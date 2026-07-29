"""What does monitoring cost per request?

    python scripts/measure_overhead.py

An end-to-end HTTP benchmark is the measurement you want and the one this machine
cannot give honestly: at load average 10 on 8 cores, run-to-run variation swamped
the effect entirely -- the first configuration measured came out slowest and the
last fastest, in the order they were run rather than the order of the treatment.
Reporting that would have been reporting the background load.

So this decomposes instead. Each piece is a short operation, repeated enough times
that the median is stable under background noise, and each is timed separately:

    inference              the baseline everything else is compared against
    + the second output    asking the graph for `features` as well as `logits`
    peek_size              reading image dimensions from the header
    record, light          the 100% path: serialise ~287 bytes, append to a file
    record, sampled        the 5% path: also write an 8 KB embedding and the image

Reported as medians against the inference cost, because a millisecond means
nothing on its own and everything next to the 60 or so the model already spends.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVING_SRC = ROOT.parent / "Project5" / "src"
MODEL = ROOT.parent / "Project5" / "models" / "eurosat_resnet50.static_int8.onnx"
sys.path.insert(0, str(SERVING_SRC))

from serving.monitor import MonitorSink  # noqa: E402
from serving.preprocess import CLASSES, decode_and_preprocess  # noqa: E402


def timeit_interleaved(cases: list[tuple[str, object, int]]) -> dict[str, tuple[float, float]]:
    """Time every variant in a round-robin, never in blocks.

    Measuring configurations one after another is how this file got its first
    wrong answer: run sequentially, asking the ONNX graph for a second output
    appeared to cost 4.74 ms, and interleaving the same two calls put it at
    0.49 ms. The machine drifts -- background load, cache state, clock -- and a
    block design lets that drift line up perfectly with the treatment. The same
    mistake, made in the same session, also produced an end-to-end benchmark in
    which more monitoring came out faster than none.

    Round-robin does not remove the drift; it spreads it evenly across the
    variants, which is all a comparison needs.
    """
    n = max(c[2] for c in cases)
    samples: dict[str, list[float]] = {label: [] for label, _, _ in cases}
    for _ in range(5):  # warm up every variant before timing any of them
        for _, fn, _ in cases:
            fn()
    for i in range(n):
        for label, fn, count in cases:
            if i >= count:
                continue
            start = time.perf_counter()
            fn()
            samples[label].append((time.perf_counter() - start) * 1000)
    out = {}
    for label, values in samples.items():
        values.sort()
        out[label] = (statistics.median(values), values[int(0.95 * len(values))])
    return out


def main() -> None:
    import onnxruntime as ort

    tile = next((ROOT / "data" / "batches" / "none_s0.50_test_n500" / "images").glob("*.png"))
    data = tile.read_bytes()
    x = decode_and_preprocess(data)

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name

    out_dir = ROOT / "artifacts" / "_overhead_tmp"
    sink_light = MonitorSink(directory=str(out_dir), sample_rate=0.0)
    sink_heavy = MonitorSink(directory=str(out_dir), sample_rate=1.0)
    probs = np.random.default_rng(0).dirichlet(np.ones(10)).astype(np.float32)
    embedding = np.random.default_rng(1).random(2048).astype(np.float32)
    counter = {"i": 0}

    def rid() -> str:
        counter["i"] += 1
        return f"{counter['i']:08x}"

    def peek():
        from io import BytesIO

        from PIL import Image
        return Image.open(BytesIO(data)).size

    measurements = [
        ("inference, logits only",
         lambda: session.run(["logits"], {name: x}), 400),
        ("inference, logits + features",
         lambda: session.run(["logits", "features"], {name: x}), 400),
        ("peek_size (header only)", peek, 2000),
        ("record, light path (100%)",
         lambda: sink_light.record(rid(), CLASSES, probs, None, data, (64, 64), 1.0), 2000),
        ("record, sampled path (5%)",
         lambda: sink_heavy.record(rid(), CLASSES, probs, embedding, data, (64, 64), 1.0), 500),
    ]

    print(f"tile: {tile.name}   filesystem for the log: {out_dir}")
    print("all variants timed round-robin, never in blocks\n")
    timings = timeit_interleaved(measurements)
    print(f"{'operation':34s} {'median ms':>10s} {'p95 ms':>9s} {'n':>6s}")
    print("-" * 63)
    results = {}
    for label, _, n in measurements:
        med, p95 = timings[label]
        results[label] = med
        print(f"{label:34s} {med:10.3f} {p95:9.3f} {n:6d}")

    base = results["inference, logits only"]
    second_output = results["inference, logits + features"] - base
    light = results["peek_size (header only)"] + results["record, light path (100%)"]
    heavy = light + second_output + (
        results["record, sampled path (5%)"] - results["record, light path (100%)"]
    )

    # The inference comparison is at the edge of what this machine can resolve:
    # a ~0.5 ms effect inside a measurement whose p95 sits 30 ms above its median.
    # When the difference comes out negative -- as it does, the two-output call
    # timing faster than the one-output call, which cannot happen -- the honest
    # report is a bound, not a point estimate. Inventing a number here would be
    # worse than admitting the instrument is too blunt.
    resolvable = second_output > 0
    writes_sampled = results["record, sampled path (5%)"] + results["peek_size (header only)"]

    print(f"\ninference baseline                 {base:.2f} ms")
    print(f"added on an unsampled request      {light:.3f} ms  ({100 * light / base:.2f}% of inference)")
    if resolvable:
        print(f"added on a sampled request         {heavy:.3f} ms  ({100 * heavy / base:.2f}% of inference)")
        print(f"  of which the second ONNX output  {second_output:+.3f} ms")
    else:
        print(f"added on a sampled request         {writes_sampled:.3f} ms of writes, plus the")
        print(f"                                   second ONNX output, which is BELOW this")
        print(f"                                   machine's resolution (measured {second_output:+.3f} ms,")
        print(f"                                   i.e. noise). A separate 300-sample run put")
        print(f"                                   it at +0.49 ms; treat it as under 1 ms.")

    for rate in (0.05, 0.30):
        mean_cost = light + rate * (writes_sampled - light + max(second_output, 0.5))
        print(f"average at {rate:.0%} sampling            ~{mean_cost:.3f} ms  "
              f"({100 * mean_cost / base:.2f}% of inference)")

    for f in out_dir.rglob("*"):
        if f.is_file():
            f.unlink()
    print("\nNote: the log directory sits on the same filesystem as the repo. On a "
          "container writing to an overlay or a network volume, the write cost is higher.")


if __name__ == "__main__":
    main()
