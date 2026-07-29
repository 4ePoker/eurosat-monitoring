"""Generate a batch of fake production traffic for a scenario.

    python -m monitoring.drift.generate --scenario new_sensor --severity 0.6 --n 500

Written to disk rather than generated on the fly, so a batch is a fixed artifact you
can re-measure after changing the detector.

Tiles are PNG. An earlier version saved JPEG q95, which added a second round of
compression to every batch including the control and cost 2.8 accuracy points.
Compression is the treatment in one scenario; it can't also be a constant in all of
them.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from monitoring.eurosat import EUROSAT_CLASSES, load_split
from monitoring.drift import scenarios as S

DEFAULT_OUT = Path(__file__).resolve().parents[3] / "data" / "batches"


def _sample_indices(
    labels: np.ndarray, n: int, weights: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Choose which images arrive.

    Without weights: a plain random sample, so the batch inherits the split's
    class mix. With weights: sample per class to hit the requested proportions,
    without replacement so no tile is counted twice (duplicates would fake a
    tighter distribution than reality).
    """
    if weights is None:
        return rng.choice(len(labels), size=min(n, len(labels)), replace=False)

    wanted = np.floor(weights * n).astype(int)
    # Hand the rounding remainder to the classes with the largest fractional
    # part, so the batch size is exactly n.
    remainder = n - wanted.sum()
    if remainder > 0:
        order = np.argsort(-(weights * n - wanted))
        wanted[order[:remainder]] += 1

    chosen: list[int] = []
    for label, k in enumerate(wanted):
        if k <= 0:
            continue
        pool = np.flatnonzero(labels == label)
        if k > len(pool):
            # Not enough held-out tiles of this class: take what exists and say
            # so, rather than silently resampling and understating variance.
            print(
                f"  ! {EUROSAT_CLASSES[label]}: wanted {k}, only {len(pool)} available "
                f"in this split; taking all of them."
            )
            k = len(pool)
        chosen.extend(rng.choice(pool, size=k, replace=False).tolist())
    return np.array(chosen)


def generate_batch(
    scenario_name: str,
    severity: float,
    n: int,
    split: str = "test",
    seed: int = 0,
    out_root: Path = DEFAULT_OUT,
    name: str | None = None,
) -> Path:
    scenario = S.get(scenario_name)
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")

    rng = np.random.default_rng(seed)
    samples = load_split(split)
    labels = np.array([label for _, label in samples])

    idx = _sample_indices(labels, n, scenario.weights_vector(), rng)
    rng.shuffle(idx)  # interleave classes: a batch is a stream, not a sorted list

    batch_name = name or f"{scenario_name}_s{severity:.2f}_{split}_n{len(idx)}"
    out_dir = out_root / batch_name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, i in enumerate(idx):
        src_path, label = samples[i]
        true_class = EUROSAT_CLASSES[label]

        img = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.float32) / 255.0
        shifted = scenario.transform(img, severity, rng)
        out_name = f"{position:05d}_{src_path.stem}.png"
        Image.fromarray((np.clip(shifted, 0, 1) * 255).astype(np.uint8)).save(
            img_dir / out_name, format="PNG", optimize=False
        )

        rows.append(
            {
                "filename": out_name,
                "source": str(src_path),
                "true_class": true_class,
                "observed_class": scenario.observed_label(true_class),
                "scenario": scenario_name,
                "severity": f"{severity:.2f}",
                "split": split,
                "seed": seed,
            }
        )

    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    relabelled = sum(r["true_class"] != r["observed_class"] for r in rows)
    meta = {
        "batch": batch_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": scenario_name,
        "description": scenario.description,
        "drift_types": list(scenario.drift_types),
        "severity": severity,
        "split": split,
        "seed": seed,
        "n_images": len(rows),
        "n_relabelled": relabelled,
        "pixels_altered": scenario.transform is not S.T.identity,
        "class_counts": {
            c: sum(r["observed_class"] == c for r in rows) for c in EUROSAT_CLASSES
        },
        "notes": scenario.notes,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"  {batch_name}: {len(rows)} tiles -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="none", help="scenario name (see --list)")
    parser.add_argument("--severity", type=float, default=0.5, help="0 = identity, 1 = extreme")
    parser.add_argument("-n", type=int, default=500, help="batch size")
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="reference batches come from train; production batches from test",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default=None, help="override the batch directory name")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--sweep",
        default=None,
        help="comma-separated severities, e.g. 0,0.25,0.5,0.75,1.0 (one batch each)",
    )
    args = parser.parse_args()

    if args.list:
        for s in S.SCENARIOS.values():
            types = ", ".join(s.drift_types) or "none (control)"
            print(f"{s.name:24s} [{types}]\n    {s.description}")
        return

    severities = (
        [float(v) for v in args.sweep.split(",")] if args.sweep else [args.severity]
    )
    for sev in severities:
        generate_batch(
            args.scenario, sev, args.n, args.split, args.seed, args.out, args.name
        )


if __name__ == "__main__":
    main()
