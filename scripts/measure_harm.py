"""Measure what each drift scenario actually costs in accuracy.

    python scripts/measure_harm.py                 # every batch under data/batches
    python scripts/measure_harm.py --batch autumn_s0.70_test_n500

This is the step that turns the generator from a data factory into a benchmark.
In production we will never know the accuracy; here we do, because the batches
were built from the held-out test split and the manifest carries the labels. So
this run produces the *ground truth of harm* that brick 3's detector has to be
validated against. Without it we would be tuning an alarm with no idea what it
is supposed to catch.

Two accuracy columns are reported, and the difference between them is the whole
point of concept drift:

    acc_old_rule -- scored against `true_class`,     the labelling rule of the training set
    acc_new_rule -- scored against `observed_class`, the labelling rule in force today

They are identical for every scenario except the concept ones. There, the model
did not change, the images did not change, and the accuracy still falls -- purely
because the definition of "correct" moved.

Alongside the summary, each batch gets a `predictions.csv`: the per-image
prediction log. That file is the raw material for brick 3, and it is deliberately
written with only the columns a real production logger could emit (plus the
labels, which live here in the lab and would not exist in production).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from monitoring.inference import Predictor, entropy, margin

BATCH_ROOT = Path(__file__).resolve().parents[1] / "data" / "batches"


def score_batch(batch_dir: Path, predictor: Predictor) -> dict:
    rows = list(csv.DictReader((batch_dir / "manifest.csv").open()))
    paths = [batch_dir / "images" / r["filename"] for r in rows]

    probs = predictor.predict_paths(paths)
    pred_idx = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    ent = entropy(probs)
    marg = margin(probs)
    classes = predictor.classes

    true_idx = np.array([classes.index(r["true_class"]) for r in rows])
    obs_idx = np.array([classes.index(r["observed_class"]) for r in rows])

    with (batch_dir / "predictions.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["filename", "true_class", "observed_class", "pred_class",
             "confidence", "entropy", "margin", *classes]
        )
        for i, r in enumerate(rows):
            writer.writerow(
                [r["filename"], r["true_class"], r["observed_class"], classes[pred_idx[i]],
                 f"{conf[i]:.6f}", f"{ent[i]:.6f}", f"{marg[i]:.6f}",
                 *[f"{p:.6f}" for p in probs[i]]]
            )

    meta = json.loads((batch_dir / "meta.json").read_text())
    pred_counts = np.bincount(pred_idx, minlength=len(classes))
    return {
        "batch": batch_dir.name,
        "scenario": meta["scenario"],
        "drift_types": ",".join(meta["drift_types"]) or "-",
        "severity": meta["severity"],
        "n": len(rows),
        "acc_old_rule": float((pred_idx == true_idx).mean()),
        "acc_new_rule": float((pred_idx == obs_idx).mean()),
        "mean_confidence": float(conf.mean()),
        "mean_entropy": float(ent.mean()),
        "mean_margin": float(marg.mean()),
        "pred_distribution": {c: int(n) for c, n in zip(classes, pred_counts)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=None, help="one batch directory name")
    parser.add_argument("--root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--out", type=Path, default=BATCH_ROOT / "harm_summary.json")
    args = parser.parse_args()

    batches = (
        [args.root / args.batch]
        if args.batch
        else sorted(d for d in args.root.iterdir() if (d / "manifest.csv").is_file())
    )

    predictor = Predictor()
    print(f"model: {predictor.model_path}\n")

    results = []
    header = f"{'batch':36s} {'drift':14s} {'acc(old)':>9s} {'acc(new)':>9s} {'conf':>7s} {'entropy':>8s}"
    print(header)
    print("-" * len(header))
    for batch_dir in batches:
        r = score_batch(batch_dir, predictor)
        results.append(r)
        print(
            f"{r['batch']:36s} {r['drift_types']:14s} "
            f"{r['acc_old_rule']:9.3f} {r['acc_new_rule']:9.3f} "
            f"{r['mean_confidence']:7.3f} {r['mean_entropy']:8.3f}"
        )

    # Scoring a single batch must not wipe the others: --batch merges into the
    # existing summary. (It used to overwrite, which silently emptied the harm
    # column of every report downstream -- the summary file still existed and
    # still parsed, so nothing complained.)
    merged: dict[str, dict] = {}
    if args.out.is_file():
        merged = {r["batch"]: r for r in json.loads(args.out.read_text())}
    merged.update({r["batch"]: r for r in results})
    args.out.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["batch"]), indent=2) + "\n")
    print(f"\nwrote {args.out} ({len(merged)} batches)")


if __name__ == "__main__":
    main()
