"""Score a candidate model on the whole bench, per batch and per class.

    python -m monitoring.promote.evaluate --model ../Project5/models/eurosat_resnet50.onnx

Per batch, because a model retrained for a new sensor will obviously win on the new
sensor; the question is what it broke elsewhere. Per class, because overall accuracy
can rise while a rare class collapses.

Only batches with labels are scored, so production windows are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from monitoring.inference import Predictor

ROOT = Path(__file__).resolve().parents[3]
BATCH_ROOT = ROOT / "data" / "batches"
ARTIFACTS = ROOT / "artifacts" / "evaluations"


def labelled_batches(root: Path = BATCH_ROOT) -> list[Path]:
    """Batches whose manifest carries a ground-truth class."""
    out = []
    for d in sorted(root.iterdir()):
        manifest = d / "manifest.csv"
        if not manifest.is_file():
            continue
        rows = list(csv.DictReader(manifest.open()))
        if rows and rows[0].get("observed_class"):
            out.append(d)
    return out


def evaluate_batch(batch_dir: Path, predictor: Predictor) -> dict:
    rows = list(csv.DictReader((batch_dir / "manifest.csv").open()))
    paths = [batch_dir / "images" / r["filename"] for r in rows]
    classes = predictor.classes

    probs = predictor.predict_paths(paths)
    pred = probs.argmax(axis=1)
    truth = np.array([classes.index(r["observed_class"]) for r in rows])

    per_class = {}
    for k, name in enumerate(classes):
        mask = truth == k
        per_class[name] = {
            "n": int(mask.sum()),
            "recall": round(float((pred[mask] == k).mean()), 4) if mask.any() else None,
        }

    return {
        "batch": batch_dir.name,
        "n": len(rows),
        "accuracy": round(float((pred == truth).mean()), 4),
        "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
        "per_class": per_class,
    }


def evaluate(model_path: Path, name: str | None = None) -> dict:
    predictor = Predictor(model_path=model_path)
    batches = labelled_batches()
    print(f"model: {model_path}\n")
    results = []
    for batch_dir in batches:
        r = evaluate_batch(batch_dir, predictor)
        results.append(r)
        print(f"  {r['batch']:36s} n={r['n']:4d}  acc={r['accuracy']:.4f}")

    report = {
        "model": str(model_path),
        "model_name": name or Path(model_path).stem,
        "size_mb": round(Path(model_path).stat().st_size / 1e6, 1),
        "n_batches": len(results),
        "mean_accuracy": round(float(np.mean([r["accuracy"] for r in results])), 4),
        "batches": results,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"{report['model_name']}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    evaluate(args.model, args.name)


if __name__ == "__main__":
    main()
