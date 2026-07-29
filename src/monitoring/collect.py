"""Turns a window of production logs into a batch in the format the detector reads.

Two tiers, matching what the serving sink writes: output features for every request,
image features and embeddings only for the sampled fraction.

No labels. `manifest.csv` keeps the columns and leaves them empty, because in
production nobody knows the true class.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from monitoring.eurosat import EUROSAT_CLASSES
from monitoring.features.extract import OUTPUT_FEATURES
from monitoring.features.image import FEATURE_NAMES, image_features

ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = ROOT / "data" / "batches"


def read_records(log_dir: Path, window: str | None = None) -> list[dict]:
    """Load prediction records, optionally restricted to one daily file."""
    pattern = f"predictions-{window}.jsonl" if window else "predictions-*.jsonl"
    files = sorted(log_dir.glob(pattern))
    if not files:
        raise SystemExit(f"no prediction logs matching {pattern} under {log_dir}")
    records = []
    for path in files:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect(log_dir: Path, window: str | None, name: str | None) -> dict:
    records = read_records(log_dir, window)
    sampled = [r for r in records if r.get("sampled")]
    samples_dir = log_dir / "samples"

    batch_name = name or f"production_{window or 'all'}_n{len(records)}"
    out_dir = BATCH_ROOT / batch_name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    rows, embeddings, missing = [], [], 0
    for rec in sampled:
        rid = rec["request_id"]
        img_path = samples_dir / f"{rid}.img"
        emb_path = samples_dir / f"{rid}.npy"
        if not img_path.is_file() or not emb_path.is_file():
            missing += 1
            continue

        image = Image.open(img_path).convert("RGB")
        # Copied into the batch as PNG so the batch is self-contained and
        # lossless; the features are computed from the original bytes above,
        # never from this copy.
        out_name = f"{rid}.png"
        image.save(out_dir / "images" / out_name, format="PNG")

        arr = np.asarray(image, dtype=np.float32) / 255.0
        feats = image_features(arr)
        rows.append({
            "request_id": rid,
            "filename": out_name,
            "pred_class": rec["pred_class"],
            **{n: f"{v:.6g}" for n, v in zip(FEATURE_NAMES, feats)},
            **{n: f"{float(rec[n]):.6g}" for n in OUTPUT_FEATURES},
        })
        embeddings.append(np.load(emb_path))

    if not rows:
        raise SystemExit(
            f"{len(sampled)} records marked sampled but no usable image/embedding pairs "
            f"found under {samples_dir}. Was MONITOR_SAMPLE_RATE set to 0?"
        )

    # features.csv and embeddings.npy in exactly the layout brick 3 reads.
    header = ["filename", "pred_class", *FEATURE_NAMES, *OUTPUT_FEATURES]
    with (out_dir / "features.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    np.save(out_dir / "embeddings.npy", np.stack(embeddings).astype(np.float32))

    # The manifest keeps the label columns and leaves them empty. In production
    # they are unknowable; an empty column says so, a missing column hides it.
    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "source", "true_class", "observed_class",
                         "scenario", "severity", "split", "seed"])
        for r in rows:
            writer.writerow([r["filename"], f"request:{r['request_id']}", "", "",
                             "production", "", "live", ""])

    # The 100% tier: output features for every request, sampled or not.
    all_header = ["request_id", "ts", "pred_class", *OUTPUT_FEATURES, "inference_ms"]
    with (out_dir / "output_features_all.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(r["pred_class"] for r in records)
    meta = {
        "batch": batch_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": "production",
        "description": "Live traffic collected from the serving monitor.",
        "drift_types": [],
        "severity": None,
        "split": "live",
        "window": window or "all",
        "n_requests_total": len(records),
        "n_requests_sampled": len(rows),
        "effective_sample_rate": round(len(rows) / max(len(records), 1), 4),
        "n_sampled_missing_files": missing,
        "labels_available": False,
        "predicted_class_counts": {c: counts.get(c, 0) for c in EUROSAT_CLASSES},
        "notes": (
            "No true labels: accuracy is not computable here. Output features and the "
            "predicted-class distribution cover 100% of traffic; image features and "
            "embeddings cover the sampled fraction only."
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True, help="the serving MONITOR_DIR")
    parser.add_argument("--window", default=None, help="a YYYYMMDD daily file; omit for all")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    meta = collect(args.logs, args.window, args.name)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
