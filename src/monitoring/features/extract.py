"""Extract all three feature layers for a batch in one pass.

    python -m monitoring.features.extract --batch autumn_s0.70_test_n500

    21 image features   what the incoming signal looks like    -> features.csv
    4 output features   how the model is behaving              -> features.csv
    2048-d embedding    what the model sees in the image       -> embeddings.npy

Labels are deliberately excluded. Everything in features.csv is computable in
production from the request and the response alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from monitoring.features.image import FEATURE_NAMES, image_features
from monitoring.inference import Predictor, entropy, margin

BATCH_ROOT = Path(__file__).resolve().parents[3] / "data" / "batches"

OUTPUT_FEATURES = ("pred_confidence", "pred_entropy", "pred_margin", "pred_top2_gap")


def extract_batch(batch_dir: Path, predictor: Predictor) -> dict:
    rows = list(csv.DictReader((batch_dir / "manifest.csv").open()))
    paths = [batch_dir / "images" / r["filename"] for r in rows]

    # Layer 1: straight from the bytes, no model involved.
    img_feats = np.stack(
        [image_features(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)
         for p in paths]
    )

    # Layers 2 and 3: one forward pass gives both.
    probs, embeddings = predictor.predict_paths(paths, with_features=True)
    conf = probs.max(axis=1)
    ent = entropy(probs)
    marg = margin(probs)
    top2 = np.sort(probs, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    out_feats = np.stack([conf, ent, marg, gap], axis=1)

    header = ["filename", "pred_class", *FEATURE_NAMES, *OUTPUT_FEATURES]
    with (batch_dir / "features.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(rows):
            writer.writerow(
                [r["filename"], predictor.classes[int(probs[i].argmax())],
                 *[f"{v:.6g}" for v in img_feats[i]],
                 *[f"{v:.6g}" for v in out_feats[i]]]
            )

    np.save(batch_dir / "embeddings.npy", embeddings.astype(np.float32))

    return {
        "batch": batch_dir.name,
        "n": len(rows),
        "n_image_features": len(FEATURE_NAMES),
        "embedding_dim": int(embeddings.shape[1]),
        "model": predictor.model_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default=None)
    parser.add_argument("--root", type=Path, default=BATCH_ROOT)
    args = parser.parse_args()

    batches = (
        [args.root / args.batch]
        if args.batch
        else sorted(d for d in args.root.iterdir() if (d / "manifest.csv").is_file())
    )
    predictor = Predictor()
    print(f"model: {predictor.model_path}\n")

    summary = []
    for batch_dir in batches:
        info = extract_batch(batch_dir, predictor)
        summary.append(info)
        print(f"  {info['batch']:36s} {info['n']:5d} tiles  "
              f"{info['n_image_features']} image features + {info['embedding_dim']}-d embedding")

    (args.root / "extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
