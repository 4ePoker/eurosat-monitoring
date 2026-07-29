"""Build a production batch from RESISC45: real domain shift, not simulated.

    python scripts/build_resisc_batch.py --n 500

Everything in `data/batches` so far was fabricated by us, and that is the single
biggest weakness of the whole project: a detector tuned on our own transforms
might be good at detecting *our transforms* rather than drift. This batch is the
antidote. RESISC45 is a different dataset entirely -- Google Earth imagery rather
than Sentinel-2, a different ground sample distance, different colour processing,
different continents. Nobody invented the shift; it is simply what happens when
you point a model at somebody else's data.

Two decisions worth arguing about:

RESIZE TO 64x64. RESISC45 ships 256x256. Feeding that straight in would let the
image-feature extractor report drift purely because the tiles are four times
wider -- sharpness, blockiness and high-frequency energy are all resolution
dependent, so a size change alone moves them. Holding the format constant keeps
the measurement about content and sensor, which is the thing we want to test.

MATCH THE CLASS MIX. Sampling RESISC45's own proportions would layer a label
shift on top of the data shift, and the diagnosis rule works by separating those
two. So the batch is drawn to match the reference's proportions among the mapped
classes, leaving data shift as the dominant signal.

The honest caveats, which belong in the README next to the result:

* The class mapping below is a judgement call, not ground truth. A RESISC45
  "river" at Google Earth resolution and a EuroSAT "river" at 10 m/px are not
  obviously the same concept, so the accuracy measured here carries label noise
  on top of genuine model error. Treat the number as a floor, not a measurement.
* Two EuroSAT classes (HerbaceousVegetation, PermanentCrop) have no confident
  analogue and are simply absent, which leaves a residual label shift of about a
  fifth of the reference mass.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from monitoring.eurosat import EUROSAT_CLASSES, load_split

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "external" / "resisc45_test.parquet"
OUT_ROOT = ROOT / "data" / "batches"

# RESISC45 label index -> EuroSAT class. Only mappings we would defend out loud;
# the ambiguous ones (chaparral, wetland, terrace, golf_course) are left out
# rather than guessed, because a wrong mapping shows up as model error and would
# make the model look worse than it is.
MAPPING: dict[int, str] = {
    13: "Forest",
    32: "River",
    21: "SeaLake",
    14: "Highway",              # freeway
    18: "Industrial",           # industrial_area
    11: "Residential",          # dense_residential
    23: "Residential",          # medium_residential
    31: "AnnualCrop",           # rectangular_farmland
    8: "AnnualCrop",            # circular_farmland
    22: "Pasture",              # meadow
}
UNMAPPED = ("HerbaceousVegetation", "PermanentCrop")


def reference_mix() -> dict[str, float]:
    """Proportions of the mapped classes in the EuroSAT training split."""
    labels = [label for _, label in load_split("train")]
    counts = np.bincount(labels, minlength=len(EUROSAT_CLASSES))
    mix = {
        c: float(counts[i]) for i, c in enumerate(EUROSAT_CLASSES) if c not in UNMAPPED
    }
    total = sum(mix.values())
    return {c: v / total for c, v in mix.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", default="resisc45_real_n500")
    parser.add_argument(
        "--size", type=int, default=64,
        help="tile size to write. 64 matches the EuroSAT format (one variable changed); "
             "256 keeps RESISC45's native detail (a different, also-real scenario)",
    )
    args = parser.parse_args()

    if not PARQUET.is_file():
        raise SystemExit(f"missing {PARQUET}; download the RESISC45 test split first")

    table = pq.read_table(PARQUET)
    labels = np.array(table.column("label").to_pylist())
    images = table.column("image").to_pylist()

    by_eurosat: dict[str, list[int]] = {}
    for resisc_idx, eurosat_class in MAPPING.items():
        by_eurosat.setdefault(eurosat_class, []).extend(np.flatnonzero(labels == resisc_idx).tolist())

    rng = np.random.default_rng(args.seed)
    mix = reference_mix()
    wanted = {c: int(round(mix[c] * args.n)) for c in by_eurosat}

    chosen: list[tuple[int, str]] = []
    for eurosat_class, k in wanted.items():
        pool = np.array(by_eurosat[eurosat_class])
        if k > len(pool):
            print(f"  ! {eurosat_class}: wanted {k}, only {len(pool)} available")
            k = len(pool)
        chosen.extend((int(i), eurosat_class) for i in rng.choice(pool, size=k, replace=False))
    rng.shuffle(chosen)

    out_dir = OUT_ROOT / args.name
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for position, (idx, eurosat_class) in enumerate(chosen):
        img = Image.open(io.BytesIO(images[idx]["bytes"])).convert("RGB")
        img = img.resize((args.size, args.size), Image.BILINEAR)
        name = f"{position:05d}_resisc{idx}.png"
        img.save(img_dir / name, format="PNG")
        rows.append({
            "filename": name, "source": f"resisc45_test#{idx}",
            "true_class": eurosat_class, "observed_class": eurosat_class,
            "scenario": "resisc45_real", "severity": "1.00",
            "split": "external", "seed": args.seed,
        })

    with (out_dir / "manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "meta.json").write_text(json.dumps({
        "batch": args.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": "resisc45_real",
        "description": "Real domain shift: NWPU-RESISC45 (Google Earth) scored by a Sentinel-2 model.",
        "drift_types": ["data"],
        "severity": 1.0,
        "split": "external",
        "seed": args.seed,
        "n_images": len(rows),
        "n_relabelled": 0,
        "pixels_altered": False,
        "class_counts": {c: sum(r["true_class"] == c for r in rows) for c in EUROSAT_CLASSES},
        "notes": (
            "Not synthetic. The shift comes from using another dataset entirely. "
            "Class mapping is a judgement call and carries label noise; two EuroSAT "
            "classes have no analogue and are absent, leaving a residual label shift."
        ),
    }, indent=2) + "\n")

    print(f"  {args.name}: {len(rows)} tiles -> {out_dir}")
    for c, n in sorted(((c, sum(r['true_class'] == c for r in rows)) for c in by_eurosat), key=lambda kv: -kv[1]):
        print(f"      {c:22s} {n:4d}")


if __name__ == "__main__":
    main()
