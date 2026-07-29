"""Build a side-by-side contact sheet of the generated drift batches.

Dev tooling, not part of the monitoring runtime. Its job is to make the drift
visible to a human, because a drift score you cannot eyeball is a drift score
you will not trust when it wakes you at 3am.

The batches are generated with the same seed and no class reweighting, so
position i in each batch is the same source tile -- which makes the columns
directly comparable.

    python scripts/contact_sheet.py --out assets/drift_scenarios.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BATCH_ROOT = Path(__file__).resolve().parents[1] / "data" / "batches"

# (directory name, row label). Ordered from "invisible" to "obvious".
ROWS = [
    ("none_s0.50_test_n500", "none (control)"),
    ("recompress_s0.80_test_n500", "recompress  0.8"),
    ("haze_s0.60_test_n500", "haze  0.6"),
    ("autumn_s0.70_test_n500", "autumn  0.7"),
    ("new_sensor_s0.25_test_n500", "new_sensor  0.25"),
    ("new_sensor_s0.50_test_n500", "new_sensor  0.50"),
    ("new_sensor_s1.00_test_n500", "new_sensor  1.00"),
]

TILE = 96
LABEL_W = 150
PAD = 4


def load_tile(batch: Path, position: int) -> tuple[Image.Image, str]:
    with (batch / "manifest.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    row = rows[position]
    img = Image.open(batch / "images" / row["filename"]).convert("RGB")
    return img.resize((TILE, TILE), Image.NEAREST), row["true_class"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("assets/drift_scenarios.png"))
    parser.add_argument("--positions", default="0,1,2,3,5,8", help="which tiles to show")
    args = parser.parse_args()

    positions = [int(p) for p in args.positions.split(",")]
    width = LABEL_W + len(positions) * (TILE + PAD)
    height = 22 + len(ROWS) * (TILE + PAD)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    # Column headers: the true class of each source tile.
    _, classes = zip(*[load_tile(BATCH_ROOT / ROWS[0][0], p) for p in positions])
    for col, name in enumerate(classes):
        draw.text((LABEL_W + col * (TILE + PAD), 6), name[:14], fill="black")

    for r, (batch_name, label) in enumerate(ROWS):
        y = 22 + r * (TILE + PAD)
        draw.text((4, y + TILE // 2 - 6), label, fill="black")
        batch = BATCH_ROOT / batch_name
        for c, position in enumerate(positions):
            tile, _ = load_tile(batch, position)
            sheet.paste(tile, (LABEL_W + c * (TILE + PAD), y))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.width}x{sheet.height})")


if __name__ == "__main__":
    main()
