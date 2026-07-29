"""EuroSAT data access, plus a copy of the train/val/test split from eurosat-mlops.

The split has to match exactly: the reference set comes from train (what the model
learned) and production batches from test (what it never saw). If they overlapped
we'd be comparing a sample against itself.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# Alphabetical, matching torchvision's ImageFolder and Project 1's
# EUROSAT_CLASSES. The index of a class in this tuple is its integer label.
EUROSAT_CLASSES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)

# Project 1's defaults (conf/config.yaml: seed=42; conf/data/eurosat.yaml:
# val_fraction=0.15, test_fraction=0.15). Changing these here would silently
# break the disjointness guarantee, so they are named constants, not literals.
SPLIT_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Configure, don't hardcode (the rule Project 5 landed on). The default points
# at a sibling checkout so a fresh clone works without ceremony.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[3] / "Project1" / "data" / "raw" / "eurosat" / "2750"


def find_data_root() -> Path:
    """Locate the EuroSAT image tree (the directory holding the class folders)."""
    root = Path(os.environ.get("EUROSAT_DATA", DEFAULT_DATA_ROOT))
    if not root.is_dir():
        raise FileNotFoundError(
            f"EuroSAT data not found at {root}. Point EUROSAT_DATA at the "
            f"directory containing the 10 class folders (Project 1 keeps it at "
            f"data/raw/eurosat/2750)."
        )
    return root


def list_samples(root: Path | None = None) -> list[tuple[Path, int]]:
    """All (path, label) pairs in torchvision ImageFolder order.

    The order matters: it defines the global indices that the split shuffles.
    ImageFolder sorts class directories and, inside each, sorts filenames as
    strings -- so `Forest_1000.jpg` comes before `Forest_999.jpg`. Sorting
    numerically here would produce a different, wrong split.
    """
    root = root or find_data_root()
    samples: list[tuple[Path, int]] = []
    for label, class_name in enumerate(EUROSAT_CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for name in sorted(p.name for p in class_dir.iterdir() if p.is_file()):
            samples.append((class_dir / name, label))
    return samples


def stratified_split(
    targets: list[int],
    val_fraction: float = VAL_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = SPLIT_SEED,
) -> tuple[list[int], list[int], list[int]]:
    """Reproduce Project 1's train/val/test split exactly.

    A port of `eurosat.data.datamodule._stratified_split`: one RNG, consumed class
    by class in ascending label order, then one extra shuffle per split. The
    sequence of RNG calls has to match, not just the seed.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in sorted(by_class):
        idxs = np.array(by_class[label])
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test : n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val :].tolist())

    for split in (train_idx, val_idx, test_idx):
        rng.shuffle(split)
    return train_idx, val_idx, test_idx


def load_split(split: str, root: Path | None = None) -> list[tuple[Path, int]]:
    """Return the (path, label) pairs belonging to one split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split {split!r}; expected train, val or test.")
    samples = list_samples(root)
    targets = [label for _, label in samples]
    train_idx, val_idx, test_idx = stratified_split(targets)
    chosen = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return [samples[i] for i in chosen]


def verify_split(root: Path | None = None) -> dict[str, object]:
    """Sanity-check the reproduced split: sizes, disjointness, stratification.

    This is the cheap insurance the whole project rests on. If it ever fails,
    every drift number downstream is suspect.
    """
    samples = list_samples(root)
    targets = [label for _, label in samples]
    train_idx, val_idx, test_idx = stratified_split(targets)

    sets = [set(train_idx), set(val_idx), set(test_idx)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), (
        "Splits overlap -- reference and production batches would share images."
    )
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(samples), "Split loses samples."

    def class_mix(idx: list[int]) -> dict[str, float]:
        counts = np.bincount([targets[i] for i in idx], minlength=len(EUROSAT_CLASSES))
        return {c: round(float(n) / len(idx), 4) for c, n in zip(EUROSAT_CLASSES, counts)}

    return {
        "total": len(samples),
        "sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "train_mix": class_mix(train_idx),
        "test_mix": class_mix(test_idx),
    }


if __name__ == "__main__":  # `python -m monitoring.eurosat` as a quick check
    import json

    print(json.dumps(verify_split(), indent=2))
