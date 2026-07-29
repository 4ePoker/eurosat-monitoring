"""The reference profile: what "normal" is, frozen into an artifact.

Every drift number is a comparison against this, which makes it as load-bearing as the
model. Two numbers computed against different profiles aren't comparable and nothing
in a chart will tell you that.

The PCA basis lives here for the same reason. Refitting it silently rebases all the
history.

The reference comes from the train split, since that's the distribution the weights
actually encode.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from monitoring.detect.embeddings import fit_pca
from monitoring.eurosat import EUROSAT_CLASSES
from monitoring.features.extract import OUTPUT_FEATURES
from monitoring.features.image import FEATURE_NAMES

IMAGE_FEATURES = tuple(FEATURE_NAMES)
ALL_FEATURES = IMAGE_FEATURES + tuple(OUTPUT_FEATURES)


def load_feature_matrix(batch_dir: Path, names: tuple[str, ...] = ALL_FEATURES) -> np.ndarray:
    rows = list(csv.DictReader((batch_dir / "features.csv").open()))
    return np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float64)


def load_predicted_classes(batch_dir: Path) -> list[str]:
    return [r["pred_class"] for r in csv.DictReader((batch_dir / "features.csv").open())]


@dataclass
class ReferenceProfile:
    features: np.ndarray          # (n, 25) image + output features
    embeddings: np.ndarray        # (n, 2048)
    pca: PCA
    feature_names: tuple[str, ...]
    #: How often the model predicted each class on the reference. The baseline
    #: for label-shift detection: in production this is the only view of the
    #: class mix we get, since true labels never arrive.
    predicted_counts: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return len(self.features)

    # -- build -------------------------------------------------------------

    @classmethod
    def fit_from_batch(cls, batch_dir: Path, n_components: int = 32) -> "ReferenceProfile":
        features = load_feature_matrix(batch_dir)
        embeddings = np.load(batch_dir / "embeddings.npy").astype(np.float64)
        pca = fit_pca(embeddings, n_components=n_components)
        preds = load_predicted_classes(batch_dir)
        counts = np.array([sum(p == c for p in preds) for c in EUROSAT_CLASSES], dtype=float)

        digest = hashlib.sha256()
        digest.update(features.tobytes())
        digest.update(embeddings.tobytes())

        meta = {
            "source_batch": batch_dir.name,
            "n_samples": len(features),
            "n_features": features.shape[1],
            "embedding_dim": int(embeddings.shape[1]),
            "pca_components": n_components,
            "pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
            "content_sha256": digest.hexdigest()[:16],
            "predicted_counts": {c: int(n) for c, n in zip(EUROSAT_CLASSES, counts)},
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        return cls(features, embeddings, pca, ALL_FEATURES, counts, meta)

    # -- persist -----------------------------------------------------------

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "features.npy", self.features)
        np.save(out_dir / "predicted_counts.npy", self.predicted_counts)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        np.savez(
            out_dir / "pca.npz",
            components=self.pca.components_,
            mean=self.pca.mean_,
            explained_variance=self.pca.explained_variance_,
            explained_variance_ratio=self.pca.explained_variance_ratio_,
        )
        (out_dir / "meta.json").write_text(
            json.dumps({**self.meta, "feature_names": list(self.feature_names)}, indent=2) + "\n"
        )
        return out_dir

    @classmethod
    def load(cls, in_dir: Path) -> "ReferenceProfile":
        meta = json.loads((in_dir / "meta.json").read_text())
        npz = np.load(in_dir / "pca.npz")
        pca = PCA(n_components=len(npz["components"]))
        pca.components_ = npz["components"]
        pca.mean_ = npz["mean"]
        pca.explained_variance_ = npz["explained_variance"]
        pca.explained_variance_ratio_ = npz["explained_variance_ratio"]
        pca.n_components_ = len(npz["components"])
        pca.n_features_in_ = npz["components"].shape[1]
        return cls(
            features=np.load(in_dir / "features.npy"),
            embeddings=np.load(in_dir / "embeddings.npy"),
            pca=pca,
            feature_names=tuple(meta["feature_names"]),
            predicted_counts=np.load(in_dir / "predicted_counts.npy"),
            meta={k: v for k, v in meta.items() if k != "feature_names"},
        )
