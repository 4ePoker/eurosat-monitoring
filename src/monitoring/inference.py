"""Runs the deployed ONNX model over files on disk.

Preprocessing is imported from eurosat-serving rather than reimplemented, so the
monitor sees exactly what the API sees. Batched here because we own the whole batch,
unlike the API which handles one request at a time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# Configure, don't hardcode -- same convention as EUROSAT_DATA.
DEFAULT_SERVING_SRC = Path(__file__).resolve().parents[3] / "Project5" / "src"
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "Project5" / "models" / "eurosat_resnet50.static_int8.onnx"


def _import_serving():
    """Put Project 5's serving package on the path and hand back its preprocess."""
    src = Path(os.environ.get("SERVING_SRC", DEFAULT_SERVING_SRC))
    if not (src / "serving" / "preprocess.py").is_file():
        raise FileNotFoundError(
            f"Project 5's serving package not found under {src}. Point SERVING_SRC at it."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from serving import preprocess  # noqa: PLC0415 - deliberate late import

    return preprocess


class Predictor:
    """The deployed classifier, applied to files on disk instead of HTTP uploads."""

    def __init__(self, model_path: str | Path | None = None, batch_size: int = 32):
        self._pp = _import_serving()
        self.classes: tuple[str, ...] = self._pp.CLASSES
        path = str(model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL))
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"Model not found at {path}. Project 5 keeps model files out of git -- "
                f"regenerate them with its scripts/export_onnx.py and scripts/quantize.py."
            )
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = path
        self.batch_size = batch_size

    def _preprocess_file(self, path: Path) -> np.ndarray:
        # serving.preprocess.preprocess() returns (1, 3, 224, 224); drop the
        # batch axis here and stack ourselves.
        return self._pp.preprocess(Image.open(path))[0]

    def predict_paths(
        self, paths: list[Path], with_features: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Probabilities of shape (n, 10), optionally with (n, 2048) embeddings.

        The embeddings come out of the same forward pass, so asking for them costs
        nothing. That's why eurosat-serving's export was changed instead of running
        the network twice.
        """
        out = np.empty((len(paths), len(self.classes)), dtype=np.float32)
        feats: list[np.ndarray] = []
        want = ["logits", "features"] if with_features else ["logits"]
        for start in range(0, len(paths), self.batch_size):
            chunk = paths[start : start + self.batch_size]
            x = np.stack([self._preprocess_file(p) for p in chunk])
            result = self.session.run(want, {self.input_name: x})
            logits = result[0]
            z = logits - logits.max(axis=1, keepdims=True)
            e = np.exp(z)
            out[start : start + len(chunk)] = e / e.sum(axis=1, keepdims=True)
            if with_features:
                feats.append(result[1])
        return (out, np.concatenate(feats)) if with_features else out


def entropy(probs: np.ndarray) -> np.ndarray:
    """Predictive entropy per row, in nats. High = the model is spreading its bets.

    This is one of the few damage signals available in production without
    labels, which is exactly why it is worth measuring here, where we *can*
    check whether it tracks the real accuracy drop.
    """
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def margin(probs: np.ndarray) -> np.ndarray:
    """Gap between the top two classes. Small = the decision was nearly a coin flip."""
    top2 = np.sort(probs, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]
