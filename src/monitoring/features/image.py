"""21 features per tile, standing in for 12288 raw pixel values.

Every group here earns its place from a measured scenario: band statistics catch the
seasonal and haze shifts, the sharpness group catches JPEG recompression (which costs
44 accuracy points while leaving colour statistics untouched), texture catches
resolution loss.

Computed at 64x64 regardless of what arrives. Blockiness and high-frequency energy are
defined relative to the pixel grid, so the same scene at 256x256 gives different
numbers for reasons that have nothing to do with drift.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

#: All image features are measured at this resolution, regardless of input size.
CANONICAL_SIZE = 64

# Rec. 601 luma. The exact weights barely matter here; being consistent does.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)

FEATURE_NAMES: tuple[str, ...] = (
    # --- band statistics: colour and exposure -----------------------------
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "brightness",
    "saturation",
    "corr_rg", "corr_rb", "corr_gb",
    # --- sharpness and compression artefacts ------------------------------
    "laplacian_var",
    "gradient_mean",
    "hf_energy_ratio",
    "blockiness",
    # --- texture ----------------------------------------------------------
    "entropy",
    "global_contrast",
    "local_contrast",
    # --- dynamic range and noise ------------------------------------------
    "frac_dark_clipped",
    "frac_bright_clipped",
    "noise_sigma",
)


def _grayscale(img: np.ndarray) -> np.ndarray:
    return img @ _LUMA


def _laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian on the interior. High response = fine detail."""
    return (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )


def _blockiness(gray: np.ndarray, block: int = 8) -> float:
    """Excess gradient across 8x8 block seams, over the gradient inside blocks.

    JPEG quantises each 8x8 block on its own, so the seams become steps the scene
    never had. Zero means the seams look like any other neighbouring pixels.

    A difference in intensity units, not a ratio. The first version divided by the
    interior gradient and blew up on near-uniform tiles (open water), giving a mean
    of 8.8 with a standard deviation of 107.

    The baseline isn't zero, since EuroSAT ships as JPEG already. That's fine: the
    monitor always compares against the reference, never against an absolute.
    """
    dv = np.abs(np.diff(gray, axis=1))          # horizontal neighbour differences
    dh = np.abs(np.diff(gray, axis=0))
    # Column j of dv is |I[:, j+1] - I[:, j]|; a block edge sits where j+1 is a
    # multiple of `block`.
    edge_cols = np.arange(dv.shape[1]) % block == (block - 1)
    edge_rows = np.arange(dh.shape[0]) % block == (block - 1)
    if edge_cols.sum() == 0 or (~edge_cols).sum() == 0:
        return 0.0
    edge = np.concatenate([dv[:, edge_cols].ravel(), dh[edge_rows, :].ravel()])
    inner = np.concatenate([dv[:, ~edge_cols].ravel(), dh[~edge_rows, :].ravel()])
    return float(edge.mean() - inner.mean())


def _hf_energy_ratio(gray: np.ndarray, cutoff: float = 0.5) -> float:
    """Share of spectral power above `cutoff` x Nyquist.

    Blur removes high frequencies, sensor noise adds them, so one number moving in
    opposite directions separates the two.

    JPEG lowers it as well (-0.47 sigma on `recompress`), which surprised me until I
    remembered that discarding high-frequency DCT coefficients *is* the compression.
    So this can't separate blurred from compressed on its own; blockiness does that.
    """
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    total = spectrum.sum()
    return float(spectrum[radius > cutoff].sum() / total) if total > 1e-12 else 0.0


def _entropy(gray: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of the intensity histogram, in bits. Flat scene = low."""
    hist, _ = np.histogram(gray, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-8 else 0.0


def image_features(img: np.ndarray) -> np.ndarray:
    """One tile (H, W, 3) float32 in [0, 1] -> a vector matching FEATURE_NAMES.

    Tiles of any size are accepted and resized to CANONICAL_SIZE first, so the
    returned vector is comparable regardless of what the provider delivered.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {img.shape}")

    if img.shape[0] != CANONICAL_SIZE or img.shape[1] != CANONICAL_SIZE:
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((CANONICAL_SIZE, CANONICAL_SIZE), Image.Resampling.BILINEAR)
        img = np.asarray(pil, dtype=np.float32) / 255.0

    gray = _grayscale(img)
    lap = _laplacian(gray)

    # Saturation as HSV defines it: how far the pixel is from grey.
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    # Local contrast: variability of 8x8 block means. Falls when detail is lost
    # at a scale larger than a single pixel -- which is what coarser ground
    # sample distance does.
    h, w = gray.shape
    bh, bw = h // 8, w // 8
    block_means = gray[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8).mean(axis=(1, 3))

    # Robust noise estimate: the MAD of the Laplacian. Real structure is sparse
    # in the Laplacian, so its median absolute deviation tracks the noise floor
    # rather than the content. 0.6745 converts MAD to a Gaussian sigma.
    mad = np.median(np.abs(lap - np.median(lap)))
    noise_sigma = float(mad / 0.6745 / np.sqrt(20.0))  # /sqrt(sum of squared kernel weights)

    return np.array(
        [
            img[..., 0].mean(), img[..., 1].mean(), img[..., 2].mean(),
            img[..., 0].std(), img[..., 1].std(), img[..., 2].std(),
            gray.mean(),
            sat.mean(),
            _correlation(img[..., 0], img[..., 1]),
            _correlation(img[..., 0], img[..., 2]),
            _correlation(img[..., 1], img[..., 2]),
            lap.var(),
            np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean(),
            _hf_energy_ratio(gray),
            _blockiness(gray),
            _entropy(gray),
            gray.std(),
            block_means.std(),
            float((gray < 0.02).mean()),
            float((gray > 0.98).mean()),
            noise_sigma,
        ],
        dtype=np.float32,
    )
