"""Pixel-level transforms that imitate real causes of drift in satellite imagery.

Each takes an image as float32 RGB in [0, 1] and a severity in [0, 1]. Two rules hold
everywhere: severity 0 is the identity, and the result is deterministic given the same
inputs and RNG.

The physics is simplified. What matters is that each transform moves the kind of
statistic the real cause would move.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Sentinel-2 RGB tiles: band order is R, G, B (EuroSAT's JPEG version).
R, G, B = 0, 1, 2


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0, out=x if x.flags.writeable else None)


def identity(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """The control. Used by the `none` scenario to prove the monitor's false-positive rate."""
    return img


def seasonal_shift(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Green-up / senescence: the same land, photographed in a different season.

    Autumn drains chlorophyll: green drops, red rises as vegetation browns, and the
    tile desaturates as fields go from striped green to stubble. The label doesn't
    change -- an AnnualCrop parcel in October is still AnnualCrop.
    """
    out = img.copy()
    out[..., G] *= 1.0 - 0.45 * severity
    out[..., R] *= 1.0 + 0.25 * severity
    out[..., B] *= 1.0 + 0.05 * severity
    # Pull towards the tile's own mean: fields lose their internal contrast
    # once the crop is cut.
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = mean + (out - mean) * (1.0 - 0.30 * severity)
    return _clip(out)


def radiometric_shift(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """A different sensor, or a different processing level, feeding the same pipeline.

    Swapping Sentinel-2 for Landsat-9, or L1C for L2A, changes gain, offset and tone
    curve. Same scene, different numbers. Modelled as gain + bias + gamma.
    """
    out = img.copy()
    gain = 1.0 + 0.35 * severity
    bias = 0.06 * severity
    gamma = 1.0 + 0.55 * severity
    out = np.power(_clip(out * gain + bias), gamma)
    # Sensors differ per band, not uniformly -- a global stretch would be too
    # easy to detect and too easy to correct.
    out[..., B] *= 1.0 + 0.12 * severity
    return _clip(out)


def resolution_loss(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Coarser ground sample distance: 10 m/px Sentinel-2 replaced by 30 m/px Landsat.

    Same ground, same 64x64 tile, but the detail is gone. Several EuroSAT classes
    are separable mostly by texture (Highway vs River, PermanentCrop vs
    HerbaceousVegetation), so this hurts accuracy while barely touching colour.
    """
    if severity <= 0:
        return img
    h, w = img.shape[:2]
    factor = 1.0 + 3.0 * severity  # up to a ~4x loss of effective resolution
    small = (max(1, int(round(w / factor))), max(1, int(round(h / factor))))
    pil = Image.fromarray((_clip(img.copy()) * 255).astype(np.uint8))
    down = pil.resize(small, Image.BILINEAR)
    up = down.resize((w, h), Image.BILINEAR)
    return np.asarray(up, dtype=np.float32) / 255.0


def atmospheric_haze(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Thin cloud, cirrus or aerosol: the scene seen through a white veil.

    Additive path radiance plus reduced contrast, weighted towards blue since
    Rayleigh scattering is stronger at short wavelengths. This one comes and goes
    with the weather, which is why the trigger needs persistence.
    """
    out = img.copy()
    veil = np.array([0.82, 0.88, 1.00], dtype=np.float32)  # bluish white
    alpha = 0.55 * severity
    out = out * (1.0 - alpha) + veil * alpha
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = mean + (out - mean) * (1.0 - 0.35 * severity)
    return _clip(out)


def sensor_noise(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """Lower signal-to-noise: an older sensor, a darker scene, higher gain."""
    if severity <= 0:
        return img
    sigma = 0.06 * severity
    return _clip(img + rng.normal(0.0, sigma, size=img.shape).astype(np.float32))


def jpeg_recompress(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    """An upstream ETL change nobody told you about.

    Somebody lowers the JPEG quality in the tiling service to save storage. Nothing
    in the world changed and the images still look fine to a human, but the
    high-frequency content the CNN relies on is now full of 8x8 artefacts.
    """
    if severity <= 0:
        return img
    quality = int(round(95 - 80 * severity))  # 95 -> 15
    pil = Image.fromarray((_clip(img.copy()) * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=max(1, quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


def compose(*fns):
    """Chain transforms, all sharing the same severity and RNG.

    Real causes are rarely pure: changing satellite provider changes
    calibration *and* resolution *and* noise at once. Composing keeps the
    scenario honest instead of testing the detector on a single clean axis.
    """

    def _composed(img: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
        for fn in fns:
            img = fn(img, severity, rng)
        return img

    _composed.__name__ = "+".join(fn.__name__ for fn in fns)
    return _composed
