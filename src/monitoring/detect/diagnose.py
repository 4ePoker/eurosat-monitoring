"""Turns detector output into a verdict and an action.

Rules rather than a learned model: this authorises spending money and replacing a
production artifact, so it has to be readable.

The discriminator is a ratio. How far the predicted class distribution moved, over how
far the pixels moved:

    label shift    tiles are ordinary, only the proportions changed, so predictions
                   move a lot and pixels move moderately.  Measured: 1.24, 1.24
    data drift     every tile is altered, so pixels move hard and predictions less.
                   Measured: 0.29 to 0.73

The thresholds below all come from measured batches. The control sets the noise floor,
the two label shifts set the ratio boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Calibrated: at n=500 the control batch reaches 0.088 (KS) and 0.068 (TV).
# Anything under these is a quiet day.
KS_NOISE_FLOOR = 0.15
TV_NOISE_FLOOR = 0.15

#: The window size the floors above were calibrated at.
CALIBRATION_N = 500
REFERENCE_N = 1000


def scaled_noise_floor(n_current: int, n_reference: int = REFERENCE_N,
                       base: float = KS_NOISE_FLOOR) -> float:
    """Noise floor, corrected for how many samples the window actually has.

    A fixed threshold turned out to be a bug. Real traffic gave windows of ~140
    sampled requests instead of the bench's 500, and the control batch (no drift at
    all) went from max KS 0.088 to 0.134, p95 0.183. Against a fixed 0.15 that's a
    false alarm on more than one quiet day in twenty.

    KS is the largest gap between two empirical distributions and small samples are
    lumpier, so the gap is bigger by construction. Its null scales with
    sqrt((n+m)/(n*m)) and the floor scales the same way. The constant is pinned so
    n = CALIBRATION_N still gives the threshold the bench validated.
    """
    if n_current <= 0:
        return base
    calib = ((CALIBRATION_N + n_reference) / (CALIBRATION_N * n_reference)) ** 0.5
    actual = ((n_current + n_reference) / (n_current * n_reference)) ** 0.5
    return base * actual / calib

# Calibrated: damaging batches top out at 0.72, both label shifts sit at 1.24.
LABEL_SHIFT_RATIO = 1.0

# Output-feature drift above this maps to damage beyond ~40 accuracy points on
# the bench; below it, damage clusters around 30. Calibrated, like everything
# else here, against measured batches rather than chosen.
OUTPUT_KS_SEVERE = 0.50

# `blockiness` dominating the table is a compression fingerprint, and nothing in
# nature produces it. Measured 0.861 on `recompress`, 0.00x everywhere else.
BLOCKINESS_KS = 0.30

# Below this many samples of a class, its per-class accuracy is not measurable
# and the monitor must say so instead of implying everything is fine.
MIN_CLASS_SAMPLES = 30


@dataclass(frozen=True)
class Diagnosis:
    verdict: str
    action: str
    retrain: bool
    confidence: str
    evidence: list[str]
    warnings: list[str]


def diagnose(
    image_ks: float,
    output_ks: float,
    predicted_tv: float,
    top_features: list[tuple[str, float, str]],
    predicted_counts: dict[str, int],
    domain_auc: float | None = None,
    n_current: int = CALIBRATION_N,
    n_reference: int = REFERENCE_N,
) -> Diagnosis:
    """Turn the detector outputs into a verdict, an action, and a retrain flag."""
    evidence: list[str] = []
    warnings: list[str] = []
    ratio = predicted_tv / max(image_ks, 1e-9)
    ks_floor = scaled_noise_floor(n_current, n_reference, KS_NOISE_FLOOR)
    tv_floor = scaled_noise_floor(n_current, n_reference, TV_NOISE_FLOOR)

    # A class the batch barely contains cannot be monitored, whatever the
    # headline number says. This fires independently of the verdict because it
    # is a blind spot, not a drift.
    for name, count in sorted(predicted_counts.items(), key=lambda kv: kv[1]):
        if count < MIN_CLASS_SAMPLES:
            warnings.append(
                f"only {count} predictions of {name}: its accuracy is not measurable in this window"
            )

    top_name, top_ks = (top_features[0][0], top_features[0][1]) if top_features else ("", 0.0)

    # --- rule 1: is anything happening at all? ----------------------------
    if image_ks < ks_floor and output_ks < ks_floor and predicted_tv < tv_floor:
        return Diagnosis(
            verdict="no drift detected",
            action="none",
            retrain=False,
            confidence="high",
            evidence=[f"image KS {image_ks:.3f}, output KS {output_ks:.3f}, "
                      f"predicted TV {predicted_tv:.3f} -- all below the floor "
                      f"{ks_floor:.3f}, scaled for a window of {n_current} samples"],
            warnings=warnings + [
                "concept drift is invisible to every signal here: if the labelling "
                "rule changed, this verdict is wrong and only a freshly labelled "
                "sample would reveal it"
            ],
        )

    # --- rule 2: compression fingerprint = a broken pipeline, not a stale model
    if top_name == "blockiness" and top_ks >= BLOCKINESS_KS:
        return Diagnosis(
            verdict="data quality: upstream compression changed",
            action="fix the ingestion pipeline; do NOT retrain",
            retrain=False,
            confidence="high",
            evidence=[
                f"blockiness dominates the feature table (KS {top_ks:.3f})",
                "8x8 block artefacts are produced by JPEG encoders, not by landscapes",
                "retraining here would teach the model to accept corrupted input "
                "and hide the defect inside the weights",
            ],
            warnings=warnings,
        )

    # --- rule 3: predictions moved further than pixels = the mix changed ---
    if ratio >= LABEL_SHIFT_RATIO:
        return Diagnosis(
            verdict="label shift: the class mix changed, the imagery did not",
            action="apply prior correction at serving time; do NOT retrain",
            retrain=False,
            confidence="high" if ratio >= 1.15 else "medium",
            evidence=[
                f"predicted-distribution TV {predicted_tv:.3f} against image KS "
                f"{image_ks:.3f} (ratio {ratio:.2f}, label shifts measured at 1.24)",
                "individual tiles look like training data; only their proportions changed",
                "measured damage for this pattern was under 2 accuracy points",
            ],
            warnings=warnings,
        )

    # --- rule 4: the imagery itself changed --------------------------------
    # Severity reads output drift rather than input drift, because across the bench
    # output KS tracks real accuracy loss at Spearman +0.95 and image KS only +0.78.
    # RESISC45 is the example: image features move moderately (0.49) and it costs 63
    # points, while haze moves them more than anything (0.86) and costs 30.
    severity = "severe" if output_ks >= OUTPUT_KS_SEVERE else "moderate"
    return Diagnosis(
        verdict=f"data drift ({severity}): the incoming imagery changed",
        action="candidate for retraining -- confirm persistence across windows first",
        retrain=True,
        confidence="high" if output_ks >= OUTPUT_KS_SEVERE else "medium",
        evidence=[
            f"output KS {output_ks:.3f} (the damage predictor: Spearman +0.95 "
            f"against measured accuracy loss), image KS {image_ks:.3f}, ratio {ratio:.2f}",
            "top movers: " + ", ".join(f"{n} {v:.2f}{d}" for n, v, d in top_features[:3]),
        ]
        + ([f"domain classifier AUC {domain_auc:.3f}"] if domain_auc is not None else []),
        warnings=warnings + [
            "a single window is not a trend: transient causes (haze, cloud) revert "
            "on their own and retraining on them makes the model worse"
        ],
    )


def predicted_total_variation(reference_counts: np.ndarray, current_counts: np.ndarray) -> float:
    """Total variation distance between two predicted-class distributions, in [0, 1]."""
    p = reference_counts / max(reference_counts.sum(), 1)
    q = current_counts / max(current_counts.sum(), 1)
    return float(0.5 * np.abs(p - q).sum())
