"""Two-sample KS test per feature, with a Benjamini-Hochberg correction.

KS rather than a standardised mean difference, because one feature is constant across
the reference and dividing by its zero standard deviation produced an "effect size" of
976 for a shift of two hundredths of a percent.

The p-values are diagnostic only. With 500 samples a day almost anything is
significant, so the trigger reads the statistic and its threshold is calibrated
against the control batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class FeatureDrift:
    name: str
    ks: float           # KS statistic in [0, 1]: the effect size that decides
    p_value: float
    p_adjusted: float   # Benjamini-Hochberg, across all features in the batch
    ref_mean: float
    cur_mean: float

    @property
    def direction(self) -> str:
        return "+" if self.cur_mean >= self.ref_mean else "-"


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Control the false discovery rate across the features tested together.

    Testing 25 features at p<0.05 gives roughly one spurious hit per batch by
    construction. BH rescales so that the *expected proportion* of false hits
    among those declared significant stays under the level, which is the
    guarantee that actually matters when you test a panel every day.
    """
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    # enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def feature_drift(
    reference: np.ndarray, current: np.ndarray, names: tuple[str, ...]
) -> list[FeatureDrift]:
    """KS-test every column of `current` against the same column of `reference`."""
    if reference.shape[1] != current.shape[1] != len(names):
        raise ValueError("reference, current and names must agree on the feature count")

    stats_, ps = [], []
    for j in range(reference.shape[1]):
        result = stats.ks_2samp(reference[:, j], current[:, j])
        stats_.append(float(result.statistic))
        ps.append(float(result.pvalue))
    adjusted = benjamini_hochberg(np.array(ps))

    return [
        FeatureDrift(
            name=names[j], ks=stats_[j], p_value=ps[j], p_adjusted=float(adjusted[j]),
            ref_mean=float(reference[:, j].mean()), cur_mean=float(current[:, j].mean()),
        )
        for j in range(len(names))
    ]


def summarise(drifts: list[FeatureDrift], threshold: float) -> dict:
    """Collapse the per-feature table into the few numbers an alert carries."""
    ks = np.array([d.ks for d in drifts])
    above = [d for d in drifts if d.ks >= threshold]
    return {
        "max_ks": float(ks.max()),
        "mean_ks": float(ks.mean()),
        "n_above_threshold": len(above),
        "n_features": len(drifts),
        "top": [
            (d.name, round(d.ks, 3), d.direction)
            for d in sorted(drifts, key=lambda d: -d.ks)[:5]
        ],
    }
