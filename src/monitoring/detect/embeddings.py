"""Three ways to ask whether a 2048-dimensional cloud of points moved.

    (a) PCA + KS    project onto the reference's main axes and test those
    (b) MMD         one kernel two-sample test, with a permutation null
    (c) domain      train a classifier to tell reference from current; AUC is
        classifier  the score, 0.5 means indistinguishable

All three are implemented so the choice could be settled by measurement. You can't
KS-test 2048 dimensions directly: at p<0.05 that's ~100 features firing by chance
every day.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from monitoring.detect.tests import benjamini_hochberg
from scipy import stats


# ---------------------------------------------------------------- (a) PCA + KS

def fit_pca(reference_embeddings: np.ndarray, n_components: int = 32) -> PCA:
    """Fit the projection ONCE, on the reference. This basis is an artefact.

    Refitting it later silently rebases every historical drift number, so it
    belongs in the model registry next to the model, with a version, not in a
    scratch variable. See brick 4.
    """
    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(reference_embeddings)
    return pca


def pca_drift(pca: PCA, reference: np.ndarray, current: np.ndarray) -> dict:
    """KS-test each principal component of the embeddings."""
    ref_p, cur_p = pca.transform(reference), pca.transform(current)
    ks, ps = [], []
    for j in range(ref_p.shape[1]):
        r = stats.ks_2samp(ref_p[:, j], cur_p[:, j])
        ks.append(float(r.statistic))
        ps.append(float(r.pvalue))
    adjusted = benjamini_hochberg(np.array(ps))
    ks = np.array(ks)
    return {
        "max_ks": float(ks.max()),
        "mean_ks": float(ks.mean()),
        "n_significant": int((adjusted < 0.05).sum()),
        "n_components": len(ks),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


# -------------------------------------------------------------------- (b) MMD

def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    sq = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def mmd_drift(
    reference: np.ndarray, current: np.ndarray, n_permutations: int = 200, seed: int = 0
) -> dict:
    """Maximum Mean Discrepancy with a permutation test.

    MMD measures the distance between the two samples' means in a kernel feature
    space; zero means identical. The permutation test says how large that distance
    gets by chance, which is what turns it into a decision.

    Bandwidth from the median heuristic rather than a hyperparameter, so there's one
    fewer knob someone can turn later without noticing.
    """
    rng = np.random.default_rng(seed)
    m, n = len(reference), len(current)
    z = np.vstack([reference, current])

    sq = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1) if len(z) < 400 else None
    if sq is None:  # memory-friendlier for the sizes we actually use
        sq = (z * z).sum(1)[:, None] + (z * z).sum(1)[None, :] - 2.0 * z @ z.T
        sq = np.maximum(sq, 0.0)
    median_sq = np.median(sq[sq > 0])
    gamma = 1.0 / median_sq if median_sq > 0 else 1.0
    k = np.exp(-gamma * sq)

    def mmd2(idx_a: np.ndarray, idx_b: np.ndarray) -> float:
        kaa = k[np.ix_(idx_a, idx_a)]
        kbb = k[np.ix_(idx_b, idx_b)]
        kab = k[np.ix_(idx_a, idx_b)]
        a, b = len(idx_a), len(idx_b)
        # unbiased: drop the diagonal, which is a self-similarity of 1
        term_a = (kaa.sum() - np.trace(kaa)) / (a * (a - 1))
        term_b = (kbb.sum() - np.trace(kbb)) / (b * (b - 1))
        return float(term_a + term_b - 2.0 * kab.mean())

    observed = mmd2(np.arange(m), np.arange(m, m + n))

    null = np.empty(n_permutations)
    all_idx = np.arange(m + n)
    for i in range(n_permutations):
        perm = rng.permutation(all_idx)
        null[i] = mmd2(perm[:m], perm[m:])

    # +1 in numerator and denominator: a permutation test can never report p=0.
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return {
        "mmd2": observed,
        "p_value": p_value,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        # How many null standard deviations out the observation sits: a scale-free
        # magnitude, so batches remain comparable to each other.
        "z_vs_null": float((observed - null.mean()) / (null.std() + 1e-12)),
    }


# ------------------------------------------------------- (c) domain classifier

def domain_classifier_drift(
    reference: np.ndarray, current: np.ndarray, seed: int = 0
) -> dict:
    """Can a classifier tell the two batches apart? AUC is the answer.

    0.5 = indistinguishable, 1.0 = trivially separable. Cross-validated, so the
    score measures generalisable separation rather than memorisation -- without
    that, any classifier with enough capacity reaches 1.0 on any two samples and
    the metric is worthless.
    """
    x = np.vstack([reference, current])
    y = np.concatenate([np.zeros(len(reference)), np.ones(len(current))])

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, scores))

    # Which inputs carried the separation -- free interpretability, and the
    # reason this method can diagnose while MMD only detects.
    model.fit(x, y)
    weight = np.abs(model.coef_[0])
    return {
        "auc": auc,
        "top_dims": np.argsort(-weight)[:5].tolist(),
        "top_weights": np.round(np.sort(weight)[::-1][:5], 3).tolist(),
    }
