"""Score every batch with every detector and print them side by side.

    python -m monitoring.detect.run --build-reference
    python -m monitoring.detect.run

The question isn't whether a detector finds drift, they all do eventually. It's
whether it stays quiet on the control and on the label shifts (which cause no damage),
and still catches recompression, which costs 45 points while being invisible to colour
statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from monitoring.detect import embeddings as emb
from monitoring.detect.diagnose import diagnose, predicted_total_variation
from monitoring.detect.reference import (
    ALL_FEATURES, IMAGE_FEATURES, ReferenceProfile, load_feature_matrix,
    load_predicted_classes,
)
from monitoring.eurosat import EUROSAT_CLASSES
from monitoring.detect.tests import feature_drift, summarise
from monitoring.features.extract import OUTPUT_FEATURES

ROOT = Path(__file__).resolve().parents[3]
BATCH_ROOT = ROOT / "data" / "batches"
PROFILE_DIR = ROOT / "artifacts" / "reference_profile"
REFERENCE_BATCH = "reference_train"

# Calibrated against the control batch, not chosen a priori. See the report the
# CLI prints: `none` must sit below every one of these.
KS_THRESHOLD = 0.15

IMG_IDX = [ALL_FEATURES.index(n) for n in IMAGE_FEATURES]
OUT_IDX = [ALL_FEATURES.index(n) for n in OUTPUT_FEATURES]


def _predicted_counts(batch_dir: Path) -> np.ndarray:
    preds = load_predicted_classes(batch_dir)
    return np.array([sum(p == c for p in preds) for c in EUROSAT_CLASSES], dtype=float)


def score_batch(profile: ReferenceProfile, batch_dir: Path, n_permutations: int) -> dict:
    features = load_feature_matrix(batch_dir)
    embeddings = np.load(batch_dir / "embeddings.npy").astype(np.float64)

    # Layer 1 and 2, split apart on purpose: their *ratio* is the diagnostic.
    # Input moving while output stays put is the signature of a shift the model
    # does not care about -- which is precisely the alarm we must not raise.
    img = summarise(
        feature_drift(profile.features[:, IMG_IDX], features[:, IMG_IDX], IMAGE_FEATURES),
        KS_THRESHOLD,
    )
    out = summarise(
        feature_drift(profile.features[:, OUT_IDX], features[:, OUT_IDX], tuple(OUTPUT_FEATURES)),
        KS_THRESHOLD,
    )

    # Layer 3, three ways. The domain classifier runs on the PCA projection
    # rather than the raw 2048: with 1500 samples and 2048 dimensions a linear
    # model separates any two clouds perfectly, and would report drift always.
    ref_pca = profile.pca.transform(profile.embeddings)
    cur_pca = profile.pca.transform(embeddings)
    domain = emb.domain_classifier_drift(ref_pca, cur_pca)

    counts = _predicted_counts(batch_dir)
    tv = predicted_total_variation(profile.predicted_counts, counts)

    verdict = diagnose(
        image_ks=img["max_ks"],
        output_ks=out["max_ks"],
        predicted_tv=tv,
        top_features=img["top"],
        predicted_counts={c: int(n) for c, n in zip(EUROSAT_CLASSES, counts)},
        domain_auc=domain["auc"],
        n_current=len(features),
        n_reference=profile.n,
    )

    return {
        "batch": batch_dir.name,
        "image_features": img,
        "output_features": out,
        "predicted_tv": tv,
        "pca_ks": emb.pca_drift(profile.pca, profile.embeddings, embeddings),
        "mmd": emb.mmd_drift(profile.embeddings, embeddings, n_permutations=n_permutations),
        "domain_classifier": domain,
        "diagnosis": {
            "verdict": verdict.verdict,
            "action": verdict.action,
            "retrain": verdict.retrain,
            "confidence": verdict.confidence,
            "evidence": verdict.evidence,
            "warnings": verdict.warnings,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--profile", type=Path, default=PROFILE_DIR)
    parser.add_argument("--root", type=Path, default=BATCH_ROOT)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()

    if args.build_reference:
        profile = ReferenceProfile.fit_from_batch(args.root / REFERENCE_BATCH)
        profile.save(args.profile)
        print(json.dumps(profile.meta, indent=2))
        return

    profile = ReferenceProfile.load(args.profile)
    print(f"reference profile: {profile.meta['source_batch']} "
          f"(n={profile.n}, sha={profile.meta['content_sha256']})\n")

    harm = {r["batch"]: r for r in json.loads((args.root / "harm_summary.json").read_text())}
    batches = sorted(
        d for d in args.root.iterdir()
        if (d / "embeddings.npy").is_file() and d.name != REFERENCE_BATCH
    )

    header = (f"{'batch':28s} {'imgKS':>6s} {'outKS':>6s} {'predTV':>7s} {'ratio':>6s} "
              f"{'MMD z':>7s} {'domAUC':>7s} {'harm':>6s}  {'retrain':>7s}  verdict")
    print(header)
    print("-" * 130)

    results = []
    control_acc = harm.get("none_s0.50_test_n500", {}).get("acc_new_rule", 0.952)
    for batch_dir in batches:
        r = score_batch(profile, batch_dir, args.permutations)
        results.append(r)
        acc = harm.get(batch_dir.name, {}).get("acc_new_rule")
        damage = (control_acc - acc) * 100 if acc is not None else float("nan")
        ratio = r["predicted_tv"] / max(r["image_features"]["max_ks"], 1e-9)
        flag = "RETRAIN" if r["diagnosis"]["retrain"] else "no"
        print(f"{batch_dir.name.replace('_test_n500', ''):28s} "
              f"{r['image_features']['max_ks']:6.3f} {r['output_features']['max_ks']:6.3f} "
              f"{r['predicted_tv']:7.3f} {ratio:6.2f} {r['mmd']['z_vs_null']:7.1f} "
              f"{r['domain_classifier']['auc']:7.3f} {damage:6.1f}  {flag:>7s}  "
              f"{r['diagnosis']['verdict']}")

    (args.root / "detection_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.root / 'detection_summary.json'}")


if __name__ == "__main__":
    main()
