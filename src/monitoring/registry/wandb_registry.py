"""Model registry on W&B, for a pair rather than a model.

The reference profile isn't independent of the model: its embeddings came out of it,
its PCA basis was fitted on those embeddings, its thresholds were calibrated on that
combination. Swap the model and all of it is wrong, in a way that keeps producing
plausible numbers.

So the unit of promotion is the bundle. Registered together, promoted together.

    python -m monitoring.registry.wandb_registry register
    python -m monitoring.registry.wandb_registry promote --version v0 --alias production

Defaults to offline mode so a fresh clone runs without an account. Promotion needs
`wandb login` and WANDB_MODE=online, since moving an alias is a server-side operation.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = ROOT / "artifacts" / "reference_profile"
BATCH_ROOT = ROOT / "data" / "batches"
DEFAULT_MODEL = ROOT.parent / "Project5" / "models" / "eurosat_resnet50.static_int8.onnx"

PROJECT = os.environ.get("WANDB_PROJECT", "eurosat-mlops")
ENTITY = os.environ.get("WANDB_ENTITY") or None
MODE = os.environ.get("WANDB_MODE", "offline")

MODEL_ARTIFACT = "eurosat-classifier"
PROFILE_ARTIFACT = "reference-profile"


@dataclass(frozen=True)
class BundleMetrics:
    """Everything a human needs to decide whether this bundle may go to production."""

    control_accuracy: float
    reference_accuracy: float
    n_batches_benchmarked: int
    correct_verdicts: int
    thresholds: dict[str, float]
    detector_calibration: list[dict]


def collect_metrics() -> BundleMetrics:
    """Read the bench results off disk, to register alongside the artifacts.

    A registry entry saying only "resnet50, 24 MB" can't support a promotion
    decision. What a reviewer needs is which batches it was measured on, what it got
    right, and what the thresholds were at the time.
    """
    from monitoring.detect import diagnose as dg

    harm = {r["batch"]: r for r in json.loads((BATCH_ROOT / "harm_summary.json").read_text())}
    detection = {r["batch"]: r for r in json.loads((BATCH_ROOT / "detection_summary.json").read_text())}

    control = harm["none_s0.50_test_n500"]["acc_new_rule"]
    calibration = []
    for name, det in sorted(detection.items()):
        acc = harm.get(name, {}).get("acc_new_rule")
        calibration.append({
            "batch": name,
            "damage_points": round((control - acc) * 100, 1) if acc is not None else None,
            "image_ks": round(det["image_features"]["max_ks"], 3),
            "output_ks": round(det["output_features"]["max_ks"], 3),
            "predicted_tv": round(det["predicted_tv"], 3),
            "domain_auc": round(det["domain_classifier"]["auc"], 3),
            "verdict": det["diagnosis"]["verdict"],
            "retrain": det["diagnosis"]["retrain"],
        })

    return BundleMetrics(
        control_accuracy=control,
        reference_accuracy=harm["reference_train"]["acc_new_rule"],
        n_batches_benchmarked=len(calibration),
        correct_verdicts=sum(1 for c in calibration if c["verdict"]),
        thresholds={
            "ks_noise_floor": dg.KS_NOISE_FLOOR,
            "tv_noise_floor": dg.TV_NOISE_FLOOR,
            "label_shift_ratio": dg.LABEL_SHIFT_RATIO,
            "output_ks_severe": dg.OUTPUT_KS_SEVERE,
            "blockiness_ks": dg.BLOCKINESS_KS,
            "min_class_samples": float(dg.MIN_CLASS_SAMPLES),
        },
        detector_calibration=calibration,
    )


def register(model_path: Path, profile_dir: Path, notes: str = "") -> dict:
    """Log the bundle as one new version of each artefact, with lineage between them."""
    metrics = collect_metrics()
    profile_meta = json.loads((profile_dir / "meta.json").read_text())

    run = wandb.init(
        project=PROJECT, entity=ENTITY, mode=MODE,
        job_type="register-bundle", name="register-bundle",
        notes=notes or "monitoring bundle: served model + the reference it is measured against",
    )

    model_art = wandb.Artifact(
        MODEL_ARTIFACT, type="model",
        description="EuroSAT ResNet50, ONNX static INT8, two outputs (logits + 2048-d features).",
        metadata={
            "format": "onnx",
            "quantization": "static-int8",
            "outputs": ["logits", "features"],
            "embedding_dim": profile_meta["embedding_dim"],
            "size_mb": round(model_path.stat().st_size / 1e6, 1),
            "test_accuracy_control_batch": metrics.control_accuracy,
        },
    )
    model_art.add_file(str(model_path), name=model_path.name)
    run.log_artifact(model_art, aliases=["latest"])

    # Draws the lineage edge in W&B's graph. It's a server call, so offline runs
    # raise on it; the dependency itself is recorded in the profile's metadata below
    # either way (`built_from_model`).
    if MODE == "online":
        run.use_artifact(model_art)

    profile_art = wandb.Artifact(
        PROFILE_ARTIFACT, type="reference-profile",
        description=(
            "What 'normal' is: reference features, embeddings, the PCA basis, and the "
            "thresholds calibrated against them. Only valid for the model above."
        ),
        metadata={
            **profile_meta,
            "built_from_model": model_art.name,
            "thresholds": metrics.thresholds,
            "benchmark_batches": metrics.n_batches_benchmarked,
        },
    )
    profile_art.add_dir(str(profile_dir))
    run.log_artifact(profile_art, aliases=["latest"])

    run.summary.update({
        "control_accuracy": metrics.control_accuracy,
        "reference_accuracy": metrics.reference_accuracy,
        "batches_benchmarked": metrics.n_batches_benchmarked,
        **{f"threshold/{k}": v for k, v in metrics.thresholds.items()},
    })
    # The full calibration evidence, as a browsable table rather than a blob.
    run.log({"detector_calibration": wandb.Table(
        columns=list(metrics.detector_calibration[0]),
        data=[list(r.values()) for r in metrics.detector_calibration],
    )})

    run_url = run.url
    run.finish()
    return {
        "mode": MODE,
        "project": PROJECT,
        "run_url": run_url,
        "model_artifact": MODEL_ARTIFACT,
        "profile_artifact": PROFILE_ARTIFACT,
        "profile_sha": profile_meta["content_sha256"],
        "control_accuracy": metrics.control_accuracy,
    }


def promote(version: str, alias: str, min_accuracy: float | None = None) -> dict:
    """Move `alias` onto a version of both artifacts, after a gate check.

    The gate is here rather than in the retraining job because promotion is the
    irreversible step, so it's the one that has to be able to refuse.
    """
    if MODE != "online":
        raise SystemExit(
            "promote needs WANDB_MODE=online and `wandb login`: moving an alias is a "
            "server-side operation. Offline runs can register versions and sync them "
            "later, but the registry pointer lives on the server."
        )

    api = wandb.Api()
    prefix = f"{ENTITY + '/' if ENTITY else ''}{PROJECT}"
    model = api.artifact(f"{prefix}/{MODEL_ARTIFACT}:{version}")
    profile = api.artifact(f"{prefix}/{PROFILE_ARTIFACT}:{version}")

    candidate = model.metadata.get("test_accuracy_control_batch")
    if min_accuracy is not None and (candidate is None or candidate < min_accuracy):
        raise SystemExit(
            f"refusing to promote: candidate accuracy {candidate} is below the "
            f"required {min_accuracy}. A challenger that does not beat the champion "
            f"does not get the alias."
        )

    # Both, or neither. Half a bundle is not deployable.
    for artefact in (model, profile):
        artefact.aliases.append(alias)
        artefact.save()

    return {"alias": alias, "version": version,
            "model": model.name, "profile": profile.name, "accuracy": candidate}


def status() -> dict:
    """What exists and where the aliases point."""
    if MODE != "online":
        local = sorted((ROOT / "wandb").glob("*run-*")) if (ROOT / "wandb").is_dir() else []
        return {
            "mode": MODE,
            "note": "offline: versions live under ./wandb until `wandb sync`",
            "local_runs": [p.name for p in local[-5:]],
        }
    api = wandb.Api()
    prefix = f"{ENTITY + '/' if ENTITY else ''}{PROJECT}"
    out: dict = {"mode": MODE, "project": PROJECT, "artifacts": {}}
    for name in (MODEL_ARTIFACT, PROFILE_ARTIFACT):
        versions = api.artifact_collection("model" if name == MODEL_ARTIFACT
                                           else "reference-profile", f"{prefix}/{name}").artifacts()
        out["artifacts"][name] = [
            {"version": a.version, "aliases": a.aliases, "created": str(a.created_at)}
            for a in versions
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="log the current bundle as a new version")
    p_reg.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p_reg.add_argument("--profile", type=Path, default=PROFILE_DIR)
    p_reg.add_argument("--notes", default="")

    p_pro = sub.add_parser("promote", help="move an alias onto a version (online only)")
    p_pro.add_argument("--version", required=True)
    p_pro.add_argument("--alias", default="production")
    p_pro.add_argument("--min-accuracy", type=float, default=None)

    sub.add_parser("status", help="show versions and where aliases point")

    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(args.model, args.profile, args.notes), indent=2))
    elif args.command == "promote":
        print(json.dumps(promote(args.version, args.alias, args.min_accuracy), indent=2))
    else:
        print(json.dumps(status(), indent=2))


if __name__ == "__main__":
    main()
