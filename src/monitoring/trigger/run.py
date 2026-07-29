"""The nightly job: read the day, fold it into the state, decide, log.

    python -m monitoring.trigger.run --window production_20260728
    python -m monitoring.trigger.run --simulate none,none,haze,new_sensor_s0.50

Meant for cron:

    0 2 * * *  cd /path/to/repo && .venv/bin/python -m monitoring.trigger.run \
                   --window "$(date -d yesterday +production_%Y%m%d)"

Firing writes a proposal, not a training job. If the drift is a new sensor, retraining
on the existing training set reproduces the existing model, since the new distribution
isn't in it. The images we kept have no labels. So the first step is human labelling.

It proposes and trains. It never deploys; promotion has its own gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from monitoring.trigger.policy import (
    COOLDOWN_DAYS, LEVEL_OFF, LEVEL_ON, MIN_WINDOW_SAMPLES, PERSISTENCE_DAYS,
    Observation, TriggerState, decide,
)

ROOT = Path(__file__).resolve().parents[3]
BATCH_ROOT = ROOT / "data" / "batches"
ARTIFACTS = ROOT / "artifacts"
STATE_PATH = ARTIFACTS / "trigger_state.json"
DECISIONS_PATH = ARTIFACTS / "decisions.jsonl"
PROPOSALS_DIR = ARTIFACTS / "proposals"

RETRAIN_VERDICT_PREFIX = "data drift"


def observation_from_batch(batch: str, day: str) -> Observation:
    """Turn a scored batch into one day's observation."""
    summary = json.loads((BATCH_ROOT / "detection_summary.json").read_text())
    record = next((r for r in summary if r["batch"].startswith(batch)), None)
    if record is None:
        raise SystemExit(
            f"no detection result for {batch!r}. Run `python -m monitoring.detect.run` first."
        )
    n = sum(1 for _ in csv.DictReader((BATCH_ROOT / record["batch"] / "features.csv").open()))
    return Observation(
        day=day,
        verdict=record["diagnosis"]["verdict"],
        retrain_eligible=bool(record["diagnosis"]["retrain"]),
        output_ks=record["output_features"]["max_ks"],
        image_ks=record["image_features"]["max_ks"],
        n_samples=n,
        batch=record["batch"],
    )


def write_proposal(decision, state: TriggerState) -> Path:
    """The artefact a human acts on. Names the work, does not do it."""
    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    obs = decision.observation
    batch_dir = BATCH_ROOT / obs["batch"]

    to_label = []
    features = batch_dir / "features.csv"
    if features.is_file():
        to_label = [r["filename"] for r in csv.DictReader(features.open())]

    proposal = {
        "created": decision.day,
        "reason": decision.reason,
        "evidence": {
            "verdict": obs["verdict"],
            "output_ks": obs["output_ks"],
            "image_ks": obs["image_ks"],
            "window_samples": obs["n_samples"],
            "consecutive_days": PERSISTENCE_DAYS,
            "recent_days": state.observations[-PERSISTENCE_DAYS:],
        },
        "next_steps": [
            {
                "step": 1,
                "action": "label the sampled tiles",
                "detail": (
                    f"{len(to_label)} tiles under {batch_dir / 'images'} need ground truth. "
                    "This is the slow, human, expensive step and it cannot be skipped: the "
                    "new distribution is absent from the existing training set, so retraining "
                    "without these produces the model we already have."
                ),
                "blocking": True,
            },
            {
                "step": 2,
                "action": "add the labelled tiles to the training data",
                "detail": "Project1/data — then `dvc add` so the new dataset version is tracked.",
                "blocking": True,
            },
            {
                "step": 3,
                "action": "retrain",
                "detail": "cd ../Project1 && dvc repro",
                "blocking": False,
            },
            {
                "step": 4,
                "action": "rebuild the reference profile against the new model",
                "detail": (
                    "python -m monitoring.detect.run --build-reference. Mandatory: the "
                    "embeddings, the PCA basis and every calibrated threshold belong to "
                    "the old model and mean nothing against a new one."
                ),
                "blocking": True,
            },
            {
                "step": 5,
                "action": "re-measure the bench and register the bundle",
                "detail": "python -m monitoring.registry.wandb_registry register",
                "blocking": False,
            },
            {
                "step": 6,
                "action": "promote, if it beats the champion",
                "detail": (
                    "python -m monitoring.registry.wandb_registry promote --version v1 "
                    "--alias production --min-accuracy <champion>. Gated on purpose: this "
                    "trigger may propose and train, never deploy."
                ),
                "blocking": False,
            },
        ],
        "tiles_awaiting_labels": to_label[:20],
        "tiles_awaiting_labels_total": len(to_label),
    }
    path = PROPOSALS_DIR / f"retrain-proposal-{decision.day}.json"
    path.write_text(json.dumps(proposal, indent=2) + "\n")
    return path


def run_day(batch: str, day: str, state_path: Path) -> None:
    state = TriggerState.load(state_path)
    obs = observation_from_batch(batch, day)
    state, decision = decide(state, obs)
    state.save(state_path)

    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_PATH.open("a") as fh:
        fh.write(json.dumps(asdict(decision)) + "\n")

    flag = "**ACTION**" if decision.action != "none" else "  ---   "
    print(f"{decision.day}  {flag}  {obs.batch[:34]:34s} "
          f"outKS={obs.output_ks:.3f} n={obs.n_samples:4d}  {decision.reason}")

    if decision.action == "propose_retraining":
        path = write_proposal(decision, state)
        print(f"\n  wrote {path}")


def simulate(sequence: list[str], state_path: Path, start: str) -> None:
    """Replay a sequence of days to see the policy behave.

    The mechanisms only exist because of what they prevent, and what they
    prevent takes more than one day to show. A simulation is the only way to
    check them without waiting a fortnight.
    """
    if state_path.is_file():
        state_path.unlink()
    if DECISIONS_PATH.is_file():
        DECISIONS_PATH.unlink()

    day = date.fromisoformat(start)
    print(f"thresholds: on>={LEVEL_ON} hold>={LEVEL_OFF}  persistence={PERSISTENCE_DAYS}d  "
          f"cooldown={COOLDOWN_DAYS}d  min window={MIN_WINDOW_SAMPLES}\n")
    for batch in sequence:
        run_day(batch.strip(), day.isoformat(), state_path)
        day += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default=None, help="batch name for one day")
    parser.add_argument("--day", default=date.today().isoformat())
    parser.add_argument("--simulate", default=None, help="comma-separated batch names, one per day")
    parser.add_argument("--start", default="2026-08-01", help="first day of a simulation")
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()

    if args.history:
        if not DECISIONS_PATH.is_file():
            raise SystemExit("no decisions recorded yet")
        for line in DECISIONS_PATH.read_text().splitlines():
            d = json.loads(line)
            print(f"{d['day']}  {d['action']:20s} {d['reason']}")
        return

    if args.simulate:
        simulate(args.simulate.split(","), args.state, args.start)
    elif args.window:
        run_day(args.window, args.day, args.state)
    else:
        parser.error("give --window or --simulate")


if __name__ == "__main__":
    main()
