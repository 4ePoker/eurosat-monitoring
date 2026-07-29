"""The state machine between "drift detected" and "spend money".

Everything upstream is memoryless, which is right for a detector and wrong for a
decision. A single window can't tell weather from a new satellite, and the response to
those is opposite.

Four brakes:

    type gate     only data drift is a retraining problem
    hysteresis    two thresholds, so the system doesn't chatter across one
    persistence   N consecutive days, because haze passes on its own
    cooldown      a retraining cycle outlives a single window
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Calibrated on the bench, in the output-feature drift that predicts damage best
# (Spearman +0.95). Damaging batches begin at 0.38 (`haze`, 30 points); the
# control sits at 0.043 and the two harmless label shifts at 0.28 and 0.36 --
# those are excluded by the type gate, but the low threshold is placed above
# them anyway so that a mis-typed day cannot keep a run alive.
LEVEL_ON = 0.38
LEVEL_OFF = 0.25

PERSISTENCE_DAYS = 3
COOLDOWN_DAYS = 14

#: A window this small cannot support a decision whatever it says -- brick 5
#: measured the noise floor climbing steeply below ~200 samples.
MIN_WINDOW_SAMPLES = 100


@dataclass
class Observation:
    """One day's worth of monitoring, reduced to what the decision needs."""

    day: str
    verdict: str
    retrain_eligible: bool      # the type gate: brick 3 said "data drift"
    output_ks: float            # the level, and the damage predictor
    image_ks: float
    n_samples: int
    batch: str = ""

    @property
    def qualifies(self) -> bool:
        return (
            self.retrain_eligible
            and self.n_samples >= MIN_WINDOW_SAMPLES
            and self.output_ks >= LEVEL_ON
        )

    @property
    def sustains(self) -> bool:
        """Enough to keep an existing run alive, though not to start one."""
        return self.retrain_eligible and self.output_ks >= LEVEL_OFF


@dataclass
class TriggerState:
    """The memory. Everything the policy needs that a single window cannot hold."""

    consecutive_days: int = 0
    run_started: str | None = None
    last_action: str | None = None
    observations: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "TriggerState":
        if not path.is_file():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class Decision:
    day: str
    action: str          # "none" | "propose_retraining"
    reason: str
    consecutive_days: int
    cooldown_until: str | None
    observation: dict


def _cooldown_until(last_action: str | None) -> str | None:
    if not last_action:
        return None
    return (date.fromisoformat(last_action) + timedelta(days=COOLDOWN_DAYS)).isoformat()


def decide(state: TriggerState, obs: Observation) -> tuple[TriggerState, Decision]:
    """Fold one day's observation into the state and say what to do about it.

    Pure: takes a state, returns a new one. That is what makes the whole policy
    testable by replaying a sequence of days, which is how the thresholds above
    were checked rather than assumed.
    """
    state.observations.append(asdict(obs))
    state.observations = state.observations[-90:]  # a quarter of history is plenty

    def finish(action: str, reason: str) -> tuple[TriggerState, Decision]:
        return state, Decision(
            day=obs.day, action=action, reason=reason,
            consecutive_days=state.consecutive_days,
            cooldown_until=_cooldown_until(state.last_action),
            observation=asdict(obs),
        )

    # --- the window has to be big enough to mean anything -------------------
    if obs.n_samples < MIN_WINDOW_SAMPLES:
        state.consecutive_days = 0
        state.run_started = None
        return finish("none", (
            f"window of {obs.n_samples} samples is below the {MIN_WINDOW_SAMPLES} "
            f"needed for a reliable reading; the noise floor climbs steeply below that"
        ))

    # --- type gate: only data drift is a retraining problem ------------------
    if not obs.retrain_eligible:
        state.consecutive_days = 0
        state.run_started = None
        return finish("none", f"verdict '{obs.verdict}' is not fixed by retraining")

    # --- level, with hysteresis ---------------------------------------------
    if obs.qualifies:
        if state.consecutive_days == 0:
            state.run_started = obs.day
        state.consecutive_days += 1
    elif obs.sustains and state.consecutive_days > 0:
        # Between the two thresholds: hold the run rather than restart the count.
        # Without this a marginal day resets the clock and a slow, real drift
        # never accumulates enough consecutive days to act on.
        state.consecutive_days += 1
    else:
        broken = state.consecutive_days
        state.consecutive_days = 0
        state.run_started = None
        return finish("none", (
            f"output KS {obs.output_ks:.3f} fell below the {LEVEL_OFF} hold threshold"
            + (f"; a run of {broken} day(s) ended -- transient, not a trend" if broken else "")
        ))

    # --- persistence ---------------------------------------------------------
    if state.consecutive_days < PERSISTENCE_DAYS:
        return finish("none", (
            f"day {state.consecutive_days} of {PERSISTENCE_DAYS} required; "
            f"one window is not a trend"
        ))

    # --- cooldown ------------------------------------------------------------
    until = _cooldown_until(state.last_action)
    if until and obs.day < until:
        return finish("none", (
            f"persistent drift, but a retraining cycle is already in flight "
            f"(acted {state.last_action}, locked out until {until})"
        ))

    state.last_action = obs.day
    state.consecutive_days = 0
    started = state.run_started
    state.run_started = None
    return finish("propose_retraining", (
        f"data drift sustained for {PERSISTENCE_DAYS}+ days since {started}, "
        f"output KS {obs.output_ks:.3f}"
    ))
