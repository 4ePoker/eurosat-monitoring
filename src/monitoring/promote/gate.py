"""Champion versus challenger: can this candidate replace production?

Four checks, in order:

    1. no regression on the batches that were already working
    2. no regression on any other batch
    3. no class recall collapse (an average can be flat while classes move underneath)
    4. everything it can't judge gets reported, not decided

Comparisons carry a noise band from the batch size. A batch of 500 at 27% accuracy has
a standard error of 2 points, so a 1.8-point move is a coin flip, not a regression.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVALUATIONS = ROOT / "artifacts" / "evaluations"

#: Batches that represent "things that were working". A regression here is the
#: one a user notices, so it is judged strictly.
BASELINE_BATCHES = ("none_s0.50_test_n500", "reference_train")

#: How much a baseline batch may drop, beyond statistical noise, and still pass.
MAX_BASELINE_REGRESSION = 0.005

#: How much any single class recall may drop, beyond noise, and still pass.
MAX_CLASS_REGRESSION = 0.05

#: Minimum class support for its recall to be judged at all. Below this the
#: estimate is too noisy to block a promotion on, and saying so is better than
#: pretending otherwise -- see the label-shift batches, where Industrial has 2.
MIN_CLASS_SUPPORT = 30


def stderr(p: float, n: int) -> float:
    """Standard error of an accuracy estimate. The width of "could be nothing"."""
    return float(np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))


@dataclass(frozen=True)
class GateResult:
    promote: bool
    checks: list[dict]
    comparisons: list[dict]
    tradeoffs: list[str]

    def summary(self) -> str:
        return "PROMOTE" if self.promote else "BLOCK"


def compare(champion: dict, challenger: dict) -> GateResult:
    champ_batches = {b["batch"]: b for b in champion["batches"]}
    comparisons, checks, blocking = [], [], []

    for cand in challenger["batches"]:
        base = champ_batches.get(cand["batch"])
        if base is None:
            continue
        delta = cand["accuracy"] - base["accuracy"]
        # Two standard errors, pooled across the two estimates: the band inside
        # which a difference is indistinguishable from sampling noise.
        band = 2 * np.hypot(stderr(base["accuracy"], base["n"]),
                            stderr(cand["accuracy"], cand["n"]))
        comparisons.append({
            "batch": cand["batch"],
            "champion": base["accuracy"],
            "challenger": cand["accuracy"],
            "delta": round(delta, 4),
            "noise_band": round(float(band), 4),
            "significant": bool(abs(delta) > band),
            "verdict": ("better" if delta > band else "worse" if delta < -band else "tie"),
        })

    # --- check 1: no significant regression where things worked ------------
    baseline_fails = [
        c for c in comparisons
        if c["batch"] in BASELINE_BATCHES
        and c["delta"] < -MAX_BASELINE_REGRESSION and c["significant"]
    ]
    checks.append({
        "check": "no regression on the working baseline",
        "passed": not baseline_fails,
        "detail": (f"regressed on {[c['batch'] for c in baseline_fails]}"
                   if baseline_fails else
                   f"baselines held: " + ", ".join(
                       f"{c['batch'].split('_')[0]} {c['delta']:+.3f}"
                       for c in comparisons if c["batch"] in BASELINE_BATCHES)),
    })
    blocking += baseline_fails

    # --- check 2: no significant regression on any other batch --------------
    other_fails = [
        c for c in comparisons
        if c["batch"] not in BASELINE_BATCHES and c["delta"] < 0 and c["significant"]
    ]
    checks.append({
        "check": "no significant regression on any batch",
        "passed": not other_fails,
        "detail": (f"{[c['batch'] for c in other_fails]}" if other_fails
                   else "all drops sit inside their noise bands"),
    })
    blocking += other_fails

    # --- check 3: no class collapse ----------------------------------------
    class_fails = []
    for cand in challenger["batches"]:
        base = champ_batches.get(cand["batch"])
        if base is None:
            continue
        for name, stats in cand["per_class"].items():
            base_stats = base["per_class"].get(name, {})
            if (stats["recall"] is None or base_stats.get("recall") is None
                    or stats["n"] < MIN_CLASS_SUPPORT):
                continue
            drop = base_stats["recall"] - stats["recall"]
            band = 2 * np.hypot(stderr(base_stats["recall"], base_stats["n"]),
                                stderr(stats["recall"], stats["n"]))
            if drop > MAX_CLASS_REGRESSION and drop > band:
                class_fails.append({
                    "batch": cand["batch"], "class": name,
                    "champion": base_stats["recall"], "challenger": stats["recall"],
                    "drop": round(drop, 4),
                })
    checks.append({
        "check": "no class recall collapse",
        "passed": not class_fails,
        "detail": (f"{len(class_fails)} class/batch regressions: "
                   + ", ".join(f"{c['class']}@{c['batch'].split('_')[0]} -{c['drop']:.2f}"
                               for c in class_fails[:4])
                   if class_fails else
                   f"every class with >={MIN_CLASS_SUPPORT} samples held"),
    })
    blocking += class_fails

    # --- what the gate refuses to decide -----------------------------------
    tradeoffs = [
        f"size: {champion['size_mb']} MB -> {challenger['size_mb']} MB "
        f"({challenger['size_mb'] / max(champion['size_mb'], 1e-9):.1f}x). "
        f"Container size and memory footprint are a deployment cost this gate "
        f"cannot weigh against accuracy on the operator's behalf.",
        "latency: not measured here. Run Project 5's scripts/benchmark.py against "
        "both models on the target CPU before deciding -- quantisation gains are "
        "hardware-dependent and cannot be inferred.",
    ]

    return GateResult(promote=not blocking, checks=checks,
                      comparisons=comparisons, tradeoffs=tradeoffs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion", required=True, help="evaluation name under artifacts/evaluations")
    parser.add_argument("--challenger", required=True)
    args = parser.parse_args()

    champion = json.loads((EVALUATIONS / f"{args.champion}.json").read_text())
    challenger = json.loads((EVALUATIONS / f"{args.challenger}.json").read_text())
    result = compare(champion, challenger)

    print(f"champion  : {champion['model_name']}  mean acc {champion['mean_accuracy']}")
    print(f"challenger: {challenger['model_name']}  mean acc {challenger['mean_accuracy']}\n")

    print(f"{'batch':36s} {'champ':>7s} {'chall':>7s} {'delta':>7s} {'+-noise':>8s}  verdict")
    print("-" * 78)
    for c in sorted(result.comparisons, key=lambda c: c["delta"]):
        print(f"{c['batch']:36s} {c['champion']:7.3f} {c['challenger']:7.3f} "
              f"{c['delta']:+7.3f} {c['noise_band']:8.3f}  {c['verdict']}")

    print("\nchecks:")
    for c in result.checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}\n         {c['detail']}")

    print("\nnot decided by this gate:")
    for t in result.tradeoffs:
        print(f"  - {t}")

    print(f"\n=> {result.summary()}")
    (EVALUATIONS / "gate_result.json").write_text(json.dumps({
        "champion": champion["model_name"], "challenger": challenger["model_name"],
        "promote": result.promote, "checks": result.checks,
        "comparisons": result.comparisons, "tradeoffs": result.tradeoffs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
