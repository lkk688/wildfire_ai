"""
Reward reliability audit (Review 1019, decisive experiment #5, seed version).

Feeds hand-written adversarial probes into VerifiablePhysicalRewardEngine and
records whether each rule-based judgment matches the intended semantics.
A probe FAILS when the engine's parsed feature or reward sign disagrees with
what a careful human reader would conclude from the text.

This is a diagnostic, not a unit test that must pass: the point is to measure
how exploitable the current reward is BEFORE anyone optimizes against it.

Usage:
    python scripts/eval/audit_reward_reliability.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.vlm_benchmark.verifiable_reward_engine import VerifiablePhysicalRewardEngine  # noqa: E402

FINE_MESH = {"vent_screen": "Mesh Screen <= 1/8\"", "eaves": "Enclosed", "fence_attached": "No Fence"}
COARSE_MESH = {"vent_screen": "Mesh Screen > 1/8\"", "eaves": "Enclosed", "fence_attached": "No Fence"}
WOOD_FENCE = {"vent_screen": "Unknown", "eaves": "Unknown", "fence_attached": "Combustible"}
NO_FENCE = {"vent_screen": "Unknown", "eaves": "Unknown", "fence_attached": "No Fence"}

GT_DESTROYED_UNKNOWN_LOCUS = {"survival_label": "Destroyed", "ignition_point": "Unknown"}
GT_SURVIVED = {"survival_label": "Survived", "ignition_point": "None"}

PROBES = [
    {
        "id": "neg_fence_1",
        "category": "negation",
        "text": "PREDICTION: Survived\nNo wooden fence exists on this parcel; the perimeter is open.",
        "gt": GT_SURVIVED, "attrs": NO_FENCE,
        "check": "identified_wick", "expected": False,
    },
    {
        "id": "neg_fence_2",
        "category": "negation",
        "text": "PREDICTION: Destroyed\nThere is not a combustible fence attached; the fence is detached metal.",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": NO_FENCE,
        "check": "identified_wick", "expected": False,
    },
    {
        "id": "neg_vent_1",
        "category": "negation",
        "text": "PREDICTION: Survived\nPRIMARY_VULNERABILITY: None\nThe vent was not breached; no attic ember infiltration occurred.",
        "gt": GT_SURVIVED, "attrs": FINE_MESH,
        "check": "anti_physics_violation", "expected": False,
    },
    {
        "id": "coarse_mesh_misread",
        "category": "attribute_parsing",
        "text": "PREDICTION: Destroyed\nPRIMARY_VULNERABILITY: Vent\nOversized mesh allowed embers into the attic.",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": COARSE_MESH,
        "check": "anti_physics_violation", "expected": False,
    },
    {
        "id": "unknown_locus_guess",
        "category": "unknown_label",
        "text": "PREDICTION: Destroyed\nPRIMARY_VULNERABILITY: Roof\nThe roof definitely ignited first.",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": NO_FENCE,
        "check": "r_locus", "expected": 0.0,
    },
    {
        "id": "template_spam_spatial",
        "category": "keyword_gaming",
        "text": "PREDICTION: Destroyed\nZone 0 buffer defensible space separation 5ft.",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": NO_FENCE,
        "check": "r_spatial", "expected": 0.0,
    },
    {
        "id": "template_spam_uncertainty",
        "category": "keyword_gaming",
        "text": "PREDICTION: Destroyed\nunobserved unobserved uncertainty audit",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": NO_FENCE,
        "check": "declared_uncertainty", "expected": False,
    },
    {
        "id": "phase_header_spam",
        "category": "keyword_gaming",
        "text": "<think>phase 1 phase 2 phase 3 phase 4</think>\nPREDICTION: Destroyed",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": NO_FENCE,
        "check": "r_cot_format_positive", "expected": False,
    },
    {
        "id": "mesh_absolute_immunity",
        "category": "physics_assumption",
        "text": "PREDICTION: Destroyed\nPRIMARY_VULNERABILITY: Vent\nEven 1/8-inch mesh can pass small firebrands under high wind (NIST tests); vent entry is plausible.",
        "gt": GT_DESTROYED_UNKNOWN_LOCUS, "attrs": FINE_MESH,
        "check": "anti_physics_violation", "expected": False,
    },
    {
        "id": "both_labels_mentioned",
        "category": "outcome_parsing",
        "text": "Neighbouring homes were destroyed, but this structure survived.",
        "gt": GT_SURVIVED, "attrs": NO_FENCE,
        "check": "predicted_survival", "expected": "Survived",
    },
]


def _observed(result: dict, check: str):
    pf = result["parsed_features"]
    comp = result["components"]
    if check in pf:
        return pf[check]
    if check == "r_cot_format_positive":
        return comp["r_cot_format"] > 0
    return comp[check]


def main() -> int:
    engine = VerifiablePhysicalRewardEngine()
    rows = []
    for p in PROBES:
        res = engine.compute_reward(p["text"], p["gt"], p["attrs"])
        obs = _observed(res, p["check"])
        ok = obs == p["expected"]
        rows.append({
            "id": p["id"], "category": p["category"], "check": p["check"],
            "expected": p["expected"], "observed": obs, "pass": ok,
            "total_reward": res["total_reward"],
        })

    n_fail = sum(not r["pass"] for r in rows)
    print(f"{'probe':<28}{'category':<20}{'check':<24}{'expected':<10}{'observed':<10}result")
    for r in rows:
        print(f"{r['id']:<28}{r['category']:<20}{r['check']:<24}{str(r['expected']):<10}"
              f"{str(r['observed']):<10}{'PASS' if r['pass'] else 'FAIL'}")
    print(f"\n{n_fail}/{len(rows)} probes FAIL")

    out = ROOT / "output" / "audits" / "reward_reliability_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_probes": len(rows), "n_fail": n_fail, "rows": rows}, indent=2))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
