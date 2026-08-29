#!/usr/bin/env python3
"""Aggregate registered inverse/observability results without rerunning them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tensor = _load("validation/_generated/tensor_local_inverse/observability_truncated_svd.json")[
        "observability"
    ]
    q1_spectrum = _load("validation/_generated/local_coefficient_inverse/spectrum_q1.json")[
        "identifiability_spectrum"
    ]["q1"]
    q4_spectrum = _load("validation/_generated/local_coefficient_inverse/spectrum_q4.json")[
        "identifiability_spectrum"
    ]["q4"]
    q1_twin = _load("validation/_generated/local_coefficient_inverse/gate3_twin_q1.json")[
        "gate_3_twin"
    ]["q1"]
    exact_regm = _load("validation/reference_data/srix_regm_femu_ranking_v1/report.json")[
        "statistics"
    ]
    observed_regm = _load(
        "validation/reference_data/srix_regm_femu_observed_ranking_v1/report.json"
    )["statistics"]
    slip = _load("validation/_generated/shared_tensor_generator/slip_law_ladder.json")
    fcc = _load("validation/_generated/shared_tensor_generator/fcc_slip_decomposition.json")
    femu = _load("validation/reference_data/p0043_spatial_noise_stationarity_v1/report.json")

    entries = [
        {
            "problem": "free tensor inverse",
            "observable_fit": {
                "metric": "least-squares objective",
                "value": 2.43e-15,
                "source": "validation/tensor_local_inverse_results.md",
            },
            "latent_recovery": {
                "registered_gauge_relative_error": tensor["truths"]["registered"][
                    "best_gauge_relative_error"
                ],
                "zero_mean_gauge_relative_error": tensor["truths"]["zero_mean"][
                    "best_gauge_relative_error"
                ],
            },
            "effective_rank": 173,
            "parameter_count": tensor["coefficients"],
            "nullity": tensor["coefficients"] - 173,
            "conclusion": "excellent kinematic fit does not determine the free latent tensor field",
        },
        {
            "problem": "compact local inverse q=1",
            "observable_fit": {"metric": "twin objective", "value": q1_twin["final_objective"]},
            "latent_recovery": {
                "field_relative_error": q1_twin["field_relative_error"],
                "coefficient_relative_error": q1_twin["coefficient_relative_error"],
            },
            "effective_rank": q1_spectrum["effective_rank"],
            "condition_number": q1_spectrum["condition_number"],
            "nullity": 0,
            "conclusion": (
                "a sufficiently constrained representation can recover an exact synthetic twin"
            ),
        },
        {
            "problem": "enriched local inverse q=4",
            "observable_fit": {
                "metric": "twin recovery",
                "status": "not run; basis is rank deficient",
            },
            "latent_recovery": {"status": "not interpreted"},
            "effective_rank": q4_spectrum["effective_rank"],
            "condition_number": q4_spectrum["condition_number"],
            "parameter_count": q4_spectrum["coefficients"],
            "nullity": q4_spectrum["coefficients"] - q4_spectrum["effective_rank"]["above_1e-6"],
            "conclusion": "basis enrichment introduces algebraic null directions",
        },
        {
            "problem": "REGM exact twin ranking",
            "observable_fit": {
                "metric": "Spearman",
                "value": exact_regm["spearman"],
                "log_pearson": exact_regm["log_pearson"],
            },
            "latent_recovery": {"status": "ranking surrogate only"},
            "conclusion": "exact-space ranking is useful before measurement transfer",
        },
        {
            "problem": "REGM observed-DIC transfer",
            "observable_fit": {
                "metric": "Spearman T1 transfer",
                "value": observed_regm["T1_transfer"]["spearman"],
                "log_pearson": observed_regm["T1_transfer"]["log_pearson"],
            },
            "latent_recovery": {"status": "not authorized"},
            "conclusion": (
                "the surrogate ranking does not survive the registered observed-space formulation"
            ),
        },
        {
            "problem": "shared FCC slip generator ladder",
            "observable_fit": {
                "metric": "unconstrained tensor e_FCC median",
                "value": fcc["unconstrained"]["e_fcc_median"],
            },
            "latent_recovery": {
                "metric": "weighted system R2",
                "value": fcc["weighted_system_r2"],
                "status": "slip-space gate false",
                "slip_space_gate": slip["bars"]["slip_space"],
            },
            "conclusion": (
                "tensor/coarse observables do not establish individual slip-system recovery"
            ),
        },
        {
            "problem": "SRIX parametric FEMU SVD",
            "observable_fit": {
                "metric": "registered geometric sensitivity",
                "status": "rank-3 geometric subspace",
            },
            "latent_recovery": {"status": "absolute experimental detectability unqualified"},
            "effective_rank": 3,
            "singular_values": femu["corner_sensitivity_svd"]["singular_values"],
            "weak_mode": "Q-b",
            "conclusion": (
                "parameter combinations, not individual parameters, are the supported object"
            ),
        },
    ]
    report = {
        "schema_version": 1,
        "method": "offline aggregation of registered inverse and observability artifacts",
        "no_forward_or_finite_difference": True,
        "central_claim": (
            "excellent agreement in the measured observable does not by itself "
            "identify a unique latent constitutive state"
        ),
        "entries": entries,
        "sources": [
            "validation/_generated/tensor_local_inverse/observability_truncated_svd.json",
            "validation/_generated/local_coefficient_inverse/spectrum_q1.json",
            "validation/_generated/local_coefficient_inverse/spectrum_q4.json",
            "validation/_generated/local_coefficient_inverse/gate3_twin_q1.json",
            "validation/reference_data/srix_regm_femu_ranking_v1/report.json",
            "validation/reference_data/srix_regm_femu_observed_ranking_v1/report.json",
            "validation/_generated/shared_tensor_generator/slip_law_ladder.json",
            "validation/_generated/shared_tensor_generator/fcc_slip_decomposition.json",
            "validation/reference_data/p0043_spatial_noise_stationarity_v1/report.json",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "entries": len(entries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
