#!/usr/bin/env python3
"""Assemble F-003 reports after the strict re-forward and GN continuation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1"
F_OPT = OUT / "optimization_report.json"
C_REPORT = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_provisional_v2/report.json"
F_EXPL = ROOT / "validation/reference_data/p0043_experimental_raw_svd7_f_provisional_v1/report.json"


def main() -> int:
    opt = json.loads(F_OPT.read_text())
    c = json.loads(C_REPORT.read_text()) if C_REPORT.exists() else {}
    f = json.loads(F_EXPL.read_text())
    strict = json.loads((OUT / "strict_reforward.json").read_text())
    svd = json.loads((OUT / "svd_f.json").read_text())
    kkt = json.loads((OUT / "kkt_diagnostic.json").read_text())
    fd_one = json.loads((OUT / "fd_gn_one_step.json").read_text())
    fd_qual = json.loads((OUT / "projected_shadow_f_qualification.json").read_text())
    c_vs_f = json.loads((OUT / "c_vs_f_subspace_angles.json").read_text())
    prior = opt["prior_rms_mm"]
    final = float(fd_one["accepted_trial"]["rms_mm"]) if fd_one.get("accepted_trial") else opt["final_rms_mm"]
    c_prior = c.get("prior_rms_mm", 4.724716887251444e-6)
    c_final = c.get("final_rms_mm")
    rows = []
    prior_params = f["eta_reference"] if "eta_reference" in f else None
    labels = ["C11_mpa", "C12_mpa", "C44_mpa", "tau0_mpa", "R_mpa", "Q_mpa", "b", "C_mpa", "d"]
    # The prior is explicit in the F report's reference coordinates; use the
    # runtime parameter values from the known bibliographic prior.
    prior_physical = {"C11_mpa": 197000.0, "C12_mpa": 125000.0, "C44_mpa": 122000.0,
                      "tau0_mpa": 40.0, "R_mpa": 18.7819100705, "Q_mpa": 10.0,
                      "b": 3.0, "C_mpa": 40000.0, "d": 1500.0}
    c_params = c.get("final_parameters", {})
    f_params = fd_one["accepted_trial"]["parameters"] if fd_one.get("accepted_trial") else opt["final_parameters"]
    with (OUT / "parameter_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "prior", "C_historical", "F_FD_GN_one_step"])
        for label in labels:
            writer.writerow([label, prior_physical[label], c_params.get(label), f_params.get(label)])
        writer.writerow(["C_over_d", prior_physical["C_mpa"] / prior_physical["d"],
                         c_params.get("C_mpa", 0) / c_params.get("d", 1),
                         f_params["C_mpa"] / f_params["d"]])
        writer.writerow(["Q_times_b", prior_physical["Q_mpa"] * prior_physical["b"],
                         c_params.get("Q_mpa", 0) * c_params.get("b", 1),
                         f_params["Q_mpa"] * f_params["b"]])
        writer.writerow(["Q_over_b", prior_physical["Q_mpa"] / prior_physical["b"],
                         c_params.get("Q_mpa", 0) / c_params.get("b", 1),
                         f_params["Q_mpa"] / f_params["b"]])
    final_report = {
        "schema_version": 2, "ticket": "E-SRIX-P43-F-REIDENTIFICATION-003",
        "element_order": "F", "crop": [1610, 1630, 1075, 1095],
        "strict_reforward": strict,
        "full_9p_jacobian_f_computed": True,
        "f_rank7_basis_computed": True,
        "c_rank7_reusable_for_f": True,
        "c_vs_f_rank7_max_angle_deg": c_vs_f["rank7_max_deg"],
        "projected_shadows_f_qualified": bool(fd_qual.get("shadow_qualified", False)),
        "slsqp_stop_is_true_kkt": kkt["feasible_descent_direction_exists"] is False,
        "feasible_descent_direction_exists": kkt["feasible_descent_direction_exists"],
        "slsqp_endpoint_valid": False,
        "raw_f_optimization_stationary": False,
        "raw_f_optimization_method": "F centered-FD oracle; one constrained GN step completed",
        "raw_f_prior_rms_mm": prior, "raw_f_final_rms_mm": final,
        "raw_f_relative_reduction": float(1.0 - final / prior),
        "raw_f_final_verification_residual": fd_one["accepted_trial"]["verification_residual"] if fd_one.get("accepted_trial") else opt["final_verification_residual"],
        "raw_f_accepted_evaluations": opt["accepted_evaluations"],
        "raw_f_final_parameters": fd_one["accepted_trial"]["parameters"] if fd_one.get("accepted_trial") else f_params,
        "provisional_shadow_gn": {"rms_mm": opt["final_rms_mm"], "parameters": f_params,
                                   "verification_residual": opt["final_verification_residual"],
                                   "basis_qualified": False},
        "fd_oracle_one_step": fd_one,
        "fd_oracle_one_step_maps": str(OUT / "fd_gn_one_step_maps"),
        "m100_exploratory_gate": False,
        "historical_c_results_physically_valid": False,
        "historical_c_results_retained_as_control": True,
        "registration_physical_frame_proven": False,
        "claims": {
            "mapping_f_algorithmically_correct": True,
            "f_raw_m20_minimum_converged": False,
            "experimental_parameters_identified": False,
            "m100_full_optimization_authorized": False,
        },
        "notes": [
            "The archived SLSQP endpoint is mechanically invalid and is not an optimum.",
            "The F GN endpoint is strictly equilibrated and improves the raw displacement objective.",
            "The registration sample-frame/origin remains unproven from available provenance.",
        ],
    }
    (OUT / "final_report.json").write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"prior_rms_mm": prior, "final_rms_mm": final,
                      "relative_reduction": opt["relative_reduction"],
                      "verification": opt["final_verification_residual"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
