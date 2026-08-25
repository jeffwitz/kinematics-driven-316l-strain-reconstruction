#!/usr/bin/env python3
"""Take one constrained GN step using the qualified F centered-FD oracle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from fem_inhouse.core.srix_parameters import DEFAULT_PARAMETER_SET, get_parameter_set
from fem_inhouse.identification.srix_parameter_coordinates import SrixTheta9
from scripts.qualify_srix_p0043_synthetic_smoke import CROP, _forward, _load_inputs, _make_path, _vector

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/reference_data/p0043_f_mapping_reidentification_v1"


def main() -> int:
    measured, angles, _ = _load_inputs(CROP)
    path = _make_path(measured, 4); scored = tuple(4 * i for i in range(1, 9))
    target = [np.asarray(s.boundary, float).copy() for s in path]
    lib = os.environ.get("MFRONT_BEHAVIOUR_LIBRARY", str(ROOT / "build/mfront/src/libBehaviour.so"))
    eta_ref = SrixTheta9.from_parameter_set(get_parameter_set(DEFAULT_PARAMETER_SET)).log_coordinates()
    low, high = eta_ref.copy(), eta_ref.copy(); low[:3] += np.log(.85); high[:3] += np.log(1.15)
    phys = np.asarray([[25,60],[8,35],[4,20],[1,6],[10000,80000],[500,3000]], float)
    low[3:], high[3:] = np.log(phys[:,0]), np.log(phys[:,1])
    j = np.load(OUT / "full_jacobian_f.npy")
    _, s, vh = np.linalg.svd(j, full_matrices=False); basis = vh.T[:, :7]
    fields, timing = _forward(SrixTheta9.from_log_coordinates(eta_ref), path, angles, lib, 4, "F")
    residual = _vector(fields, scored, target); rms0 = float(np.sqrt(np.mean(residual**2)))
    weak = vh.T[:, 7:]; z0 = np.zeros(7)
    constraint = LinearConstraint(basis, low - eta_ref, high - eta_ref)
    fun = lambda d: 0.5 * float(np.sum((residual + j @ (basis @ d))**2))
    jac = lambda d: basis.T @ j.T @ (residual + j @ (basis @ d))
    fit = minimize(fun, z0, jac=jac, method="SLSQP", constraints=(constraint,), options={"ftol":1e-18,"maxiter":500})
    accepted = None
    for alpha in (1., .5, .25, .125, .0625, .03125):
        eta = eta_ref + basis @ (alpha * fit.x)
        if np.any(eta < low - 1e-10) or np.any(eta > high + 1e-10): continue
        f, t = _forward(SrixTheta9.from_log_coordinates(eta), path, angles, lib, 4, "F")
        rr = _vector(f, scored, target); rms = float(np.sqrt(np.mean(rr**2)))
        if t["verification_residual"] <= 1e-8 and rms < rms0:
            accepted = {"alpha": alpha, "rms_mm": rms, "verification_residual": t["verification_residual"],
                        "eta": eta.tolist(), "parameters": SrixTheta9.from_log_coordinates(eta).as_runtime_overrides()}
            break
    report = {"method":"one constrained GN step with F centered-FD Jacobian", "prior_rms_mm":rms0,
              "linear_solver_success":bool(fit.success), "linear_solver_message":str(fit.message),
              "predicted_rms_mm":float(np.sqrt(2*fun(fit.x)/residual.size)), "accepted_trial":accepted,
              "singular_values":s.tolist(), "basis":basis.tolist(), "verification_prior":timing["verification_residual"]}
    (OUT / "fd_gn_one_step.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"prior_rms_mm":rms0,"predicted_rms_mm":report["predicted_rms_mm"],"accepted_trial":accepted},sort_keys=True),flush=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
