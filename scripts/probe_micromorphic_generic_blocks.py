"""Check that MFront itself can produce the four coupled tangent blocks.

The coupled micromorphic Newton needs ``dsig/deps``, ``dsig/dchi``,
``dp/deps`` and ``dp/dchi``. The host currently obtains the last three by
finite differences around the converged point, at eight extra integrations per
call.

``validation/mfront/MicromorphicJ2GenericBlocksProbe.mfront`` declares ``chi``
as a second gradient and the equivalent plastic strain as its conjugate force,
and asks MFront for the four blocks through ``@TangentOperatorBlocks``. TFEL
builds them from the converged implicit Jacobian via
``getIntegrationVariablesDerivatives_*``, so nothing is refactorised and no
residual crosses the interface.

This script integrates a load path and compares every block against central
finite differences of the same behaviour. It is a feasibility probe, not a
production path: the probe is tridimensional, and the plane-stress closure is
qualified separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MATERIAL = {
    "YoungModulus": 205000.0,
    "PoissonRatio": 0.3,
    "InitialYieldStress": 250.0,
    "HardeningCoefficient": 500.0,
    "HardeningExponent": 0.245,
    "MicromorphicCouplingModulus": 3000.0,
}


def _make(library: str):
    import mgis.behaviour as mb

    behaviour = mb.load(
        library, "MicromorphicJ2GenericBlocksProbe", mb.Hypothesis.Tridimensional
    )
    data = mb.MaterialDataManager(behaviour, 1)
    for state in (data.s0, data.s1):
        for name, value in MATERIAL.items():
            mb.setMaterialProperty(state, name, value)
        mb.setExternalStateVariable(state, "Temperature", 293.15)
    return mb, behaviour, data


def _integrate(
    mb, data, strain: np.ndarray, chi: float, dt: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Integrate one step from the current s0 and return forces and tangent."""

    data.s1.gradients[0, :6] = strain
    data.s1.gradients[0, 6] = chi
    status = mb.integrate(
        data, mb.IntegrationType.INTEGRATION_CONSISTENT_TANGENT_OPERATOR, dt, 0, 1
    )
    if status != 1:
        raise RuntimeError(f"probe integration failed with status {status}")
    return (
        np.array(data.s1.thermodynamic_forces[0, :6], dtype=float),
        float(data.s1.thermodynamic_forces[0, 6]),
        np.array(data.K[0], dtype=float).ravel(),
    )


def _snapshot(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(data.s0.gradients, dtype=float),
        np.array(data.s0.thermodynamic_forces, dtype=float),
        np.array(data.s0.internal_state_variables, dtype=float),
    )


def _restore(data, snapshot) -> None:
    data.s0.gradients[:, :] = snapshot[0]
    data.s0.thermodynamic_forces[:, :] = snapshot[1]
    data.s0.internal_state_variables[:, :] = snapshot[2]


def probe(library: str, steps: int, amplitude: float, chi_scale: float, fd_step: float) -> dict:
    """Walk a load path, comparing every returned block against central FD."""

    mb_module, _behaviour, data = _make(library)
    direction = np.array([1.0, -0.4, -0.4, 0.2, 0.0, 0.0])
    records = []

    for step in range(1, steps + 1):
        strain = (step / steps) * amplitude * direction
        chi = (step / steps) * chi_scale
        base = _snapshot(data)

        stress, pobs, tangent = _integrate(mb_module, data, strain, chi, 1.0 / steps)
        # Declaration order: dsig_ddeto (36), dsig_ddchi (6), dp_ddeto (6), dp_ddchi (1).
        if tangent.size != 49:
            raise RuntimeError(f"expected 49 tangent entries, got {tangent.size}")
        dsig_deto = tangent[:36].reshape(6, 6)
        dsig_dchi = tangent[36:42]
        dp_deto = tangent[42:48]
        dp_dchi = tangent[48]

        # Central finite differences of the SAME behaviour, each restarted from
        # the identical committed state so the comparison is same-state.
        fd_dsig_deto = np.zeros((6, 6))
        fd_dp_deto = np.zeros(6)
        for column in range(6):
            plus, minus = strain.copy(), strain.copy()
            plus[column] += fd_step
            minus[column] -= fd_step
            _restore(data, base)
            sp, pp, _ = _integrate(mb_module, data, plus, chi, 1.0 / steps)
            _restore(data, base)
            sm, pm, _ = _integrate(mb_module, data, minus, chi, 1.0 / steps)
            fd_dsig_deto[:, column] = (sp - sm) / (2 * fd_step)
            fd_dp_deto[column] = (pp - pm) / (2 * fd_step)

        chi_step = fd_step
        _restore(data, base)
        sp, pp, _ = _integrate(mb_module, data, strain, chi + chi_step, 1.0 / steps)
        _restore(data, base)
        sm, pm, _ = _integrate(mb_module, data, strain, chi - chi_step, 1.0 / steps)
        fd_dsig_dchi = (sp - sm) / (2 * chi_step)
        fd_dp_dchi = (pp - pm) / (2 * chi_step)

        def relative(candidate, reference) -> float:
            candidate = np.atleast_1d(np.asarray(candidate, dtype=float))
            reference = np.atleast_1d(np.asarray(reference, dtype=float))
            scale = float(np.linalg.norm(reference))
            error = float(np.linalg.norm(candidate - reference))
            return error / scale if scale > 0.0 else error

        # Commit the unperturbed step and move on.
        _restore(data, base)
        _integrate(mb_module, data, strain, chi, 1.0 / steps)
        mb_module.update(data)

        records.append(
            {
                "step": step,
                "equivalent_plastic_strain": pobs,
                "von_mises_stress": float(
                    np.sqrt(
                        0.5
                        * (
                            (stress[0] - stress[1]) ** 2
                            + (stress[1] - stress[2]) ** 2
                            + (stress[2] - stress[0]) ** 2
                        )
                        + 3.0 * (stress[3] ** 2 + stress[4] ** 2 + stress[5] ** 2)
                    )
                ),
                "relative_error": {
                    "dsig_deto": relative(dsig_deto, fd_dsig_deto),
                    "dsig_dchi": relative(dsig_dchi, fd_dsig_dchi),
                    "dp_deto": relative(dp_deto, fd_dp_deto),
                    "dp_dchi": relative(dp_dchi, fd_dp_dchi),
                },
                "dp_dchi": {"returned": float(dp_dchi), "finite_difference": float(fd_dp_dchi)},
            }
        )

    plastic = [r for r in records if r["equivalent_plastic_strain"] > 0.0]
    worst = {
        key: max(r["relative_error"][key] for r in plastic)
        for key in ("dsig_deto", "dsig_dchi", "dp_deto", "dp_dchi")
    } if plastic else {}
    return {
        "finite_difference_step": fd_step,
        "steps": steps,
        "amplitude": amplitude,
        "chi_scale": chi_scale,
        "plastic_steps": len(plastic),
        "worst_relative_error_on_plastic_steps": worst,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--amplitude", type=float, default=0.02)
    parser.add_argument("--chi-scale", type=float, default=0.01)
    parser.add_argument("--fd-step", type=float, action="append")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    steps = arguments.fd_step or [1.0e-7]
    report = {
        "behaviour": "MicromorphicJ2GenericBlocksProbe",
        "material": MATERIAL,
        "sweeps": [
            probe(
                arguments.library,
                arguments.steps,
                arguments.amplitude,
                arguments.chi_scale,
                h,
            )
            for h in steps
        ],
    }
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for sweep in report["sweeps"]:
        print(
            f"  h={sweep['finite_difference_step']:.0e}  "
            f"plastic steps={sweep['plastic_steps']:2d}  "
            + "  ".join(
                f"{k}={v:.2e}" for k, v in sweep["worst_relative_error_on_plastic_steps"].items()
            )
        )


if __name__ == "__main__":
    main()
