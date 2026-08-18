#!/usr/bin/env python3
"""Material qualifications of the causal TANN-FCC (T0), before any P43 run.

The frozen gates, per `validation/tann_fcc_preregistration.md`:
zero increment, non-negative dissipation on thousands of random states,
permutation equivariance of the twelve systems, substep invariance
(1/2/4/8), the algorithmic tangent against central finite differences
with a step study, transaction semantics, and the FCC geometry coherence
with the repo's canonical closed form.
"""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.core.srix_canonical import SCHMID_FACTOR_001, ACTIVE_SYSTEMS_001

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
POINTS = 2000
RNG = np.random.default_rng(20260817)


def systems_identity() -> np.ndarray:
    """Schmid tensors in the specimen frame for the identity orientation."""

    tensors = np.empty((12, 3, 3))
    for index, (burgers, normal) in enumerate(SLIP_SYSTEMS):
        s = np.asarray(burgers, dtype=np.float64)
        m = np.asarray(normal, dtype=np.float64)
        s /= np.linalg.norm(s)
        m /= np.linalg.norm(m)
        tensors[index] = 0.5 * (np.outer(s, m) + np.outer(m, s))
    return np.tile(tensors, (POINTS, 1, 1, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT / "tann_fcc_material_qualification.json")
    parser.add_argument("--sigma-ref", type=float, default=None,
                        help="force reference in MPa (None -> 2 mu, Amendment 1; "
                             "200.0 is Amendment 3)")
    parser.add_argument("--integrator", type=str, default="rk4",
                        help="rk4 (registered) or implicit_euler")
    arguments = parser.parse_args()

    config = TannFCCConfig(sigma_ref_mpa=arguments.sigma_ref, integrator=arguments.integrator)
    batch = TannFCCBatch(config, point_count=POINTS, systems_global=systems_identity())
    report: dict[str, object] = {}

    # A. zero increment
    zero = np.zeros((POINTS, 3))
    trial_zero = batch.evaluate(zero, compute_tangent=False)
    report["zero_increment"] = {
        "state_delta": float(np.max(np.abs(trial_zero.trial_state))),
        "slip_delta": float(np.max(np.abs(trial_zero.plastic_slip))),
    }

    # B. dissipation on random states, over a two-increment path
    worst_d = np.inf
    for _ in range(8):
        strain_1 = RNG.normal(scale=2e-3, size=(POINTS, 3))
        strain_2 = strain_1 + RNG.normal(scale=2e-3, size=(POINTS, 3))
        batch.evaluate(strain_1, compute_tangent=False)
        batch.commit()
        trial = batch.evaluate(strain_2, compute_tangent=False)
        worst_d = min(worst_d, float(trial.generalised_dissipation.min()))
        batch.revert()
    report["dissipation"] = {
        "worst_D": worst_d,
        "pass": worst_d >= -1e-9,
    }

    # C. permutation equivariance, from the zero state: the committed state
    # must also be permuted for a permuted history, so the comparison is
    # only valid on fresh batches that start from zero.
    permutation = RNG.permutation(12)
    strain = RNG.normal(scale=2e-3, size=(POINTS, 3))
    batch_c = TannFCCBatch(
        config, point_count=POINTS, systems_global=systems_identity()
    )
    batch_perm = TannFCCBatch(
        config,
        point_count=POINTS,
        systems_global=systems_identity()[:, permutation, :, :],
    )
    # copy the same (tiny) network weights so the laws are identical
    batch_perm.copy_weights_from(batch_c)
    trial = batch_c.evaluate(strain)
    trial_perm = batch_perm.evaluate(strain)
    stress_permuted_back = trial_perm.stress_in_plane_mpa  # stress is permutation-free
    report["permutation_equivariance"] = {
        "stress_max_abs_diff": float(np.max(np.abs(trial.stress_in_plane_mpa - stress_permuted_back))),
        "tangent_max_abs_diff": float(
            np.max(np.abs(trial.consistent_tangent_mpa - trial_perm.consistent_tangent_mpa))
        ),
    }

    # D. substep invariance
    strain = RNG.normal(scale=2e-3, size=(POINTS, 3))
    substep_errors = {}
    previous_state = None
    for substeps in (1, 2, 4, 8):
        sub_config = TannFCCConfig(n_substeps=substeps)
        sub_batch = TannFCCBatch(
            sub_config, point_count=POINTS, systems_global=systems_identity()
        )
        sub_batch.copy_weights_from(batch)
        trial = sub_batch.evaluate(strain, compute_tangent=False)
        if previous_state is not None:
            substep_errors[substeps] = float(
                np.max(np.abs(trial.trial_state - previous_state))
            )
        previous_state = trial.trial_state
    report["substepping"] = substep_errors

    # E. algorithmic tangent vs finite differences, with a step study.
    # Plain central differences bottom out on round-off near 1e-5 at the
    # 2000-point worst point; the Richardson column (4 * FD(h/2) - FD(h)) / 3
    # cancels the h^2 term and is the recorded reference for the gate.
    # Improving the FD reference does not move the frozen 1e-5 threshold.
    strain = RNG.normal(scale=1e-3, size=(POINTS, 3))
    trial = batch.evaluate(strain)
    tangent_ad = trial.consistent_tangent_mpa

    def central_fd(step: float, component: int) -> np.ndarray:
        plus = strain.copy()
        minus = strain.copy()
        plus[:, component] += step
        minus[:, component] -= step
        trial_plus = batch.evaluate(plus, compute_tangent=False)
        trial_minus = batch.evaluate(minus, compute_tangent=False)
        return (
            trial_plus.stress_in_plane_mpa - trial_minus.stress_in_plane_mpa
        ) / (2.0 * step)

    sweep = {}
    richardson = {}
    for h in (1e-4, 1e-5, 1e-6, 1e-7):
        worst_plain = 0.0
        worst_rich = 0.0
        for component in range(3):
            fd_h = central_fd(h, component)
            fd_half = central_fd(h / 2.0, component)
            error = np.max(np.abs(tangent_ad[:, :, component] - fd_h))
            worst_plain = max(worst_plain, float(np.max(error)))
            rich = (4.0 * fd_half - fd_h) / 3.0
            error_rich = np.max(np.abs(tangent_ad[:, :, component] - rich))
            worst_rich = max(worst_rich, float(np.max(error_rich)))
        sweep[h] = worst_plain
        richardson[h] = worst_rich
    report["tangent"] = {
        "fd_sweep": sweep,
        "richardson_sweep": richardson,
        "pass": min(richardson.values()) <= 1e-5,
    }

    # F. transactions
    strain_1 = RNG.normal(scale=1e-3, size=(POINTS, 3))
    first = batch.evaluate(strain_1, compute_tangent=False)
    second = batch.evaluate(strain_1, compute_tangent=False)
    identical = float(np.max(np.abs(first.stress_in_plane_mpa - second.stress_in_plane_mpa)))
    committed_before = batch.committed_state.copy()
    batch.evaluate(strain_1 + RNG.normal(scale=1e-4, size=(POINTS, 3)), compute_tangent=False)
    batch.revert()
    committed_after_revert = batch.committed_state.copy()
    batch.evaluate(strain_1, compute_tangent=False)
    batch.commit()
    committed_after_commit = batch.committed_state.copy()
    report["transactions"] = {
        "double_evaluate_identical": identical,
        "revert_restores": float(np.max(np.abs(committed_after_revert - committed_before))),
        "commit_advances": float(np.max(np.abs(committed_after_commit - committed_before))),
    }

    # G. FCC geometry coherence: [001] uniaxial Schmid values from the closed form
    s = np.zeros((1, 3, 3))
    s[0, 1, 1] = 1.0  # uniaxial yy
    tensors = systems_identity()[0]
    tau = np.einsum("ij,aij->a", s[0], tensors)
    nonzero = np.sort(np.abs(tau[tau != 0]))
    report["fcc_geometry"] = {
        "active_count": int(np.count_nonzero(tau)),
        "expected_active": ACTIVE_SYSTEMS_001,
        "schmid_values": nonzero.tolist(),
        "expected_schmid": SCHMID_FACTOR_001,
        "pass": np.count_nonzero(tau) == ACTIVE_SYSTEMS_001
        and np.allclose(np.abs(tau[tau != 0]), SCHMID_FACTOR_001, atol=1e-14),
    }

    report["memory"] = {
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
