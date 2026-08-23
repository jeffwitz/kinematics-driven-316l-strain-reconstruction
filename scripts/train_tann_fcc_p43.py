#!/usr/bin/env python3
"""Train the corrected causal TANN-FCC T0 on the P43 displacement history.

The material is replayed causally from experimental state 0.  States 1--20
are warm-up states, states 31--32 are replayed but never scored because their
displacements are interpolated, and the measured DIC is not passed through a
second low-pass.  The FEM displacement alone is mapped through the declared
legacy-profile observation approximation before comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fem_inhouse.constitutive.tann_fcc import TannFCCBatch, TannFCCConfig
from fem_inhouse.constitutive.tann_fcc_geometry import systems_from_bunge_node_grid
from fem_inhouse.identification.dic_whitening import DICSpectralTransfer
from fem_inhouse.identification.tann_fcc_adjoint import TannFCCTrajectoryAdjoint
from fem_inhouse.identification.tann_fcc_sequence import TannFCCSequence
from fem_inhouse.spectral2d import EBISpectralSolverConfig, StructuredGrid2D
from fem_inhouse.spectral2d.transforms import SpectralTransformConfig

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "validation/reference_data/dic_multistep_history_p0043_repaired_v1"
OBSERVATION_CSV = (
    ROOT
    / "validation/reference_data/dic_legacy_profile_comparison_v1"
    / "case_c_legacy_corrected_warp/sinusoidal_transfer.csv"
)
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
OUT = ROOT / "validation/_generated/shared_tensor_generator"

PIXEL_SIZE_MM = 1.84e-3
YOUNG_MPA = 205_000.0
POISSON = 0.30
ORIGIN = (1580, 1030)
PIXELS = 100
REFERENCE_STATE = 0
STATES = list(range(1, 41))
LOSS_STATES = set(range(21, 41)) - {31, 32}
HOLDOUT = {24, 28, 36, 39}
TRAINING_STATES = LOSS_STATES - HOLDOUT
SEED = 20260817


def load_crop(pixels: int) -> tuple[np.ndarray, dict]:
    """The repaired P43 displacement history, cropped like the live script."""

    report = json.loads((HISTORY / "report.json").read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    x0, y0 = ORIGIN
    source = np.load(HISTORY / "repaired_history_mm.npy", mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + pixels - bounds[0] + 1,
            y0 - bounds[2] : y0 + pixels - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    return history, report


def load_ebsd_systems(pixels: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-material-point specimen-frame Schmid tensors from the EBSD map."""

    import h5py

    x0, y0 = ORIGIN
    with h5py.File(EBSD_PATH, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"/orientation/{name}"])[
                    x0 : x0 + pixels + 1, y0 : y0 + pixels + 1
                ]
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
        schmid = np.asarray(handle["/schmid/max_schmid_factor"])[
            x0 : x0 + pixels + 1, y0 : y0 + pixels + 1
        ]
    return systems_from_bunge_node_grid(angles, max_schmid_factor=schmid)


@dataclass(frozen=True, slots=True)
class StepReport:
    step: int
    loss_raw: float
    loss_observation: float | None
    e_train: dict[int, float]
    e_holdout: dict[int, float]
    median_e_holdout: float
    gradient_norm: float
    rollout_seconds: float
    adjoint_seconds: float
    newton_iterations: int
    min_d: float
    total_d: float
    slip_activity: float


def elastic_reference(
    grid: StructuredGrid2D, history: np.ndarray, pixels: int
) -> dict[int, np.ndarray]:
    """The conversion-corrected elastic lift, per state (the live script's)."""

    from fem_inhouse.core.kelvin import KELVIN_SCALE_2D
    from fem_inhouse.identification.tensor_plastic_observability import (
        TensorPlasticObservabilityOperator,
    )
    from fem_inhouse.spectral2d.newton_ebi import pack_interior

    class _Identity:
        """Local no-op field operator (the live script's own)."""

        def apply(self, values):
            return np.asarray(values)

    operator = TensorPlasticObservabilityOperator.build(
        grid,
        young_modulus_mpa=YOUNG_MPA,
        poisson_ratio=POISSON,
        transfer=_Identity(),
        whitener=_Identity(),
    )

    def kelvin_strain(field: np.ndarray) -> np.ndarray:
        return operator.kelvin_strain(field).reshape(-1, 3)

    def stress_of(strain: np.ndarray) -> np.ndarray:
        return np.einsum("pi,pij->pj", strain.reshape(-1, 3), operator.elasticity)

    def divergence(stress_kelvin: np.ndarray) -> np.ndarray:
        voigt = stress_kelvin.reshape(-1, 3) / KELVIN_SCALE_2D
        return pack_interior(
            operator.kinematics.divergence_from_sample_stress(
                voigt.reshape((pixels, pixels, 2, 3))
            )
        )

    reference = history[REFERENCE_STATE]
    elastic: dict[int, np.ndarray] = {}
    for state in STATES:
        field = history[state] - reference
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            pixels - 1, pixels - 1, 2
        )
        elastic[state] = lifted
    return elastic


def interior_norm(field: np.ndarray) -> float:
    return float(np.linalg.norm(field[1:-1, 1:-1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=OUT / "tann_fcc_p43_run.json")
    parser.add_argument("--pixels", type=int, default=PIXELS,
                        help="crop size (100 is the registered run; smaller only for smoke tests)")
    parser.add_argument("--max-increments", type=int, default=None,
                        help="limit the trajectory to the first N states (development)")
    parser.add_argument("--sigma-ref", type=float, default=None,
                        help="force reference in MPa (None -> 2 mu per Amendment 1; "
                             "200.0 is Amendment 3)")
    parser.add_argument("--equilibrium-tolerance", type=float, default=1.0e-10,
                        help="solver equilibrium tolerance (recorded in the artifact)")
    parser.add_argument("--trajectory-checkpoint", type=Path, default=None,
                        help="per-increment restart archive (defaults to <output>.trajectory.npz)")
    parser.add_argument("--training-checkpoint", type=Path, default=None,
                        help="Adam/model checkpoint (defaults to <output>.weights)")
    parser.add_argument("--resume-training", action="store_true",
                        help="resume model and Adam state from --training-checkpoint")
    parser.add_argument("--gmres-tolerance", type=float, default=None,
                        help="GMRES relative tolerance (None -> solver default 1e-8)")
    parser.add_argument("--line-search-reductions", type=int, default=None,
                        help="maximum line-search halvings (None -> solver default 8)")
    arguments = parser.parse_args()
    pixels = arguments.pixels

    started = time.perf_counter()
    history, report = load_crop(pixels)
    systems, validity = load_ebsd_systems(pixels)
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    reference = history[REFERENCE_STATE]
    measured = np.stack([history[s] - reference for s in STATES], axis=0)
    boundary = np.concatenate(
        [np.zeros_like(measured[:1]), measured], axis=0
    )  # zero reference + one entry per increment
    state_indices = STATES
    if arguments.max_increments is not None:
        boundary = boundary[: arguments.max_increments + 1]
        measured = measured[: arguments.max_increments]
        state_indices = STATES[: arguments.max_increments]
    active_holdout = HOLDOUT & set(state_indices)
    active_training = TRAINING_STATES & set(state_indices)
    if not active_training or not active_holdout:
        raise ValueError(
            "the truncated trajectory must include at least one training and one holdout state"
        )

    observation = DICSpectralTransfer.from_sinusoidal_csv(OBSERVATION_CSV)
    solver_kwargs: dict = {}
    if arguments.gmres_tolerance is not None:
        solver_kwargs["gmres_relative_tolerance"] = arguments.gmres_tolerance
    if arguments.line_search_reductions is not None:
        solver_kwargs["maximum_line_search_reductions"] = arguments.line_search_reductions
    solver_config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=arguments.equilibrium_tolerance,
        transform=SpectralTransformConfig(backend="fftw", fftw_planner_effort="estimate"),
        **solver_kwargs,
        # The strongly plastic operating point (Amendment 3) makes some
        # increments hard for plain Newton: the solver's own adaptive
        # stepping subdivides them, which also keeps the RK4 trial
        # excursions inside the integrator's stability margin.
        adaptive_stepping_enabled=True,
        # NOTE: reference_update_mode="per_increment" was tried and is a
        # measured REGRESSION (the 25x25 run that converged 17/17 with the
        # initial reference failed at increment 3 with it) -- kept out.
        progress_callback=lambda event: (
            print(
                f"  [{event.get('event', '?')}] "
                f"{event.get('increment', '')}{event.get('newton_iteration', '')}",
                flush=True,
            )
            if event.get("event") in {"increment_converged", "increment_failed"}
            else (
                print(
                    f"  [newton_residual] {event.get('increment', '')}."
                    f"{event.get('newton_iteration', '')} "
                    f"{event.get('relative_residual', 0.0):.3e}",
                    flush=True,
                )
                if event.get("event") == "newton_residual"
                and int(event.get("newton_iteration", 0) or 0) >= 25
                else None
            )
        ),
    )
    run_config = TannFCCConfig(seed=SEED, sigma_ref_mpa=arguments.sigma_ref)
    material = TannFCCBatch(
        run_config,
        point_count=2 * pixels * pixels,
        systems_global=systems,
    )
    trajectory_checkpoint = arguments.trajectory_checkpoint or arguments.output.with_suffix(
        ".trajectory.npz"
    )
    if trajectory_checkpoint.exists():
        # A fresh trajectory must not inherit another run's increments.  A
        # partial constitutive trajectory cannot be resumed during training:
        # its initial state also depends on the network parameters, and that
        # missing sensitivity would invalidate the adjoint gradient.
        trajectory_checkpoint.unlink()
    sequence = TannFCCSequence(
        grid=grid,
        material=material,
        boundary_history=boundary,
        measured_interior=measured,
        state_indices=state_indices,
        holdout=active_holdout,
        training_states=active_training,
        observation=observation.apply_without_wrap,
        observation_adjoint=observation.adjoint_without_wrap,
        solver_config=solver_config,
        checkpoint_path=trajectory_checkpoint,
    )
    elastic = elastic_reference(grid, history, pixels)

    optimizer = torch.optim.Adam(material._network.parameters(), lr=arguments.learning_rate)
    training_checkpoint = arguments.training_checkpoint or arguments.output.with_suffix(
        ".weights"
    )
    training_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []
    first_step = 0
    if arguments.resume_training:
        checkpoint = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
        material._network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        first_step = int(checkpoint["completed_step"]) + 1
        if arguments.output.exists():
            previous = json.loads(arguments.output.read_text(encoding="utf-8"))
            steps = list(previous.get("steps", []))
    elif training_checkpoint.exists():
        training_checkpoint.unlink()
    for step in range(first_step, arguments.steps):
        rollout_started = time.perf_counter()
        result = sequence.rollout()
        rollout_seconds = time.perf_counter() - rollout_started
        records = result.records

        def e_metric(record) -> float:
            model_residual = interior_norm(record.displacement - record.measured_displacement)
            elastic_residual = interior_norm(
                elastic[record.state] - record.measured_displacement
            )
            return model_residual / max(elastic_residual, 1e-30)

        e_train = {record.state: e_metric(record) for record in records if record.training}
        e_holdout = {record.state: e_metric(record) for record in records if record.holdout}

        adjoint = TannFCCTrajectoryAdjoint(
            grid=grid,
            material=material,
            records=records,
            observation_adjoint=observation.adjoint_without_wrap,
        )
        adjoint_started = time.perf_counter()
        dtheta, _diagnostics = adjoint.sweep()
        adjoint_seconds = time.perf_counter() - adjoint_started

        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(
            material._network.parameters(), dtheta, strict=True
        ):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad += torch.from_numpy(gradient)
        gradient_norm = float(
            np.sqrt(sum(float(np.sum(g**2)) for g in dtheta))
        )
        optimizer.step()
        temporary_checkpoint = training_checkpoint.with_suffix(
            training_checkpoint.suffix + ".tmp"
        )
        torch.save(
            {
                "completed_step": step,
                "model_state_dict": material._network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "seed": SEED,
                "history_reference_state": REFERENCE_STATE,
                "states": state_indices,
            },
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(training_checkpoint)

        report_step = {
            "step": step,
            "loss_raw": result.total_loss_raw,
            "loss_observation": result.total_loss_observation,
            "e_train": e_train,
            "e_holdout": e_holdout,
            "median_e_holdout": float(np.median(list(e_holdout.values()))),
            "gradient_norm": gradient_norm,
            "rollout_seconds": rollout_seconds,
            "adjoint_seconds": adjoint_seconds,
            "newton_iterations": [
                attempt.newton_iterations
                for attempt in result.solver_diagnostics.load_step_attempts
            ],
            "dissipation_min": float(min(r.dissipation.min() for r in records)),
            "dissipation_total": float(sum(r.dissipation.sum() for r in records)),
            "slip_activity": float(
                sum(np.abs(r.committed_state[..., 0]).sum() for r in records)
            ),
        }
        steps.append(report_step)
        print(
            f"step {step}: loss_observation {result.total_loss_observation:.6e} "
            f"E_holdout {report_step['median_e_holdout']:.4f} "
            f"grad {gradient_norm:.3e} ({rollout_seconds:.0f}s + {adjoint_seconds:.0f}s)",
            flush=True,
        )
        # checkpoint after every step: a killed run keeps everything it scored
        partial = {
            "git_sha": None,
            "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "machine": __import__("platform").node(),
            "seed": SEED,
            "states": state_indices,
            "holdout": sorted(active_holdout),
            "training_states": sorted(active_training),
            "replay_only_states": sorted(
                set(state_indices) - active_holdout - active_training
            ),
            "history_reference_state": REFERENCE_STATE,
            "observation": {
                "kind": "legacy_profile_spectral_approximation",
                "source_csv": str(OBSERVATION_CSV),
                "loss": "O(u_fem) - u_dic",
            },
            "architecture": asdict(run_config),
            "optimizer": {"name": "adam", "lr": arguments.learning_rate},
            "training_checkpoint": str(training_checkpoint),
            "solver": {
                "relative_equilibrium_tolerance": solver_config.relative_equilibrium_tolerance,
                "adaptive_stepping_enabled": solver_config.adaptive_stepping_enabled,
            },
            "steps": steps,
            "total_seconds": time.perf_counter() - started,
            "ebsd_validity_fraction": float(np.mean(validity)),
            "source_report_solve_bounds": report.get("solve_bounds"),
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(partial, indent=2, default=str) + "\n")

    artifact = {
        "git_sha": None,
        "dirty": None,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": __import__("platform").node(),
        "seed": SEED,
        "states": state_indices,
        "holdout": sorted(active_holdout),
        "training_states": sorted(active_training),
        "replay_only_states": sorted(
            set(state_indices) - active_holdout - active_training
        ),
        "history_reference_state": REFERENCE_STATE,
        "observation": {
            "kind": "legacy_profile_spectral_approximation",
            "source_csv": str(OBSERVATION_CSV),
            "loss": "O(u_fem) - u_dic",
        },
        "architecture": asdict(run_config),
        "optimizer": {"name": "adam", "lr": arguments.learning_rate},
        "training_checkpoint": str(training_checkpoint),
        "solver": {
            "relative_equilibrium_tolerance": solver_config.relative_equilibrium_tolerance,
            "adaptive_stepping_enabled": solver_config.adaptive_stepping_enabled,
        },
        "steps": steps,
        "total_seconds": time.perf_counter() - started,
        "ebsd_validity_fraction": float(np.mean(validity)),
        "source_report_solve_bounds": report.get("solve_bounds"),
    }
    try:
        import subprocess

        artifact["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
        artifact["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=ROOT,
            ).stdout.strip()
        )
    except Exception:  # pragma: no cover - provenance only
        pass
    # Per-state fields of the final trajectory, beside the JSON: the
    # figures (B, C, D, F) are generated from these, never from copied
    # values. The elastic reference is included for E_n.
    fields_path = arguments.output.with_suffix(".npz")
    field_dict: dict[str, np.ndarray] = {}
    for record in records:
        prefix = f"state_{record.state}"
        field_dict[f"{prefix}_u_sim"] = record.displacement
        field_dict[f"{prefix}_u_meas"] = record.measured_displacement
        field_dict[f"{prefix}_stress"] = record.stress_in_plane_mpa
        field_dict[f"{prefix}_committed_state"] = record.committed_state
        field_dict[f"{prefix}_dissipation"] = record.dissipation
        field_dict[f"{prefix}_u_elastic"] = elastic[record.state]
    np.savez_compressed(fields_path, **field_dict)
    artifact["fields_path"] = str(fields_path)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"artifact: {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
