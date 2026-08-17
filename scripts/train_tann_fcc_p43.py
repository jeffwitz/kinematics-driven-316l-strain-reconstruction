#!/usr/bin/env python3
"""Train the causal TANN-FCC (T0) on the P43 100x100 masked-state sequence.

The first real run of `validation/tann_fcc_preregistration.md`: one causal
trajectory over states 21-40, holdout {24, 28, 32, 36, 39}, whitened
displacement loss on the interior DOF, gradients by the discrete
trajectory adjoint (`fem_inhouse.identification.tann_fcc_adjoint`), and
the per-state metric E_n relative to the elastic reference. The elastic
reference is the conversion-corrected lift of `learn_flow_direction_p43.py`
(the only current construction of it). Every real run writes one
self-contained JSON artifact; figures are generated from that artifact,
never from copied values.
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
WHITENER_CSV = (
    ROOT / "validation/reference_data/dic_measurement_chain_v4/sinusoidal_transfer.csv"
)
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
OUT = ROOT / "validation/_generated/shared_tensor_generator"

PIXEL_SIZE_MM = 1.84e-3
YOUNG_MPA = 205_000.0
POISSON = 0.30
ORIGIN = (1580, 1030)
PIXELS = 100
REFERENCE_STATE = 20
STATES = list(range(21, 41))
HOLDOUT = {24, 28, 32, 36, 39}
SEED = 20260817


def load_crop() -> tuple[np.ndarray, dict]:
    """The repaired P43 displacement history, cropped like the live script."""

    report = json.loads((HISTORY / "report.json").read_text(encoding="utf-8"))
    bounds = list(map(int, report["solve_bounds"]))
    x0, y0 = ORIGIN
    source = np.load(HISTORY / "repaired_history_mm.npy", mmap_mode="r", allow_pickle=False)
    history = np.asarray(
        source[
            :,
            x0 - bounds[0] : x0 + PIXELS - bounds[0] + 1,
            y0 - bounds[2] : y0 + PIXELS - bounds[2] + 1,
            :,
        ],
        dtype=np.float64,
    )
    return history, report


def load_ebsd_systems() -> tuple[np.ndarray, np.ndarray]:
    """Per-material-point specimen-frame Schmid tensors from the EBSD map."""

    import h5py

    x0, y0 = ORIGIN
    with h5py.File(EBSD_PATH, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"/orientation/{name}"])[
                    x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
                ]
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
        schmid = np.asarray(handle["/schmid/max_schmid_factor"])[
            x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
        ]
    return systems_from_bunge_node_grid(angles, max_schmid_factor=schmid)


@dataclass(frozen=True, slots=True)
class StepReport:
    step: int
    loss_raw: float
    loss_whitened: float | None
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


def elastic_reference(grid: StructuredGrid2D, history: np.ndarray) -> dict[int, np.ndarray]:
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
                voigt.reshape((PIXELS, PIXELS, 2, 3))
            )
        )

    reference = history[REFERENCE_STATE]
    elastic: dict[int, np.ndarray] = {}
    for state in STATES:
        field = history[state] - reference
        forcing = -divergence(stress_of(kelvin_strain(field))) / operator.quadrature_weight
        lifted = field.copy()
        lifted[1:-1, 1:-1, :] -= operator.solve_stiffness(forcing).reshape(
            PIXELS - 1, PIXELS - 1, 2
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
    arguments = parser.parse_args()
    pixels = arguments.pixels
    global PIXELS
    PIXELS = pixels

    started = time.perf_counter()
    history, report = load_crop()
    systems, validity = load_ebsd_systems()
    grid = StructuredGrid2D(pixels, pixels, PIXEL_SIZE_MM * pixels, PIXEL_SIZE_MM * pixels)
    reference = history[REFERENCE_STATE]
    boundary = np.stack([history[s] - reference for s in STATES], axis=0)
    measured = boundary.copy()
    state_indices = STATES
    if arguments.max_increments is not None:
        boundary = boundary[: arguments.max_increments]
        measured = measured[: arguments.max_increments]
        state_indices = STATES[: arguments.max_increments]

    whitener = DICSpectralTransfer.from_sinusoidal_csv(WHITENER_CSV)
    solver_config = EBISpectralSolverConfig(
        relative_equilibrium_tolerance=1.0e-10,
        transform=SpectralTransformConfig(backend="fftw", fftw_planner_effort="estimate"),
    )
    material = TannFCCBatch(
        TannFCCConfig(seed=SEED),
        point_count=2 * PIXELS * PIXELS,
        systems_global=systems,
    )
    sequence = TannFCCSequence(
        grid=grid,
        material=material,
        boundary_history=boundary,
        measured_interior=measured,
        state_indices=state_indices,
        holdout=HOLDOUT,
        whitener=lambda field: whitener.apply_without_wrap(field),
        solver_config=solver_config,
    )
    elastic = elastic_reference(grid, history)

    optimizer = torch.optim.Adam(material._network.parameters(), lr=arguments.learning_rate)
    steps: list[dict] = []
    for step in range(arguments.steps):
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

        e_train = {record.state: e_metric(record) for record in records if not record.holdout}
        e_holdout = {record.state: e_metric(record) for record in records if record.holdout}

        adjoint = TannFCCTrajectoryAdjoint(
            grid=grid,
            material=material,
            records=records,
            whitener=lambda field: whitener.apply_without_wrap(field),
        )
        adjoint_started = time.perf_counter()
        dtheta, diagnostics = adjoint.sweep()
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

        report_step = {
            "step": step,
            "loss_raw": result.total_loss_raw,
            "loss_whitened": result.total_loss_whitened,
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
            "dissipation_min": float(min(r.generalised_dissipation.min() for r in records)),
            "dissipation_total": float(sum(r.generalised_dissipation.sum() for r in records)),
            "slip_activity": float(
                sum(np.abs(r.committed_state[..., 0]).sum() for r in records)
            ),
        }
        steps.append(report_step)
        print(
            f"step {step}: loss_whit {result.total_loss_whitened:.6e} "
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
            "holdout": sorted(HOLDOUT),
            "architecture": asdict(TannFCCConfig(seed=SEED)),
            "optimizer": {"name": "adam", "lr": arguments.learning_rate},
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
        "holdout": sorted(HOLDOUT),
        "architecture": asdict(TannFCCConfig(seed=SEED)),
        "optimizer": {"name": "adam", "lr": arguments.learning_rate},
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
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        )
    except Exception:  # pragma: no cover - provenance only
        pass
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    print(f"artifact: {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
