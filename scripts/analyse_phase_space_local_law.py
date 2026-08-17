#!/usr/bin/env python3
"""Phase-space analysis of the reconstructed inelastic trajectories.

The registered test of local constitutive structure, per
`validation/phase_space_local_law_preregistration.md`: on the
observable-projected increments (the kernel of the displacement operator is
removed first), does a similar local state produce a similar inelastic
increment?

* direction dispersion: circular standard deviation of the angle between the
  increment's deviatoric direction and `s`, within state bins (quantiles on
  `|s|` x deviatoric angle x `p_eq`, minimum 50 points), aggregated over the
  best populated decile of bins;
* amplitude structure: `R^2_cond(Delta p | S)` from the within-bin variance;
* coverage of the visited domain `Omega_P43`: populated bins, best-decile
  share, participation ratio.

The full cos-angle histogram is reported: the projected-Krylov field carries
a measured zero-work boundary mass (`f_0 ~ 0.47`), so the distribution is
expected to be bimodal — the boundary population and the interior one. A
secondary reading repeats the bin statistics on the interior points
(`|cos| >= 0.3`) only; the frozen primary readings stay on all active
points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
GAUGE = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)
MIN_BIN_POINTS = 50


def gauge_norm(field: np.ndarray) -> np.ndarray:
    """Pointwise plastic-gauge norm `sqrt(z^T G_p z)`."""

    return np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", field, GAUGE, field), 0.0))


def deviator(stress: np.ndarray) -> np.ndarray:
    """In-plane deviator of a 3-component Kelvin stress (zz implied)."""

    pressure = (stress[:, 0] + stress[:, 1]) / 3.0
    result = np.empty_like(stress)
    result[:, 0] = stress[:, 0] - pressure
    result[:, 1] = stress[:, 1] - pressure
    result[:, 2] = stress[:, 2]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=OUT / "krylov_trajectories.r16.npz",
    )
    parser.add_argument("--output", type=Path, default=OUT / "phase_space_analysis.json")
    arguments = parser.parse_args()

    data = np.load(arguments.trajectories, allow_pickle=False)
    stress = data["stress"]  # (states, points, 3) predictor stress
    d_eps = data["d_eps_inel_observable"]  # (states, points, 3)
    d_eps_raw = data["d_eps_inel"]
    eps_inel = data["eps_inel_observable"]

    n_states, n_points, _ = stress.shape
    flat_stress = stress.reshape(-1, 3)
    flat_d_eps = d_eps.reshape(-1, 3)
    flat_d_raw = d_eps_raw.reshape(-1, 3)
    flat_eps = eps_inel.reshape(-1, 3)

    s = deviator(flat_stress)
    s_norm_3d = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + (s[:, 0] + s[:, 1]) ** 2)
    n_dir = deviator(flat_d_eps)
    n_norm = np.sqrt((n_dir**2).sum(axis=1))
    amplitude = gauge_norm(flat_d_eps)
    amplitude_raw = gauge_norm(flat_d_raw)
    p_eq = gauge_norm(flat_eps)

    amplitude_floor = (
        np.quantile(amplitude[amplitude > 0], 0.9) if np.any(amplitude > 0) else 0.0
    )
    active = (amplitude > 1e-6 * amplitude_floor) & (s_norm_3d > 1e-2 * s_norm_3d.max())
    print(f"samples: {int(active.sum())} of {flat_stress.shape[0]} active", flush=True)

    cos = np.zeros(flat_stress.shape[0])
    cos[active] = np.einsum("pi,pi->p", s[active], n_dir[active]) / np.maximum(
        s_norm_3d[active] * n_norm[active], 1e-300
    )
    angle = np.arccos(np.clip(cos, -1.0, 1.0))

    # State features for binning: |s|, in-plane deviatoric angle, p_eq.
    d1 = (s[:, 0] - s[:, 1]) / np.sqrt(2.0)
    d2 = s[:, 2]
    angle_s = np.arctan2(d2, d1)

    def bin_stats(mask: np.ndarray, label: str) -> dict:
        edges_s = np.quantile(s_norm_3d[mask], np.linspace(0, 1, 9))
        edges_a = np.quantile(angle_s[mask], np.linspace(0, 1, 9))
        edges_p = np.quantile(p_eq[mask], np.linspace(0, 1, 5))
        idx_s = np.digitize(s_norm_3d, edges_s[1:-1])
        idx_a = np.digitize(angle_s, edges_a[1:-1])
        idx_p = np.digitize(p_eq, edges_p[1:-1])
        bins: list[dict] = []
        for i in range(8):
            for j in range(8):
                for k in range(4):
                    in_bin = mask & (idx_s == i) & (idx_a == j) & (idx_p == k)
                    count = int(in_bin.sum())
                    if count < MIN_BIN_POINTS:
                        continue
                    c = cos[in_bin]
                    a = angle[in_bin]
                    resultant = np.mean(np.exp(2j * a))
                    circular_std = float(np.sqrt(max(-2.0 * np.log(abs(resultant)), 0.0)) / 2.0)
                    bins.append(
                        {
                            "count": count,
                            "circular_std_degrees": float(np.degrees(circular_std)),
                            "mean_angle_degrees": float(np.degrees(np.mean(a))),
                            "mean_cos": float(np.mean(c)),
                            "boundary_fraction": float(np.mean(np.abs(c) < 1e-3)),
                            "amplitude_variance": float(np.var(amplitude[in_bin])),
                        }
                    )
        bins.sort(key=lambda b: -b["count"])
        cumulative = 0
        chosen: list[dict] = []
        for b in bins:
            chosen.append(b)
            cumulative += b["count"]
            if cumulative >= 0.1 * int(mask.sum()):
                break
        weights = np.asarray([b["count"] for b in chosen], dtype=np.float64)
        weights /= max(weights.sum(), 1e-300)
        global_var = float(np.var(amplitude[mask]))
        r2_cond = (
            1.0
            - float(np.sum([b["count"] * b["amplitude_variance"] for b in bins]))
            / max(float(np.sum([b["count"] for b in bins])) * global_var, 1e-300)
            if bins
            else None
        )
        counts_all = np.asarray([b["count"] for b in bins], dtype=np.float64)
        participation = (
            float(counts_all.sum() ** 2 / np.sum(counts_all**2)) if bins else 0.0
        )
        return {
            "label": label,
            "populated_bins": len(bins),
            "chosen_bins": len(chosen),
            "samples_in_chosen": cumulative,
            "chosen_share": float(cumulative / max(int(mask.sum()), 1)),
            "circular_std_degrees_mean": float(
                np.sum(weights * np.asarray([b["circular_std_degrees"] for b in chosen]))
            ),
            "mean_angle_degrees_mean": float(
                np.sum(weights * np.asarray([b["mean_angle_degrees"] for b in chosen]))
            ),
            "mean_cos_mean": float(
                np.sum(weights * np.asarray([b["mean_cos"] for b in chosen]))
            ),
            "r2_cond_amplitude": r2_cond,
            "participation_ratio": participation,
        }

    primary = bin_stats(active, "primary_all_active")
    interior_mask = active & (np.abs(cos) >= 0.3)
    secondary = bin_stats(interior_mask, "secondary_interior")

    hist, edges = np.histogram(cos[active], bins=20, range=(-1.0, 1.0))
    payload = {
        "schema_version": 1,
        "trajectories": str(arguments.trajectories),
        "n_states": int(n_states),
        "n_points": int(n_points),
        "primary": primary,
        "secondary_interior": secondary,
        "cos_histogram": {"edges": edges.tolist(), "counts": hist.tolist()},
        "coverage": {
            "active_samples": int(active.sum()),
            "interior_samples": int(interior_mask.sum()),
            "boundary_samples_1e3": int((active & (np.abs(cos) < 1e-3)).sum()),
        },
        "amplitude_global": {
            "mean": float(np.mean(amplitude[active])),
            "std": float(np.std(amplitude[active])),
            "raw_over_observable_mean_ratio": float(
                np.mean(amplitude_raw[active]) / max(np.mean(amplitude[active]), 1e-300)
            ),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
