#!/usr/bin/env python3
"""Phase-space clustering of the admissible inelastic trajectories.

The registered family search, per
`validation/phase_space_cluster_preregistration.md`: HDBSCAN over physically
constructed feature sets of increasing richness (stress, hardening level,
orientation), with the frozen bars on reconstruction robustness (AMI across
ranks), kernel exclusion (AMI raw vs observable `p_eq`), time-mixing, and
response conditioning (direction and amplitude structure within clusters).

The response (amplitude and flow direction) is measured but never used as a
clustering feature in the primary runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from fem_inhouse.core.kelvin import PLANE_STRESS_PLASTIC_GAUGE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
EBSD_PATH = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
GAUGE = np.asarray(PLANE_STRESS_PLASTIC_GAUGE, dtype=np.float64)
ORIGIN = (1580, 1030)
PIXELS = 100
SUBCELLS = 2
MIN_CLUSTER_SIZE = 50
MIN_SAMPLES = 5
SUBSAMPLE_PER_STATE = 2000


def deviator(stress: np.ndarray) -> np.ndarray:
    pressure = (stress[:, 0] + stress[:, 1]) / 3.0
    result = np.empty_like(stress)
    result[:, 0] = stress[:, 0] - pressure
    result[:, 1] = stress[:, 1] - pressure
    result[:, 2] = stress[:, 2]
    return result


def gauge_norm(field: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", field, GAUGE, field), 0.0))


def load_orientation() -> np.ndarray:
    """Euler angles and max Schmid factor on the window, per material point."""

    x0, y0 = ORIGIN
    with h5py.File(EBSD_PATH, "r") as handle:
        phi1 = np.asarray(handle["/orientation/phi1"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi = np.asarray(handle["/orientation/Phi"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        phi2 = np.asarray(handle["/orientation/phi2"])[x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1]
        schmid = np.asarray(handle["/schmid/max_schmid_factor"])[
            x0 : x0 + PIXELS + 1, y0 : y0 + PIXELS + 1
        ]

    def to_points(field: np.ndarray) -> np.ndarray:
        # Node grid (101x101) -> element mean (100x100) -> the two subcells.
        element_mean = 0.25 * (field[1:, 1:] + field[:-1, :-1] + field[1:, :-1] + field[:-1, 1:])
        return np.repeat(element_mean[:, :, None], SUBCELLS, axis=2).reshape(-1)

    return np.stack([to_points(phi1), to_points(phi), to_points(phi2), to_points(schmid)], axis=1)


def build_features(trajectory: dict, orientation: np.ndarray) -> dict[str, np.ndarray]:
    stress = trajectory["stress"].reshape(-1, 3)  # predictor stress per (state, point)
    eps_inel_obs = trajectory["eps_inel_observable"].reshape(-1, 3)
    eps_inel_raw = trajectory["eps_inel"].reshape(-1, 3)
    s = deviator(stress)
    q = np.sqrt(1.5) * np.sqrt(
        s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + (s[:, 0] + s[:, 1]) ** 2
    )
    p = -(stress[:, 0] + stress[:, 1]) / 3.0
    d1 = (s[:, 0] - s[:, 1]) / np.sqrt(2.0)
    angle_s = np.arctan2(s[:, 2], d1)
    p_eq_obs = gauge_norm(eps_inel_obs)
    p_eq_raw = gauge_norm(eps_inel_raw)
    n_states = trajectory["stress"].shape[0]
    orientation_rep = np.tile(orientation, (n_states, 1))
    phi1, phi, phi2, schmid = orientation_rep.T
    common = {
        "q": q,
        "p": p,
        "sin_angle": np.sin(angle_s),
        "cos_angle": np.cos(angle_s),
        "p_eq_obs": p_eq_obs,
        "p_eq_raw": p_eq_raw,
        "schmid": schmid,
        "sin_phi1": np.sin(phi1),
        "cos_phi1": np.cos(phi1),
        "sin_Phi": np.sin(phi),
        "cos_Phi": np.cos(phi),
        "sin_phi2": np.sin(phi2),
        "cos_phi2": np.cos(phi2),
    }
    feature_sets = {
        "F1": np.stack(
            [common["q"], common["p"], common["sin_angle"], common["cos_angle"]], axis=1
        ),
        "F2": np.stack(
            [
                common["q"],
                common["p"],
                common["sin_angle"],
                common["cos_angle"],
                common["p_eq_obs"],
            ],
            axis=1,
        ),
        "F3": np.stack(
            [
                common["q"],
                common["p"],
                common["sin_angle"],
                common["cos_angle"],
                common["p_eq_obs"],
                common["schmid"],
            ],
            axis=1,
        ),
        "F4": np.stack(
            [
                common["q"],
                common["p"],
                common["sin_angle"],
                common["cos_angle"],
                common["p_eq_obs"],
                common["schmid"],
                common["sin_phi1"],
                common["cos_phi1"],
                common["sin_Phi"],
                common["cos_Phi"],
                common["sin_phi2"],
                common["cos_phi2"],
            ],
            axis=1,
        ),
        "F2_raw_peq": np.stack(
            [
                common["q"],
                common["p"],
                common["sin_angle"],
                common["cos_angle"],
                common["p_eq_raw"],
            ],
            axis=1,
        ),
    }
    return feature_sets, common


def cluster_and_score(features: np.ndarray, states: np.ndarray, response: dict) -> dict:
    n_total = features.shape[0]
    rng = np.random.default_rng(20260817)
    indices = np.concatenate(
        [
            np.where(states == s)[0][rng.permutation(np.sum(states == s))[:SUBSAMPLE_PER_STATE]]
            for s in np.unique(states)
        ]
    )
    scaler = StandardScaler().fit(features[indices])
    scaled_all = scaler.transform(features)
    clusterer = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES).fit(
        scaled_all[indices]
    )
    labels_sub = clusterer.labels_
    # Medoid assignment for the full population.
    unique_labels = [lab for lab in np.unique(labels_sub) if lab >= 0]
    medoids = np.stack(
        [np.median(scaled_all[indices][labels_sub == lab], axis=0) for lab in unique_labels]
    )
    labels_full = np.full(n_total, -1, dtype=int)
    if len(unique_labels):
        distances = np.stack([np.linalg.norm(scaled_all - m, axis=1) for m in medoids], axis=1)
        labels_full = np.where(distances.min(axis=1) < np.inf, distances.argmin(axis=1), -1)
    # Merge small clusters into noise.
    counts = np.bincount(labels_full[labels_full >= 0], minlength=len(unique_labels))
    keep = np.where(counts >= 0.005 * n_total)[0]
    remap = {old: new for new, old in enumerate(keep.tolist())}
    labels_full = np.asarray([remap.get(label, -1) for label in labels_full], dtype=int)
    n_clusters = len(keep)

    # Time-mixing: max single-state share per cluster.
    time_ok = True
    worst_share = 0.0
    for lab in range(n_clusters):
        members = np.where(labels_full == lab)[0]
        shares = [float(np.mean(states[members] == s)) for s in np.unique(states)]
        worst_share = max(worst_share, max(shares))
        if max(shares) > 0.8:
            time_ok = False

    # Response conditioning (on the full assigned population).
    angle = response["angle"]
    amplitude = response["amplitude"]
    assigned = labels_full >= 0
    global_std = float(np.sqrt(max(-2 * np.log(abs(np.mean(np.exp(2j * angle[assigned])))), 0)) / 2)
    within_stds, within_vars, counts_list = [], [], []
    for lab in range(n_clusters):
        mask = labels_full == lab
        a = angle[mask]
        r = np.abs(np.mean(np.exp(2j * a)))
        within_stds.append(np.sqrt(max(-2 * np.log(r), 0)) / 2)
        within_vars.append(float(np.var(amplitude[mask])))
        counts_list.append(int(mask.sum()))
    weights = np.asarray(counts_list, dtype=np.float64) / max(np.sum(counts_list), 1)
    within_std = float(np.sum(weights * np.asarray(within_stds)))
    global_var = float(np.var(amplitude[assigned]))
    r2_cluster = 1.0 - float(np.sum(weights * np.asarray(within_vars))) / max(global_var, 1e-300)

    silhouette = float(
        silhouette_score(scaled_all[indices], labels_sub)
        if len(np.unique(labels_sub)) > 1
        else np.nan
    )
    return {
        "n_clusters": n_clusters,
        "noise_share": float(np.mean(labels_full < 0)),
        "silhouette": silhouette,
        "time_mixed": bool(time_ok),
        "worst_single_state_share": float(worst_share),
        "direction_improvement": float(global_std / max(within_std, 1e-300)),
        "global_circular_std_degrees": float(np.degrees(global_std)),
        "within_cluster_circular_std_degrees": float(np.degrees(within_std)),
        "r2_cluster_amplitude": r2_cluster,
        "labels_sub": labels_sub,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectories-r8",
        type=Path,
        default=OUT / "krylov_trajectories.r8.npz",
    )
    parser.add_argument(
        "--trajectories-r16",
        type=Path,
        default=OUT / "krylov_trajectories.r16.npz",
    )
    parser.add_argument("--output", type=Path, default=OUT / "phase_space_clusters.json")
    arguments = parser.parse_args()

    orientation = load_orientation()
    r8 = np.load(arguments.trajectories_r8, allow_pickle=False)
    r16 = np.load(arguments.trajectories_r16, allow_pickle=False)
    n_states = r16["stress"].shape[0]
    n_points = r16["stress"].shape[1]
    states = np.repeat(np.arange(n_states), n_points)
    loaded = np.zeros(n_states * n_points, dtype=bool)
    for tr in (r8, r16):
        s = deviator(tr["stress"].reshape(-1, 3))
        s_norm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + (s[:, 0] + s[:, 1]) ** 2)
        loaded |= s_norm > 1e-2 * s_norm.max()
    print(f"loaded samples: {int(loaded.sum())} of {n_states * n_points}", flush=True)

    def responses(tr: dict) -> dict:
        stress = tr["stress"].reshape(-1, 3)
        d_eps = tr["d_eps_inel_observable"].reshape(-1, 3)
        s = deviator(stress)
        s_norm = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2 + (s[:, 0] + s[:, 1]) ** 2)
        n_dir = deviator(d_eps)
        n_norm = np.sqrt((n_dir**2).sum(axis=1))
        cos = np.einsum("pi,pi->p", s, n_dir) / np.maximum(s_norm * n_norm, 1e-300)
        return {
            "angle": np.arccos(np.clip(cos, -1.0, 1.0)),
            "amplitude": gauge_norm(d_eps),
        }

    report: dict[str, dict] = {}
    labels_by_set: dict[str, tuple] = {}
    for name, tr in (("r8", r8), ("r16", r16)):
        feature_sets, _ = build_features(tr, orientation)
        resp = responses(tr)
        for set_name in ("F1", "F2", "F3", "F4", "F2_raw_peq"):
            result = cluster_and_score(feature_sets[set_name][loaded], states[loaded], resp)
            labels_by_set[f"{name}_{set_name}"] = result.pop("labels_sub")
            report[f"{name}_{set_name}"] = result
            print(
                f"{name} {set_name}: clusters {result['n_clusters']}  noise "
                f"{result['noise_share']:.2f}  silhouette {result['silhouette']:.3f}  "
                f"time_mixed {result['time_mixed']}  "
                f"dir_gain {result['direction_improvement']:.2f}  "
                f"r2_amp {result['r2_cluster_amplitude']:.2f}",
                flush=True,
            )

    # Frozen bars: AMI across ranks (same set) and raw vs observable p_eq.
    ami_rank = {
        set_name: float(
            adjusted_mutual_info_score(
                labels_by_set[f"r8_{set_name}"], labels_by_set[f"r16_{set_name}"]
            )
        )
        for set_name in ("F1", "F2", "F3", "F4")
    }
    ami_kernel = float(
        adjusted_mutual_info_score(labels_by_set["r16_F2"], labels_by_set["r16_F2_raw_peq"])
    )
    report["frozen_bars"] = {
        "ami_rank": ami_rank,
        "ami_kernel_raw_vs_observable": ami_kernel,
        "ami_rank_threshold": 0.5,
        "ami_kernel_threshold": 0.5,
        "direction_gain_threshold": 1.4,
        "r2_cluster_threshold": 0.5,
    }
    payload = {
        "schema_version": 1,
        "trajectories_r8": str(arguments.trajectories_r8),
        "trajectories_r16": str(arguments.trajectories_r16),
        "ebsd": str(EBSD_PATH),
        "loaded_samples": int(loaded.sum()),
        "results": report,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report["frozen_bars"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
