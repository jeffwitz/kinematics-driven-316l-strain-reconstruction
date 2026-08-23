#!/usr/bin/env python3
"""Minimal dynamical memory per system: which family closes the phase space?

Per `validation/memory_families_preregistration.md`: three memory families
(pure accumulation, saturating, signed) evolved causally on the decomposed
activities, then `(|tau|, memory) -> |Delta gamma|` in the same pooled
leave-one-state-out k-NN as the ladder, with the family parameters tuned
per fold on the training states only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
N_SYSTEMS = 12
N_STATES = 20
SUBSAMPLE_PER_STATE = 20000
TUNE_SUBSAMPLE = 100000
K = 50
GRID_SAT = (0.5, 1.0, 2.0, 4.0)
GRID_D = (0.0, 0.5, 2.0)


def binned_r2(features: np.ndarray, gamma_abs: np.ndarray) -> float:
    """Variance reduction of |gamma| by quantile bins of the first feature."""

    edges = np.quantile(features[:, 0], np.linspace(0, 1, 11))
    idx = np.digitize(features[:, 0], edges[1:-1])
    global_var = float(np.var(gamma_abs))
    if global_var <= 0:
        return 0.0
    within, total = 0.0, 0
    for i in range(10):
        mask = idx == i
        count = int(mask.sum())
        if count < 100:
            continue
        within += count * float(np.var(gamma_abs[mask]))
        total += count
    return 1.0 - within / max(total * global_var, 1e-300)


def evolve(
    memory_kind: str, params: tuple, gamma_full: np.ndarray, sigma_gamma: float
) -> np.ndarray:
    """Causal evolution of the memory over all states, per (point, system).

    Returns the memory at the state *before* each increment (aligned like
    `Gamma`), with shape (n_states, n_points, 12, channels); F3 carries two
    channels (saturating z, signed x)."""
    n_states, n_points, n_systems = gamma_full.shape
    channels = 2 if memory_kind == "F3" else 1
    memory = np.zeros((n_states, n_points, n_systems, channels))
    z = np.zeros((n_points, n_systems))
    x = np.zeros((n_points, n_systems))
    z_sat = params[0]
    d = params[1]
    for n in range(n_states):
        memory[n, ..., 0] = z
        if channels == 2:
            memory[n, ..., 1] = x
        delta = np.abs(gamma_full[n]) / sigma_gamma
        sign = np.sign(gamma_full[n])
        if memory_kind == "F0":
            z = z + delta
        elif memory_kind == "F1":
            z = z + delta * np.maximum(1.0 - z / z_sat, 0.0)
        elif memory_kind == "F2":
            x = (1.0 - d * delta) * x + delta * sign
        else:  # F3: both
            z = z + delta * np.maximum(1.0 - z / z_sat, 0.0)
            x = (1.0 - d * delta) * x + delta * sign
    return memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=OUT / "fcc_slip_fields.npz")
    parser.add_argument("--output", type=Path, default=OUT / "memory_families.json")
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    tau_full = fields["tau"]
    gamma_full = fields["gamma"]  # flat (samples, 12)
    gamma_states = gamma_full.reshape(N_STATES, -1, N_SYSTEMS)  # for the causal evolution
    states_full = fields["states"]
    _n_states, n_points, _ = gamma_states.shape

    rng = np.random.default_rng(20260817)
    sample_idx = np.concatenate(
        [
            np.where(states_full == state)[0][
                rng.choice(np.sum(states_full == state), size=SUBSAMPLE_PER_STATE, replace=False)
            ]
            for state in range(N_STATES)
        ]
    )
    system_idx = rng.integers(0, N_SYSTEMS, size=len(sample_idx))
    tau = tau_full[sample_idx, system_idx]
    gamma = gamma_full[sample_idx, system_idx]
    gamma_abs = np.abs(gamma)
    tau_abs = np.abs(tau)
    state_vec = states_full[sample_idx]
    point_idx = sample_idx % n_points
    state_num = sample_idx // n_points

    families = {
        "F0": [(1.0, 0.0)],
        "F1": [(z, 0.0) for z in GRID_SAT],
        "F2": [(1.0, d) for d in GRID_D],
        "F3": [(z, d) for z in (1.0, 2.0) for d in (0.5, 2.0)],
    }

    fold_rows: dict[str, list] = {name: [] for name in families}
    for state in range(N_STATES):
        test = state_vec == state
        train = ~test
        train_indices = np.where(train)[0]
        tune_mask = train_indices[
            rng.choice(
                len(train_indices), size=min(TUNE_SUBSAMPLE, len(train_indices)), replace=False
            )
        ]
        sigma_gamma = max(float(np.std(gamma_abs[tune_mask])), 1e-300)

        baseline_features = np.stack([tau_abs, np.abs(gamma)], axis=1)
        baseline_train = baseline_features[train]
        scaler = StandardScaler().fit(baseline_train)
        tree = cKDTree(scaler.transform(baseline_train))
        _, idx = tree.query(scaler.transform(baseline_features[test]), k=K)
        pred = gamma_abs[train][idx].mean(axis=1)
        # Gamma baseline as the F0 family (no tuning).
        memory_gamma = np.abs(gamma_states).cumsum(axis=0) - np.abs(gamma_states)
        mem_features = np.stack([tau_abs, memory_gamma[state_num, point_idx, system_idx]], axis=1)
        scaler = StandardScaler().fit(mem_features[train])
        tree = cKDTree(scaler.transform(mem_features[train]))
        _, idx = tree.query(scaler.transform(mem_features[test]), k=K)
        pred = gamma_abs[train][idx].mean(axis=1)
        gamma_r2 = 1.0 - float(np.sum((gamma_abs[test] - pred) ** 2)) / max(
            float(np.sum((gamma_abs[test] - np.mean(gamma_abs[train])) ** 2)), 1e-300
        )
        fold_rows["F0"].append({"state": state, "r2": gamma_r2, "params": (1.0, 0.0)})

        for family in ("F1", "F2", "F3"):
            best = (-1e300, None)

            def family_features(memory: np.ndarray) -> np.ndarray:
                mem = memory[state_num, point_idx, system_idx]
                if memory.shape[-1] == 2:
                    return np.stack([tau_abs, mem[:, 0], mem[:, 1]], axis=1)
                return np.stack([tau_abs, mem[:, 0]], axis=1)

            for params in families[family]:
                memory = evolve(family, params, gamma_states, sigma_gamma)
                feats = family_features(memory)
                score = binned_r2(feats[tune_mask], gamma_abs[tune_mask])
                if score > best[0]:
                    best = (score, params)
            params = best[1]
            memory = evolve(family, params, gamma_states, sigma_gamma)
            feats = family_features(memory)
            scaler = StandardScaler().fit(feats[train])
            tree = cKDTree(scaler.transform(feats[train]))
            _, idx = tree.query(scaler.transform(feats[test]), k=K)
            pred = gamma_abs[train][idx].mean(axis=1)
            r2 = 1.0 - float(np.sum((gamma_abs[test] - pred) ** 2)) / max(
                float(np.sum((gamma_abs[test] - np.mean(gamma_abs[train])) ** 2)), 1e-300
            )
            fold_rows[family].append({"state": state, "r2": r2, "params": list(params)})
            print(
                f"  fold {state:2d} {family}: params {params}  r2 {r2:+.4f}  "
                f"(gamma baseline {gamma_r2:+.4f})",
                flush=True,
            )

    summary = {}
    for family, rows in fold_rows.items():
        r2s = [row["r2"] for row in rows]
        summary[family] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "params": [row["params"] for row in rows],
        }
    best_family = max(("F1", "F2", "F3"), key=lambda f: summary[f]["r2_mean"])
    payload = {
        "schema_version": 1,
        "bars": {"memory_bar": 0.30, "jump_bar": 0.10},
        "summary": summary,
        "best_family": best_family,
        "jump_over_gamma": summary[best_family]["r2_mean"] - summary["F0"]["r2_mean"],
        "verdict": {
            "memory_closes_the_gap": summary[best_family]["r2_mean"] >= 0.30,
            "jump_is_real": summary[best_family]["r2_mean"] - summary["F0"]["r2_mean"] >= 0.10,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k != "summary"},
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
