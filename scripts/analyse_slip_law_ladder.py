#!/usr/bin/env python3
"""The shared slip-law ladder, in leave-one-state-out.

Per `validation/slip_law_ladder_preregistration.md`: one k-NN law for all
twelve systems (the model never sees the system index), evaluated on held-out
increments, over the feature ladder

  S1: (tau^alpha)
  S2: (tau^alpha, Gamma^alpha_{n-1})
  S3: (tau^alpha, Gamma^alpha_{n-1}, Gamma^{beta!=alpha}_{n-1})
  S5: (tau^alpha, Gamma^1_{n-1}, ..., Gamma^12_{n-1})

with Gamma always causal (the history before the increment being predicted).
The ladder runs on both the L2 and the time-regularised decompositions — the
gauge test: if the laws agree, the structure is imposed by the experimental
tensors, not by the pseudo-inverse choice.
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
K = 50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=OUT / "fcc_slip_fields.npz")
    parser.add_argument("--output", type=Path, default=OUT / "slip_law_ladder.json")
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    tau_full = fields["tau"]
    states_full = fields["states"]

    def build_variant(gamma_name: str, cumulative_name: str) -> dict:
        gamma_full = fields[gamma_name]
        cumulative_full = fields[cumulative_name]
        n_samples = tau_full.shape[0]
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
        # Causal history: the cumulative up to (state, point) minus the current
        # increment's own magnitude — the history *before* the response.
        current = np.abs(gamma_full[sample_idx, :])
        history = cumulative_full[sample_idx, :] - current  # (M, 12), all systems
        gamma_history = history[np.arange(len(sample_idx)), system_idx]
        history_others = history.sum(axis=1) - gamma_history
        ladders = {
            "S1": np.stack([tau], axis=1),
            "S2": np.stack([tau, gamma_history], axis=1),
            "S3": np.stack([tau, gamma_history, history_others], axis=1),
            "S5": np.concatenate([tau[:, None], history], axis=1),
        }
        return {
            "tau": tau,
            "gamma": gamma,
            "system": system_idx,
            "state": states_full[sample_idx],
            "ladders": ladders,
        }

    report: dict[str, dict] = {}
    for name, (gamma_key, cumulative_key) in (
        ("l2", ("gamma", "gamma_cumulative")),
        ("temporal", ("gamma_temporal", "gamma_temporal_cumulative")),
    ):
        data = build_variant(gamma_key, cumulative_key)
        state_vec = data["state"]
        for ladder_name, features in data["ladders"].items():
            predictions = np.empty_like(data["gamma"])
            for state in range(N_STATES):
                test = state_vec == state
                train = ~test
                scaler = StandardScaler().fit(features[train])
                tree = cKDTree(scaler.transform(features[train]))
                _, idx = tree.query(scaler.transform(features[test]), k=K)
                predictions[test] = data["gamma"][train][idx].mean(axis=1)
            residual = data["gamma"] - predictions
            global_mean = np.mean(data["gamma"])
            r2 = 1.0 - float(np.sum(residual**2)) / max(
                float(np.sum((data["gamma"] - global_mean) ** 2)), 1e-300
            )
            per_system = {}
            for system in range(N_SYSTEMS):
                mask = data["system"] == system
                r2_system = 1.0 - float(np.sum(residual[mask] ** 2)) / max(
                    float(np.sum((data["gamma"][mask] - global_mean) ** 2)), 1e-300
                )
                per_system[str(system)] = float(r2_system)
            report[f"{name}_{ladder_name}"] = {
                "r2": float(r2),
                "per_system_r2": per_system,
                "best_per_system_r2": float(max(per_system.values())),
            }
            print(
                f"{name} {ladder_name}: R2 {r2:+.4f}  best per-system "
                f"{max(per_system.values()):+.4f}",
                flush=True,
            )

    bars = {
        "gauge_tolerance": 0.05,
        "slip_space_bar": 0.30,
        "latent_jump_bar": 0.10,
        "invariance_ratio_bar": 0.8,
        "gauge_stability": float(
            max(
                abs(report["l2_S2"]["r2"] - report["temporal_S2"]["r2"]),
                abs(report["l2_S3"]["r2"] - report["temporal_S3"]["r2"]),
            )
        )
        <= 0.05,
        "slip_space": report["l2_S2"]["r2"] >= 0.30,
        "latent_hardening": max(
            report["l2_S3"]["r2"] - report["l2_S2"]["r2"],
            report["l2_S5"]["r2"] - report["l2_S2"]["r2"],
        )
        >= 0.10,
        "invariance": report["l2_S2"]["r2"]
        >= 0.8 * report["l2_S2"]["best_per_system_r2"],
    }
    payload = {
        "schema_version": 1,
        "results": report,
        "bars": bars,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(bars, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
