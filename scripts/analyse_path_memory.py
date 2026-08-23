#!/usr/bin/env python3
"""Local path memory: is the missing information in the observable past?

Per `validation/path_memory_preregistration.md`: windows of the local
signed past of `(tau, Delta gamma)` as k-NN features, target
`|Delta gamma_n|`, leave-one-state-out, no tuning. W1-tau isolates the
loading path; W1 adds the response's own past; W2 and W4 extend both.
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
    parser.add_argument("--output", type=Path, default=OUT / "path_memory.json")
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    tau_full = fields["tau"]  # flat (samples, 12)
    gamma_full = fields["gamma"]
    states_full = fields["states"]
    tau_states = tau_full.reshape(N_STATES, -1, N_SYSTEMS)
    gamma_states = gamma_full.reshape(N_STATES, -1, N_SYSTEMS)

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
    point_idx = sample_idx % (tau_states.shape[1])
    state_num = sample_idx // tau_states.shape[1]

    def at_state(offset: int) -> np.ndarray:
        """Signed tau at state n + offset for the pooled samples (-inf if absent)."""
        n = state_num + offset
        valid = (n >= 0) & (n < N_STATES)
        result = np.full(len(sample_idx), np.nan)
        result[valid] = tau_states[n[valid], point_idx[valid], system_idx[valid]]
        return result

    def gamma_at_state(offset: int) -> np.ndarray:
        n = state_num + offset
        valid = (n >= 0) & (n < N_STATES)
        result = np.full(len(sample_idx), np.nan)
        result[valid] = gamma_states[n[valid], point_idx[valid], system_idx[valid]]
        return result

    tau_n = at_state(0)
    windows = {
        "W1-tau": np.stack([tau_n, at_state(-1), tau_n - at_state(-1)], axis=1),
        "W1": np.stack([tau_n, at_state(-1), tau_n - at_state(-1), gamma_at_state(-1)], axis=1),
        "W2": np.stack(
            [tau_n, at_state(-1), at_state(-2), gamma_at_state(-1), gamma_at_state(-2)],
            axis=1,
        ),
        "W4": np.stack(
            [
                tau_n,
                at_state(-1),
                at_state(-2),
                at_state(-3),
                at_state(-4),
                gamma_at_state(-1),
                gamma_at_state(-2),
                gamma_at_state(-3),
                gamma_at_state(-4),
            ],
            axis=1,
        ),
    }
    gamma_abs = np.abs(gamma_full[sample_idx, system_idx])
    tau_abs = np.abs(tau_n)

    # Baseline (|tau_n|, Gamma_{n-1}) in the same metric.
    cumulative = np.abs(gamma_states).cumsum(axis=0) - np.abs(gamma_states)
    gamma_history = cumulative[state_num, point_idx, system_idx]

    report: dict[str, dict] = {}
    for name, features in [
        ("baseline", np.stack([tau_abs, gamma_history], axis=1)),
        *windows.items(),
    ]:
        r2s = []
        predicted = 0
        for state in range(N_STATES):
            test = (state_num == state) & np.isfinite(features).all(axis=1)
            if test.sum() < 100:
                continue
            train = (state_num != state) & np.isfinite(features).all(axis=1)
            scaler = StandardScaler().fit(features[train])
            tree = cKDTree(scaler.transform(features[train]))
            _, idx = tree.query(scaler.transform(features[test]), k=K)
            pred = gamma_abs[train][idx].mean(axis=1)
            residual = gamma_abs[test] - pred
            r2s.append(
                1.0
                - float(np.sum(residual**2))
                / max(float(np.sum((gamma_abs[test] - np.mean(gamma_abs[train])) ** 2)), 1e-300)
            )
            predicted += int(test.sum())
        report[name] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "predicted_samples": predicted,
        }
        print(
            f"  {name:9s}: R2 {report[name]['r2_mean']:+.4f}  "
            f"(std {report[name]['r2_std']:.4f}, n {predicted})",
            flush=True,
        )

    best_window = max(windows, key=lambda w: report[w]["r2_mean"])
    best_r2 = report[best_window]["r2_mean"]
    baseline_r2 = report["baseline"]["r2_mean"]
    payload = {
        "schema_version": 1,
        "bars": {"closure": 0.30, "partial": 0.10},
        "results": report,
        "best_window": best_window,
        "jump_over_baseline": best_r2 - baseline_r2,
        "reading": (
            "closure candidate" if best_r2 >= 0.30 else "partial" if best_r2 >= 0.10 else "nothing"
        ),
        "loading_vs_response_gap": report["W1"]["r2_mean"] - report["W1-tau"]["r2_mean"],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
