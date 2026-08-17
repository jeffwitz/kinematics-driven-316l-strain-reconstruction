#!/usr/bin/env python3
"""The resistance coordinate: is the overstress the phase variable?

Per `validation/resistance_coordinate_preregistration.md`: the simplest
internal resistance `r = tau_ref (a + b Gamma~_self + c Gamma~_others)`,
the overstress `xi = |tau| - r`, and the same leave-one-state-out k-NN as
the ladder, with the three parameters tuned per fold on the training states
only (binned-variance score on a subsample, then the full-fold evaluation).
The baseline is the S2 rung in magnitude form: `(|tau|, Gamma~^alpha)`.
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
GRID = (0.0, 0.3, 1.0, 3.0)


def binned_r2(xi: np.ndarray, gamma_abs: np.ndarray, gamma_hist: np.ndarray) -> float:
    """Variance reduction of |gamma| by quantile bins of (xi, gamma_hist)."""

    bins_xi = np.quantile(xi, np.linspace(0, 1, 11))
    bins_h = np.quantile(gamma_hist, np.linspace(0, 1, 6))
    idx_xi = np.digitize(xi, bins_xi[1:-1])
    idx_h = np.digitize(gamma_hist, bins_h[1:-1])
    global_var = float(np.var(gamma_abs))
    if global_var <= 0:
        return 0.0
    within = 0.0
    total = 0
    for i in range(10):
        for j in range(5):
            mask = (idx_xi == i) & (idx_h == j)
            count = int(mask.sum())
            if count < 100:
                continue
            within += count * float(np.var(gamma_abs[mask]))
            total += count
    return 1.0 - within / max(total * global_var, 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=OUT / "fcc_slip_fields.npz")
    parser.add_argument("--output", type=Path, default=OUT / "resistance_coordinate.json")
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    tau_full = fields["tau"]
    gamma_full = fields["gamma"]
    cumulative_full = fields["gamma_cumulative"]
    states_full = fields["states"]

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
    current = np.abs(gamma_full[sample_idx, :])
    history = cumulative_full[sample_idx, :] - current
    history_self = history[np.arange(len(sample_idx)), system_idx]
    history_others = history.sum(axis=1) - history_self
    state_vec = states_full[sample_idx]

    fold_rows: list[dict] = []
    for state in range(N_STATES):
        test = state_vec == state
        train = ~test
        train_indices = np.where(train)[0]
        tune_mask = train_indices[
            rng.choice(len(train_indices), size=min(TUNE_SUBSAMPLE, len(train_indices)), replace=False)
        ]
        tau_ref = float(np.median(np.abs(tau[tune_mask])))
        gamma_ref = max(float(np.median(history_self[tune_mask])), 1e-300)
        h_self = history_self / gamma_ref
        h_others = history_others / gamma_ref
        tau_abs = np.abs(tau)

        best = (-1e300, None)
        for a in GRID:
            for b in GRID:
                for c in GRID:
                    xi = tau_abs - tau_ref * (a + b * h_self + c * h_others)
                    score = binned_r2(xi[tune_mask], gamma_abs[tune_mask], h_self[tune_mask])
                    if score > best[0]:
                        best = (score, (a, b, c))
        a, b, c = best[1]
        xi = tau_abs - tau_ref * (a + b * h_self + c * h_others)

        def knn_r2(features_train: np.ndarray, features_test: np.ndarray) -> float:
            scaler = StandardScaler().fit(features_train)
            tree = cKDTree(scaler.transform(features_train))
            _, idx = tree.query(scaler.transform(features_test), k=K)
            prediction = gamma_abs[train][idx].mean(axis=1)
            residual = gamma_abs[test] - prediction
            return 1.0 - float(np.sum(residual**2)) / max(
                float(np.sum((gamma_abs[test] - np.mean(gamma_abs[train])) ** 2)), 1e-300
            )

        baseline_features = np.stack([tau_abs, h_self], axis=1)
        resistance_features = np.stack([xi, h_self], axis=1)
        baseline_r2 = knn_r2(baseline_features[train], baseline_features[test])
        resistance_r2 = knn_r2(resistance_features[train], resistance_features[test])
        fold_rows.append(
            {
                "state": state,
                "tuned_a": a,
                "tuned_b": b,
                "tuned_c": c,
                "tune_score": best[0],
                "baseline_r2": baseline_r2,
                "resistance_r2": resistance_r2,
            }
        )
        print(
            f"  fold {state:2d}: (a,b,c)=({a},{b},{c})  baseline {baseline_r2:+.4f}  "
            f"resistance {resistance_r2:+.4f}",
            flush=True,
        )

    base = float(np.mean([row["baseline_r2"] for row in fold_rows]))
    res = float(np.mean([row["resistance_r2"] for row in fold_rows]))
    payload = {
        "schema_version": 1,
        "bars": {"overstress_bar": 0.30, "jump_bar": 0.10},
        "baseline_r2_mean": base,
        "resistance_r2_mean": res,
        "jump": res - base,
        "folds": fold_rows,
        "parameter_spread": {
            "a": float(np.std([row["tuned_a"] for row in fold_rows])),
            "b": float(np.std([row["tuned_b"] for row in fold_rows])),
            "c": float(np.std([row["tuned_c"] for row in fold_rows])),
        },
        "verdict": {
            "overstress_is_the_coordinate": res >= 0.30,
            "jump_is_real": res - base >= 0.10,
            "parameters_stable": float(np.std([row["tuned_b"] for row in fold_rows])) <= 0.3,
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "folds"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
