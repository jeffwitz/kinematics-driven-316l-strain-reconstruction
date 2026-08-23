#!/usr/bin/env python3
"""Path-memory closure check: magnitude-respecting windows, three predictors.

Per `validation/path_memory_closure_check_preregistration.md`: the same
windows in magnitude form with sign-reversal indicators, evaluated by k-NN
(the reference), linear ridge and histogram gradient boosting — if none
reaches 0.10, the closure is real and the discovery path closes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/_generated/shared_tensor_generator"
N_SYSTEMS = 12
N_STATES = 20
SUBSAMPLE_PER_STATE = 20000
K = 50


def r2_score(y_true: np.ndarray, y_pred: np.ndarray, baseline: float) -> float:
    residual = y_true - y_pred
    return 1.0 - float(np.sum(residual**2)) / max(float(np.sum((y_true - baseline) ** 2)), 1e-300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, default=OUT / "fcc_slip_fields.npz")
    parser.add_argument("--output", type=Path, default=OUT / "path_memory_closure_check.json")
    arguments = parser.parse_args()

    fields = np.load(arguments.fields, allow_pickle=False)
    tau_full = fields["tau"]
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
    point_idx = sample_idx % tau_states.shape[1]
    state_num = sample_idx // tau_states.shape[1]

    def at(offset: int) -> np.ndarray:
        n = state_num + offset
        valid = (n >= 0) & (n < N_STATES)
        result = np.full(len(sample_idx), np.nan)
        result[valid] = tau_states[n[valid], point_idx[valid], system_idx[valid]]
        return result

    def gamma_at(offset: int) -> np.ndarray:
        n = state_num + offset
        valid = (n >= 0) & (n < N_STATES)
        result = np.full(len(sample_idx), np.nan)
        result[valid] = gamma_states[n[valid], point_idx[valid], system_idx[valid]]
        return result

    t0, tm1 = at(0), at(-1)
    windows = {
        "baseline": None,
        "W1m": np.stack(
            [np.abs(t0), np.abs(tm1), np.abs(t0 - tm1), (t0 * tm1 < 0).astype(float)], axis=1
        ),
        "W1mg": np.stack(
            [
                np.abs(t0),
                np.abs(tm1),
                np.abs(t0 - tm1),
                (t0 * tm1 < 0).astype(float),
                np.abs(gamma_at(-1)),
            ],
            axis=1,
        ),
        "W2m": np.stack(
            [
                np.abs(t0),
                np.abs(tm1),
                np.abs(at(-2)),
                (t0 * tm1 < 0).astype(float),
                (tm1 * at(-2) < 0).astype(float),
                np.abs(gamma_at(-1)),
                np.abs(gamma_at(-2)),
            ],
            axis=1,
        ),
    }
    cumulative = np.abs(gamma_states).cumsum(axis=0) - np.abs(gamma_states)
    baseline_features = np.stack([np.abs(t0), cumulative[state_num, point_idx, system_idx]], axis=1)
    windows["baseline"] = baseline_features

    target = np.abs(gamma_full[sample_idx, system_idx])
    report: dict[str, dict] = {}
    for name, features in windows.items():
        finite = np.isfinite(features).all(axis=1)
        for predictor in ("knn", "ridge", "boost"):
            r2s = []
            for state in range(N_STATES):
                test = (state_num == state) & finite
                train = (state_num != state) & finite
                if test.sum() < 100:
                    continue
                scaler = StandardScaler().fit(features[train])
                x_train = scaler.transform(features[train])
                x_test = scaler.transform(features[test])
                baseline = float(np.mean(target[train]))
                if predictor == "knn":
                    tree = cKDTree(x_train)
                    _, idx = tree.query(x_test, k=K)
                    pred = target[train][idx].mean(axis=1)
                elif predictor == "ridge":
                    pred = Ridge(alpha=1.0).fit(x_train, target[train]).predict(x_test)
                else:
                    pred = (
                        HistGradientBoostingRegressor(
                            max_iter=100, early_stopping=False, random_state=20260817
                        )
                        .fit(x_train, target[train])
                        .predict(x_test)
                    )
                r2s.append(r2_score(target[test], pred, baseline))
            report[f"{name}_{predictor}"] = {
                "r2_mean": float(np.mean(r2s)),
                "r2_std": float(np.std(r2s)),
            }
            print(
                f"  {name:9s} {predictor:6s}: R2 {report[f'{name}_{predictor}']['r2_mean']:+.4f}",
                flush=True,
            )

    best = max(report, key=lambda k: report[k]["r2_mean"])
    payload = {
        "schema_version": 1,
        "bars": {"reopen": 0.10},
        "results": report,
        "best": best,
        "best_r2": report[best]["r2_mean"],
        "reading": "reopen" if report[best]["r2_mean"] >= 0.10 else "closed",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
