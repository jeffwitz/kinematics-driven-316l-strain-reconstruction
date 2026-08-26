#!/usr/bin/env python3
"""Data-only full-field EBSD/DIC Schmid and registration audit (no FEM)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS
from fem_inhouse.core.crystal_orientation import rotations_from_euler_bunge_deg
from scripts.plot_p0043_raw_svd7_evm_maps import _evm

ROOT = Path(__file__).resolve().parents[1]
HDF5 = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
OUT = ROOT / "validation/reference_data/p0043_fullfield_schmid_registration_v1"
PIXEL_MM = 0.00184


def _schmid(angles: np.ndarray, axis: int) -> np.ndarray:
    q = rotations_from_euler_bunge_deg(angles)
    direction = np.zeros(3); direction[axis] = 1.0
    # Q maps sample vectors to crystal vectors.
    t = np.einsum("...ij,j->...i", q, direction)
    values = []
    for burgers, normal in SLIP_SYSTEMS:
        d = np.asarray(burgers, float); d /= np.linalg.norm(d)
        n = np.asarray(normal, float); n /= np.linalg.norm(n)
        values.append(np.abs(np.sum(t * d, axis=-1) * np.sum(t * n, axis=-1)))
    return np.max(np.stack(values), axis=0)


def _metrics(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    idx = np.flatnonzero(valid.ravel())
    if idx.size > 250_000:
        idx = rng.choice(idx, 250_000, replace=False)
    a, b = x.ravel()[idx], y.ravel()[idx]
    return {"n_pixels": int(valid.sum()), "pearson": float(np.corrcoef(a, b)[0, 1]),
            "spearman": float(spearmanr(a, b).statistic)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(43001)
    with h5py.File(HDF5, "r") as h:
        angles = np.stack([np.asarray(h[f"orientation/{n}"]) for n in ("phi1", "Phi", "phi2")], axis=-1)
        u = np.asarray(h["displacement/U"], dtype=float) * PIXEL_MM
        v = np.asarray(h["displacement/V"], dtype=float) * PIXEL_MM
        archived = np.asarray(h["schmid/max_schmid_factor"], dtype=float)
    exx = np.gradient(u, PIXEL_MM, axis=1)
    eyy = np.gradient(v, PIXEL_MM, axis=0)
    exy = 0.5 * (np.gradient(u, PIXEL_MM, axis=0) + np.gradient(v, PIXEL_MM, axis=1))
    evm = _evm(np.stack([u, v], axis=-1))
    maps: dict[str, np.ndarray] = {"epsilon_yy": eyy, "evm": evm}
    rows = []
    for pname, transformed in {
        "identity": angles, "rot180": np.rot90(angles, 2, axes=(0, 1)),
        "flip_x": angles[::-1], "flip_y": angles[:, ::-1],
    }.items():
        for axis, label in ((0, "EBSD_X"), (1, "EBSD_Y")):
            m = _schmid(transformed, axis)
            maps[f"{pname}_{label}"] = m
            item = {"spatial_hypothesis": pname, "traction_axis": label}
            item.update({f"epsilon_yy_{k}": val for k, val in _metrics(m, eyy, rng).items()})
            item.update({f"evm_{k}": val for k, val in _metrics(m, evm, rng).items()})
            threshold_m = np.nanpercentile(m, 90); threshold_e = np.nanpercentile(eyy, 90)
            valid = np.isfinite(m) & np.isfinite(eyy)
            high_m, high_e = (m >= threshold_m) & valid, (eyy >= threshold_e) & valid
            item["top10_iou_epsilon_yy"] = float(np.sum(high_m & high_e) / np.sum(high_m | high_e))
            item["top10_enrichment_epsilon_yy"] = float(np.mean(high_e[high_m]) / np.mean(high_e[valid]))
            rows.append(item)
    with (OUT / "schmid_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Null distribution on a fixed random pixel sample; preserves the marginal Schmid distribution.
    valid = np.isfinite(maps["identity_EBSD_Y"]) & np.isfinite(eyy)
    flat_m, flat_e = maps["identity_EBSD_Y"][valid], eyy[valid]
    sample = rng.choice(flat_m.size, min(100_000, flat_m.size), replace=False)
    observed = float(spearmanr(flat_m[sample], flat_e[sample]).statistic)
    null = np.empty(1000)
    for i in range(null.size): null[i] = spearmanr(rng.permutation(flat_m[sample]), flat_e[sample]).statistic
    null_percentile = float(np.mean(null <= observed)); pvalue = float((1 + np.sum(null >= observed)) / (null.size + 1))
    np.savetxt(OUT / "null_permutations.csv", null, delimiter=",", header="spearman_null", comments="")
    np.savez_compressed(OUT / "maps.npz", **maps)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    panels = [(eyy, "DIC epsilon_yy"), (evm, "DIC EVM"), (maps["identity_EBSD_X"], "Schmid EBSD X"),
              (maps["identity_EBSD_Y"], "Schmid EBSD Y"), (maps["identity_EBSD_Y"] - maps["identity_EBSD_X"], "Schmid Y-X"),
              (maps["identity_EBSD_Y"], "Schmid + registration")]
    for ax, (field, title) in zip(axes.flat, panels):
        im = ax.imshow(field, origin="lower", cmap="viridis"); ax.set_title(title); fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(OUT / "fullfield_schmid_overview.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(null, bins=40, alpha=.8); ax.axvline(observed, color="r", label=f"observed={observed:.3f}"); ax.legend(); ax.set_xlabel("Spearman null"); fig.savefig(OUT / "null_distribution.png", dpi=180); plt.close(fig)
    report = {"schema_version": 1, "hdf5": str(HDF5), "shape": list(angles.shape[:2]), "pixel_size_mm": PIXEL_MM,
              "dic_displacement_units": "mm (HDF5 px converted using 1.84 um/px)", "results": rows,
              "null": {"metric": "Spearman(identity, EBSD_Y, epsilon_yy)", "n": 1000, "observed": observed,
                       "percentile": null_percentile, "empirical_pvalue": pvalue},
              "statuses": {"fullfield_analysis_completed": True, "ebsd_global_geometry_known": False,
                           "ebsd_axis_metadata_found": False, "registration_proven": False,
                           "fem_used": False, "material_identification_used": False}}
    (OUT / "provenance_report.json").write_text(json.dumps({"hdf5": str(HDF5), "shape": list(angles.shape[:2]), "pixel_size_mm": PIXEL_MM, "orientation_attrs": "read from HDF5", "displacement_attrs": "read from HDF5"}, indent=2) + "\n")
    (OUT / "final_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "shape": angles.shape[:2], "observed_spearman": observed, "null_pvalue": pvalue}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
