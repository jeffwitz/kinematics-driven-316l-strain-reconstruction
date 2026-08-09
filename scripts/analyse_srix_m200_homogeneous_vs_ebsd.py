"""Compare the controlled P43 M200 homogeneous and EBSD SRIX runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / "validation/_generated/performance"
DATA = ROOT / "data/processed/case_study"
EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")


def _load(stem: str) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    report = json.loads((PERF / f"{stem}.json").read_text())
    arrays = dict(np.load(PERF / f"{stem}.fields.npz"))
    return report, arrays


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a).ravel()
    bb = np.asarray(b).ravel()
    return float(np.corrcoef(aa, bb)[0, 1])


def _quantiles(a: np.ndarray) -> dict[str, float]:
    q = np.quantile(np.abs(a), [0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(("p50", "p90", "p95", "p99", "max"), q, strict=True)
    }


def _top_jaccard(a: np.ndarray, b: np.ndarray, fraction: float = 0.05) -> float:
    n = a.size
    count = max(1, int(np.ceil(fraction * n)))
    am = np.zeros(n, dtype=bool)
    bm = np.zeros(n, dtype=bool)
    am[np.argpartition(a.ravel(), -count)[-count:]] = True
    bm[np.argpartition(b.ravel(), -count)[-count:]] = True
    return float(np.count_nonzero(am & bm) / np.count_nonzero(am | bm))


def _timing(report: dict[str, object]) -> dict[str, object]:
    timings = report["timings"]
    assert isinstance(timings, dict)
    return {
        "elapsed_seconds": report["elapsed_seconds"],
        "newton_iterations": report["newton_iterations"],
        "iterations_per_increment": report["iterations_per_increment"],
        "final_residual": report["final_residual"],
        "material_seconds": timings["material_seconds"],
        "material_integration_seconds": timings["material_integration_seconds"],
        "krylov_seconds": timings["gmres_seconds"],
        "krylov_overhead_seconds": timings["krylov_overhead_seconds"],
        "jacobian_seconds": timings["jacobian_seconds"],
        "preconditioner_seconds": timings["preconditioner_seconds"],
        "substep_points": timings["material_native_substep_points"],
        "substep_cache_hits": timings["material_native_substep_cache_hits"],
        "substep_cache_misses": timings["material_native_substep_cache_misses"],
    }


def main() -> int:
    homogeneous, h = _load("srix_p43_m200_homogeneous_structural_fd")
    ebsd, e = _load("srix_p43_m200_ebsd_structural_fd")

    h_acc = h["accumulated_slip"]
    e_acc = e["accumulated_slip"]
    h_max = np.max(np.abs(h["plastic_slip"]), axis=(-2, -1))
    e_max = np.max(np.abs(e["plastic_slip"]), axis=(-2, -1))
    h_dom = np.argmax(np.abs(h["plastic_slip"]), axis=-1)
    e_dom = np.argmax(np.abs(e["plastic_slip"]), axis=-1)

    ux = np.load(DATA / "displacement_x_mm.npy", mmap_mode="r")
    uy = np.load(DATA / "displacement_y_mm.npy", mmap_mode="r")
    crop = ebsd["crop_nodes"]
    x0, x1, y0, y1 = (int(v) for v in crop)
    dic = np.stack((ux[x0 : x1 + 1, y0 : y1 + 1], uy[x0 : x1 + 1, y0 : y1 + 1]), axis=-1)
    interior = (slice(1, -1), slice(1, -1))
    displacement = {
        name: {
            "rmse_mm": float(np.sqrt(np.mean((arrays["displacement"] - dic) ** 2))),
            "relative_l2": _relative(arrays["displacement"], dic),
            "interior_rmse_mm": float(
                np.sqrt(np.mean((arrays["displacement"][interior] - dic[interior]) ** 2))
            ),
            "interior_relative_l2": _relative(
                arrays["displacement"][interior], dic[interior]
            ),
        }
        for name, arrays in (("homogeneous", h), ("ebsd", e))
    }

    report = {
        "status": "completed_srix_m200_homogeneous_vs_ebsd",
        "configuration": {
            "mesh": [200, 200],
            "increments": 8,
            "crop_nodes": list(crop),
            "behaviour": "fcc_forest_rubin_srix",
            "backend": "mfront-structural-plane-stress",
            "mfront_threads": 4,
            "krylov_blas_threads": 1,
            "fftw_threads": 1,
            "ebsd_file": str(EBSD),
            "boundary_sha256": ebsd["boundary_sha256"],
        },
        "runs": {"homogeneous": _timing(homogeneous), "ebsd": _timing(ebsd)},
        "field_comparison_ebsd_vs_homogeneous": {
            "displacement_relative_l2": _relative(e["displacement"], h["displacement"]),
            "stress_relative_l2": _relative(e["stress_in_plane_mpa"], h["stress_in_plane_mpa"]),
            "reaction_relative_l2": _relative(e["reaction_forces"], h["reaction_forces"]),
            "accumulated_slip_relative_l2": _relative(e_acc, h_acc),
            "plastic_slip_relative_l2": _relative(e["plastic_slip"], h["plastic_slip"]),
            "equivalent_plastic_slip_relative_l2": _relative(
                e["equivalent_plastic_slip"], h["equivalent_plastic_slip"]
            ),
            "stress_component_correlation": [
                _corr(e["stress_in_plane_mpa"][..., i], h["stress_in_plane_mpa"][..., i])
                for i in range(3)
            ],
            "accumulated_slip_correlation": _corr(e_acc, h_acc),
            "max_system_slip_correlation": _corr(e_max, h_max),
            "dominant_system_agreement": float(np.mean(e_dom == h_dom)),
            "top_5_percent_accumulated_slip_jaccard": _top_jaccard(e_acc, h_acc),
            "top_5_percent_max_system_slip_jaccard": _top_jaccard(e_max, h_max),
            "accumulated_slip_quantiles_homogeneous": _quantiles(h_acc),
            "accumulated_slip_quantiles_ebsd": _quantiles(e_acc),
            "max_system_slip_quantiles_homogeneous": _quantiles(h_max),
            "max_system_slip_quantiles_ebsd": _quantiles(e_max),
        },
        "displacement_vs_dic": displacement,
        "source_reports": {
            "homogeneous": "srix_p43_m200_homogeneous_structural_fd.json",
            "ebsd": "srix_p43_m200_ebsd_structural_fd.json",
        },
    }
    output = PERF / "srix_p43_m200_homogeneous_vs_ebsd.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
