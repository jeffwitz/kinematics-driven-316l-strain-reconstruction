#!/usr/bin/env python3
"""Close the non-mechanical parts of the P43 EBSD/DIC registration audit."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fem_inhouse.core.fcc_interaction_matrix import SLIP_SYSTEMS

ROOT = Path(__file__).resolve().parents[1]
EBSD = Path("/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5")
OUT = ROOT / "validation/reference_data/p0043_ebsd_mapping_audit_v1"
CROP = (1610, 1630, 1075, 1095)
PIXEL_SIZE_UM = 1.84
MIN_SCHMID = 1.0 / np.sqrt(6.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _angles_and_archived() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x0, x1, y0, y1 = CROP
    with h5py.File(EBSD, "r") as handle:
        angles = np.stack(
            [
                np.asarray(handle[f"orientation/{name}"][x0:x1, y0:y1], dtype=float)
                for name in ("phi1", "Phi", "phi2")
            ],
            axis=-1,
        )
        archived = np.asarray(
            handle["schmid/max_schmid_factor"][x0:x1, y0:y1], dtype=float
        )
        attrs = {
            "orientation_attrs": {
                name: {key: value.tolist() if hasattr(value, "tolist") else value
                       for key, value in handle[f"orientation/{name}"].attrs.items()}
                for name in ("phi1", "Phi", "phi2")
            },
            "schmid_attrs": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in handle["schmid/max_schmid_factor"].attrs.items()
            },
            "hdf5_root_attrs": {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in handle.attrs.items()
            },
        }
    return angles, archived, attrs


def _schmid_for_axis(angles: np.ndarray, axis_index: int) -> np.ndarray:
    phi1, capital, phi2 = np.deg2rad(np.moveaxis(angles, -1, 0))
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(capital), np.sin(capital)
    c3, s3 = np.cos(phi2), np.sin(phi2)
    rotation = np.empty((*angles.shape[:2], 3, 3), dtype=float)
    rotation[..., 0, 0] = c1 * c3 - s1 * s3 * c2
    rotation[..., 0, 1] = s1 * c3 + c1 * s3 * c2
    rotation[..., 0, 2] = s3 * s2
    rotation[..., 1, 0] = -c1 * s3 - s1 * c3 * c2
    rotation[..., 1, 1] = -s1 * s3 + c1 * c3 * c2
    rotation[..., 1, 2] = c3 * s2
    rotation[..., 2, 0] = s1 * s2
    rotation[..., 2, 1] = -c1 * s2
    rotation[..., 2, 2] = c2
    loading_in_crystal = rotation[..., :, axis_index]
    values = []
    for burgers, normal in SLIP_SYSTEMS:
        direction = np.asarray(burgers, dtype=float)
        direction /= np.linalg.norm(direction)
        plane_normal = np.asarray(normal, dtype=float)
        plane_normal /= np.linalg.norm(plane_normal)
        values.append(np.abs(np.sum(loading_in_crystal * direction, axis=-1)
                             * np.sum(loading_in_crystal * plane_normal, axis=-1)))
    return np.max(np.stack(values, axis=0), axis=0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    angles, archived, hdf5_attrs = _angles_and_archived()
    candidates = np.stack([_schmid_for_axis(angles, axis) for axis in range(3)])
    valid = np.isfinite(archived) & (archived >= MIN_SCHMID - 1e-6) & (archived <= 0.5 + 1e-6)
    correlations = []
    errors = []
    for _axis, candidate in enumerate(candidates):
        correlations.append(float(np.corrcoef(candidate[valid], archived[valid])[0, 1]))
        errors.append(float(np.linalg.norm(candidate[valid] - archived[valid])
                           / np.linalg.norm(archived[valid])))
    selected = int(np.argmax(np.asarray(correlations)))
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    images = [archived, candidates[selected], candidates[selected] - archived, candidates[0]]
    titles = ["HDF5 Schmid", f"reconstruit axe {selected}", "différence", "reconstruit axe 0"]
    for axis, image, title in zip(axes, images, titles, strict=True):
        plot = axis.imshow(image.T, origin="lower", aspect="equal", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
        figure.colorbar(plot, ax=axis, shrink=0.8)
    figure.suptitle("P43 M20 — contrôle Schmid EBSD indépendant de la FEM")
    figure.savefig(OUT / "schmid_map_registered.png", dpi=220)
    plt.close(figure)
    np.savez_compressed(
        OUT / "schmid_map_registered.npz",
        angles=angles,
        archived=archived,
        candidates=candidates,
        valid=valid,
    )
    provenance = {
        "schema_version": 1,
        "ebsd_file": str(EBSD),
        "ebsd_sha256": _sha256(EBSD),
        "pixel_size_um": PIXEL_SIZE_UM,
        "p43_crop_indices": list(CROP),
        "dic_crop_contract": (
            "DIC canonical crop uses the same numerical bounds in the maintained workflow"
        ),
        "ebsd_crop_provenance": (
            "CP_dataset.h5 attributes state rows 400:4000 and cols 1211:4311; "
            "P43 subcrop method is not recorded"
        ),
        "axis_status": {
            "dic_canonical_axis_identity_proven": True,
            "dic_axis_direction_proven": True,
            "ebsd_axis_identity_proven": False,
            "ebsd_axis_direction_proven": False,
        },
        "crop_status": {
            "same_array_indices_declared": True,
            "physical_origin_registration_proven": False,
            "registration_transform": "unknown",
        },
        "sample_frame_status": {
            "ebsd_to_fem_rotation_proven_from_acquisition_metadata": False,
            "internal_schmid_axis_diagnostic": True,
            "selected_internal_axis": selected,
            "axis_correlations_with_archived_schmid": correlations,
            "axis_relative_errors_with_archived_schmid": errors,
        },
        "mapping_status": {
            "structured_mesh_element_order": "F",
            "explicit_f_mapping_passed": True,
            "spectral_solver_order": "C by documented contract; not silently changed",
        },
        "schmid_diagnostic": {
            "valid_pixels": int(valid.sum()),
            "invalid_pixels": int((~valid).sum()),
            "selected_axis": selected,
            "selected_correlation": correlations[selected],
            "selected_relative_error": errors[selected],
            "interpretation": (
                "internal consistency check only; does not prove "
                "EBSD-to-DIC sample-frame registration"
            ),
        },
        "hdf5_attrs": hdf5_attrs,
        "statuses": {
            "ebsd_element_order_correct": True,
            "ebsd_dic_axis_identity_proven": False,
            "ebsd_dic_axis_direction_proven": False,
            "ebsd_dic_origin_registration_proven": False,
            "ebsd_dic_crop_registration_proven": False,
            "ebsd_fem_sample_frame_proven": False,
            "historical_c_results_physically_valid": False,
            "historical_c_results_retained_as_control": True,
        },
    }
    (OUT / "registration_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n"
    )
    (OUT / "axis_registration.md").write_text(
        "# P43 axis registration\n\n"
        "DIC canonical axis 0 is transverse x and axis 1 is tensile y, as documented. "
        "The CP_dataset orientation datasets record shape, crop and pixel size but no "
        "physical EBSD axis direction. Therefore EBSD-to-DIC axis identity and sign "
        "remain unproven; no flip/transpose is selected from FEM fit.\n"
    )
    (OUT / "sample_frame_registration.md").write_text(
        "# P43 sample-frame registration\n\n"
        "The internal Schmid diagnostic selects one Euler-derived sample axis as the "
        "best match to the archived Schmid dataset, but this is an internal HDF5 "
        "consistency check, not acquisition metadata. The fixed EBSD-to-FEM rotation "
        "is therefore not proven and no unregistered +/-90 degree correction is applied.\n"
    )
    audit_report_path = OUT / "report.json"
    audit_report = json.loads(audit_report_path.read_text()) if audit_report_path.is_file() else {}
    audit_report.update({
        "schmid_audit": {
            "status": "completed_internal_consistency_check",
            "selected_axis": selected,
            "axis_correlations": correlations,
            "axis_relative_errors": errors,
            "interpretation": "does not prove EBSD-to-DIC sample-frame registration",
        },
        "ebsd_dic_axis_registration_proven": False,
        "ebsd_dic_axis_direction_proven": False,
        "ebsd_crystal_sample_frame_registration_proven": False,
        "p43_historical_cp_results_spatially_trustworthy": False,
        "mechanics_run": True,
        "mechanics_run_kind": "classical M20 prior C/F comparison archived separately",
    })
    audit_report_path.write_text(json.dumps(audit_report, indent=2) + "\n")
    f_report_path = ROOT / (
        "validation/reference_data/"
        "p0043_experimental_raw_svd7_f_provisional_v1/report.json"
    )
    f_evm_path = ROOT / (
        "validation/reference_data/"
        "p0043_experimental_raw_svd7_f_provisional_v1/p0043_raw_svd7_evm_metrics.json"
    )
    if f_report_path.is_file():
        f_report = json.loads(f_report_path.read_text())
        final_dir = ROOT / "validation/reference_data/raw_femu_f_v1"
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f_report_path, final_dir / "report.json")
        if f_evm_path.is_file():
            shutil.copy2(f_evm_path, final_dir / "localization_metrics.json")
        source_dir = f_report_path.parent
        for filename in ("p0043_raw_svd7_evm_historical_overview.png",
                         "p0043_raw_svd7_evm_maps.png",
                         "p0043_raw_svd7_evm_rms_by_state.png"):
            source = source_dir / filename
            if source.is_file():
                destination = (
                    final_dir / "field_overview.png"
                    if filename.endswith("overview.png")
                    else final_dir / filename
                )
                shutil.copy2(source, destination)
        with (final_dir / "history.csv").open("w", newline="") as stream:
            records = f_report.get("evaluation_history", [])
            writer = csv.DictWriter(stream, fieldnames=("evaluation", "rms_mm", "z", "parameters"))
            writer.writeheader()
            for record in records:
                writer.writerow({
                    key: (
                        json.dumps(record.get(key), sort_keys=True)
                        if key in {"z", "parameters"}
                        else record.get(key)
                    )
                    for key in writer.fieldnames
                })
        prior = f_report.get("evaluation_history", [{}])[0].get("parameters", {})
        final = f_report.get("final_parameters", {})
        with (final_dir / "parameter_comparison.csv").open("w", newline="") as stream:
            names = sorted(set(prior) | set(final))
            writer = csv.DictWriter(stream, fieldnames=("parameter", "prior", "final"))
            writer.writeheader()
            writer.writerows({"parameter": name, "prior": prior.get(name), "final": final.get(name)}
                             for name in names)
        f_converged = bool(
            f_report.get("optimizer", {}).get("success", False)
            and f_report.get("optimizer", {}).get("status") == 1
            and float(f_report.get("final_timing", {}).get("verification_residual", 1.0)) < 1e-9
        )
        registration = json.loads((OUT / "registration_provenance.json").read_text())
        registration["raw_femu_f"] = {
            "completed": True,
            "minimum_converged": f_converged,
            "report": str(f_report_path),
            "prior_rms_mm": f_report.get("prior_rms_mm"),
            "final_rms_mm": f_report.get("final_rms_mm"),
            "relative_rms_reduction": f_report.get("relative_rms_reduction"),
            "gauss_newton_predicted_relative_reduction": f_report.get(
                "gauss_newton_predicted_relative_reduction"
            ),
            "optimizer": f_report.get("optimizer"),
            "final_verification_residual": f_report.get("final_timing", {}).get(
                "verification_residual"
            ),
        }
        registration["statuses"].update({
            "raw_m20_f_identification_completed": True,
            "raw_m20_f_minimum_converged": f_converged,
        })
        (OUT / "registration_provenance.json").write_text(
            json.dumps(registration, indent=2, sort_keys=True, default=str) + "\n"
        )
    print(json.dumps({
        "selected_axis": selected,
        "axis_correlations": correlations,
        "axis_relative_errors": errors,
        "output": str(OUT),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
