# -*- coding: utf-8 -*-
"""
fem_test_driver.py – run fem_pixel on the SAME problem as case5_test.py
(size/preprocessing from test_config.py) and, if the Abaqus grid npy files
exist, print a field-by-field comparison against the final ODB frame.
"""
import os
import sys
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../fem/1_codes
ROOT     = os.path.dirname(os.path.dirname(CODE_DIR))       # test root
sys.path.insert(0, CODE_DIR)   # fem_pixel
sys.path.insert(0, ROOT)       # test_config
from fem_pixel import run_fem, _vm
from test_config import (X_SIZE, Y_SIZE, ELEMENT_SIZE, SCALE_FACTOR, N_EXP,
                         JOB_NAME, NX, NY, load_case5_inputs, FEM_OUT, AB_OUT)

FEM_DIR    = FEM_OUT   # all FEM outputs live here (final_validation/fem_single)
ABAQUS_DIR = AB_OUT    # Abaqus npy for comparison (final_validation/abaqus)
os.makedirs(FEM_DIR, exist_ok=True)

HARDENING = 'tabular'   # match the Abaqus *Plastic table
N_INC     = 40

tag = f"fem_{JOB_NAME}_{HARDENING}"

# ── reuse fields saved by visual_fem_test.py (same solve endpoint)? ──────────
# Enabled when run via run_all_fem.py (FEM_REUSE_FRAMES=1) and the saved
# final-state npys match the current window - avoids solving twice.
def _load_saved():
    d = {}
    for key in ("U", "S", "E", "PE", "PEEQ", "RF"):
        p = os.path.join(FEM_DIR, f"{tag}_{key}.npy")
        if not os.path.isfile(p):
            return None
        d[key] = np.load(p)
    if d["S"].shape[:2] != (NX, NY):
        return None
    return d

result = None
if os.environ.get("FEM_REUSE_FRAMES", "0") == "1":
    result = _load_saved()
    if result is not None:
        print(f"[driver] reusing final-state fields saved by visual_fem_test "
              f"(prefix {tag}) - no re-solve")

if result is None:
    # ── identical preprocessing to case5_test.py (shared loader) ─────────────
    center_disp_x, center_disp_y, yield_cropped, K_cropped = load_case5_inputs()

    # ── solve ─────────────────────────────────────────────────────────────────
    result = run_fem(center_disp_x, center_disp_y,
                     yield_cropped, K_cropped, N_EXP,
                     X_SIZE, Y_SIZE, ELEMENT_SIZE, SCALE_FACTOR,
                     E_mod=205000., nu=0.3,
                     N_inc=N_INC, hardening=HARDENING, verbose=True)

    for key in ("U", "S", "E", "PE", "PEEQ", "RF"):
        np.save(os.path.join(FEM_DIR, f"{tag}_{key}.npy"), result[key])
    print(f"FEM results saved with prefix {tag} in {FEM_DIR}")

# ── compare with Abaqus grid npy (final frame), if available ─────────────────
def _load(name):
    p = os.path.join(ABAQUS_DIR, f"{JOB_NAME}_{name}.npy")
    return np.load(p)[-1] if os.path.isfile(p) else None   # last frame

ab = {k: _load(k) for k in ("u1", "u2", "s11", "s22", "s12", "e11", "e22", "e12")}

if all(v is not None for v in ab.values()):
    fem = {
        "u1":  result["U"][..., 0],  "u2":  result["U"][..., 1],
        "s11": result["S"][..., 0], "s22": result["S"][..., 1], "s12": result["S"][..., 2],
        "e11": result["E"][..., 0], "e22": result["E"][..., 1],
        "e12": result["E"][..., 2],   # both are engineering shear gamma12
    }
    print("\n=== FEM vs Abaqus (final frame) ===")
    print(f"{'field':>6} {'max|diff|':>12} {'rms(ab)':>12} {'rel_max %':>10}")
    for k in ("u1", "u2", "s11", "s22", "s12", "e11", "e22", "e12"):
        a, f = ab[k], fem[k]
        if a.shape != f.shape:
            print(f"{k:>6}  SHAPE MISMATCH ab{a.shape} fem{f.shape} "
                  f"(stale Abaqus npy from another window size? rerun run_all_test.py)")
            continue
        d = np.abs(a - f)
        scale = np.sqrt(np.nanmean(a**2))
        rel = 100.0 * np.nanmax(d) / scale if scale > 0 else float("nan")
        print(f"{k:>6} {np.nanmax(d):12.4e} {scale:12.4e} {rel:10.3f}")
    if ab['s11'].shape == fem['s11'].shape:
        svm_ab  = np.sqrt(ab['s11']**2 + ab['s22']**2 - ab['s11']*ab['s22'] + 3*ab['s12']**2)
        svm_fem = _vm(result["S"].reshape(-1, 3)).reshape(svm_ab.shape)
        print(f"\n  svm  max|diff| = {np.nanmax(np.abs(svm_ab-svm_fem)):.4f} MPa "
              f"(abaqus max {np.nanmax(svm_ab):.2f} MPa)")
else:
    missing = [k for k, v in ab.items() if v is None]
    print(f"\nAbaqus npy files not found ({missing}) - run the Abaqus part first "
          f"(run_all_test.py) for the comparison.")
