# -*- coding: utf-8 -*-
"""
visual_fem_test.py – same plot as visual_test.py but for the fem_pixel solver.

Part 1: if the FEM frame files are missing (or FORCE_RERUN), solve the 10x10
        problem at N_FRAMES load fractions (same ramp Abaqus applies) and save
        fem_test_10x10_{s11,s22,s12,e11,e22,e12}.npy  with shape (n_frames, nx, ny).
Part 2: identical plotting to visual_test.py (DIC cropped to the FEA window,
        22_U_V_again_ dataset, capped at frame 40) with run_tag=fem_test_10x10.
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CODE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../fem/1_codes
ROOT     = os.path.dirname(os.path.dirname(CODE_DIR))       # test root
sys.path.insert(0, CODE_DIR)   # fem_pixel
sys.path.insert(0, ROOT)       # test_config
from test_config import FEM_OUT
FEM_DIR = FEM_OUT     # all FEM outputs live here (final_validation/fem_single)
os.makedirs(FEM_DIR, exist_ok=True)

# ═════════════════════ Part 1: generate FEM frame data ═══════════════════════
from test_config import (X_SIZE, Y_SIZE, ELEMENT_SIZE, SCALE_FACTOR, N_EXP,
                         FEM_TAG, JOB_NAME, NX as FEA_NX, NY as FEA_NY,
                         load_case5_inputs)

run_tag   = FEM_TAG
N_FRAMES  = 20          # load fractions 0..1 (plus zero frame), like ODB frames
HARDENING = 'tabular'
FORCE_RERUN = False

# s/e are per-ELEMENT (FEA_NX, FEA_NY); u1/u2 are per-NODE (FEA_NX+1, FEA_NY+1);
# time is the pseudo-time (0..1) of each frame, matching Abaqus's frameValue
# convention for a single step of period 1.0.
_vars = ("s11", "s22", "s12", "e11", "e22", "e12")
_nodal_vars = ("u1", "u2")
_frame_files = {v: os.path.join(FEM_DIR, f"{run_tag}_{v}.npy") for v in _vars}
_nodal_frame_files = {v: os.path.join(FEM_DIR, f"{run_tag}_{v}.npy") for v in _nodal_vars}
_time_file = os.path.join(FEM_DIR, f"{run_tag}_time.npy")

def _cache_valid():
    """Existing frame files must match the CURRENT window size. Also requires
    the nodal (u1/u2) and time frame files - added later than s/e, so any
    cache from before that addition is treated as stale and re-solved."""
    for p in _frame_files.values():
        if not os.path.isfile(p):
            return False
        if np.load(p, mmap_mode="r").shape[1:] != (FEA_NX, FEA_NY):
            print(f"[FEM frames] cached {os.path.basename(p)} has wrong shape "
                  f"for {FEA_NX}x{FEA_NY} window -> re-solving")
            return False
    for p in _nodal_frame_files.values():
        if not os.path.isfile(p):
            print(f"[FEM frames] {os.path.basename(p)} missing (older cache "
                  f"predates u1/u2 frame export) -> re-solving")
            return False
        if np.load(p, mmap_mode="r").shape[1:] != (FEA_NX + 1, FEA_NY + 1):
            return False
    if not os.path.isfile(_time_file):
        print(f"[FEM frames] {os.path.basename(_time_file)} missing (older "
              f"cache predates time export) -> re-solving")
        return False
    return True

if FORCE_RERUN or not _cache_valid():
    from fem_pixel import run_fem

    center_disp_x, center_disp_y, yield_cropped, K_cropped = load_case5_inputs()

    # ONE incremental solve; fields are snapshotted at each load fraction
    # (path-consistent plasticity, ~N_FRAMES x faster than re-solving per level)
    fracs = [k / N_FRAMES for k in range(1, N_FRAMES + 1)]
    r = run_fem(center_disp_x, center_disp_y,
                yield_cropped, K_cropped, N_EXP,
                X_SIZE, Y_SIZE, ELEMENT_SIZE, SCALE_FACTOR,
                E_mod=205000., nu=0.3,
                N_inc=N_FRAMES, snapshot_fractions=fracs,
                hardening=HARDENING, verbose=True)

    # save the FINAL-state fields too (same solve endpoint) so
    # fem_test_driver.py can reuse them instead of re-solving
    _final_tag = f"fem_{JOB_NAME}_{HARDENING}"
    for key in ("U", "S", "E", "PE", "PEEQ", "RF"):
        np.save(os.path.join(FEM_DIR, f"{_final_tag}_{key}.npy"), r[key])
    print(f"[FEM frames] final-state fields saved with prefix {_final_tag}")

    frames = {v: [np.zeros((FEA_NX, FEA_NY))] for v in _vars}   # frame 0 = zeros
    nodal_frames = {v: [np.zeros((FEA_NX + 1, FEA_NY + 1))] for v in _nodal_vars}
    time_frames = [0.0]
    for f in fracs:
        snap = r["frames"][f]
        frames["s11"].append(snap["S"][..., 0]); frames["s22"].append(snap["S"][..., 1])
        frames["s12"].append(snap["S"][..., 2])
        frames["e11"].append(snap["E"][..., 0]); frames["e22"].append(snap["E"][..., 1])
        frames["e12"].append(snap["E"][..., 2])   # engineering gamma12, same as ODB export
        nodal_frames["u1"].append(snap["U"][..., 0])
        nodal_frames["u2"].append(snap["U"][..., 1])
        time_frames.append(f)   # pseudo-time 0..1, same convention as Abaqus frameValue

    for v in _vars:
        np.save(_frame_files[v], np.array(frames[v]))
    for v in _nodal_vars:
        np.save(_nodal_frame_files[v], np.array(nodal_frames[v]))
    np.save(_time_file, np.array(time_frames))
    print(f"[FEM frames] saved {run_tag}_*.npy (incl. u1/u2/time) in {FEM_DIR}")
else:
    print(f"[FEM frames] found existing {run_tag}_*.npy (set FORCE_RERUN=True to redo)")

# ═════════════════════ Part 2: plotting (as visual_test.py) ══════════════════
nu = 0.30
E = 205000.0

sigma_y = 124.0
K_ludwik = 380
n_ludwik = 0.245

macro_path = r"C:\Users\adil.kilinc\Desktop\Thesis\3_data\stress_strain.npy"

# same dataset the FEA BCs are built from
dic_dir_U = r"C:\Users\adil.kilinc\Desktop\Thesis\3_data\22_U_V_again_"
dic_dir_V = r"C:\Users\adil.kilinc\Desktop\Thesis\3_data\22_U_V_again_"
dic_max_frame = 40   # FEM is loaded up to frame 40 -> stop the DIC curve there too
dic_baseline_n = 5

smooth_k = 7 # use odd values like 1,3,5,7...

# crop DIC field to the SAME central window the FEM was run on
CROP_TO_FEA_REGION = True

from test_config import crop_center as crop_center_arr, window_tag

if CROP_TO_FEA_REGION:
    smooth_k = 1                 # no smoothing on an 11x11 window (FEM has none)

# 'affine'  : least-squares plane fit over the window; slopes = mean strains
#             (avoids the noisy gradient-mean that made the DIC curve loop)
# 'gradient': original visual28 behaviour
STRAIN_MODE = "affine" if CROP_TO_FEA_REGION else "gradient"

num_dir = FEM_DIR
out_dir = FEM_DIR
os.makedirs(out_dir, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40

def list_dic_frames(folder_u, folder_v, max_k=None):
    def _idxs(folder, prefix):
        idx = []
        for fn in os.listdir(folder):
            m = re.match(rf"{re.escape(prefix)}_(\d+)\.npy$", fn)
            if m:
                idx.append(int(m.group(1)))
        return set(idx)
    iu = _idxs(folder_u, "U")
    iv = _idxs(folder_v, "V")
    inter = sorted(iu.intersection(iv))
    if max_k is not None:
        inter = [k for k in inter if k <= int(max_k)]
    return inter

def load_dic_uv(frame_idx):
    U = np.load(os.path.join(dic_dir_U, f"U_{frame_idx}.npy")).astype(float)
    V = np.load(os.path.join(dic_dir_V, f"V_{frame_idx}.npy")).astype(float)
    if CROP_TO_FEA_REGION:
        # RAW orientation, no transpose (see orientation_check.py)
        U = crop_center_arr(U, FEA_NX + 1, FEA_NY + 1)
        V = crop_center_arr(V, FEA_NX + 1, FEA_NY + 1)
        return U, V
    return U.T, V.T   # legacy whole-field mode keeps the visual28 transpose

def _boxfilter_sum(a, k):
    p = k // 2
    a = np.pad(a, ((p, p), (p, p)), mode="edge")
    s = a.cumsum(axis=0).cumsum(axis=1)
    s = np.pad(s, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]

def smooth2d_nanmean(a, k):
    k = int(k)
    if k <= 1:
        return a.astype(float, copy=True)
    a = a.astype(float, copy=False)
    v = np.isfinite(a).astype(float)
    a0 = np.where(np.isfinite(a), a, 0.0)
    num = _boxfilter_sum(a0, k)
    den = _boxfilter_sum(v, k)
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0.0)
    return out

def strain_from_disp(U, V, dx=1.0):
    dU_dy, dU_dx = np.gradient(U, dx, dx)
    dV_dy, dV_dx = np.gradient(V, dx, dx)
    exx = dV_dx
    eyy = dU_dy
    exy = 0.5 * (dU_dx + dV_dy)
    return exx, exy, eyy

def stress_vm(s11, s12, s22):
    return np.sqrt(s11**2 + s22**2 - s11*s22 + 3.0*s12**2)

def plane_stress_hooke(E, nu, exx, exy, eyy):
    pref = E / (1.0 - nu**2)
    sxx = pref * (exx + nu * eyy)
    syy = pref * (eyy + nu * exx)
    G = E / (2.0 * (1.0 + nu))
    sxy = 2.0 * G * exy
    return sxx, sxy, syy

def evm_classic_ezz0(exx, exy, eyy):
    return np.sqrt((2.0 / 3.0) * (exx**2 + eyy**2 + 2.0 * exy**2))

def evm_plane_stress_inv(exx, exy, eyy, nu):
    return np.sqrt(
        (4.0 / (9.0 * (1.0 - nu + nu**2))) *
        ((exx**2 + eyy**2 - exx * eyy) + 3.0 * (1.0 - nu)**2 * exy**2)
    )

def sigma_vm_from_evm_stitched_ludwik(evm, E, nu, sigma_y, K, n):
    G = E / (2.0 * (1.0 + nu))
    evm = np.asarray(evm, dtype=float)
    eps_y = sigma_y / (3.0 * G)
    sig_el = 3.0 * G * evm
    eps_p = np.maximum(evm - eps_y, 0.0)
    sig_pl = sigma_y + K * np.power(eps_p, n)
    return np.where(evm <= eps_y, sig_el, sig_pl)

def sigma_uniaxial_stitched_ludwik(eyy, E, sigma_y, K, n):
    eyy = np.asarray(eyy, dtype=float)
    eps_y = sigma_y / E
    sig_el = E * eyy
    eps_p = np.maximum(eyy - eps_y, 0.0)
    sig_pl = sigma_y + K * np.power(eps_p, n)
    return np.where(eyy <= eps_y, sig_el, sig_pl)

def mean_masked(a, mask):
    aa = np.where(mask, a, np.nan)
    return float(np.nanmean(aa))

def window_mean_strains(U, V, mask):
    """Window-mean (exx, exy, eyy) in the MESH frame of the transposed/cropped
    arrays (matches the FE BCs): u_x = V along axis0, u_y = U along axis1."""
    if STRAIN_MODE == "affine":
        fin = mask & np.isfinite(U) & np.isfinite(V)
        ii, jj = np.nonzero(fin)
        if ii.size < 6:
            return np.nan, np.nan, np.nan
        A = np.column_stack([np.ones(ii.size), ii.astype(float), jj.astype(float)])
        bu = np.linalg.lstsq(A, U[fin], rcond=None)[0]
        bv = np.linalg.lstsq(A, V[fin], rcond=None)[0]
        exx = bv[1]                    # d(u_x)/dx = dV/d(axis0)
        eyy = bu[2]                    # d(u_y)/dy = dU/d(axis1)
        exy = 0.5 * (bv[2] + bu[1])    # tensorial shear
        return exx, exy, eyy
    Us = smooth2d_nanmean(U, smooth_k)
    Vs = smooth2d_nanmean(V, smooth_k)
    exx, exy, eyy = strain_from_disp(Us, Vs, dx=1.0)
    return mean_masked(exx, mask), mean_masked(exy, mask), mean_masked(eyy, mask)

def load_numeric_curve(folder, run_tag, var, prefer_weighted=True):
    p = os.path.join(folder, f"{run_tag}_{var}.npy")
    if os.path.isfile(p):
        arr = np.load(p).astype(float)
        if arr.ndim == 1:
            return arr
        if arr.ndim >= 2:
            n = arr.shape[0]
            return np.asarray([np.nanmean(arr[k]) for k in range(n)], dtype=float)
    return None

exp_macro = np.load(macro_path).astype(float)
macro_eps_pct = exp_macro[0, :].astype(float)
macro_sig = exp_macro[1, :].astype(float)
mmin = min(macro_eps_pct.size, macro_sig.size)
macro_eps_pct = macro_eps_pct[:mmin]
macro_sig = macro_sig[:mmin]
mm = np.isfinite(macro_eps_pct) & np.isfinite(macro_sig)
macro_eps_pct = macro_eps_pct[mm]
macro_sig = macro_sig[mm]

dic_frames = list_dic_frames(dic_dir_U, dic_dir_V, max_k=dic_max_frame)
if len(dic_frames) == 0:
    raise RuntimeError("No matching DIC U_k/V_k files found")

nb = int(min(max(dic_baseline_n, 1), len(dic_frames)))
mask = None
for fr in dic_frames[:nb]:
    U, V = load_dic_uv(fr)
    m = np.isfinite(U) & np.isfinite(V)
    mask = m if mask is None else (mask & m)

mask = mask if mask is not None else np.ones_like(load_dic_uv(dic_frames[0])[0], dtype=bool)
mask = mask.copy()
if STRAIN_MODE != "affine":
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False

baseline_exx = []
baseline_exy = []
baseline_eyy = []

for fr in dic_frames[:nb]:
    U, V = load_dic_uv(fr)
    exx_m, exy_m, eyy_m = window_mean_strains(U, V, mask)
    baseline_exx.append(exx_m)
    baseline_exy.append(exy_m)
    baseline_eyy.append(eyy_m)

exx0_m = float(np.nanmean(np.asarray(baseline_exx, dtype=float)))
exy0_m = float(np.nanmean(np.asarray(baseline_exy, dtype=float)))
eyy0_m = float(np.nanmean(np.asarray(baseline_eyy, dtype=float)))

dic_x_eyy_pct = np.zeros(len(dic_frames), dtype=float)
dic_evm_classic = np.zeros(len(dic_frames), dtype=float)
dic_evm_ps = np.zeros(len(dic_frames), dtype=float)

dic_sig_hooke_vm_stitched = np.zeros(len(dic_frames), dtype=float)
dic_sig_3G_classic_stitched = np.zeros(len(dic_frames), dtype=float)
dic_sig_3G_ps_stitched = np.zeros(len(dic_frames), dtype=float)
dic_sig_uniax_stitched = np.zeros(len(dic_frames), dtype=float)

G = E / (2.0 * (1.0 + nu))
eps_y_vm = sigma_y / (3.0 * G)

for i, fr in enumerate(dic_frames):
    U, V = load_dic_uv(fr)
    exx_w, exy_w, eyy_w = window_mean_strains(U, V, mask)

    exx_m = exx_w - exx0_m
    exy_m = exy_w - exy0_m
    eyy_m = eyy_w - eyy0_m

    dic_x_eyy_pct[i] = 100.0 * eyy_m

    sxx_m, sxy_m, syy_m = plane_stress_hooke(E, nu, exx_m, exy_m, eyy_m)
    sig_vm_el = float(stress_vm(sxx_m, sxy_m, syy_m))

    evm_c = float(evm_classic_ezz0(exx_m, exy_m, eyy_m))
    evm_ps = float(evm_plane_stress_inv(exx_m, exy_m, eyy_m, nu))
    dic_evm_classic[i] = evm_c
    dic_evm_ps[i] = evm_ps

    sig_ep_c = float(sigma_vm_from_evm_stitched_ludwik(evm_c, E, nu, sigma_y, K_ludwik, n_ludwik))
    sig_ep_ps = float(sigma_vm_from_evm_stitched_ludwik(evm_ps, E, nu, sigma_y, K_ludwik, n_ludwik))

    dic_sig_hooke_vm_stitched[i] = sig_vm_el if evm_ps <= eps_y_vm else sig_ep_ps
    dic_sig_3G_classic_stitched[i] = sig_ep_c
    dic_sig_3G_ps_stitched[i] = sig_ep_ps
    dic_sig_uniax_stitched[i] = float(sigma_uniaxial_stitched_ludwik(eyy_m, E, sigma_y, K_ludwik, n_ludwik))

m_dic = np.isfinite(dic_x_eyy_pct)
dic_x_eyy_pct = dic_x_eyy_pct[m_dic]
dic_evm_classic = dic_evm_classic[m_dic]
dic_evm_ps = dic_evm_ps[m_dic]
dic_sig_hooke_vm_stitched = dic_sig_hooke_vm_stitched[m_dic]
dic_sig_3G_classic_stitched = dic_sig_3G_classic_stitched[m_dic]
dic_sig_3G_ps_stitched = dic_sig_3G_ps_stitched[m_dic]
dic_sig_uniax_stitched = dic_sig_uniax_stitched[m_dic]

num_e11 = load_numeric_curve(num_dir, run_tag, "e11")
num_e22 = load_numeric_curve(num_dir, run_tag, "e22")
num_e12 = load_numeric_curve(num_dir, run_tag, "e12")

num_s11 = load_numeric_curve(num_dir, run_tag, "s11")
num_s22 = load_numeric_curve(num_dir, run_tag, "s22")
num_s12 = load_numeric_curve(num_dir, run_tag, "s12")

have_num_strains = (num_e11 is not None) and (num_e22 is not None) and (num_e12 is not None)
have_num_stress = (num_s11 is not None) and (num_s22 is not None) and (num_s12 is not None)

num_x_eyy_pct = None
num_evm_classic = None
num_evm_ps = None

num_sig_fem_vm_stitched = None
num_sig_3G_classic_stitched = None
num_sig_3G_ps_stitched = None
num_sig_uniax_stitched = None

if have_num_strains:
    n = min(len(num_e11), len(num_e22), len(num_e12))
    exx = np.asarray(num_e11[:n], dtype=float)
    exy = np.asarray(num_e12[:n], dtype=float)
    eyy = np.asarray(num_e22[:n], dtype=float)

    num_x_eyy_pct = 100.0 * eyy
    num_evm_classic = evm_classic_ezz0(exx, exy, eyy)
    num_evm_ps = evm_plane_stress_inv(exx, exy, eyy, nu)

    num_sig_3G_classic_stitched = sigma_vm_from_evm_stitched_ludwik(num_evm_classic, E, nu, sigma_y, K_ludwik, n_ludwik)
    num_sig_3G_ps_stitched = sigma_vm_from_evm_stitched_ludwik(num_evm_ps, E, nu, sigma_y, K_ludwik, n_ludwik)
    num_sig_uniax_stitched = sigma_uniaxial_stitched_ludwik(eyy, E, sigma_y, K_ludwik, n_ludwik)

    m = np.isfinite(num_x_eyy_pct) & np.isfinite(num_evm_classic) & np.isfinite(num_evm_ps)
    num_x_eyy_pct = num_x_eyy_pct[m]
    num_evm_classic = num_evm_classic[m]
    num_evm_ps = num_evm_ps[m]
    num_sig_3G_classic_stitched = np.asarray(num_sig_3G_classic_stitched)[m]
    num_sig_3G_ps_stitched = np.asarray(num_sig_3G_ps_stitched)[m]
    num_sig_uniax_stitched = np.asarray(num_sig_uniax_stitched)[m]

if have_num_stress:
    if have_num_strains:
        n = min(len(num_s11), len(num_s22), len(num_s12), len(num_evm_ps))
        s11 = np.asarray(num_s11[:n], dtype=float)
        s12 = np.asarray(num_s12[:n], dtype=float)
        s22 = np.asarray(num_s22[:n], dtype=float)
        evm_ps_k = np.asarray(num_evm_ps[:n], dtype=float)
    else:
        n = min(len(num_s11), len(num_s22), len(num_s12))
        s11 = np.asarray(num_s11[:n], dtype=float)
        s12 = np.asarray(num_s12[:n], dtype=float)
        s22 = np.asarray(num_s22[:n], dtype=float)
        evm_ps_k = None

    sig_vm_el = stress_vm(s11, s12, s22)

    if evm_ps_k is not None:
        sig_vm_ep = sigma_vm_from_evm_stitched_ludwik(evm_ps_k, E, nu, sigma_y, K_ludwik, n_ludwik)
        num_sig_fem_vm_stitched = np.where(evm_ps_k <= eps_y_vm, sig_vm_el, sig_vm_ep)
    else:
        num_sig_fem_vm_stitched = sig_vm_el

    m = np.isfinite(num_sig_fem_vm_stitched)
    num_sig_fem_vm_stitched = np.asarray(num_sig_fem_vm_stitched)[m]

    if have_num_strains:
        num_x_eyy_pct_s = 100.0 * np.asarray(num_e22[:n], dtype=float)[m]
        num_evm_ps_s = np.asarray(num_evm_ps[:n], dtype=float)[m]
    else:
        num_x_eyy_pct_s = None
        num_evm_ps_s = None
else:
    num_x_eyy_pct_s = None
    num_evm_ps_s = None

fig = plt.figure(figsize=(12, 13))
ax = plt.gca()

ax.plot(
    macro_eps_pct,
    macro_sig,
    linewidth=3,
    linestyle="-",
    marker="o",
    markersize=7,
    color="#000000",
    label="Macro measured"
)

ax.plot(
    100.0 * dic_evm_ps[:-2],
    dic_sig_3G_ps_stitched[:-2],
    linewidth=3,
    linestyle=(0, (1, 2)),
    marker="v",
    markersize=7,
    color="#17becf",
    label="DIC reconstructed"
)

if (num_evm_ps is not None) and (num_sig_3G_ps_stitched is not None):
    nplot = min(len(num_evm_ps), len(num_sig_3G_ps_stitched))

    ax.plot(
        100.0 * num_evm_ps[:nplot],
        num_sig_3G_ps_stitched[:nplot],
        linewidth=3,
        linestyle=(0, (8, 3)),
        marker="<",
        markersize=7,
        color="#8c564b",
        label="FEM reconstructed"
    )

if (num_evm_ps_s is not None) and (num_sig_fem_vm_stitched is not None):
    nplot_s = min(len(num_evm_ps_s), len(num_sig_fem_vm_stitched))

    ax.plot(
        100.0 * num_evm_ps_s[:nplot_s],
        num_sig_fem_vm_stitched[:nplot_s],
        linewidth=3,
        linestyle=(0, (4, 2, 1, 2)),
        marker="s",
        markersize=7,
        color="#2ca02c",
        label="FEM stress"
    )

ax.set_xlabel(r"$\varepsilon_{vM}$ (%)")
ax.set_ylabel(r"$\sigma_{vM}$ (MPa)")

ax.grid(True, alpha=0.3)

ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.f"))

ax.legend(frameon=True, fontsize=30, loc="lower right",facecolor="white",
    edgecolor="black",
    framealpha=0.6)
suffix = "_dic_cropped" if CROP_TO_FEA_REGION else ""
fig.savefig(
    os.path.join(out_dir, f"S{sigma_y}_K{K_ludwik}_{n_ludwik}_{run_tag}_{window_tag()}{suffix}.pdf"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Saved to:", out_dir)
print("run_tag:", run_tag)
print("DIC frames used:", len(dic_frames), "from", dic_frames[0], "to", dic_frames[-1])
print("DIC baseline frames:", nb)
print("smooth_k:", smooth_k)
print("Numerical mode: fem_pixel frame-averaged curves (10x10 window)")
