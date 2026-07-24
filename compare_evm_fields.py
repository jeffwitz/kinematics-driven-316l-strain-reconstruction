"""
compare_evm_fields.py - side-by-side εvM fields: DIC | Abaqus | FEM (10x10 test)

DIC    : U_40/V_40 from 22_U_V_again_, cropped to the same central window,
         element(cell)-centred strains from nodal displacement differences
         (same discretisation as the FEA), baseline (frames 1..5) subtracted.
Abaqus : test_10x10_{e11,e22,e12}.npy (last ODB frame, from odb_process_test).
FEM    : fem_test_10x10_{e11,e22,e12}.npy (last frame, from visual_fem_test),
         falls back to fem_test_10x10_tabular_E.npy from fem_test_driver.

All three use the plane-stress εvM of visual28 and share one colour scale.
Output: evm_fields_dic_abaqus_fem.pdf/.png in this folder.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

import sys

sys.path.insert(0, TEST_DIR)
from test_config import (
    AB_OUT,
    FEM_OUT,
    FEM_TAG,
    FINAL_VALIDATION_DIR,
    JOB_NAME,
    PX_TO_MM,
    crop_center,
    window_tag,
)
from test_config import (
    DIC_DIR as dic_dir,
)
from test_config import (
    DIC_FINAL_FRAME as dic_final_frame,
)
from test_config import (
    NX as FEA_NX,
)
from test_config import (
    NY as FEA_NY,
)

ABAQUS_DIR = AB_OUT
FEM_DIR = FEM_OUT
OUT_DIR = FINAL_VALIDATION_DIR  # comparison figures land at the top level

nu = 0.30
dic_baseline_n = 5

# Gaussian smoothing of the DIC eps_vM field before plotting (px).
# 0 = OFF: raw per-element strains, no filtering of any kind.
DIC_EVM_SMOOTH_SIGMA = 0.0


def evm_ps(exx, exy, eyy):
    return np.sqrt(
        (4.0 / (9.0 * (1.0 - nu + nu**2)))
        * ((exx**2 + eyy**2 - exx * eyy) + 3.0 * (1.0 - nu) ** 2 * exy**2)
    )


# ── DIC field ────────────────────────────────────────────────────────────────
def _pad_to(a, n_rows, n_cols):
    # edge-pad when the nodal grid needs one more row/col than the pixel
    # arrays have (full-field windows) - same as test_config.load_case5_inputs
    pr = max(0, n_rows - a.shape[0])
    pc = max(0, n_cols - a.shape[1])
    if pr or pc:
        a = np.pad(a, ((0, pr), (0, pc)), mode="edge")
    return a


def load_uv(k):
    # RAW orientation, no transpose (validated by orientation_check.py):
    # axis0=rows=transverse=mesh x, axis1=cols=loading=mesh y
    U = np.load(os.path.join(dic_dir, f"U_{k}.npy")).astype(float)
    V = np.load(os.path.join(dic_dir, f"V_{k}.npy")).astype(float)
    U = _pad_to(U, FEA_NX + 1, FEA_NY + 1)
    V = _pad_to(V, FEA_NX + 1, FEA_NY + 1)
    U = crop_center(U, FEA_NX + 1, FEA_NY + 1)
    V = crop_center(V, FEA_NX + 1, FEA_NY + 1)
    return U, V


def cell_strains(U, V):
    """DIC strains exactly per the pipeline convention:
    1) np.gradient of U, V (pixel units, spacing 1 px -> dimensionless strain)
    2) average each strain COMPONENT from nodes to element centres
    3) eps_vM computed afterwards, per element.
    Mesh convention: axis0=x, axis1=y, V=u_x, U=u_y. Therefore
    exx=dV/d0, eyy=dU/d1, exy=0.5*(dV/d1+dU/d0) (tensorial shear)."""
    dU0, dU1 = np.gradient(U)  # spacing = 1 px
    dV0, dV1 = np.gradient(V)

    # MESH frame of the RAW cropped arrays (matches the FE BCs):
    #   u_x = V along axis0 (transverse),  u_y = U along axis1 (loading)
    exx_n = dV0  # d(u_x)/dx
    eyy_n = dU1  # d(u_y)/dy
    exy_n = 0.5 * (dV1 + dU0)  # tensorial shear

    def n2c(a):  # node -> cell-centre average
        return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])

    return n2c(exx_n), n2c(exy_n), n2c(eyy_n)


# baseline strain fields (mean over first frames), subtracted per element
base = None
n_base = 0
for k in range(1, dic_baseline_n + 1):
    fu = os.path.join(dic_dir, f"U_{k}.npy")
    fv = os.path.join(dic_dir, f"V_{k}.npy")
    if not (os.path.isfile(fu) and os.path.isfile(fv)):
        continue
    U, V = load_uv(k)
    s = np.stack(cell_strains(U, V))
    base = s if base is None else base + s
    n_base += 1
base = base / n_base if n_base else 0.0

U, V = load_uv(dic_final_frame)
exx_d, exy_d, eyy_d = cell_strains(U, V)
if n_base:
    exx_d = exx_d - base[0]
    exy_d = exy_d - base[1]
    eyy_d = eyy_d - base[2]
evm_dic = evm_ps(exx_d, exy_d, eyy_d)
if DIC_EVM_SMOOTH_SIGMA > 0:
    from scipy.ndimage import gaussian_filter

    finite = np.isfinite(evm_dic)
    filled = np.where(finite, evm_dic, 0.0)
    w = gaussian_filter(finite.astype(float), DIC_EVM_SMOOTH_SIGMA)
    evm_dic = np.where(
        w > 1e-6, gaussian_filter(filled, DIC_EVM_SMOOTH_SIGMA) / np.maximum(w, 1e-6), np.nan
    )


# ── Abaqus field (last frame) ────────────────────────────────────────────────
def last_frame(fname, folder):
    p = os.path.join(folder, fname)
    if not os.path.isfile(p):
        return None
    a = np.load(p)
    return a[-1] if a.ndim == 3 else a


ab_e11 = last_frame(f"{JOB_NAME}_e11.npy", ABAQUS_DIR)
ab_e22 = last_frame(f"{JOB_NAME}_e22.npy", ABAQUS_DIR)
ab_e12 = last_frame(f"{JOB_NAME}_e12.npy", ABAQUS_DIR)
evm_ab = None
if all(a is not None for a in (ab_e11, ab_e22, ab_e12)):
    # Abaqus E12 is engineering gamma12 -> /2 for tensorial shear,
    # consistent with the DIC strains above
    evm_ab = evm_ps(ab_e11, ab_e12 / 2.0, ab_e22)

# ── FEM field (last frame) ───────────────────────────────────────────────────
fm_e11 = last_frame(f"{FEM_TAG}_e11.npy", FEM_DIR)
fm_e22 = last_frame(f"{FEM_TAG}_e22.npy", FEM_DIR)
fm_e12 = last_frame(f"{FEM_TAG}_e12.npy", FEM_DIR)
evm_fem = None
if all(a is not None for a in (fm_e11, fm_e22, fm_e12)):
    # FEM e12 is engineering gamma12 -> /2 for tensorial shear
    evm_fem = evm_ps(fm_e11, fm_e12 / 2.0, fm_e22)
else:
    p = os.path.join(FEM_DIR, f"fem_{JOB_NAME}_tabular_E.npy")
    if os.path.isfile(p):
        Efld = np.load(p)  # (nx, ny, 3): e11, e22, gamma12
        evm_fem = evm_ps(Efld[..., 0], Efld[..., 2] / 2.0, Efld[..., 1])

# ── plot ─────────────────────────────────────────────────────────────────────
fields = [("DIC", evm_dic), ("Abaqus", evm_ab), ("FEM", evm_fem)]
fields = [(t, f) for t, f in fields if f is not None]
if len(fields) == 0:
    raise RuntimeError("No fields available - run the pipeline first")

# drop solver fields whose npy shape does not match the CURRENT window
# (stale outputs from a previous window size)
_good = []
for t, f in fields:
    if t != "DIC" and f.shape != (FEA_NX, FEA_NY):
        print(
            f"[warn] {t} field shape {f.shape} != current window "
            f"({FEA_NX},{FEA_NY}) - stale npy from another size, skipping panel. "
            f"Rerun that chain with the current size."
        )
    else:
        _good.append((t, f))
fields = _good
if len(fields) == 0:
    raise RuntimeError("No fields match the current window size")

vmax = max(np.nanpercentile(f, 99.5) for _, f in fields) * 100.0
vmin = 0.0

# figure sized so each panel is the SAME physical width (11in) and uses the
# SAME font sizes as partition_visualize.py / partition_blended_visualize.py
# (those use figsize=(33,10.5) for 3 panels = 11in/panel) - matching panel
# width, not just point size, is what makes the text look the same size
# across scripts with different panel counts.
fig, axes = plt.subplots(1, len(fields), figsize=(11 * len(fields), 10.5), constrained_layout=True)
if len(fields) == 1:
    axes = [axes]

for ax, (title, f) in zip(axes, fields, strict=True):
    im = ax.imshow(
        100.0 * f.T, cmap="jet", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    ax.set_title(title, fontsize=22)
    ax.set_xlabel("x (elem)", fontsize=16)
    ax.set_ylabel("y (elem)", fontsize=16)
    ax.tick_params(labelsize=13)
    mean_pct = 100.0 * float(np.nanmean(f))
    ax.text(
        0.02,
        0.98,
        f"mean = {mean_pct:.4f} %",
        transform=ax.transAxes,
        fontsize=15,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

cbar = fig.colorbar(im, ax=axes, shrink=0.85)
cbar.set_label(r"$\varepsilon_{vM}$ (%)", fontsize=18)
cbar.ax.tick_params(labelsize=13)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT_DIR, f"evm_fields_dic_abaqus_fem_{window_tag()}.{ext}"), dpi=250)
plt.close(fig)

# consistency check: window-mean strain COMPONENTS are fixed by the boundary
# displacements, so DIC and FE means must agree closely here even though the
# interior fields (and mean eps_vM) differ. If these disagree, something IS wrong.
print("\nwindow-mean strain components (should match between DIC and FE):")
print(f"  {'src':7s} {'e_xx':>12s} {'e_yy':>12s} {'e_xy(tens)':>12s}")
print(f"  {'DIC':7s} {np.nanmean(exx_d):12.3e} {np.nanmean(eyy_d):12.3e} {np.nanmean(exy_d):12.3e}")
if all(a is not None for a in (ab_e11, ab_e22, ab_e12)):
    print(
        f"  {'Abaqus':7s} {np.nanmean(ab_e11):12.3e} {np.nanmean(ab_e22):12.3e} {np.nanmean(ab_e12) / 2:12.3e}"
    )
if all(a is not None for a in (fm_e11, fm_e22, fm_e12)):
    print(
        f"  {'FEM':7s} {np.nanmean(fm_e11):12.3e} {np.nanmean(fm_e22):12.3e} {np.nanmean(fm_e12) / 2:12.3e}"
    )
print("  (note: DIC includes the baseline-frame subtraction; FE BCs are raw frame-40,")
print("   so a small offset equal to the baseline pre-strain is expected. Axis naming:")
print("   FE e11 pairs with the DIC component along the same mesh axis.)")

print("\nFields plotted:", [t for t, _ in fields])
for t, f in fields:
    print(f"  {t:7s} evm: mean={100 * np.nanmean(f):.4f}%  max={100 * np.nanmax(f):.4f}%")
if evm_ab is not None and evm_fem is not None and evm_ab.shape == evm_fem.shape:
    d = np.abs(evm_ab - evm_fem)
    print(f"  |Abaqus-FEM| evm: max={100 * np.nanmax(d):.5f}%  mean={100 * np.nanmean(d):.5f}%")
print(f"Saved: evm_fields_dic_abaqus_fem_{window_tag()}.pdf/.png in {OUT_DIR}")

# ═════════════════════ u2 (axial displacement) comparison ════════════════════
# DIC | Abaqus | FEM nodal u2 in mm + |sim-DIC| panel. Boundary nodes are
# prescribed BCs (must match ~exactly); interior = prediction vs measurement.
u2_dic = (
    crop_center(
        _pad_to(
            np.load(os.path.join(dic_dir, f"U_{dic_final_frame}.npy")).astype(float),
            FEA_NX + 1,
            FEA_NY + 1,
        ),
        FEA_NX + 1,
        FEA_NY + 1,
    )
    * PX_TO_MM
)

u2_ab = None
p = os.path.join(ABAQUS_DIR, f"{JOB_NAME}_u2.npy")
if os.path.isfile(p):
    a = np.load(p)
    a = a[-1] if a.ndim == 3 else a
    if a.shape == (FEA_NX + 1, FEA_NY + 1):
        u2_ab = a
    else:
        print(
            f"[warn] Abaqus u2 shape {a.shape} != window ({FEA_NX + 1},{FEA_NY + 1}) - skipping panel."
        )

u2_fem = None
p = os.path.join(FEM_DIR, f"fem_{JOB_NAME}_tabular_U.npy")
if os.path.isfile(p):
    a = np.load(p)[..., 1]
    if a.shape == (FEA_NX + 1, FEA_NY + 1):
        u2_fem = a
    else:
        print(f"[warn] FEM U shape {a.shape} != window - skipping panel.")

u2_fields = [
    (t, f) for t, f in (("DIC", u2_dic), ("Abaqus", u2_ab), ("FEM", u2_fem)) if f is not None
]


def _u2_report(name, a, b):
    d = np.abs(a - b)
    inner = d[1:-1, 1:-1]
    bnd = d.copy()
    bnd[1:-1, 1:-1] = np.nan
    print(f"\n{name}:")
    print(
        f"  overall : max|diff| = {np.nanmax(d):.4e} mm   rms = {np.sqrt(np.nanmean(d**2)):.4e} mm"
    )
    print(f"  boundary: max|diff| = {np.nanmax(bnd):.4e} mm  (should be ~0: BCs are prescribed)")
    print(
        f"  interior: max|diff| = {np.nanmax(inner):.4e} mm   rms = {np.sqrt(np.nanmean(inner**2)):.4e} mm"
    )


if u2_ab is not None and u2_fem is not None:
    _u2_report("u2 Abaqus vs FEM (solver check)", u2_ab, u2_fem)
_sim = u2_fem if u2_fem is not None else u2_ab
if _sim is not None:
    _u2_report(
        f"u2 {'FEM' if u2_fem is not None else 'Abaqus'} vs DIC (prediction vs measurement)",
        _sim,
        u2_dic,
    )

if len(u2_fields) > 0:
    vmin2 = min(np.nanmin(f) for _, f in u2_fields)
    vmax2 = max(np.nanmax(f) for _, f in u2_fields)
    n_p = len(u2_fields) + (1 if _sim is not None else 0)
    fig, axes = plt.subplots(1, n_p, figsize=(11 * n_p, 10.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, f) in zip(axes, u2_fields, strict=True):
        im = ax.imshow(
            f.T, cmap="jet", origin="lower", vmin=vmin2, vmax=vmax2, interpolation="nearest"
        )
        ax.set_title(f"{title}  u2", fontsize=22)
        ax.set_xlabel("x (node)", fontsize=16)
        ax.set_ylabel("y (node)", fontsize=16)
        ax.tick_params(labelsize=13)
        ax.text(
            0.02,
            0.98,
            f"mean = {np.nanmean(f):.4e} mm",
            transform=ax.transAxes,
            fontsize=15,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
    cbar = fig.colorbar(im, ax=list(axes[: len(u2_fields)]), shrink=0.8)
    cbar.set_label(r"$u_2$ (mm)", fontsize=18)
    cbar.ax.tick_params(labelsize=13)
    if _sim is not None:
        ax = axes[len(u2_fields)]
        im2 = ax.imshow(
            np.abs(_sim - u2_dic).T, cmap="magma", origin="lower", interpolation="nearest"
        )
        ax.set_title(f"|{'FEM' if u2_fem is not None else 'Abaqus'} - DIC|", fontsize=22)
        ax.set_xlabel("x (node)", fontsize=16)
        ax.set_ylabel("y (node)", fontsize=16)
        ax.tick_params(labelsize=13)
        cbar2 = fig.colorbar(im2, ax=[ax], shrink=0.8)
        cbar2.set_label("|diff| (mm)", fontsize=18)
        cbar2.ax.tick_params(labelsize=13)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(OUT_DIR, f"u2_fields_dic_abaqus_fem_{window_tag()}.{ext}"), dpi=250
        )
    plt.close(fig)
    print(f"Saved: u2_fields_dic_abaqus_fem_{window_tag()}.pdf/.png in {OUT_DIR}")
