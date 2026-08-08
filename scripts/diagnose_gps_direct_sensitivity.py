"""Direct sensitivity of the GPS local system: C_sens vs shadow Schur vs DSL.

CdC step 1-2: the GPS local system is F(x, eps_a) = 0 with x = (deel[6],
dg[12]) in R^18 and eps_a the imposed in-plane strain. The consistent tangent
MFront returns is built from the DSL's assumption on dF/ddeto; the shadow
Schur is exact but pays a full 3D integration. This script computes the
EXACT derivative of the system MFront actually solves, by re-implementing
the 18 equations of Fcc316LForestRubinSrixGps.mfront in numpy -- including
the declared Jacobian blocks -- and validates the implementation by checking
F(x*) = 0 at the converged state of the checkpoint.

At the converged state:

    A = dF/dx            (18x18, the local Newton matrix, re-implemented)
    B = dF/deps_a        (18x3,  analytically: only the three kinematic
                          in-plane rows of feel carry -I; everything else
                          is zero -- deto enters nowhere else)
    A X = -B             (three right-hand sides, one LU factorisation)
    C_sens = (dsig_a/dx) X

Comparison on the responsible points (96, 95, 59) at the checkpoint
increment 6:

    |C_sens - C_shadow| / |C_shadow|   -- target <= 1e-10
    |C_sens - C_DSL_projected| / |C_DSL|  -- reproduces the ~3e-3 gap

Usage:

    .venv/bin/python scripts/diagnose_gps_direct_sensitivity.py \
        --output validation/_generated/performance/gps_direct_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CROP_20X20 = (1610, 1630, 1075, 1095)
EBSD_ORIENTATION_H5 = "/home/jeff/CNRS/Theses/Adil/essais/9_numerical/CP_dataset.h5"
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
GPS = "mfront-native-generalised-plane-stress"
REFERENCE = "mfront-3d-condensed-plane-stress"
TOP_POINTS = (96, 95, 59)
_PLANE = (0, 1, 3)
_TRANSVERSE = (2, 4, 5)
_SQRT_TWO = np.sqrt(2.0)

#: The twelve octahedral Schmid tensors in Kelvin storage, copied VERBATIM
#: from the generated Fcc316LForestRubinSrixGpsSlipSystems.ixx (the compiled
#: truth; do not recompute from Miller indices).
_MUS: tuple[tuple[float, ...], ...] = (
    (0.0, 0.408248290463863035, -0.408248290463863035, 0.288675134594812895, -0.288675134594812895, 0.0),
    (0.408248290463863035, 0.0, -0.408248290463863035, 0.288675134594812895, 0.0, -0.288675134594812895),
    (0.408248290463863035, -0.408248290463863035, 0.0, 0.0, 0.288675134594812895, -0.288675134594812895),
    (0.0, 0.408248290463863035, -0.408248290463863035, 0.288675134594812895, 0.288675134594812895, 0.0),
    (0.408248290463863035, 0.0, -0.408248290463863035, 0.288675134594812895, 0.0, 0.288675134594812895),
    (0.408248290463863035, -0.408248290463863035, -0.0, 0.0, -0.288675134594812895, 0.288675134594812895),
    (0.0, -0.408248290463863035, 0.408248290463863035, 0.288675134594812895, -0.288675134594812895, 0.0),
    (0.408248290463863035, -0.0, -0.408248290463863035, -0.288675134594812895, 0.0, -0.288675134594812895),
    (0.408248290463863035, -0.408248290463863035, -0.0, 0.0, -0.288675134594812895, -0.288675134594812895),
    (0.0, -0.408248290463863035, 0.408248290463863035, 0.288675134594812895, 0.288675134594812895, 0.0),
    (0.408248290463863035, -0.0, -0.408248290463863035, -0.288675134594812895, 0.0, 0.288675134594812895),
    (0.408248290463863035, -0.408248290463863035, 0.0, 0.0, 0.288675134594812895, 0.288675134594812895),
)


def _gps_rotate(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """R s R^T in Kelvin storage, exactly the gpsRotate function of the law.

    `r` is the 3x3 rotation matrix whose rows are the nine arguments of the
    MFront private function. The formula is transcribed from the .mfront: the
    diagonal-to-shear mixing carries sqrt(2), not 1/sqrt(2).
    """

    sq2 = _SQRT_TWO
    s0, s1, s2, s3, s4, s5 = s
    r00, r01, r02 = r[0]
    r10, r11, r12 = r[1]
    r20, r21, r22 = r[2]
    out = np.empty(6, dtype=float)
    out[0] = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2 + sq2 * (
        r00 * r01 * s3 + r00 * r02 * s4 + r01 * r02 * s5
    )
    out[1] = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2 + sq2 * (
        r10 * r11 * s3 + r10 * r12 * s4 + r11 * r12 * s5
    )
    out[2] = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2 + sq2 * (
        r20 * r21 * s3 + r20 * r22 * s4 + r21 * r22 * s5
    )
    out[3] = sq2 * (r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2) + (
        r00 * r11 + r01 * r10
    ) * s3 + (r00 * r12 + r02 * r10) * s4 + (r01 * r12 + r02 * r11) * s5
    out[4] = sq2 * (r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2) + (
        r00 * r21 + r01 * r20
    ) * s3 + (r00 * r22 + r02 * r20) * s4 + (r01 * r22 + r02 * r21) * s5
    out[5] = sq2 * (r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2) + (
        r10 * r21 + r11 * r20
    ) * s3 + (r10 * r22 + r12 * r20) * s4 + (r11 * r22 + r12 * r21) * s5
    return out


def _kelvin_dot(a: np.ndarray, b: np.ndarray) -> float:
    """Double contraction in Kelvin storage = plain component sum."""

    return float(np.dot(a, b))


def _deviator(s: np.ndarray) -> np.ndarray:
    trace = s[0] + s[1] + s[2]
    out = s.copy()
    out[0] -= trace / 3.0
    out[1] -= trace / 3.0
    out[2] -= trace / 3.0
    return out


class GpsLocalSystem:
    """The 18-unknown GPS system, re-implemented exactly from the .mfront.

    x = (deel[6], dg[12]); F = (feel[6], fg[12]). All quantities Kelvin,
    stress in MPa. The residual and the Jacobian are transcribed from
    Fcc316LForestRubinSrixGps.mfront, including the declared derivative
    blocks, with theta = 1.
    """

    def __init__(
        self,
        *,
        q: np.ndarray,
        sig0: np.ndarray,
        p: np.ndarray,
        a: np.ndarray,
        d: np.ndarray,
        m: np.ndarray,
        r: float,
        tau0: float,
        q_hard: float,
        b: float,
        c_hard: float,
        d_hard: float,
        deqeps: float = 1.0e-14,
    ) -> None:
        self.q = np.asarray(q, dtype=float)
        # The .mfront passes the nine Q components to gpsRotate in the order
        # (Q11, Q12, Q13, Q21, ...) which its gpsRotate reads as ROW-MAJOR, so
        # the rotation actually applied to the residual and to sig is Q^T
        # (measured: gpsRotate(sig, Q^T) reproduces the bridge's global stress
        # to 1e-13, Q does not).
        self.rotation = self.q.T
        self.sig0 = np.asarray(sig0, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.m = np.asarray(m, dtype=float)
        self.r = float(r)
        self.tau0 = float(tau0)
        self.q_hard = float(q_hard)
        self.b = float(b)
        self.c_hard = float(c_hard)
        self.d_hard = float(d_hard)
        self.deqeps = float(deqeps)
        self.mus = np.asarray(_MUS, dtype=float)
        self.gps_modulus = 122000.0
        # Constant rotated objects of @InitLocalVariables.
        q_t = self.q.T
        self.gps_rot_t = np.stack(
            [_gps_rotate(np.eye(6)[j], q_t) for j in range(6)]
        )
        self.gps_dc = np.stack(
            [_gps_rotate(self.d @ np.eye(6)[j], q_t) for j in range(6)]
        )
        self.gps_rot_m = np.stack(
            [_gps_rotate(mus_i, q_t) for mus_i in self.mus]
        )

    def stress(self, deel: np.ndarray) -> np.ndarray:
        """sig = sig0 + D:deel, the StandardElasticity brick with theta = 1."""

        return self.sig0 + self.d @ deel

    def residual(self, x: np.ndarray, deto: np.ndarray | None = None) -> np.ndarray:
        """F(x): the 18 equations, transcribed from the @Integrator."""

        deel = x[:6]
        dg = x[6:]
        theta = 1.0
        sig = self.stress(deel)

        de = _deviator(deel)
        for i in range(12):
            de = de + dg[i] * self.mus[i]
        deq = float(np.sqrt(2.0 * _kelvin_dot(de, de) / 3.0))
        feel = np.zeros(6, dtype=float)
        fg = dg.copy()
        gps_plastic = np.zeros(6, dtype=float)
        if deq >= self.deqeps:
            flow_slope = deq / self.r
            ndeq = (2.0 / (3.0 * deq)) * de
            exp_bp = np.exp(-self.b * (self.p + theta * np.abs(dg)))
            for i in range(12):
                tau = _kelvin_dot(sig, self.mus[i])
                r_hard = self.tau0
                for j in range(12):
                    r_hard += self.q_hard * self.m[i, j] * (1.0 - exp_bp[j])
                da = (dg[i] - self.d_hard * self.a[i] * abs(dg[i])) / (
                    1.0 + theta * self.d_hard * abs(dg[i])
                )
                x_back = self.c_hard * (self.a[i] + theta * da)
                overstress = abs(tau - x_back) - r_hard
                f = max(overstress, 0.0)
                dflow = flow_slope if overstress > 0.0 else 0.0
                sgn = 1.0 if tau - x_back > 0.0 else -1.0
                gps_plastic = gps_plastic + dg[i] * self.mus[i]
                fg[i] -= flow_slope * f * sgn
        # Residual in the global frame: kinematic rows 0,1,3; closure rows 2,4,5.
        gps_residual = _gps_rotate(deel + gps_plastic, self.rotation) - deto
        gps_sigma_global = _gps_rotate(sig, self.rotation)
        feel[0] = gps_residual[0]
        feel[1] = gps_residual[1]
        feel[3] = gps_residual[3]
        feel[2] = gps_sigma_global[2] / self.gps_modulus
        feel[4] = gps_sigma_global[4] / self.gps_modulus
        feel[5] = gps_sigma_global[5] / self.gps_modulus
        return np.concatenate((feel, fg))

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """A = dF/dx (18x18), the declared blocks of the @Integrator."""

        deel = x[:6]
        dg = x[6:]
        theta = 1.0
        sig = self.stress(deel)
        dfeel_ddeel = np.zeros((6, 6), dtype=float)
        dfeel_ddg = np.zeros((6, 12), dtype=float)
        dfg_ddeel = np.zeros((12, 6), dtype=float)
        dfg_ddg = np.eye(12, dtype=float)

        de = _deviator(deel)
        for i in range(12):
            de = de + dg[i] * self.mus[i]
        deq = float(np.sqrt(2.0 * _kelvin_dot(de, de) / 3.0))
        if deq >= self.deqeps:
            flow_slope = deq / self.r
            ndeq = (2.0 / (3.0 * deq)) * de
            exp_bp = np.exp(-self.b * (self.p + theta * np.abs(dg)))
            for i in range(12):
                tau = _kelvin_dot(sig, self.mus[i])
                r_hard = self.tau0
                for j in range(12):
                    r_hard += self.q_hard * self.m[i, j] * (1.0 - exp_bp[j])
                da = (dg[i] - self.d_hard * self.a[i] * abs(dg[i])) / (
                    1.0 + theta * self.d_hard * abs(dg[i])
                )
                x_back = self.c_hard * (self.a[i] + theta * da)
                overstress = abs(tau - x_back) - r_hard
                f = max(overstress, 0.0)
                dflow = flow_slope if overstress > 0.0 else 0.0
                sgn = 1.0 if tau - x_back > 0.0 else -1.0
                damplitude = f * sgn / self.r
                # dfeel_ddeel rows: 0,1,3 from gpsRotT; 2,4,5 from gpsDc.
                for j in range(6):
                    dfeel_ddeel[0, j] = self.gps_rot_t[j][0]
                    dfeel_ddeel[1, j] = self.gps_rot_t[j][1]
                    dfeel_ddeel[3, j] = self.gps_rot_t[j][3]
                    dfeel_ddeel[2, j] = theta * self.gps_dc[j][2] / self.gps_modulus
                    dfeel_ddeel[4, j] = theta * self.gps_dc[j][4] / self.gps_modulus
                    dfeel_ddeel[5, j] = theta * self.gps_dc[j][5] / self.gps_modulus
                # dfeel_ddg column i: gpsRotM[i] with transverse rows zeroed.
                col = self.gps_rot_m[i].copy()
                col[2] = 0.0
                col[4] = 0.0
                col[5] = 0.0
                dfeel_ddg[:, i] = col
                # dfg_ddeel row i.
                mus_d = np.array(
                    [self.mus[i] @ self.d[:, k] for k in range(6)]
                )
                dfg_ddeel[i, :] = -dflow * theta * mus_d - damplitude * ndeq
                # dfg_ddg diagonal.
                sgn_gi = 1.0 if dg[i] > 0.0 else -1.0
                dda_ddg = (1.0 - self.d_hard * self.a[i] * sgn_gi) / (
                    1.0 + theta * self.d_hard * abs(dg[i])
                ) ** 2
                dfg_ddg[i, i] += dflow * self.c_hard * theta * dda_ddg
                for j in range(12):
                    sgn_gj = 1.0 if dg[j] > 0.0 else -1.0
                    dr = (
                        self.q_hard
                        * self.m[i, j]
                        * theta
                        * self.b
                        * exp_bp[j]
                        * sgn_gj
                    )
                    dfg_ddg[i, j] += (
                        dflow * dr * sgn - damplitude * _kelvin_dot(ndeq, self.mus[j])
                    )
        # The kinematic rows of dfeel_ddeel do not depend on the slips and are
        # filled above only inside the plastic branch; in the elastic branch
        # they are zero -- transcribed as declared (rows 0,1,3 are constants).
        top = np.concatenate((dfeel_ddeel, dfeel_ddg), axis=1)
        bottom = np.concatenate((dfg_ddeel, dfg_ddg), axis=1)
        return np.concatenate((top, bottom), axis=0)

    def stress_a_sensitivity(self, x: np.ndarray) -> np.ndarray:
        """dsig_a/dx (3x18, Kelvin): the in-plane rows of d(gpsRotate(sig))/dx.

        sig = sig0 + D:deel depends only on deel, so dsig_a/ddg = 0 and
        dsig_a/ddeel_j = (gpsRotate(D e_j)) in-plane rows = gpsDc[j] rows.
        """

        out = np.zeros((3, 18), dtype=float)
        for j in range(6):
            rotated = self.gps_dc[j]
            out[0, j] = rotated[0]
            out[1, j] = rotated[1]
            out[2, j] = rotated[3]
        return out

    def tangent_to_engineering(self, c_kelvin: np.ndarray) -> np.ndarray:
        """Kelvin 3x3 in-plane tangent to engineering, the bridge's scaling."""

        stress_scale = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
        strain_scale = np.array([1.0, 1.0, 1.0 / _SQRT_TWO])
        return c_kelvin * stress_scale[:, None] * strain_scale[None, :]


def _extract_elastic_stiffness(library: str) -> np.ndarray:
    """D (6x6 Kelvin) from the elastic tangent of the raw law at a tiny step.

    The raw 3D law's consistent tangent at a sub-threshold strain is exactly
    the brick's D. Evaluated with the identity rotation so D stays in the
    material frame.
    """

    from fem_inhouse.core.mfront import MFront3DMaterialPointBatch
    from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS

    batch = MFront3DMaterialPointBatch(
        library,
        behaviour_spec=MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix"),
        point_count=1,
        behaviour_name="Fcc316LForestRubinSrix",
    )
    strain = np.zeros((1, 6), dtype=float)
    strain[0, 0] = 1.0e-6
    trial = batch.evaluate(strain, time_increment=1.0, collect_observables=False)
    tangent = np.asarray(trial.consistent_tangent_kelvin_mpa)[0]
    batch.revert()
    return tangent


def _load_case(arguments: argparse.Namespace):
    from scripts.benchmark_tri2_j2_krylov import _load_case as load

    mesh = arguments.crop_nodes[1] - arguments.crop_nodes[0]
    grid, _, yield_stress, coefficient, boundary = load(mesh, arguments.crop_nodes)
    return grid, yield_stress, coefficient, boundary


def _run_backend(backend, arguments, grid, yield_stress, coefficient, boundary):
    from scripts.diagnose_gps_tangent_localisation import _run_backend as run

    return run(backend, arguments, grid, yield_stress, coefficient, boundary)


def _build_material(backend, arguments, grid, yield_stress, coefficient):
    from scripts.diagnose_gps_tangent_localisation import _build_material as build

    return build(backend, arguments, grid, yield_stress, coefficient)


def _checkpoint_calls(recording, increment: int):
    from scripts.diagnose_gps_tangent_localisation import _checkpoint_calls as calls

    return calls(recording, increment)


def _native(obj: object) -> object:
    return getattr(obj, "_bridge", None) or obj


def _isv_offsets(material: object) -> dict[str, np.ndarray]:
    from fem_inhouse.core.mfront import _declared_internal_slices

    native = _native(material)
    return _declared_internal_slices(
        native._mgis,
        native._behaviour,
        native._mgis.Hypothesis.Tridimensional,
        native._specification,
    )


def _converged_increment(
    material: object,
    snapshot: object,
    strain: np.ndarray,
    dt: float,
    point: int,
) -> dict[str, np.ndarray]:
    """Evaluate the GPS from the snapshot and read the converged (deel, dg).

    Returns the local unknowns at the converged state of this evaluation, for
    ONE point, plus the committed stress needed by the re-implementation.
    """

    native = _native(material)
    manager = native._manager
    material.restore_state(snapshot)
    trial = material.evaluate(
        strain.reshape(-1, 3),
        time_increment=dt,
        consistent_tangent=True,
    )
    offsets = _isv_offsets(material)
    s0_isv = np.asarray(manager.s0.internal_state_variables)[point]
    s1_isv = np.asarray(manager.s1.internal_state_variables)[point]
    eel0 = s0_isv[offsets["elastic_strain"]]
    eel1 = s1_isv[offsets["elastic_strain"]]
    g0 = s0_isv[offsets["plastic_slip"]]
    g1 = s1_isv[offsets["plastic_slip"]]
    p = s0_isv[offsets["equivalent_plastic_slip"]]
    a = s0_isv[offsets["back_strain"]]
    sig0 = np.asarray(manager.s0.thermodynamic_forces)[point].copy()
    deel = eel1 - eel0
    dg = g1 - g0
    # The DSL's deto is the INCREMENT of the imposed gradient: the bridge
    # writes the total into s1.gradients, the brick forms s1 - s0. Measured:
    # gpsRotate(deel + plastic, Q^T) = s1.gradients - s0.gradients to 1e-20.
    deto_increment = (
        np.asarray(manager.s1.gradients)[point] - np.asarray(manager.s0.gradients)[point]
    ).copy()
    return {
        "deel": deel,
        "dg": dg,
        "sig0": sig0,
        "p": p,
        "a": a,
        "deto_increment": deto_increment,
        "tangent_returned": np.asarray(trial.tangent_in_plane_mpa)[point].copy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-nodes", nargs=4, type=int, default=CROP_20X20)
    parser.add_argument("--increments", type=int, default=8)
    parser.add_argument("--library", default="build/mfront/src/libBehaviour.so")
    parser.add_argument("--ebsd-orientation-h5", type=Path, default=Path(EBSD_ORIENTATION_H5))
    parser.add_argument("--paired-parameter-set", default=PAIRED_PARAMETER_SET)
    parser.add_argument("--mfront-threads", type=int, default=4)
    parser.add_argument("--maximum-newton-iterations", type=int, default=40)
    parser.add_argument("--checkpoint-increment", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation/_generated/performance/gps_direct_sensitivity.json"),
    )
    arguments = parser.parse_args()

    from fem_inhouse.core.fcc_interaction_matrix import build_interaction_matrix
    from fem_inhouse.core.crystal_parameter_pairs import resolve_paired_crystal_parameters

    grid, yield_stress, coefficient, boundary = _load_case(arguments)
    material_gps, recording_gps, result_gps = _run_backend(
        GPS, arguments, grid, yield_stress, coefficient, boundary
    )
    material_ref, recording_ref, result_ref = _run_backend(
        REFERENCE, arguments, grid, yield_stress, coefficient, boundary
    )
    increment = arguments.checkpoint_increment
    calls_gps = _checkpoint_calls(recording_gps, increment)
    calls_ref = _checkpoint_calls(recording_ref, increment)
    strain_gps = np.asarray(calls_gps[0]["strain"], dtype=float)
    strain_ref = np.asarray(calls_ref[0]["strain"], dtype=float)
    dt = float(calls_gps[0]["time_increment"])
    snapshot_gps = recording_gps.committed_snapshots[increment - 2]
    snapshot_ref = recording_ref.committed_snapshots[increment - 2]

    # Parameters applied by the factory to the GPS behaviour.
    overrides, _ = resolve_paired_crystal_parameters(
        paired_parameter_set=arguments.paired_parameter_set,
        law="forest_rubin_srix",
    )
    d = _extract_elastic_stiffness(arguments.library)
    m = build_interaction_matrix((1.0, 1.0, 0.6, 1.8, 1.6, 12.3, 1.6))
    rotations = np.asarray(material_gps._rotations, dtype=float)

    rows = []
    for point in TOP_POINTS:
        state = _converged_increment(
            material_gps, snapshot_gps, strain_gps, dt, point
        )
        q = rotations[point]
        system = GpsLocalSystem(
            q=q,
            sig0=state["sig0"],
            p=state["p"],
            a=state["a"],
            d=d,
            m=m,
            r=float(overrides["SrixOverstressModulus"]),
            tau0=float(overrides["tau0"]),
            q_hard=float(overrides["Q"]),
            b=float(overrides["b"]),
            c_hard=float(overrides["C"]),
            d_hard=float(overrides["d"]),
        )
        x = np.concatenate((state["deel"], state["dg"]))
        # The DSL's deto: the INCREMENT of the imposed gradient (s1 - s0),
        # read from the converged evaluation.
        deto = state["deto_increment"]
        residual_at_convergence = system.residual(x, deto)
        a_jac = system.jacobian(x)
        # B = dF/deps_a (Kelvin): -I on the kinematic in-plane rows, 0 elsewhere.
        b_mat = np.zeros((18, 3), dtype=float)
        b_mat[0, 0] = -1.0
        b_mat[1, 1] = -1.0
        b_mat[3, 2] = -1.0
        from scipy.linalg import solve

        x_sens = solve(a_jac, -b_mat)
        dsig_dx = system.stress_a_sensitivity(x)
        c_sens_kelvin = dsig_dx @ x_sens
        c_sens = system.tangent_to_engineering(c_sens_kelvin)

        # C_shadow: the reference Schur on the GPS state (from the blocks run).
        from scripts.diagnose_gps_tangent_blocks import (
            _block_analysis,
            _schur_plane_stress,
            _raw_3d_tangent,
        )

        # Reference tangent on the GPS-transplanted state is costly to rebuild;
        # reuse the archived block analysis of gps_tangent_blocks.json.
        blocks_path = Path(
            "validation/_generated/performance/gps_tangent_blocks.json"
        )
        archived = json.loads(blocks_path.read_text(encoding="utf-8"))
        block_row = next(
            row for row in archived["rows"] if row["point"] == point
        )
        schur_ref = _schur_plane_stress(
            np.asarray(block_row["blocks_reference_on_gps_state"]["caa"])
            if False
            else np.asarray(
                _rebuild_tangent_from_blocks(block_row["blocks_reference_on_gps_state"])
            )
        )
        c_dsl = state["tangent_returned"]
        rel_shadow = float(
            np.linalg.norm(c_sens - schur_ref)
            / max(np.linalg.norm(schur_ref), 1.0e-30)
        )
        rel_dsl = float(
            np.linalg.norm(c_sens - c_dsl) / max(np.linalg.norm(c_dsl), 1.0e-30)
        )
        rows.append(
            {
                "point": point,
                "residual_norm_at_convergence": float(np.linalg.norm(residual_at_convergence)),
                "relative_to_shadow": rel_shadow,
                "relative_to_dsl": rel_dsl,
                "c_sens": c_sens.tolist(),
                "c_shadow": schur_ref.tolist(),
                "c_dsl": c_dsl.tolist(),
            }
        )
        print(
            f"point {point}: |F(x*)| = {np.linalg.norm(residual_at_convergence):.3e} | "
            f"C_sens vs shadow {rel_shadow:.3e} | C_sens vs DSL {rel_dsl:.3e}"
        )

    payload = {
        "schema_version": 1,
        "configuration": {
            "crop_nodes": arguments.crop_nodes,
            "increments": arguments.increments,
            "checkpoint_increment": increment,
        },
        "rows": rows,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _rebuild_tangent_from_blocks(blocks: dict[str, object]) -> np.ndarray:
    """Rebuild the 6x6 Kelvin tangent from the archived block analysis."""

    tangent = np.zeros((6, 6), dtype=float)
    caa = np.asarray(blocks["caa"])
    cab = np.asarray(blocks["cab"])
    cba = np.asarray(blocks["cba"])
    cbb = np.asarray(blocks["cbb"])
    tangent[np.ix_(_PLANE, _PLANE)] = caa
    tangent[np.ix_(_PLANE, _TRANSVERSE)] = cab
    tangent[np.ix_(_TRANSVERSE, _PLANE)] = cba
    tangent[np.ix_(_TRANSVERSE, _TRANSVERSE)] = cbb
    return tangent


if __name__ == "__main__":
    raise SystemExit(main())
