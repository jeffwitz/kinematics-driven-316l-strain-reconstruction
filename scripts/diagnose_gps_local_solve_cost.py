"""Where the joint UMAT solve spends its time, against the raw 18-unknown law.

The GPS backend makes 3.3 times FEWER constitutive calls than the nested
reference -- exactly what the closure-inside-the-law was for -- and loses
anyway, because each call costs about eight times more. This script isolates
that factor from the sub-stepping, by timing both laws at the same state with
sub-stepping disabled.

Both laws are driven to the same committed state, the GPS one with the
transverse strain its own closure found, so the raw law integrates the very
same problem with the transverse strain imposed.

Usage:
    MFRONT_BEHAVIOUR_LIBRARY=$PWD/build/mfront/src/libBehaviour.so \\
    .venv/bin/python scripts/diagnose_gps_local_solve_cost.py
"""
import os
import time

import numpy as np

from fem_inhouse.core.mfront import (
    _ENGINEERING_TO_KELVIN_STRAIN_SCALE,
    _PLANE_STRESS_COMPONENTS,
    MFront3DMaterialPointBatch,
)
from fem_inhouse.core.mfront_behaviours import MFRONT_BEHAVIOURS
from fem_inhouse.core.plane_stress_material import create_plane_stress_material_batch

lib = os.environ["MFRONT_BEHAVIOUR_LIBRARY"]
N, INC, MAX = 400, 12, 0.02
rot = np.broadcast_to(np.eye(3), (N,3,3)).copy()

def raw():
    return MFront3DMaterialPointBatch(
        lib, behaviour_spec=MFRONT_BEHAVIOURS.get("fcc_forest_rubin_srix"), point_count=N,
        rotation_global_to_material=rot, thread_count=4,
        behaviour_name="Fcc316LForestRubinSrix", behaviour_parameters=None)

def gps():
    b = create_plane_stress_material_batch(
        "mfront-native-generalised-plane-stress", np.full((N,1),250.), np.full((N,1),500.), 0.245,
        young_modulus_mpa=205000., poisson_ratio=0.3, hardening_mode="ludwik",
        plastic_strain_max=0.2, plastic_table_points=1000, first_positive_plastic_strain=1e-6,
        mfront_library=lib, mfront_threads=4, mfront_behaviour_id="fcc_forest_rubin_srix_gps",
        constitutive_options={"parameter_set": "316l_srix_exploratory_r1"})
    return b, (b._material if hasattr(b, "_material") else b)

# Drive the GPS law two increments (converges without sub-steps), record eps_zz.
b, inner = gps()
inner._maximum_substeps = 1
for k in (1, 2):
    e = np.tile((k/INC)*MAX*np.array([1.,-0.4,0.]), (N,1))
    b.evaluate(e, time_increment=1.0 / INC)
    b.commit()
target = np.tile((3/INC)*MAX*np.array([1.,-0.4,0.]), (N,1))
ezz = np.asarray(inner._manager.s0.gradients)[:, [2,4,5]].copy()

def timeit(fn, repeats=20):
    fn()
    ts = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - started)
    return float(np.median(ts))

t_gps = timeit(lambda: (b.evaluate(target, time_increment=1.0/INC), b.revert()))

r = raw()
for k in (1, 2):
    tot = np.zeros((N, 6))
    step = np.tile((k / INC) * MAX * np.array([1.0, -0.4, 0.0]), (N, 1))
    tot[:, _PLANE_STRESS_COMPONENTS] = step * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
    tot[:, [2, 4, 5]] = ezz * (k / 2)
    r.evaluate(tot, time_increment=1.0 / INC)
    r.commit()
tot = np.zeros((N, 6))
tot[:, _PLANE_STRESS_COMPONENTS] = target * _ENGINEERING_TO_KELVIN_STRAIN_SCALE
tot[:, [2, 4, 5]] = ezz * 1.5
t_raw = timeit(lambda: (r.evaluate(tot, time_increment=1.0/INC), r.revert()))

print(f"raw 18 inconnues : {t_raw*1e3:7.3f} ms  -> {t_raw/N*1e6:6.2f} us/point")
print(f"GPS 21 inconnues : {t_gps*1e3:7.3f} ms  -> {t_gps/N*1e6:6.2f} us/point   (sans sous-pas)")
print(f"rapport par integration : {t_gps/t_raw:.2f}x")
