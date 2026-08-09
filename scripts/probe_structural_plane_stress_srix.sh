#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
source_mfront="${1:-$repo_root/mfront/Fcc316LForestRubinSrix.mfront}"
behaviour_name="${2:-${STRUCTURAL_BEHAVIOUR_NAME:-Fcc316LForestRubinSrix}}"
probe_dir=$(mktemp -d /tmp/structural-plane-stress-srix.XXXXXX)
generated_output="${STRUCTURAL_PLANE_STRESS_OUTPUT:-$probe_dir/Fcc316LForestRubinSrixStructuralProbe.mfront}"
set +u
source /home/jeff/.local/share/tfel/env/env.sh
set -u
python_bin="$repo_root/.venv/bin/python"

STRUCTURAL_BEHAVIOUR_NAME="$behaviour_name" "$python_bin" - "$source_mfront" "$generated_output" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
target = Path(sys.argv[2])
generated_behaviour = __import__("os").environ["STRUCTURAL_BEHAVIOUR_NAME"]
if generated_behaviour != "Fcc316LForestRubinSrix":
    source = source.replace("Fcc316LForestRubinSrix", generated_behaviour)
slip_class = generated_behaviour + "SlipSystems"
source = source.replace(
    "@Behaviour Fcc316LForestRubinSrix;",
    f"@Behaviour {generated_behaviour};",
    1,
)
aux = r'''
@MaterialProperty real Q11;
@MaterialProperty real Q12;
@MaterialProperty real Q13;
@MaterialProperty real Q21;
@MaterialProperty real Q22;
@MaterialProperty real Q23;
@MaterialProperty real Q31;
@MaterialProperty real Q32;
@MaterialProperty real Q33;
@Parameter real CondensedTangent = 1.;
@AuxiliaryStateVariable real structuralTotalStrain[6];
structuralTotalStrain.setEntryName("StructuralTotalStrain");
@AuxiliaryStateVariable strain ezz;
@AuxiliaryStateVariable strain eyz;
@AuxiliaryStateVariable strain exz;
@AuxiliaryStateVariable real structuralJacobian[324];
structuralJacobian.setEntryName("StructuralJacobian");

@Private {
  Stensor rotate(const Stensor& s) const {
    const auto r00 = this->Q11, r01 = this->Q21, r02 = this->Q31;
    const auto r10 = this->Q12, r11 = this->Q22, r12 = this->Q32;
    const auto r20 = this->Q13, r21 = this->Q23, r22 = this->Q33;
    const auto q = sqrt(real(2));
    const auto s0 = s(0), s1 = s(1), s2 = s(2);
    const auto s3 = s(3), s4 = s(4), s5 = s(5);
    auto r = s;
    r(0) = r00*r00*s0+r01*r01*s1+r02*r02*s2
         + q*(r00*r01*s3+r00*r02*s4+r01*r02*s5);
    r(1) = r10*r10*s0+r11*r11*s1+r12*r12*s2
         + q*(r10*r11*s3+r10*r12*s4+r11*r12*s5);
    r(2) = r20*r20*s0+r21*r21*s1+r22*r22*s2
         + q*(r20*r21*s3+r20*r22*s4+r21*r22*s5);
    r(3) = q*(r00*r10*s0+r01*r11*s1+r02*r12*s2)
         +(r00*r11+r01*r10)*s3+(r00*r12+r02*r10)*s4
         +(r01*r12+r02*r11)*s5;
    r(4) = q*(r00*r20*s0+r01*r21*s1+r02*r22*s2)
         +(r00*r21+r01*r20)*s3+(r00*r22+r02*r20)*s4
         +(r01*r22+r02*r21)*s5;
    r(5) = q*(r10*r20*s0+r11*r21*s1+r12*r22*s2)
         +(r10*r21+r11*r20)*s3+(r10*r22+r12*r20)*s4
         +(r11*r22+r12*r21)*s5;
    return r;
  }
};
'''
integrator = r'''
  constexpr auto Gref = real(210000.);
  constexpr ushort transverse[3] = {2, 4, 5};
  using jacobian_type = decltype(this->jacobian);
  using indexing_policy = typename jacobian_type::indexing_policy;
  constexpr indexing_policy indexing{};
  constexpr auto system_size = indexing.size(0);
  const auto sig_global = rotate(this->sig);
  auto K_material = Stensor(real(0));
  for (ushort i = 0; i != StensorSize; ++i) {
    K_material(i) = this->fzeros(i) + this->deto(i);
  }
  const auto K_global = rotate(K_material);
  tmatrix<StensorSize, StensorSize, real> rotation;
  for (ushort j = 0; j != StensorSize; ++j) {
    auto basis = Stensor(real(0));
    basis(j) = 1;
    const auto column = rotate(basis);
    for (ushort i = 0; i != StensorSize; ++i) {
      rotation(i, j) = column(i);
    }
  }
  const auto raw_jacobian = this->jacobian;
  for (ushort i = 0; i != StensorSize; ++i) {
    this->fzeros(i) = K_global(i) - this->deto(i);
    for (ushort j = 0; j != system_size; ++j) {
      auto value = real(0);
      for (ushort k = 0; k != StensorSize; ++k) {
        value += rotation(i, k) * raw_jacobian(k, j);
      }
      this->jacobian(i, j) = value;
    }
  }
  for (ushort k = 0; k != 3; ++k) {
    const auto i = transverse[k];
    this->fzeros(i) = sig_global(i) / Gref;
    for (ushort j = 0; j != system_size; ++j) {
      auto value = real(0);
      if (j < StensorSize) {
        for (ushort l = 0; l != StensorSize; ++l) {
          value += rotation(i, l) * this->D_tdt(l, j);
        }
      }
      this->jacobian(i, j) = value / Gref;
    }
  }
  for (ushort i = 0; i != system_size; ++i) {
    for (ushort j = 0; j != system_size; ++j) {
      this->structuralJacobian[i * system_size + j] = this->jacobian(i, j);
    }
  }
'''
marker = "\n}\n\n@UpdateAuxiliaryStateVariables"
if source.count(marker) != 1:
    raise SystemExit(f"unexpected SRIX integrator marker count: {source.count(marker)}")
source = source.replace(marker, "\n" + integrator + "}\n\n@UpdateAuxiliaryStateVariables", 1)
source = source.replace("@StateVariable strain g[Nss];", aux + "\n@StateVariable strain g[Nss];", 1)
source = source.replace("@UpdateAuxiliaryStateVariables {", """@UpdateAuxiliaryStateVariables {
  const auto& ss = SLIP_SYSTEM_CLASS<real>::getSlipSystems();
  auto material_total = this->deel;
  for (ushort i = 0; i != Nss; ++i) {
    material_total += this->dg[i] * ss.mus[i];
  }
  const auto global_total = rotate(material_total);
  for (ushort i = 0; i != StensorSize; ++i) {
    this->structuralTotalStrain[i] = global_total(i);
  }
  this->ezz = global_total(2);
  this->eyz = global_total(5);
  this->exz = global_total(4);""".replace("SLIP_SYSTEM_CLASS", slip_class), 1)
tangent = r'''

@TangentOperator {
  constexpr ushort active_columns[3] = {0, 1, 3};
  using jacobian_type = decltype(this->jacobian);
  using indexing_policy = typename jacobian_type::indexing_policy;
  constexpr indexing_policy indexing{};
  constexpr auto system_size = indexing.size(0);
  tmatrix<system_size, system_size, real> A;
  for (ushort i = 0; i != system_size; ++i) {
    for (ushort j = 0; j != system_size; ++j) {
      A(i, j) = this->structuralJacobian[i * system_size + j];
    }
  }
  TinyPermutation<system_size> permutation;
  if (!TinyMatrixSolve<system_size, real, false>::decomp(A, permutation)) {
    return false;
  }
  this->Dt = Stensor4(real(0));
  tmatrix<StensorSize, StensorSize, real> rotation;
  for (ushort j = 0; j != StensorSize; ++j) {
    auto basis = Stensor(real(0));
    basis(j) = 1;
    const auto column = rotate(basis);
    for (ushort i = 0; i != StensorSize; ++i) {
      rotation(i, j) = column(i);
    }
  }
  for (ushort column = 0; column != 3; ++column) {
    tvector<system_size, real> rhs(real(0));
    rhs(active_columns[column]) = 1;
    if (!TinyMatrixSolve<system_size, real, false>::back_substitute(
            A, permutation, rhs)) {
      return false;
    }
    auto stress_material = Stensor(real(0));
    for (ushort i = 0; i != StensorSize; ++i) {
      for (ushort j = 0; j != StensorSize; ++j) {
        stress_material(i) += this->D_tdt(i, j) * rhs(j);
      }
    }
    const auto stress_global = rotate(stress_material);
    for (ushort i = 0; i != StensorSize; ++i) {
      this->Dt(i, active_columns[column]) = stress_global(i);
    }
  }
};
'''
source += tangent
target.write_text(source)
PY

if [[ "${STRUCTURAL_PLANE_STRESS_GENERATE_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "$generated_output"
  exit 0
fi

cd "$probe_dir"
mfront --obuild --interface=generic "$generated_output" >/dev/null
BEHAVIOUR_NAME="$behaviour_name" LIBRARY="$probe_dir/src/libBehaviour.so" "$python_bin" - <<'PY'
import os
import mgis.behaviour as mgis
import numpy as np

behaviour = mgis.load(os.environ["LIBRARY"], os.environ["BEHAVIOUR_NAME"],
                      mgis.Hypothesis.Tridimensional)
q = np.array([
    [0.6517403912340062, 0.7532585459971657, 0.08852132690137686],
    [-0.7326322075147665, 0.5950699920075869, 0.33036608954935215],
    [0.19617469496901108, -0.2801664995932355, 0.9396926207859084],
])
assert np.max(np.abs(q @ q.T - np.eye(3))) < 1e-12
assert abs(np.linalg.det(q) - 1.0) < 1e-12
t = q.T

def rotate(values):
    s = np.array([[values[0], values[3]/np.sqrt(2), values[4]/np.sqrt(2)],
                  [values[3]/np.sqrt(2), values[1], values[5]/np.sqrt(2)],
                  [values[4]/np.sqrt(2), values[5]/np.sqrt(2), values[2]]])
    r = t @ s @ t.T
    return np.array([r[0, 0], r[1, 1], r[2, 2], np.sqrt(2)*r[0, 1],
                     np.sqrt(2)*r[0, 2], np.sqrt(2)*r[1, 2]])

data = mgis.MaterialDataManager(behaviour, 1)
for state in (data.s0, data.s1):
    for row in range(3):
        for column in range(3):
            mgis.setMaterialProperty(state, f"Q{row + 1}{column + 1}", q[row, column])
for state in (data.s0, data.s1):
    mgis.setExternalStateVariable(state, "Temperature", 293.15)
scale = 1.0e-2 if os.environ["BEHAVIOUR_NAME"].endswith("Srix") else 1.0e-4
strain = scale * np.array([1.0, -2.0e-1, 0.0, 3.0e-1, 0.0, 0.0])
time_increment = 1.0 if os.environ["BEHAVIOUR_NAME"].endswith("Srix") else 1.0e-3
data.s1.gradients[0] = strain
result = mgis.integrate(data, mgis.IntegrationType.IntegrationWithoutTangentOperator,
                        time_increment, 0, 1)
if result != 1:
    raise SystemExit(f"SRIX structural closure integration failed: result={result}")
stress_global = rotate(np.asarray(data.s1.thermodynamic_forces[0]))
transverse = np.abs(stress_global[[2, 4, 5]])
if float(np.max(transverse)) > 1.0e-7:
    raise SystemExit(f"SRIX plane-stress closure failed: {transverse}")
total = np.asarray(data.s1.internal_state_variables[0])
offset = 0
structural_indices = []
for variable in behaviour.internal_state_variables:
    width = 6 if variable.type == mgis.VariableType.Stensor else 1
    if variable.name.startswith("StructuralTotalStrain"):
        structural_indices.extend(range(offset, offset + width))
    offset += width
if len(structural_indices) != 6:
    raise SystemExit(f"could not locate StructuralTotalStrain metadata: {structural_indices}")
structural_total = total[structural_indices]
kinematic_error = np.max(np.abs(structural_total[[0, 1, 3]] - strain[[0, 1, 3]]))
if float(kinematic_error) > 1.0e-10:
    raise SystemExit(f"SRIX in-plane kinematics failed: {kinematic_error}")

def integrate_with_tangent(value):
    trial = mgis.MaterialDataManager(behaviour, 1)
    for state in (trial.s0, trial.s1):
        for row in range(3):
            for column in range(3):
                mgis.setMaterialProperty(state, f"Q{row + 1}{column + 1}", q[row, column])
    for state in (trial.s0, trial.s1):
        mgis.setExternalStateVariable(state, "Temperature", 293.15)
    trial.s1.gradients[0] = value
    status = mgis.integrate(
        trial, mgis.IntegrationType.IntegrationWithConsistentTangentOperator,
        time_increment, 0, 1,
    )
    if status != 1:
        raise SystemExit(f"{os.environ['BEHAVIOUR_NAME']} tangent integration failed: {status}")
    return rotate(np.asarray(trial.s1.thermodynamic_forces[0])), np.asarray(trial.K[0]).copy()

tangent_stress, tangent = integrate_with_tangent(strain)
active = np.array([0, 1, 3])
fd_errors = []
for step in (1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7):
    fd = np.zeros((3, 3))
    for column, component in enumerate(active):
        plus = strain.copy()
        minus = strain.copy()
        plus[component] += step
        minus[component] -= step
        plus_stress = integrate_with_tangent(plus)[0]
        minus_stress = integrate_with_tangent(minus)[0]
        fd[:, column] = (plus_stress[active] - minus_stress[active]) / (2 * step)
    error = np.max(np.abs(fd - tangent[np.ix_(active, active)])) / np.max(
        np.abs(tangent[np.ix_(active, active)]))
    fd_errors.append(float(error))
    if error > 1.0e-4:
        print("FD=", fd)
        print("Tangent=", tangent[np.ix_(active, active)])
        raise SystemExit(f"{os.environ['BEHAVIOUR_NAME']} tangent failed for h={step}: {error}")
inactive = np.delete(np.arange(6), active)
inactive_error = float(np.max(np.abs(tangent[:, inactive])))
print(f"generic {os.environ['BEHAVIOUR_NAME']} structural plane-stress probe: passed "
      f"(max transverse stress={np.max(transverse):.3e}, "
      f"max in-plane strain error={kinematic_error:.3e}, "
      f"tangent FD errors={fd_errors}, max inactive-column={inactive_error:.3e}, "
      f"internal-state-size={total.size})")
PY
