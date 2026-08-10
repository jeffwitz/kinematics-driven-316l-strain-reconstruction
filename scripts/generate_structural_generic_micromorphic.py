#!/usr/bin/env python3
"""Generate the validation-only structural closure from the generic J2 probe.

The constitutive equations are kept in the source probe.  Only the mechanical
residual rows and their tangent right-hand sides are replaced; the chi/pobs
couple and the plasticity equations remain untouched.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

INTEGRATOR = r'''@Integrator {
  if (!plastic) {
    // In-plane kinematics and the three traction-free structural equations.
    feel(0) = deel(0) - deto(0);
    feel(1) = deel(1) - deto(1);
    feel(3) = deel(3) - deto(3);
    feel(2) = sig(2) / young;
    feel(4) = sig(4) / young;
    feel(5) = sig(5) / young;
    fp = dp;
    dfeel_ddeel = Stensor4::Id();
    dfeel_ddeel(2, 2) = De(2, 2) / young;
    dfeel_ddeel(2, 4) = De(2, 4) / young;
    dfeel_ddeel(2, 5) = De(2, 5) / young;
    dfeel_ddeel(4, 2) = De(4, 2) / young;
    dfeel_ddeel(4, 4) = De(4, 4) / young;
    dfeel_ddeel(4, 5) = De(4, 5) / young;
    dfeel_ddeel(5, 2) = De(5, 2) / young;
    dfeel_ddeel(5, 4) = De(5, 4) / young;
    dfeel_ddeel(5, 5) = De(5, 5) / young;
    dfeel_ddp = Stensor(0.);
    dfp_ddeel = Stensor(0.);
    dfp_ddp = 1;
    return true;
  }
  const auto seq = sigmaeq(sig);
  const auto n = 3 * deviator(sig) / (2 * seq);
  const auto p_ = p + theta * dp;
  const auto chi_ = chi + theta * dchi;
  const auto Rp = p_ <= p0
                      ? sy0 + K * p_ * pow(p0, ludwik_n - 1) + Hchi * (p_ - chi_)
                      : sy0 + K * pow(p_, ludwik_n) + Hchi * (p_ - chi_);
  const auto dR_dp = p_ < p0 ? K * pow(p0, ludwik_n - 1) + Hchi
                             : K * ludwik_n * pow(p_, ludwik_n - 1) + Hchi;

  // K = deel + dp*n is recovered from the standard elastic rows.  The
  // in-plane rows keep their kinematic equations; the transverse rows are
  // replaced by the three traction-free equations.
  feel(0) = deel(0) - deto(0) + dp * n(0);
  feel(1) = deel(1) - deto(1) + dp * n(1);
  feel(3) = deel(3) - deto(3) + dp * n(3);
  feel(2) = sig(2) / young;
  feel(4) = sig(4) / young;
  feel(5) = sig(5) / young;
  fp = (seq - Rp) / young;

  const auto dn_dsig = (3 * k4 / 2 - (n ^ n)) / seq;
  dfeel_ddeel = Stensor4::Id() + dp * (dn_dsig * De) * theta;
  dfeel_ddeel(2, 0) = De(2, 0) / young;
  dfeel_ddeel(2, 1) = De(2, 1) / young;
  dfeel_ddeel(2, 2) = De(2, 2) / young;
  dfeel_ddeel(2, 3) = De(2, 3) / young;
  dfeel_ddeel(2, 4) = De(2, 4) / young;
  dfeel_ddeel(2, 5) = De(2, 5) / young;
  dfeel_ddeel(4, 0) = De(4, 0) / young;
  dfeel_ddeel(4, 1) = De(4, 1) / young;
  dfeel_ddeel(4, 2) = De(4, 2) / young;
  dfeel_ddeel(4, 3) = De(4, 3) / young;
  dfeel_ddeel(4, 4) = De(4, 4) / young;
  dfeel_ddeel(4, 5) = De(4, 5) / young;
  dfeel_ddeel(5, 0) = De(5, 0) / young;
  dfeel_ddeel(5, 1) = De(5, 1) / young;
  dfeel_ddeel(5, 2) = De(5, 2) / young;
  dfeel_ddeel(5, 3) = De(5, 3) / young;
  dfeel_ddeel(5, 4) = De(5, 4) / young;
  dfeel_ddeel(5, 5) = De(5, 5) / young;
  dfeel_ddp(0) = n(0);
  dfeel_ddp(1) = n(1);
  dfeel_ddp(3) = n(3);
  dfeel_ddp(2) = 0;
  dfeel_ddp(4) = 0;
  dfeel_ddp(5) = 0;
  dfp_ddeel = theta * (n | De) / young;
  dfp_ddp = -theta * dR_dp / young;
}'''

TANGENT = r'''@TangentOperator {
  static_cast<void>(smt);
  // Only the three imposed in-plane components enter the closure.
  dfeel_ddeto = -Stensor4::Id();
  // Keep the standard elastic RHS only for the imposed in-plane components.
  // Stensor4 indices are tensorial/Kelvin-aware; zeroing columns explicitly
  // is safer than assigning three diagonal scalar entries.
  dfeel_ddeto(2, 2) = 0;
  dfeel_ddeto(2, 4) = 0;
  dfeel_ddeto(2, 5) = 0;
  dfeel_ddeto(4, 2) = 0;
  dfeel_ddeto(4, 4) = 0;
  dfeel_ddeto(4, 5) = 0;
  dfeel_ddeto(5, 2) = 0;
  dfeel_ddeto(5, 4) = 0;
  dfeel_ddeto(5, 5) = 0;
  dfp_ddeto = Stensor(0.);
  auto ddeel_ddeto = Stensor4{};
  auto ddp_ddeto = Stensor{};
  getIntegrationVariablesDerivatives_eto(ddeel_ddeto, ddp_ddeto);

  dfeel_ddchi = Stensor(0.);
  dfp_ddchi = plastic ? theta * Hchi / young : real(0);
  auto ddeel_ddchi = Stensor{};
  auto ddp_ddchi = real{};
  getIntegrationVariablesDerivatives_chi(ddeel_ddchi, ddp_ddchi);

  dsig_ddeto = De * ddeel_ddeto;
  dsig_ddchi = De * ddeel_ddchi;
  dpobs_ddeto = ddp_ddeto;
  dpobs_ddchi = ddp_ddchi;
}'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    text, count_integrator = re.subn(
        r"@Integrator \{.*?\n\}\n\n@ComputeFinalThermodynamicForces",
        INTEGRATOR + "\n\n@ComputeFinalThermodynamicForces",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text, count_tangent = re.subn(
        r"@TangentOperator \{.*?\n\}\s*$",
        TANGENT,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count_integrator != 1 or count_tangent != 1:
        raise RuntimeError("generic probe layout did not match the expected blocks")
    text = text.replace(
        "@Behaviour MicromorphicJ2GenericBlocksProbe;",
        "@Behaviour MicromorphicJ2GenericStructuralPlaneStressProbe;",
    )
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
