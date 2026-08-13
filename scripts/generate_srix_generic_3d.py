#!/usr/bin/env python3
"""Generate the validation-only 3-D GenericBehaviour SRIX formulation.

The production SRIX behaviour remains the oracle.  This generator deliberately
keeps the transformation explicit so the generic local system can be reviewed
and compiled independently before any bridge integration.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def generate(source: str) -> str:
    text = source.replace("@DSL Implicit;", "@DSL ImplicitGenericBehaviour;")
    text = text.replace(
        "@Behaviour Fcc316LForestRubinSrix;",
        "@Behaviour Fcc316LForestRubinSrixGeneric3D;",
    )
    text = text.replace(
        "Fcc316LForestRubinSrixSlipSystems",
        "Fcc316LForestRubinSrixGeneric3DSlipSystems",
    )
    text = re.sub(r"\bD\b", "Ce", text)
    text = re.sub(r"@Brick StandardElasticity\{.*?\n\};\n", "", text, flags=re.S)
    text = text.replace(
        "@ExternalStateVariable strain chi;\nchi.setEntryName(\"NonlocalEquivalentPlasticStrain\");",
        """@Gradient StrainStensor eto;
eto.setGlossaryName(\"Strain\");
@ThermodynamicForce StressStensor sig;
sig.setGlossaryName(\"Stress\");
@Gradient strain chi;
chi.setEntryName(\"NonlocalEquivalentPlasticStrain\");
@ThermodynamicForce strain pobs;
pobs.setEntryName(\"AccumulatedSlipOutput\");
@StateVariable StrainStensor eel;
@StateVariable strain Gamma;
Gamma.setEntryName(\"AccumulatedSlip\");""",
    )
    text = text.replace(
        "@AuxiliaryStateVariable strain a[Nss];",
        "@AuxiliaryStateVariable strain a[Nss];\n\n@LocalVariable Stensor4 Ce;",
    )
    text = text.replace(
        "@Integrator {",
        """@Predictor {
  deel = deto;
  for (unsigned short i=0; i!=Nss; ++i) { dg[i] = 0; }
  dGamma = 0;
}

@ComputeThermodynamicForces {
  sig = Ce * eel;
  pobs = Gamma;
}

@ComputeFinalThermodynamicForces {
  sig = Ce * eel;
  pobs = Gamma;
}

@TangentOperatorBlocks{dsig_ddeto, dsig_ddchi, dpobs_ddeto, dpobs_ddchi};

@InitLocalVariables {
  // Cubic elasticity in the material frame, matching the production brick.
  Ce = Stensor4(0.);
  Ce(0,0) = 197000.; Ce(1,1) = 197000.; Ce(2,2) = 197000.;
  Ce(0,1) = 125000.; Ce(0,2) = 125000.; Ce(1,0) = 125000.;
  Ce(1,2) = 125000.; Ce(2,0) = 125000.; Ce(2,1) = 125000.;
  Ce(3,3) = 122000.; Ce(4,4) = 122000.; Ce(5,5) = 122000.;
}

@Integrator {
  using size_type = unsigned short;
  const auto& ss = Fcc316LForestRubinSrixGeneric3DSlipSystems<real>::getSlipSystems();
  const auto& m = ss.him;
  feel = deel - deto;
  dfeel_ddeel = Stensor4::Id();
  for (size_type i=0; i!=Nss; ++i) {
    feel += dg[i] * ss.mus[i];
    dfeel_ddg(i) = ss.mus[i];
    fg[i] = dg[i];
    dfg_ddg(i, i) = 1.;
  }
  fGamma = dGamma;
  dfGamma_ddGamma = 1.;
  for (size_type i=0; i!=Nss; ++i) {
    fGamma -= theta * abs(dg[i]);
    dfGamma_ddg(i) = -theta * (dg[i] > 0 ? 1 : (dg[i] < 0 ? -1 : 0));
  }
""",
        1,
    )
    text = text.replace(
        "  }  // end of the Deq >= deqeps branch\n}\n\n@UpdateAuxiliaryStateVariables",
        """  }  // end of the Deq >= deqeps branch
}

@TangentOperator {
  dfeel_ddeto = -Stensor4::Id();
  dfeel_ddchi = Stensor(0.);
  dfGamma_ddeto = Stensor(0.);
  dfGamma_ddchi = 0.;
  auto ddeel_ddeto = Stensor4{};
  auto dGamma_ddeto = Stensor{};
  getIntegrationVariablesDerivatives_eto(ddeel_ddeto, dGamma_ddeto);
  auto ddeel_ddchi = Stensor{};
  auto dGamma_ddchi = real{};
  getIntegrationVariablesDerivatives_chi(ddeel_ddchi, dGamma_ddchi);
  dsig_ddeto = Ce * ddeel_ddeto;
  dsig_ddchi = Ce * ddeel_ddchi;
  dpobs_ddeto = dGamma_ddeto;
  dpobs_ddchi = dGamma_ddchi;
}

@UpdateAuxiliaryStateVariables""",
        1,
    )
    duplicate = (
        "  using size_type = unsigned short;\n"
        "  const auto& ss = Fcc316LForestRubinSrixGeneric3DSlipSystems<real>::getSlipSystems();\n"
        "  const auto& m = ss.him;\n"
    )
    if text.count(duplicate) > 1:
        head, tail = text.rsplit(duplicate, 1)
        text = head + tail
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()), encoding="utf-8")


if __name__ == "__main__":
    main()
