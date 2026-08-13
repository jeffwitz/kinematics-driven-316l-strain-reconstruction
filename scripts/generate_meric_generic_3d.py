#!/usr/bin/env python3
"""Generate the validation-only 3-D GenericBehaviour form of Méric-Cailletaud."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def generate(source: str) -> str:
    text = source.replace("@DSL Implicit;", "@DSL ImplicitGenericBehaviour;")
    text = text.replace(
        "@Behaviour Fcc316LMericCailletaud;",
        "@Behaviour Fcc316LMericCailletaudGeneric3D;",
    )
    text = text.replace(
        "Fcc316LMericCailletaudSlipSystems",
        "Fcc316LMericCailletaudGeneric3DSlipSystems",
    )
    text = re.sub(r"\bD\b", "Ce", text)
    text = re.sub(r"@Brick StandardElasticity\{.*?\n\};\n", "", text, flags=re.S)
    text = text.replace(
        "@StateVariable strain g[Nss];\n"
        'g.setEntryName("ViscoplasticSlip");',
        """@Gradient StrainStensor eto;
eto.setGlossaryName(\"Strain\");
@Gradient strain chi;
chi.setEntryName(\"NonlocalEquivalentPlasticStrain\");
@ThermodynamicForce StressStensor sig;
sig.setGlossaryName(\"Stress\");
@ThermodynamicForce strain pobs;
pobs.setEntryName(\"AccumulatedSlipOutput\");
@StateVariable StrainStensor eel;
@StateVariable strain g[Nss];
g.setEntryName(\"ViscoplasticSlip\");
@StateVariable strain Gamma;
Gamma.setEntryName(\"AccumulatedSlip\");
@LocalVariable Stensor4 Ce;""",
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

@InitLocalVariables {
  Ce = Stensor4(0.);
  Ce(0,0) = 197000.; Ce(1,1) = 197000.; Ce(2,2) = 197000.;
  Ce(0,1) = 125000.; Ce(0,2) = 125000.; Ce(1,0) = 125000.;
  Ce(1,2) = 125000.; Ce(2,0) = 125000.; Ce(2,1) = 125000.;
  Ce(3,3) = 244000.; Ce(4,4) = 244000.; Ce(5,5) = 244000.;
}

@TangentOperatorBlocks{dsig_ddeto, dsig_ddchi, dpobs_ddeto, dpobs_ddchi};

@Integrator {
  feel = deel - deto;
  dfeel_ddeel = Stensor4::Id();
  fGamma = dGamma;
  dfGamma_ddGamma = 1.;
  for (unsigned short i=0; i!=Nss; ++i) {
    fGamma -= abs(dg[i]);
    dfGamma_ddg(i) = -(dg[i] > 0 ? 1 : (dg[i] < 0 ? -1 : 0));
  }
""",
        1,
    )
    text = text.replace(
        "@UpdateAuxiliaryStateVariables {",
        """@TangentOperator {
  dfeel_ddeto = -Stensor4::Id();
  dfeel_ddchi = Stensor(0.);
  dsig_ddchi = Stensor(0.);
  dpobs_ddchi = 0.;
  auto ddeel_ddeto = Stensor4{};
  auto dgs_ddeto = tfel::math::tvector<Nss, tfel::math::derivative_type<strain, Stensor>>{};
  auto dGamma_ddeto = Stensor{};
  getIntegrationVariablesDerivatives_eto(ddeel_ddeto, dgs_ddeto, dGamma_ddeto);
  dsig_ddeto = Ce * ddeel_ddeto;
  dpobs_ddeto = dGamma_ddeto;
}

@UpdateAuxiliaryStateVariables {""",
        1,
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(args.source.read_text()), encoding="utf-8")


if __name__ == "__main__":
    main()
