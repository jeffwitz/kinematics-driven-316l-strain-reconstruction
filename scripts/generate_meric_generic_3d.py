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
@ThermodynamicForce StressStensor sig;
sig.setGlossaryName(\"Stress\");
@StateVariable StrainStensor eel;
@StateVariable strain g[Nss];
g.setEntryName(\"ViscoplasticSlip\");
@LocalVariable Stensor4 Ce;""",
    )
    text = text.replace(
        "@Integrator {",
        """@Predictor {
  deel = deto;
  for (unsigned short i=0; i!=Nss; ++i) { dg[i] = 0; }
}

@ComputeThermodynamicForces {
  sig = Ce * eel;
}

@ComputeFinalThermodynamicForces {
  sig = Ce * eel;
}

@InitLocalVariables {
  Ce = Stensor4(0.);
  Ce(0,0) = 197000.; Ce(1,1) = 197000.; Ce(2,2) = 197000.;
  Ce(0,1) = 125000.; Ce(0,2) = 125000.; Ce(1,0) = 125000.;
  Ce(1,2) = 125000.; Ce(2,0) = 125000.; Ce(2,1) = 125000.;
  Ce(3,3) = 244000.; Ce(4,4) = 244000.; Ce(5,5) = 244000.;
}

@TangentOperatorBlocks{dsig_ddeto};

@Integrator {
  feel = deel - deto;
  dfeel_ddeel = Stensor4::Id();
""",
        1,
    )
    text = text.replace(
        "@UpdateAuxiliaryStateVariables {",
        """@TangentOperator {
  dfeel_ddeto = -Stensor4::Id();
  auto ddeel_ddeto = Stensor4{};
  getIntegrationVariablesDerivatives_eto(ddeel_ddeto);
  dsig_ddeto = Ce * ddeel_ddeto;
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
