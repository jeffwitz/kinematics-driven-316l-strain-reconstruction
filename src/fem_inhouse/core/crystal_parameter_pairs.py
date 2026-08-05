"""Explicitly paired parameter sets for the Meric--SRIX comparison.

The pair is built from the existing immutable preset registries.  No material
constant is duplicated here: this module only locks the common backbone and
the two distinct flow rules together.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from fem_inhouse.core.single_crystal_presets import (
    FCC_INTERACTION_COEFFICIENTS,
    CubicElasticity,
    get_preset,
    srix_overstress_modulus_from_meric,
)
from fem_inhouse.core.srix_parameters import (
    ParameterOrigin,
    get_parameter_set,
)

CrystalLaw = Literal["meric_cailletaud", "forest_rubin_srix"]
PAIRED_PARAMETER_SET = "316l_guilhem2013_nasri2018_meric_srix_rate_1e-3"
BACKBONE_IDENTIFIER = "316l_guilhem2013_nasri2018"
REFERENCE_STRAIN_RATE_S_INV = 1.0e-3


@dataclass(frozen=True, slots=True)
class CrystalMaterialBackbone:
    identifier: str
    material: str
    elasticity: CubicElasticity
    interaction_matrix: tuple[float, ...]
    interaction_convention: str
    slip_system_family: str
    slip_system_count: int
    tau0_mpa: float
    q_mpa: float
    b: float
    c_mpa: float
    d: float
    temperature_k: float
    provenance: Mapping[str, ParameterOrigin]

    def canonical_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "material": self.material,
            "elasticity": {
                "C11_mpa": self.elasticity.c11_mpa,
                "C12_mpa": self.elasticity.c12_mpa,
                "C44_mpa": self.elasticity.c44_mpa,
            },
            "interaction_matrix": list(self.interaction_matrix),
            "interaction_convention": self.interaction_convention,
            "slip_system_family": self.slip_system_family,
            "slip_system_count": self.slip_system_count,
            "tau0_mpa": self.tau0_mpa,
            "Q_mpa": self.q_mpa,
            "b": self.b,
            "C_mpa": self.c_mpa,
            "d": self.d,
            "temperature_k": self.temperature_k,
            "provenance": {
                name: origin.record() for name, origin in sorted(self.provenance.items())
            },
        }

    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_record(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MericCailletaudFlowParameters:
    norton_strength_mpa: float
    norton_exponent: float
    origin: ParameterOrigin


@dataclass(frozen=True, slots=True)
class ForestRubinSrixFlowParameters:
    overstress_modulus_mpa: float
    origin: ParameterOrigin
    transposition_method: str | None
    reference_strain_rate_s_inv: float | None


@dataclass(frozen=True, slots=True)
class PairedCrystalLawSet:
    identifier: str
    backbone: CrystalMaterialBackbone
    meric: MericCailletaudFlowParameters
    srix: ForestRubinSrixFlowParameters
    pairing_status: str
    pairing_scope: str
    notes: str

    def validate(self) -> None:
        if self.backbone.identifier != BACKBONE_IDENTIFIER:
            raise ValueError("paired 316L set must use the registered common backbone")
        if len(self.backbone.interaction_matrix) != FCC_INTERACTION_COEFFICIENTS:
            raise ValueError("the paired FCC interaction matrix must have seven coefficients")
        if self.backbone.slip_system_count != 12:
            raise ValueError("the paired FCC backbone must expose twelve slip systems")
        source = get_preset(BACKBONE_IDENTIFIER)
        srix_source = get_parameter_set("316l_srix_transposed_from_nasri2018_rate_1e-3")
        if (
            self.backbone.elasticity != source.elasticity
            or srix_source.elasticity != source.elasticity
        ):
            raise ValueError("Méric and SRIX do not share the registered cubic elasticity")
        if (
            self.backbone.interaction_matrix != source.interaction_matrix
            or srix_source.interaction_matrix != source.interaction_matrix
        ):
            raise ValueError("Méric and SRIX do not share the registered interaction matrix")
        common = (
            (self.backbone.tau0_mpa, source.initial_crss_mpa, srix_source.tau0_mpa),
            (self.backbone.q_mpa, source.isotropic_saturation_mpa, srix_source.q_mpa),
            (self.backbone.b, source.isotropic_rate, srix_source.b),
            (self.backbone.c_mpa, source.kinematic_modulus_mpa, srix_source.c_mpa),
            (self.backbone.d, source.kinematic_recovery, srix_source.d),
        )
        if any(len(set(values)) != 1 for values in common):
            raise ValueError("Méric and SRIX do not share the registered hardening backbone")
        expected_r = srix_overstress_modulus_from_meric(
            norton_strength_mpa=self.meric.norton_strength_mpa,
            norton_exponent=self.meric.norton_exponent,
            reference_strain_rate=self.srix.reference_strain_rate_s_inv or 0.0,
        )
        relative = abs(self.srix.overstress_modulus_mpa - expected_r) / max(
            abs(expected_r), 1.0e-30
        )
        if relative >= 1.0e-13:
            raise ValueError(f"paired SRIX R does not match its Meric transposition: {relative:g}")
        if self.pairing_status != "analytical_transposition":
            raise ValueError("the registered pair must record analytical_transposition")

    def manifest(self) -> dict[str, object]:
        self.validate()
        backbone = self.backbone.canonical_record()
        return {
            "paired_parameter_set": self.identifier,
            "paired_set_selected_explicitly": True,
            "comparison_authorized": True,
            "backbone": {
                **backbone,
                "sha256": self.backbone.sha256(),
                "slip_systems": {
                    "crystal_structure": "FCC",
                    "family": self.backbone.slip_system_family,
                    "count": self.backbone.slip_system_count,
                },
                "hardening": {
                    "tau0_mpa": self.backbone.tau0_mpa,
                    "Q_mpa": self.backbone.q_mpa,
                    "b": self.backbone.b,
                    "C_mpa": self.backbone.c_mpa,
                    "d": self.backbone.d,
                },
            },
            "flow_rules": {
                "meric_cailletaud": {
                    "name": "meric_cailletaud",
                    "K_mpa": self.meric.norton_strength_mpa,
                    "n": self.meric.norton_exponent,
                    "time_dependent": True,
                    "duration_normalization": "total pseudo-time is one",
                },
                "forest_rubin_srix": {
                    "name": "forest_rubin_srix",
                    "R_mpa": self.srix.overstress_modulus_mpa,
                    "status": self.srix.origin.status,
                    "transposition_method": self.srix.transposition_method,
                    "reference_strain_rate_s_inv": self.srix.reference_strain_rate_s_inv,
                },
            },
            "pairing_status": self.pairing_status,
            "pairing_scope": self.pairing_scope,
            "notes": self.notes,
            "comparison_limits": [
                "The common 316L backbone is a literature prior, not an identification on P43.",
                "SRIX R is analytically transposed from Meric K and n at 1e-3 s^-1.",
                "The transposition matches a reference [001] loading condition and "
                "does not make the laws globally equivalent.",
                "The physical strain rate of the P43 DIC experiment is not documented.",
            ],
        }


def _build_pair() -> PairedCrystalLawSet:
    source = get_preset(BACKBONE_IDENTIFIER)
    srix = get_parameter_set("316l_srix_transposed_from_nasri2018_rate_1e-3")
    assert source.norton_strength_mpa is not None
    assert source.norton_exponent is not None
    assert source.initial_crss_mpa is not None
    assert source.isotropic_saturation_mpa is not None
    assert source.isotropic_rate is not None
    assert source.kinematic_modulus_mpa is not None
    assert source.kinematic_recovery is not None
    backbone = CrystalMaterialBackbone(
        identifier=source.identifier,
        material=source.material,
        elasticity=source.elasticity,
        interaction_matrix=source.interaction_matrix,
        interaction_convention=source.interaction_convention,
        slip_system_family="<0,1,-1>{1,1,1}",
        slip_system_count=12,
        tau0_mpa=float(source.initial_crss_mpa),
        q_mpa=float(source.isotropic_saturation_mpa),
        b=float(source.isotropic_rate),
        c_mpa=float(source.kinematic_modulus_mpa),
        d=float(source.kinematic_recovery),
        temperature_k=293.15,
        provenance={
            "backbone": ParameterOrigin(
                status="literature_prior",
                reference=source.provenance,
                note="Shared historical 316L backbone; not identified on P43.",
            )
        },
    )
    pair = PairedCrystalLawSet(
        identifier=PAIRED_PARAMETER_SET,
        backbone=backbone,
        meric=MericCailletaudFlowParameters(
            norton_strength_mpa=float(source.norton_strength_mpa),
            norton_exponent=float(source.norton_exponent),
            origin=ParameterOrigin(status="literature_prior", reference=source.provenance),
        ),
        srix=ForestRubinSrixFlowParameters(
            overstress_modulus_mpa=float(srix.overstress_modulus_mpa),
            origin=ParameterOrigin(
                status="analytical_transposition",
                reference="Forest and Rubin 2016, equation (16)",
                note="Transposed from the paired Meric K,n values.",
            ),
            transposition_method="forest_rubin_equation_16",
            reference_strain_rate_s_inv=srix.reference_strain_rate,
        ),
        pairing_status="analytical_transposition",
        pairing_scope="same_316l_backbone_matched_at_001_reference_rate",
        notes="The flow rule is the only constitutive difference intended by this pair.",
    )
    pair.validate()
    return pair


PAIRED_CRYSTAL_PARAMETER_SETS: dict[str, PairedCrystalLawSet] = {
    PAIRED_PARAMETER_SET: _build_pair()
}


def get_paired_crystal_parameter_set(identifier: str) -> PairedCrystalLawSet:
    try:
        return PAIRED_CRYSTAL_PARAMETER_SETS[identifier]
    except KeyError:
        known = ", ".join(sorted(PAIRED_CRYSTAL_PARAMETER_SETS))
        raise KeyError(
            f"unknown paired crystal parameter set {identifier!r}; registered: {known}"
        ) from None


def _common_mfront_overrides(pair: PairedCrystalLawSet) -> dict[str, float]:
    return {
        "tau0": pair.backbone.tau0_mpa,
        "Q": pair.backbone.q_mpa,
        "b": pair.backbone.b,
        "C": pair.backbone.c_mpa,
        "d": pair.backbone.d,
        **_elastic_brick_options(pair.backbone.elasticity),
    }


def _elastic_brick_options(elasticity: CubicElasticity) -> dict[str, float]:
    young = elasticity.young_modulus_100_mpa
    poisson = elasticity.poisson_ratio_100
    shear = elasticity.shear_modulus_mpa
    return {
        **{f"YoungModulus{i}": young for i in (1, 2, 3)},
        **{f"PoissonRatio{pair}": poisson for pair in ("12", "23", "13")},
        **{f"ShearModulus{pair}": shear for pair in ("12", "23", "13")},
    }


def resolve_paired_crystal_parameters(
    *, paired_parameter_set: str, law: CrystalLaw
) -> tuple[dict[str, float], dict[str, object]]:
    pair = get_paired_crystal_parameter_set(paired_parameter_set)
    pair.validate()
    overrides = _common_mfront_overrides(pair)
    if law == "meric_cailletaud":
        overrides.update({"K": pair.meric.norton_strength_mpa, "n": pair.meric.norton_exponent})
    elif law == "forest_rubin_srix":
        overrides["SrixOverstressModulus"] = pair.srix.overstress_modulus_mpa
        overrides["MinimumEquivalentStrainIncrement"] = 1.0e-14
    else:  # pragma: no cover - Literal protects callers, guard public API anyway.
        raise ValueError(f"unsupported crystal law {law!r}")
    manifest = pair.manifest()
    manifest["law"] = law
    flow_rules = manifest["flow_rules"]
    assert isinstance(flow_rules, dict)
    manifest["flow_rule"] = flow_rules[law]
    manifest["material_backbone"] = pair.backbone.canonical_record()
    return overrides, manifest
