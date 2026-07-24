import pytest

from fem_inhouse.config import CaseStudyConfig, MaterialConfig, MeshConfig, SolverConfig


def test_article_case_defaults_are_explicit() -> None:
    material = MaterialConfig()
    assert material.young_modulus_mpa == 205_000.0
    assert material.poisson_ratio == 0.30
    assert material.hardening_exponent == 0.245
    assert material.plastic_strain_max == 0.2
    assert material.plastic_table_points == 1_000
    assert material.first_positive_plastic_strain == 1e-6
    assert SolverConfig().hardening_mode == "ludwik"
    assert SolverConfig().constitutive_backend == "mfront"


def test_mesh_uses_article_pixel_scale() -> None:
    mesh = MeshConfig(nx=3_600, ny=3_100)
    assert mesh.element_size_mm == pytest.approx(0.00184)
    assert mesh.physical_size_mm == pytest.approx((6.624, 5.704))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: MaterialConfig(young_modulus_mpa=0), "young_modulus"),
        (lambda: MaterialConfig(poisson_ratio=0.5), "poisson_ratio"),
        (lambda: MaterialConfig(hardening_exponent=0), "hardening_exponent"),
        (lambda: MaterialConfig(plastic_strain_max=0), "plastic_strain_max"),
        (lambda: MaterialConfig(plastic_table_points=2), "plastic_table_points"),
        (
            lambda: MaterialConfig(first_positive_plastic_strain=0.2),
            "first_positive_plastic_strain",
        ),
        (lambda: MeshConfig(nx=0, ny=1), "nx and ny"),
        (lambda: MeshConfig(nx=1, ny=1, base_pixel_size_mm=0), "base_pixel_size_mm"),
        (lambda: MeshConfig(nx=1, ny=1, scale_factor=0), "scale_factor"),
        (lambda: SolverConfig(increments=0), "increments"),
        (lambda: SolverConfig(max_newton_iterations=0), "max_newton_iterations"),
        (lambda: SolverConfig(residual_tolerance=1.0), "residual_tolerance"),
        (lambda: SolverConfig(minimum_step_divisor=1), "minimum_step_divisor"),
        (
            lambda: SolverConfig(hardening_mode="unsupported"),  # type: ignore[arg-type]
            "hardening_mode",
        ),
        (
            lambda: SolverConfig(constitutive_backend="unsupported"),  # type: ignore[arg-type]
            "constitutive_backend",
        ),
        (lambda: SolverConfig(mfront_library=""), "mfront_library"),
        (lambda: SolverConfig(mfront_threads=0), "mfront_threads"),
    ],
)
def test_invalid_configuration_is_rejected(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_complete_case_configuration_composes_validated_parts() -> None:
    config = CaseStudyConfig(mesh=MeshConfig(nx=10, ny=12))
    assert config.mesh.physical_size_mm == pytest.approx((0.0184, 0.02208))
    assert config.material.plastic_table_points == 1_000
    assert config.solver.require_pypardiso is True
