import numpy as np
import pytest

from fem_inhouse.examples import reduced_biaxial_case
from fem_inhouse.partitioning import PartitionLayout
from fem_inhouse.postprocessing import field_error_metrics, interface_gradient_ratio
from fem_inhouse.solver import run_case_study
from fem_inhouse.workflows import PartitionWorkflow


@pytest.fixture(scope="module")
def homogeneous_reference():
    case = reduced_biaxial_case(nx=6, ny=6, constitutive_backend="python")
    result = run_case_study(
        case.config,
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
    )
    return case, result


@pytest.mark.parametrize("padding", [0, 1])
def test_partitioned_homogeneous_solution_matches_monolithic(
    tmp_path,
    homogeneous_reference,
    padding,
) -> None:
    case, reference = homogeneous_reference
    layout = PartitionLayout((6, 6), (2, 2), padding=padding)
    workflow = PartitionWorkflow(
        config=case.config,
        layout=layout,
        displacement_x_mm=case.displacement_x_mm,
        displacement_y_mm=case.displacement_y_mm,
        yield_stress_mpa=case.yield_stress_mpa,
        hardening_coefficient_mpa=case.hardening_coefficient_mpa,
        output_directory=tmp_path / f"padding-{padding}",
    )

    assert workflow.solve_pending() == [0, 1, 2, 3]
    displacement = workflow.stitch("U")
    stress = workflow.stitch("S")
    strain = workflow.stitch("E")
    equivalent_plastic_strain = workflow.stitch("PEEQ")

    np.testing.assert_allclose(displacement, reference.displacement_mm, rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(stress, reference.stress_mpa, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(strain, reference.total_strain, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        equivalent_plastic_strain,
        reference.equivalent_plastic_strain,
        rtol=1e-9,
        atol=1e-12,
    )
    metrics = field_error_metrics(reference.stress_mpa[..., 0], stress[..., 0])
    assert metrics.relative_l2_error < 1e-10
    assert interface_gradient_ratio(displacement[..., 0], layout) == pytest.approx(
        1.0,
        abs=1e-10,
    )
