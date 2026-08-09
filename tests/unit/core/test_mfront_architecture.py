from fem_inhouse.core import mfront
from fem_inhouse.core.mfront_3d import MFront3DMaterialPointBatch
from fem_inhouse.core.mfront_condensation import MFront3DCondensedPlaneStressBatch
from fem_inhouse.core.mfront_gps.adapter import MFrontNativeGeneralisedPlaneStressBatch
from fem_inhouse.core.mfront_native import MFrontMaterialPointBatch


def test_historical_mfront_imports_are_compatibility_exports() -> None:
    assert mfront.MFrontMaterialPointBatch is MFrontMaterialPointBatch
    assert mfront.MFront3DMaterialPointBatch is MFront3DMaterialPointBatch
    assert (
        mfront.MFront3DCondensedPlaneStressBatch
        is MFront3DCondensedPlaneStressBatch
    )
    assert (
        mfront.MFrontNativeGeneralisedPlaneStressBatch
        is MFrontNativeGeneralisedPlaneStressBatch
    )


def test_gps_layers_are_separate_modules() -> None:
    assert MFrontNativeGeneralisedPlaneStressBatch.__module__.endswith(
        "mfront_gps.adapter"
    )
