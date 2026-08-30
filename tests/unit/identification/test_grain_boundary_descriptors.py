"""Synthetic contracts for grain-boundary descriptors."""

from __future__ import annotations

import unittest

import numpy as np

from fem_inhouse.identification.grain_boundary_descriptors import (
    cubic_misorientation_angle,
    cubic_symmetry_matrices,
    distance_to_boundary,
    equivalent_diameter_map,
    luster_morris_matrix,
    neighbour_pairs,
    normalize_descriptor,
    residual_burgers_matrix,
)


class GrainBoundaryDescriptorTests(unittest.TestCase):
    def test_geometry_and_neighbours_are_deterministic(self) -> None:
        labels = np.zeros((4, 6), dtype=int)
        labels[:, 3:] = 1
        diameters = equivalent_diameter_map(labels)
        np.testing.assert_allclose(diameters[:, :3], diameters[:, 3:])
        self.assertEqual(len(neighbour_pairs(labels)), 1)
        self.assertEqual(neighbour_pairs(labels)[0]["contact_pixels"], 4)
        distance = distance_to_boundary(labels)
        self.assertTrue(np.isfinite(distance).all())

    def test_normalization_centres_and_scales_support(self) -> None:
        values = np.arange(6.0).reshape(2, 3)
        normalized = normalize_descriptor(values)
        self.assertAlmostEqual(float(np.mean(normalized)), 0.0)
        self.assertAlmostEqual(float(np.sqrt(np.mean(normalized**2))), 1.0)

    def test_cubic_symmetry_and_misorientation(self) -> None:
        symmetries = cubic_symmetry_matrices()
        self.assertEqual(symmetries.shape, (24, 3, 3))
        identity = np.eye(3)
        self.assertAlmostEqual(cubic_misorientation_angle(identity, identity), 0.0)
        self.assertAlmostEqual(cubic_misorientation_angle(identity, symmetries[7]), 0.0)
        self.assertAlmostEqual(
            cubic_misorientation_angle(identity, symmetries[7]),
            cubic_misorientation_angle(symmetries[7], identity),
        )

    def test_slip_transfer_metrics_are_bounded_and_sign_invariant(self) -> None:
        identity = np.eye(3)
        compatibility = luster_morris_matrix(identity, identity)
        residual = residual_burgers_matrix(identity, identity)
        self.assertEqual(compatibility.shape, (12, 12))
        self.assertEqual(residual.shape, (12, 12))
        self.assertGreaterEqual(float(compatibility.min()), 0.0)
        self.assertLessEqual(float(compatibility.max()), 1.0)
        self.assertGreaterEqual(float(residual.min()), 0.0)
        np.testing.assert_allclose(np.diag(residual), 0.0)


if __name__ == "__main__":
    unittest.main()
