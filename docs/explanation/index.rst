Explanation
===========

These pages answer “why?” and connect the software design to the mechanics and
the supplied article.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Scientific goal
      :link: scientific_goal
      :link-type: doc

      Why DIC fields need mechanical reconstruction and what local descriptors
      mean.

   .. grid-item-card:: Constitutive law
      :link: material_law
      :link-type: doc

      Plane stress, J2 plasticity, Ludwik hardening, origin regularization, and
      the removed PEEQ cap.

   .. grid-item-card:: MFront and Newton
      :link: mfront_newton
      :link-type: doc

      How MGIS states, tensor conventions, consistent tangents, and
      trial/commit/revert fit the global nonlinear solve.

   .. grid-item-card:: Condensed 3D material law
      :link: mfront_3d_condensation
      :link-type: doc

      How a six-component MFront law is locally reduced to plane stress
      without changing the two-dimensional finite-element solve.

   .. grid-item-card:: Complete plane-stress tensors
      :link: plane_stress_tensors
      :link-type: doc

      Why the transverse strain is not zero, how the accepted 2D state is
      completed, and where MFront exposes its native axial variables.

   .. grid-item-card:: Partitioning
      :link: partitioning
      :link-type: doc

      Why overlap is used, how cores are owned, and how fields are stitched.

   .. grid-item-card:: Spatial-width diagnostic
      :link: nonlocality_diagnostic
      :link-type: doc

      Why an element-centred Helmholtz filter can test a localization-width
      hypothesis without changing the mechanical solution.

   .. grid-item-card:: Coupled micromorphic plasticity
      :link: micromorphic_plasticity
      :link-type: doc

      How the energetic H-chi coupling feeds the Helmholtz field back into
      J2 plastic evolution while preserving transactional Newton updates.

   .. grid-item-card:: P43 coupled results
      :link: p43_coupled_results
      :link-type: doc

      Detailed visual and quantitative interpretation of the first
      band-containing coupled campaign, including temporary conclusions.

   .. grid-item-card:: Validation strategy
      :link: validation
      :link-type: doc

      Distinguish code verification, constitutive comparison, DIC agreement,
      and external Abaqus validation.
