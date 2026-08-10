Kinematics-Driven 316L Strain Reconstruction
=============================================

Measured DIC kinematics are turned into a plane-stress boundary-value problem
whose interior fields satisfy compatibility, the constitutive law and
equilibrium. The verified local J2/Ludwik baseline is implemented nominally in
MFront. Its localization is nevertheless too narrow and intense in
band-containing regions. An energetic micromorphic interaction therefore
introduces a coupling modulus and a spatial length, which are explored through
a staged F0/F1/F2 design.

.. image:: _static/workflow.*
   :alt: DIC displacement, local descriptors, mechanical boundary-value problem and reconstructed fields.
   :align: center
   :width: 100%

.. include:: _generated/current_conclusion.inc
   :parser: myst_parser.sphinx_

Choose what you need
--------------------

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: Tutorial
      :link: tutorials/first_reconstruction
      :link-type: doc

      Learn by running a local reconstruction, then compare it with one
      coupled solution.

   .. grid-item-card:: How-to
      :link: how-to/index
      :link-type: doc

      Prepare data, run calculations, inspect campaigns, compare fields and
      operate identification.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      Look up models, tensors, inputs, outputs, convergence, CLI, APIs and
      evidence.

   .. grid-item-card:: Explanation
      :link: explanation/index
      :link-type: doc

      Follow the scientific argument from measured kinematics to the present
      identifiability boundary.

Recommended journeys
--------------------

**Understand the science**
   :doc:`explanation/from_dic_to_mechanics` →
   :doc:`explanation/local_baseline` →
   :doc:`explanation/missing_spatial_interaction` →
   :doc:`explanation/dic_synthetic_measurement_tests` →
   :doc:`explanation/micromorphic_model` →
   :doc:`explanation/parameter_identification` →
   :doc:`explanation/current_evidence` →
   :doc:`explanation/scope_and_prediction`

**Understand the spectral mechanics evidence**
   :doc:`explanation/spectral_mechanics/index` →
   :doc:`explanation/spectral_mechanics/solver_pipeline` →
   :doc:`explanation/spectral_mechanics/one_point_instability` →
   :doc:`explanation/spectral_mechanics/tet2_newton_gmres` →
   :doc:`explanation/spectral_mechanics/ebi_srix_falsification`

**Reproduce the results**
   :doc:`how-to/install` → :doc:`how-to/prepare_case` →
   :doc:`how-to/run_local_reconstruction` →
   :doc:`how-to/run_coupled_reconstruction` →
   :doc:`how-to/run_identification`

**Run 316L crystal plasticity, having never used MFront**
   :doc:`how-to/install` → :doc:`how-to/run_316l_crystal_plasticity` →
   :doc:`how-to/choose_mfront_backend` →
   :doc:`how-to/use_srix_crystal_law` →
   :doc:`explanation/forest_rubin_srix`

**Extend the model**
   :doc:`reference/model_contract` →
   :doc:`reference/numerics/mfront_transaction` →
   :doc:`reference/numerics/three_dimensional_condensation` →
   :doc:`reference/numerics/mfront_structural_plane_stress` →
   :doc:`how-to/choose_mfront_backend` →
   :doc:`how-to/use_srix_crystal_law` →
   :doc:`how-to/add_mfront_behaviour`

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/index

.. toctree::
   :hidden:
   :caption: How-to

   how-to/index

.. toctree::
   :hidden:
   :caption: Reference

   reference/index

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/first_reconstruction
   tutorials/first_coupled_comparison
   tutorials/first_full_dirichlet_spectral_reconstruction
