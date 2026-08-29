Kinematics-Driven 316L Strain Reconstruction
=============================================

This repository is a research platform for full-field computational mechanics
and inverse identification from measured kinematics.  It develops
full-Dirichlet spectral mechanics, three-dimensional constitutive laws under
structural plane stress, matrix-free sensitivities and adjoints, and
observation-aware inverse methods.  P43 is the primary current experimental
demonstrator; its data provenance and material conclusions remain under
consolidation.

MFront remains the qualified reference backend.  The native Python SRIX
backend reproduces that behaviour on the CPU and exposes the point-local
NumPy/Numba architecture intended for a future CuPy/GPU implementation,
including a coupled plane-stress closure that a generic MFront bridge cannot
perform internally.

.. image:: _static/workflow.*
   :alt: DIC displacement, crystal orientations, mechanical reconstruction and validation.
   :align: center
   :width: 100%

Choose what you need
--------------------

* **Learn by example** — :doc:`tutorials/index`
* **Use the software** — :doc:`how-to/index`
* **Understand the science** — :doc:`explanation/index`
* **Methodological landscape** — :doc:`explanation/methodological_landscape`
* **Look up a contract** — :doc:`reference/index`
* **Check the evidence** — :doc:`evidence/index`
* **Develop and maintain** — :doc:`maintainers/index`

Recommended journeys
--------------------

**Understand the methodological landscape**
   :doc:`explanation/methodological_landscape` →
   :doc:`explanation/spectral_mechanics/plastic_inverse_reuse` →
   :doc:`explanation/identification/femu_identification`

**First reconstruction**
   :doc:`tutorials/first_reconstruction` →
   :doc:`how-to/mechanics/compare_fields` →
   :doc:`reference/data/output_contract`

**Understand native SRIX**
   :doc:`explanation/constitutive/forest_rubin_srix` →
   :doc:`reference/numerics/native_srix_backend` →
   :doc:`how-to/crystal-plasticity/run_316l_crystal_plasticity`

**Understand the full-Dirichlet spectral solver**
   :doc:`explanation/spectral_mechanics/scientific_question` →
   :doc:`explanation/spectral_mechanics/full_dirichlet_formulation` →
   :doc:`explanation/spectral_mechanics/solver_pipeline` →
   :doc:`explanation/spectral_mechanics/tet2_newton_gmres`

**Find a scientific result**
   :doc:`evidence/index` →
   :doc:`reference/evidence/evidence_registry` →
   :doc:`how-to/reproduce/reproduce_ebi_falsification`

.. toctree::
   :hidden:
   :caption: Start here

   tutorials/index

.. toctree::
   :hidden:
   :caption: Use the software

   how-to/index

.. toctree::
   :hidden:
   :caption: Look things up

   reference/index

.. toctree::
   :hidden:
   :caption: Understand the science

   explanation/index

.. toctree::
   :hidden:
   :caption: Scientific evidence

   evidence/index

.. toctree::
   :hidden:
   :caption: Developer / maintainer

   maintainers/index
