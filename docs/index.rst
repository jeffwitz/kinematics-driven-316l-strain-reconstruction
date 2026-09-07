Kinematics-Driven 316L Strain Reconstruction
=============================================

This repository is a research platform for full-field computational mechanics
and inverse identification from measured kinematics in 316L stainless steel.
It combines DIC kinematics and EBSD crystal orientations with mechanically
admissible reconstruction, three-dimensional constitutive laws under
structural plane stress, spectral and matrix-free mechanics, and
observation-aware inverse methods.

Project lineage
---------------

The project grew out of the `DIC-based in-house FEM implementation developed
by Adil Kılınç <https://github.com/kilincadil/DIC-based-inHouse-FEM>`_. That
implementation provided the initial computational basis for the
kinematics-driven reconstruction workflow, notably the structured
finite-element treatment of DIC-prescribed boundary conditions and the
J2/Ludwik reference model. It remains an explicit part of the scientific and
software provenance of this project.

The present repository is maintained as a **separate downstream research
project**. Since the initial import, its architecture and scientific scope have
diverged substantially. The historical solver has been refactored into a
modular package, while the project now includes structural plane-stress
condensation of 3D laws, MFront/MGIS integration, Forest--Rubin SRIX crystal
plasticity, native NumPy/Numba kernels, full-Dirichlet spectral mechanics,
matrix-free nonlinear solves and sensitivities, DIC/EBSD qualification,
non-local formulations, inverse identification, observability analysis,
automated validation and reproducible scientific evidence.

Keeping the repositories distinct preserves the provenance and maintenance
boundaries of both developments while allowing generic improvements to be
transferred independently when useful. Citation metadata retain Adil Kılınç
and the authors of the associated scientific work.

Current scope
-------------

P43 is the primary current experimental demonstrator; its data provenance and
material conclusions remain under consolidation. The project distinguishes
between reconstruction, identification and prediction, and between implemented
capabilities and claims supported by registered evidence.

MFront remains the qualified reference backend. The native Python SRIX backend
reproduces that behaviour on the CPU and exposes the point-local NumPy/Numba
architecture intended for a future CuPy/GPU implementation, including a
coupled plane-stress closure that a generic MFront bridge cannot perform
internally.

J2/Ludwik is retained as the historical and numerical baseline. The spectral
and matrix-free developments target mechanically admissible full-Dirichlet
reconstruction, scalable nonlinear solves, sensitivities and inverse problems
rather than a general-purpose finite-element replacement.

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
* **Check the evidence and claim boundaries** — :doc:`evidence/index`
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

**Find a scientific result and its evidence**
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
