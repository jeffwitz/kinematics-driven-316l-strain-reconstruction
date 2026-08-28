Kinematics-Driven 316L Strain Reconstruction
=============================================

Measured DIC kinematics and EBSD crystal orientations are combined in a
mechanical reconstruction problem.  The current stack uses three-dimensional
constitutive laws under structural plane stress, a matrix-free spectral/FEM
solver, and full-field comparison for qualification and identification.

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
* **Look up a contract** — :doc:`reference/index`
* **Check the evidence** — :doc:`evidence/index`
* **Develop and maintain** — :doc:`maintainers/index`

Recommended journeys
--------------------

**First reconstruction**
   :doc:`tutorials/first_reconstruction` →
   :doc:`how-to/compare_fields` →
   :doc:`reference/output_contract`

**Understand native SRIX**
   :doc:`explanation/constitutive/forest_rubin_srix` →
   :doc:`reference/numerics/native_srix_backend` →
   :doc:`how-to/crystal-plasticity/run_316l_crystal_plasticity`

**Find a scientific result**
   :doc:`evidence/index` →
   :doc:`reference/evidence_registry` →
   :doc:`how-to/reproduce_ebi_falsification`

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
