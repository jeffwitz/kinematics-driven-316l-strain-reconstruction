Kinematics-Driven 316L Strain Reconstruction
=============================================

.. container:: landing-title

   Mechanically admissible reconstruction of microscale strain fields

.. container:: landing-subtitle

   This software starts from displacements measured by digital image
   correlation (DIC), associates local elastoplastic descriptors with the
   measurement grid, and solves a plane-stress finite-element problem. It is
   not a general replacement for Abaqus and it does not model each grain. Its
   purpose is to make the measured kinematics compatible with mechanical
   equilibrium for the published 316L case study.

.. image:: _static/workflow.*
   :alt: Complete workflow from DIC measurements to reconstructed finite-element fields.
   :align: center
   :width: 100%

Choose your path
----------------

The documentation follows `Diátaxis <https://diataxis.fr/>`_. Choose the
section that matches what you need now.

.. grid:: 1 2 2 4
   :gutter: 2

   .. grid-item-card:: Tutorial
      :link: tutorials/first_reconstruction
      :link-type: doc

      Run a first DIC-driven reconstruction and understand each result as it
      appears.

   .. grid-item-card:: How-to guides
      :link: how-to/index
      :link-type: doc

      Install MFront, prepare data, run partitions, resume a campaign, and
      inspect saved outputs.

   .. grid-item-card:: Reference
      :link: reference/index
      :link-type: doc

      Look up formats, units, axes, parameters, CLI commands, result fields,
      Python APIs, and architecture decisions.

   .. grid-item-card:: Explanation
      :link: explanation/index
      :link-type: doc

      Understand the scientific goal, J2/Ludwik law, MFront–Newton coupling,
      partitioning, and validation strategy.

What the software guarantees
----------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item::

      **Supported scope**

      * structured rectangular CPS4 mesh;
      * small strain and plane stress;
      * isotropic J2 plasticity with Ludwik hardening;
      * complete 3D stress and strain tensors reconstructed after convergence;
      * DIC-prescribed boundary displacements;
      * resumable, traceable partitioned computation;
      * preserved ``U``, ``S``, ``E``, ``PE``, ``PEEQ``, and ``RF`` fields.

   .. grid-item::

      **Outside the scope**

      * general-purpose finite-element analysis;
      * crystal plasticity or grain-property identification;
      * finite transformations, contact, or dynamics;
      * certified Abaqus parity without the original ``.inp`` and ODB files;
      * interpretation of local descriptors as intrinsic grain constants.

Validated state
---------------

.. container:: result-strip

   The **510 × 460-element** corner partition has converged with the analytical
   MFront law in **10 min 50.08 s**, over 20 increments without a cutback. All
   six fields, logs, hashes, and control maps are preserved. The complete
   11.16-million-element ROI has not yet been stitched.

Start here
----------

* :doc:`tutorials/first_reconstruction` if this is your first visit;
* :doc:`explanation/scientific_goal` for the scientific question;
* :doc:`explanation/material_law` for the constitutive model;
* :doc:`explanation/plane_stress_tensors` for complete 3D tensor reconstruction;
* :doc:`explanation/mfront_3d_condensation` for the experimental generic
  condensation of a 3D material law;
* :doc:`how-to/install` to install TFEL/MFront, MGIS, and the Python package;
* :doc:`reference/input_contract` to check input arrays;
* :doc:`reference/results` to interpret the validated campaigns.

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/first_reconstruction

.. toctree::
   :hidden:
   :caption: How-to guides

   how-to/index
   how-to/install
   how-to/prepare_data
   how-to/run_partitioned
   how-to/inspect_results

.. toctree::
   :hidden:
   :caption: Reference

   reference/index
   reference/input_contract
   reference/configuration
   reference/cli
   reference/output_contract
   reference/results
   reference/api
   reference/architecture_decisions

.. toctree::
   :hidden:
   :caption: Explanation

   explanation/index
   explanation/scientific_goal
   explanation/material_law
   explanation/plane_stress_tensors
   explanation/mfront_newton
   explanation/mfront_3d_condensation
   explanation/partitioning
   explanation/validation
