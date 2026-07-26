How-to guides
=============

Each page answers one operational question. If you are new to the project,
start with :doc:`../tutorials/first_reconstruction`.

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: Install the software
      :link: install
      :link-type: doc

      Build TFEL/MFront and MGIS, install the Python package, and compile the
      behaviour.

   .. grid-item-card:: Prepare input data
      :link: prepare_data
      :link-type: doc

      Verify raw arrays, select an explicit repair policy, and create canonical
      solver inputs.

   .. grid-item-card:: Run partitioned calculations
      :link: run_partitioned
      :link-type: doc

      Prepare, distribute, resume, and stitch the ROI partitions.

   .. grid-item-card:: Inspect a campaign
      :link: inspect_results
      :link-type: doc

      Check statuses, fingerprints, residuals, DIC boundary conditions, and
      saved maps.

   .. grid-item-card:: Diagnose spatial width
      :link: diagnose_nonlocality
      :link-type: doc

      Filter an existing padded partition, sweep Helmholtz lengths, and
      compare the retained core with DIC.

   .. grid-item-card:: Run coupled P154
      :link: run_coupled_nonlocal_p154
      :link-type: doc

      Derive H-ref, execute the smoke and validation profiles, and preserve
      every coupled field and diagnostic.

   .. grid-item-card:: Identify ell and H-chi
      :link: run_joint_nonlocal_identification
      :link-type: doc

      Run the staged F0/F1 design, build the Pareto front, and generate an
      approval-gated F2 proposal without launching it.
