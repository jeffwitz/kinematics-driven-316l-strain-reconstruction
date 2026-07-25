Command-line interface
======================

The installed entry point is ``fem-inhouse``.

Top-level commands
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Command
     - Purpose
   * - ``backend``
     - Verify and print the sparse linear-solver backend.
   * - ``validate``
     - Run the reduced analytical validation case.
   * - ``example``
     - Run and save the reduced example.
   * - ``prepare-case``
     - Verify raw DIC data and create canonical solver inputs.
   * - ``layout``
     - Write an article partition-layout manifest.
   * - ``partition``
     - Prepare, inspect, solve, resume, or stitch a partition campaign.
   * - ``compare-fields``
     - Compare co-registered fields against pre-declared thresholds.
   * - ``diagnose-nonlocality``
     - Sweep element-centred Helmholtz filters over a saved padded partition.
   * - ``estimate-nonlocal-reference``
     - Derive the pre-registered coupling-modulus sweep from a local partition.
   * - ``validate-coupled-nonlocal``
     - Compare raw local and coupled fields with DIC on one retained core.
   * - ``plot-coupled-alpha-fields``
     - Create reproducible raw EVM/PEEQ comparison figures for P154.

Run ``fem-inhouse COMMAND --help`` for the authoritative options installed with
the current source revision.

Global option
-------------

``--verbose``
   Enable progress and Newton-iteration logging.

``prepare-case`` modes
----------------------

Required paths are ``--raw`` and ``--output``. Principal options are:

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning
   * - ``--hardening-scale-mpa``
     - Convert the local hardening multiplier to MPa.
   * - ``--nonfinite-policy {error,nearest}``
     - Reject or explicitly repair non-finite multipliers.
   * - ``--nodal-completion edge-pad-upper``
     - Complete the upper nodal row and column.
   * - ``--crop-nx``, ``--crop-ny``
     - Create a deterministic central crop.

``partition`` contract
----------------------

Required campaign options:

.. code-block:: text

   --input DIRECTORY
   --output DIRECTORY
   --count {25,100}
   --padding ELEMENTS

``--parts-x N --parts-y M`` can replace ``--count`` for an explicit layout;
the two forms are mutually exclusive.

Exactly one action is required:

.. list-table::
   :header-rows: 1

   * - Action
     - Effect
   * - ``--list-pending``
     - Validate the campaign and print incomplete partition IDs.
   * - ``--partition-id N``
     - Solve one independent partition.
   * - ``--solve-pending``
     - Solve all incomplete partitions sequentially.
   * - ``--stitch {U,S,E,PE,PEEQ,RF}``
     - Stitch one complete global field.

Solver options include ``--increments``, ``--max-newton-iterations``,
``--residual-tolerance``,
``--constitutive-backend {python,mfront,mfront-native-plane-stress,mfront-3d-condensed-plane-stress}``,
``--mfront-library``, and ``--mfront-threads``.

Coupled MFront campaigns add ``--nonlocal-plasticity``,
``--nonlocal-length-um``, ``--nonlocal-coupling-modulus-mpa``,
``--nonlocal-relaxation``, ``--nonlocal-tolerance``, and
``--nonlocal-max-iterations``.

``estimate-nonlocal-reference`` contract
-----------------------------------------

.. code-block:: text

   --input PREPARED_CASE
   --campaign COMPLETED_LOCAL_CAMPAIGN
   --partition-id N
   --output REPORT.json
   --alphas 0 0.25 0.5 1 2

The command verifies the hardening-map fingerprint, reads ``PEEQ`` only from
the retained core, and reports the median Ludwik tangent together with every
``alpha * H_ref`` candidate. It refuses a genuinely coupled source campaign
and does not overwrite an existing report without ``--overwrite``.

``validate-coupled-nonlocal`` contract
--------------------------------------

.. code-block:: text

   --input PREPARED_CASE
   --local-campaign COMPLETED_LOCAL_CAMPAIGN
   --coupled-campaign COMPLETED_COUPLED_CAMPAIGN
   --partition-id N
   --output REPORT.json

The two campaigns must use identical input hashes, layout, material,
mechanical settings, and increment schedule. A nonlocal campaign with
``Hchi=0`` is accepted as a mechanically local smoke-test reference.

The command verifies every loaded output hash, reconstructs the DIC and FEM
historical EVM fields with the same displacement operator, and evaluates the
frozen P154 criteria only on the retained core. It never applies a Helmholtz
filter to the final FEM field. A complete report that fails one or more
scientific criteria is still written; the command then returns status 2.

``diagnose-nonlocality`` contract
---------------------------------

Required paths and identifiers are:

.. code-block:: text

   --input PREPARED_CASE
   --campaign SAVED_PARTITION_CAMPAIGN
   --partition-id N
   --output DIRECTORY

Exactly one length unit is required:

.. code-block:: text

   --lengths-mm VALUES...
   --lengths-um VALUES...
   --lengths-pixels VALUES...

The command adds zero, sorts and deduplicates valid lengths. Optional controls
are ``--include-peeq``, ``--mode {exploratory,confirmatory}``,
``--decision-thresholds FILE``, ``--top-fractions``, ``--dic-quantiles``,
``--minimum-padding-length-ratio``, ``--save-fields {all,best,none}``, and
``--overwrite``. Confirmatory mode requires a YAML or JSON thresholds file.
Existing non-empty output is never replaced unless ``--overwrite`` is present.

Filtered fields are post-processing products and are not valid
``partition --stitch`` field names.

``plot-coupled-alpha-fields`` contract
--------------------------------------

This command compares the local P154 campaign with the three registered
coupled campaigns (alpha = 0.5, 1, 2). It reconstructs the historical total
equivalent strain from saved nodal displacements, using the same operator as
the validation workflow, and extracts only the manifest-declared core:

.. code-block:: text

   fem-inhouse plot-coupled-alpha-fields \
     --input data/processed/case_study \
     --local-campaign results/constitutive-local-p0154-pad128 \
     --campaign-a050 results/constitutive-nonlocal-p0154-pad128-a050 \
     --campaign-a100 results/constitutive-nonlocal-p0154-pad128-a100 \
     --campaign-a200 results/constitutive-nonlocal-p0154-pad128-a200 \
     --partition-id 154 \
     --output validation/figures/p154-alpha-comparison \
     --include-optional-fields

The command accepts ``--dpi``, ``--format {png,pdf,svg}``, the three robust
colour-limit percentile options, ``--include-optional-fields``, and
``--overwrite``. PNG, PDF, and SVG are generated by default. The primary EVM
figures always use raw converged FEM displacements: no Helmholtz filter is
applied. ``PEEQ`` is an internal plasticity variable and is shown only as a
mechanism diagnostic, not as an experimental PEEQ comparison. The output
``plot_metadata.json`` records campaign hashes, core bounds, colour limits,
metrics, alpha/Hchi values, plotting options, and the explicit no-post-filter
contract.

Exit behaviour
--------------

The CLI returns a non-zero status for invalid contracts, unavailable required
backends, incompatible manifests, non-convergence, corrupt outputs, and failed
comparison thresholds. It does not convert these conditions into warnings.
