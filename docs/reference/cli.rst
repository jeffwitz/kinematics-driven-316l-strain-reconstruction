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
   * - ``select-dic-partition``
     - Rank DIC partitions by robust EVM spatial heterogeneity.
   * - ``identify-nonlocal``
     - Run explicit F0/F1 identification stages and generate approval-gated
       F2 proposals.

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
``--residual-tolerance``, ``--minimum-step-divisor``,
``--constitutive-backend {python,mfront,mfront-native-plane-stress,mfront-3d-condensed-plane-stress}``,
``--mfront-library``, and ``--mfront-threads``.

Coupled MFront campaigns add ``--nonlocal-plasticity``,
``--nonlocal-length-um``, ``--nonlocal-coupling-modulus-mpa``,
``--nonlocal-relaxation``, ``--nonlocal-relaxation-strategy``,
``--nonlocal-minimum-relaxation``, ``--nonlocal-maximum-relaxation``,
``--nonlocal-aitken-residual-growth-factor``, ``--nonlocal-tolerance``,
``--nonlocal-max-iterations``, and
``--nonlocal-record-iteration-history``.

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

``identify-nonlocal`` contract
------------------------------

Every action requires a versioned ``--config``. The supported actions are:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Action
     - Effect
   * - ``inspect``
     - Verify inputs and display the planned F0/F1/F2 scope without writing.
   * - ``screen-frozen``
     - Run or reuse the DCT-only frozen-PEEQ screen.
   * - ``run-low-fidelity``
     - Replay reduced coupled mechanics; ``--design`` selects the historical
       sparse DOE, ``--identifiability-design`` selects the homogeneous
       saturation/constant-A/fixed-alpha experiment, and repeated
       ``--point alpha:ell_um`` selects explicit points. These selectors are
       mutually exclusive.
   * - ``collect-results``
     - Build an immutable consolidated F1/F2 CSV and JSON table.
   * - ``profile-h``
     - Profile the amplitude objective at each sampled length using bounded
       PCHIP interpolation.
   * - ``select-candidates``
     - Construct the amplitude-localization Pareto front and its knee.
   * - ``generate-high-fidelity-manifest``
     - Write the configured bounded set of proposed F2 commands. For a
       discriminating design it refuses while points are incomplete or an
       alpha profile has no plateau. It cannot execute them.
   * - ``report``
     - Generate the DOE report and SVG/PNG/PDF figures without mechanics.
   * - ``prepare-transfer-validation``
     - Freeze at most three candidates in a no-recalibration transfer
       manifest. It cannot execute them.

``--dry-run`` is supported by calculation and manifest actions.
``--workers`` limits F1 workers and defaults to one. Caches are reused only
when their complete physical, numerical, observation and Git fingerprints
match. A point failure receives an individual status; the F1 design continues
and the command returns status 2.

The homogeneous P43 configuration additionally saves converged ``U``, ``E``,
and ``PEEQ`` snapshots at configured load fractions. Snapshot names, hashes and
fractions belong to the immutable point manifest; missing or altered
snapshots prevent cache reuse.

See :doc:`../how-to/run_joint_nonlocal_identification` for the complete P43
sequence and :doc:`../explanation/joint_nonlocal_identification` for the
scientific limits of F0, F1 and F2.

``plot-coupled-alpha-fields`` contract
--------------------------------------

This command compares one local campaign with exactly three coupled
campaigns. It reconstructs the historical total
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

The generic repeated form supports other pre-registered alpha sets. For
example, the P43 sweep uses:

.. code-block:: text

   fem-inhouse plot-coupled-alpha-fields \
     --input data/processed/case_study \
     --local-campaign results/constitutive-local-p0043-pad150 \
     --coupled-campaign 1 results/constitutive-nonlocal-p0043-pad150-a100 \
     --coupled-campaign 2 results/constitutive-nonlocal-p0043-pad150-a200 \
     --coupled-campaign 4 results/constitutive-nonlocal-p0043-pad150-a400 \
     --partition-id 43 \
     --output validation/figures/p0043-alpha-comparison

The legacy ``--campaign-a050``, ``--campaign-a100``, and
``--campaign-a200`` options remain supported. They must not be mixed with
``--coupled-campaign``.

The command accepts ``--dpi``, ``--format {png,pdf,svg}``, the three robust
colour-limit percentile options, ``--include-optional-fields``, and
``--overwrite``. PNG, PDF, and SVG are generated by default. The primary EVM
figures always use raw converged FEM displacements: no Helmholtz filter is
applied. ``PEEQ`` is an internal plasticity variable and is shown only as a
mechanism diagnostic, not as an experimental PEEQ comparison. The output
``plot_metadata.json`` records campaign hashes, core bounds, colour limits,
metrics, alpha/Hchi values, plotting options, and the explicit no-post-filter
contract.

``select-dic-partition`` contract
---------------------------------

Before calibrating a non-local parameter, rank candidate ROIs using only the
DIC kinematics:

.. code-block:: text

   fem-inhouse select-dic-partition \
     --input data/processed/case_study \
     --output validation/dic_partition_heterogeneity_10x10.json \
     --parts-x 10 --parts-y 10 --padding 150

The command reconstructs ``EVM_HISTORICAL`` from DIC displacements and ranks
all partitions from the dominant coherent q85 high-strain component. The
primary score combines its aspect ratio, occupied area, contrast, and boundary
contacts after light denoising and binary cleanup. It also records kurtosis,
coefficient of variation, quantile-tail contrast, gradient RMS, and extrema as
secondary diagnostics. This is a prospective ROI-selection diagnostic, not a
material parameter fit; the selected maps must still be inspected before any
costly FEM campaign.

Exit behaviour
--------------

The CLI returns a non-zero status for invalid contracts, unavailable required
backends, incompatible manifests, non-convergence, corrupt outputs, and failed
comparison thresholds. It does not convert these conditions into warnings.
