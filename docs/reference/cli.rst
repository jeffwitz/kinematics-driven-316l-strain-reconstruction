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

Exit behaviour
--------------

The CLI returns a non-zero status for invalid contracts, unavailable required
backends, incompatible manifests, non-convergence, corrupt outputs, and failed
comparison thresholds. It does not convert these conditions into warnings.
