#!/usr/bin/env bash
# Finish the P43 (ell, alpha) campaign once the mechanics has stopped.
#
# Protocol: validation/p0043_small_parameter_matrix_preregistration.md
#
# Waits for the matrix driver to exit, observes every completed point through
# the symmetric operator, then scores both DISFlow profiles. Each step is
# resumable, so re-running this after an interruption costs only what is
# genuinely missing.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOGS=results/mm-matrix-logs

while pgrep -f run_p0043_parameter_matrix >/dev/null; do sleep 60; done
echo "=== mechanics finished $(date '+%H:%M:%S')"

$PY scripts/replay_p0043_matrix_observations.py 2>&1 | tee "$LOGS/replay.log" | tail -5
echo "=== observations finished $(date '+%H:%M:%S')"

for profile in legacy_script_2021 declared_medium_v4; do
  echo "=== scoring $profile $(date '+%H:%M:%S')"
  $PY scripts/analyse_p0043_parameter_matrix.py --profile "$profile" \
    2>&1 | tee "$LOGS/selection_${profile}.log" | tail -30
done
echo "=== campaign complete $(date '+%H:%M:%S')"
