#!/bin/bash
# Build and run the tests of every module that has its own tests directory (FML/*/tests, FML/*/*/tests), each on
# 1, 2 and 4 MPI tasks where the module uses MPI. FML/tests (make -C FML test) and SimulationCode/tests are run
# separately. Usage: run_module_tests.sh [directory to skip ...]   e.g. Cosmology/EinsteinBoltzmannSolver/tests
cd "$(dirname "$0")/../../FML" || exit 1
failed=""
for d in */tests */*/tests; do
  [ -f "$d/Makefile" ] || continue
  skip=false
  for s in SimulationCode/tests "$@"; do
    [ "$d" = "$s" ] && skip=true
  done
  $skip && continue
  echo "::group::$d"
  start=$(date +%s)
  if make -C "$d" -j4 run MPIRUN="${MPIRUN:-mpirun --oversubscribe}" NP_LIST="${NP_LIST:-1 2 4}"; then
    result=passed
  else
    result=FAILED
    failed="$failed $d"
  fi
  echo "::endgroup::"
  echo "$d: $result ($(( $(date +%s) - start )) s)"
done
if [ -n "$failed" ]; then
  echo "Failed module tests:$failed"
  exit 1
fi
echo "All module tests passed"
