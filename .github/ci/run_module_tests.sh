#!/bin/bash
# Build and run the tests of every module that has its own tests directory (FML/*/tests, FML/*/*/tests), each on
# 1, 2 and 4 MPI tasks where the module uses MPI. FML/tests (make -C FML test) and SimulationCode/tests are run
# separately. Usage: run_module_tests.sh [directory to skip ...]   e.g. Cosmology/EinsteinBoltzmannSolver/tests
cd "$(dirname "$0")/../../FML" || exit 1
# Oversubscription is an Open MPI option; MPICH/Hydra rejects it.
# An explicit MPIRUN setting takes precedence over launcher detection.
if [ -z "${MPIRUN:-}" ]; then
  MPIRUN=mpirun
  mpi_version=$(mpirun --version 2>/dev/null)
  case "$mpi_version" in
    *"Open MPI"*|*"OpenRTE"*) MPIRUN="mpirun --oversubscribe" ;;
  esac
fi
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
  if make -C "$d" -j4 run MPIRUN="$MPIRUN" NP_LIST="${NP_LIST:-1 2 4}"; then
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
