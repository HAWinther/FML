#!/bin/bash
#==========================================================================
# Run the accuracy tests in this folder (LCDM_*): each one is a 128^3 simulation
# whose P(k) at z = 0 is compared to MG-PICOLA run with the same random numbers
# and the same time-stepping (see README.md)
#
# Usage: ./run_accuracy_tests.sh <path/to/nbody> <outdir> [test ...]
#   MPIRUN : how to launch (default "mpirun -np 2"; MPIRUN="" for a serial build)
#   TOL    : max allowed relative difference in P(k) (default 0.002)
#
# We compare the P(k) the code computes every step (CIC on the force grid, window
# deconvolved), which is the estimator MG-PICOLA was modified to use for these tests
#==========================================================================
set -u
here="$(cd "$(dirname "$0")" && pwd)"
exe="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
outdir="$2"
shift 2
tests="${*:-$(cd "$here" && ls -d LCDM_*/ | tr -d /)}"
MPIRUN="${MPIRUN-mpirun -np 2}"
TOL="${TOL-0.002}"

mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
failed=""
for t in $tests; do
    work="$outdir/$t"
    mkdir -p "$work"
    # Use the test's parameterfile, but read the input from the test folder and skip particle output
    cp "$here/$t/parameterfile.lua" "$work/parameterfile.lua"
    printf '\n-- Added by run_accuracy_tests.sh\nic_input_filename = "%s"\noutput_particles = false\n' \
        "$here/$t/input_power_spectrum.dat" >> "$work/parameterfile.lua"
    printf "%-48s" "$t"
    start=$(date +%s)
    if ! (cd "$work" && $MPIRUN "$exe" parameterfile.lua > log.txt 2>&1); then
        echo "FAILED to run (see $work/log.txt)"
        failed="$failed $t"
        continue
    fi
    printf "ran in %3d s   " "$(( $(date +%s) - start ))"
    if ! python3 "$here/compare_pofk.py" "$here/$t/mgpicola_pofk_z0.000.txt" \
        "$work/output/pofk_TestSim_cb_z0.000.txt" --tol "$TOL"; then
        failed="$failed $t"
    fi
done
[ -z "$failed" ] || { echo "Failed:$failed"; exit 1; }
echo "All accuracy tests passed"
