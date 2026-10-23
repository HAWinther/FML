#!/bin/bash
#==========================================================================
# Run the COLASolver regression cases
#
# Usage: ./run.sh <path/to/nbody> <outdir> [case ...]
#   MPIRUN  : how to launch (default "mpirun -np 2"; set MPIRUN="" for a serial build)
#   OMP_NUM_THREADS is set to 1 so that results are reproducible
#
# Then compare two runs with: python3 compare.py <reference_outdir> <outdir>
#==========================================================================
set -u
here="$(cd "$(dirname "$0")" && pwd)"
exe="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
outdir="$2"
shift 2
MPIRUN="${MPIRUN-mpirun -np 2}"
# Cases needing optional features (e.g. hdf5) are only run if the executable has them
features="$($MPIRUN "$exe" --features 2>/dev/null | grep '^FEATURES:' | cut -d: -f2)"
cases="${*:-$(python3 "$here/cases.py" --list $features)}"
export OMP_NUM_THREADS=1

mkdir -p "$outdir"
outdir="$(cd "$outdir" && pwd)"
python3 "$here/cases.py" "$outdir"

failed=""
for c in $cases; do
    printf "%-28s" "$c"
    start=$(date +%s)
    if (cd "$outdir/$c" && $MPIRUN "$exe" parameterfile.lua > log.txt 2>&1); then
        echo "ok   ($(( $(date +%s) - start )) s)"
    else
        echo "FAILED (see $outdir/$c/log.txt)"
        failed="$failed $c"
    fi
done
[ -z "$failed" ] || { echo "Failed:$failed"; exit 1; }
