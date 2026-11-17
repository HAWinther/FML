# Survey validation

Build with `make`. A bounded replay can be run with
`python3 run.py --bin-dir "$PWD" --output /path/to/fresh/smoke --smoke`.
This checks Cartesian Gaussian/Poisson targets; it is not a coverage claim.
MPI/pencil builds accept `USE_MPI=true USE_PENCIL_GRID=true`; see `run.py --help`
for MPI binaries, launcher and rank controls. Full calibration runs are opt-in.

[Survey model contracts](../../docs/SURVEY.md),
[historical results](../../docs/VALIDATION.md), and
[archived data](../../docs/ARCHIVE.md).
Keep generated run output outside the library source tree.
