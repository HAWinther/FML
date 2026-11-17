# Small PM HMC validation

Build with `make`; run the dependency-free smoke checks with
`python3 smoke.py --exe ./pm_hmc --mpi false`. The library test target
`pm-hmc-smoke` calls this driver. MPI options are `--mpi true --mpirun "..."`
and `--np-list "1 2 4"`; pencil builds use `USE_MPI=true USE_PENCIL_GRID=true`.

The optional full experiment is in run.py; inspect `--help` and use an external
output directory. [Historical results](../../../docs/VALIDATION.md) and their
[raw archive](../../../docs/ARCHIVE.md) are separate from the maintained source.
