# Bounded Eulerian numerical/resource screen

This maintained driver compares the same frozen synthetic tracer-field data,
physical initial realization, Fourier modes and cutoffs across numerical choices.
It does not download a catalog, calibrate a posterior or establish a posterior
mean-shift budget. The physics/constants of native contract version 1 are fixed
in `Main.cpp` and recorded in `plan.json`; mesh/tolerance choices are CLI arguments.
Change/version both files for a different physical study.

`mesh` is the compact sampled/observation side length, `product` is the padded
LPT/particle/operator side length (zero means shared), and `density` is the
PSWF deposition side length. Product and particle refinements are **coupled** in
the current model; this study cannot claim separate convergence of those roles.
Initial Fourier Gaussian coefficients are keyed by integer wavevector and scaled
by 1/sqrt(mesh^3). This preserves the same physical spectrum when mesh changes,
without using the same real-cell random draw on incompatible meshes.

The reference creates one Gaussian tracer field using a PSWF N24 product/N48
fine grid at epsilon=1e-12. The cases freeze the initial and output bands and
assess alpha=0.9,1,1.1. The fixed-IC conditional target, scalar gradients and
active-field scores are compared. A reference Jacobian gives a local Fisher
score distance and amplitude-shift proxy. Neither metric is a posterior mean
shift, posterior SD or proof of the planned 0.1 posterior-SD scientific gate.
The scaled Fisher rank and eigenvalues are reported; rank deficiency cannot
pass the amplitude proxy. All gates and budgets are in the frozen plan.

Each case is a fresh process. Native output records cold/warm forward timing,
three forward+gradient timings, a two-draw/two-leapfrog HMC smoke cost, and
getrusage peak RSS. MPI reports both maximum per-rank peak and sum of rank peaks;
the latter is not simultaneous job RSS or unique physical memory. MPI timing
uses the slowest rank, with a barrier before each timed call. Serial builds
use no OpenMP; MPI runs use two threads per process. Compare MPI scaling to
its own one-rank build. Three repeats and one RSS measurement are a small
hardware-specific screen, not a statistical speed or production-scaling result.
The RSS peak includes setup, assessment and the HMC block; it is not a sampler-only
or field-storage measurement.

```sh
make -C FML/FieldLevel/validation/galaxy/numerics numerical_screen USE_MPI=false
python3 FML/FieldLevel/validation/galaxy/numerics/run.py \
  --exe /path/to/serial/numerical_screen --output /external/numerical-screen
# Optional separate MPI slab and pencil builds, using the local compiler/FFTW:
# add --mpi-slab /path/to/slab/numerical_screen \
#     --mpi-pencil /path/to/pencil/numerical_screen --mpirun 'mpirun ...'
```

The driver pins the plan, native executable hashes, generated data hash and each
completed output inventory. Raw modes, timings and logs stay outside FieldLevel.
Each numerical case is deterministic, bounded and retained even when its runtime
or numerical gates fail. There is no extension-until-pass policy. Cases exceeding
the predeclared total budgets remain unrun with a reason. `--resume` verifies the
contract/inventory and reuses completed cases without rerunning them. Ctrl-C stops
the current process group; an interrupted incomplete case restarts from its
beginning. The short HMC smoke is not a restartable inference campaign.

The native command is
`generate|assess PREFIX MESH PRODUCT DENSITY EPSILON DATAFILE-or-dash`.
Generation runs serially; assessment reads that frozen data and writes one owned
mode table per state/rank, a scalar Jacobian table and completion metadata. The
Python comparison gathers only this small test's mode tables; the library's
production forward/reverse stay distributed.

[Recorded results](../../../docs/validation/studies/numerical_screen/RESULTS.md)
separate numerical errors, state-coordinate counts, observed total RSS and runtime.
