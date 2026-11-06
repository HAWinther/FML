# Short-range orbit and cluster evolution (item 47)

This optional test driver compares **actual SimulationCode steps** with an
independent adaptive direct-force evolution. It is controlled small-N GR
validation, not a production halo calibration, equilibrium construction, COLA
implementation or short-range subcycling feature. Production code is unchanged.

## Equations and independent reference

Both evolutions use comoving box coordinates and canonical velocities
`v = a^3 E(a) dx/da`, with the same initialized LCDM background E(a). The direct
reference solves

```
dx/da = v / [a^3 E(a)]
dv/da = A(a,x) / [a^3 E(a)]
A_i = [3 OmegaM a / (8 pi M)] sum_j m_j F_periodic,soft(x_i-x_j).
```

It uses a separate full-state adaptive GSL RKF45 integrator, not the production
Quinn kick/drift coefficients. The direct periodic force comes from
`ClusteredForceReference.h`: independently integrated spline density, Ewald images
and reciprocal structure factors, with no production mesh or short-force kernel
calls. Primary inverse split 3.5/mode bound 8 is compared against 5/10 at every
saved output. Both use images -1..1; the earlier static experiment cross-checks
this reference family against the older pairwise oracle.

Each fixture is integrated twice with absolute/relative ODE tolerances 1e-10 and
1e-12. Their maximum normalized position/velocity differences must be below 1e-6;
spatial force refinement must agree below 1e-10 relative RMS at every output.
Thus reference time and force accuracy are checked separately from the PM solver.
Ordinary mechanical energy conservation is not imposed on this expanding,
explicitly time-dependent system. Canonical total momentum should be constant.

## Fixtures

Box size is 100 comoving Mpc/h, OmegaM=0.3, and the fixed comoving spline support
is 1.25 Mpc/h. Every fixture starts at a=0.5:

| Label | Contents | End a | Default Quinn drift counts |
|---|---|---:|---|
| circular | Two equal masses, separation 0.06 box, approximately circular initial tangential velocities | 0.65 | 32, 64, 128, 256 |
| eccentric | Same pair with tangential speed multiplied by 0.65; center initially at x=0.995 and canonical bulk vx=0.08 | 0.65 | 128, 256, 512, 1024 |
| cluster | Sixteen unequal masses with deterministic shell positions and tangent velocities | 0.54 | 32, 64, 128, 256 |
| merger | Two approaching eight-particle clusters, initial center separation 0.08 box | 0.53 | 32, 64, 128, 256 |

These are prescribed finite-N states, not distribution-function samples or
virialized halo models. The near-circular label does not imply an exact Kepler
orbit in the expanding periodic background. The eccentric case crosses the
periodic boundary and MPI ownership boundaries. The cluster fixtures include
softened close encounters. Fixtures and masses are identical in both evolutions.

## Pipeline and measurements

The driver initializes a real `NBodySimulation` and replaces its temporary IC
catalogue with the controlled fixture. It then calls the actual `step` with
coefficients from `simulation.time_stepping().steps`. Normal assignment, force
interlacing, model-owned short kicks, cosmological drifts and particle migration
all execute. This does not test IC accuracy, the ordinary `run()` output scheduler,
lightcone analysis or snapshot IO.

Sixteen output windows end with kick-only velocity synchronization. Positions
and velocities are compared at the **same a**, rather than mixing full-step
positions with staggered velocities. Splitting uniform KDK steps this way adds
repeated boundary force evaluations but leaves the underlying drift grid intact.
Positions are unwrapped continuously through periodic crossings; each drift is
required to be shorter than a quarter box to make this unambiguous. Saved IDs
are checked for uniqueness and complete ownership.

The CSVs retain every particle's prescribed mass, unwrapped position, canonical
velocity and owning rank at all 17 output times. Summary/series files report:

- Maximum mass-weighted RMS position error, divided by the initial RMS radius.
- Maximum mass-weighted RMS velocity error, divided by the initial velocity dispersion.
- Center-of-mass error and canonical momentum change, with those same initial scales.
- Pair separation and orbital phase difference modulo 2*pi at the saved outputs.
- Cluster RMS radius and velocity dispersion, with direct-reference values.

A second-order convergence window is required: at least one adjacent refinement
must reduce each error by a factor between 2.5 and 6, and the finest error must be
less than one fifth of the coarsest. Finest position/velocity budgets are 0.005/0.01
of their **initial scales**. These are not errors relative to instantaneous speed
at eccentric pericenter, nor percentages of the box length. Momentum change must
remain below 1e-9 of the initial velocity-dispersion scale.

Additional runs check exact tree/cell trajectory agreement, interlacing, a 32³
mesh, and a PM-only control. The control keeps the same continuous gradient and
deconvolution as the split solve; it is not a tuned or universal PM baseline.
Compact point catalogues can be especially sensitive to this unfiltered force
choice. A separately recorded fiducial-gradient PM check also remains much less
accurate than the resolved split force on these fixtures. Neither comparison
establishes PM performance on ordinary cosmological particle loads.

The MPI runner repeats the finest cell evolution on 2/4 ranks, compares the full
saved trajectories to one rank, and requires actual ownership migration in the
boundary-crossing pair. The small fixtures and reference are replicated test-only
work; no distributed-memory performance claim follows from these tests.

## Initial results (double precision)

| Fixture | Finest steps | Max. position error / initial radius | Max. velocity error / initial dispersion |
|---|---:|---:|---:|
| circular | 256 | 0.001016 | 0.001638 |
| eccentric | 1024 | 0.001264 | 0.00516 |
| cluster | 256 | 0.0004244 | 0.0009532 |
| merger | 256 | 0.0003777 | 0.004119 |

Refinement ratios approach four in every fixture, consistent with second-order
Quinn evolution. The eccentric fixture at 256 steps still has errors about
0.0200/0.0825 in these position/velocity normalizations; at 1,024 steps these fall
to 0.00126/0.00516. Accurate forces alone do not eliminate the close-orbit timestep
constraint.

The two direct ODE tolerances differ by at most 3.46e-7 in the reported normalized
metrics. Exact tree/cell saved trajectories agree below 6.6e-14; MPI differences
are below 4.5e-16 in these metrics. The boundary pair exercises three ownership
changes on both 2 and 4 ranks. These small tree fixtures use exact traversal;
finite-opening dynamical drift is not calibrated by this experiment.

The full workflow completes 40 serial executions and 48 executions in the MPI
suite (including serial direct references and 1/2/4-rank evolution comparisons).
Sanitized cluster smoke runs pass 16 pipeline synchronization checks and 17
reference force-refinement checks. New driver/header code is instrumented; linked
background objects were prebuilt without sanitizer instrumentation. The explicit
Clang CMake build and manual MPI build pass; the Make target was checked with a
dry run rather than freshly built through Make.

## Reproduction

The longer driver is outside default Make/CTest targets:

```sh
cd FML/SimulationCode/tests
make benchmark_short_range_dynamics USE_MPI=false
python3 regression/short_range_dynamics.py --binary ./benchmark_short_range_dynamics \
    --output /tmp/fml-dynamics-serial --plot

# Clean and rebuild before changing MPI/compiler/precision.
make clean
make benchmark_short_range_dynamics USE_MPI=true
python3 regression/short_range_dynamics.py --binary ./benchmark_short_range_dynamics \
    --output /tmp/fml-dynamics-mpi --mpi --launcher mpirun
```

CMake users explicitly build `benchmark_short_range_dynamics` and pass
`build/tests/benchmark_short_range_dynamics`. The runner otherwise uses only the
Python standard library; `--plot` requires matplotlib. A smaller plumbing run is
`--fixtures circular --steps 32 64 128 --no-variants`. Custom step lists apply to
all selected fixtures and must contain at least three increasing multiples of
16. The gates and mesh are chosen for the defaults; spatial error floors or an
unresolved orbit may require changing both mesh and timestep resolution.

Outputs include exact commands and Lua files, source/binary/parameter hashes,
Git revision/status, full trajectories, convergence ratios, reference refinement,
MPI comparisons and plots. Raw forces and initial conditions are not inferred
from plotted curves. The initial plotting labels were shortened after execution;
the original executed runner snapshot and separate rendering provenance are kept
with the exported results, with non-plot Python code unchanged.

## Remaining item 47 work

This completes the controlled small-N dynamical check following the clustered
static-force experiment. Larger-N halo evolution, finite-opening dynamical
momentum drift, broader force/mesh/timestep calibration and performance/ghost-memory
measurements remain. Common-timestep COLA is now covered in
[COLA_SHORT_RANGE.md](COLA_SHORT_RANGE.md); short-range subcycling remains a separate
implementation milestone. No automatic acceleration-based timestep controller
is introduced here.
