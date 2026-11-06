# Common-timestep COLA with short-range forces

`force_short_range=true` now supports the existing **1LPT/2LPT COLA frames**,
including scale-dependent frames. The force ordering was already implemented;
this change validates it and removes the blanket COLA rejection. No force,
frame-update or kick/drift formula was changed.

## Supported combinations

| Evolution | Short-range timestep methods |
|---|---|
| Ordinary single-cold-species N-body | Quinn |
| COLA, order 1 or 2 | Quinn or Tassev |

Scale-dependent COLA retains the existing model growth and source conventions.
Screening/exact nonlinear solvers, massive neutrinos/separate species, enforced
LPT trajectories, sheet/NUFFT forces, early force switches and noncontinuous
Green/gradient kernels remain excluded. FastPM/BullFrog remain excluded with the
short-range option. `simulation_cola_LPT_order=0` disables COLA: Quinn then reduces
to ordinary N-body, while Tassev remains excluded from that combination.

Start with the lengths/backend settings in [SHORT_RANGE.md](SHORT_RANGE.md), and
select the existing COLA options:

```lua
simulation_use_cola = true
simulation_cola_LPT_order = 2         -- or 1; ic_LPT_order must be >= this
simulation_use_scaledependent_cola = false -- true for a scale-dependent frame
simulation_enforce_LPT_trajectories = false
timestep_method = "Quinn"            -- or "Tassev" with COLA
force_greens_function_kernel = "continuous"
force_gradient_kernel = "continuous"
```

The default remains PM-only when `force_short_range=false`. No new COLA parameters
are introduced. A model without scale-dependent growth retains the existing
normalization of `simulation_use_scaledependent_cola` back to ordinary COLA.

## Why the existing ordering works

COLA stores residual velocity u relative to its LPT trajectory. With the same
kick coefficient for both gravity pieces, the existing sequence is

```
u' = velocity_scale*u + dt_kick*(A_long + A_short - A_LPT)
x' = x + dt_drift*u' + delta_x_LPT.
```

Here accelerations are -grad Phi; the mesh API's +grad Phi is applied with its
existing minus sign. The ordinary mesh kick scales the residual velocity once;
the unshifted particle kick adds the short acceleration once; the existing COLA
update subtracts the full LPT-frame acceleration once and supplies the LPT drift.
The frame potentials are not Gaussian filtered. Interlacing averages only the
mesh force and does not multiply the particle kick or frame correction.

The lightcone path splits the existing frame kick/drift around a temporary
restoration of physical velocities. It retains the unsplit PM potential estimate;
this feature does not add a particle-pair lightcone potential or new relativistic
modified-gravity accuracy.

## Manufactured one-step reference

The diagnostic driver uses the default fiducial particle type, with mass, 1LPT/2LPT
fields and q storage. Seven off-grid unequal-mass sources are placed independently
of prescribed nonzero frames. Tests cover combined kick/drift, kick-only and
drift-only calls, a nonunit residual-velocity scale of 0.7, frame orders 1/2,
interlacing, and MPI ownership changes from moving to q and back.

The full body-force reference is a separately summed periodic Newtonian Ewald
force plus converged Yukawa image sums for finite-range models. All pair
separations lie outside the spline support; this is not a new independent check
of softened Yukawa forces. GR uses a manufactured EdS background and independent
closed frame factors

```
delta_x_1 = (a_new-a)/a_ini * Psi_1
F_frame,1 = 3/2 * a^2/a_ini * Psi_1
delta_x_2 = (a_new^2-a^2)/a_ini^2 * Psi_2
F_frame,2 = 5 * a^3/a_ini^2 * Psi_2.
```

For DGP/f(R), frame acceleration is instead checked through a five-point derivative
`a^3 E(a) d[v_LPT(a)]/da` of the public model growth/velocity functions, rather than
calling the production COLA kick/source-factor routines. This tests consistency
with the model's existing LPT approximation; it is not an independent derivation
of nonlinear modified-gravity LPT.

Scale-dependent tests prescribe Fourier potentials with one x mode for 1LPT and
one y mode for 2LPT. q values are grid nodes, the potentials carry the interpolation
halos, and initial stored displacement fields contain deliberately different
sentinels. This checks recomputation from the potentials, frame signs/normalization,
q restoration, IDs/masses and physical-velocity restoration. The synthetic step
coefficients isolate assembly; actual timestep coefficients are tested separately.

## End-to-end checks and measured accuracy

Seeded 8³-particle runs use a 64 Mpc/h box, IC mesh 16³, force mesh 32³, PCS,
deconvolution, sigma8=0.1 and a=0.1..1. Both methods are tested with GR 1/2LPT,
DGP 2LPT, and f(R) ordinary 2LPT plus scale-dependent 1/2LPT. Initial true positions
match their N-body counterparts exactly; true velocity RMS differences are below
4.7e-12 in code units. Final comparisons restore the full LPT velocity, rather
than comparing COLA residual velocities to N-body velocities. Physical here means
frame-restored full velocity; the saved values remain canonical code units, not km/s.

Four step counts (4/8/16/32) test self-refinement and compare to 128-step ordinary
N-body runs using the same force. The 64-to-128-step N-body differences are up to
1.54e-6 box units in position RMS and 9.77e-7 in canonical velocity RMS; these
references have finite timestep error. COLA and N-body are not expected to agree
bitwise at finite resolution, nor does this establish a force-error bound from
those state differences.

| Frame | Quinn: position / velocity RMS vs N-body128 | Tassev: position / velocity RMS vs N-body128 |
|---|---:|---:|
| GR1 | 1.32e-06 / 6.15e-07 | 6.49e-06 / 2.98e-06 |
| GR2 | 1.33e-06 / 6.28e-07 | 6.49e-06 / 2.99e-06 |
| DGP2 | 1.41e-06 / 6.88e-07 | 6.84e-06 / 3.33e-06 |
| fR2 | 1.17e-06 / 6.46e-07 | 5.6e-06 / 2.83e-06 |
| fRscale1 | 1.36e-06 / 6.62e-07 | 7.61e-06 / 3.97e-06 |
| fRscale2 | 1.38e-06 / 6.78e-07 | 7.63e-06 / 4.01e-06 |

RMS values are per coordinate: positions in box units and velocities in the
library's canonical code units. The mild-load comparisons are validation cases,
not evidence for a universally preferable timestep method or a production speedup.

The suite also checks exact tree/cell and default-table/analytic kernels,
interlacing, the order-zero Quinn limit, lightcone-on/off state equality, rejected
unsupported settings and serial/MPI 1/2/4 consistency. The manufactured-frame
maximum residual-velocity error is below 1.77e-8 and position error below 1e-10.
Supported end-to-end MPI states agree below 1.23e-17 RMS; lightcone-on/off differences
are below 6.5e-19. These near-roundoff comparisons are between matching numerical
routes, not absolute errors relative to an exact cosmological solution.

## Reproduction

```sh
cd FML/SimulationCode/tests
make test_short_range_simulation USE_MPI=false
python3 regression/short_range_cola.py --binary ./test_short_range_simulation \
    --output /tmp/fml-short-cola-serial

# Clean/rebuild before changing compiler, MPI or precision.
make clean
make test_short_range_simulation USE_MPI=true
python3 regression/short_range_cola.py --binary ./test_short_range_simulation \
    --output /tmp/fml-short-cola-mpi --mpi --launcher mpirun
```

CMake users explicitly build target `test_short_range_simulation`; the longer
Python suite is outside default CTest runs. The runner uses the Python standard
library and retains full states, Lua inputs, logs, convergence CSVs and
source/binary/parameter hashes. The driver modes are `frame-check` for the oracle,
`true-state` for physical-velocity output, and the existing `single-kick`/default
run modes. Default raw output remains residual velocity when COLA is active;
physical comparisons must select `true-state`.

The serial COLA workflow has 105 executions and the MPI workflow 117. The
ordinary N-body short-range regression also passes in serial/MPI, and existing
COLA-frame/model unit tests pass. Production `nbody` smoke runs exercise GR/Tassev
and scale-dependent f(R)/Quinn. Build options, raw evidence and the review patch
are retained with the exported validation artifacts. Sanitized manufactured-frame
smoke runs pass all nine checks for GR and scale-dependent f(R) (new driver/header
code instrumented, linked background objects prebuilt without instrumentation).
Unsupported order-zero Tassev settings also reject promptly on 2/4 ranks.

Short-range subcycling is still absent: all these cases use one common force
clock. Nonlinear/halo calibration, finite-opening tree drift in COLA, observational
convergence and model-specific LPT theory limits remain application-dependent
follow-ups. The close-orbit timestep constraints in [DYNAMICS.md](DYNAMICS.md)
remain relevant with COLA; a coordinate-frame split does not resolve close
encounters automatically.
