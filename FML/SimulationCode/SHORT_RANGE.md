# Common-timestep PM plus short-range gravity

Backlog #47 provides opt-in periodic unscreened forces in 3D N-body and COLA
evolution, with a common timestep for the long and short forces. Ordinary N-body
uses Quinn; COLA supports Quinn and Tassev with its existing 1LPT/2LPT frames.
Short-range subcycling remains separate work. The default is PM-only evolution.

The implementation reuses [ShortRangeGravity](../ShortRangeGravity/README.md):
its Gaussian/spline force split, cell-list P3M or monopole-tree backend, and selective
multi-slab MPI ghosts. No particle-payload all-gather is added to the simulation.

## Configuration

Starting from a valid unscreened N-body parameter file, add:

```lua
simulation_use_cola = false
simulation_enforce_LPT_trajectories = false
timestep_method = "Quinn"
force_greens_function_kernel = "continuous"
force_gradient_kernel = "continuous"
force_density_method = "assignment"
force_interpolation_method = "assignment"
force_early_method = "none"

force_short_range = true
force_short_range_backend = "cell"       -- or "tree"
force_short_range_split_scale = 3.125     -- comoving Mpc/h
force_short_range_softening_support = 0.2 -- spline support h, comoving Mpc/h
force_short_range_cutoff_tolerance = 1e-6
force_short_range_table_points = 16384    -- 0 evaluates the analytic pair kernel
force_short_range_opening_angle = 0.3     -- tree only; 0 gives exact leaf sums
```

The lengths above illustrate a 100 Mpc/h box with a 64³ force mesh: the split is
two mesh cells. They are not calibrated production recommendations. Select the split,
softening, cutoff, mesh and time resolution together and measure convergence.
The support h is **2.8 times the Plummer-equivalent epsilon**. Both lengths are
fixed comoving values in this first implementation. The library converts them to
unit-box coordinates explicitly. The resulting cutoff must be smaller than L/2,
and h must not exceed it. Kernel settings are validated before simulation setup.

`force_short_range=false` requires no new lengths and follows the PM-only path.
With the option enabled, the supported scope requires:

- 3D unscreened gravity, with screening and exact nonlinear solvers disabled; no massive neutrinos, separate baryon/neutrino particles or N-body gauge
  source; the single cold species can carry finite nonnegative particle masses;
- ordinary N-body or 1LPT/2LPT COLA (including scale-dependent frames), without enforced LPT trajectories;
- Quinn timesteps, or Tassev with COLA, and continuous Fourier Laplacian/gradient kernels; select these
  explicitly because the usual fiducial gradient can switch to a discrete stencil;
- assignment-based density and interpolation; no sheet, NUFFT or early-time switch;
- stable globally unique particle IDs. Standard generated ICs already assign these.

Force interlacing and window deconvolution remain available. Interlacing order and
assignment choices keep their existing validation. Small-scale aliasing and particle
sampling effects are not removed by the analytic split; resolve the Gaussian long
force and test assignment/interpolation convergence. The initial validation used
double FFT grids; float-FFT production calibration remains work to do.

For COLA configuration, frame ordering and validation, see
[COLA_SHORT_RANGE.md](COLA_SHORT_RANGE.md).

## Force assembly and time factors

Let x be the unit-box position, C(a)=3 OmegaM a/2, and M the global sum of the source
particle masses. The PM force API supplies **+grad_x Phi**, while the short-range
API supplies an acceleration **-grad_x Phi**. The update is

```
v_new = velocity_scale * v_old - delta_time_kick * F_long
                                + delta_time_kick * A_short.
```

In GR, the long force uses a density filtered by `exp[-k_box² (split_scale/L)²]`.
The model owns both the matched mesh solve and the particle kick (see below).
The short kernel uses normalization `C(a)/(4*pi*M)` multiplied by the model coupling,
with unit-box positions/lengths. The
actual mass sum is reduced globally, matching mass-normalized mesh deposition;
neither a rank's local count nor its owned-plus-ghost mass is used as M.

The existing kick integral is applied once, with no additional scale-factor or H0
factor. The short update does not rescale velocity again. There is no extra drift
or independent short-force clock. Ghosts and the cell/tree search are rebuilt at
each nonzero force kick after ownership migration. The `ShortRangeForce` timer
reports this work separately from the PM solve.

For force interlacing, each shifted realization contributes only to the averaged
long force. Positions are restored exactly, the existing velocity scaling/PM kick
is applied, and **one** unshifted short kick follows before drifting. The short
correction is not added separately to every shifted realization.

The original lightcone potential is calculated from the **unfiltered** density,
before the long-force filter; interlaced runs build the original unshifted density
for that potential. Thus lightcone output retains its unsplit PM potential estimate.
It does not acquire a particle-pair small-scale potential or TreePM-resolution
relativistic corrections. No claim of such potential accuracy is made here.

## GravityModel extension

`GravityModel::GeffOverG_highk(a)` defaults to the existing
`GeffOverG(a, 1e12)`, with k in H0/c units. A new model implementing only
`GeffOverG` therefore inherits the short-range implementation. Models with known
asymptotes override this method. This finite fiducial is an approximation for a
new model, and should be checked against its physical high-k behavior.

`short_range_comoving_range(a)` defaults to **0, denoting infinite range**.
Override it to supply a comoving Mpc/h Yukawa range. f(R), Symmetron and m,beta
convert their physical inverse mass to this convention, including the factor 1/a.
DGP, JBD and GR use infinite range. The mu BZ model uses its analytic comoving
range; scale-dependent tables retain the generic default.

The model describes a pair potential outside the spline support as

```
Phi_pair = -G M/r [mu0 + (muinf-mu0) exp(-r/lambda)]
mu0 = GeffOverG(a,0), muinf = GeffOverG_highk(a).
```

The force enhancement is `mu0+(muinf-mu0)*(1+r/lambda)*exp(-r/lambda)`.
For f(R), this retains the overall `1/(1+fR)` prefactor and the relative 1/3
Yukawa amplitude. Infinite range gives a Newtonian pair kernel of strength muinf.
The finite-range kernel requires positive mu0; the infinite-range kernel requires
nonnegative muinf. Invalid or inconsistent model descriptors fail collectively.

`compute_force_for_step` uses the matched long filter from ShortRangeGravity,
which includes the Yukawa mass-dependent Gaussian factor. It also retains a
resolved Fourier remainder:

```
mu_mesh(k) = GeffOverG(a,k) - mu_pair(k) + mu_pair_long(k).
```

Thus future models with arbitrary scale dependence retain their mesh response,
even when inheriting the infinite-range pair surrogate. For the analytic Yukawa
models the remainder vanishes. The approximation below the mesh scale remains
the chosen surrogate; this is not an exact real-space representation of an
arbitrary Geff. Override the limit/range for an appropriate physical model.
No extra Geff factor is applied after constructing this mesh response.
The existing virtual force solver remains in use when short-range forces are off.

For screened models set `gravity_model_screening=false` and the model's
`gravity_model_*_exact_solution=false`. Nonlinear screening is not represented by
a density-independent pair kernel and remains explicitly rejected.

## Validation and reproduction

The existing SimulationCode build includes the feature. The diagnostic driver runs
the real `NBodySimulation` pipeline and saves full-precision final states by ID:

```
cd FML/SimulationCode/tests
make test_short_range_force test_gravity_short_range test_short_range_simulation USE_MPI=false
./test_short_range_force
./test_gravity_short_range
python3 regression/short_range.py --binary ./test_short_range_simulation \
    --output /tmp/fml-short-range-serial

# Clean and rebuild before changing MPI/compiler/precision options.
make clean
make test_short_range_force test_gravity_short_range test_short_range_simulation USE_MPI=true
mpirun -np 4 ./test_short_range_force
mpirun -np 4 ./test_gravity_short_range
python3 regression/short_range.py --binary ./test_short_range_simulation \
    --output /tmp/fml-short-range-mpi --mpi --launcher "mpirun"
```

The force test covers unequal masses, physical/unit-box conversions, MPI empty
ranks, both exact backends, disabled/zero kicks, non-unit velocity handling and
collective invalid input. Actual deposition/interpolation and the short correction
are compared to independent periodic Ewald image/Fourier sums on 16³/32³/64³ meshes.
It also checks momentum balance and the gain over unsplit PM on a clustered fixture.

The end-to-end script checks timestep convergence against 64 steps; exact tree/cell
agreement; interlacing; the tiny-split PM-only limit; rejected unsupported modes;
and MPI state agreement. Isolated calls of the **actual simulation step** compare
the complete kick to Ewald forces with a non-unit velocity scale, independently
checking that long filtering and the short contribution occur once. Lightcone runs
exercise the unsplit potential path with/without interlacing. See the exported
review and [SHORT_RANGE_RESULTS.md](SHORT_RANGE_RESULTS.md) for numerical results.

## Clustered static-force validation

An optional larger experiment now covers concentrated and overlapping halos, a
periodic-boundary halo, and mesh-relative translations. It uses independent
softened Ewald references, samples error distributions and radial dependence,
and measures exact/finite-tree momentum balance in serial and MPI. See
[CLUSTERED_FORCE.md](CLUSTERED_FORCE.md) for scope, results and reproduction.
It does not yet validate time evolution of these clustered fixtures.

## Controlled orbit and cluster evolution

A second optional driver evolves near-circular/eccentric pairs and compact
16-particle cluster/merger fixtures through the actual simulation step, comparing
synchronized states to independent adaptive direct Ewald evolution. It checks
reference refinement, second-order timestep convergence, exact tree/cell and
interlaced trajectories, canonical momentum and MPI migration. See
[DYNAMICS.md](DYNAMICS.md) for units, results, reproduction and validation limits.

## Limits and remaining #47 work

This is a common-timestep implementation, not completion of the full item.
Short-range subcycling, broader nonlinear COLA calibration, screened/multispecies force
prescriptions, particle-pair lightcone potentials and production halo/orbit,
power-spectrum and memory/performance calibration remain. The tree's finite opening
can introduce asymmetric force errors; validate its force-error distribution,
momentum drift and rank dependence against exact/cell results. The cutoff controls
an individual pair window, not relative error in a cancelled net force.

Short-range forces can require much smaller timesteps than PM alone. This initial
milestone adds no automatic acceleration/softening timestep controller: increase
`timestep_nsteps` and test the observables you use. MPI ghosts may be large for
narrow slabs or a large cutoff. Per-kick owned/ghost/search buffers are additional
workspace; no total-memory reduction or production speedup is claimed. These tests
are controlled verification cases, not a claim of calibrated nonlinear cosmology.
