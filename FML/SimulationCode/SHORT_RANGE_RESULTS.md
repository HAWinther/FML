# Initial common-timestep short-range validation

Recorded 5 November 2026. This covers the first ordinary N-body milestone of #47,
not COLA, subcycling or production nonlinear calibration. See [the contract](SHORT_RANGE.md).

## Force adapter and actual PM assembly

The focused suite passes 15 checks under Apple Clang, GCC12, ASan/UBSan, and
OpenMPI/Clang17 at 1/2/4 ranks, including OpenMP, FFT bounds checking and manual
`NO_AUTO_FML_SETUP` initialization. The sparse four-rank fixture includes an empty
owned-particle rank. Cell and exact-tree adapters agree with a replicated direct
reference after independently normalizing by global mass. The replicated source
list exists only in the small test oracle, not the production adapter.

Actual PCS deposition, Fourier force differentiation, interpolation and the short
correction are compared to periodic Ewald image/Fourier sums with an unrelated
reference split. For the seven-point unequal-mass off-grid clustered fixture,
relative RMS force errors are:

| Force calculation | Relative RMS error |
|---|---:|
| Unsplit PM, 16³ | 0.589184 |
| Combined, 16³ | 0.00136265 |
| Combined, 32³ | 0.0000603862 |
| Combined, 64³ | 0.00000196315 |
| Combined with two-shift interlacing, 32³ | 0.00000176725 |

The comparison uses split scale 0.04 and spline support 0.002 in unit-box coordinates,
cutoff tolerance 1e-10 and analytic pair coefficients. All particle separations in
this particular Ewald fixture exceed the softening support. The force library's
separate tests cover spline interiors and softened periodic references. The new
adapter also checks physical-length rescaling, mass-weighted momentum balance,
disabled/zero kicks, kick sign and preservation of pre-existing velocity scaling.
The mesh error must fall by at least a factor five and reach below 0.005; the double
momentum threshold is 1e-9. These are controlled fixture criteria, not generic force
accuracy guarantees for a cosmological catalogue.

## The real SimulationCode step

The diagnostic driver isolates one actual `NBodySimulation::step` at a=0.5 with
velocity_scale=0.7, kick integral 0.001 and no drift. Seven equal-mass particles on
a 64³ force mesh are checked against the independent Ewald reference. Relative RMS
errors inferred from the complete velocity update are 6.4353e-5 without interlacing
and 2.61761e-6 with two shifts (limit 0.002). Positions remain exactly unchanged.
This exercises actual placement of the filter/short kick, including its single
application and the GR epoch normalization, independently of the adapter comparison.

## Full evolution and timestep convergence

A 512-particle GR/LCDM run evolves from z=9 to z=0 on a 16³ force mesh with Quinn KDK,
PCS assignment, fixed seed and sigma8=0.1 normalization. Full-precision states are
saved by ID, so snapshot float rounding is not part of the comparison. RMS unit-box
position errors against 64 steps are:

| Steps | RMS position difference |
|---|---:|
| 4 | 2.14815e-4 |
| 8 | 7.68772e-5 |
| 16 | 2.14970e-5 |
| 32 | 4.54249e-6 |

The two finer halvings must each improve by more than 1/0.35, testing behavior beyond
simple first-order error reduction on this case. These results are consistent with
second-order refinement toward the finite-step reference. They do not establish
halo/orbit convergence or accuracy of this deliberately small mesh/particle setup.

Exact-tree and cell-list evolutions agree to about 9e-19 in the saved state components.
A vanishingly small split with no relevant short neighbors matches the PM-only run
to about 1.5e-15. Two/four-shift interlaced runs complete. MPI 1/2/4-rank states
agree to about 2e-16 maximum absolute difference over unit-box positions and code
velocities. Interlaced MPI runs and isolated kicks also pass. Lightcone runs exercise
the original unsplit potential path with/without interlacing in serial and MPI;
they are smoke tests, not a reference calibration of relativistic observables.

Unsupported COLA, non-continuous gradients, sheet density, FastPM timesteps,
modified gravity and finite-difference forces are rejected. The modified-gravity
negative case provides a valid mu-model configuration, so it tests the new scope
check rather than failing because of missing model parameters.

## Remaining validation

The default PM path and existing model/COLA/lightcone tests are compatibility
checks; no COLA short-range implementation is claimed. Production halo/orbit and
power-spectrum convergence, float FFT calibration, larger clustered rank/thread
workloads, finite-opening tree errors/momentum drift and peak workspace/RSS remain.
No production speedup is claimed: the purpose is a usable higher-resolution force
workflow, with added short-range work. Subcycling and its cosmological time/frame
factors are the next integration stage, after common-timestep calibration.
