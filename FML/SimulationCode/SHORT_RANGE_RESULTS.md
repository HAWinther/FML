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


## Model-owned unscreened extension (2026-11-05)

`GravityModel` now provides a generic finite-k coupling fallback and a matched
mesh/pair split. Analytic limits/ranges are supplied for GR, f(R), DGP, JBD,
Symmetron, m,beta and BZ mu. Screening and exact nonlinear solvers remain excluded.

- New model test: 30 checks in serial and MPI (1/2/4 ranks). Includes a model
  implementing only the old Geff interface, a deliberately non-Yukawa response,
  analytic response checks across epochs and k, the f(R) one-third relative
  amplitude, a differentiated Yukawa potential, preserved input density, an
  independent erfc particle kick with unequal masses, and collective failure
  when only rank zero returns an invalid limit.
- Long Fourier coefficients match independent analytic expressions within
  8e-8 absolute error (test tolerance 2e-7). This includes the existing
  `compute_delta_fifth_force` coupling-spline approximation; it is not a
  machine-precision claim for that interpolation.
- Explicit Clang CMake build and CTest run of the new model suite pass. The
  default local GCC CMake build encounters a macOS stdio/FILE header error in
  unchanged Spline.cpp, before reaching this extension.
- Existing 15-check force suite passes in serial, address/undefined sanitizers,
  and MPI 1/2/4. Existing model, COLA-frame, parameter and lightcone tests pass.
- Complete serial/MPI integration script passes, including unscreened f(R),
  DGP, Symmetron, m,beta and BZ mu. Exact tree/cell state differences are at most
  1.2e-16 in serial; MPI/backend comparisons are at most 3.4e-16. Screening and
  exact nonlinear-solver configurations are rejected.
- GR timestep errors against 64 steps remain 2.14815e-4, 7.68772e-5,
  2.14970e-5 and 4.54248e-6 for 4/8/16/32 steps. The tiny-split PM limit differs
  by at most 1.45e-15. The saved PM-only baseline is bitwise unchanged; the
  enabled GR eight-step state shifts by at most 3.96e-9 from the previous
  implementation because the matched filter now uses the coupling spline.
- Real-step isolated Ewald checks and unsplit lightcone potential checks pass
  with and without interlacing in serial and MPI.

These checks verify the implementation and its linear kernels. They do not
calibrate nonlinear modified-gravity observables or justify a generic surrogate
as an exact pair force for arbitrary new theories. Such models preserve their
resolved Fourier remainder but need physical small-scale validation.


## Clustered static-force experiment (2026-11-06)

Added an optional reproducible experiment with 2,048 unequal-mass sources and
128 sampled targets per fixture: concentrated halo, overlapping halos, a periodic
boundary halo, and a globally translated copy. Twelve cases vary mesh, split,
softening, interlacing, tree opening and kernel-table use. It calls the model-owned
mesh solve and kick; no production force behavior changes.

The independent oracle combines direct Ewald images and reciprocal structure
factors with a separate quadrature of the compact spline density. Inverse splits
5/7 with Fourier bounds 10/16 agree within 2.0e-14 relative RMS. A separate older
pairwise oracle checks one target per fixture. Sampled targets include 172–361
directed pairs inside the baseline spline support.

All 190 full-size checks pass in serial and MPI 1/2/4 ranks. An address/undefined-
sanitized 128-source/16-target smoke run passes 138 checks (new driver/header
code instrumented, prebuilt background objects uninstrumented). Per-case sampled
MPI force differences are below 2.4e-15 relative RMS. Cell RMS errors decrease
from 2.85e-4–3.90e-4 on 16³ meshes to 5.72e-7–6.98e-7 on 64³ meshes; 64³
99th-percentile errors are below 1e-5. Exact backends retain normalized
mass-weighted momentum balance near roundoff.

Angle 0.6 genuinely exercises tree approximation: RMS errors are
2.44e-4–3.05e-4 and normalized instantaneous momentum imbalance reaches 1.05e-5.
Angle 0.3 still traverses to exact leaf sums on these particular catalogues;
that agreement is not evidence of a tree speedup. The figures retain radial
error distributions and absolute errors normalized by reference RMS, alongside
relative errors with an explicit denominator floor near cancellations.

See [CLUSTERED_FORCE.md](CLUSTERED_FORCE.md) for reproduction and the complete
contract. These are static GR tests; orbit/cluster time evolution, cosmological
timestep convergence, modified-gravity reference forces and production
performance/ghost-memory calibration remain open.


## Controlled orbit and cluster evolution (2026-11-06)

Added a second optional driver using actual SimulationCode steps and a separate
full-state adaptive RKF45 reference with direct periodic softened Ewald forces.
Near-circular and boundary-crossing eccentric pairs, a compact unequal-mass
16-particle cluster and two approaching eight-particle clusters all pass the
serial and MPI 1/2/4-rank workflow. Reference spatial-force refinement is checked
at every output; ODE tolerances 1e-10/1e-12 differ by less than 3.46e-7 in the
normalized trajectory metrics. Position/velocity outputs are synchronized at
16 common epochs and positions are unwrapped across the periodic boundary.

Temporal refinement approaches a factor-four error reduction per doubling.
At 256 steps, maximum position/velocity RMS errors relative to their initial
radius/dispersion scales are 0.00102/0.00164 (near-circular),
0.000424/0.000953 (cluster) and 0.000378/0.00412 (merger).
The eccentric pair needs 1,024 steps to reach 0.00126/0.00516; at 256 steps
it still has 0.0200/0.0825 errors in these same initial-scale normalizations.
This exposes the close-orbit timestep requirement without changing the solver.

Exact tree/cell trajectories agree below 6.6e-14 in these metrics; the MPI
comparisons agree below 4.5e-16. The boundary pair has three ownership changes
on both 2 and 4 ranks. Interlaced finest trajectories meet the same accuracy
budgets. Canonical momentum remains constant near roundoff.

PM-only runs are matched continuous-gradient controls, not optimized PM baselines;
these compact particle loads are sensitive to the unfiltered kernel. A separate
fiducial-gradient PM check also remains much less accurate than the split force
on these fixtures. These comparisons do not establish general PM performance.
No mechanical-energy conservation claim is made for the expanding system.
Sanitized cluster smoke runs pass both the pipeline (16 synchronization checks)
and reference (17 spatial-refinement checks); new driver/headers are instrumented
while linked background objects are prebuilt without instrumentation.

See [DYNAMICS.md](DYNAMICS.md) and the exported trajectories, convergence plots
and execution/rendering provenance. Small-N dynamics do not calibrate larger
production halos or finite-opening tree drift. COLA, short-range subcycling and
automatic acceleration-based timestep selection remain separate work.


## Common-timestep COLA enablement (2026-11-06)

The existing short-kick placement before the COLA correction passed validation;
no force or frame-update formula change was necessary. Removed the blanket COLA
rejection and allowed Quinn or Tassev with 1LPT/2LPT COLA, including scale-dependent
frames. Ordinary N-body stays Quinn-only. Enforced LPT, screening/exact nonlinear
solvers, massive/separate species, unsupported density/interpolation methods and
noncontinuous kernels remain excluded. An effective order-zero COLA setting cannot
bypass the Tassev restriction.

The serial workflow passes 105 executions; the MPI workflow passes 117, including
1/2/4-rank cases. Nonzero manufactured frames use independent EdS factors or
numerically differentiated model frame velocities, plus independent periodic
Newtonian/Yukawa body forces. All nine checks per manufactured run pass for kick,
drift and combined steps, including nonunit residual scaling and physical velocity
restoration. Maximum velocity discrepancy is 1.77e-8 in code units, position below
1e-10. No new claim about softened Yukawa or nonlinear MG LPT theory follows.

Seeded mild-load trajectories cover GR 1/2LPT, DGP 2LPT, f(R) ordinary 2LPT and
scale-dependent 1/2LPT with Quinn/Tassev and 4/8/16/32 steps. Initial full positions
match N-body exactly and full velocity RMS differences are below 4.7e-12. At 32
steps, position RMS differences from 128-step N-body are below 7.63e-6 box units;
velocity RMS differences are below 4.01e-6 canonical units. The reference's own
64-to-128-step differences reach 1.54e-6/9.77e-7, so these are finite-resolution
state comparisons, not exact-solution error bounds or bitwise COLA/N-body claims.

Exact tree/cell, default table/analytic kernels, interlacing, lightcone-on/off state
transparency, order-zero Quinn reduction and negative input checks pass. Matching
MPI state differences are below 1.23e-17 RMS; lightcone differences below 6.5e-19.
Ordinary N-body short-range regressions pass in serial/MPI, existing COLA-frame/model
unit tests pass, and production nbody smoke runs pass for GR/Tassev and scaled f(R)/Quinn.
Sanitized GR/scaled-f(R) manufactured-frame smoke runs pass (new driver/headers
instrumented; linked background objects prebuilt without instrumentation).

See [COLA_SHORT_RANGE.md](COLA_SHORT_RANGE.md) for configuration, equations and
reproduction. Subcycling, production nonlinear/halo and finite-opening tree COLA
calibration, and model-specific theory limits remain open. No speedup is claimed.
