# Discrete PM pullback (#52)

`ParticleMeshAdjoint.h` adds `FRAMEWORK::ParticleMeshAdjointT<Grid>` (default
`ParticleMeshAdjoint` uses slabs) to the [PM forward map](PARTICLE_MESH.md).
It differentiates the actual fixed-background, fixed-schedule discrete map with
respect to its initial white field and spectrum parameters. The companion [periodic Gaussian posterior/MAP adapter](PM_POSTERIOR.md)
now connects it to inference. HMC scientific validation remains a later milestone. Background, schedule and force discretization are
held fixed.

```cpp
#include <FML/FieldLevel/ParticleMeshAdjoint.h>
// model is a ParticleMeshForwardT<Grid>, work is its workspace.
const auto & result = model.evolve(white, theta, work);
FML::FIELDLEVEL::FRAMEWORK::PMCotangent cotangent;
cotangent.matter_density = density_loss_gradient; // Compact LOCAL owned force cells.
FML::FIELDLEVEL::FRAMEWORK::ParticleMeshAdjointT<Grid> adjoint(model);
auto gradient = adjoint.pullback(cotangent, work, {4});
// gradient.white: LOCAL initial cells.
// gradient.spectrum: replicated GLOBAL derivatives; do not reduce again.
```

The [standalone sensitivity example (archived)](ARCHIVE.md) uses a smooth
density statistic and works in serial, MPI slabs or pencils:

```sh
make -C FML/FieldLevel/example particle_mesh_adjoint USE_MPI=true USE_PENCIL_GRID=true
FML_PENCILS="2 2" mpirun -np 4 FML/FieldLevel/example/particle_mesh_adjoint
```

As for the forward example, clean before switching compilation flags. No GSL
dependency is introduced by the adjoint header or test.

## Cotangent and derivative conventions

`PMCotangent` supplies Euclidean derivatives of a scalar objective with respect
to the forward result's arrays: `initial_density`, `matter_density`, three
`displacement` fields and three canonical `momentum` fields. Pairings are sums
over owned cells/particles and MPI ranks. Particle and force-grid layouts differ
when the two meshes differ; callers must use the matching output layout.
Each empty field means zero, independently on each rank. Nonempty fields must
have the exact local size and finite values. There are no implicit cell-volume,
particle-mass, averaging or likelihood factors: include them in the cotangent
when required by the objective.

`gradient.white` is local and follows the initial-field layout. `gradient.spectrum`
has the same order as `model.parameters()` and is already reduced over ranks.
These derivatives are of the supplied objective, without priors. There are no
derivatives of scalar diagnostic outputs or cosmological background parameters.
Wrapped-position sensitivities may be supplied as displacement sensitivities
away from periodic wrap discontinuities.

The workspace caches a copy of the forward white field and spectrum parameters
before mutating output. A pullback requires a successful forward evaluation from
the same model object. Later changes to the caller's original input arrays do
not change that cached evaluation. Reverse work uses a separate scratch workspace
and leaves the public forward result unchanged, allowing repeated cotangents and
checkpoint-budget comparisons. A failed forward still invalidates the cache.

## Discrete reverse operations

Each KDK step is reversed in this order:

1. Last kick: preserve the momentum cotangent and add `K_last J_F(x_new)^T p_bar`
   to the position cotangent.
2. Drift: add `D x_bar` to the momentum cotangent.
3. First kick: add `K_first J_F(x_old)^T p_bar` at the old particle positions.

For `F(x) = gather_x[G deposit_x(mass)]`, the force pullback includes both
the derivative of the gather weights at the particle positions and the
derivative of density deposition. Gather cotangents scatter to the owning
mesh nodes using the same sparse routing. On the real periodic mesh each
continuous spectral force component obeys `G_c^T = -G_c`; DC and each even-grid
Nyquist derivative component vanish exactly as in the forward kernel. The
normalized forward/unnormalized inverse Fourier transforms introduce no extra
mesh factor in this real-to-real transpose. The deposition branch retains the
forward `N_force^3 / N_particles^3` normalization.

ZA start cotangents include both initial displacement and canonical momentum.
The initial displacement kernel is transposed in Fourier space and combined
with any direct initial-density cotangent. The existing distributed linear
model supplies the spectrum/whitening pullback, including its Hermitian weights
and real/Fourier normalization. Its local spectrum derivatives are reduced once
by this API. No changes to the existing linear/LPT gradient implementation are
required.

CIC is continuous but only piecewise differentiable in position. The code uses
the cell selected by `floor`, taking that cell's polynomial derivative. At an
exact cell face there need not be a unique derivative. Finite-difference probes
must resolve a fixed branch; larger perturbations may cross faces along the
trajectory. This is not a globally smooth assignment/HMC suitability claim.

## Bounded checkpoints and replay

`PMAdjointOptions::max_checkpoints` is the number of additional midpoint states
that may be retained beyond the initial particle state; allowed values are 0..32
and must agree on all ranks. Each retained state contains six local particle
arrays (displacement and momentum), with no mesh/force/routing tape. Recursive
binary subdivision retains midpoint states while reversing the right interval,
then releases them before reversing the left interval. If the budget is exhausted,
prefixes are recomputed from the nearest retained start. Zero additional slots
is supported and trades quadratic replay work for minimum checkpoint storage.
This initial policy is bounded, not an optimal Revolve schedule.

`gradient.report` provides:

- `replayed_steps`: actual repeated KDK steps, including the forward replay of
  each step being reversed. It excludes the original forward call.
- `force_evaluations`: primal force reconstructions during replay and pullback.
  Transposed spectral operations are additional work, not counted here.
- `peak_checkpoint_states`: simultaneous retained particle states, including
  the initial state, at most `max_checkpoints + 1`.
- `checkpoint_payload_bytes`: that peak times `6 * local_particles * sizeof(double)`.

The payload is not total peak RSS or a total workspace budget. Additional fixed
storage includes the original forward result/cache, a scratch PM workspace,
position/momentum cotangents, one old-position array triplet during a reverse
step and an IC cotangent Fourier grid. MPI routing buffers and allocator/FFT
workspace are not included. Retained particle/mesh payload stays bounded in
trajectory length for a fixed checkpoint budget; scalar schedules and forward
diagnostics still grow with the node count. Zero/low budgets can be expensive for long schedules.
Checkpoints here are temporary in-memory reverse states, not restart files.

## Validation and remaining work

The registered `particle-mesh-adjoint` target tests all four output groups,
joint/field/spectrum directions, individual initial cells and spectrum components,
zero/invalid/rank-local optional cotangents, cached input ownership, preserved
forward results and multiple checkpoint budgets. It compares reverse contractions
with centered finite-difference approximations to `J direction`, including a
step-size sweep; no separate analytic tangent/JVP API is claimed. Slab/pencil,
unequal-box and rank/thread tests use canonical gradients.

```sh
make -C FML/FieldLevel/tests particle-mesh-adjoint USE_MPI=true NP_LIST="1 2 4"
# A CSV path exports the step sweep and a canonical gradient snapshot:
mpirun -np 4 FML/FieldLevel/tests/test_particle_mesh_adjoint /tmp/pm_gradients.csv
```

See [the validation record and plots](validation/tests/PM_ADJOINT_RESULTS.md) for measured
derivative errors, checkpoint costs and compiler/rank coverage.

The [periodic posterior/MAP milestone](PM_POSTERIOR.md) is implemented. Small HMC
experiments are the next #52 milestone. Scientific coverage, trajectories near assignment faces, alternative
smooth assignment kernels, cosmological derivatives, survey/tracer/RSD operators,
disk checkpoints, 2LPT/COLA starts and broader performance validation remain open.

## Force-window convention

The pullback uses the same selected `PMSettings::force_window` as the forward
trajectory. Opt-in `CICCompensated` multiplies each skew-adjoint force component
by the real even inverse squared CIC window. Its transpose is therefore still
the negative of that filtered component, with identical DC/Nyquist exclusions.
Both the force-gather position derivative and the density-force-deposition reverse
path use this convention. [Window-specific tests and results](validation/studies/pm_window_validation/RESULTS.md)
include full-output cotangents, independent value differences and replay checks.
This does not add derivatives with respect to the discrete window choice.
