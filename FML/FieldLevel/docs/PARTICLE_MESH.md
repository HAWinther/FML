# Periodic particle-mesh forward evolution (#52)

`ParticleMesh.h` supplies `FRAMEWORK::ParticleMeshForwardT<Grid>` (default
`ParticleMeshForward` uses slabs). The first PM milestone maps local whitened
initial cells and spectrum parameters into evolved matter and particle fields.
It supports `FFTWGrid<3>` and `PencilGrid<3>` with a copied model domain.
The companion [discrete adjoint](PM_ADJOINT.md) now differentiates this map with
bounded in-memory checkpoints/replay. A [periodic Gaussian posterior/MAP adapter](PM_POSTERIOR.md) now supplies the first
inference application. [Bounded HMC checks](validation/studies/pm_full_validation/RESULTS.md) cover
small mocks. [Dense particle sampling](validation/studies/pm_oversampling_validation/RESULTS.md) now passes
bounded spatial-refinement checks after the original sparse-load failure; broader
calibration remains open under [PM-S1](KNOWN_ISSUES.md). Tracer/bias/RSD models and an analytic
tangent API remain subsequent milestones. The existing LPT inference API
and its evolution enum are unchanged.

## Example

```cpp
#include <FML/FieldLevel/ParticleMesh.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
FF::Configuration c;
c.mesh = 8; c.density_mesh = 16; c.boxsize = 200;
c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff = .08;
c.evolution = FF::Evolution::Linear; // Describes the IC generator.
FF::Spectrum power;
power.evaluate = [ (archived)](ARCHIVE.md) { return FF::Power{1000., {}}; };
FF::PMSettings options;
options.background.omega_m = .3;
options.scale_factors = {.1, .15, .2, .3, .4, .6, .8, 1.};
FF::ParticleMeshForward model(c, power, options);
auto work = model.workspace();
const auto & output = model.evolve(model.white_noise(91), model.initial_parameters(), work);
```

FML/MPI must already be initialized (the usual FML startup does this). Every
rank participates in construction, workspace construction and evolution in the
same order. Use the pencil frontend initialization before construction, or pass
an explicit compatible domain. The [runnable example (archived)](ARCHIVE.md)
selects its backend through the usual frontend build flag:

```sh
make -C FML/FieldLevel/example particle_mesh USE_MPI=true USE_PENCIL_GRID=true
FML_PENCILS="2 2" mpirun -np 4 FML/FieldLevel/example/particle_mesh
```

Clean before switching backend/MPI flags. The production header needs FFTW and
the existing FieldLevel dependencies, with no GSL dependency. The optional
native NBody comparison test requires `USE_GSL=true` because `NBody.h` includes
the ODE interface. The independent dense-DFT test runs without GSL.

## Dynamics and units

The initial spectrum is the real-space linear density spectrum at `a=1`, in
the same volume/Fourier units as `DistributedLinearModelT`. The generator keeps
the existing zero/Hermitian and strict initial-cutoff conventions. Only spectrum
parameters are exposed; its incidental bias/noise parameters are not PM inputs.
`growth_rate` in the initial configuration must be zero.

There is one equal-mass particle per initial-grid cell, initially at
`x(q)=q/N + D(a_initial) psi(q)`, with ZA growing-mode canonical momentum.
The initial LPT displacement is derived from that same band-limited density.
The fixed background is flat matter plus Lambda,
`E(a)^2 = Omega_m/a^3 + 1 - Omega_m`, with `0 < Omega_m <= 1` and no radiation,
massive neutrinos or modified gravity. Growth is normalized to `D(1)=1`;
the matter/Lambda growing solution uses its integral representation. Simpson
quadrature uses 128 panels and an endpoint-regularizing coordinate for growth;
EdS growth and time integrals are analytic. Extreme parameter limits are outside
the recorded validation range.

Coordinates/displacements are fractions of the periodic box, and
`p = a^2 dx/d(H0 t)`. Thus physical peculiar velocity is `H0 * L * p / a`,
when `H0` and `L` have compatible units. Positions wrap for assignment, while
displacements accumulate without wrapping. With `g = -grad inverse_Laplacian(delta)`,
the equations in scale factor are

```text
dx/da = p / (a^3 E(a))
dp/da = (3 Omega_m / 2) g / (a^2 E(a))
```

Each interval uses a kick to its arithmetic scale-factor midpoint, a full drift,
then a kick from midpoint to endpoint, using the force at the appropriate
particle positions. The fixed schedule must contain 2..100001 strictly increasing
finite nodes in `[1e-4,1]`. Its dimension, values and background are checked
collectively before trajectory execution. It is not an adaptive timestep solver.

CIC scatters mass onto the force mesh, normalizing to mean density one. The
default continuous spectral force is `i k delta(k)/k^2` in box coordinates. DC and each
even-grid Nyquist derivative component are zero. The default `PMForceWindow::None` has no assignment-window deconvolution.
There is no implicit smoothing, interlacing, short-range force or force-dependent cutoff.
CIC gathers the three force components at the same particle positions. This is
the same discretization as native NBody with continuous kernels and deconvolution
disabled. Mesh/particle discreteness remains a numerical modeling choice.

## Optional CIC force compensation

```cpp
c.mesh = 64; c.density_mesh = 32; // Dense particle sampling; keep IC cutoff below both Nyquists.
options.force_window = FF::PMForceWindow::CICCompensated;
```

This explicit opt-in convention multiplies the spectral force by `1/W_CIC^2`,
where `W_CIC=product_i sinc(pi*n_i/M)^2` for integer wavevector n and force mesh M.
It compensates CIC deposition and force gathering. The real even multiplier is
also used by the discrete adjoint; DC/component Nyquist handling stays the same.
The original `None` convention remains the default and preserves existing targets.

This setting changes dynamics. It does **not** deconvolve `matter_density`, change
its likelihood response, remove aliasing or suppress particle noise. The inverse
window can amplify short-scale errors, especially for sparse loads. In the recorded N8/M64 linear control, its
longitudinal response is about 53.65 rather than the fluid value 1. Validate the
particle sampling, time/cutoff choices and posterior for the intended application.
[The controlled comparison](validation/studies/pm_window_validation/RESULTS.md) reports raw-density,
common-window diagnostic and posterior-gradient results separately. A common-window
plot is postprocessing, not an implemented PM observation adapter.

Compensated targets use checkpoint numerical-law tag 2. Default targets retain
tag-1 identity bytes and old checkpoint compatibility. Cross-window restart is
rejected even at a zero state. Settings remain immutable in a constructed model;
make a new model/workspace to choose a different convention.

## Ownership and workspace

The model retains particles on their initial Lagrangian rank. Sparse
`DistributedAssignmentT` routing scatters to the owning force-grid boxes and
returns requested node values; there is no global particle or mesh replication.
This handles particles crossing slabs/pencils and wrapping the box. Its existing
checked MPI-int transport limits still apply; it does not add unbounded-count
support to that router. Cell edges must align on both meshes. Slabs retain the
initial model's divisibility restrictions; pencils support unequal boxes.

`Configuration::mesh=N` controls both the initial white/IC lattice and N^3
particles; `density_mesh=M` controls the force and observation mesh. **N may exceed
M.** Both meshes must be even and at least 4. The strict initial cutoff must be
below `pi*min(N,M)/boxsize`; a periodic posterior also checks its observation
cutoff against M. Increasing N at fixed physical cutoff adds sampling and prior-null
white coordinates, not new supported physical modes. Particle/white memory grows
as N^3; the force workspace grows as M^3. There is no independent particle lattice.

For slabs both N and M must be divisible by `NTasks`. For pencils, process dimensions must fit
both grids, including `px<=M` and `py<=M/2+1` for the force rFFT transpose, and
all copied domain edges must align on both meshes. Unsupported coarse-force
process layouts reject collectively before force-workspace allocation.

The workspace retains one linear workspace, a density grid, three force grids,
six local particle fields, output fields and sparse routing buffers, plus a cached
copy of the white input and spectrum parameters for the companion pullback. It stores
no trajectory of grids/particles. Reconstruction of the initial fields has
temporary allocations; this is not a measured peak-memory guarantee. There is
one scalar diagnostic per schedule node. Repeated calls reset the trajectory,
and workspaces are movable but bound to their original model object.

`evolve()` returns a reference owned by the workspace, valid until its next
evolution or destruction. Copy the result when an owning snapshot is needed.
`result()` rejects access before a successful evolution or after a failed one.
Outputs have separate particle and force-density `Model::Layout` descriptors:

- `initial_density`: local linear density at the `a=1` reference, on the IC mesh.
- `displacement`, `momentum`: three local particle fields in Lagrangian order.
- `position(q)`: wrapped final position computed from the displacement and ID.
- `matter_density`: compact owned real cells of the final force mesh, without padding.
- `diagnostics`: scale factor, mean-mass error, density RMS and mean momentum.

The density is the CIC mesh density, without later filtering or Fourier response.
Diagnostics measure mass and momentum, not total energy or posterior calibration.
A fine force mesh can resolve particle lattice structure even for unperturbed
particles. Coincident grids and aligned integer particle oversampling give a
homogeneous uniform load; arbitrary mesh ratios need not.

## Validation and next stages

The registered `particle-mesh` test target checks analytic homogeneous and growing
ICs, independent dense CIC/direct DFT EdS evolution, optional native migrating
NBody evolution, conservation, repeatability, layout/ownership and collective
input errors. MPI tests compare slab, x/y pencils and 2x2 pencils, including
unequal boxes. An optional output filename runs a fixed-mesh timestep refinement
experiment against a finer trajectory.

```sh
make -C FML/FieldLevel/tests particle-mesh USE_MPI=true NP_LIST="1 2 4"
# Optional native NBody comparison:
make -C FML/FieldLevel/tests clean
make -C FML/FieldLevel/tests particle-mesh USE_MPI=true USE_GSL=true NP_LIST="1 2 4"
```

See [the recorded validation and convergence plot](validation/tests/PM_FORWARD_RESULTS.md) for
measured agreement and sanitizer/compiler limits. These bounded checks do not
establish production cosmological accuracy. The [reverse discrete map and bounded
replay](PM_ADJOINT.md) and [periodic posterior/MAP](PM_POSTERIOR.md) are implemented
in companion headers. Bounded HMC and same-layout restart are implemented;
[dense-sampling validation](validation/studies/pm_oversampling_validation/README.md) records the new N>M
forward/adjoint/posterior/restart checks. Broader calibration and full cosmological
derivatives, 2LPT starts, COLA, RSD/tracer bias,
changed-rank restarts and survey integration remain separate extensions.

## Read-only trajectory observer

`evolve_observed(white, parameters, workspace, callback)` invokes a local callback
`callback(node_index, const Result&)` before deposition at each force/position
node. Use `settings().scale_factors[node_index]` for its time; `Result::a` remains
final-output metadata. Positions are current at the callback; momentum can be
staggered, and density/diagnostics are not completed node outputs. All ranks participate; callbacks must not call collectives.
Exceptions propagate collectively and invalidate the workspace. The default
`evolve()` compiles out the observer and preserves its communication/arithmetic.
See [stationarity diagnostics](validation/studies/pm_stationarity_validation/README.md).

## Optional TSC assignment

`PMSettings::assignment` defaults to `PMAssignment::CIC`. Select the quadratic
B-spline for both force deposition/gather and the final matter density with:

```cpp
PMSettings settings;
settings.assignment = PMAssignment::TSC;
settings.force_window = PMForceWindow::TSCCompensated; // or None
```

In one mesh coordinate, let `j=floor(u+1/2)`, `t=u-j`. The weights on
`j-1,j,j+1` are `(1/2)(1/2-t)^2`, `3/4-t^2`, `(1/2)(1/2+t)^2`.
Their derivatives with respect to box-fraction positions are mesh size times
`t-1/2`, `-2t`, `t+1/2`. The 3D stencil is their tensor product over 27 periodic
nodes. Weight and first derivative remain continuous across half-cell support
changes, including periodic wrapping. Second derivatives need not be continuous.
All nodes, including zero-weight nodes, participate in sparse routing.

The discrete adjoint uses the same stencil derivatives in source deposition,
force gathering and final-density pullbacks. It retains checkpoint-budget replay
and the original initial-field generator, time integrals and Fourier force law.
`TSCCompensated` removes both TSC force windows: the product of `sinc^3` per
assignment, squared for deposit plus gather. `CICCompensated` and TSC assignment
are rejected together, as is the opposite mismatch. `None` is valid with either
assignment. Compensation does not remove particle/grid aliasing.

The reported density retains its selected assignment window. If
`PMObservationSettings::reference_cic_mesh` is set, a TSC model uses the fixed
coarse-CIC/fine-TSC Fourier response. With equal reference and force meshes this
can amplify the raw TSC signal. Noise is added after that response; raw
reconstruction fields remain unchanged by the response. A reference mesh is a
window convention, not a new data layout or exact regridding/alias correction.

TSC has a distinct PM checkpoint target identity, even at an indistinguishable
zero saved state. Default CIC checkpoint identities and numerical paths remain
unchanged. Assignment choices must agree across MPI ranks. Slab and pencil
backends share the same sparse routing, with constant-size local stencil work;
TSC visits 27 nodes rather than CIC's eight and costs more per force evaluation.

See the [bounded matched-assignment experiment](validation/studies/pm_tsc_validation/README.md).
This changes the discretized forward model and requires matched data/likelihood
conventions; it is not a transparent optimization of a fixed CIC target.
