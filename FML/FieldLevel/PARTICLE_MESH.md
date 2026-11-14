# Periodic particle-mesh forward evolution (#52)

`ParticleMesh.h` supplies `FRAMEWORK::ParticleMeshForwardT<Grid>` (default
`ParticleMeshForward` uses slabs). The first PM milestone maps local whitened
initial cells and spectrum parameters into evolved matter and particle fields.
It supports `FFTWGrid<3>` and `PencilGrid<3>` with a copied model domain.
The companion [discrete adjoint](PM_ADJOINT.md) now differentiates this map with
bounded in-memory checkpoints/replay. A [periodic Gaussian posterior/MAP adapter](PM_POSTERIOR.md) now supplies the first
inference application. Tracer/bias/RSD models, an analytic tangent API and HMC
scientific validation require subsequent milestones. The existing LPT inference API
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
power.evaluate = [](double, const auto &) { return FF::Power{1000., {}}; };
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
an explicit compatible domain. The [runnable example](example/ParticleMesh.cpp)
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
continuous spectral force is `i k delta(k)/k^2` in box coordinates. DC and each
even-grid Nyquist derivative component are zero. There is no assignment-window
deconvolution, smoothing, interlacing, short-range force or force-dependent cutoff.
CIC gathers the three force components at the same particle positions. This is
the same discretization as native NBody with continuous kernels and deconvolution
disabled. Mesh/particle discreteness remains a numerical modeling choice.

## Ownership and workspace

The model retains particles on their initial Lagrangian rank. Sparse
`DistributedAssignmentT` routing scatters to the owning force-grid boxes and
returns requested node values; there is no global particle or mesh replication.
This handles particles crossing slabs/pencils and wrapping the box. Its existing
checked MPI-int transport limits still apply; it does not add unbounded-count
support to that router. Cell edges must align on both meshes. Slabs retain the
initial model's divisibility restrictions; pencils support unequal boxes.

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
The larger force mesh can resolve particle lattice structure even for unperturbed
particles; zero density contrast is expected for a uniform lattice only when
particle and force meshes coincide.

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

See [the recorded validation and convergence plot](tests/PM_FORWARD_RESULTS.md) for
measured agreement and sanitizer/compiler limits. These bounded checks do not
establish production cosmological accuracy. The [reverse discrete map and bounded
replay](PM_ADJOINT.md) and [periodic posterior/MAP](PM_POSTERIOR.md) are implemented
in companion headers. Bounded HMC scientific validation is next. Full cosmological
derivatives, 2LPT starts, COLA, RSD/tracer bias,
changed-rank restarts and survey integration remain separate extensions.
