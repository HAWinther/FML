# Survey clustering workflow

The existing `GalaxiesToBox.h` converts RA/Dec/redshift catalogues into Cartesian
coordinates. The new headers use those **physical Cartesian coordinates**, not
FML's normalized periodic particle positions:

- `SurveyMesh.h`: weighted catalogue points, physical geometry, CIC painting and
  MPI routing from arbitrary initial catalogue distributions.
- `SurveyPower.h`: FKP-field monopole, quadrupole and hexadecapole with local
  first-point (Yamamoto) or fixed line of sight.
- `SurveyWindow.h`: random-selection window multipoles through ell=8 and a
  multipole mixing matrix with an optional global integral constraint.
- `SurveyReconstruction.h`: smoothed selection-corrected displacement solve,
  RecIso/RecSym data/random shifts, masks and solver diagnostics.

FFTW is required. MPI/OpenMP and single-precision FFTW builds are supported. MPI
meshes must have at least one x slice per rank and Nmesh must be divisible by the rank
count. Use the same geometry/options on every rank. Catalogue operations are
collective, including ranks with empty local catalogues; catalogues contain each
object on exactly one initial rank. Results/statistics are available on all ranks.

## Coordinates, weights and normalization

```cpp
using namespace FML::SURVEY;
Geometry geometry;
geometry.nmesh = 128;
geometry.boxsize = 1600.;
geometry.minimum = {-400., -800., -200.};
geometry.observer = {0., 0., 0.};
// geometry.local_los = false; geometry.fixed_los = {0., 0., 1.};

std::vector<CatalogPoint> data, randoms; // locally stored objects, arbitrary rank ownership
// Each point: physical position, complete effective weight, selection density nbar.
// Example effective weight: systematic_weight * fkp_weight(nbar, P0).
```

All lengths (positions, box size, smoothing radius, separation bins) use the same
physical units; k uses their inverse and P uses length cubed. An observer is a
physical point independent of the mesh minimum. Geometry is cubic; RA/Dec and
redshift-to-distance conversions remain explicit caller choices.

The box is a padded FFT embedding, not a periodically wrapped survey. Each input
position must lie at least one mesh cell from the edges, specifically mesh
coordinates `1 <= u[d] < Nmesh-2`. CIC clouds crossing the padded boundary are
rejected. Positive-weight objects at the observer are rejected for local LOS.
Weights and nbar must be finite and nonnegative.

The normalization convention is explicit:

```
alpha = sum_data(weight) / sum_random(weight)
F(x) = painted_data(x) - alpha * painted_random(x)
I2 = alpha * sum_random(nbar * weight^2)
```

`PowerOptions::i2 > 0` overrides I2, useful when selection normalization is
estimated externally. Otherwise random objects need physical nbar values. nbar
is the underlying selection number density before the supplied effective weight.
The library does not infer a physical number density from a bounding-box volume
or silently change the normalization for systematics/FKP weight conventions.

## Fourier multipoles

```cpp
PowerOptions options;
options.edges = {0., .01, .02, .03, .04, .05}; // lower inclusive, upper exclusive
options.ells = {0, 2, 4};
auto power = estimate_power(data, randoms, geometry, options);
```

The field is painted as weighted counts per cell. Tensor-weighted FFTs evaluate
`A_ell(k) = sum_x F(x) L_ell(khat . los(x)) exp(-ik.x)` using LOS at mesh nodes.
The shell average is `(2ell+1) Re[A0 conj(Aell)] / I2`. The mesh FFT's internal
normalization is undone explicitly. DC is excluded and Hermitian multiplicities
include odd meshes. Returned `k`, `nmodes`, `raw`, `shotnoise`, and `power` arrays
preserve empty bins as zero power/count with a midpoint k.

Weighted Poisson contact noise is computed from the actual catalogue LOS tensor
moments, with weights squared for data and alpha-squared weights squared for
randoms. Higher-order contact terms are retained for finite anisotropic mode
shells rather than forced to zero. `subtract_shotnoise=false` returns raw power
while still reporting the contact terms.

CIC compensation is enabled by default. It divides the FFT amplitudes by the CIC
window, and is distinct from the survey window below. The shot terms include the
aliased images of the CIC-painted Poisson noise (Jing 2005): the continuum value
times prod_i [1 - 2/3 sin^2(k_i h/2)], divided by the squared CIC window when
compensating (`aliased_shotnoise = false` gives the continuum value; for Poisson
points that leaves +5% of the shot noise at k h = 1.6-2.6). There is no interlacing,
and the aliased images of the clustering signal are not corrected. Use scales
well below the mesh Nyquist frequency, resolve LOS variation with the mesh, and
check mesh-resolution convergence; compensation alone does not remove aliasing.
Only even auto-spectrum multipoles 0, 2, 4 are supported currently.

## Window multipoles and model convolution

```cpp
std::vector<double> separation_edges; // physical separations, increasing
// Fill edges, with the last <= boxsize/2.
auto window = estimate_window(randoms, geometry, separation_edges,
                              power.alpha, power.i2, 8);
// Theory input bins: finer than the measured bins and extending well beyond the last one
// (the window mixes power from higher k into the measured bins; with the input cut at the
// last measured edge the convolved P0 of a 500 Mpc/h cube was 22% low in the last bin)
std::vector<double> input_edges;
for (int i = 0; i <= 200; ++i)
    input_edges.push_back(0.005 * i); // 0 to 1 h/Mpc
auto matrix = make_binned_window_matrix(window, input_edges, options.edges,
                                {0,2,4}, {0,2,4}, true);
// Input vector: all P0 bandpowers, then P2 bandpowers, then P4 bandpowers (on input_edges).
std::vector<double> theory_bandpowers;
// Fill theory_bandpowers with 3 * (input_edges.size()-1) values.
auto predicted = matrix.apply(theory_bandpowers);
```

The selection field is `W = alpha * painted_random / cell_volume`.
FFT cross-correlations of its LOS tensors estimate

```
Q_L(s) = (2L+1)/I2 * angular_average_s integral dx W(x) W(x+s) L_L(shat.los(x)) .
```

CIC self pairs from the same random object are removed exactly, including their
cloud extent and node LOS. This is not merely a subtraction at zero separation.
The optional `subtract_self=false` instead retains all mesh autocorrelation
pairs. `Q0(0)` is estimated with the supplied I2, not renormalized by fiat.
The reported `nvectors` identifies separation bins with no lattice samples.

The separations are the mesh lattice vectors, and the lattice vectors of a shell
are not isotropically distributed (cubic symmetry, only a few vectors at small s).
A plain average over them does not average L_ell to zero for ell >= 4, so the large
isotropic part of the pair function leaks into Q4, Q6 and Q8: for a 500 Mpc/h cube
on a 7.8 Mpc/h mesh the plain average gave Q4 = +4.4, -0.24, -0.25 in the first
10 Mpc/h bins (exact: 0.005, 0.010, 0.016) and errors of +-0.05 out to 100 Mpc/h.
Each shell therefore uses `mean(G F) - mean(G) [mean(F) - <F>_sphere]` for the pair
function G and angular factor F of every LOS tensor, which is exact for a pair
function that is isotropic on the shell; the errors become ~0.005-0.01 (also checked
for a local LOS against a Monte Carlo pair integral). Within about two cells the
shells contain only a few lattice directions and the anisotropy of the pair function
itself is not resolved there; use separation bins at least one cell wide. The ell > 0
averages exclude the s = 0 vector. For a local LOS the window costs one FFT per LOS
tensor (95 for maxell = 8).

Window estimation requires each catalogue's extent plus CIC support to be below
half the box along every axis, preventing periodic images from masquerading as
survey pairs. Choose a separation range covering the window's relevant support;
if the survey's diagonal extent is large, enlarge the padded box accordingly.
There is no implicit correction for a truncated separation range.

Matrix construction and `apply` are local operations and can be called on rank 0
only once the collective window estimate is available.

`make_window_matrix` evaluates output at specified k nodes. The
`make_binned_window_matrix` convenience API integrates each output shell with
k-squared weighting using 16-point quadrature. This is a continuous shell
average, distinct from the exact finite-mesh mode distribution.

The matrix multiplies correlation-function multipoles by Q, using the exact
Legendre triple-product coefficients, and transforms them back with spherical
Bessel kernels. Input P is piecewise constant in each k bin; output is at the
specified k nodes, not a shell average. It uses 16-point quadrature within each
input k bin and within each separation bin, with xi of the input bandpower evaluated
at the nodes and Q interpolated linearly between the bin separations `s` (one point
per separation bin is not converged unless k ds << 1: 10-25% errors at
k = 0.1-0.2 h/Mpc for ds = 10 Mpc/h). The table needs `edges` (filled by
`estimate_window`). Refine k/separation bins and the mesh to check convergence. For measured bandpowers, use the binned helper or explicitly average fine
output k sampling over the measured bins/modes.
Window multipoles through `max(input_ell)+max(output_ell)` are required.

The optional **global integral constraint** subtracts `C Q_ell(s)` from the
convolved correlation multipoles, with C chosen so the convolved monopole at
k=0 vanishes. This is the conventional local-plane-parallel global-IC model;
it is not an exact arbitrary-mask projection of all wide-angle terms. Radial or
angular constraints imposed by catalogue construction need additional operators.
This implementation does not include wide-angle corrections beyond the
first-point/local-plane-parallel conventions.

## Survey BAO reconstruction

```cpp
ReconstructionOptions rec;
rec.bias = 2.;
rec.growth_rate = .8;
rec.smoothing_radius = 15.;
rec.convention = ReconstructionConvention::RecSym;
// rec.mask = [](const Vector & x) { return ...; }; // optional additional selection mask

auto shifted = reconstruct(data, randoms, geometry, rec);
auto reconstructed_power = estimate_power(shifted.data, randoms, geometry,
                                         options, &shifted.randoms);
```

Data and random CIC meshes are smoothed separately with a Gaussian. Their ratio
forms `delta_tracer = D_smoothed/(alpha R_smoothed)-1`. Cells below
`minimum_random_fraction * Rbar`, with `Rbar = sum R_smoothed^2 / sum R_smoothed` the
mean smoothed random density of the survey (independent of the padding), and cells
excluded by an additional mask have source zero. With the default 0.01 the source
extends about 2.3 smoothing radii beyond the survey edges, where the ratio of the
smoothed tails extrapolates the density near the edge; a stricter threshold or a
`mask` of the true footprint confines it. Bias/growth are constant scalars in this implementation.
Use weights appropriate for the reconstruction selection function; these need
not be the same FKP weights chosen later for power estimation.

The displacement solves

```
-div[(I + (f/b) n n) grad(phi)] = delta_tracer / b
Psi = grad(phi)
```

with the local radial n or the fixed LOS. A positive-definite finite-volume-style
stencil pairs forward gradients with their adjoint divergence. A Fourier
Laplacian preconditioner accelerates conjugate gradients. The padded box has
periodic potential boundary conditions, so padding/resolution convergence
remains necessary for a survey. The constant source mode is projected out and
reported. Cell-centered displacements are read with CIC interpolation.

- Data always move by `-Psi - f (Psi.n)n`.
- RecSym randoms move by the same displacement plus RSD shift.
- RecIso randoms move by `-Psi` only.

All LOS values for shifts are taken at the original positions. Returned objects
retain their local order, weights and nbar; MPI queries route interpolations and
return them to their original rank. Positions are never wrapped or clipped.
If a shifted catalogue exhausts the padded region, enlarge the geometry before
passing it back to the estimator. Original unshifted randoms define power
normalization/selection; shifted randoms define its negative density field.
Post-reconstruction weighted contact noise uses shifted randoms, without claiming
an exact correction for reconstruction-induced sampling correlations.

`SolverReport` records convergence, iteration count, actual final residual and
removed source mean. `solve_displacement` returns nonconvergence explicitly;
`reconstruct` throws if it would otherwise return an unconverged shifted catalogue.
Nonfinite arithmetic and invalid source/geometry fail explicitly. The lower-level
`solve_displacement` and `sample_displacement` APIs support independent checks and
reuse of an already prepared displacement field.

## Survey mocks (`SurveyMocks.h`)

Mock catalogues from a periodic box (as make_survey): a snapshot projection around an observer (one snapshot
tiled around the observer; the redshift of an object comes from its distance, its dynamics from the snapshot;
this is not an evolving lightcone), RA/Dec and redshifts, an angular footprint, a radial selection and randoms.

```cpp
using namespace FML::SURVEY::MOCKS;
std::vector<MockObject> objects = objects_from_hod(hod_mock.galaxies); // or objects_from_particles(...)
MockOptions o;
o.boxsize = 1000; o.observer = {500, 500, 500};          // Mpc/h
o.z_min = 0.05; o.z_max = 0.2;
o.hubble_over_c = [](double z) { return std::sqrt(0.3 * std::pow(1 + z, 3) + 0.7) / 2997.92458; }; // h/Mpc
o.footprint = ra_dec_box(120, 240, 0, 60);                 // or any (ra, dec) -> completeness callback
o.target_nbar = [](double z) { return 3e-4; };             // (h/Mpc)^3; empty: keep everything
auto data = make_survey(objects, o);                       // collective (parent density summed over tasks)
auto randoms = make_randoms(o, data.parent_nbar);          // random_factor randoms per expected object
auto points = to_catalog_points<FML::SURVEY::CatalogPoint>(data.galaxies); // for SurveyPower etc.
```

Conventions: comoving Mpc/h, peculiar velocities in km/s, H(z)/c supplied by the user (no cosmology dependency),
RA/Dec in degrees in the survey frame x_s = R (x - observer) (`rotation`, or `rotation_from_pole`).
z_obs = z_cos + (1 + z_cos) v_r/c (receding objects are redder); `redshift_space = false` gives z_obs = z_cos;
positions are r(z_obs) r_hat (and `position_real` = r(z_cos) r_hat). Per object the catalogue stores the
`selection_probability` that was applied (target n(z)/parent density x completeness; an error if it exceeds 1),
the `completeness` of the footprint (also when `apply_completeness = false`) and `nbar`, the expected density of
the final catalogue (for FKP weights and I2); FKP weights are made by the analysis, not here, so nothing is applied
twice. Random decisions use Philox streams keyed by (seed, object id and subid, replica), so the catalogue is the
same for any number of MPI tasks or threads; randoms are keyed by their index and split over the tasks. Tiling
repeats structures when r(z_max) > L/2 (`MockCatalog::repeated`; each object keeps its source id and replica).
Velocities above 0.03 c are not reached by the search shell.

Deep or wide surveys without repetition: `make_remapping(M)` with an integer matrix of determinant 1 gives the
volume-preserving cuboid of Carlson & White (2010) (edges L1 x L2 x L3 = 1 in box units, e.g. {{5,2,1},{2,1,0},
{0,0,1}} gives 5.48 x 0.447 x 0.408); `remap_objects` moves the objects into it (one to one) and the survey is
made with `periodic = false`, `extent` = the cuboid in Mpc/h (`MockCatalog::truncated` when the sphere r(z_max)
around the observer leaves it; the footprint must then stay inside). Healpix footprints:
`healpix_footprint(map)` with a RING-ordered completeness map (`read_healpix_ascii`, or with USE_HEALPIX
`read_healpix_fits` in any ordering); the pixelization is built in (checked against healpy). `mock_example/` makes a mock and randoms from a
uniform box; `tests/test_mocks.cpp` checks r(z) against EdS (4e-9), the coordinate round trip back to the box
(1e-6 Mpc/h), the redshift-velocity relation, the footprint area and n(z) of data and randoms (Poisson), the
feasibility error, 1 vs 2 MPI tasks, the remapping (one to one, uniform, volume 1; a pencil to 3x the box depth
without repeated objects) and the Healpix pixels against healpy. MANGLE footprints, lightcone input and a
`make_survey` executable follow (backlog item 53).

## Build, example and validation

`workflow_example/` generates a small synthetic Cartesian survey and writes
pre/post reconstruction multipoles, Q0..Q8 and a model-mixing matrix. It is an
interface demonstration, not a converged science analysis:

```
make -C FML/Survey/workflow_example
mpirun -np 4 FML/Survey/workflow_example/survey_workflow
make -C FML/Survey/tests run
```

Normal FML Makefile configuration applies. Tests compare FFT multipoles against
a direct DFT, window multipoles against explicit weighted random pairs, Gaussian
window convolution against an analytic result, and fixed/local-LOS displacement
solves against manufactured right-hand sides built with the same discrete operator (a check of the solver; an
independent spectral solution agreed to slope 0.995 and correlation 0.99995 in the Oct 2026 review). They also
check IC nulling, MPI ownership/order, masks, normalization, RecIso/RecSym,
shifted-random spectra and collective invalid-input behavior. These establish
numerical/convention checks; no production-survey mock recovery or external
pypower/pyrecon numerical comparison has yet been performed.

References for the conventions:
[Bianchi et al. (2015)](https://arxiv.org/abs/1505.05341),
[Wilson et al. (2017)](https://arxiv.org/abs/1511.07799),
[pypower API](https://pypower.readthedocs.io/en/latest/api/api.html), and
[pyrecon reconstruction conventions](https://pyrecon.readthedocs.io/en/latest/api/api.html).
