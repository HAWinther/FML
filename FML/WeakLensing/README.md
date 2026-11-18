# Weak lensing and projected galaxy–matter observables

This is the initial implementation of backlog #37: flat-geometry Born convergence
maps for source bins, periodic flat-sky shear, extended-Limber angular spectra,
and finite-line-of-sight galaxy–matter profiles with comoving excess surface
density. It provides projection/analysis operators, not a ray tracer or a
calibrated survey likelihood.

`WeakLensing.h` is standalone C++17 (no FML setup, FFTW, GSL or MPI dependency).
`FlatSky.h` adds an optional `FFTWGrid<2>` adapter using the existing serial/MPI
FFT backend (any 2D grid with FML's generic grid interface works too, e.g.
`PencilGrid<2>`). Namespace: `FML::WEAKLENSING`.

## Units and physical assumptions

| Quantity | Convention |
|---|---|
| chi, R, Pi, c/H0 | Comoving Mpc/h; default c/H0 = 2997.92458 Mpc/h |
| k and P(k,z) | h/Mpc and (Mpc/h)^3 |
| delta_m and xi_gm | Dimensionless total-matter contrast and galaxy–matter correlation |
| source density n(z), n(chi) | Probability/count density per redshift or per comoving Mpc/h; explicitly normalized internally |
| radial galaxy window | Normalized p(chi), units (Mpc/h)^-1 |
| kappa, gamma, C_ell | Dimensionless; angular patch widths in radians |
| w_gm | Comoving Mpc/h |
| rho_m0 | Present mean matter density, (Msun/h)/(Mpc/h)^3 |
| DeltaSigma_com | (Msun/h)/(Mpc/h)^2 |
| 4 pi G/c^2 | Caller-supplied (Mpc/h)/(Msun/h) for critical-density conversion |

The density kernel assumes flat FRW geometry, GR's standard matter/Weyl-potential
relation and the Born density-projection approximation. Supply the matching
redshift–distance relation, Omega_m and **total** matter field/power. A CDM+baryon
field in a massive-neutrino model is not automatically total matter. No modified
Weyl response, curvature, post-Born/lens–lens coupling, reduced-shear correction,
intrinsic alignments, source clustering, magnification bias, shape noise or
shot-noise subtraction or RSD/relativistic number-count terms are included.
(`CosmicShear.h`, below, adds NLA intrinsic alignments, a Sigma(z) on the lensing
kernel, photo-z shifts and shear calibration.)

## Source bins and kernels

```cpp
#include <FML/WeakLensing/WeakLensing.h>
using namespace FML::WEAKLENSING;
auto screen = Sources::plane(2000.);
Sources weighted_planes({1500.,2500.}, {1.,3.});
// n_z is a density, not bin counts. chi_of_z returns comoving Mpc/h.
auto population = Sources::from_redshift_density(z_nodes, n_z, chi_of_z);
// Alternative: source density per comoving distance.
auto population_chi = Sources::from_distance_density(chi_nodes, n_chi);
FlatGeometry geometry{Omega_m, 2997.92458, z_of_chi};
auto W = convergence_kernel(population, geometry);
```

`Sources(chi,masses)` sorts/merges positive-distance planes and normalizes
nonnegative probability masses. The density factories use trapezoid **quadrature
masses in their declared measure**, then map the nodes to source distances. Thus
n(z) includes dz weights, not dchi weights with a missing Jacobian. They do not
claim an exact continuous distribution: refine the source table and verify
convergence. Finite table support is truncated and renormalized;
`input_normalization()` reports the original integral/mass. Zero-weight observer
nodes are allowed, but positive probability at chi=0, negative/zero total mass,
nonfinite values and nonmonotone density grids are rejected.

The efficiency and kernel are

```
g(chi) = sum_{chi_s>chi} p_s (1 - chi/chi_s)
W_kappa(chi) = (3 Omega_m / 2) chi (1+z(chi)) g(chi) / (c/H0)^2
```

Positive cumulative tail sums make efficiency queries O(log N_sources).
Callbacks must return finite values on all queried intervals. The geometry
callback is not an implicit cosmology solver or redshift extrapolator.

## Streaming Born maps

```cpp
BornPlan plan(shell_edges, source_bins, geometry);
BornAccumulator maps(plan, local_pixel_count);
for (std::size_t j=0; j<plan.shells(); ++j)
    maps.add_shell(j, delta_shell.data(), delta_shell.size());
const auto & kappa_bins = maps.maps(); // available only after every shell
```

Each input is an angular map of **density contrast**, represented as constant
radially inside its shell. Geometric weights integrate W over the finite shell,
splitting at source-plane distances (including a source inside a shell). The
represented radial density still needs shell-resolution convergence. Lenses
outside the supplied radial interval are omitted; the plan exposes that interval
rather than silently adding unprovided data.

Only source-bin weights and local output maps persist; input shells can be
released after addition. Pixels may be any common angular ordering, including
local pieces of a spherical map. Every shell must use the same pixel directions
and resolution. Different onion-map NSIDE values must first be reprojected to a
common representation; counts/mass maps must first become contrast maps.
Missing/masked pixels must not silently be treated as physical zero contrast.
There is no automatic mask correction or HEALPix reader.

The core does no MPI collectives or pixel replication. Each rank supplies its
owned pixels and an identical radial/source plan. Empty ownership is supported.
Input size, ordering, finiteness and update overflow are checked before mutation;
duplicate/out-of-order shells and premature result access are rejected. In an MPI
application, the caller coordinates errors from these rank-local APIs.

## Periodic flat-sky shear

```cpp
#include <FML/WeakLensing/FlatSky.h> // requires USE_FFTW
// kappa is a real-space, distributed FFTWGrid<2> on a periodic angular patch.
auto shear = shear_from_convergence(kappa, theta_x_radians, theta_y_radians);
```

The maps can also be another 2D grid type, e.g. `FML::GRID::PencilGrid<2>` (include
`<FML/PencilDomain/PencilGrid.h>`): `shear_from_convergence` returns
`ShearMapsT<Grid>` with maps made like the input (`ShearMaps` is the `FFTWGrid<2>`
version). The tests run on both.

The fixed Cartesian convention is

```
gamma1(l) = (lx^2-ly^2)/(lx^2+ly^2) kappa(l)
gamma2(l) = 2 lx ly/(lx^2+ly^2) kappa(l)
```

Input is preserved; the two output grids have owned real pixels and no halos.
All ranks participate in the existing FFT backend. Angular widths, grid status,
local finite pixels and cross-rank geometry agreement are checked collectively.
DC is removed: shear cannot recover a convergence mass sheet. For even N, **both
Nyquist lines are discarded in both components**, avoiding ambiguous mixed
Nyquist derivatives and preserving a real spin-2 field. Odd meshes have no such
lines. This is pure gravitational E shear from scalar convergence, not a
full-sky spin-2 transform, a finite-mask inversion or reduced shear.

## Angular spectra

```cpp
PowerSpectrum mm{Pmm_of_k_z, k_min, k_max, z_min, z_max};
auto c_ij = limber_spectrum(ells, W_i, W_j, mm, geometry, radial_edges);
RadialSelection lenses(chi_nodes, n_chi_lens);
PowerSpectrum gm{Pgm_of_k_z, k_min, k_max, z_min, z_max};
auto c_gk = limber_spectrum(ells, lenses, W_j, gm, geometry, radial_edges);
```

The generic one-pair integrator evaluates

```
C_ell = integral dchi Wa(chi) Wb(chi) P((ell+1/2)/chi,z(chi)) / chi^2
```

Use total Pmm for convergence pairs and galaxy–matter **cross** power Pgm for
foreground–convergence pairs. Galaxy bias belongs in Pgm if the foreground window
is a normalized density; do not insert it twice. Signed cross power is allowed.
No noise term is added/subtracted. The same generic operator can use other
consistent windows and power providers; it does not infer their tracer physics.

`RadialSelection` is a continuous piecewise-linear p(chi), normalized by its
trapezoid integral and zero outside support. If starting with a foreground n(z),
convert to p(chi) using dz/dchi before constructing this window. This continuous
window is distinct from discrete source quadrature masses.

Supply panel boundaries at source planes, foreground-window endpoints/knots and
other discontinuities; refine radial panels and Gauss order (2–64) until stable.
Quadrature nodes avoid chi=0. Nonzero integrands querying outside declared k/z
power coverage throw; there is no hidden extrapolation or silent k cut. Zero
windows skip the power query. ell>=2 is a numerical input constraint, **not** a
claim that Limber/flat-sky predictions are accurate at low ell. Wide-angle,
non-Limber, full-sky shear factors and oscillatory angular correlation transforms
remain extensions. In the supported flat-sky limit C_EE=C_kappakappa and C_BB=0.

## Projected galaxy–matter profiles

```cpp
auto w = [&](double R) { return projected_galaxy_matter(R, Pi_max, xi_gm); };
double ds = excess_surface_density(R, rho_m0, w);
double invcrit = inverse_sigma_critical_comoving(chi_lens, z_lens, sources,
                                                four_pi_G_over_c2);
double gamma_t = ds * invcrit; // thin lens, R = chi_lens * theta
```

```
w_gm(R;Pi_max) = 2 integral_0^Pi_max xi_gm(sqrt(R^2+Pi^2)) dPi
DeltaSigma_com = rho_m0 [2/R^2 integral_0^R R' w_gm(R') dR' - w_gm(R)]
1/SigmaCrit_com = (4 pi G/c^2) (1+z_lens) chi_lens g(chi_lens)
DeltaSigma_phys = (1+z_lens)^2 DeltaSigma_com
```

The excess-density integral uses a scaled radius and subtracts the outer value
inside the integrand, so a constant projected sheet cancels exactly. xi_gm is an isotropic real-space input; anisotropic/RSD correlations depending
on (R,Pi) require a separate extension. xi_gm and w_gm are supplied functions; no undocumented power-to-correlation transform or
bias model is inserted. w_gm must cover central radii needed by the interior mean.
Finite Pi_max is part of the observable definition: verify Pi and quadrature
convergence rather than assuming an infinite projection. The critical-density
conversion assumes one lens plane and unclustered source weights; converting a
broad foreground population requires its own lens/source integration, not one
mean lens redshift. Surface density here is per (Mpc/h)^2, not per pc^2.

## Tomographic cosmic shear (`CosmicShear.h`)

`CosmicShear.h` (header-only, standalone like `WeakLensing.h`, same units) adds what a cosmic-shear
likelihood (KiDS, DES, ...) needs on top of the kernels above: intrinsic alignments, photometric-redshift
and shear-calibration nuisances, the correlation functions and COSEBIs. The matter power spectrum is the
caller's: nonlinear, **total** matter, including any baryonic feedback.

```cpp
#include <FML/WeakLensing/CosmicShear.h>
using namespace FML::WEAKLENSING;
ShearModel model;
model.geometry = {Omega_m, 2997.92458, z_of_chi};   // flat; Omega_m in the lensing prefactor
model.chi_of_z = chi_of_z;                          // Mpc/h
model.power = {P_nl_of_k_z, k_min, k_max, 0, z_max};// h/Mpc, (Mpc/h)^3; 0 outside [k_min, k_max]
model.growth = D_of_z;                              // D(z)/D(0), for NLA
// model.lensing_strength = Sigma_of_z;             // modified gravity: Sigma(z) on the lensing kernel
ShearSystematics sys{delta_z, m, NLA{A_IA, eta_IA}}; // per-bin shifts and m (empty: 0)
auto cl  = shear_spectra(nz_bins, model, sys, ells); // cl.total[cl.pair(i, j)][ell], also GG, GI, II
auto xi  = shear_correlation_binned(cl, theta_edges);  // xi_+-, bin-averaged with weight theta
LogCOSEBIs cosebis(0.5 * arcmin, 300 * arcmin, 5);   // tabulates W_n(ell) once (~3 s for ell <= 1e5)
auto En  = cosebis.E(cl);                            // En[pair][n-1]
```

Limber spectra of shear and NLA intrinsic alignments:

```
C_ell^ij = S(ell) int dchi/chi^2 [q_i q_j + (q_i p_j + p_i q_j) F + p_i p_j F^2] P((ell+1/2)/chi, z)
F(z)     = -A_IA C1 rho_crit Omega_m / D(z) [(1+z)/(1+z_pivot)]^eta_IA
C_ell^ij -> (1 + m_i)(1 + m_j) C_ell^ij,       n_i(z) -> n_i(z - delta_z_i)
```

with q_i the convergence kernel above (times `lensing_strength` if set), p_i = n_i(z) dz/dchi normalised,
C1 rho_crit = 5e-14 rho_crit = 0.013877 (CosmoSIS as used by KiDS-1000 and DES, and CCL; the KiDS-1000 chain's
chi^2 values need it: with 0.0134 they differ by 1.5 at A_IA = 2), z_pivot = 0.62 by default, and
S(ell) = (ell-1)ell(ell+1)(ell+2)/(ell+1/2)^4 only with `ShearSettings::spin_factor` (CCL includes it,
CosmoSIS does not; it matters only at ell < 10). A positive delta_z moves the sources to higher redshift (as
CosmoSIS' `photoz_bias`); n(z) is linear between its nodes, cut at z = 0 and refined to `dz_max` = 0.002.
The radial integral uses Gauss-Legendre panels of width min(20 Mpc/h, 0.1 chi).

The correlation functions are the flat-sky transforms
xi_+-(theta) = int dell ell/(2 pi) C_ell J_{0,4}(ell theta) over the ell range of the spectra (C_ell a
cubic spline in ln ell of ln C_ell, zero outside; take ell up to ~1e5 for arcminute scales), integrated
between the zeros of the Bessel function. `hankel` does the same for any callable C(ell) and J_0, J_2, J_4.

`LogCOSEBIs` are the logarithmic COSEBIs of Schneider, Eifler & Krause (2010, arXiv:1002.2136) used by
KiDS-1000 (Asgari et al. 2021):

```
E_n = int dell ell/(2 pi) C^E_ell W_n(ell),  W_n(ell) = int dtheta theta T_+n(theta) J_0(ell theta)
    = 1/2 int dtheta theta [T_+n xi_+ + T_-n xi_-]      (B_n: the difference; EB() from xi_+-)
```

T_+n is a polynomial of degree n+1 in ln(theta/theta_min) with int theta T_+ = int theta^3 T_+ = 0 and
(1/Delta theta) int T_+m T_+n = delta_mn. SEK10 solve for its monomial coefficients with ~40 digits; here it is
expanded in the polynomials orthonormal for dtheta on [theta_min, theta_max] (from the Stieltjes recurrence),
where the conditions are a well-conditioned Gram-Schmidt step in double precision (orthonormal to 1e-14
for n = 20). `normalization(n)` and `roots(n)` are SEK10's N_n and r_ni, for comparison with tabulated
files. W_n is tabulated at construction: direct Gauss-Legendre below ell theta = 30, above it the Hankel
asymptotic form of J_0 with Filon's method, so the cost does not grow with ell (~3 s on two cores for
n <= 5, ell <= 1e5); E() then costs ~1 ms per pair.

Validation (`tests/test_cosmic_shear.cpp`, references by `tests/generate_cosmic_shear_reference.py`):

| Check | Agreement |
|---|---|
| C_ell, 3 bins with photo-z shifts, ell = 2 - 2e4, vs QUADPACK with the exact n(z) | 1e-5 (6e-6 from the tabulated n(z)) |
| same, GG and GG + NLA vs CCL 3.3.6 (same P(k, z), chi(z), D(z)) | 3e-4 (CCL is 3e-4 low at ell = 2e4 for the low-z bin) |
| NLA part (GI + II) vs CCL | 3e-5 of the signal |
| xi_+- vs QUADPACK of the same C_ell, 0.5' - 300' | 6e-8 |
| Hankel J_0, J_4 vs analytic transforms | 1e-10 |
| COSEBIs N_n, r_ni, T_+-(theta), n <= 5, 0.5' - 300', vs SEK10 in 60-digit arithmetic | 1e-14 |
| W_n(ell) (asymptotic + Filon) vs brute force, ell = 3 - 7e4 | 3e-7 |
| E_n in ell space vs real space (T_+-, xi_+-); B_n of a pure E mode | 7e-6; 5e-6 of E_n |

Not included: the data files and likelihoods (KiDS-1000 COSEBIs, DES Y3 xi_+-), the theory pipeline
(nonlinear total-matter P(k, z) on a grid from `CosmologyFitting::Theory`), bandpowers, galaxy-galaxy lensing
and clustering (3x2pt), beyond-Limber or full-sky corrections, reduced shear, source clustering, and TATT
intrinsic alignments.

## Build, example and validation

```sh
make -C FML/WeakLensing/tests run            # test_weak_lensing (np = 1, 2, 4) and test_cosmic_shear (np = 1)
make -C FML/WeakLensing/example
OMP_NUM_THREADS=2 mpirun -np 2 FML/WeakLensing/example/weak_lensing_example
# Scalar-only path, from a clean/separate build:
make -C FML/WeakLensing/tests clean
make -C FML/WeakLensing/tests run USE_MPI=false USE_OMP=false USE_FFTW=false
```

The example uses a clearly labeled toy flat distance law and analytic spectra;
it streams two source bins, constructs shear and prints Ckk/Cgk and projected
profiles. Replace its background/field/power inputs for scientific applications.
Build configuration changes require clean or separate object directories.

Tests include analytic source efficiencies and finite-shell integrals, streamed
local ownership/error cases, source measure conversion, radial refinement,
analytic Limber auto/cross/foreground spectra, Gaussian finite-Pi profiles,
physical/comoving consistency and collective FFT mode/sign/aspect/B-mode/DC/
Nyquist checks. `tests/Reference.h` contains twelve independent EdS/analytic-power
QUADPACK references generated by `tests/generate_reference.py` (SciPy 1.12.0);
normal tests need no Python. These validate numerical conventions, not production
cosmological calibration. See the task review for executed build configurations.

Existing [SimulationCode onion-map projection](../SimulationCode/src/Lightcone/Lightcone.h)
and [CMB spectrum remapping](../Cosmology/Spectra/Lensing.h) remain separate. This
module does not change either pipeline. Full-sky adapters, observational masks,
precision cosmology/reference-code comparisons and production convergence/scaling
studies remain follow-ups.

Equations follow standard Born/flat-sky projections, e.g.
[MillenniumTNG lensing definitions](https://doi.org/10.1093/mnras/stad2205) and
[extended-Limber source-distribution treatment](https://www.aanda.org/articles/aa/pdf/2025/01/aa51494-24.pdf).
