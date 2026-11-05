# Periodic isotropic three-point correlation multipoles

Initial scope of backlog #33: unweighted point catalogues in a periodic cubic box,
with radially binned isotropic 3PCF multipoles through ell=16. C++17 and `Global.cpp`
are required; FFTW and GSL are not. The existing periodic KD-tree supplies neighbors.
The harmonic factorization follows [Slepian & Eisenstein (2015)](https://arxiv.org/abs/1506.02040).
This implementation has its own tests and is not a port or reference-code calibration.

## Usage and ownership

```cpp
#include <FML/ThreePointCorrelation/ThreePointCorrelation.h>
namespace T = FML::THREEPOINT;
std::vector<T::Sample> data, random; // Sample{position[3], uint64_t id}
// Fill data and an independent, uniform random catalogue over the full box.
auto result = T::periodic(data, random,
    {500., {20., 45., 70., 100.}, 4}); // boxsize, shell edges, maximum ell
const double zeta2 = result.zeta[result.index(2, 0, 1)];
const double DDD0 = result.moment(0, 0, 0, 0, 0, 1);
```

`periodic` is collective over `FML::communicator`. Initialize FML once, e.g.
`FML::FMLSetup::init(&argc,&argv)`. **Both full catalogues, their order and options
must be identical on every rank.** A collective validity check and input fingerprint
reject inconsistent inputs before neighbor work. This is a replicated-catalogue
algorithm; do not pass only locally owned MPI slabs. Primary indices are divided
cyclically among ranks, and only the fixed-size moments array is summed. Every rank
receives the full result. Floating-point reduction order can change roundoff.
There is no hidden catalogue gather and no distributed tree or halo exchange.

Call from outside OpenMP regions. The initial implementation uses one serial primary
loop and reusable harmonic buffers per MPI rank. Enabling OpenMP elsewhere does not
thread this loop. Concurrent collective calls on the same communicator are unsupported.

Coordinates and edges use the same caller-selected length unit. Each position must
be finite and in `[0,L)`; it is not silently wrapped. IDs must be unique **within**
each catalogue. The two catalogues have separate ID namespaces; equal IDs across
catalogues do not cause exclusion. The random catalogue must be statistically
independent of the data and uniform over the entire box. It must not be a copy or
subset of the data. Each catalogue needs at least three objects.

Edges are strictly increasing and satisfy `0 < rmin < rmax < L/2`, with 1–128 shells.
Maximum ell is 0–16. Unrepresentable squared distances/shell normalization are rejected.
Radial shells are half-open `[edge[b],edge[b+1])` and use minimum-image vectors from
the primary to each neighbor. The primary cannot be its own neighbor. Zero-length
vectors, including coincident distinct objects, fall outside the positive inner edge.
Only the two primary-to-neighbor distances are binned: the third side is unrestricted.
These are primary-centered, ordered-leg counts, not unique unordered triangles.

## Estimator and normalization

For catalogue types `c,a,b` in `{D,R}`, the returned raw moment is

```
M_cab(ell,i,j) = sum over distinct objects (p,q,r)
    1[p in c, q in a, r in b] 1[r_pq in shell i] 1[r_pr in shell j]
    P_ell(unit(r_pq) dot unit(r_pr)).
```

Objects must be distinct when drawn from the same catalogue. Cross-catalogue objects
are distinct even if they have matching numerical IDs. A triangle contributes once
for each selected primary and each ordered assignment of its two other vertices.
The result layout is `[c*4+a*2+b][ell][i][j]`; type 0 is data and type 1 random.
`moment(...)` and `index(...)` check indices. `catalogue_sizes` records normalization
counts. `options` records the geometry and angular truncation.

Let `nR=c+a+b` and `nD=3-nR`, interpreting R as 1 and D as 0. Normalize each orientation
by its own distinct-object count:

```
Q_cab = M_cab / [(N_D)_nD (N_R)_nR],   (N)_k = N(N-1)...(N-k+1)
f_i = 4*pi/3 * [(edge[i+1]/L)^3 - (edge[i]/L)^3]
zeta_ell(i,j) = (2*ell+1)/(f_i*f_j) * sum_cab (-1)^nR Q_cab(ell,i,j).
zeta(r1,r2,mu) = sum_ell zeta_ell(r1,r2) P_ell(mu).
```

The signed sum uses the density-contrast construction of
[Szapudi & Szalay](https://arxiv.org/abs/astro-ph/9704241), with analytic periodic
normalization in place of a measured random denominator. It expands all three DDR and
all three DRR orientations. It removes the uniform and two-point/disconnected terms;
raw DDD alone is not the 3PCF. The denominator uses **analytic periodic shell volumes**,
not a noisy measured RRR ratio. RRR still appears in the contrast numerator. Therefore
this is a periodic estimator, not a survey edge-corrected estimator. It is radially
volume averaged and angularly exact in the retained multipoles; finite ell truncates
the reconstructed angular function, not the individual multipole estimates.

Factorial normalizations account for sampling without replacement within each
catalogue. No extra Poisson self-triplet subtraction should be applied to these
already distinct-object moments. The mean density is estimated from the catalogue;
finite-volume/integral-constraint effects in a cosmological sample still require
application-specific treatment. A finite random catalogue adds sampling variance.
Increase its density and test convergence for science measurements. Arbitrary weights,
selection functions, edge corrections and count covariance are not implemented.

## Harmonic factorization, memory and cost

For each primary, accumulate real orthonormal spherical harmonics in each radial shell:

```
a_lm^t(i) = sum of Y_lm(unit separation) over neighbors of type t in shell i
M contribution = 4*pi/(2*ell+1) * sum_m a_lm^a(i) a_lm^b(j)
                 - 1[a=b and i=j] * number_of_neighbors_a(i).
```

The last term removes the same neighbor from occupying both legs. Its contribution
is one for **every** ell by the addition theorem at angle zero. It is not confined
to the monopole. Center exclusions are applied before accumulating harmonics.
No neighbor-pair or triangle list is stored or traversed by the estimator.

With fixed radial/angular choices, work per primary is proportional to the number
of returned neighbors, rather than its square. More explicitly it is
`O((Nneighbor + Nbin^2) * (ellmax+1)^2)`, in addition to tree lookup/build cost.
Memory per rank is two replicated catalogues/trees, an `8*(ellmax+1)*Nbin^2` double
moment array plus the output, `O(Nbin*(ellmax+1)^2)` harmonic buffers and a reusable
neighbor list. Existing KD-tree queries may allocate internal scratch; no measured
allocation reduction or production speedup is claimed here. MPI parallelizes primaries
but does not remove replicated-catalogue memory or guarantee optimal load balance.

## Build and validation

```
cd FML/ThreePointCorrelation/tests
make USE_MPI=false
make run USE_MPI=false
# With an MPI compiler/configuration:
make clean
make run USE_MPI=true NP_LIST="1 2 4"

cd ../example
make USE_MPI=false
./threepoint
```

Compiler and library paths follow `FML/Makefile.config` / `Makefile.local`.
The example is an independent uniform Poisson toy; its nonzero estimates are finite
sample noise, not a calibrated cosmological prediction. Only rank zero prints.

The regression compares all eight mixed-catalogue moments and the normalized
contrast to independent explicit triplet/Legendre enumeration through ell=16.
It also tests analytic collinear/right triangles, diagonal self-products, shell
boundaries, monopole partition sums, coordinate transformations and input failures.
Independent uniform mocks test a zero contrast and a nonzero raw disconnected DDD
baseline. MPI runs additionally test rank-dependent valid and invalid inputs.
See [VALIDATION.md](VALIDATION.md) for measured results and limits.

Remaining #33 work includes fixed/local-LOS anisotropic statistics, bispectrum
multipoles, weighted/survey estimators with proper edge mixing, distributed catalogue
ownership, OpenMP/load balancing and independent cosmological/reference-code validation.
