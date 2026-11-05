# Initial periodic 3PCF validation

Recorded 5 November 2026. This validates the initial isotropic scope of #33;
anisotropic estimators, survey geometry and production cosmological calibration remain.

## Builds and execution

The regression passes 27 checks per serial/one-rank run with Apple Clang, GCC 12,
ASan/UBSan, and OpenMPI/Clang17. Two- and four-rank runs pass 30 checks each, including
three collective inconsistent-input failures. MPI builds cover automatic FML setup
with OpenMP enabled elsewhere and explicit `NO_AUTO_FML_SETUP` initialization without
OpenMP. The estimator's primary loop itself is serial within each rank.

No FFTW/GSL dependencies are used. Both new Makefiles build/run with the shared FML
configuration in serial and MPI modes. The example's 30 output rows agree across
compilers and 1/2/4 MPI ranks within 1e-10 absolute zeta (actual differences are saved
in the accompanying validation artifact). These are replicated-catalogue runs, not
MPI slab ownership tests.

## Independent checks

- All eight D/R orientations, all radial-bin pairs and ell=0–16 agree with explicit
  ordered distinct-triplet enumeration using independent Legendre recurrences.
  Raw-moment comparisons use 2e-10 scaled by max(1, absolute moment). The normalized
  contrast is independently assembled with factorial counts and shell volumes.
- A collinear three-object catalogue has DDD moments 6 for even ell and 2 for odd
  ell through ell=16. A right triangle has dipole 2 sqrt(2). These check angular
  normalization independently of enumeration.
- A shell containing only an isolated pair gives zero DDD for every ell: the same
  neighbor cannot fill both legs. Center exclusions, half-open radial edges,
  coincident coordinates with distinct IDs and monopole shell partition sums are
  checked separately.
- Input permutation, rigid box rotation, periodic translation/boundary crossings
  and radial-leg exchange preserve the results within roundoff.
- Invalid/nonfinite positions, duplicate IDs, undersized catalogues, unsupported
  orders, nonincreasing/zero/half-box edges and unrepresentable distance/shell
  scales are rejected. MPI tests check inconsistent options, inconsistent valid
  catalogues and invalid input on only one rank.

## Independent uniform ensemble

48 independent uniform data/random pairs (100 data and 160 random points) use a
unit box, one shell [0.04,0.32) and ell=0–2. The mean and standard error across mocks
are approximately:

| ell | Mean contrast multipole | Ensemble SEM |
|---|---:|---:|
| 0 | -0.00112508 | 0.00378106 |
| 1 | 0.00383853 | 0.00684294 |
| 2 | 0.00507323 | 0.00668791 |

All are consistent with zero using the explicit five-SEM-plus-0.005 regression
criterion. The raw DDD monopole separately retains its nonzero analytic uniform
baseline f_shell² (tested within 8% on the ensemble mean). This is a negative
control against interpreting raw triplet clustering as the connected contrast.
These mocks establish unclustered normalization, not nonlinear cosmological recovery,
error covariance or a science-quality random-density prescription.

## Small timing illustration

An opt-in scratch comparison uses 140 data and 210 random points, three shells
[0.02,0.14,0.26,0.4] and ellmax=4, Apple Clang -O3. Harmonic and explicit-triplet
raw moments agree. Timings are included in the exported review artifact. The
harmonic path includes validation, KD-tree construction and MPI-serial reduction;
the direct path is the deliberately simple test oracle, not an optimized established
code. This is an illustration of avoiding neighbor-pair enumeration, not a production
benchmark or a guarantee of MPI/OpenMP scaling. No peak-RSS or allocation speedup
was measured.

## Remaining validation and extensions

Cosmological analytic/reference-code comparisons, weighted catalogues, random-density
and scale convergence, covariance/error calibration, survey edges, anisotropic LOS
statistics, bispectrum multipoles, distributed catalogue memory and thread/load
scaling remain. Fixed finite-volume catalogue normalization also retains the usual
mean-density/integral-constraint considerations described in the README.
