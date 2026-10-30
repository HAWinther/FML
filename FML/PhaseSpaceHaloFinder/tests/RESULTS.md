# Initial phase-space finder validation, 28 October 2026

This records a first independent single-snapshot implementation. It is not a production completeness/purity calibration or a claim of ROCKSTAR equivalence.

## Regression and manufactured-truth checks

GCC 12 `-O2 -Wall -Wextra`, OpenMP; MPICH with 2 threads per rank for MPI. Serial sanitizer build: Clang 19, AddressSanitizer plus UndefinedBehaviorSanitizer, with leak detection. C++ is formatted using `FML/.clang-format` and clang-format 19.

- Standalone and main-suite targets pass in serial/OpenMP, on 2 and 4 MPI ranks and under sanitizers.
- Existing FoF, KD-tree, Newtonian short-range gravity and Yukawa regression executables pass after the shared spline-potential extraction. The entire unrelated FML test suite was not rerun.
- Exact-opening isolated binding potentials agree with independent direct pair sums, excluding self potential; unequal-mass pair energies and circular-velocity shells agree with analytic results.
- Two velocity-separated but spatially overlapping 200-particle populations are recovered as two objects with their correct memberships. Particle permutation, axis rotation, velocity boost, periodic translation, unit rescaling and thread-count changes preserve the tested memberships.
- The smooth-host/cold-subhalo case retains the 150-particle subhalo, recovers 772 of 800 injected host particles in its eligible FoF parent and removes all five high-energy debris particles. The density-saddle check removes a false extra seed that appeared without that check. This is one controlled case, not a universal false-positive bound.
- Isotropic Plummer realizations at 1000 and 4000 particles give one recovered object, bound masses .968 and .95975 (input total mass 1), and Vmax 5.24675 and 5.13085 versus the untruncated analytic value 5.06557. Vmax deviations are about 3.6% and 1.3%; the initial FoF boundary omits some outer particles. Plummer membership agrees exactly between the complete serial and distributed analyses on tested ranks. Normal-distribution fixture streams can differ between C++ standard libraries, so the Clang fixture realization need not be identical to GCC's.
- MPI tests cover periodic and multi-slab FoF links, chains wider than a slab, empty particle ranks, tiny message chunks, complete-parent routing, local assignment return, SO-seed spatial routing and global ID uniqueness. Option mismatch, parent-size and iteration/work limits fail explicitly.
- The text-input example smoke test writes two halo rows and 400 particle assignments. FML particle import, explicit HOD mass/radius and exclusive halo-history adapters are exercised.

## Planted subhaloes (29 October 2026)

Host Plummer sphere (8000 particles, scale radius .02), two field haloes, a subhalo (400 or 800 particles, scale
radius .004) on a circular orbit at .03-.08 from the host centre and 20000 background particles (G = 1, box 1,
softening .002), 48 realizations. With the symmetric union of Poisson-consistent seeds the subhalo was found in
34; with one-way merging into the most consistent larger seed in 40 (no case lost; one more spurious extra object
over the 48 runs). The test suite contains one of the recovered cases. The remaining misses are subhaloes at
1.5-2.5 host scale radii: there a lower-level 6D group that contains the subhalo and many host particles assigns
all of them to the subhalo seed, whose inflated phase-space scales then win the host particles at the next level,
and the host-dominated membership is self-bound. Recovery of subhaloes deep inside host cores is therefore not
reliable yet. The finder took 0.9, 2.1, 10 and 32 s (2 threads) for 3.2e4, 6.3e4, 1.3e5 and 2.5e5 particles of
these configurations, roughly N^1.6, so very large parents (the 1e6 default `max_parent_particles`) are slow.

## Optional external ROCKSTAR comparison

Reference: [yt-project/rockstar at 10869df8e16ca4da2f4a929bccd00c4bcae937fd](https://github.com/yt-project/rockstar/tree/10869df8e16ca4da2f4a929bccd00c4bcae937fd), reporting version 0.99.9-RC3. It was built outside FML with GCC 12, C99 `-O2 -fcommon`, without source modifications. No external source is bundled. The reproduction harness is `compare_rockstar.py` plus the optional `rockstar_fixture` target.

The same isolated 4000-particle Plummer realization has scale radius .015 Mpc/h and truncation radius .15 Mpc/h. Finite truncation and normalization slightly modify the analytic equilibrium. It uses boxsize 1 Mpc/h, a=1, Omega_m=.3, Omega_Lambda=.7, h=.7 and mass per particle 20,815,247.025 Msun/h. FML uses G derived from `FML/Units/Units.h`. Both use initial linking parameter .28 and input resolution/support scale .0005 Mpc/h. The single-process ROCKSTAR calculation is nonperiodic; this cloud is away from the box boundaries. Softening, centring, unbinding and mass-boundary conventions differ. Particle-ID overlap against ROCKSTAR was not measured.

| Quantity | FML | ROCKSTAR |
|---|---|---|
| Objects | 1 | 1 |
| Reported mass (Msun/h) | 7.9909733e+10 (exclusive isolated-bound) | 8.014e+10 (bound M200b) |
| Vmax (km/s) | 97.09516431 | 97.05143700 |

Vmax differs by 0.0451%; reported masses differ by -0.2873% despite their distinct definitions. Centre separation is 0.00046179 Mpc/h. This catches gross errors in a controlled case; it does not validate cosmological halo statistics or subhalo purity.

## Cosmological comparison with ROCKSTAR (30 October 2026)

COLASolver snapshot: 128^3 particles in 64 Mpc/h, force grid 384^3 (cell 0.167 Mpc/h), 40 steps from z = 30,
z = 0, m_p = 1.10e10 Msun/h. ROCKSTAR as above (header Omega_Lambda set to 1 - Omega_m; ROCKSTAR rejects the
radiation term) with FILE_FORMAT = "GADGET2", FORCE_RES = .1667, MASS_DEFINITION = "200b", b = .28,
MIN_HALO_OUTPUT_SIZE = 20, FULL_PARTICLE_CHUNKS = 1. FML: `example/phase_halos parts out 64 .1667 4.30091e-9 1 100`
(Mpc/h, km/s, Msun/h, Hubble flow in the binding). `tests/compare_rockstar_cosmo.py` converts the snapshot and
compares (mutual best particle overlap; N = bound particles: FML inclusive bound, ROCKSTAR mbound_200b; ROCKSTAR
hosts are not inside R200b of a more massive halo). This is a PM-resolution snapshot: haloes are softened at the
cell size and subhaloes are not reliable, so it tests host detection and Vmax, not subhalo completeness.

| | FML | ROCKSTAR |
|---|---|---|
| Objects (hosts) | 5685 (4666) | 2915 (2268) |
| Time (example program, including text input) | 30 s (2 threads), 34 s (1 thread); was 71 s | 7 s (1 process) |

| Hosts, N bound | ROCKSTAR hosts matched | median Vmax FML/ROCKSTAR (default; exclusive) | FML hosts matched |
|---|---|---|---|
| 50-100 | 95.0% | 1.010 | 64.6% |
| 100-1000 | 96.3% | 1.004; 1.001 | 97.6% |
| > 1000 | 96.8% | 0.976; 0.936 | 98.6% |

Vmax functions N(>V) above 150, 200, 300, 500 km/s: hosts 724/797, 362/395, 125/142, 22/29 (exclusive Vmax:
707, 336, 111, 14); all objects 952/1010, 471/489, 162/167, 32/33.

- Hosts with more than 100 particles are found by both (96-99% matched both ways).
- The exclusive Vmax was 6% low for massive hosts because subhalo members are excluded; the default now follows
  ROCKSTAR (object plus descendants, softening floor, density limit). The remaining 2.4% for hosts above 1000
  particles: ROCKSTAR checks binding once against the combined host + substructure potential, our members are
  unbound object by object (Vmax from all particles around our centres is 1.005 of ROCKSTAR's).
- 2400 extra FML hosts with 20-50 particles have no particles in a printed ROCKSTAR halo; ROCKSTAR finds the
  same groups (2499 internal haloes, median 30 particles) but does not print them (fewer than 20 bound particles
  inside R200b; the COLA haloes are diffuse at the force resolution). Output definition, not detection.
- Subhaloes: about a third of ROCKSTAR's 20-100 particle subhaloes are matched; not interpretable at this
  resolution. A real N-body snapshot is needed for the core-subhalo question.
- ROCKSTAR's Vmax is 0 for small haloes (no radius above the density threshold); so is the default FML Vmax.
- Speed (finder only, 1 thread): 71.5 s before, of which 57.8 s binding potentials: the unbinding passes after
  the first recomputed the full tree potential (39 of 49 s) for large objects losing a few particles per pass, and
  the opening angle .3 makes 1e3-1e4 particle objects close to direct sums (3300 interactions per particle);
  std::pow in the softened Green function took 13%. Now later passes subtract the removed particles directly while
  that is cheaper than a new tree, and the Green function uses multiplications: 24.7 s (20.0 s on 2 threads),
  2.1e9 instead of 9.4e9 interactions; 2 of 5686 objects change membership (at most .05% of their particles).
  ROCKSTAR's looser monopole criterion (dmin = bmax + sqrt(bmax^2 + <x^2>), tested, not adopted) gave 18.7 s but
  changed 245 objects by up to 11% of their particles; opening angle .7 gave 21.6 s and changed 26 objects by up to
  1.7%. The remaining time outside the potential (about 14 s, FoF and 6D refinement) runs on one thread.

## Small local performance cases

Median of three separate process runs, without our own concurrent builds/tests. Full `find_distributed` pipeline, including validation, label convergence, routing, hierarchy, unbinding and properties. The input is one Plummer parent; resolving many parents and the largest-parent load imbalance is a separate scaling problem. Peak RSS includes runtime/startup, outputs and the synthetic fixture; it is process peak across the increasing-size sequence, not isolated workspace. Every rank generates the same fixture before selecting its owned particles for this benchmark; the library does not all-gather those particles. Timings on a shared machine are approximate.

| Particles | Serial, 1 thread (s) | Serial, 4 threads (s) | Peak RSS, 4-thread process (MiB) |
|---|---|---|---|
| 1000 | 0.0376 | 0.0318 | 5.13 |
| 4000 | 0.3214 | 0.2277 | 10.02 |
| 16000 | 3.1029 | 1.6177 | 25.45 |

| Particles | MPI ranks (2 threads each) | Maximum-rank time (s) | Max ghosts | Max particles routed to one owner | Max process peak RSS (MiB) |
|---|---|---|---|---|---|
| 1000 | 1 | 0.0346 | 0 | 1000 | 26.96 |
| 1000 | 2 | 0.0384 | 473 | 978 | 29.04 |
| 1000 | 4 | 0.0395 | 473 | 971 | 28.90 |
| 4000 | 1 | 0.2626 | 0 | 4000 | 31.48 |
| 4000 | 2 | 0.2662 | 1549 | 3917 | 33.34 |
| 4000 | 4 | 0.2447 | 1549 | 3873 | 32.60 |
| 16000 | 1 | 2.1895 | 0 | 16000 | 47.46 |
| 16000 | 2 | 2.2184 | 4855 | 15702 | 48.16 |
| 16000 | 4 | 2.1998 | 4855 | 15511 | 43.54 |

All cases recover one object. A single parent remains on one rank, so more MPI ranks do not materially accelerate its phase-space analysis. OpenMP accelerates potential evaluation; much hierarchy and property work remains serial. Work in these small cases grows faster than particle count, and repeated potential evaluations are substantial. More aggressive spatial FoF linking, reusable query buffers, faster potential/centre evaluation, work-balanced parent assignment and distributed processing of exceptionally large parents are worthwhile future performance work. The last requires a different ownership/work distribution contract.

Production follow-ups: cosmological ROCKSTAR catalogue/ID comparisons, completeness and purity across resolved mergers and tidal stripping, calibrated seed thresholds, temporal stabilization, SO or tidal host boundaries, orphan reassignment conventions, percolating parents, large-parent load balancing and total-memory controls. The current output is a documented isolated-bound, exclusive-membership catalogue, not a general SO/virial catalogue.
