# Initial validation and performance, 28 October 2026

GCC 12, C++17, `-O2 -Wall -Wextra`, OpenMP; MPICH for MPI. Serial sanitizer build uses Clang 19 with AddressSanitizer and UndefinedBehaviorSanitizer, including leak detection. Builds use the repository `FML/.clang-format` (clang-format 19). These are small local-machine checks, not production performance or force-error guarantees.

## Validation

- New force tests pass in serial/OpenMP and on 2 and 4 MPI ranks (2 threads per rank).
- Independent cubic-spline enclosed-mass quadrature agrees with the implemented softened force; the Gaussian and softened split recombines correctly inside and outside the support.
- Independent periodic Ewald sums agree with filtered FML PM plus short-range forces at multiple separations, orientations, split scales and softening supports. Worst measured relative error: **2.95e-11 in double**, **1.83e-5 with float FFT grids**, including weak corner forces. The float test explicitly allows additional FFT rounding; the double criterion is unchanged.
- Cell lists and exact-opening octrees agree with direct sums to summation precision for uniform, clustered, periodic, coincident and zero-mass cases. Cell-list results are identical across tested thread counts. Finite-opening trees accept nodes and converge toward direct forces as the opening tightens.
- Ghost tests cover selective exchange, cutoffs wider than a slab, empty particle slabs, periodic boundaries, multiple message chunks and collective invalid-input rejection; no duplicated source IDs are introduced.
- Length and mass unit rescaling preserves the dimensionless FML acceleration.
- AddressSanitizer/UndefinedBehaviorSanitizer and the existing NBody and particle regression executables pass. The entire unrelated library suite was not rerun for this isolated module addition.

## Serial/OpenMP benchmark

8000 particles, unit box, `rs=.02`, spline support `.005`, cutoff tolerance `1e-6`; tree opening `.3`, leaf capacity 8. Clustered means half the particles occupy a cube of side .01, with the other half uniform. Median of three separate process runs, with no concurrent builds or tests. Cell/tree times include construction and force evaluation. The direct reference scans all 63,992,000 directed nonself pairs.

| Threads | Distribution | Direct (s) | Cell list (s) | Octree (s) | Cell speedup over direct |
|---|---|---|---|---|---|
| 1 | Uniform | 0.9609 | 0.1196 | 0.1607 | 8.03x |
| 1 | Clustered | 1.1795 | 0.6565 | 0.8438 | 1.80x |
| 4 | Uniform | 0.2359 | 0.0278 | 0.0428 | 8.49x |
| 4 | Clustered | 0.2681 | 0.1530 | 0.2182 | 1.75x |

The cell list visits 7,998,090 candidates for the uniform case and 22,033,866 for the clustered case. The tree visits 2,223,540 and 16,817,870 leaf candidates respectively, plus accepted monopoles. The uniform tree result is effectively exact in this case; clustered relative RMS error is 3.64e-5. **The conservative tree is slower than the cell list in both small cases.** Fewer leaf pairs do not automatically compensate for traversal, identity lookup and construction costs. Tree acceptance/traversal tuning and larger dense-halo cases are needed before claiming a TreePM advantage. Cell-list complexity still deteriorates with clustering or a cutoff covering many cells.

## MPI benchmark

Same 8000-particle cases; 2 OpenMP threads per rank, so total CPU resources increase with rank count. Maximum-rank phase time, median of three runs; barriers before timed phases prevent the preceding direct reference's load imbalance from being charged to ghost exchange. Timings exclude those barriers. The benchmark alone retains a replicated input for its independent direct reference; the library ghost implementation exchanges only required sources.

| Ranks | Distribution | Ghost exchange (s) | Cell build + force (s) | Tree build + force (s) | Max ghost payload (MiB) |
|---|---|---|---|---|---|
| 1 | Uniform | 0.0014 | 0.0627 | 0.1032 | 0.0000 |
| 1 | Clustered | 0.0003 | 0.3466 | 0.4407 | 0.0000 |
| 2 | Uniform | 0.0016 | 0.0269 | 0.0425 | 0.0952 |
| 2 | Clustered | 0.0005 | 0.3067 | 0.4327 | 0.0481 |
| 4 | Uniform | 0.0039 | 0.0189 | 0.0312 | 0.0990 |
| 4 | Clustered | 0.0009 | 0.3223 | 0.4507 | 0.2003 |

Uniform cell-force work scales usefully in this small run, with exchange overhead becoming more visible. The clustered case does not scale well because the compact halo stays in one spatial slab. A gravity tree alone does not solve domain load imbalance. Maximum-rank tree RMS error remains below 5e-5 for these cases. Ghost payload measures received particle count times `sizeof(Particle)`, excluding capacity and all other workspace; it is not RSS.

Further validation should cover mesh assignment/interpolation convergence for off-grid sources, force-error percentiles including weak forces, larger clustered MPI workloads and total peak workspace/RSS. COLASolver force assembly and time integration remain backlog item 47.

## Newtonian-plus-Yukawa extension

The Newtonian model remains the default; `Kernel(options, NewtonYukawa{alpha, lambda})` selects the additional model. Newtonian compatibility was checked against commit `c2413cf` using hexadecimal double output for 3000 radii over three split scales, plus Fourier multipliers and cutoffs (12,003 values in total): the outputs match bit for bit with GCC 12 `-O2`. Explicit Newtonian and zero-alpha Yukawa paths also agree exactly in the regression test.

The Yukawa kernel has 100-digit independent Ewald reference data covering 30 combinations of radius and range. These check both component kernels and Newtonian-minus-Yukawa differences. The maximum scaled long-kernel reference error is 1.76e-13. Numerical potential derivatives validate forces at two softening supports, including the evaluation switch at `q=1`. Small/large range limits and the small physical remainder at `alpha=-1, lambda=1e8` are tested explicitly. The numerical Yukawa softening is defined through the spline effective separation; it is not the field of an extended spline density.

The combined filtered PM force and short correction agrees with an independent periodic Yukawa image sum. The maximum error, scaled by `max(1, norm(alpha*Yukawa_reference))`, is about 1.72e-13 in double and 4.39e-6 with float FFT grids. This is a component-scaled error, not a relative guarantee for a cancelled total force. The periodic reference tests use multiple split scales, ranges and both positive and negative strengths; the existing Newtonian periodic Ewald checks remain unchanged.

The clustered, 1000-particle tree test now requires actual accepted monopoles and checks both opening angles .3 and .1 with leaf capacity 1. The worst error scaled by `max(1, norm(Newtonian_direct_force))` is 0.00477 for .3 and 0.000361 for .1. The coarser test accepts 97,786 nodes across its parameter cases. Thus tighter opening improves accuracy in this case; a geometric opening angle alone does not guarantee relative accuracy near force cancellation. The default leaf capacity remains 16. Cutoffs bound absolute component tails and MPI exchange checks model/strength/range agreement collectively.

Validation includes serial/OpenMP, float FFT grids, 2/4 MPI ranks, AddressSanitizer/UndefinedBehaviorSanitizer with leak detection, and the existing NBody and particle regressions. Whole-library tests outside this module and those adjacent regressions were not rerun. New tests are registered in the main suite and standalone module Makefile. The same FFT count is used for the new PM operator; extra special-function/series work occurs in pair kernels, and short-range Yukawa tree acceptance is deliberately more conservative. No production Yukawa performance claim is made.
