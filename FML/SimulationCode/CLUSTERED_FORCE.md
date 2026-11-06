# Clustered short-range force validation (item 47)

This is a controlled **static GR force** experiment. It exercises the model-owned
mesh solve and particle kick with mass-weighted PCS deposition/interpolation,
window deconvolution and continuous Fourier derivatives. It does not evolve halo
orbits, establish equilibrium, calibrate modified gravity, or complete item 47.
No production solver behavior changes in this milestone.

## Fixtures and independent reference

The default has 2,048 unequal-mass particles and 128 reference targets in each of
four reproducible catalogues: a concentrated truncated Plummer-profile halo, two
overlapping halos, a halo across the periodic x boundary, and a translated copy
of the first halo. Radial scale is 0.0125 and truncation radius 0.14 in box units.
The merger centers are separated by 0.09. These describe generating profiles;
particle masses vary independently and the samples are not claimed to be
virialized. Targets use stratified particle IDs with deterministic hash offsets,
so both merger populations can be sampled. Each source contributes to each
sampled reference force.

The reference is a separate Ewald implementation using direct real-space image
sums and reciprocal-space source structure factors. It does not call production
short-range kernels. Compact spline softening is obtained by integrating its
normalized radial density with piecewise Gauss-Legendre quadrature, rather than
calling the production softened-force polynomial. The default compares inverse
splits 5 and 7, Fourier bounds 10 and 16, and real images -1..1 in every dimension;
the result is also checked for one target against the older pairwise Ewald oracle
(inverse split 6, modes -12..12 and images -2..2). Softening corrections are
included in that cross-check. The simulation split is unrelated to these choices.

Box size is 100 comoving Mpc/h and epoch a=0.8 in LCDM with OmegaM=0.3. The force
normalization is C/(4*pi*M), C=3*OmegaM*a/2, with M the global source-mass sum.
The benchmark reads the short acceleration through a unit model-owned kick;
it does not interpret that unit kick as a physical trajectory.

## Measured quantities and regression gates

Twelve cases per fixture cover unsplit PM, 16³/32³/64³ cell solves, two-way
interlacing at 32³, exact tree and opening angles 0.3/0.6, split lengths 3/4/5
Mpc/h, spline supports 0.25/0.5 Mpc/h and analytic/default-tabulated short kernels.
The cutoff tolerance is 1e-8. Sampled close-pair counts confirm that compact
softening is actually exercised in the full-size experiment.

The summary CSV contains relative RMS force error, median/95th/99th/maximum
particle errors, 95th-percentile direction error, and normalized momentum imbalance
`|sum m_i a_i| / sum m_i |a_i|`. Momentum uses **all particles**; reference error
statistics use the sampled targets. Per-particle CSVs retain radius, mass, full
reference/actual vectors, direction error, and absolute error divided by reference
RMS. Per-particle relative errors use denominator `max(|a_ref|, 0.001*a_ref_RMS)`;
the absolute-error statistic remains interpretable near force cancellations.

Gates require independent-reference agreement below 1e-10 relative RMS, exact
tree/cell agreement below 1e-12, exact-backend momentum imbalance below 1e-10,
resolved cell/interlaced RMS below 0.005 with 99th-percentile error below 0.02,
and clear mesh convergence/improvement over PM. Finite-tree budgets are RMS
0.005/0.02 and 99th-percentile 0.025/0.1 for angles 0.3/0.6. These are regression
gates for these fixtures, not universal force-error guarantees. The full-size
suite also requires angle 0.6 to differ measurably from exact sums, preventing
an accidentally exact traversal from masquerading as approximation validation.

The MPI runner requires numerical rank invariance for exact backends, but records
and separately bounds finite-opening tree differences: accepted nodes can depend
on rank layout. Test catalogues and references are deliberately replicated; the
benchmark's small diagnostic force arrays are reduced across ranks. This is not
an MPI memory-scaling or performance benchmark. Per-case wall times exclude the
reference construction and are diagnostic only.

## Build and run

The experiment is optional, outside the default fast tests and CTest suite:

```sh
cd FML/SimulationCode/tests
make benchmark_clustered_force USE_MPI=false
python3 regression/clustered_force.py --binary ./benchmark_clustered_force \
    --output /tmp/fml-clustered-serial --plot

# Clean/rebuild before changing compiler, MPI or precision options.
make clean
make benchmark_clustered_force USE_MPI=true
python3 regression/clustered_force.py --binary ./benchmark_clustered_force \
    --output /tmp/fml-clustered-mpi --mpi --launcher mpirun
```

`--plot` needs matplotlib; all other runner functions use the Python standard
library. The MPI command runs 1/2/4 ranks. CMake users explicitly build target
`benchmark_clustered_force` and use `build/tests/benchmark_clustered_force`.
For a short plumbing check use `--particles 128 --targets 16`; small fixtures do
not necessarily exercise tree approximation or close softening. Supported ranges
are 128..16,384 particles and 16..N targets; large reference runs can be expensive.

Outputs retain raw forces, summary distributions, logs, exact commands, binary
and relevant source hashes, Git revision/status and MPI comparisons. Optional
PNG/PDF figures show mesh convergence, error CDFs and radial tree errors. The
saved Git revision alone does not identify uncommitted source; use the hashes too.

## Initial serial results (double FFT grids)

| Fixture | Cell RMS, 16³ | Cell RMS, 64³ | Tree RMS, theta=0.6 | Tree momentum imbalance |
|---|---:|---:|---:|---:|
| halo | 0.000285 | 5.72e-07 | 0.000286 | 5.71e-06 |
| merger | 0.00039 | 6.97e-07 | 0.000304 | 8.75e-06 |
| boundary | 0.0003 | 6.04e-07 | 0.000244 | 7.36e-06 |
| shifted | 0.000302 | 6.07e-07 | 0.000261 | 1.05e-05 |

The full suite passes 190 checks in serial and MPI 1/2/4 ranks.
An address/undefined-sanitized 128-source/16-target smoke run passes 138 checks
(new driver/header code instrumented; prebuilt background objects uninstrumented). Across MPI
rank counts, per-case sampled-force differences are below 2.4e-15 relative RMS.
The two independent Ewald settings differ by less than 2.0e-14 relative RMS.
The baseline sampled targets include 172 or 361 directed softened pairs, depending
on fixture, confirming that these are not exclusively point-force tests.

At angle 0.3 these particular fixtures still traverse to exact leaf sums; their
agreement does not demonstrate useful tree acceleration. Angle 0.6 does exercise
approximation and shows a small but measurable instantaneous momentum imbalance.
It is not a measurement of accumulated center-of-mass drift over an orbit.
The 64³ cell 99th-percentile errors are below 1e-5 on these fixtures.

Controlled small-N orbit/cluster evolution against a converged direct-reference
integrator is now covered in [DYNAMICS.md](DYNAMICS.md). Production halo evolution,
finite-opening dynamical drift, broader force-error distributions and tree/ghost
performance calibration remain follow-ups.
