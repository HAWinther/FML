> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# Dense-particle PM validation (#52, PM-S1)

`ParticleMeshForwardT` now permits an IC/particle lattice **N larger than the
force mesh M**. There is still one particle per IC cell: this does not introduce
an independent particle-count parameter. The initial physical Fourier band must
lie strictly below both Nyquists. The equal-mass CIC deposit/gather, continuous
spectral force, ZA start and KDK evolution are unchanged.

[Results](RESULTS.md) record passing bounded convergence criteria with dense
sampling. The [original sparse-particle study](../pm_resolution_validation/RESULTS.md)
and its failed criteria are preserved. Dense sampling improves that experiment;
it is not a universal choice of N/M or a production-accuracy guarantee.

## Experiment

This follows the original study's matched 18 real initial Fourier coordinates,
box length 1, P=0.002, IC k<10, observed k<15, noise power 1e-4, Omega_m=0.3 and
scale factors 0.2 to 0.6. The same noisy N=16/M=32/16-step reference data are reused
within this campaign. Physical initial coefficients are matched across N to 2e-12.
The observed band has 56 real degrees of freedom.

| Family | Fixed | Trials | Finite reference |
| --- | --- | --- | --- |
| Force | N=64, 16 steps | M=8,16,32,64 | M=64 |
| Particle/IC | M=16, 16 steps | N=16,32,48,64 | N=64 |
| Particle/IC | M=32, 16 steps | N=32,48,64 | N=64 |
| Time | N=32, M=16 | 4,8,16,32 steps | 32 steps |

There are 12 unique comparisons plus the data-generation run. Each family varies
one factor and uses its own finest finite reference. Density norms include rFFT
Hermitian weights. Gradients are projected onto the same 18 orthonormal initial
Fourier coordinates plus log-amplitude/noise, rather than comparing differently
normalized raw white-grid derivatives. The original finest-correction thresholds
remain field <5%, projected gradient <15%, and density discrepancy <0.1 noise SD
per observed real degree, with improvement over the coarsest trial.

At N=32/M=16/16 steps, independent forward-value central differences check a field
and a mixed field/scalar direction over eight h values. Multiple refined values
must agree (best last-four scaled error <2e-5; at least two <1e-4). The independent
NumPy CIC/FFT/gather and native tiny-kick probes now include N>M. Both kick sizes
and both displacement amplitudes are retained. No new long inference chains are
run in this follow-up.

## Reproduce

Build the existing bounded native drivers with the normal compiler/FFTW overrides:

```sh
make -C FML/FieldLevel/pm_resolution_validation pm_resolution pm_force_probe USE_MPI=false
python3 FML/FieldLevel/pm_oversampling_validation/run.py --bin-dir /path/to/binaries --output /path/to/fresh/OUT --jobs 2
```

NumPy and Matplotlib are required. This explicit campaign is outside routine CI.
The force probe uses one rank. `--phase cases` generates native comparison outputs;
`--phase force` reruns the independent force checks against existing cases;
`--phase analyze` regenerates both analyses from existing outputs. Analysis writes
reports/plots and returns nonzero for failed criteria. Do not overwrite the old
study's report or use its analysis script for these different refinement families.
Logs and exact argument vectors are saved by phase.

The PM unit/integration tests cover both mesh orderings:

```sh
make -C FML/FieldLevel/tests particle-mesh particle-mesh-adjoint particle-mesh-posterior particle-mesh-checkpoint USE_MPI=true NP_LIST="1 2 4"
```

Use clean builds when changing MPI/OpenMP/compiler flags. See [RESULTS.md](RESULTS.md)
for the actual serial/MPI/thread/sanitizer scope and remaining limitations.
