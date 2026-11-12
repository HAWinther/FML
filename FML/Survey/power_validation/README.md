# Survey Fourier multipoles: external reference and convergence

This optional experiment extends backlog item 25 beyond the small direct-DFT regression. It compares weighted fixed/local-LOS multipoles to the actual **pypower** implementation, checks the declared shot-noise prescription independently, and measures convergence toward an unpainted catalogue DFT. It does not require a simulation snapshot or modify the estimator's scientific formulas.

## Run

Build the driver with the normal FML compiler/FFTW settings:

```
make -C FML/Survey/power_validation USE_MPI=false
python3 FML/Survey/power_validation/regression/power_reference.py \
    --binary FML/Survey/power_validation/survey_power_catalog --output power_validation
```

The default experiment requires NumPy. Add `--pypower` in an environment with the cosmodesi [pypower package](https://github.com/cosmodesi/pypower), pmesh and their MPI/FFT dependencies installed. The similarly named electrical-power package is unrelated. The tested external sources are pinned in [RESULTS.md](RESULTS.md); installing the reference dependencies is optional and separate from building FML. Use normal Python execution, without `-O`, because validation assertions must remain enabled.

For FML MPI agreement, build a separate MPI driver (clean between build configurations), and pass:

```
--mpi-binary /path/to/mpi/survey_power_catalog --launcher "mpirun"
```

This runs FML at 2 and 4 ranks and compares its vectors to the serial driver. Open MPI users may need `--oversubscribe`; MPICH users should omit it. The pypower comparison itself is a single-process run; MPI consistency is checked on FML. All commands have bounded subprocess timeouts.

## What is compared

The seeded fixture contains 768 data and 2,304 random points in an asymmetric Cartesian survey region, with unequal weights, spatially varying supplied nbar and a deterministic modulation of data weights. It is a numerical fixture, not a calibrated cosmological mock. The experiment runs meshes 32 and 64, local first-point or oblique fixed LOS, with and without CIC compensation. It covers I2 derived from random nbar as well as an explicit override.

pypower performs its own catalogue CIC painting and spherical-harmonic FFT multipoles. The comparison deliberately matches:

- Positive-k bin edges, FFT box/node locations and observer (pypower coordinates are translated so its observer is the origin).
- `interlacing=0`, CIC painting, double precision, no mode oversampling.
- Explicit `wnorm=I2`; this checks FML's result against the same independently calculated normalization, not against pypower's different automatic mesh-based normalization convention.
- Raw multipoles before any shot subtraction. With compensation enabled, pypower receives `compensations='cic'` on its uncompensated painted RealField, matching FML's sinc-squared amplitude correction. Its default noninterlaced CatalogMesh uses `cic-sn`, a different transfer function.

For an oblique fixed LOS, pypower's explicit mu bins can lose exactly parallel modes when floating-point mu rounds just outside [-1,1]. The runner expands only the two mu endpoints by `16*epsilon_double`, retaining those modes without editing external source code. Mode counts and mean k must then match exactly/to roundoff. This workaround and the reference commits are part of the recorded comparison convention.

Raw spectra agree to roundoff. **Default corrected spectra are not claimed to agree**: FML reports finite-lattice higher-multipole contact terms and aliased CIC noise, while pypower's usual constant shot subtraction has a different convention. The NumPy oracle independently evaluates the FML-declared contact prescription using explicit per-object Legendre factors and full signed Fourier-mode sums. It checks weights squared, alpha squared, angular factors and the CIC alias/compensation factor. This validates the stated prescription, not an exact self-pair removal from each finite CIC realization or a reconstruction-induced noise correction.

For continuum convergence, the oracle evaluates the unpainted catalogue Fourier sums at the same signed modes, with LOS at each object. FML's LOS is evaluated at mesh nodes. The reported error is the RMS raw multipole difference divided by the continuum raw **monopole** RMS; this avoids dividing by nearly zero higher multipoles. Meshes 32/64/128 use identical catalogues, physical box and k bins. A common translation of catalogues, observer and box minimum is also tested.

Input/output files, command logs, pypower arrays, summary JSON and binary/runner/input hashes are written under `--output`. Preserve the driver/source version and external package commits with these artifacts. Timings in the logs are diagnostic timings, not a scaling benchmark.

## Driver text format

`survey_power_catalog input.txt output.csv` reads:

1. `nmesh boxsize`
2. Three box-minimum coordinates.
3. Three observer coordinates.
4. `local_los fixed_los_x fixed_los_y fixed_los_z` (boolean is 0/1).
5. `i2_override compensate_cic aliased_shotnoise number_of_edges`
6. All k edges.
7. `number_of_data number_of_randoms`
8. Data rows, then random rows: `x y z weight nbar`.

Each rank reads the small fixture and keeps its assigned rows, using deliberately different data/random ownership. This is a validation IO format, not a distributed production catalogue reader. The estimator itself routes its owned catalogue data normally. Rank zero writes CSV columns `ell,bin,k,nmodes,raw,shotnoise,power,alpha,i2`. The driver estimates orders 0/2/4 with shot subtraction enabled, retaining both raw and noise columns.

See [RESULTS.md](RESULTS.md) for actual runs and remaining validation boundaries. Production recovery for a target survey still needs its own mask, selection, padding/mesh/k convergence, noise model and independent mocks; this experiment does not validate survey-window modeling or BAO reconstruction.
