# SimulationCode

A fast and flexible particle-mesh (PM) and COLA N-body code for ΛCDM and modified gravity, built on
[FML](../../README.md). A single parameter file sets up the whole run: the initial conditions, the evolution and the
analysis done during the run, such as power spectra, halos and lightcones. Every time-consuming step is parallelised
with MPI and OpenMP.

The code uses slab-based parallelisation, so it is not meant for very high-resolution simulations. It is well suited
to fast approximate (COLA) simulations and large suites of runs, to testing new models and methods, and to analysing
snapshots from other simulations. There is more documentation on [the website](https://fml.wintherscoming.no/nbodycode.php).

## Features

- **[Models](MODELS.md)**
  - Cosmologies (the background): ΛCDM, w0waCDM, DGP, Jordan–Brans–Dicke, quintessence, early dark energy, or a
    tabulated H(z).
  - Gravity models (growth and forces): GR, Hu–Sawicki f(R), DGP, Jordan–Brans–Dicke, symmetron, dilaton,
    (m(a), β(a)) and a phenomenological μ(k, a).
  - Screening can be included in four ways: linear only, approximate screening, approximate screening with the linear
    field equation, or exactly with multigrid solvers.
- **[Initial conditions](INITIAL_CONDITIONS.md)**
  - 1LPT, 2LPT and 3LPT, from Gaussian fields or local, equilateral and orthogonal non-Gaussian fields, with optional
    fixed amplitudes and paired phases.
  - Random fields compatible with N-GenIC and L-PICOLA, or independent of the number of MPI tasks (Philox).
  - Input from a power spectrum, CAMB, CLASS, or FML's own Einstein–Boltzmann solver run at start-up; or particles
    from GADGET snapshots.
  - Ultralight-axion dark matter in the initial conditions (`ic_axion_fraction`: the solver's mixed axion + CDM
    model, the particles all the dark matter).
- **[Forces, time stepping and COLA](EVOLUTION.md)**
  - PM forces with NGP to PQS mass assignment and a choice of kernels; interlaced and NUFFT forces.
  - COLA, also with scale-dependent growth.
  - Quinn, Tassev, FastPM and BullFrog time stepping, including starting from the Big Bang.
  - Optional [short-range forces](SHORT_RANGE.md) (P3M or TreePM).
- **[Matter content](SPECIES.md)**
  - CDM+baryons as one species or as two.
  - Massive neutrinos as a linear field (with any masses) or as δf particles.
  - The N-body gauge, for radiation and GR on the largest scales.
- **[On-the-fly analysis and output](ANALYSIS.md)**
  - P(k) at every step with nonlinear predictions (EuclidEmulator2, and the f(R) fitting function).
  - Redshift-space multipoles and bispectra.
  - Hybrid-bias spectra with Zel'dovich and one-loop predictions.
  - FoF and spherical-overdensity halos with mass functions.
  - Lightcones with HEALPix maps.
  - Baryonification.
  - GADGET binary or HDF5 snapshots.

## Building

The code needs FFTW3, GSL and Lua (only for reading the parameter file), plus MPI and OpenMP for parallel runs. HDF5
(GADGET HDF5 output) and HEALPix (lightcone maps) are optional. With make, the settings are in
`../Makefile.config`; put the paths for your machine in `../Makefile.local`.

```bash
make                         # ./nbody; e.g. make USE_HDF5=true USE_HEALPIX=true, or USE_MPI=false
```

Or with CMake: the same executable and options, listed at the top of [CMakeLists.txt](CMakeLists.txt).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # e.g. -DFML_USE_HDF5=ON -DCMAKE_PREFIX_PATH=$HOME/local
cmake --build build -j
ctest --test-dir build                            # the tests in tests/
```

The CMake options are:

- `FML_USE_MPI`, `FML_USE_OMP`, `FML_USE_FFTW_THREADS`
- `FML_USE_MEMORYLOG` (on by default)
- `FML_USE_HDF5`, `FML_USE_HEALPIX`, `FML_USE_CHUNKPIX`
- `FML_USE_MINIMAL_PARTICLES` (off by default)
- `NDIM_SPACE` (3)
- `FML_FFTW_PRECISION` (double, float or long)
- `FML_NATIVE_ARCH` (`-march=native`, on by default)
- `FML_BUILD_TESTS` (on by default)

CMake finds GSL, Lua, MPI, OpenMP and HDF5 with its own modules and `CMAKE_PREFIX_PATH`. It finds FFTW with
`FFTW_ROOT`, and HEALPix with `HEALPIX_ROOT` and `CFITSIO_ROOT`. The CMake and Makefile builds give bit-identical
results on the regression cases (`tests/regression`).

By default the code is compiled for one particle type that can run every configuration. That type takes 120 bytes per
particle in 3D. `USE_MINIMAL_PARTICLES=true` (or `-DFML_USE_MINIMAL_PARTICLES=ON`) compiles all the particle types
and uses the smallest one a run needs, between 56 and 104 bytes (up to 120 with particle masses). The results are the same, but compiling takes 7
times longer. See [DEVELOPING.md](DEVELOPING.md#particle-types).

## Running

```bash
mpirun -np 4 ./nbody parameterfile.lua
./nbody --parameters PARAMETERS.md                            # write the reference of all the parameters
./nbody --write-solver-defaults internal_solver_default.ini   # the settings of the internal Einstein–Boltzmann solver
./nbody --write-transfers parameterfile.lua tables.txt        # its transfer tables, to be read with ic_type_of_input = tables
```

[parameterfile.lua](parameterfile.lua) is a commented example of a parameter file. [PARAMETERS.md](PARAMETERS.md)
lists every parameter with its type, default value, when it is used and the allowed values. Values are checked when
the file is read. A name that is not a parameter, such as a misspelling, gives a warning that suggests the intended
name.

When you combine MPI and OpenMP, give the threads of each task their own cores. When OpenMPI runs only 1-2 tasks it
binds each task to a single core, so all of that task's threads share it; the code warns about this at startup. Use
e.g. `OMP_NUM_THREADS=4 mpirun --map-by slot:PE=4 -np 2 ./nbody parameterfile.lua`, or `mpirun --bind-to none`.

At the end of the run the code prints how long the main parts of a step took: density assignment and FFT, P(k) every
step, force, kick and drift.

## Output

Everything goes to `output_folder`, labelled with `simulation_name`:

- the CDM+baryon (and total matter) P(k) at every step, together with linear theory and a nonlinear prediction
- at each of the `output_redshifts`, the particles (GADGET binary or HDF5) and the analysis that is switched on:
  - P(k) and its multipoles
  - bispectra
  - hybrid-bias spectra
  - halo catalogues and mass functions
  - baryonified spectra
- lightcone particles and maps

[ANALYSIS.md](ANALYSIS.md) describes the files and their conventions.

## Documentation

| Document | Contents |
|---|---|
| [PARAMETERS.md](PARAMETERS.md) | Every parameter: type, default, when it is used, allowed values |
| [MODELS.md](MODELS.md) | The cosmologies and gravity models, screening |
| [INITIAL_CONDITIONS.md](INITIAL_CONDITIONS.md) | Random fields, non-Gaussianity, input spectra and transfer functions (CAMB, CLASS, the internal solver), snapshots as initial conditions |
| [EVOLUTION.md](EVOLUTION.md) | PM forces, the force grid, accurate forces on a coarse grid, time stepping, starting from the Big Bang, COLA |
| [SPECIES.md](SPECIES.md) | Separate CDM and baryons, neutrino masses, neutrino particles, the N-body gauge |
| [ANALYSIS.md](ANALYSIS.md) | P(k), multipoles, halos, bispectra, hybrid-bias spectra, baryonification, snapshots and lightcones |
| [SHORT_RANGE.md](SHORT_RANGE.md), [COLA_SHORT_RANGE.md](COLA_SHORT_RANGE.md) | Short-range forces with PM and with COLA ([results](SHORT_RANGE_RESULTS.md), [clustered force](CLUSTERED_FORCE.md), [dynamics](DYNAMICS.md)) |
| [DEVELOPING.md](DEVELOPING.md) | The code layout, adding parameters, particle types and models |
| [tests/README.md](tests/README.md) | The tests: comparisons with MG-PICOLA, accuracy checks and regression tests |

## Accuracy and caveats

The code has a regression test suite and accuracy checks. These include comparisons with MG-PICOLA to better than
0.1%, linear-theory and convergence tests, and comparisons with CAMB, CLASS and the Einstein–Boltzmann solver. Each
document gives the measured accuracy of the options it describes. Not every combination of the many models and
options has been tested, so check the settings you use against a reference. Some refinements are still missing, such
as the higher-order LPT kernels of all models beyond ΛCDM, but their effect should be small. Options that do not work
together give an error; the list is at the end of [PARAMETERS.md](PARAMETERS.md).

## Contributors

Many people have helped improve the code, among them Bartolomeo Fiorini, Guilherme Brando, Kazuya Koyama, Daniela
Saadeh, Herman Slettmoen and Renate Mauland-Hus.
