# FML

[![CI](https://github.com/HAWinther/FML/actions/workflows/ci.yml/badge.svg)](https://github.com/HAWinther/FML/actions/workflows/ci.yml)

**FML is a C++17 library for large-scale-structure cosmology.** It has the pieces you need to set up, run and analyse
cosmological simulations and to compare them with theory. The library covers initial conditions, N-body and COLA
evolution, halo finding, clustering statistics, perturbation theory and an Einstein–Boltzmann solver. Most of it is
header-only and templated on the particle type and the number of dimensions, so you can use your own particle
struct directly. It runs in serial or in parallel with MPI (slab decomposition) and OpenMP.

The library also contains [SimulationCode](FML/SimulationCode/README.md), a complete PM/COLA N-body code for ΛCDM and
modified gravity. It is driven by a parameter file and computes power spectra, halos and other outputs on the fly.

## What's inside

**Simulations and initial conditions**
- Gaussian and non-Gaussian random fields (N-GenIC compatible or Philox), fixed amplitudes and glass initial
  conditions.
- 1LPT, 2LPT, 3LPT and arbitrary-order LPT displacements, with transfer functions from CAMB, CLASS or the built-in
  solver.
- PM and COLA N-body evolution with short-range and TreePM-style forces. Includes massive neutrinos and wave
  (fuzzy) dark matter.
- Modified gravity in SimulationCode: f(R), DGP, symmetron, dilaton, Jordan–Brans–Dicke, μ-parametrised and more.
  Screening is handled either exactly with multigrid solvers or with fast approximations.

**Measuring statistics**
- Power spectra and multipoles, bispectra and higher polyspectra, interlacing and higher-order mass assignment.
- Survey estimators with windows and BAO reconstruction.
- Pair counting, 2PCF and 3PCF multipoles, counts-in-cells, kNN distributions and marked or environment-split
  correlations.
- Cosmic-web classification, Minkowski functionals and topology, void finders and weak-lensing maps and profiles.

**Halos and galaxies**
- Friends-of-friends, spherical overdensity and a phase-space halo finder.
- Merger histories, HOD mocks and baryonification.

**Theory**
- Einstein–Boltzmann solver for ΛCDM and extensions: massive neutrinos, dark energy, Horndeski and more.
- Halofit, HMcode-2016 and HMcode-2020, a general halo model, ReACT, EuclidEmulator2 and spherical collapse.
- One-loop perturbation theory for matter and biased tracers: Lagrangian and IR-resummed Eulerian PT in redshift
  space, hybrid-bias EFT spectra, wiggle/no-wiggle splits and FFTLog.

**Inference and numerics**
- MCMC and HMC samplers, field-level inference, cosmology fitting and ensemble statistics with control variates.
- Multigrid solver, ODE solvers, splines, NUFFT, KD-tree and FFTW grids with MPI.

The [capability matrix](FML/CAPABILITIES.md) lists every module with its scope, dependencies, parallel behaviour and
tests.

## Example

The power spectrum of a particle catalogue. Positions are in [0, 1), and every MPI task keeps the particles in its
slab:

```cpp
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <iostream>

// Any particle type with get_pos() works: no base class needed
struct Particle {
    double x[3];
    double * get_pos() { return x; }
    constexpr int get_ndim() const { return 3; }
};

int main() {
    // A uniform random catalogue in the unit box
    std::vector<Particle> catalogue(100000);
    FML::RANDOM::RandomGenerator rng;
    for (auto & p : catalogue)
        for (auto & x : p.x)
            x = rng.generate_uniform();

    // Distribute the particles over the MPI tasks
    FML::PARTICLE::MPIParticles<Particle> part;
    part.create(catalogue.data(), catalogue.size(), 2 * catalogue.size() / FML::NTasks, FML::xmin_domain,
                FML::xmax_domain, true);

    // P(k) on a 128^3 grid with cloud-in-cell assignment and interlacing
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(64);
    FML::CORRELATIONFUNCTIONS::compute_power_spectrum<3>(
        128, part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), pofk, "CIC", true);
    pofk.scale(1000.0); // a box of 1000 Mpc/h: k in h/Mpc, P in (Mpc/h)^3

    if (FML::ThisTask == 0)
        for (int i = 0; i < pofk.n; i++)
            std::cout << pofk.kbin[i] << " " << pofk.pofk[i] << "\n";
}
```

Compile it from the folder that contains `FML/`, linking `FML/Global/Global.cpp`:

```sh
c++ -std=c++17 -O3 -I. -DUSE_FFTW pofk.cpp FML/Global/Global.cpp -lfftw3 -o pofk
mpicxx -std=c++17 -O3 -I. -DUSE_MPI -DUSE_FFTW pofk.cpp FML/Global/Global.cpp -lfftw3_mpi -lfftw3 -o pofk  # with MPI
```

Each module folder has examples (`example/`) and a README with its conventions.

## Building

You need a C++17 compiler and make. FFTW is needed for anything with Fourier transforms. MPI, OpenMP, GSL, Lua, HDF5,
CGAL and Python are optional and only needed by the modules that use them. Set paths and compilers in
`FML/Makefile.local`; the defaults are in [Makefile.config](FML/Makefile.config).

```sh
git clone --recursive https://github.com/HAWinther/FML/   # --recursive also fetches the example data

make -C FML examples -j4                                  # build the examples
make -C FML test MPIRUN=mpirun NP_LIST="1 2 4"           # run the regression tests with 1, 2 and 4 MPI tasks
make -C FML/tests USE_MPI=false USE_OMP=false USE_FFTW=false USE_GSL=false USE_LUA=false test_core   # minimal serial test
```

The example data are a git submodule. Only some examples need them; if you cloned without `--recursive`, fetch them
with `git submodule update --init --recursive`. The [build guide](FML/BUILDING.md) explains the options and how to
use FML from your own code. Clean the objects when you change the compiler, precision or feature flags.

## Python

A small [Python package](FML/PythonAnalysis/README.md) has catalogue statistics: kNN distances and CDFs, and
counts-in-spheres PDFs and moments, with NumPy results.

```sh
python -m pip install ./FML/PythonAnalysis
python FML/PythonAnalysis/examples/catalogue_statistics.py
```

There is also a [SWIG wrapper tutorial](FML/MakePythonWrapper) and a pybind11 wrapper for the
[halo model](FML/HaloModel/PythonWrapper/README); build them with `make -C FML python`.

## Documentation

- [Capability matrix](FML/CAPABILITIES.md): every module, with its scope, dependencies, parallel behaviour and tests.
- [Build and API guide](FML/BUILDING.md): compiling, linking, MPI and FFTW conventions.
- [References](FML/REFERENCES.md): the papers and external codes behind each module.
- [SimulationCode](FML/SimulationCode/README.md): the N-body code, its [parameters](FML/SimulationCode/PARAMETERS.md)
  and [tests](FML/SimulationCode/tests/README.md).
- [Changelog](FML/CHANGELOG.md) and [development backlog](FML/DEVELOPMENT_BACKLOG.md).
- [Website](https://fml.wintherscoming.no/) and [Doxygen reference](https://fml.wintherscoming.no/doxygen/).

The regression tests run in CI on every push. They compare against analytic results or reference codes where
possible. The module READMEs describe what has been validated and over which range, so check them before relying on
a method in a new regime.
