# FML

FML is a C++17 library for large-scale-structure cosmology: particles and grids, Fourier statistics, initial conditions, perturbation theory, N-body/COLA tools, halo/void analysis and numerical solvers. Algorithms use templates to adapt to particle types and dimension. MPI slab decomposition and OpenMP are available where the module supports them; catalogue ownership and collective-call requirements remain part of each API.

Start with the [capability matrix](FML/CAPABILITIES.md) for implemented scope, dependencies, parallel contracts and linked regression examples. The [build and API guide](FML/BUILDING.md) is the reference for C++17 compilation, optional features and integration. The [development backlog](FML/DEVELOPMENT_BACKLOG.md) distinguishes delivered work, production-validation follow-ups and proposed capabilities.

## Build

A C++17 compiler and make are required. Dependency-free modules can run in serial; Fourier transforms need FFTW, and other modules optionally use MPI, GSL, Lua, CGAL, HDF5 or Python. Configure paths and compilers in `FML/Makefile.local` or on the command line; defaults are in [Makefile.config](FML/Makefile.config). Clean objects when changing compiler, precision or feature flags.

```sh
# Most examples, with the dependencies selected by their Makefiles:
make -C FML examples -j4

# Standard regression suite, normally using MPI/FFTW/GSL/Lua:
make -C FML test MPIRUN=mpirun NP_LIST="1 2 4"

# Minimal serial core test (start with clean test objects):
make -C FML/tests USE_MPI=false USE_OMP=false USE_FFTW=false USE_GSL=false USE_LUA=false test_core
FML/tests/test_core
```

The optional example data are a submodule. Clone with `git clone --recursive https://github.com/HAWinther/FML/` or initialize it later with `git submodule update --init --recursive`. Not every example requires those data. See the build guide for clean configuration changes, linking `Global.cpp`, FFTW precision/thread selection and extra dependencies.

## Python analysis

The [small Python package](FML/PythonAnalysis/README.md) offers an owned 3D catalogue, kNN distances/CDFs and spherical count/PDF/moment workflows with NumPy results. Its extension is serial and does not own an MPI/FFTW lifecycle.

```sh
# Inside a Python virtual environment:
python -m pip install ./FML/PythonAnalysis
python FML/PythonAnalysis/examples/catalogue_statistics.py
python -m unittest discover -s FML/PythonAnalysis/tests -v
```

The [SWIG tutorial](FML/MakePythonWrapper) and specialized [halo-model wrapper](FML/HaloModel/PythonWrapper/README) remain separate; `make -C FML python` builds those legacy targets. Python is optional for the C++ library.

## Documentation and validation

Local module guides and tests linked from the capability matrix describe conventions and numerical limits. [COLASolver](FML/COLASolver/README.md) has its own [parameter reference](FML/COLASolver/PARAMETERS.md) and [test guide](FML/COLASolver/tests/README.md). The [changelog](FML/CHANGELOG.md) records other changes. The [website](https://fml.wintherscoming.no/) and [Doxygen reference](https://fml.wintherscoming.no/doxygen/) provide additional material; local documentation tracks this checkout and may be newer.

Regression coverage establishes specific numerical contracts, not universal production accuracy. Survey estimators, initial-condition providers, nonlinear force models and halo/void algorithms have module-specific validation and supported regimes. Consult those contracts before combining workflows or interpreting physical units and shot-noise corrections.
