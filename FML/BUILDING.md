# Building and integrating FML

This is the shared build/API guide. The [capability matrix](CAPABILITIES.md) links module scope, ownership and examples; individual module guides specify numerical conventions. FML's supported library configuration requires **C++17** (`-std=c++17`), including `if constexpr`, `std::optional` and `std::variant`. Some isolated headers may support older standards, as documented locally; that does not lower the library requirement. This guide does not promise support for every historical compiler version.

## Configuration

The shared [Makefile.config](Makefile.config) supplies compilers, feature flags, dependency paths and compile/link rules. A module Makefile chooses defaults and required objects. Put machine-specific settings in `FML/Makefile.local`, or pass make variables on the command line, which take precedence. Do not put flags such as `-DUSE_MPI` in isolation: matching compiler/link/runtime settings are also required.

```make
# Example Makefile.local; replace paths for your installation.
CXX_SERIAL = c++
CXX_MPI = mpicxx
LOCAL_PREFIX = /path/to/dependencies
# FFTW_INCLUDE, FFTW_LIB, GSL_INCLUDE, GSL_LIB, LUA_INCLUDE, ... can override individually.
```

| Make variable | Effect and requirement |
| --- | --- |
| `USE_MPI=true` | MPI compiler, headers and libraries; uses `FML::communicator`, normally MPI_COMM_WORLD |
| `USE_OMP=true` | OpenMP compile/link support; control worker count with OMP_NUM_THREADS |
| `USE_FFTW=true` | FFTW transforms; with MPI, also link an MPI-enabled FFTW build |
| `USE_FFTW_THREADS=true` | FFTW thread support and corresponding FFTW thread library; separate from ordinary OpenMP loops |
| `USE_GSL=true` | GSL-backed splines, ODEs and relevant numerical/cosmology workflows |
| `USE_LUA=true` | Lua parameter-file parsing; configure matching version/include/library paths |
| `USE_CGAL=true` | CGAL triangulation/tessellation examples and their dependencies |
| `USE_HDF5=true` | HDF5-enabled snapshot/output paths; configure appropriate includes/libraries |
| `USE_PYTHON=true` | Python/NumPy includes and embedding link options for applicable legacy examples; does not build/install the analysis package |
| `USE_MEMORYLOG=true` | Enable the shared build's memory logging defines; coverage is module dependent |
| `USE_DEBUG=true`, `USE_SANITIZER=true` | Module debug defines and address-sanitizer build options; not substitutes for numerical validation |

Not every target supports every combination. In particular, disabling all dependencies is not a recipe for building every cosmology/FFT regression. Choose an appropriate target from its module Makefile. `USE_FFTW_THREADS` needs the matching FFTW threading build; `USE_OMP` alone does not mean FFTs are multithreaded. [Makefile.config](Makefile.config) remains authoritative for exact variable names and link defaults.

## Rebuilds, examples and tests

Objects do not encode feature flags in their filenames. Clean before changing MPI/serial mode, precision, compiler or dependency options; use separate build directories for simultaneous configurations. Header dependencies are recorded in `.d` files, but a changed command-line define does not itself invalidate old objects. Several examples use `Main.cpp`; avoid sharing their object directories. Keep user data/output outside paths removed by a target's `clean` rule.

```sh
# From the repository root; start with clean objects when switching options.
make -C FML/tests clean
make -C FML/tests CXX_SERIAL=c++ USE_MPI=false USE_OMP=false USE_FFTW=false USE_GSL=false USE_LUA=false test_core
FML/tests/test_core

# Standard dependency-enabled test suite, built and run with requested task counts:
make -C FML test MPIRUN=mpirun NP_LIST="1 2 4"

# Focused modules:
make -C FML/CountsInCells/tests run USE_MPI=false USE_OMP=false
make -C FML/tests test_fourier_resampling
mpirun -np 4 FML/tests/test_fourier_resampling

# Examples and solver:
make -C FML examples -j4
make -C FML examples-extra -j4
make -C FML/SimulationCode -j4 USE_HDF5=true
```

`examples-extra` selects dependency-heavy examples such as CGAL/Python-related workflows. The SWIG wrappers are built by `make -C FML python`; import with the interpreter matching their build headers and NumPy installation. The new [analysis package](PythonAnalysis/README.md) instead uses `python -m pip install ./FML/PythonAnalysis` from the repository root, or `make -C FML python-analysis PYTHON=/path/to/venv/bin/python`. `make -C FML test-python-analysis PYTHON=...` tests an already installed package.

The [CI workflow](../.github/workflows/ci.yml) records the Linux regression configuration. Linked tests describe coverage; listing a target here does not assert every example was rerun during the current documentation change. External test data and optional libraries are required only by the targets that use them. Production convergence/accuracy checks remain separate from build/unit tests.

### Example data (`TestData` submodule)

Some examples read input files from the `TestData` git submodule ([HAWinther/TestData](https://github.com/HAWinther/TestData)),
by the relative path `../../../TestData/...` from their example directory. The library, the module tests, `FML/tests` and
CI do not use it, so a plain clone leaves it empty. Fetch it before running these examples:

```bash
git submodule update --init TestData   # from the repository root
```

| Example | Files used |
| --- | --- |
| ComputePowerSpectra, FriendsOfFriends, TidalTensor, Triangulation, GadgetUtils (`Test.cpp`) | `particles_B1024.txt` |
| GadgetUtils (`Main.cpp`) | `ExampleGadgetSnapshot/` |
| RamsesUtils | `ExampleRamsesSnapshot/` |
| PairCounting, Survey | `ExampleSurveyData/` (galaxies and randoms) |
| PairCounting, KDTree, MultigridSolver (`Reconstruction.cpp`) | `galaxies_redshiftspace.txt`, `voids_realspace.txt` |

New examples should generate their own input (e.g. a mock or a Gaussian random field) instead of adding to this list.

## Include paths and linked sources

Include the directory **containing** `FML`, not `FML/` itself:

```cpp
#include <FML/MPIParticles/MPIParticles.h>
struct Tracer {
    double pos[3]{};
    double *get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
int main() {
    FML::PARTICLE::MPIParticles<Tracer> particles;
    particles.create_particle_grid(8, 1.5, FML::xmin_domain, FML::xmax_domain);
}
```

Most algorithms are headers, but link `FML/Global/Global.cpp` for shared state and helpers. Additional modules need their implementation objects, for example Spline.cpp, ODESolver.cpp, ParameterMap.cpp, FileUtils.cpp or FFTLog.cpp; copy the linked-object list from a relevant tested Makefile. A serial minimal particle program can be built from the repository root with:

```sh
c++ -std=c++17 -I. particles.cpp FML/Global/Global.cpp -o particles
```

Particle interfaces use [compile-time reflection](ParticleTypes/ReflectOnParticleMethods.h), rather than requiring one base class. Required accessors and optional mass/velocity/ID/serialization methods depend on the algorithm. Coordinates used by MPI particles and many mesh algorithms are normalized to [0,1), whereas physical-box analysis APIs accept explicit box lengths. Read the selected module's unit and ownership contract; a common C++ type does not imply interchangeable conventions.

## MPI, lifetime and error contracts

Normal executable builds automatically set up FML/MPI/FFTW according to enabled flags. For an embedding application that owns the runtime, compile all relevant objects consistently with `NO_AUTO_FML_SETUP`, initialize MPI and any FFTW/thread facilities yourself, select `FML::communicator`, then call `FML::init_fml()`. Do not mix automatically managed and externally managed runtime objects or finalize an MPI runtime another component still uses. The Python analysis package enforces its separate serial/no-auto-setup configuration.

Collective APIs must be called on every participating rank in the same order, including ranks with zero work. Pass complete replicated catalogues only where required; pass slab-owned particles only to distributed APIs that specify that contract. [MPI count checks](Global/MPICounts.md), [particle migration](MPIParticles/Migration.md) and [Fourier routing plans](FFTWGrid/FourierResampling.md) document message/count/storage limits. A per-message cap is not a total process-memory bound.

Error behavior is API specific: newer numerical summaries often throw standard exceptions, legacy assertions may call `assert_mpi`/abort, and solver termination may return an explicit status. Do not assume all C++ calls are recoverable exceptions, nor catch a local exception and let other ranks enter a collective. [Multigrid termination](MultigridSolver/TerminationStatus.md) and the [Python exception contract](PythonAnalysis/README.md) give concrete examples. Plans/workspaces and returned views have documented ownership; do not retain references across `free`, reallocation or incompatible plan reuse.

## FFT precision and grid layout

FFTWGrid defaults to double precision. `SINGLE_PRECISION_FFTW` or `LONG_DOUBLE_PRECISION_FFTW` selects the matching scalar/complex type; provide the corresponding FFTW libraries as well. For example, pass `FFTW_LINK=-lfftw3f` and `FFTW_MPI_LINK=-lfftw3f_mpi` for float, plus the matching thread library when enabled. These are C++ defines supplied consistently to every dependent object; the shared Makefile does not infer precision libraries from arbitrary defines.

FFTWGrid uses a non-transposed r2c slab layout with a final axis of `nmesh/2+1` complex coefficients and real-array padding. The MPI constructor currently requires dimension >1 and a mesh divisible by task count. Forward transforms are normalized; raw Fourier resampling needs no extra volume factor. Physical spectra, assignment-window corrections, shot noise and survey windows have separate conventions in their module guides. Clean/rebuild when precision or layout-related options change.
