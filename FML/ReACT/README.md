# ReACT

FML's optional C++ interface to the official [ACTio-ReACTio v2](https://github.com/nebblu/ACTio-ReACTio)
engine, pinned to `a36dda02db6b537a4ddcc7bf044c446c10a58373`. It uses the upstream spherical-collapse,
halo-model and one-loop perturbation-theory calculations; it does not approximate screening by
passing a modified linear spectrum through an unmodified GR fitting formula.

The module supports flat GR, Hu--Sawicki f(R) with n=1, LCDM-background normal-branch DGP,
constant-w quintessence, and CPL smooth dark energy, including massive neutrinos. The solver's
`nonlinear=react` option supports `lcdm`, `fr`, `w0wa`, and `dgp` with `dgp_branch=-1` and
`dgp_lcdm_background=1`. Other DGP expansion histories, arbitrary EFT/modified-gravity models,
curvature, running, isocurvature and baryonic feedback are not supported by this integration.

## Build

Normal FML builds require no additional packages and keep the ReACT backend disabled.
Calling it in that configuration throws an explanatory exception. To enable it, first install
GSL, CMake, Git and a C++ compiler with OpenMP, then run from the repository root:

```sh
python3 FML/ReACT/build_backend.py --gsl-prefix "$HOME/local"
make -C FML/Cosmology/EinsteinBoltzmannSolver clean
make -C FML/Cosmology/EinsteinBoltzmannSolver -j4 USE_REACT=true REACT_LIB="$PWD/FML/ReACT/.backend/lib"
```

The helper downloads pinned upstream ReACT and SUNDIALS 4.1.0 sources into `.build`, compiles
them, and installs the backend privately in `.backend`. Both directories are git-ignored;
nothing is installed system-wide. It also installs the upstream ReACT and Copter license files.
Override `--prefix`, `--build-dir`, `--react-source`, `--sundials-source`, `--cc`, `--cxx` or
`--gsl-prefix` as needed. Existing source checkouts must have the pinned revisions. On macOS,
`--cc /usr/bin/clang --cxx g++` works with the MacPorts GCC C++ compiler and Apple's C compiler.
This also avoids older GCC C-header incompatibilities with newer Xcode SDKs.

For an already installed SUNDIALS 4.1.0, the smaller backend-only build is:

```sh
cmake -S FML/ReACT -B /tmp/fml-react-build \
  -DREACT_SOURCE=/path/to/ACTio-ReACTio -DSUNDIALS_ROOT=/path/to/sundials \
  -DGSL_ROOT_DIR=/path/to/gsl -DCMAKE_INSTALL_PREFIX=/path/to/backend
cmake --build /tmp/fml-react-build -j4
cmake --install /tmp/fml-react-build
```

Use the same compatible C++ runtime as FML. `REACT_LIB` is the directory containing
`libfml_react_backend.so` (Linux) or `.dylib` (macOS); the Makefile adds its runtime search path.
**Clean/rebuild when changing `USE_REACT` or library paths:** the shared Makefiles do not track
command-line flag changes in existing object files. The SimulationCode CMake build includes
the disabled-backend module so its usual internal solver still links; enabling the backend is
currently configured through FML's Makefiles, not SimulationCode's CMake options.

## Solver

From `FML/Cosmology/EinsteinBoltzmannSolver`, after the enabled build:

```sh
./cmb model=fr fR0=1e-5 fR_n=1 nonlinear=react pert_k_max=1 compute_cells=false
./cmb model=dgp Omega_rc=0.1 dgp_branch=-1 dgp_lcdm_background=1 nonlinear=react pert_k_max=1 compute_cells=false
./cmb model=w0wa w0=-0.9 wa=0.1 m_nu=0.3 nonlinear=react pert_k_max=1 compute_cells=false
```

The prediction is

    P_NL,target(k,z) = R(k,z) P_NL,pseudo(k,z).

The solver uses **HMcode-2020 for the matched pseudo cosmology**: flat GR/LCDM with clustering
matter density equal to the target's total present-day matter density, no massive neutrinos,
and a linear spectrum matched to the target total spectrum separately at each redshift. Its
primordial amplitude need not equal the target's. A separate LCDM Boltzmann calculation with
the target's original primordial parameters and neutrino content supplies the upstream
auxiliary cold spectrum needed for concentrations. These two reference cosmologies must not
be confused. With a massless GR target the solver enforces the exact R=1 identity; the raw
library interface retains the upstream result and its small numerical real/pseudo mismatch.

The same prediction supplies the fourth matter-power column and the existing Limber
nonlinear-lensing path. Linear total and cold power are unchanged. ReACT uses target spectra
at each output redshift, so scale-dependent growth is not applied a second time.

Settings:

| Parameter | Default | Meaning |
|---|---:|---|
| `react_n_z` | 12 | Number of samples uniform in ln(a), at least 4 |
| `react_n_k` | 256 | Logarithmic input/output k samples from 10^-5 to 100 h/Mpc, at least 32 |
| `react_mass_loop` | 30 | Spherical-collapse mass samples, 10--1000 |
| `react_z_max` | 2.5 | Maximum reaction redshift, greater than 0 and no greater than 2.5 |

**Redshift boundary:** upstream restricts reactions to z<=2.5. Above `react_z_max`, the solver
uses the matched HMcode-2020 pseudo baseline with R=1 and prints this assumption. This is an
explicit high-redshift approximation, not a validated ReACT prediction; it can introduce a
boundary discontinuity. The standalone table instead throws on redshift extrapolation.
At k below the table it returns R=1; above the table it holds the endpoint reaction fixed.
Neither the k=100 numerical endpoint nor the full table is a claim of physical calibration.

Upstream's target-power interface restricts 1.5e-9<=A_s<=2.5e-9 and 0.001<=k_pivot<=0.1 /Mpc, but with
power-spectrum input it uses A_s, k_pivot and n_s only to convert P to a transfer function and back, so they cancel:
FML always passes A_s = 2.1e-9 and k_pivot = 0.05/Mpc, and any A_s works. Use sufficient linear-spectrum
coverage (`pert_k_max` at least 1/Mpc for precision work) and check convergence in both the
Boltzmann settings and ReACT sampling. The small CLI examples used for implementation smoke
tests are not precision cosmological predictions.

## Library interface

```cpp
#include <FML/ReACT/ReACT.h>
using FML::COSMOLOGY::ReACT;
ReACT::Input in;
// Fill in.cosmology, increasing in.k and in.z, and [z][k] grids:
// in.total, in.cold, in.lcdm_cold.
ReACT::Options options;
options.model = ReACT::Model::HuSawicki;
options.parameters[0] = 1e-5;
auto result = ReACT::compute(in, options);
ReACT table(result);
double R = table.reaction(-std::log1p(z), k_hmpc);
```

All module input/output k use **h/Mpc** and all powers use **(Mpc/h)^3**. Arrays are **[z][k]**
with strictly increasing redshift; FML handles the upstream reversal/flattening. Cold power is
relative to the mean CDM+baryon density. `lcdm_cold` uses the same cosmological and primordial
parameters (including neutrinos) as the target but with LCDM expansion and GR perturbations.
Input grids must cover k<=10^-4 through k>=50 h/Mpc for the internal integrals. With neutrinos,
the upstream reconstruction from powers assumes coherent adiabatic transfer functions.

The result exposes `reaction`, target `linear`, upstream Takahashi `pseudo`, their product
`nonlinear`, and `sigma8_at_lowest_z`. Thus `result.nonlinear` uses the **upstream Halofit pseudo**,
whereas the solver deliberately uses the new **FML HMcode-2020 pseudo**. For a different
baseline, multiply `result.reaction` by a separately computed, linearly matched GR pseudo
spectrum. Multiplying a generic same-primordial-parameters LCDM spectrum gives the wrong
normalization in general.

Backend calls are serialized across FML ReACT instances because upstream uses process-global
tables and error storage. Upstream also resets GSL's error handler to the default (abort) one; the wrapper
restores the caller's handler after every call (FML switches it off and checks status codes). Completed FML tables can be queried concurrently. The interface checks
shapes, ordering, ranges, parameters and finite positive values, propagates upstream error messages,
and validates returned powers. Upstream internal fatal errors can still terminate the process;
the wrapper does not change the upstream numerical engine or its fatal-error policy.

## Tests and references

`FML/Cosmology/EinsteinBoltzmannSolver/tests/test_react_physics.cpp` tests the solver's `nonlinear=react` with
non-trivial reactions (~8 min on 2 cores with the backend; trivial without it):

- the solver's f(R) P_NL equals R from the library (on another k/z grid) times HMcode-2020 of the pseudo cosmology
  evaluated directly (3e-5);
- the f(R) enhancement P_f(R)/P_LCDM against the N-body fitting function of Winther et al. 2019
  (`Emulator/FofrFittingFunction.h`, ELEPHANT cosmology): 2.3% (|fR0| = 1e-6) and 2.9% (1e-5) for k <= 1 h/Mpc,
  z <= 1; up to 6.7% at k = 3 h/Mpc, z = 0 for 1e-5. At k = 0.05-0.1 h/Mpc ReACT x HMcode-2020 dips 1-2% below 1
  for 1e-6, where the simulations give >= 1: HMcode-2020 responds to the non-LCDM shape of the pseudo spectrum
  (the 1-halo/2-halo transition). Upstream's Halofit pseudo is worse (-6% at k = 0.1), which is why the solver
  uses HMcode-2020;
- R -> R_GR for |fR0| = 1e-8 and Omega_rc = 1e-6 (5e-4), and P_NL/P_L = LCDM's on large scales;
- w0wa (w0 = -0.9, wa = 0.1) and LCDM with sum m_nu = 0.3 eV: ReACT x pseudo agrees with HMcode-2020 run directly
  to 0.7% and 0.8% (k <= 3 h/Mpc, z <= 1);
- A_s = 3.2e-9 runs (a smaller f(R) enhancement than at 2.43e-9: more screening).

```sh
make -C FML/ReACT/tests run                         # no-backend contract/interpolation checks
make -C FML/ReACT/tests clean
OMP_NUM_THREADS=1 make -C FML/ReACT/tests run USE_REACT=true REACT_LIB="$PWD/FML/ReACT/.backend/lib"
```

Checked-in fixtures are generated independently through the official Python wrapper using CAMB
linear spectra, at z=0,0.5,1,2 and 128 k points from 10^-5 to 100 h/Mpc. Cases: GR, fR0=10^-5,
Omega_rc=0.1, and CPL w0=-0.9, wa=0.1 with sum(mnu)=0.3 eV. The f(R)/nDGP target linear spectra
are generated using upstream modified growth from the LCDM seed. The comparison checks array
ordering, units, linear-input preservation, reaction and upstream pseudo power. These are
interface/reference checks, not independent N-body validation or a calibration claim for the
HMcode-2020-composed solver prediction. The pseudo spectra are reproduced to rounding on any platform, the reactions
only to upstream's numerical reproducibility: rounding-level changes of the input change R by up to ~5e-4 (spherical
collapse with CVODE and bisection) and Linux and macOS builds differ by up to 1.3e-3 (f(R), 10 mass samples);
GR is reproduced to 1e-10. The test tolerances reflect this. The fixtures use 10 mass samples to keep runtime low;
production defaults use 30. To regenerate:

```sh
OMP_NUM_THREADS=1 python3 FML/ReACT/tests/generate_reference.py \
  --upstream /path/to/ACTio-ReACTio --backend /path/to/libfml_react_backend.so
```

Scientific references: [halo-model reactions](https://arxiv.org/abs/1812.05594),
[ReACT](https://arxiv.org/abs/2005.12184),
[massive neutrinos with modified gravity](https://arxiv.org/abs/2105.12114),
[ReACT v2](https://arxiv.org/abs/2210.01094).
