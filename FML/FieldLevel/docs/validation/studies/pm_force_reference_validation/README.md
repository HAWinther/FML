> Historical validation record. Original commands and full raw outputs are
> preserved in the [external archive](../../../ARCHIVE.md).

# PM force reference with controlled particle sampling

This #52 follow-up tests the ambiguity in the [122-mode force comparison](../pm_nonlinear_validation/RESULTS.md):
its finest force mesh M64 had only N64 particles, one particle per force cell.
It measures whether increasing the particle lattice changes that reference and
whether force-grid sensitivity remains with a denser load on **both** grids.

The strong91 fixture is unchanged: 122 real IC directions (integer squared
wavenumber 1..9), 256 observed real directions (1..16), P=0.002, box length 1,
Omega_m=0.3, ZA start a=0.2 and final a=1. This is one initial realization and
one fixed noise seed, not an ensemble or a shell-crossing accuracy certificate.

## Fixed protocol

| Family | Configurations | Finite reference |
| --- | --- | --- |
| Particles at M32 | N64,96,128; 32 steps | N128/M32 |
| Particles at M64 | N64,96,128; 32 steps | N128/M64 |
| Force grid | M32,64; fixed N128, 32 steps | M64 |
| Time | 32,64 steps; fixed N128/M64 | 64 steps |

Seven forward trajectories, run one at a time. No adjoint or MAP runs are added.
Physical initial coefficients must match within 2e-12. The original density
thresholds remain <5% relative observed-band field change and <0.1 noise SD per
observed real degree, for each last finite correction. Particle corrections must
also improve from the coarsest trial. Mass error must be <1e-10 and absolute
mean canonical momentum <1e-10. These conservation checks do not establish
physical accuracy. Check output finiteness, support and complete diagnostic nodes.

The independent variable is N (both white/IC and particle lattice), with fixed
physical initial support. Basis grids are not stored. The forward workspace keeps
current fields; no reverse trajectory/checkpoint allocation is requested.
The driver exposes a separate forward-only metadata file and emits no gradient
file. Therefore this pass **cannot close the previous posterior-gradient criteria**.
Every reference is finite. Even N128/M64 (eight particles per force cell) needs
convergence evidence; it is not assumed to be exact.

`reference.data.csv` and `reference.initial.csv` freeze the original strong91
N64/M32/32-step mock from commit `a2093a9`, noise seed 92/power 1e-4, so a new
run does not need to repeat the original campaign. The data header checks target
identity; hashes are recorded in the plan. Data are used only for a value-only
posterior diagnostic. Density comparisons depend on the common physical ICs,
not on fitting these data. No forces, assignment windows or physics are changed.

## Reproduce

Build the shared driver using the usual compiler/library overrides:

```sh
make -C FML/FieldLevel/pm_nonlinear_validation pm_nonlinear USE_MPI=true USE_PENCIL_GRID=true
python3 FML/FieldLevel/pm_force_reference_validation/run.py --bin-dir /path/to/binaries --output /path/to/fresh/OUT --np 4 --mpirun "mpirun --oversubscribe" --timeout-seconds 300
```

Use a launcher appropriate to the installed MPI; clean/separate builds when
changing flags. OMP_NUM_THREADS=1 is set for the explicit campaign. `--np 0` runs
a serial binary directly. The default timeout is 300 seconds **per trajectory**;
timeouts and keyboard interruptions terminate the launched process group and leave logs. Argument
vectors, child PIDs and return codes are saved. `--phase cases|analyze|all` supports
separate execution/analysis. Analysis retains failed criteria and returns nonzero.
NumPy and Matplotlib are required. This is outside routine CI.

The shared native driver accepts an optional final `FORWARD_ONLY=1` argument:

```text
pm_nonlinear PREFIX N M STEPS SEED POWER FINAL_A DATA_PATH 0 0 1
```

Only forward-only mode permits N up to 128; M remains <=64. It requires external
compatible data and forbids FD/MAP requests. The original CLI/default mode and
N<=64 gradient/MAP bound remain unchanged. See [RESULTS.md](RESULTS.md) for actual
runtime/backend coverage, measured accuracy and remaining limits.
