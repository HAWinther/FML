# Regression tests for the COLASolver

A set of small simulations (32^3 particles, about 30 seconds in total) that together go through
all the main code paths of the solver:

| Case | What it covers |
|---|---|
| `cola_quinn_all` | COLA, Quinn time-stepping, output at the initial redshift, sigma8 normalization, P(k), multipoles, bispectrum, FoF, Gadget output |
| `nocola_tassev_log` | No COLA, Tassev time-stepping, logarithmic spacing, total number of steps split over outputs, transfer function input, IC from GR growth, fixed amplitude + reversed phases, MT19937, FML output |
| `scaledep_cola_neutrinos` | Scale-dependent COLA, massive neutrinos, transfer info file input |
| `fofr_cola1_tsc` | f(R) gravity with screening, 1LPT COLA, TSC assignment |
| `powerlaw_spacing` | Power-law spacing of the time steps |
| `lightcone_dgp` | Lightcone (ascii output), DGP gravity |
| `nbody_gauge` | N-body gauge: IC from N-body gauge transfer functions plus the linear source from radiation, neutrinos and GR |
| `fine_force_grid` | Force grid twice the particle grid (fiducial gradient kernel becomes discrete_2pt), PCS assignment |
| `scaledep_cola_fine_force_grid` | Scale-dependent COLA with the force grid twice the particle grid (lattice offset); identical P(k) to `fine_force_grid` in GR |
| `hybridbias_zeldovich` | Hybrid bias spectra with the Zel'dovich prediction |
| `fofr_exact_multigrid` | f(R) with the exact field equation (multigrid solver) |
| `dgp_exact_multigrid` | DGP with the exact field equation (multigrid solver: mixed derivatives, cube ordering of the sweeps, mean of the residual subtracted) |
| `fastpm_cola_fofr` | FastPM time-stepping with COLA, logarithmic spacing, f(R) gravity |
| `bullfrog_lightcone` | BullFrog time-stepping (no COLA) with the lightcone |
| `read_particles_exact` | Reading particles (from `cola_quinn_all`) with exact LPT reconstruction |
| `read_particles_grid` | Reading particles with LPT reconstruction from the density field |
| `read_phases` | Taking the phases of the IC from particles |
| `hdf5_output` | As `cola_quinn_all` but with GADGET HDF5 output (only run if compiled with `USE_HDF5=true`) |
| `read_particles_hdf5` | Reading the HDF5 particles from `hdf5_output` (only run if compiled with `USE_HDF5=true`) |
| `nongaussian` | Non-gaussian (local fNL) IC |

The cases are defined in `cases.py`.

## Usage

Use these when changing the code: run the cases with the code before and after the change and
compare the outputs. Results are reproducible (`OMP_NUM_THREADS=1` and a fixed seed), so a pure
refactoring should give identical outputs.

```bash
# Reference: build the code before your change
make && tests/regression/run.sh ./nbody /tmp/reg_ref

# ...make your change, rebuild, and run again
make && tests/regression/run.sh ./nbody /tmp/reg_new

# Compare (--exact: every number and every binary file must be identical)
python3 tests/regression/compare.py /tmp/reg_ref /tmp/reg_new --exact
```

- `run.sh <nbody> <outdir> [case ...]` runs all cases (or the ones named). Set `MPIRUN` to choose
  how to launch, e.g. `MPIRUN="mpirun -np 4"` or `MPIRUN=""` for a serial build (default `mpirun -np 2`).
- `compare.py <ref> <new>` compares all output files. Without `--exact` text files are compared
  with a relative tolerance (`--rtol`, default 1e-6) and binary files are skipped. Use this
  when comparing across machines or compilers, where results are not bit-identical.

The `read_*` cases use the particles written by `cola_quinn_all`, so run that case first (the default
order does this).
