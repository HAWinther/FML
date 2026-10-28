# Tests

## Snapshot baryonification

`make run` in this folder builds the fast parameter/unit-suffix and snapshot-copy tests (no GSL/FFTW/Lua required).
The end-to-end check needs the normal COLASolver executable and numpy:

```bash
python3 tests/check_baryonification.py ./nbody /tmp/baryonification
```

It exercises nonzero feedback at two outputs, checks that original outputs and later evolution are byte-identical
to a run without correction, and verifies the empty-catalogue identity control, mass/radius conversions, recorded
settings, preserved IDs/velocities, and invalid configurations. HDF5 corrected output is tested when compiled in
(with dataset checks when h5py is available). Run with MPIRUN="mpirun -np 1", 2 or 4 to check the MPI decompositions.
The integration uses reduced profile-table resolution for speed; numerical profile tests live in `FML/Baryonification/tests`.

All the scripts take the `nbody` executable and a folder to put the outputs in. They use
`mpirun -np 2` by default; set `MPIRUN` to change this (e.g. `MPIRUN=""` for a serial build).

## Accuracy: comparison to MG-PICOLA (`LCDM_*`)

The `LCDM_*` folders compare to L-PICOLA / MG-PICOLA run with the same random number generator and
the same time-stepping. When these things are the same we should get *very* close results. For these
tests MG-PICOLA was also modified to evaluate the power-spectrum exactly as we do it for the P(k) we
compute every step (CIC on the force grid, window deconvolved, no shot-noise subtraction).

```bash
tests/run_accuracy_tests.sh ./nbody /tmp/accuracy      # about 15 s per test
```

Each test is a 128^3 simulation and passes if P(k) at z = 0 agrees with MG-PICOLA to 0.2% for all
k up to the Nyquist frequency (0.8 h/Mpc). Currently the agreement is better than 0.08%. Use
`compare_pofk.py <reference> <pofk-file> --verbose` to see the comparison bin by bin.

NB: the k we output is slightly different (we output the mean of the k-modes in each bin, while the
picola results are the raw k-bins) so the rows are matched by index.

The `pofk_TestSim_cb_z0.000.txt` files are the results from the time the tests were made.

## Non-gaussian initial conditions (`check_nongaussian.py`)

```bash
python3 tests/check_nongaussian.py ./nbody /tmp/nongaussian   # about 15 s, needs numpy
```

Runs gaussian IC and local fNL = 0, +300, -300 with the same seed and checks that fNL = 0 gives the
same P(k) as the gaussian IC, that the bispectrum responds linearly to fNL and that the large-scale
reduced bispectrum increases with fNL.

## Phenomenological modified gravity (`check_mu_gravity.py`)

```bash
python3 tests/check_mu_gravity.py ./nbody /tmp/mu   # about 2 minutes, needs numpy and scipy
```

Linear-amplitude runs of GR and `gravity_model = mu` (mu0 = 0.5 and a Bertschinger-Zukin mu(a,k)), without COLA and
with COLA (scale-dependent COLA for the k-dependent BZ mu), and checks P_mu / P_GR at z = 0 against (D_mu / D_GR)^2
from an independent solution of the growth equation (to 0.1% for k <= 0.1 h/Mpc). The scale-dependent COLA runs also
check that the COLA frame is right with the particle lattice offset from the force grid (2x force grid).

## Time-stepping (`check_timestepping.py`)

```bash
python3 tests/check_timestepping.py ./nbody /tmp/timestepping   # about 15 minutes, needs numpy
```

Runs the same simulation (64^3, z = 19 -> 0) with all the time-stepping methods and 5, 10, 20 and 40 steps
(FastPM and BullFrog also with steps uniform in the growth factor, `-D`)
and prints how far P(k) at z = 0 is from a reference run with 200 steps. Checks that FastPM and BullFrog
get large scales right with few steps and that all methods converge.

## N-body gauge (`check_nbody_gauge.py`)

```bash
python3 tests/check_nbody_gauge.py ./nbody /tmp/nbgauge   # about 3 minutes, needs numpy, scipy and classy
```

Makes the N-body gauge input with CLASS and runs a large box (8000 Mpc/h, 128^3, tiny amplitude so it is
linear) from z = 49 with the N-body gauge source, with the source set to zero and with the standard setup.
Checks that the source changes P(k) on the largest scales as linear theory predicts (0.96% at k = 1e-3 h/Mpc,
reproduced to 0.01%) and that the run then follows the CLASS growth as well as the standard run follows its
own linear theory (the PM force resolution slows the growth by about 0.5% at this k in both).

## Force grid finer than the particle grid (`check_force_grid.py`)

```bash
python3 tests/check_force_grid.py ./nbody /tmp/forcegrid   # about 20 s, needs numpy
```

A linear (tiny amplitude) run with a force grid twice the particle grid (GRID_FACTOR=3 or 4 for other
ratios). With the continuous gradient kernel and the particles on the grid nodes the regular particle
lattice of the initial conditions gives spurious forces and the growth of the individual large-scale modes
scatters by 40-70%; with the fiducial settings (half-cell lattice offset, discrete_4pt kernel up to 3x and discrete_2pt beyond) the growth on
the largest scales is right to < 1.5% with < 1% scatter between modes.

## Lightcone (`check_lightcone.py`)

```bash
python3 tests/check_lightcone.py ./nbody /tmp/lightcone   # about 15 s, needs numpy and scipy (healpy for the maps)
```

32^3 particles in 256 Mpc/h. `cola`: the lightcone positions with COLA and scale-dependent COLA (with a k-dependent
mu(a,k): it is only used with scale-dependent growth) in 10 steps are within a median 0.085 Mpc/h of those of a 200
step run (0.25 without COLA; 0.49 when the LPT displacement of the step was counted twice). `batches`: an octant
with replicas where a step has more crossings than particles per task, with 2 and 4 tasks, with and without
`plc_output_in_batches`: all give the same particles (the number expected from the mean density) and every batch has
one file per task. `zfinish`: with `plc_z_finish` inside a step the number of particles just above r(z_finish) is
the same as with 200 steps. `coverage`: without replicas an incomplete octant is an error. `healpix`: the octant
maps (zero outside, zero mean inside) are the full sky maps restricted to the octant and their C_ell the
pseudo-C_ell divided by fsky (without USE_HEALPIX: the maps are an error when reading the parameters). Run the tests
separately by naming them after the outdir.

## Initial conditions (`check_initial_conditions.py`)

```bash
python3 tests/check_initial_conditions.py ./nbody /tmp/ics   # about 15 s, needs numpy
```

Checks that the velocities at the initial redshift are the same with and without COLA (0 steps) for CDM + baryon
particles with linear massive neutrinos, one species and the N-body gauge; that massive neutrinos without the
transfer functions give an error; that reading particles with exact LPT reconstruction from an offset lattice
(fiducial offset with a 2x force grid, half_force_cell) reproduces the P(k) of the original run, that the
reconstruction from the density field (PCS) with COLA is within 2.5%, and that a file with another BoxSize or h, or
exact reconstruction with scale-dependent growth and COLA, is an error; and the warning for an input P(k) that does
not reach the largest k of the IC grid.

## COLA and the LPT growth factors (`check_cola.py`, `test_cola_frame.cpp`)

```bash
python3 tests/check_cola.py ./nbody /tmp/cola   # about 10 s, needs numpy and scipy
cd tests && make test_cola_frame && ./test_cola_frame
```

`check_cola.py`: `simulation_use_scaledependent_cola = true` gives standard COLA in GR (with a message, the same
result as false) and scale-dependent COLA with massive neutrinos, and exact LPT reconstruction of read particles works
without COLA; with `simulation_enforce_LPT_trajectories = true` the velocities at z = 20, 1 and 0 are the 2LPT
velocities of the displacements (to 4e-5; they stayed close to the initial velocities); and D2/D1^2 of DGP (H0 rc =
0.5, 1, 2 with screening and 1 without) agrees with an independent solution of the growth equations to 3e-6 (it was
0.5-2% off). `test_cola_frame` (no MPI): a linear mode integrated with the Quinn and Tassev steps (20 steps from a =
0.05, linear, logarithmic and a^0.5 spacing) has the growth and velocity within the expected time-stepping error
(with logarithmic spacing -0.38% / -0.31% for Quinn, -0.76% / -0.84% when the kick was normalized at the midpoint of
the velocity times instead of the time of the force), and the COLA frame kicks of the LPT force add up to minus the
change of the LPT velocity of the growth factors to 1e-7 (GR, DGP with and without screening; 1% off for 2LPT in DGP
before the source factors were included).

## Regression tests (`regression/`)

Small simulations covering all the main code paths, for checking that a change does not change the
results (or only changes what it should). See `regression/README.md`.
