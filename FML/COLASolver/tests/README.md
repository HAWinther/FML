# Tests

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

## Time-stepping (`check_timestepping.py`)

```bash
python3 tests/check_timestepping.py ./nbody /tmp/timestepping   # about 4 minutes, needs numpy
```

Runs the same simulation (64^3, z = 19 -> 0) with all the time-stepping methods and 5, 10, 20 and 40 steps
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

## Regression tests (`regression/`)

Small simulations covering all the main code paths, for checking that a change does not change the
results (or only changes what it should). See `regression/README.md`.
