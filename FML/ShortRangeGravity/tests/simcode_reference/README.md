# Frozen SimulationCode TreePM reference (for TreePMCode)

SimulationCode's common-timestep PM + short-range N-body (`force_short_range`, backlog item 47) is to be removed from
SimulationCode once TreePMCode reproduces it. These are reference runs of it, made with SimulationCode at FMLDevelop
`b7cc4758` (November 26 2026) and frozen here, so that TreePMCode can be checked against them after the
SimulationCode code is gone. The static force of the same library pieces is checked separately against an
independent Ewald sum by `../benchmark_clustered_force` (`../CLUSTERED_FORCE.md`).

## The runs

ΛCDM (Ωm = 0.3175, h = 0.671), GR, 32³ particles in 64 Mpc/h (mean separation 2 Mpc/h, particle k_Ny = 1.57 h/Mpc),
2LPT at z = 19 from `SimulationCode/input/example_power_spectrum_cb_z0.000.txt` (seed 1234, GSL, lattice offset by
half a force cell), plain N-body with Quinn KDK and logarithmic steps (3/4 of them from z = 19 to 1, 1/4 from 1 to 0).
Force: 64³ CIC grid with the Gaussian long-range filter, continuous Green's function and gradient; short-range force
with the cell list, split scale r_s = 1.25 force cells = 1.25 Mpc/h, cubic-spline softening support h = 0.14 Mpc/h
(h = 2.8 ε, ε = 1/40 of the separation), cutoff tolerance 1e-6. The full parameter files are in `parameters/`
(the input file path is relative to `FML/`).

| Case | What |
|---|---|
| `treepm_1024` | **The reference**: 1024 steps |
| `treepm_256`, `treepm_512` | The same with 256 and 512 steps (time-step convergence) |
| `pm_fiducial_256` | PM only, SimulationCode's fiducial PM (4-point gradient on this 2× grid), 256 steps: the size of the short-range contribution |
| `treepm_fofr_512` | Unscreened Hu–Sawicki f(R), n = 1, f_R0 = 1e-5, 512 steps: the model-owned Yukawa short-range force (compare with `treepm_512`) |

All cases start from the same particles (GR initial conditions): `ic_z19.000.gadget`. `treepm_1024_z0.000.gadget` is
the reference at z = 0. Both are single GADGET-1 files (32768 particles of type 1, float32 positions in Mpc/h and
velocities in the GADGET convention v_pec/√a km/s, int32 IDs; particle mass 70.49 × 10¹⁰ M☉/h).

`pofk/` has P(k) of the cb particles at z = 1 and 0 for every case (64³ PCS grid, interlacing, no shot-noise
subtraction; columns k [h/Mpc], P(k), linear P(k), and a nonlinear prediction).

## Accuracy of the reference (summary.json)

| | z = 1, k ≤ 1 h/Mpc | z = 1, up to k_Ny | z = 0, k ≤ 1 h/Mpc | z = 0, up to k_Ny |
|---|---|---|---|---|
| 512 vs 1024 steps | 0.13% | 0.25% | 0.64% | 1.2% |
| 256 vs 1024 steps | 0.71% | 1.5% | 1.8% | 4.9% |
| PM only (fiducial) vs reference | 6.3% | 6.3% | 4.9% | 22% |
| 2 MPI ranks vs 1 (256 steps) | 0 | 0 | 0 | 0 (to the printed digits) |

The step error falls by about 4 per halving, as expected for a second-order integrator, so the 1024-step reference is
converged to about 0.2% for k ≤ 1 h/Mpc at z = 0 (the 512–1024 difference is 3/4 of the 512-step error). The short-range force changes P(k) by 5–22% here, well above that.
Unscreened f(R) has +11% (k = 0.2 h/Mpc) to +36% (k_Ny) more power than GR at z = 0.
(The maxima in `summary.json` stop just below k_Ny; the table includes the k_Ny bin.)

## Checking a new code against it

Start from `ic_z19.000.gadget` (or regenerate the same initial conditions), evolve to z = 1 and 0 with the same force
split and softening, measure P(k) as above and run

```sh
python3 check_against_reference.py my_pofk_z1.000.txt my_pofk_z0.000.txt          # tolerance 2% for k <= 1 h/Mpc
```

It passes `treepm_512` (0.6%) and fails `pm_fiducial_256` (6.3% at z = 1, 4.9% at z = 0, for k ≤ 1 h/Mpc). Individual time steps and a
finite tree opening angle are not expected to reproduce the common-step cell-list run exactly; 2% leaves room for
that. Positions of individual particles in haloes are chaotic and should not be compared directly.

`make_simcode_reference.py` made all of this (it needs a SimulationCode with `force_short_range`, i.e. before its
removal): `python3 make_simcode_reference.py --nbody path/to/nbody` (about 25 minutes on 2 cores).
