# TreePMCode

A simple TreePM N-body code in the spirit of GADGET, for collisionless particles: CDM (or CDM+baryons as one set of
particles), or CDM and baryons as two sets. No hydrodynamics. Built from FML's building blocks, separate from AMRCode
(the RAMSES-like code) and SimulationCode.

```
make && mpirun -np 4 ./treepm parameterfile.txt
cd tests && make run
```

Needs MPI, FFTW and GSL (as the rest of FML, `../Makefile.config` / `Makefile.local`). The number of tasks must
divide the PM grid `Nmesh`.

## What it does

1. **Initial conditions** as AMRCode: 1LPT/2LPT from a linear P(k) at z = 0 (CAMB-format file, or Eisenstein & Hu
   normalized to `sigma8`) back-scaled with the model's growth, or from transfer functions (`transfer_file`,
   `transfer_source = tables | camb | class`), N-GenIC seeding (`random_generator = GSL`). CDM and baryons as two sets
   of particles (`particle_baryons = true`) from their transfer functions (`baryon_lattice = same | offset`). Or the
   particles of a GADGET file (`ic_gadget_file`, one set of type-1 particles; zini from its header).
2. **Domain.** The particles are on a Peano-Hilbert domain (`Global/Domain.h`), rebuilt with equal particles per task
   every step (`hilbert_bits`).
3. **Gravity** split with a Gaussian of scale `r_s = asmth` PM cells (GADGET-2's split and cubic-spline softening,
   `ShortRangeGravity/`):
   - long range: the density on the `Nmesh^3` slab grid from the particles on their Hilbert domain
     (`particles_to_grid_any_domain`), the filtered Poisson equation with continuous kernels (window deconvolved,
     `deconvolve`), and the force interpolated back to the particles (`interpolate_grids_to_particles_any_domain`,
     the inverse: each task gathers the part of the force grids its particles need);
   - short range: the ghosts within the cutoff from the other tasks (`exchange_domain_ghosts` on the Peano-Hilbert
     domain) and the monopole octree (`backend = tree`, `leaf_size`; `opening_criterion = geometric` with
     `opening_angle`, or `relative` as GADGET-2 with `relative_tolerance` = ErrTolForceAcc) or the cell list
     (`backend = cell`) over local + ghost particles. Softening `softening` (Plummer-equivalent, in units of the mean
     particle spacing, comoving; the spline support is 2.8 times it), cutoff where the pair force is
     `cutoff_tolerance` of Newton's.
4. **Time stepping** (`timestep_method`):
   - `fixed` (default): kick-drift-kick with a common step, `nsteps` uniform in log(a) (plus the outputs), or
     `nsteps_between_outputs` (one number per output: uniform in log(a) from zini to the first output, then between
     the outputs).
   - `gadget`: as GADGET-2. The PM (long-range) force on a global step: the rms displacement in a step is at most
     `max_rms_displacement` (MaxRMSDisplacementFac, 0.25) times min(mean particle spacing, r_s), and the step at most
     `max_dlna` (MaxSizeTimestep, in log a, 0.025); the outputs are hit exactly. The short-range force on individual
     steps dt_i = sqrt(2 eta eps / |a_i|) (`eta` = ErrTolIntAccuracy, 0.025; eps the softening, |a_i| the last total
     acceleration d^2x/dtau^2), in power-of-two bins of the PM step (at most `max_bins`): half a long-range kick, then
     kick-drift-kick sub-steps where every particle is drifted and only the active ones get the short-range force
     (ghosts and tree of all particles at the current positions, the force of the active targets), and at the end of
     the PM step (all synchronized) the new domain, the PM force and the other half long-range kick. Bins change only
     when synchronized. The friction (coupled quintessence, dark scattering) acts in the short-range kicks; the opening
     long-range kick is weighted so that the friction of the step applies once.
5. **Models** (as AMRCode, `Cosmology.h`): LCDM, `CoupledQuintessence` / `FadingDarkSector` (background, Poisson
   factor, CDM friction) and `CovariantDarkScattering` (wCDM, CDM drag). With CDM and baryon particles the source is
   weighted by f_c(a), f_b(a) (the particle mass), and with coupled quintessence the CDM also feels 2 beta^2 f_c times
   the force of the CDM alone (a second force evaluation with the CDM as the only source).

Units: positions in [0,1), tau = log(a), v = a^2 dx/dt / (H0 B): `dx/dtau = v / (a^2 E)`,
`dv/dtau = -Gamma v - (3/2) OmegaM P(a) / (a E) D phi`, `D^2 phi = delta`.

Outputs: `pofk_<label>_z<z>.txt` (k, P(k) without shot-noise subtraction, linear P(k); with two species also
`pofk_cdm_` and `pofk_baryon_`; `pofk_ngrid`, `pofk_assignment`, interlaced), optionally GADGET files and a PNG image
at `output_redshifts`.

## Files

| File | |
|---|---|
| `TreePM.h` | `TreePMParticle`, `TreePMParameters`, `TreePMSimulation` (domain, PM + short-range force, KDK step) |
| `Cosmology.h` | The background, models, growth factors, kick/drift/friction integrals (as AMRCode's, kept separate) |
| `Setup.h` | Parameters of a run, initial conditions, step times, outputs |
| `ParameterFile.h`, `LinearPower.h` | The parameter file and the input P(k) |
| `Main.cpp` | The program |
| `tests/TestTreePM.cpp` | Blob force, Zel'dovich plane wave, two fluids |

Library parts written for this code: `Interpolation/DistributedDeposit.h` (`interpolate_from_partitioned_mesh`,
`interpolate_grids_to_particles_any_domain`) and `ShortRangeGravity/SlabGhosts.h` (`exchange_peano_hilbert_ghosts`,
and `exchange_domain_ghosts` for a PeanoHilbert domain), with tests in `tests/test_distributed_deposit.cpp` and
`ShortRangeGravity/tests/test_short_range.cpp`.

## Tests (`tests/`, `make run`: 1, 2 and 4 tasks)

- **Blob** (`testtreepm blob`): a Gaussian blob (sigma = 0.02, 2^17 particles, PM 64^3) against the exact periodic
  force M(<r)/(4 pi r^2) - r/3: rms radial error 3.7 / 2.1 / 1.3 / 0.4 % at r/sigma in [0.5,1) / [1,2) / [2,4) /
  [4,8) (the particle noise). 2^13 particles: cell list = exact tree to 2e-15, tree with opening 0.5 within 6e-5 of it,
  PM 32^3 instead of 64^3 within 0.4% (the split moves with the grid).
- **Zel'dovich plane wave** (`zeldovich`, 32^3, z = 49 to D b = 0.88, softening 1/4 spacing): rms error 2.2% in x and
  2.5% in v, amplitude 1.001; tree = cell list to 1e-16; the same positions on 1, 2 and 4 tasks (checksum to 14
  digits). With GADGET-like softening (1/30) the velocities near shell crossing are 20% off: the sheets of particles
  come closer than the spacing within a sheet, and the discrete lattice forces between them are not smoothed.
- **Two fluids** (`twofluid`, 16^3 of each species, softening 1 spacing): LCDM: both species move together; coupled
  quintessence (beta = 0.3): D_b / D_c = 0.78396 against 0.78477 from the linear two-fluid growth; covariant dark
  scattering (alpha = 1, w = -0.98): 1.0501 against 1.0514.
- **GADGET-2 steps** (`gadget`): the plane wave with 40 PM steps and individual short-range steps follows Zel'dovich
  (2.2% in x, 2.5% in v) and the run with 160 common steps to 1.3e-3 of the amplitude; the two-fluid split with
  individual steps is the same as with common steps (0.78396). (Before the opening long-range kick was weighted for the
  friction it was 0.7813: the friction of half a step counted twice.)

## First runs

64^3 in 128 Mpc/h from z = 49 (EBS transfer tables, the same IC as AMRCode's and SimulationCode's 64^3 runs), PM 128^3,
softening 1/30 spacing, 2 tasks. P(k) against EuclidEmulator2 (B = P_nl / P_lin for this cosmology), as
[P - V/N] / [P_lin R] / B with R = P / P_lin of each k-bin at z = 49 (removes the realization of the small box):

| z = 0, k [h/Mpc] | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| TreePM, 40 common steps (2m26s) | 0.92 | 0.90 | 0.86 | 0.68 | 0.54 | 0.33 | 0.13 |
| TreePM, 160 common steps (11 min) | 0.93 | 0.95 | 1.00 | 0.87 | 0.84 | 0.76 | 0.42 |
| TreePM, `gadget` steps (157 PM steps, up to 16 sub-steps; 21 min) | 0.93 | 0.96 | 1.02 | 0.90 | 0.89 | 0.86 | 0.51 |
| AMRCode, levels 6-10, 40 fixed steps | 0.93 | 0.91 | 0.90 | 0.74 | 0.73 | 0.65 | 0.35 |
| AMRCode, `timestep_method = ramses` (63 steps) | 0.94 | 0.92 | 0.92 | 0.76 | 0.73 | 0.63 | 0.35 |
| SimulationCode PM 128^3, 100 Quinn steps | 0.94 | 0.96 | 0.98 | 0.84 | 0.78 | 0.62 | 0.24 |

(z = 1: TreePM `gadget` 0.96 / 0.91 / 0.90 / 0.83 at k = 0.3 / 0.5 / 1 / 1.5.) Below k ~ 0.3 all codes are 4-7% under
EE2: the small box (no larger modes, few modes). Above k ~ 1 the mass resolution of 64^3 particles in 128 Mpc/h
(7e11 Msun/h) limits every code. Among the codes TreePM with GADGET steps is closest to EE2; the common step had to be
far shorter for the TreePM force resolution, while AMRCode does not change with RAMSES' steps (its resolution, not its
steps, limits it).

## Against the SimulationCode TreePM reference

`ShortRangeGravity/tests/simcode_reference`: 32^3 particles in 64 Mpc/h from `ic_z19.000.gadget` (z = 19), PM 64^3
CIC, r_s = 1.25 Mpc/h, softening support 0.14 Mpc/h (`softening = 0.025`), cutoff tolerance 1e-6, 768 + 256 log steps
(`nsteps_between_outputs = 768, 256`), P(k) on 64^3 PCS with interlacing (`pofk_assignment = PCS`), OmegaR of
T_CMB = 2.7255 K and N_eff = 3.046.

**The frozen reference does not deconvolve the CIC window** (SimulationCode's default `force_deconvolve_window = false`,
not set in its Lua files): the long-range force on the 64^3 grid is suppressed by W^2(k) near r_s, so the result
depends on the split. TreePMCode (which deconvolves, as GADGET-2) is 7% above `treepm_1024` at k = 1 h/Mpc at z = 1
and fails `check_against_reference.py`. The evidence (256 steps, P / reference at z = 1, k = 0.5 / 1 / 1.5 h/Mpc):

| | k = 0.5 | 1.0 | 1.5 |
|---|---|---|---|
| TreePMCode, `deconvolve = false` / SimulationCode `treepm_256` (the same settings) | 1.0003 | 1.0003 | 1.0015 |
| TreePMCode, PM 64^3, r_s = 1.25 Mpc/h / `treepm_256` | 1.044 | 1.077 | 1.084 |
| TreePMCode, PM 256^3, r_s = 0.31 Mpc/h / `treepm_256` | 1.043 | 1.072 | 1.080 |
| SimulationCode, PM 256^3, r_s = 0.31 Mpc/h / `treepm_256` | 1.044 | 1.076 | 1.086 |

With the window deconvolved the result does not depend on the split (64^3 or 256^3), and the two codes agree.
SimulationCode with `force_deconvolve_window = true` (otherwise the parameters of `treepm_1024`, 1024 steps, 8.5 min on
2 tasks) gives the corrected reference; against it, max |P / P_ref - 1|:

| | z = 1, k <= 1 | z = 1, to k_Ny | z = 0, k <= 1 | z = 0, to k_Ny | time (2 tasks) |
|---|---|---|---|---|---|
| TreePMCode, cell list, 1024 steps | 0.01% | 0.13% | 0.11% | 0.7% | 8.7 min |
| TreePMCode, tree, `opening_criterion = relative`, `relative_tolerance = 0.005` | 0.03% | 0.24% | 0.16% | 2.0% | 5.3 min |
| the same, `relative_tolerance = 0.002` | 0.08% | 0.30% | 0.23% | 0.9% | 5.8 min |
| (the same, 0.005, before the quadrupole-trace term in the tree) | 0.16% | 0.60% | 0.69% | 3.8% | 4.9 min |
| TreePMCode, cell list, 256 steps | 0.8% | 5.2% | 2.0% | 9.7% | 2.0 min |
| SimulationCode, 256 steps (PM 256^3) | 0.9% | 4.8% | 1.9% | 9.2% | 3.3 min |
| The frozen `treepm_1024` (window not deconvolved) | 7.2% | 7.9% | 4.9% | 6.6% | |

(At 256 steps both codes have the same time-step error.) The tree with the relative criterion is the faster backend
here (the geometric criterion accepts almost no nodes at this resolution: it is a direct sum within the cutoff, as the
cell list); the monopole alone biased the force low on the scale of the nodes (P(k) 4% low at k_Ny at z = 0), see
`ShortRangeGravity/README.md`.

## Limitations / next steps

- **Individual time steps:** the ghosts and the tree are rebuilt at every sub-step (GADGET-2 drifts the tree nodes
  instead); a larger box at higher resolution against EE2 (and GADGET-4) is the next validation.
- **Tree performance:** the relative opening criterion (GADGET-2) is 1.6-1.8 times faster than the cell list on the
  reference runs; the walk itself (recursive, per target) and the ghosts / tree rebuilt at every (sub-)step are next.
  Full quadrupoles (only the trace is included) would allow larger nodes.
- **CDM and baryons on the same lattice:** with a force that resolves the particle spacing each baryon is pulled back
  to its CDM partner, which damps the CDM-baryon difference (the split of coupled quintessence is 0.916 instead of
  0.785 with softening 1/4 spacing on the plane wave). Use a softening of about the spacing for two species, or the
  offset lattice; a proper treatment (e.g. adaptive softening, Angulo et al. 2013) is open.
- Load balancing by work (now by particle count), timings and memory at larger N, comparison with GADGET-4.
