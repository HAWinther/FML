# Initial conditions and input

The particle positions and velocities are made with 1LPT, 2LPT or 3LPT (`ic_LPT_order`) from a Gaussian or
non-Gaussian initial density field. When the initial conditions are made the code stores what the particle type has
room for: the initial Lagrangian position, a unique ID and the 1LPT and 2LPT displacement fields (for COLA).

## Random fields

The initial field can be generated in Fourier space or from a white-noise field in real space, with a choice of
random number generator. `ic_random_field_backend = "ngenic"` (the default) is the generator of 2LPTic, L-PICOLA and
MG-PICOLA, so runs can start from exactly the same initial conditions as those codes (including their line seeding
with repeated seeds); `ic_random_generator` (`GSL` or `MT19937`) applies to it. `"philox"` is the mode-keyed
alternative, independent of the number of MPI tasks and threads. See [RandomFields](../RandomFields/README.md) for
the reproducibility conventions.

Amplitude-fixed fields (Angulo & Pontzen 2016, arXiv:1603.05253; Villaescusa-Navarro et al. 2018,
arXiv:1806.01871) are an option, and the phases can be inverted for paired-fixed simulations.

## Non-Gaussian initial conditions

Local, equilateral and orthogonal fNL. The non-Gaussianity is added to the potential Φ = (3/5) g(z) ζ at
`ic_fnl_redshift`, with g = D/a the linear growth suppression of the potential: `ic_fnl_redshift = 0` is the usual
large-scale-structure convention, a redshift in the matter era such as 30 the primordial/CMB convention
Φ = (3/5) ζ; fNL_LSS = fNL_CMB / g(z = 0), about 1.27 fNL_CMB for OmegaM = 0.32.

## Input spectra

The initial conditions need a CDM+baryon power spectrum or transfer function in the CAMB format (k in h/Mpc and P in
(Mpc/h)^3, or T in Mpc^2). Simulations with massive neutrinos need the transfer functions of all species at all
redshifts: from FML's own Einstein–Boltzmann solver (`ic_type_of_input = internal`, or its tables written once with
`./nbody --write-transfers` and read with `ic_type_of_input = tables`), CAMB (`camb`, a transfer info file listing
the CAMB transfer files) or CLASS (`class`); see [Transfer function input](#transfer-function-input) below. The
input should preferably be at z = 0, so that the scaling back to the initial redshift is as accurate as possible,
but any redshift works (set `ic_input_redshift`).

## Transfer function input

The transfer functions (`ic_type_of_input = camb`, `class`, `internal` or `tables`) are read with
[FML/TransferProviders](../TransferProviders/README.md) (`src/TransferInput.cpp`): one `TransferProvider`, shared by the
gravity model (the growth factors with massive neutrinos), the initial conditions (species ratios, two species,
neutrino particles, the N-body gauge) and the linear predictions in the outputs. Power/transfer preparation and the
species-ratio helpers use the shared [InitialConditions module](../InitialConditions/README.md). Simple P(k,z=0) input
remains supported without a Boltzmann solve.

- `camb` (the old name `transferinfofile` still works, with a warning): CAMB transfer files listed in a transfer info
  file. The columns are read by name from the header line (or by position without one: 13 columns CAMB, 9 AxionCAMB,
  7 old CAMB or CLASS with `format = camb`),
  so `ic_type_of_input_fileformat` is no longer needed (it is ignored with a warning). Files not found in the folder
  of the info file are looked for next to it.
- `class`: native CLASS output (`format = class`, `output = mTk,vTk`, `write_background = yes`), with
  `ic_input_filename` the root of the run (e.g. `output/run_` for `output/run_z1_tk.dat`, ..., `output/run_background.dat`)
  and `ic_input_gauge` the gauge CLASS was run in (`synchronous`, CLASS's default, or `newtonian`, which is converted to
  the CDM-synchronous gauge at the tabulated redshifts). h and the presence of massive neutrinos are checked against
  the simulation. `tests/check_class_input.py` runs a simulation with massive neutrinos from CLASS in both gauges and
  from CAMB for the same cosmology: P_cb and P_total agree to 1.4e-3 (synchronous) and 7e-4 (Newtonian) at all outputs.
- `internal` (the old name `internal_transfer` also works): FML's
  Einstein-Boltzmann solver ([Cosmology/EinsteinBoltzmannSolver](../Cosmology/EinsteinBoltzmannSolver/README.md)) is
  run when the simulation starts, with the cosmology of the simulation, so no input file is needed:
  - the background parameters (h, Omegab, OmegaCDM, OmegaK, TCMB, Neff) and the massive neutrinos as the simulation
    has them: N_nu = 3 degenerate species of mass Mnu/3 (the solver's `deg_ncdm`; the degenerate species are solved as
    one), the primordial spectrum (As, ns, kpivot; rescaled with `ic_sigma8_normalization`);
  - `cosmology_model` with its parameters: the solver has the same models (LCDM, w0waCDM, DGP, JBD, Quintessence,
    EDE), with their perturbations (smooth dark energy for w0waCDM, the scalar field of quintessence and EDE, the
    modified gravity of DGP and JBD). Not `Tabulated` (no perturbations). The gravity model of the simulation
    (`gravity_model`) is separate: the IC are those of the cosmology model at zini.
  - The other settings of the solver are its defaults unless given in a solver .ini file (`ic_input_filename`,
    optional) or as `ic_internal_solver_parameters = {"name=value", ...}` in the parameter file (applied after the
    .ini, so they override it):

    ```lua
    ic_type_of_input = "internal"
    ic_input_filename = ""   -- or "solver.ini" with "name = value" lines
    ic_internal_solver_parameters = {"Yp=0.245", "z_reion=8", "precision=fast"}
    ```

    [`internal_solver_default.ini`](internal_solver_default.ini) has the settings that are yours here
    (recombination, reionization, the background range and the precision of the perturbations) with their defaults
    and descriptions: copy it, change what you need and give it as `ic_input_filename` (as it is, it gives the same
    results as no file). It is generated from the solver's parameter table with
    `./nbody --write-solver-defaults internal_solver_default.ini` (CI checks that it is up to date). All the
    parameters of the solver are in
    [`params/default.ini`](../Cosmology/EinsteinBoltzmannSolver/params/default.ini) (`./cmb --write-defaults` writes
    it). What is whose:

    | Solver parameters | |
    |---|---|
    | `h`, `OmegaB`, `OmegaCDM`, `OmegaK`, `TCMB`, `Neff`, `m_nu`, `deg_ncdm`, `m_nu_list`, `T_ncdm_over_TCMB`, `model` and its parameters, `A_s`, `n_s`, `k_pivot`, `preset` | Set by the simulation (`cosmology_*`): an error if given |
    | `alpha_s` | 0 (the primordial spectrum of the simulation has no running): an error if given |
    | `pert_k_min`, `pert_k_max`, `pert_n_k`, `pert_dk_max` | The k-grid below, set from the IC grid: an error if given |
    | Recombination and reionization (`Yp`, `Yp_bbn`, `reionization`, `z_reion`, `recombination`, `rec_*`, ...) | Yours |
    | Precision (`precision = fast`, `ode_abserr`, `ode_relerr`, `lmax`, `lmax_nu`, `lmax_ncdm`, `n_q_ncdm`, `ncdm_fluid_trigger`, `tc_*`, `method_*`, `pert_x_start`, `pert_n_x_*`, `bg_*`, ...) | Yours (`bg_x_end` is raised past the last output if needed) |
    | Isocurvature (`f_cdi`, ...) | An error when the solver starts (adiabatic IC only) |
    | Tensors, line of sight, lensing, outputs (`r`, `lensing`, `ells`, `los_*`, `output_dir`, `write_*`, ...) | Accepted but unused (only the perturbations are solved) |

    The log prints the solver parameters given. A model of the solver that `cosmology_model` does not have (f(R),
    Horndeski, ...) cannot be used: the model is always that of the simulation.
  - Use `ic_input_redshift = ic_initial_redshift` (the transfer functions are then taken at zini, with no scaling by
    the growth factor).

  The solver runs on task 0 (with OpenMP) and its tables are sent to the other tasks. k: 15 points per decade from
  1e-4 h/Mpc (or half the box fundamental mode) and 50 per decade in [0.01, 0.6] h/Mpc where the BAO are (the
  interpolation is then good to ~1e-4) up to max(5 h/Mpc, 1.5 x the largest mode of the IC grid); above that the
  tables are extrapolated as the CAMB and CLASS tables. The perturbations are solved to a = 1.1 and tabulated at 128
  scale factors from zini (half the scale factor) to then. Time: ~30 s on 2 cores (kmax = 5 h/Mpc, with or without
  massive neutrinos); the cost grows with kmax (the photon and neutrino hierarchies are solved until today).
  Test (the regression cases with CAMB input, 32^3 particles, same seed): P(k) at z = 0 and 1 for k < 0.5 h/Mpc agrees
  with the CAMB runs to 3e-4 without neutrinos and to 0.4% with 0.4 eV (3 degenerate species in both) and with
  neutrino particles; the 0.2 eV CAMB input has one massive species and two massless (1-1.5% different, as expected).
  The solver's neutrino velocities near its fluid switch differ from CLASS at high redshift (see
  [TransferProviders](../TransferProviders/README.md)).

- `tables`: the tables of the internal solver in one text file, so that the solver runs once per cosmology instead
  of at every start. Write them with the parameterfile you would run with `ic_type_of_input = internal` (and
  `force_nbody_gauge = true` for N-body gauge tables, which then include the source):

  ```bash
  ./nbody --write-transfers parameterfile.lua transfers_lcdm.txt
  ```

  and run with `ic_type_of_input = "tables"` and `ic_input_filename = "transfers_lcdm.txt"`. The file
  (`FML/TransferProviders/TransferTablesFile.h`) is self-describing: a header with the cosmology (the background
  parameters with the neutrino masses, `cosmology_model` and its parameters), the primordial spectrum, the gauge, the
  settings of the solver and the column names, then one row per (a, k) with delta/R and theta/(aH R) of every
  species (and the N-body gauge source). When it is read the cosmology, the primordial spectrum and the gauge are
  checked against the simulation (an error if they differ), and the tables must reach the initial redshift (they
  start at half the scale factor of the `ic_initial_redshift` they were written for). A run from the file gives the
  same results as `internal` (the particles and P(k) were identical in the test; the values are written with 10
  digits). The tests use `input/transfer_tables` (LCDM with 0, 0.1, 0.2 and 0.4 eV and in the N-body gauge, written
  for zini = 20; `tests/make_transfer_tables.py` remakes them).

### Ultralight axions in the initial conditions only

With `ic_type_of_input = internal` (or `tables` written that way) `ic_axion_fraction > 0` makes the share
`ic_axion_fraction` of the dark matter (`cosmology_OmegaCDM`) an ultralight axion of mass 10^`ic_axion_log10_m` eV in
the initial conditions only: the solver runs its `model = axion` (mixed axion + CDM dark matter as axionCAMB;
`ic_axion_switch_mH` is its `ax_switch_mH`) and the CDM of its tables is CDM + axion (density and momentum weighted,
the axion in the CDM-synchronous gauge), so the particles, which are all the dark matter (+ baryons), start with the
axion's suppression of the small-scale power and the matching velocities. The simulation itself is ΛCDM
(`cosmology_model = LCDM`): its background is the same once the axion redshifts as matter (the solver's Ω_ax + Ω_c =
`cosmology_OmegaCDM`, Λ from closure), and its dynamics neglect the axion's pressure (quantum pressure / Jeans scale)
after the start. The axion must be a matter-like fluid before the start of the tables (half the initial scale factor;
an error otherwise): with the default switch m ≳ 10⁻²⁶ eV for z_ini ~ 50. `cosmology_model = LCDM` only, and not with
`force_nbody_gauge`. With `ic_input_redshift = ic_initial_redshift` the IC are those of the axion model at zini;
with a later input redshift the scaling back uses the growth of the gravity model (ΛCDM), which keeps the axion's
late-time suppression in the IC.

```lua
ic_type_of_input = "internal"
ic_axion_fraction = 0.1
ic_axion_log10_m = -22
```

The CAMB and CLASS tables are interpolated in (log a, log k) and extrapolated above their k_max as a power law, as
before. Compared with the old reader (`LinearTransferData`, interpolation in z) the regression outputs of the runs with
transfer tables change by up to 3e-4 in P(k) (5e-4 for P_nu).

## Initial conditions from a snapshot

The particles can also be read from GADGET binary or HDF5 files (`ic_reconstruct_gadgetfilepath`; the library also
has a RAMSES reader) and evolved from there. For COLA the LPT displacement fields are reconstructed from the
particle distribution, assuming the initial conditions are 1LPT: this works well for initial conditions at not too
low redshift. This is mostly useful for tests, e.g. comparisons with high-resolution simulations run with RAMSES,
GADGET or PKDGRAV from the same initial conditions.
