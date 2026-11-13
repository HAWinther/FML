# Matter content: CDM, baryons, neutrinos and the N-body gauge

By default the particles are CDM+baryons (one species, "cb") and massive neutrinos are a linearly evolved grid
(`force_linear_massive_neutrinos`). CDM and baryons can be separate species, and the neutrinos can be particles.

## CDM and baryons as separate particles

By default CDM and baryons are one set of particles ("cb"). With `particle_baryons = true` they are two sets (one MPIParticles container each, `particle_Npart_1D^3` of each) with masses in the ratio Omegab / OmegaCDM. Both feel the gravity of the total CDM+baryon density delta_cb = (OmegaCDM delta_cdm + Omegab delta_b) / OmegaCB (plus linear massive neutrinos), so the baryons are collisionless (no hydrodynamics). What this gets right is the difference between baryons and CDM left over from before decoupling (the baryons lag behind the CDM and catch up). The initial conditions are made from the CDM and baryon transfer functions at zini (`ic_type_of_input = camb`, `class`, `internal` or `tables`):

- delta_i(k) = delta_cb(k) T_i(k,zini) / T_cb(k,zini) from the same random field;
- 1LPT velocities with f_i = f_cb + dlog(T_i/T_cb)/dlna, where f_cb is the growth rate of the gravity model. The total CDM+baryon velocity is then exactly the growing mode, and the relative velocity comes from the transfer functions. (The growth rates computed directly from the CAMB transfer functions are inaccurate near the ends of the redshift range of the table: from z = 49 they gave 0.8% too much CDM+baryon power.)
- the 2LPT displacement of the total field for both species.

By default both species start from the same particle lattice (`particle_baryon_lattice = "same"`). Interleaved lattices offset by half a particle spacing (`"offset"`, the usual choice in tree codes) give a large spurious baryon-CDM difference on small scales in this PM code. Tests: 64^3 particles of each species in 256 Mpc/h, z_ini = 49, 2x force grid, P(k) at z = 0. The relative difference is 1 - sqrt(P_b/P_cdm), compared with CAMB 1 - T_b/T_cdm:

| k (h/Mpc)                                    | 0.025 | 0.05  | 0.1   | 0.2   | 0.3   | 0.4    |
|----------------------------------------------|-------|-------|-------|-------|-------|--------|
| CAMB 1 - T_b/T_cdm (%)                       | 0.54  | 0.55  | 0.64  | 0.44  | 0.52  | 0.49   |
| linear amplitude, same lattice (%)           | 0.51  | 0.53  | 0.60  | 0.45  | 0.55  | 0.55   |
| linear amplitude, offset lattices (%)        | 0.54  | 0.62  | 0.96  | 2.06  | 4.40  | 8.06   |
| P_cb (two species) / P_cb (one species) - 1, same lattice (%) | 0.00 | 0.00 | -0.01 | 0.00 | 0.00 | 0.00 |
| ... offset lattices, linear amplitude (%)    | -0.06 | -0.14 | -0.55 | -2.4  | -5.6  | -10.5  |
| ... offset lattices, normal amplitude, 1x grid, 20 BullFrog steps (%) | 0.08 | 0.13 | 0.21 | -0.31 | -2.0 | -2.8 |

With the same lattice the CDM+baryon P(k) is the same as that of one species, and the baryon-CDM difference follows linear theory on all scales. With 20 BullFrog steps and a 1x force grid at normal amplitude it is 0.51 / 0.54 / 0.60 / 0.58 / 0.64 / 0.69 % (the nonlinear evolution changes it a little on small scales). With offset lattices the difference grows spuriously for k > k_Ny/8 (worse with a 1x force grid). Time stepping: BullFrog with 20 steps gets the CDM+baryon P(k) at z = 0 to 0.04% for k < 0.1 h/Mpc (relative to 300 logarithmic Quinn steps), and FastPM with 20 steps to 0.3%.

COLA: the two species share the COLA frame of the total CDM+baryon field (the LPT displacements of delta_cb, also scale-dependent), and their initial velocities are set relative to it: the baryon-CDM relative velocity, a^2 H (f_i - f_cb / R_i) Psi_i, is evolved by the PM part like everything else. The CDM+baryon P(k) is then the same as for one species with COLA (to < 0.01%, also with scale-dependent COLA and linear neutrinos). With offset lattices the frame of the baryons is that of their own lattice (scale-dependent COLA keeps the LPT potentials of the shifted CDM+baryon field for them, two more grids of size `ic_nmesh`; without them the baryon velocities at z_ini were 22% off). The relative mode is free of forces (its code velocity is constant), so the standard kick and drift integrate it exactly. The modified Tassev operators assume a residual that grows like LPT and get it wrong, so with baryons COLA needs `timestep_method = Quinn` or `FastPM`. P(k) of delta_c - delta_b (the relative mode) relative to Quinn without COLA with 200 logarithmic steps (64^3 of each species in 256 Mpc/h, z_ini = 49, 2x grid, z = 0; COLA with 20 linear steps):

| k (h/Mpc)                                       | 0.025 | 0.05  | 0.1   | 0.2   | 0.4   |
|-------------------------------------------------|-------|-------|-------|-------|-------|
| COLA + Quinn, linear amplitude (%)              | 0.0   | 0.0   | 0.0   | 0.0   | +0.1  |
| COLA + FastPM, linear amplitude (%)             | -4.1  | -3.5  | -3.6  | -3.0  | -3.2  |
| COLA + Tassev, linear amplitude (%)             | +34   | +29   | +30   | +25   | +26   |
| COLA + Tassev, logarithmic steps (%)            | +1.3  | +1.1  | +1.2  | +1.0  | +1.0  |
| COLA + Quinn, normal amplitude (%)              | -0.5  | -0.5  | -0.6  | -1.1  | -0.8  |
| scale-dependent COLA + Quinn, 0.4 eV linear neutrinos, normal amplitude (%) | -0.4 | -0.4 | -0.6 | -0.8 | -0.6 |

With COLA + Quinn the CDM+baryon P(k) is within 0.2% of the reference for k < 0.6 h/Mpc (Tassev: +1.1% at k = 0.6). NB: at normal amplitude the relative mode on large scales is about 30% smaller (in amplitude, at z = 0 and k = 0.03 h/Mpc) than linear theory with the default `particle_baryon_relative_mode = "displacement"`, with and without COLA, see below. Measure it from delta_c - delta_b: 1 - sqrt(P_b/P_c) is biased low at normal amplitude by the power the two species share.

### The baryon-CDM relative mode: displacements or masses

`particle_baryon_relative_mode` sets how the relative density delta_c - delta_b is represented:

- `"displacement"` (default): each species is displaced by its own linear field and has its own velocity (as above). The relative mode is then carried by the separation of each CDM-baryon pair (rms ~50 kpc/h at z = 0, from the same lattice point). Its size is conserved, but its direction is scrambled in collapsed regions (the coherent part of the pair separations is 0.9 where 1+delta < 0.5 and 0.1-0.2 where 1+delta > 30), so the net divergence of these dipoles, the relative density, is lost in halos: P(delta_c - delta_b) is about 0.5 of linear theory on large scales at z = 0 (propagator G = 0.70), and it gets worse with resolution. With a force grid finer than 2x the particle grid even the linear relative mode is wrong (a spurious discrete mode from the pairs attracting each other once the force resolves their separation; the code warns).
- `"mass"` (recommended for the relative mode): both species follow the total CDM+baryon flow (the same displacements and velocities as one species) and the relative density is carried by the particle masses m_i = 1 + eps_i(q), eps_i(k) = delta_cb(k,zini) (T_i - T_cb)(k,z=0) / T_cb(k,zini) (the compensated mode of Hahn, Rampf & Uhlemann 2021: w_c eps_c + w_b eps_b = 0, so delta_cb and the forces are those of one species). The masses carry the linear relative mode of z = 0, which is constant at late times in the absence of pressure: at linear amplitude it is exactly the CAMB relative mode at z = 0 and too large by (T_c - T_b)(z=0) / (T_c - T_b)(z) at earlier times: 1.5% at z = 1, 4% at z = 3, 9% at z = 10 (in amplitude, k >= 0.05 h/Mpc). The density assignment weights the particles by their masses (so do the P(k) of each species, FoF and SO halos), and the masses are written to the GADGET files (MassTable = 0 for types 1 and 2, a MASS block in binary files, `PartTypeN/Masses` in HDF5).
- `"mass_ini"`: common displacements, masses with eps_i = T_i/T_cb - 1 at zini and each species keeps its own velocity (f_i T_i/T_cb Psi_cb), so the relative mode grows from zini as in linear theory, but the part that grows (~20% of the amplitude at z = 0) is again carried by displacements.

Tests (64^3 particles of each species in 256 Mpc/h, z_ini = 49, 200 logarithmic Quinn steps, 2x force grid, the same seed): P(delta_c - delta_b) at z = 0 relative to the relative mode of a run with 1e-4 times the power (displacement mode, which follows linear theory), and the propagator G = P_cross / P_lin:

| k (h/Mpc) | displacement P/P_lin, G | mass P/P_lin, G | mass_ini P/P_lin, G |
| :-: | :-: | :-: | :-: |
| 0.031 | 0.49, 0.70 | 1.03, 1.01 | 0.87, 0.93 |
| 0.055 | 0.52, 0.70 | 1.04, 1.00 | 0.92, 0.94 |
| 0.100 | 0.49, 0.60 | 0.98, 0.88 | 0.86, 0.82 |
| 0.197 | 1.19, 0.34 | 2.13, 0.55 | 1.86, 0.50 |

The mass encoding at linear amplitude is 1.5-3% (in amplitude) above the linear-amplitude reference at k = 0.03-0.3 (the N-body evolution of the relative mode from zini, with displacements or with mass_ini, ends slightly below the CAMB relative mode at z = 0), so relative to its own linear run "mass" has G = 0.98 / 0.98 / 0.86 at k = 0.031 / 0.055 / 0.10: the remaining loss is the nonlinear propagator of the flow that carries the masses. On small scales P(delta_c - delta_b) is dominated by nonlinear power in all the modes. The CDM+baryon P(k) is the same in the three modes (to 0.02% for k < 1.5 h/Mpc). With a 4x force grid "mass" gives the same relative mode (linear amplitude: G = 1.000; normal amplitude: G = 0.98-1.00 relative to the 2x grid), while "mass_ini" loses 15-20% of the linear relative mode (G = 0.82-0.86) and "displacement" 70-98% of its power. With COLA (Quinn) the mass modes give the same relative mode as without (to 0.1%). The mass modes need `particle_baryon_lattice = "same"` and use a particle type with a mass (8 more bytes per particle).

Output: P(k) of CDM (`pofk_cdm_zX.txt`), baryons (`pofk_baryon_zX.txt`) and CDM+baryons (`pofk_zX.txt`, as before) with linear theory P_cb (T_i/T_cb)^2. The same files are written for every step. GADGET (binary and HDF5) files have the CDM as type 1 and the baryons as type 2 (a collisionless type: type 0 is SPH gas in GADGET, which many readers expect to have gas fields). Baryon IDs follow the CDM IDs.

Dark scattering (`cosmology_model = DarkScattering`, see [MODELS.md](MODELS.md)): the CDM particles feel the full
drag Gamma and the baryon particles none (with one set of CDM+baryon particles: f_c Gamma). Both species take the
steps of the CDM+baryon fluid (the same scale factors) with their own kick, drift and velocity factors
(`src/SpeciesTimeStepping.h`): Quinn and Tassev from the friction of the species, FastPM from the linear growth of
CDM and baryons as two fluids sourced by the same delta_cb (D_c, D_b; the kick with the force of D_cb), so each
species follows its linear growing mode exactly. BullFrog is not supported (its coefficients are those of one
fluid). With COLA the frame is the CDM+baryon one (friction f_c Gamma); a species with another friction gets
(velocity factor of the species - that of the frame) v_LPT in every kick, the part of the change of its LPT velocity
that the frame's kick does not have (without it the CDM-baryon split below was -5.6% / -4.5% instead of
-6.2% / -1.3%). The growth from z = 49 to 0 relative to wCDM (w = -0.9, xi = 50 barn/GeV, 1e-4 times the power,
32^3 of each species in 1000 Mpc/h, 2x force grid, k < 0.05 h/Mpc), each run against wCDM with the same time
stepping, and the linear growth of the N-body equations (two fluids with the drag, the dark energy at rest):

| | CDM | baryons | CDM+baryons |
|---|---|---|---|
| Two-fluid linear growth (the N-body equations) | -6.16% | -1.31% | -5.42% |
| Quinn, 200 logarithmic steps | -6.12% | -1.30% | -5.38% |
| FastPM, 10 steps | -6.12% | -1.30% | -5.38% |
| COLA + Quinn, 10 steps | -6.14% | -1.25% | -5.39% |
| COLA + FastPM, 10 steps | -6.12% | -1.34% | -5.38% |
| one species (f_c Gamma), Quinn, 200 steps | | | -5.42% |

The Einstein-Boltzmann solver (its transfer functions at z = 49 and 0) agrees with this on small scales but its
dark energy is not at rest on large scales: the drag also pulls the dark energy along with the dark matter
(Gamma_DE = Gamma_c R with R = rho_c / ((1 + w) rho_DE) >> 1), which reduces the relative velocity and so the drag.
CDM / baryons, simulation (Quinn) / solver: -6.16% / -4.90%, -1.32% / -0.64% at k = 0.008 h/Mpc; -6.11% / -5.81%,
-1.30% / -1.09% at k = 0.032; -5.97% / -5.95%, -1.22% / -1.19% at k = 0.088. The N-body equations (as in
Baldi & Simpson 2015 and the other N-body work on these models) neglect the velocity of the dark energy, so they
overestimate the effect on large scales (here k < ~0.05 h/Mpc; more for w closer to -1, see the covariantised model
below).

Coupled quintessence (`cosmology_model = CoupledQuintessence`, see [MODELS.md](MODELS.md)): the CDM particles feel
gravity and the fifth force of the CDM density, D^2 Phi_5 = 2 beta^2 1.5 OmegaCDM m(a) a delta_c (a second kick
after the PM kick, in the same grids: `kick_cdm_fifth_force`), and the velocity term +beta phi' v; the baryons only
gravity. delta_cb is weighted with the CDM fraction f_c(a) of the varying CDM mass. The steps are as for dark
scattering, FastPM with the two-fluid growth with the couplings (the CDM feels 1 + 2 beta^2 of the CDM). The growth
from z = 49 to 0 relative to uncoupled quintessence (lambda = 0.5, beta = 0.15, as above):

| | CDM | baryons | CDM+baryons | delta_b / delta_c at z = 0 |
|---|---|---|---|---|
| Einstein-Boltzmann solver | +14.23% | +10.74% | +13.32% | 0.9327 |
| Quinn, 200 logarithmic steps | +14.10% | +10.58% | +13.18% | 0.9324 |
| FastPM, 10 steps | +14.11% | +10.59% | +13.18% | 0.9325 |
| COLA + Quinn, 10 steps | +14.11% | +10.58% | +13.19% | 0.9323 |
| COLA + FastPM, 10 steps | +14.10% | +10.59% | +13.18% | 0.9325 |

(uncoupled: delta_b / delta_c = 0.9960, solver 0.9961). Per k the simulations agree with the solver to 4-7e-4 at
k = 0.02-0.03 h/Mpc (one species), with larger differences in the largest-scale bin (4e-3) and towards k_Ny of the
particles (the resolution, 1-3e-3 at k = 0.05-0.09). The growth of the species from z = 49 includes the baryons
catching up with the CDM (delta_b / delta_c = 0.865 at z = 49, 0.898 uncoupled), so it is not that of the two-fluid
growing mode (CDM +14.4%, baryons +7.0%); its D_b / D_c = 0.9355 is the ratio the solver approaches by z = 0.

With the same lattice keep the force grid at most twice the particle grid: a finer grid resolves the attraction
between the CDM and baryon particles that start at the same point, which damps the CDM-baryon difference the drag
builds up (CDM / baryons with Quinn: -6.01% / -1.22% with a 1x grid, -6.12% / -1.30% 2x, -5.90% / -2.62% 4x,
-5.55% / -4.63% 8x).

Covariantised dark Thomson-like scattering (`cosmology_model = CovariantDarkScattering`, see [MODELS.md](MODELS.md)):
as dark scattering with the drag Gamma = alpha a^3 / (Omega_c0 E) of the dark matter. The growth from z = 49 to 0
relative to alpha = 0 (alpha = 1, w = -0.98, as arXiv:2510.12551; set up as above, k < 0.05 h/Mpc):

| | CDM | baryons | CDM+baryons | delta_b / delta_c at z = 0 |
|---|---|---|---|---|
| Two-fluid linear growth (the N-body equations) | -9.13% | -0.32% | -7.77% | 1.0969 (growing mode) |
| Quinn, 200 logarithmic steps | -9.03% | -0.32% | -7.69% | 1.0915 |
| FastPM, 10 steps | -9.04% | -0.32% | -7.70% | 1.0915 |
| COLA + Quinn, 10 steps | -9.07% | -0.57% | -7.76% | 1.0891 |
| COLA + FastPM, 10 steps | -9.00% | -0.76% | -7.73% | 1.0863 |
| one species (f_c Gamma), Quinn, 200 steps | | | -7.91% | |

The drag grows as a^3 (Gamma = 3.7 per ln a today, 0.26 at a = 0.5), so it acts mostly at z < 1, and one set of
CDM+baryon particles (f_c Gamma) overestimates the suppression of the CDM+baryon growth by 2.4e-3 (the momentum-
weighted one-fluid approximation; with separate particles the CDM and baryons follow the two-fluid growth). With
w = -0.98 the dark energy is strongly dragged along (R ~ 1/(1 + w)): relative to the solver the N-body suppression
of the CDM growth is -9.1% / -6.2% at k = 0.008 h/Mpc, -9.0% / -8.5% at k = 0.032 and -8.7% / -8.7% at k >= 0.07
(v_DE / v_c today in the solver: 0.43 at k = 0.0075, 0.13 at 0.016, 0.03 at 0.033, < 0.01 at k > 0.07 h/Mpc); the
baryons agree to 1e-4 for k >= 0.05. So the N-body equations (the dark energy at rest, as in the N-body work on this
model) hold for k > ~0.05 h/Mpc.

Not implemented for two species yet (the code stops with a message): COLA with the Tassev operators, reading the IC from file, NUFFT / sheet densities, force interlacing and the early-time force, the N-body gauge, the lightcone, multipoles and the bispectrum (the full list is the table at the end of [PARAMETERS.md](PARAMETERS.md)). Use `ic_input_redshift = ic_initial_redshift` (the two species are made from the transfer functions at zini).

## Neutrino masses

There are N_nu = 3 neutrino species with the temperature T_nu = T_CMB (Neff/3)^(1/4) (4/11)^(1/3). Their masses are
given in one of two ways:

- `cosmology_OmegaMNu > 0`: 3 equal masses, with the total Mnu from OmegaMNu (the non-relativistic relation), as before.
- `cosmology_neutrino_masses_eV = {m1, m2, m3}` (up to 3 entries in eV; 0 or not listed: massless), e.g.
  `{0.0, 0.0086, 0.05}` for the normal hierarchy with a massless lightest neutrino. `cosmology_OmegaMNu` must then be
  0; OmegaMNu (the particle masses, f_nu, the Poisson normalization) follows from the masses with the same
  non-relativistic relation.

The background (`Cosmology`, FML/Cosmology/Background) has each massive species with its own Fermi-Dirac density.
The linear neutrinos (`force_linear_massive_neutrinos`) and the scale-dependent growth use the transfer functions of
all the neutrinos together, so they work for any masses as long as the transfer functions are those of the same masses
(`ic_type_of_input = internal` runs the solver with them; equal masses are one hierarchy there, so 2 or 3 distinct
masses cost more: an estimated 1.3-2x the solver time of one mass). CAMB and CLASS input must be made with the same
masses (the code only checks that both have massive neutrinos). The nonlinear EuclidEmulator2 boost takes the total mass.
Neutrino particles with different masses: see below.

## Massive neutrinos as particles (delta-f method)

By default massive neutrinos are included with linear theory (`force_linear_massive_neutrinos`). With `particle_neutrinos = true` they are particles instead (`particle_neutrino_Npart_1D^3` of them, default `particle_Npart_1D`), evolved with the delta-f method of Elbers et al. 2021 ([arXiv:2010.07321](https://arxiv.org/abs/2010.07321)):

- Initial conditions: the bulk part is made like the CDM, delta_nu(k) = delta_cb(k) T_nu(k,zini) / T_cb(k,zini) from the same random field, with 1LPT displacements and velocities from the neutrino growth rate f_nu(k,zini) of the transfer functions. Each particle then gets a thermal momentum q = y k_B T_nu0 / c in a random direction with y drawn from the Fermi-Dirac distribution y^2 / (e^y + 1) (the random numbers depend only on `ic_random_seed` and the particle ID, not on the number of tasks). The neutrinos are N_nu = 3 degenerate species with m_nu = Sum m_nu / 3.
- Relativistic equations of motion. The code velocity is the comoving momentum q / (m H0 B). With u = q / (a m c): dx = v dt_drift / sqrt(1 + u^2) (u at the middle of the drift) and dv = -F dt_kick (1 + 2u^2) / sqrt(1 + u^2) (the geodesic equation in the Newtonian gauge).
- delta-f weights w = 1 - f0(y) / f_i, where f_i = f0(y_ini) (1 + delta_ini) is the phase-space density the particle carries (conserved along its path) and f0(y) the unperturbed distribution at its current momentum. The weighted density (N_grid^3 / N_nu) Sum w_i W(x - x_i) is delta_nu: the particles only carry the perturbation, so the shot noise is that of w (Sum w^2 / N^2 instead of 1 / N). `particle_neutrino_deltaf = false` uses plain particles (w = 1), for testing.
- The force is computed from delta_m = (1 - f_nu) delta_cb + f_nu delta_nu with f_nu = OmegaMNu / OmegaM (the same non-relativistic mass weighting as the linear option; the neutrino energy density is larger than the rest mass density at early times: by 33% at z = 50 and 7% at z = 20 for 0.1 eV, 3% at z = 50 for 0.4 eV).

Needs `ic_type_of_input = camb`, `class`, `internal` or `tables` with tables up to zini and accurate massive neutrino transfer functions (of the same masses as the simulation). `internal` (or its `tables`) needs nothing else; `input/transfer_tables/lcdm_nu0.1.txt` and `lcdm_nu0.4.txt` are the tables of the tests (OmegaM = 0.3175, OmegaMNu = 0.0024 and 0.0095, `cosmology_OmegaCDM = 0.2661` and `0.259`). With CAMB (`scripts/make_camb_transfer_tables.py`) use an accuracy boost: with the default accuracy CAMB underestimates P_nu below the free-streaming scale (for 0.1 eV by 3 / 13 / 16% at k = 0.014 / 0.05 / 0.1 h/Mpc at z = 0, relative to CAMB with AccuracyBoost = 3 and to CLASS without the fluid approximation).

Tests: 1024 Mpc/h, 128^3 CDM+baryon and 128^3 neutrino particles, 256^3 force grid, z_ini = 50, 40 logarithmic Quinn steps, fixed amplitude IC. P_nu at z = 0 (shot noise subtracted) relative to linear theory (CAMB, from the measured P_cb at z_ini), and P_cb / P_total compared with the same simulation with linear neutrinos:

| k (h/Mpc)                                         | 0.01  | 0.02  | 0.03  | 0.05  | 0.075 | 0.1   | 0.15  |
|---------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|
| 0.4 eV: P_nu / P_nu,lin                           | 0.997 | 0.992 | 1.008 | 1.019 | 1.005 | 0.993 | 1.019 |
| 0.4 eV: P_cb (particles) / P_cb (linear nu) - 1 (%) | 0.17 | 0.09 | 0.04 | 0.01 | 0.01 | 0.02 | 0.03 |
| 0.4 eV: P_total ... - 1 (%)                       | 0.17  | 0.06  | 0.03  | 0.08  | 0.11  | 0.18  | 0.28  |
| 0.1 eV: P_nu / P_nu,lin                           | 1.050 | 1.032 | 1.035 | 0.995 | 0.950 | 0.958 | 0.931 |
| 0.1 eV: P_cb (particles) / P_cb (linear nu) - 1 (%) | 0.08 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.1 eV: P_total ... - 1 (%)                       | 0.08  | 0.02  | 0.01  | 0.00  | 0.00  | 0.01  | 0.01  |

(P_cb / P_cb,lin of the simulation itself is 0.95-1.05 on these scales because of the time stepping and the force resolution, and P_nu follows it; relative to that P_nu agrees with linear theory to 2% for 0.4 eV and to 7% for 0.1 eV; the difference for 0.1 eV does not change with the number of steps or z_ini.) The shot noise of the neutrinos at z = 0 is 136 (Mpc/h)^3 for 0.4 eV (rms weight 0.52) and 4.2 (Mpc/h)^3 for 0.1 eV (rms weight 0.09), compared with V / N = 512 (Mpc/h)^3 for plain particles; it equals P_nu,lin at k = 0.12 h/Mpc in both cases. The weights grow as gravity changes the neutrino momenta, so the gain is largest at high redshift and for light neutrinos (the rms weight is 0.14 for 0.4 eV and 0.02 for 0.1 eV at z = 2). For heavy neutrinos the bulk velocities at late times are comparable to the thermal velocities and the weights become of order one. The relativistic corrections change P_nu by about 2% for 0.1 eV. The neutrinos add about 15% to the run time with as many neutrino as CDM particles.

Output: P(k) of the neutrinos (`pofk_nu_zX.txt`, with the shot noise in the header) and of the total matter (`pofk_total_zX.txt`) at the outputs and every step. GADGET (binary and HDF5) files have the neutrinos as type 3 (HDF5: with their weights in `PartType3/Weights`; the particle mass is that of a plain particle, the effective mass is mass * weight). Their velocities are q / (a m), which differs from the peculiar velocity by the factor sqrt(1 + u^2). Neutrino IDs follow the CDM (and baryon) IDs.

Different masses (`cosmology_neutrino_masses_eV`) are one set of particles: all species have the same temperature and
number density, so each particle is one of the massive species at random (from a hash of its ID: the same split for
every number of tasks and every seed, equal numbers of each on average). Its thermal velocity uses its own mass (y is
drawn as for one mass, v = y (T_nu0 / m_i) c), and so does its delta-f weight; the weight also carries m_i / <m> (the
mean mass of the massive species), so the weighted density is the mass-density contrast of all the neutrinos (the
GADGET `Weights` then include this factor). The bulk part of the initial conditions (delta_nu and the growth rate) is
that of all the neutrinos together (the transfer functions of their sum) for every particle; with very different
masses the heavier ones cluster more already at zini, which this misses (small at high zini). With equal masses
nothing changes (bitwise the same as before). Not run with unequal masses yet.

Not implemented for neutrino particles yet (the code stops with a message): time stepping other than Quinn, COLA, reading the IC from file, NUFFT / sheet densities, force interlacing and the early-time force, the N-body gauge, the lightcone, multipoles and the bispectrum (see the table at the end of [PARAMETERS.md](PARAMETERS.md)). The initial conditions use the same density perturbation for all momenta (Elbers et al. use momentum-dependent initial conditions from integrating linear geodesics, which is more accurate for light neutrinos at high redshift).

## Radiation, neutrinos and GR: the N-body gauge

On very large scales (k < ~0.01 h/Mpc) a Newtonian simulation misses the effect of the radiation
(photons and massless neutrinos), and of the massive neutrinos, on the growth, and the meaning of the
particle positions is ambiguous. In the N-body gauge (Fidler et al. 2015, 2016) the particles follow the
Newtonian equations exactly to linear order in GR if the force is sourced by
rho_cb (delta_cb + delta_extra), where delta_extra(k,z) is a linear field that contains everything else
(photons, neutrinos and a metric term). With `force_nbody_gauge = true` we add this source to the density
field every time we compute the force, and the initial conditions are made from the N-body gauge transfer
functions (with the scale-dependent growth rate for the velocities).

With `ic_type_of_input = "internal"` nothing else is needed: the Einstein-Boltzmann solver gives the N-body
gauge transfer functions and the source directly, for LCDM, w0waCDM, quintessence and EDE (with GR). The
source is computed from the comoving Poisson equation as a sum of small terms, with smoothed time
derivatives; integrating the linear growth with it reproduces the solver's N-body gauge delta_cb to
about 1e-5 (LCDM, with and without massive neutrinos) and 1e-3 at k = 2e-4 h/Mpc for w0wa.

With `ic_type_of_input = "camb"` the input is made with CLASS (`pip install classy`) for the cosmology in
the parameterfile:

```bash
python3 scripts/make_nbody_gauge_transfer.py parameterfile.lua --outdir nbgauge_input
```

This writes the N-body gauge transfer functions (CAMB format, with a transfer info file) and the table
T_extra(k,z) (`force_nbody_gauge_filename`), checks the gauge transformation against CLASS, shows how big
the effect is for your initial redshift and prints the lines to put in the parameterfile. Only for LCDM
with GR; the code checks that the table was made for the same cosmology and that H(z) agrees.

For a start at z = 49 the effect on P(k) at z = 0 is about 2% at k = 1e-3 h/Mpc (radiation, without
massive neutrinos) and much less than 0.1% for k > 0.01 h/Mpc. `tests/check_nbody_gauge.py` checks that the
simulation follows the linear N-body gauge evolution from CLASS on the largest scales.
