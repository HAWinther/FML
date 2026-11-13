# Models

A simulation has a **cosmology** (`cosmology_model`): the background expansion, H(a) and the densities. It also has
a **gravity model** (`gravity_model`), which sets the growth of the perturbations: the LPT growth factors of the
initial conditions and COLA, and how the force is computed from the density field. The two are chosen separately,
so one can be changed while the other stays fixed. All their parameters are in [PARAMETERS.md](PARAMETERS.md).

The background comes from FML's background module (`FML/Cosmology/Background`, shared with the Einstein–Boltzmann
solver): `src/Cosmology.h` maps `cosmology_model` and its parameters to a model of that module's registry. How to
add a model is in [DEVELOPING.md](DEVELOPING.md#adding-a-model).

## Cosmologies

* All cosmologies have CDM, baryons, photons, neutrinos (massless and massive), curvature, a cosmological constant
  and the primordial parameters (As, ns, kpivot). The massive neutrinos are treated exactly in the background
  (splines of the Fermi–Dirac integrals).

* LCDM: ΛCDM.

* w0waCDM: dark energy with the Chevallier–Polarski–Linder equation of state w(a) = w0 + wa (1 − a).

* JBD: Jordan–Brans–Dicke, with a time-varying gravitational constant. Two parameters: the Brans–Dicke parameter
  wBD (GR as wBD → ∞) and the value of Newton's constant today relative to the measured one (useful e.g. to test how
  well cosmology constrains it).

* DGP: the self-accelerating DGP cosmology. One parameter: the cross-over scale.

* Quintessence: a scalar field phi as the dark energy, with the potential V = V0 u(phi) (phi in units of the reduced Planck mass): `exponential` (u = exp(-lambda phi)), `inverse_power_law` (u = phi^-alpha, Ratra-Peebles, has a tracker solution), `pngb` (u = 1 + cos(phi/f), axion-like thawing) or `hilltop` (u = (1 - (phi/mu)^p)^2, flat top at phi = 0). The field starts frozen at `cosmology_quintessence_phi_ini` deep in the radiation era (optionally with a velocity `cosmology_quintessence_dphidlna_ini`) and V0 is found so that the dark energy density today is 1 - OmegaM - OmegaR - OmegaK. If the field oscillates around a minimum many times per e-fold it is replaced by a w = 0 fluid. The code stops with a message if no V0 works (the field rolls or oscillates too early). w(a) is in the column `w_phi` of `cosmology_<name>.txt`.

* EDE: axion-like early dark energy, V = m^2 f^2 (1 - cos(phi/f))^n, plus a cosmological constant. Parametrized by the peak fraction `cosmology_ede_fEDE` of the total density, the redshift of the peak `cosmology_ede_zc`, the initial angle `cosmology_ede_theta_ini` and `cosmology_ede_n` (m and f are found by shooting; takes ~10-20 s). Only the background is modified (the EDE perturbations are neglected), so start the simulation well after zc with initial conditions from a Boltzmann code that includes EDE.

* Tabulated: the background from a file (`cosmology_tabulated_filename`): `cosmology_tabulated_format = "H"` (columns z, H/H0, used as given), `"CLASS"` (a CLASS `_background.dat` file; the columns z and H [1/Mpc] are used, e.g. any hi_class background) or `"w"` (columns a, w(a) of the dark energy). Matter and radiation are those of the parameter file (they should agree with the table: the largest dark energy fraction above z = 10 is printed as a check). `input/class_background_w0wa.dat` is an example (CLASS, w0 = -0.9, wa = 0.1, the cosmology of the regression tests).

* DarkScattering: elastic dark matter - dark energy scattering (Simpson 2010; Baldi & Simpson 2015), the
  Einstein-Boltzmann solver's `dark_scattering`. The background is wCDM with a constant `cosmology_ds_w` (!= -1);
  only momentum is exchanged, so the scattering is a velocity friction of the dark matter, dv/dlna = -Gamma v + force
  with Gamma = (1 + w) c xi rho_DE / H per ln a (xi = `cosmology_ds_xi` = sigma_D / m_c in barn/GeV; the dark energy
  is smooth on N-body scales, so it does not move). Gamma > 0 (w > -1) slows the growth down, w < -1 speeds it up. It
  enters the growth equations (all LPT orders, the COLA frame and the IC velocities) and the kick and drift factors of
  all the time-stepping methods (`Cosmology::velocity_friction`, see `FML/NBody/TimeStepping.h`). The baryons do not
  scatter: with the CDM and baryons as one set of particles they feel f_c = OmegaCDM / OmegaCB of the drag (the
  momentum of the fluid with v_b = v_c). With `particle_baryons = true` the CDM particles feel the full drag and
  the baryon particles none, at the same steps with their own kick, drift and velocity factors (FastPM: from the
  two-fluid linear growth, so both species follow it exactly; COLA: the CDM+baryon frame plus the change of the LPT
  velocity that the frame's friction does not have; not with BullFrog), see [SPECIES.md](SPECIES.md). Neutrino
  particles are not supported with it yet. The IC from the internal solver (`ic_type_of_input = internal` or `tables`) include the scattering
  (`cosmology_ds_cs2` is the sound speed of the dark-energy perturbations there). The dark energy at rest is a
  large-scale approximation: the drag also moves the dark energy with the dark matter, so the solver has less
  suppression on large scales (w = -0.9, xi = 50: -4.9% instead of -6.2% for the CDM at k = 0.008 h/Mpc; the same for
  k > ~0.07 h/Mpc), see [SPECIES.md](SPECIES.md).

* CovariantDarkScattering: the covariantised dark Thomson-like scattering (Beltran Jimenez et
  al. 2020; N-body: arXiv:2510.12551), Q^mu = alpha_bar (u_DE^mu - u_c^mu), the internal
  solver's `covariant_dark_scattering`. As DarkScattering (wCDM background with `cosmology_ds_w`, momentum exchange
  only, `cosmology_ds_cs2` for the IC) with the dark-matter drag Gamma = alpha a^3 / (Omega_c0 E(a)) per ln a
  (`cosmology_cds_alpha` = 8 pi G alpha_bar / (3 H0^3); the conformal rate alpha H0 a^4 / Omega_c0), the same for
  one set of CDM+baryon particles (f_c Gamma) and with `particle_baryons = true`. The drag grows as a^3 and acts
  mostly at z < 1. Tested (alpha = 1, w = -0.98, 1e-4 times the power): the simulations (Quinn, FastPM, COLA +
  Quinn, COLA + FastPM) follow the growth of the N-body equations to 1e-3 (separate CDM and baryons; one set of
  particles overestimates the CDM+baryon suppression by 2.4e-3 with this strong drag), and the solver for k > ~0.05
  h/Mpc; on larger scales the solver's dark energy moves with the dark matter and the suppression is smaller (CDM:
  -6.2% instead of -9.1% at k = 0.008 h/Mpc), see [SPECIES.md](SPECIES.md).

* CoupledQuintessence: a scalar field phi (reduced Planck masses) with V ~ exp(-lambda phi) coupled to the CDM, whose
  particles have the mass m ~ exp(-beta phi) (Wetterich 1995; Amendola 2000): the background and the internal
  Einstein-Boltzmann solver's `coupled_quintessence` (`cosmology_cq_lambda`, `cosmology_cq_beta`; beta is Amendola's
  sqrt(2/3) beta). The CDM density is rho_c(a) a^3 = OmegaCDM m(a)/m(1) (`Cosmology::cdm_mass_factor`; the Poisson
  source `clustering_mass_factor`). Needs `gravity_model = CoupledQuintessence` (the force and the velocity term of the
  CDM, below).

* IDE: interacting dark energy (Valiviita, Majerotto & Maartens 2008, arXiv:0804.0232), the internal solver's `ide`: a dark-energy
  fluid with a constant `cosmology_ide_w` (!= -1) and the energy exchange Q = xi H rho_DE along the dark-matter
  4-velocity (`cosmology_ide_xi`; > 0: energy flows to the dark matter; the doom factor xi / (3 (1 + w)) must be <= 0;
  `cosmology_ide_cs2` for the IC). The CDM density follows rho_c' = -3 rho_c + xi rho_DE (`cdm_mass_factor`). As in
  the solver the exchange is perturbed by the local expansion of the dark matter, so on sub-horizon scales a CDM
  particle gains xi/3 rho_DE times the change of its volume and keeps its momentum: the particles carry their own
  masses (`Simulation::update_ide_masses`, before the force of every step; the volume from the number density of the
  particles, CIC on the particle grid). This dilution term (delta_c' = delta_n' (1 - q/3) - q delta_c, q = xi
  rho_DE / rho_c) is most of the effect: xi = -0.1, w = -0.9 gives 10% more growth of delta_cb from z = 49 to 0 than
  xi = 0, but only 2% if all the masses changed alike. The growth factors include the mass perturbation (the source
  (1 + m1) delta_n, `GravityModel::ide_mass_growth`), so the COLA frame follows the particles. Tested against the
  solver (xi = -0.1, w = -0.9, 1e-4 times the power, 32^3 particles in 1000 Mpc/h): the extra growth over xi = 0
  agrees to 6e-4 for k < 0.02 h/Mpc (CDM 7e-4, baryons 1e-4 with `particle_baryons = true`), and is too small by
  8e-3 for k < 0.05 h/Mpc (k_N/2 of the particle grid: the volume is smoothed on the scale of the particle grid;
  deconvolving that is unstable in the nonlinear regime); COLA (30 steps) as Quinn (200 steps). Q does not depend on
  rho_c, so in voids the dark matter can run out (xi < 0; the masses stop at zero, the CDM+baryon particles at their
  baryons): the nonlinear regime of the model is extreme (P(k) relative to xi = 0 at z = 0 is 5% below the linear
  ratio on large scales in a 256 Mpc/h box). Not with massive neutrinos; with `particle_baryons` only with
  `particle_baryon_relative_mode = displacement` and not with BullFrog. The particles need a mass (the default
  particle type has one; it is 120 bytes in 3D).

* Horndeski: a reduced Horndeski theory from the table of `scripts/horndeski_frontend.py`
  (`cosmology_horndeski_filename`; the internal solver's `horndeski_table`), see the gravity model Horndeski below.

* CubicGalileon: the cubic Galileon on its tracker (the internal solver's `cubic_galileon`; no parameter beyond those
  of ΛCDM, the dark energy is the Galileon with Omega = 1 - the other species). Use it with `gravity_model =
  CubicGalileon` (with `gravity_model = GR` it is the background alone).

For Quintessence, EDE and Tabulated the dark energy is smooth (no perturbations) and the gravity model is GR (they can be combined with the other gravity models, but those use the background only through H(a)). The backgrounds were checked against an independent solution (scipy): H(a) agrees to < 1e-4 and w(a) to < 1e-5 for a < 1, and the EDE peak fraction and redshift to < 1e-3.

## Gravity models

* The GravityModel base class solves the LPT growth equations up to 3LPT for a general effective Newton's constant
  G_eff(k, a)/G, which covers most of the common modified-gravity models; a model only provides those functions.
  A model that does not fit this pattern can override the growth solver.

* GR: the force from the Poisson equation.

* f(R): the Hu–Sawicki f(R) model, the standard test model for chameleon screening. Two parameters: the power n of
  the f(R) function and fR0, which sets the range of the fifth force of the scalar field. fR0 → 0 is GR.

* DGP: the DGP braneworld model (a cubic Galileon), the standard test model for Vainshtein screening. With a ΛCDM
  cosmology it is the normal branch, with the DGP cosmology the original self-accelerating model.

* Horndeski: a generic reduced Horndeski theory, S = ∫ √-g [K(φ, X) − G3(φ, X) □φ + G4(φ) R] (luminal
  gravitational waves), as Hi-COLA (Wright et al. 2023, arXiv:2209.01666). `scripts/horndeski_frontend.py` takes K,
  G3 and G4 (sympy expressions, a JSON file; examples in `input/horndeski`), derives the background equations from the
  minisuperspace action, solves them (from today with dφ/dlna from the Friedmann constraint, or from early times with
  a parameter shot for E(a = 1) = 1; use the latter for attractor solutions such as trackers) and writes the table
  `_gravity.txt`: H(a), the quasi-static force and the α's of Bellini & Sawicki (arXiv:1404.3713), all derived from
  the Lagrangian (and `_background.txt`, H(z) for `cosmology_model = Tabulated`). Use it with `cosmology_model =
  Horndeski` (`cosmology_horndeski_filename`; the background, and the IC of `ic_type_of_input = internal`: the
  Einstein-Boltzmann solver's `horndeski_table`, the full linear perturbations with these α's) and `gravity_model =
  Horndeski` (its table by default). The force (Kimura, Kobayashi & Yamamoto 2012, arXiv:1111.6749;
  `FML/Cosmology/Perturbations/QuasiStaticGravity_Horndeski.h`) is the Newtonian force rescaled by G_G4/G = 1/(2 G4)
  (the running Planck mass) plus a fifth force with the DGP equation of a time-dependent r_c(a) and beta(a): G_eff/G
  = 1/(2 G4) + 1/(3 beta), the 2LPT source and Vainshtein screening of DGP (G3X ≠ 0; without it the fifth force is
  linear): linear, the screening approximation (the density smoothed with `gravity_model_horndeski_smoothing_*`, a
  Gaussian by default as Hi-COLA) or the exact equation with DGP's multigrid solver
  (`gravity_model_horndeski_exact_solution`, needs r_c^2 > 0). The table must be made for the cosmology of the run
  (Ω_m and Ω_r are checked: give the frontend h, T_CMB and N_eff); flat, no massive neutrinos. Tested: the frontend
  gives the cubic Galileon's H(a), G_eff, r_c and α's and Brans-Dicke's G_eff = (4 + 2 omega) / ((3 + 2 omega) phi),
  Sigma = 1/phi and α's (`tests/check_horndeski_frontend.py`); in the solver the cubic Galileon from its Lagrangian
  reproduces `cubic_galileon` (growth 1e-6, P(k) 2e-5, C_ℓ^TT 3e-6) and Brans-Dicke's full solution has the
  quasi-static G_eff of the table on small scales to 1e-6 (`test_horndeski_table`); in the simulations the cubic
  Galileon from its Lagrangian gives the P(k) of `CubicGalileon` to 1e-5 (the IC from the internal solver alike; linear,
  screening approximation, multigrid). Unlike Hi-COLA the 2LPT growth has the theory's second-order source and the
  IC are those of the model.

  K-mouflage (`input/horndeski/kmouflage.json`; Babichev, Deffayet & Ziour 2009, Brax & Valageas 2014) is run as in
  Hi-COLA (Sen Gupta, Fiorini & Baker 2024, arXiv:2407.00855): in the Jordan frame, where the Einstein-frame action
  R/2 + λ² K(X_E) (K = -1 + X_E + K0 X_E^2, matter on A² g_E, A = exp(β φ)) is the reduced Horndeski theory G4 =
  1/(2 A²), K_J = λ² K(A² X / λ²) / A⁴ - 6 β² X / A² (written in the field 1/A², Brans-Dicke-like). The Newtonian
  force is rescaled by A² and the fifth force, G_eff/G = A² (1 + 2 β² / K_X), is linear: G3X = 0, so the DGP form has
  no Vainshtein term and the kinetic (K-mouflage) screening is not modelled, as in Hi-COLA. The example is n = 2, K0 =
  1, β = 0.2 (Hi-COLA's main model; λ shot for H0, the field started on its radiation-era attractor). Note that the
  table fixes the Jordan-frame Ω_m (the run's), not the Einstein-frame one of Hi-COLA and MG-GLAM. Tested: H(a), A²
  and G_eff of the table agree with an independent Einstein-frame integration mapped to the Jordan frame to 2e-7
  (`check_horndeski_frontend.py`); the solver's full perturbations have the table's G_eff on small scales to 1e-6
  (`test_horndeski_table`); the simulation's growth agrees with the solver's transfer functions to 4e-5
  (k = 0.1 h/Mpc). Regression case `horndeski_kmouflage_cola`.

* JBD: Jordan–Brans–Dicke. The gravity model of the cosmology with the same name (and it needs that cosmology).

* CubicGalileon: the force of the cubic Galileon (needs `cosmology_model = CubicGalileon`). Its quasi-static field
  equation is that of the DGP brane-bending mode with a time-dependent cross-over scale, (r_c H0/c)^2 = E^2 / (2
  Omega_phi(a) Omega_phi0), and beta = (1 - 2 dlnH/dlna - Omega_phi(a)) / (3 Omega_phi(a))
  (`FML/Cosmology/Perturbations/QuasiStaticGravity_CubicGalileon.h`): G_eff/G = 1 + 1/(3 beta) (1.03 at z = 1, 1.77
  today; it agrees with the Einstein-Boltzmann solver to 0.1%), the 2LPT source and the Vainshtein screening of DGP
  with these r_c(a) and beta(a) (approximate screening: the spherical top-hat factor of the local density, also in
  underdense regions, with the density smoothed as for DGP: `gravity_model_cubic_galileon_smoothing_filter` and
  `_smoothing_scale_over_boxsize`; exact: `gravity_model_cubic_galileon_exact_solution`, DGP's multigrid solver). In
  deep voids
  (delta < -0.6 today) the equation has no real solution: the screening factor and the multigrid solver use the
  limit there (a zero discriminant; the number of such cells is printed), as Barreira et al. 2013 (arXiv:1306.3219),
  who found the same for a > ~0.8. The multigrid solver then stalls at a residual of ~1e-4 (it stops after
  1000 V-cycles, slow). Tested: linear growth from z = 49 to 0 over GR on the same background +2.03% (linear,
  screening and multigrid alike) vs +2.07% from the quasi-static growth equation; at normal amplitude (256 Mpc/h, 64^3
  particles, COLA 30 steps) P(k)/P_GR at z = 0 is 1.04-1.21 for k = 0.03-1.5 h/Mpc without screening, 1.04-1.065
  with the exact equation and 1.03-1.06 with the screening approximation (without
  `gravity_model_screening_enforce_largescale_linear`, which restores the linear 1.04 on large scales); Barreira et
  al. found <~8% more power in the nonlinear regime.

* Symmetron: a symmetron with V = -(φ/φ*)^N/N + (φ/φ*)^M/M (symmetry breaking at `gravity_model_symmetron_assb`).

* Dilaton: the environment-dependent dilaton of Brax, Davis, Li, Winther & Zhao 2012 (arXiv:1206.3568), with the
  coupling β(a) and the mass m(a) = m0 a^-r of the generalised model.

* mbeta: the general (m(a), β(a)) parametrisation of chameleon-like models, β(a) = β0 a^n and m(a) = m0 H0 a^m.

  The linear theory of Symmetron, Dilaton and mbeta (and f(R)) is also in the Einstein-Boltzmann solver: the models
  `symmetron`, `dilaton`, `mbeta` (`fr`) of `cmb`, the ΛCDM background with the same quasi-static μ(k, a)
  (`FML/Cosmology/Perturbations/PerturbationModel_QuasiStatic.h`), for the CMB and the linear P(k). Its scale-dependent
  growth agrees with the quasi-static growth equation of the force here to 2e-4 (`test_quasistatic_models`). The IC of
  `ic_type_of_input = internal` follow `cosmology_model` (ΛCDM for these), as before.

* CoupledQuintessence: the force and the velocity term of the CDM of the cosmology with the same name (and it needs
  that cosmology, as JBD; it takes beta and phi(a) from the background). In the N-body equations (the sub-horizon limit
  of those of the Einstein-Boltzmann solver) the CDM feels G (1 + 2 beta^2) from the CDM (the fifth force of the light
  field, unscreened and scale independent on N-body scales) and G from the baryons, and has the velocity term
  +beta phi' v (`velocity_friction_cdm` = -beta dphi/dlna, through the velocity-friction machinery of DarkScattering:
  an acceleration for beta phi' > 0); the baryons feel gravity only. With the CDM and baryons as one set of particles
  they are a momentum-weighted fluid: G_eff/G = 1 + 2 beta^2 f_c(a)^2 (`GeffOverG`) and the friction f_c(a)
  (-beta phi') with f_c(a) = rho_c/rho_cb, in the growth equations (all LPT orders, COLA, IC) and the force. With
  `particle_baryons = true` the PM force is gravity alone (`GeffOverG_gravity`) and the CDM gets its fifth force from
  the CDM density in a second kick (`cdm_fifth_force_coupling` = 2 beta^2) and its own friction (see
  [SPECIES.md](SPECIES.md)). Not with massive neutrinos. Tested against the solver (lambda = 0.5, beta = 0.15: 13% more
  growth from z = 49 to 0 than uncoupled quintessence): the one-fluid growth of the code agrees with it to 6e-4 for
  k >= 0.02 h/Mpc, and the CDM+baryon growth of the simulations (Quinn, FastPM, COLA + Quinn, COLA + FastPM) to
  4-7e-4 at k = 0.02-0.03 h/Mpc (4e-3 in the largest-scale bin and 1-3e-3 at k = 0.05-0.09, the resolution). Halo masses are those at the output (times `cb_mass_factor(a)`); the masses in the snapshot
  files are those of today's density parameters.

* mu: Phenomenological modified gravity, the Poisson equation with G_eff/G = mu(a,k) and no screening (as in linear theory, also on small scales): `gravity_model_mu_parametrization = "DE"` with mu = 1 + mu0 Omega_DE(a)/Omega_DE(a=1) (the "late-time" parametrisation used by Planck and DES), `"BZ"` with mu = (1 + beta1 lambda1^2 k^2 a^s) / (1 + lambda1^2 k^2 a^s) (Bertschinger & Zukin 2008; k in h/Mpc, lambda1 in Mpc/h), or `"table"` with mu(a) or mu(a,k) from a file with rows `a mu` or `a k mu` on a regular grid (e.g. computed with hi_class; splined in log a and log k, the closest value outside the table; example: `input/mu_table_BZ_example.txt`). The growth factors (1LPT-3LPT) are solved with mu(a,k), so a k-dependent mu gives scale-dependent growth (use scale-dependent COLA, or COLA with the large-scale growth and the rest from the PM part). Only the potential that moves the particles is modified (Sigma, the lensing potential, is not used: the lightcone potential maps are GR). Test (`tests/check_mu_gravity.py`: 64^3 particles, 256 Mpc/h, 2x grid, linear amplitude, IC from GR growth at z = 19): P_mu/P_GR at z = 0 agrees with (D_mu/D_GR)^2 from an independent solution of the growth equation to 0.05% for k <= 0.1 h/Mpc (mu0 = 0.5: P_mu/P_GR = 1.131; BZ with beta1 = 4/3, lambda1 = 10 Mpc/h, s = 4: 1.002-1.18 for k = 0.03-0.6) and to 0.2-1.2% up to k = 0.6 h/Mpc (the force resolution of the PM grid), with and without scale-dependent COLA; the tables reproduce the parametrisations to 0.001%. At normal amplitude COLA (30 steps, scale-dependent for BZ) gives the same response P_mu/P_GR as 200 Quinn steps to 0.02% for k < 0.6 h/Mpc.

The theory of f(R), DGP, the cubic Galileon, JBD, the symmetron, (m(a), beta(a)) and mu (the DE parametrisation) - G_eff(k,a)/G, the f(R) scalaron mass and f_R(a), the DGP beta(a) and its second-order (2LPT) source, and the screening (its type, potential or density, its threshold and the screening factor) - is not defined here but in `FML/Cosmology/Perturbations/QuasiStaticGravity_*.h`, on the run's background, and is the same as in FML's Einstein-Boltzmann solver and the hybrid bias one-loop kernels. Applying the screening to the density field and the multigrid solvers are here (the f(R) solver takes the background curvature R(a) of the theory). The growth factors are solved by `FML/Cosmology/Growth/LPTGrowthFactors.h`.

## Screening

For the models with screening (f(R), DGP, the cubic Galileon, generic Horndeski, the symmetron, the dilaton and (m(a), β(a))) there are four ways to
include the nonlinear evolution of the scalar field:

* Linear: simulate with the linearised field equations. Fast, and agrees with linear theory on large scales.

* Approximate screening (Winther & Ferreira 2014, arXiv:1403.6492; for Vainshtein screening in COLA see also Brando,
  Koyama & Winther 2023, arXiv:2303.09549). Almost as fast and quite accurate (it can be tuned to be more
  accurate).

* Approximate screening together with the linear field equation, so that the linear evolution on large scales is
  exact. This is usually the best choice.

* Exact: solve the full field equation with a multigrid solver, as high-resolution codes such as ECOSMOG, ISIS and
  MG-GADGET do. This is slow and mostly useful for testing; the multigrid solver is easy to extend to other models.
  For f(R) the solver works with u = sqrt(|f_R|) by default (`gravity_model_fofr_multigrid_method = "sqrt"`), which
  converges for any f_R0, n and redshift; the original ln|f_R| variable (`"log"`) can give NaN for small f_R0 or at
  high redshift (e.g. |f_R0| = 1e-6 in a 64^3 run).
