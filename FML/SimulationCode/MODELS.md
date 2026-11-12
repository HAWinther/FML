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
  (`cosmology_ds_cs2` is the sound speed of the dark-energy perturbations there).

* CoupledQuintessence: a scalar field phi (reduced Planck masses) with V ~ exp(-lambda phi) coupled to the CDM, whose
  particles have the mass m ~ exp(-beta phi) (Wetterich 1995; Amendola 2000): the background and the internal
  Einstein-Boltzmann solver's `coupled_quintessence` (`cosmology_cq_lambda`, `cosmology_cq_beta`; beta is Amendola's
  sqrt(2/3) beta). The CDM density is rho_c(a) a^3 = OmegaCDM m(a)/m(1) (`Cosmology::cdm_mass_factor`; the Poisson
  source `clustering_mass_factor`). Needs `gravity_model = CoupledQuintessence` (the force and the velocity term of the
  CDM, below).

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

* JBD: Jordan–Brans–Dicke. The gravity model of the cosmology with the same name (and it needs that cosmology).

* Symmetron: a symmetron with V = -(φ/φ*)^N/N + (φ/φ*)^M/M (symmetry breaking at `gravity_model_symmetron_assb`).

* Dilaton: the environment-dependent dilaton of Brax, Davis, Li, Winther & Zhao 2012 (arXiv:1206.3568), with the
  coupling β(a) and the mass m(a) = m0 a^-r of the generalised model.

* mbeta: the general (m(a), β(a)) parametrisation of chameleon-like models, β(a) = β0 a^n and m(a) = m0 H0 a^m.

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
  growth from z = 49 to 0 than uncoupled quintessence): the one-fluid growth agrees with it to 1e-4, and the CDM+baryon
  growth of the simulations (Quinn, FastPM, COLA + Quinn, COLA + FastPM) to 7-8e-4 (the resolution: 1.6e-4 in the
  largest-scale bin). Halo masses are those at the output (times `cb_mass_factor(a)`); the masses in the snapshot
  files are those of today's density parameters.

* mu: Phenomenological modified gravity, the Poisson equation with G_eff/G = mu(a,k) and no screening (as in linear theory, also on small scales): `gravity_model_mu_parametrization = "DE"` with mu = 1 + mu0 Omega_DE(a)/Omega_DE(a=1) (the "late-time" parametrisation used by Planck and DES), `"BZ"` with mu = (1 + beta1 lambda1^2 k^2 a^s) / (1 + lambda1^2 k^2 a^s) (Bertschinger & Zukin 2008; k in h/Mpc, lambda1 in Mpc/h), or `"table"` with mu(a) or mu(a,k) from a file with rows `a mu` or `a k mu` on a regular grid (e.g. computed with hi_class; splined in log a and log k, the closest value outside the table; example: `input/mu_table_BZ_example.txt`). The growth factors (1LPT-3LPT) are solved with mu(a,k), so a k-dependent mu gives scale-dependent growth (use scale-dependent COLA, or COLA with the large-scale growth and the rest from the PM part). Only the potential that moves the particles is modified (Sigma, the lensing potential, is not used: the lightcone potential maps are GR). Test (`tests/check_mu_gravity.py`: 64^3 particles, 256 Mpc/h, 2x grid, linear amplitude, IC from GR growth at z = 19): P_mu/P_GR at z = 0 agrees with (D_mu/D_GR)^2 from an independent solution of the growth equation to 0.05% for k <= 0.1 h/Mpc (mu0 = 0.5: P_mu/P_GR = 1.131; BZ with beta1 = 4/3, lambda1 = 10 Mpc/h, s = 4: 1.002-1.18 for k = 0.03-0.6) and to 0.2-1.2% up to k = 0.6 h/Mpc (the force resolution of the PM grid), with and without scale-dependent COLA; the tables reproduce the parametrisations to 0.001%. At normal amplitude COLA (30 steps, scale-dependent for BZ) gives the same response P_mu/P_GR as 200 Quinn steps to 0.02% for k < 0.6 h/Mpc.

The theory of f(R), DGP, JBD, the symmetron, (m(a), beta(a)) and mu (the DE parametrisation) - G_eff(k,a)/G, the f(R) scalaron mass and f_R(a), the DGP beta(a) and its second-order (2LPT) source, and the screening (its type, potential or density, its threshold and the screening factor) - is not defined here but in `FML/Cosmology/Perturbations/QuasiStaticGravity_*.h`, on the run's background, and is the same as in FML's Einstein-Boltzmann solver and the hybrid bias one-loop kernels. Applying the screening to the density field and the multigrid solvers are here (the f(R) solver takes the background curvature R(a) of the theory). The growth factors are solved by `FML/Cosmology/Growth/LPTGrowthFactors.h`.

## Screening

For the models with screening (f(R), DGP, the symmetron, the dilaton and (m(a), β(a))) there are four ways to
include the nonlinear evolution of the scalar field:

* Linear: simulate with the linearised field equations. Fast, and agrees with linear theory on large scales.

* Approximate screening (Winther & Ferreira 2014). Almost as fast and quite accurate (it can be tuned to be more
  accurate).

* Approximate screening together with the linear field equation, so that the linear evolution on large scales is
  exact. This is usually the best choice.

* Exact: solve the full field equation with a multigrid solver, as high-resolution codes such as ECOSMOG, ISIS and
  MG-GADGET do. This is slow and mostly useful for testing; the multigrid solver is easy to extend to other models.
  For f(R) the solver works with u = sqrt(|f_R|) by default (`gravity_model_fofr_multigrid_method = "sqrt"`), which
  converges for any f_R0, n and redshift; the original ln|f_R| variable (`"log"`) can give NaN for small f_R0 or at
  high redshift (e.g. |f_R0| = 1e-6 in a 64^3 run).
