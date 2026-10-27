#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Models/StandardSpecies.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace FML::COSMOLOGY {

using namespace Constants;

BackgroundCosmology::BackgroundCosmology(const CosmologyParameters& params_in)
  : BackgroundCosmology(params_in, Models::build(params_in)) {}

BackgroundCosmology::BackgroundCosmology(const CosmologyParameters& params_in, Model model_in)
  : params(params_in), model(std::move(model_in)), H0(params_in.H0()), OmegaK0(params_in.OmegaK),
    K(-params_in.OmegaK * params_in.H0() * params_in.H0() / (Constants::c * Constants::c)) {
  params.validate();
  if (!model.gravity) throw std::invalid_argument("BackgroundCosmology: the model has no gravity");
  by_role.resize(int(Species::Role::Other) + 1);
  for (const auto& s : model.species) {
    if (!s) throw std::invalid_argument("BackgroundCosmology: null species in model " + model.name);
    by_role[int(s->role())].push_back(s.get());
    if (s->role() == Species::Role::MassiveNeutrinos) {
      if (ncdm) throw std::invalid_argument("BackgroundCosmology: only one massive neutrino species is supported");
      ncdm = dynamic_cast<const MassiveNeutrinoSpecies*>(s.get());
      if (!ncdm) throw std::invalid_argument("BackgroundCosmology: massive neutrinos must be a MassiveNeutrinoSpecies");
    }
  }
  model.gravity->init_background(*this);
}

// =============================================================================
// Species sums
// =============================================================================

double BackgroundCosmology::rho_species(double x) const {
  double r = 0.0;
  for (const auto& s : model.species) r += s->rho(x);
  return r;
}
double BackgroundCosmology::drho_species(double x) const {
  double r = 0.0;
  for (const auto& s : model.species) r += s->drhodx(x);
  return r;
}
double BackgroundCosmology::ddrho_species(double x) const {
  double r = 0.0;
  for (const auto& s : model.species) r += s->ddrhoddx(x);
  return r;
}
double BackgroundCosmology::rho_of(Species::Role role, double x) const {
  double r = 0.0;
  for (const Species* s : by_role[int(role)]) r += s->rho(x);
  return r;
}
double BackgroundCosmology::drho_of(Species::Role role, double x) const {
  double r = 0.0;
  for (const Species* s : by_role[int(role)]) r += s->drhodx(x);
  return r;
}
double BackgroundCosmology::ddrho_of(Species::Role role, double x) const {
  double r = 0.0;
  for (const Species* s : by_role[int(role)]) r += s->ddrhoddx(x);
  return r;
}
double BackgroundCosmology::p_of(Species::Role role, double x) const {
  double r = 0.0;
  for (const Species* s : by_role[int(role)]) r += s->p(x);
  return r;
}

double BackgroundCosmology::pNcdm(double x) const        { return ncdm ? ncdm->p(x) : 0.0; }
double BackgroundCosmology::dpNcdmdx(double x) const     { return ncdm ? ncdm->dpdx(x) : 0.0; }
double BackgroundCosmology::pseudo_pNcdm(double x) const { return ncdm ? ncdm->pseudo_p(x) : 0.0; }
double BackgroundCosmology::M_ncdm_of_x(double x) const  { return ncdm ? ncdm->M_of_x(x) : 0.0; }
double BackgroundCosmology::ncdm_prefactor() const       { return ncdm ? ncdm->prefactor() : 0.0; }

double BackgroundCosmology::E2_of_x(double x) const      { return model.gravity->E2(x, *this); }
double BackgroundCosmology::dE2dx_of_x(double x) const   { return model.gravity->dE2dx(x, *this); }
double BackgroundCosmology::ddE2ddx_of_x(double x) const { return model.gravity->ddE2ddx(x, *this); }

// =============================================================================
// Expansion rate. With Hp = H0 e^x sqrt(E2) and L = E2'/E2:
//   Hp'  = Hp (1 + L/2)
//   Hp'' = Hp [ (1 + L/2)^2 + (E2''/E2 - L^2)/2 ]
// =============================================================================

double BackgroundCosmology::H_of_x(double x) const {
  return H0 * std::sqrt(E2_of_x(x));
}

double BackgroundCosmology::Hp_of_x(double x) const {
  return std::exp(x) * H_of_x(x);
}

double BackgroundCosmology::dHpdx_of_x(double x) const {
  const double L = dE2dx_of_x(x) / E2_of_x(x);
  return Hp_of_x(x) * (1.0 + 0.5 * L);
}

double BackgroundCosmology::ddHpddx_of_x(double x) const {
  const double E2 = E2_of_x(x);
  const double L  = dE2dx_of_x(x) / E2;
  const double M  = ddE2ddx_of_x(x) / E2;
  return Hp_of_x(x) * ((1.0 + 0.5 * L) * (1.0 + 0.5 * L) + 0.5 * (M - L * L));
}

double BackgroundCosmology::OmegaB_of_x(double x)     const { return rhoB(x) / E2_of_x(x); }
double BackgroundCosmology::OmegaCDM_of_x(double x)   const { return rhoCDM(x) / E2_of_x(x); }
double BackgroundCosmology::OmegaGamma_of_x(double x) const { return rhoGamma(x) / E2_of_x(x); }
double BackgroundCosmology::OmegaNu_of_x(double x)    const { return rhoNu(x) / E2_of_x(x); }
double BackgroundCosmology::OmegaNcdm_of_x(double x)  const { return rhoNcdm(x) / E2_of_x(x); }
double BackgroundCosmology::OmegaK_of_x(double x)     const { return rhoK(x) / E2_of_x(x); }
double BackgroundCosmology::OmegaDE_of_x(double x)    const { return rhoDE(x) / E2_of_x(x); }

double BackgroundCosmology::TCMB_of_x(double x) const {
  return params.TCMB * std::exp(-x);
}

// =============================================================================
// Initial conditions: exact matter + radiation solution
//   eta(a) = 2c a / (H0 [sqrt(OmR + OmM a) + sqrt(OmR)])
//   t(a)   = 2 OmR^{3/2} / (3 H0 OmM^2) f(y),  y = OmM a / OmR,
//            f(y) = (y - 2) sqrt(1 + y) + 2  (series for small y to avoid cancellation)
// =============================================================================

double BackgroundCosmology::eta_initial(double x) const {
  const double a = std::exp(x);
  // Massive neutrinos are relativistic at x_start and counted as radiation
  const double OmR = rhoGamma(0.0) + rhoNu(0.0) + rhoNcdm(x) * a * a * a * a, OmM = rhoB(0.0) + rhoCDM(0.0);
  return 2.0 * c * a / (H0 * (std::sqrt(OmR + OmM * a) + std::sqrt(OmR)));
}

double BackgroundCosmology::t_initial(double x) const {
  const double a = std::exp(x);
  const double OmR = rhoGamma(0.0) + rhoNu(0.0) + rhoNcdm(x) * a * a * a * a, OmM = rhoB(0.0) + rhoCDM(0.0);
  if (OmR <= 0.0) return 2.0 / (3.0 * H0 * std::sqrt(OmM)) * std::pow(a, 1.5);  // pure matter
  const double y = OmM * a / OmR;
  const double f = (y < 1e-3)
    ? y * y * (0.75 + y * (-0.25 + y * (9.0 / 64.0 - y * 3.0 / 32.0)))
    : (y - 2.0) * std::sqrt(1.0 + y) + 2.0;
  return 2.0 * std::pow(OmR, 1.5) / (3.0 * H0 * OmM * OmM) * f;
}

// =============================================================================
// Solve for eta(x) and t(x)
// =============================================================================

void BackgroundCosmology::solve(double x_start_in, double x_end_in, int n_points) {
  Utils::Timer timer("BackgroundCosmology::solve");
  if (!(x_end_in > x_start_in)) throw std::invalid_argument("BackgroundCosmology::solve: need x_end > x_start");
  x_start = x_start_in;
  x_end   = x_end_in;

  const auto x_array = Utils::linspace(x_start, x_end, n_points);

  // y = (eta, t):  deta/dx = c / Hp,  dt/dx = 1 / H
  ODEFunction rhs = [&](double x, const double*, double* dydx) {
    const double H = H_of_x(x);
    dydx[0] = c / (std::exp(x) * H);
    dydx[1] = 1.0 / H;
    return GSL_SUCCESS;
  };

  ODESolver ode(1e-3, 0.0, 1e-12);
  ode.solve(rhs, x_array, {eta_initial(x_start), t_initial(x_start)}, gsl_odeiv2_step_rk8pd);

  eta_spline.create(x_array, ode.get_data_by_component(0), "eta_of_x");
  t_spline.create(x_array, ode.get_data_by_component(1), "t_of_x");
  solved = true;

  timer.print();
}

void BackgroundCosmology::require_solved(const char* func) const {
  if (!solved) throw std::runtime_error(std::string("BackgroundCosmology::") + func + ": call solve() first");
}

double BackgroundCosmology::eta_of_x(double x) const {
  require_solved("eta_of_x");
  return eta_spline(x);
}

double BackgroundCosmology::detadx_of_x(double x) const {
  return c / Hp_of_x(x);
}

double BackgroundCosmology::t_of_x(double x) const {
  require_solved("t_of_x");
  return t_spline(x);
}

double BackgroundCosmology::eta0() const { return eta_of_x(0.0); }
double BackgroundCosmology::t0()   const { return t_of_x(0.0); }

// =============================================================================
// Distances
// =============================================================================

double BackgroundCosmology::comoving_distance_of_x(double x) const {
  return eta0() - eta_of_x(x);
}

// sinK(chi) = chi * S(K chi^2) with S(u) = sin(sqrt u)/sqrt u (u > 0) or sinh(sqrt -u)/sqrt -u (u < 0).
// cotK(chi) = (1/chi) * C(K chi^2) with C(u) = sqrt(u) cot(sqrt u) or sqrt(-u) coth(sqrt -u).
// Series for small |u|: S = 1 - u/6 + u^2/120 - u^3/5040,  C = 1 - u/3 - u^2/45 - 2u^3/945.
double BackgroundCosmology::sinK(double chi) const {
  const double u = K * chi * chi;
  if (std::fabs(u) < 1e-4) return chi * (1.0 - u / 6.0 * (1.0 - u / 20.0 * (1.0 - u / 42.0)));
  const double s = std::sqrt(std::fabs(u));
  return chi * (u > 0.0 ? std::sin(s) : std::sinh(s)) / s;
}

double BackgroundCosmology::cotK(double chi) const {
  const double u = K * chi * chi;
  if (std::fabs(u) < 1e-4) return (1.0 - u / 3.0 - u * u / 45.0 - 2.0 * u * u * u / 945.0) / chi;
  const double s = std::sqrt(std::fabs(u));
  return (u > 0.0 ? s / std::tan(s) : s / std::tanh(s)) / chi;
}

double BackgroundCosmology::transverse_distance_of_x(double x) const {
  return sinK(comoving_distance_of_x(x));
}

double BackgroundCosmology::angular_distance_of_x(double x) const {
  return std::exp(x) * transverse_distance_of_x(x);
}

double BackgroundCosmology::luminosity_distance_of_x(double x) const {
  return transverse_distance_of_x(x) * std::exp(-x);
}

// =============================================================================
// Characteristic epochs
// =============================================================================

namespace {
  // Root of f on [a,b], or NaN if there is no sign change
  double root_or_nan(const std::function<double(double)>& f, double a, double b) {
    if (f(a) * f(b) > 0.0) return std::numeric_limits<double>::quiet_NaN();
    return Utils::find_root(f, a, b);
  }
}

double BackgroundCosmology::x_matter_radiation_equality() const {
  return root_or_nan([&](double x) {
    return std::log((rhoB(x) + rhoCDM(x)) / (rhoGamma(x) + rhoNu(x)));
  }, x_start, x_end);
}

double BackgroundCosmology::x_matter_DE_equality() const {
  return root_or_nan([&](double x) {
    return (rhoB(x) + rhoCDM(x)) - rhoDE(x);
  }, std::max(x_start, -10.0), x_end);
}

double BackgroundCosmology::x_acceleration_onset() const {
  return root_or_nan([&](double x) { return dHpdx_of_x(x) / Hp_of_x(x); },
                     std::max(x_start, -10.0), x_end);
}

// =============================================================================
// Output
// =============================================================================

void BackgroundCosmology::info() const {
  const auto f = std::cout.flags();
  std::cout << std::setprecision(6);
  std::cout << "\nBackgroundCosmology: model " << model.name << ", gravity " << model.gravity->info() << "\n";
  std::cout << "  Species: ";
  for (size_t i = 0; i < model.species.size(); i++) std::cout << (i ? "\n           " : "") << model.species[i]->info();
  std::cout << "\n";
  std::cout << "  E^2(x=0) (should be 1)   = " << E2_of_x(0.0) << "\n";
  std::cout << "  OmegaB, CDM, DE (today)  = " << OmegaB_of_x(0.0) << ", " << OmegaCDM_of_x(0.0)
            << ", " << OmegaDE_of_x(0.0) << "\n";
  if (has_massive_nu())
    std::cout << "  OmegaNcdm (today)        = " << OmegaNcdm_of_x(0.0) << "\n";
  std::cout << "  OmegaGamma, Nu, K        = " << OmegaGamma_of_x(0.0) << ", " << OmegaNu_of_x(0.0)
            << ", " << OmegaK_of_x(0.0) << "\n";
  if (K != 0.0) {
    std::cout << "  Curvature radius 1/sqrt|K| = " << 1.0 / std::sqrt(std::fabs(K)) / Mpc << " Mpc ("
              << (K > 0.0 ? "closed" : "open") << "), sqrt|K| eta0 = " << std::sqrt(std::fabs(K)) * eta0() << "\n";
  }

  auto print_epoch = [&](const char* name, double x) {
    std::cout << "  " << std::left << std::setw(25) << name << "x = " << std::setw(11) << x
              << "  z = " << std::setw(11) << std::exp(-x) - 1.0;
    if (solved && std::isfinite(x))
      std::cout << "  t = " << t_of_x(x) / yr << " yr";
    std::cout << "\n";
  };
  print_epoch("Matter-radiation eq.:", x_matter_radiation_equality());
  print_epoch("Matter-DE equality:",   x_matter_DE_equality());
  print_epoch("Acceleration onset:",   x_acceleration_onset());

  if (solved) {
    std::cout << "  Age t0                   = " << t0() / Gyr << " Gyr\n";
    std::cout << "  Conformal time eta0      = " << eta0() / Mpc << " Mpc  (eta0/c = "
              << eta0() / c / Gyr << " Gyr)\n";
  }
  std::cout << "\n";
  std::cout.flags(f);
}

void BackgroundCosmology::output(const std::string& filename, double x_min, double x_max,
                                 int n_points) const {
  require_solved("output");
  if (x_min == 0.0 && x_max == 0.0) { x_min = x_start; x_max = x_end; }

  std::ofstream fp(filename);
  if (!fp) throw std::runtime_error("BackgroundCosmology::output: cannot open " + filename);

  const double H0unit = H0_over_h;  // 100 km/s/Mpc
  fp << "# Background cosmology. Units: eta, chi, d_A, d_L in Mpc; t in Gyr; H, Hp, dHp, ddHp in (100 km/s/Mpc)\n"
     << "# x  eta  t  H  Hp  dHpdx  ddHpddx  OmegaB  OmegaCDM  OmegaGamma  OmegaNu  OmegaK  OmegaDE"
        "  chi  d_A  d_L  OmegaNcdm  w_ncdm\n";
  fp << std::scientific << std::setprecision(10);
  for (double x : Utils::linspace(x_min, x_max, n_points)) {
    fp << x << " "
       << eta_of_x(x) / Mpc << " "
       << t_of_x(x) / Gyr << " "
       << H_of_x(x) / H0unit << " "
       << Hp_of_x(x) / H0unit << " "
       << dHpdx_of_x(x) / H0unit << " "
       << ddHpddx_of_x(x) / H0unit << " "
       << OmegaB_of_x(x) << " "
       << OmegaCDM_of_x(x) << " "
       << OmegaGamma_of_x(x) << " "
       << OmegaNu_of_x(x) << " "
       << OmegaK_of_x(x) << " "
       << OmegaDE_of_x(x) << " "
       << comoving_distance_of_x(x) / Mpc << " "
       << angular_distance_of_x(x) / Mpc << " "
       << luminosity_distance_of_x(x) / Mpc << " "
       << OmegaNcdm_of_x(x) << " " << (has_massive_nu() ? pNcdm(x) / rhoNcdm(x) : 0.0) << "\n";
  }
}

} // namespace FML::COSMOLOGY
