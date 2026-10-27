// =============================================================================
// Tests of the dark scattering model (models/DarkScattering.h). Run with `make test`.
//
//   * Background: rho_DE ~ a^{-3(1+w)}, closure; the velocity-locking epoch Gamma_DE/Hp = 200
//   * xi = 0 (wCDM) and xi > 0: the Einstein constraints that are not evolved (Poisson, momentum) hold with
//     dark matter and dark energy; with scattering they hold only if the momentum exchange cancels
//   * Drag: the growth of the dark matter follows the quasi-static two-fluid growth equations with the
//     momentum transfer to the dark energy (dark-energy velocity from the run)
//   * Direction: for w > -1 the scattering suppresses the growth, for w < -1 it enhances it
// (The wCDM part, xi = 0, is compared with CLASS's fluid dark energy in python/compare_class_wcdm.py.)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Models/DarkScattering.h>

#include <cmath>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
  CosmologyParameters ds(double w, double xi) {
    CosmologyParameters p;
    p.model = "dark_scattering";
    p.model_params = {{"ds_w", w}, {"ds_xi", xi}};
    return p;
  }

  struct Run {
    BackgroundCosmology cosmo;
    RecombinationHistory rec;
    Perturbations pert;
    Run(const CosmologyParameters& p, const PerturbationOptions& o)
      : cosmo(p), rec((cosmo.solve(), cosmo)), pert(cosmo, rec, o) {
      rec.solve();
      pert.solve();
    }
  };

  const FluidDarkEnergy& fluid(const BackgroundCosmology& b) {
    for (const auto& s : b.species())
      if (auto f = dynamic_cast<const FluidDarkEnergy*>(s.get())) return *f;
    throw std::runtime_error("no dark energy fluid");
  }

  // Gamma_DE / Hp and Gamma_c / Hp at x
  void rates(const BackgroundCosmology& b, double x, double& gde, double& gc) {
    const auto& f = fluid(b);
    gde = std::exp(x) * c * f.xi * b.rhoCDM(x) * f.rho_crit0 / b.Hp_of_x(x);
    gc = (1.0 + f.w) * f.rho(x) / b.rhoCDM(x) * gde;
  }

  void constraints(const Run& r, double& eP, double& eM) {
    const auto& cs = r.cosmo;
    const auto& pt = r.pert;
    const auto& f = fluid(cs);
    eP = eM = 0.0;
    for (double k : pt.get_k_array())
      for (double x : {-2.0, -1.0, 0.0}) {
        const double H0a2 = std::pow(cs.get_H0() * std::exp(x), 2), Hp = cs.Hp_of_x(x), ckH = c * k / Hp;
        const double rc = cs.rhoCDM(x), rb = cs.rhoB(x), rg = cs.rhoGamma(x), rn = cs.rhoNu(x), rd = f.rho(x);
        const double drho = H0a2 * (rc * pt.get_delta_cdm(x, k) + rb * pt.get_delta_b(x, k) + 4.0 * rg * pt.get_Theta(0, x, k) +
                                    4.0 * rn * pt.get_Nu(0, x, k) + rd * pt.get("delta_de", x, k));
        const double flux = H0a2 * (-rc * pt.get_v_cdm(x, k) - rb * pt.get_v_b(x, k) + 4.0 * rg * pt.get_Theta(1, x, k) +
                                    4.0 * rn * pt.get_Nu(1, x, k) - (1.0 + f.w) * rd * pt.get("v_de", x, k));
        const double D = drho + 3.0 * flux / ckH;
        eP = std::max(eP, std::fabs(c * c * k * k * pt.get_Phi(x, k) / (1.5 * D) - 1.0));
        eM = std::max(eM, std::fabs((pt.get_dPhidx(x, k) - pt.get_Psi(x, k)) / (-1.5 * flux / (ckH * Hp * Hp)) - 1.0));
      }
  }

  // delta_c(0)/delta_c(x0) from the quasi-static two-fluid equations with drag,
  //   delta_c'' + (2 + dlnH/dx + g_c) delta_c' = (3/2)(rho_c delta_c + rho_b delta_b)/E^2 + g_c (ck/Hp) v_DE
  //   delta_b'' + (2 + dlnH/dx) delta_b'       = (3/2)(rho_c delta_c + rho_b delta_b)/E^2
  // started from the run's values at x0, with the run's dark-energy velocity
  double growth(const Run& r, double k, double x0) {
    const auto& b = r.cosmo;
    const double h = 1e-3;
    double dc = r.pert.get_delta_cdm(x0, k), db = r.pert.get_delta_b(x0, k);
    double dcx = (r.pert.get_delta_cdm(x0 + h, k) - r.pert.get_delta_cdm(x0 - h, k)) / (2 * h);
    double dbx = (r.pert.get_delta_b(x0 + h, k) - r.pert.get_delta_b(x0 - h, k)) / (2 * h);
    auto acc = [&](double x, double dc, double db, double dcx, double dbx, double& ac, double& ab) {
      const double E2 = b.E2_of_x(x), L = 0.5 * b.dE2dx_of_x(x) / E2;
      double gde, gc;
      rates(b, x, gde, gc);
      const double src = 1.5 * (b.rhoCDM(x) * dc + b.rhoB(x) * db) / E2;
      ac = -(2.0 + L + gc) * dcx + src + gc * c * k / b.Hp_of_x(x) * r.pert.get("v_de", x, k);
      ab = -(2.0 + L) * dbx + src;
    };
    const int n = 20000;
    const double hx = -x0 / n;
    for (int i = 0; i < n; i++) {
      const double x = x0 + i * hx;
      double k1c, k1b, k2c, k2b, k3c, k3b, k4c, k4b;
      acc(x, dc, db, dcx, dbx, k1c, k1b);
      acc(x + hx / 2, dc + hx / 2 * dcx, db + hx / 2 * dbx, dcx + hx / 2 * k1c, dbx + hx / 2 * k1b, k2c, k2b);
      acc(x + hx / 2, dc + hx / 2 * (dcx + hx / 2 * k1c), db + hx / 2 * (dbx + hx / 2 * k1b), dcx + hx / 2 * k2c,
          dbx + hx / 2 * k2b, k3c, k3b);
      acc(x + hx, dc + hx * (dcx + hx / 2 * k2c), db + hx * (dbx + hx / 2 * k2b), dcx + hx * k3c, dbx + hx * k3b, k4c, k4b);
      dc += hx * (dcx + hx / 6 * (k1c + k2c + k3c));
      db += hx * (dbx + hx / 6 * (k1b + k2b + k3b));
      dcx += hx / 6 * (k1c + 2 * k2c + 2 * k3c + k4c);
      dbx += hx / 6 * (k1b + 2 * k2b + 2 * k3b + k4b);
    }
    return dc / r.pert.get_delta_cdm(x0, k);
  }
}

int main() {
  const double w = -0.9, xi = 30.0;
  std::cout << "Background (w = " << w << ")\n";
  {
    BackgroundCosmology b(ds(w, xi));
    check_close("E(0) = 1", b.E2_of_x(0.0), 1.0, 1e-13);
    check_rel("rho_DE ~ a^{-3(1+w)}", b.rhoDE(-1.0) / b.rhoDE(0.0), std::exp(3.0 * (1.0 + w)), 1e-13);
    double gde, gc;
    rates(b, 0.0, gde, gc);
    std::cout << "  today: Gamma_DE/Hp = " << gde << ", Gamma_c/Hp = " << gc << "\n";
    check("Gamma_c << Gamma_DE today (rho_c >> (1 + w) rho_DE)", gc < gde);
    check("model needs w != -1", [] { try { BackgroundCosmology bb(ds(-1.0, 1.0)); } catch (...) { return true; } return false; }());
  }

  PerturbationOptions o;
  o.n_k = 5;
  o.k_min = 0.001 / Mpc;
  o.k_max = 0.2 / Mpc;
  std::cout << "Perturbations: Einstein constraints with dark matter and dark energy\n";
  Run r0(ds(w, 0.0), o), r(ds(w, xi), o);
  {
    double eP, eM;
    constraints(r0, eP, eM);
    check_close("xi = 0 (wCDM): Poisson constraint", eP, 0.0, 1e-6);
    check_close("xi = 0 (wCDM): momentum constraint", eM, 0.0, 1e-6);
    constraints(r, eP, eM);
    check_close("xi = 30 b/GeV: Poisson constraint", eP, 0.0, 1e-6);
    check_close("xi = 30 b/GeV: momentum constraint (exchange cancels)", eM, 0.0, 1e-6);
  }

  std::cout << "Drag on the dark matter\n";
  {
    const double k = r.pert.get_k_array().back(), x0 = -3.0;
    const double code = (r.pert.get_delta_cdm(0.0, k) / r.pert.get_delta_cdm(x0, k)) /
                        (r0.pert.get_delta_cdm(0.0, k) / r0.pert.get_delta_cdm(x0, k));
    const double ref = growth(r, k, x0) / growth(r0, k, x0);
    std::cout << "  growth from z = 19 to 0, xi = " << xi << " b/GeV relative to xi = 0: " << code
              << ", growth equations with drag: " << ref << "\n";
    check_close("growth with drag = two-fluid growth equations", code / ref, 1.0, 5e-4);
    check("w > -1: scattering suppresses the growth", code < 0.99);
    Run rp(ds(-1.1, xi), o), rp0(ds(-1.1, 0.0), o);
    const double codep = (rp.pert.get_delta_cdm(0.0, k) / rp.pert.get_delta_cdm(x0, k)) /
                         (rp0.pert.get_delta_cdm(0.0, k) / rp0.pert.get_delta_cdm(x0, k));
    std::cout << "  w = -1.1: " << codep << "\n";
    check("w < -1: scattering enhances the growth", codep > 1.01);
  }

  return Test::summary();
}
