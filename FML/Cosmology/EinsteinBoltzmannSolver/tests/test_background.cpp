// =============================================================================
// Tests for BackgroundCosmology (and a model with CPL dark energy) against analytic results
// and independent quadrature. Run with `make test`.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/Model.h>
#include <FML/Cosmology/Models/StandardSpecies.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <functional>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

// Independent E(a) = H/H0 for LCDM + curvature, written out directly from the parameters
static double E_reference(const CosmologyParameters& p, double a) {
  return std::sqrt(p.OmegaM() / (a * a * a) + p.OmegaR() / (a * a * a * a) + p.OmegaK / (a * a) +
                   p.OmegaLambda());
}

// eta(a) = int_0^a c da' / (a'^2 H(a')),   t(a) = int_0^a da' / (a' H(a'))
// Substituting a' = u^2 removes the integrable endpoint behaviour at a' -> 0.
static double eta_reference(const CosmologyParameters& p, double a) {
  return Utils::integrate([&](double u) {
    const double ap = u * u;
    return 2.0 * u * c / (ap * ap * p.H0() * E_reference(p, ap));
  }, 0.0, std::sqrt(a), 0.0, 1e-12);
}
static double t_reference(const CosmologyParameters& p, double a) {
  return Utils::integrate([&](double u) {
    const double ap = u * u;
    return 2.0 * u / (ap * p.H0() * E_reference(p, ap));
  }, 0.0, std::sqrt(a), 0.0, 1e-12);
}

// Check the analytic derivatives of Hp against finite differences of Hp itself.
// Errors are normalised by max(|derivative|, Hp).
static void check_Hp_derivatives(const BackgroundCosmology& cosmo, const std::string& label,
                                 double x_max = 4.0) {
  double max_err1 = 0.0, max_err2 = 0.0;
  const double h = 1e-4;
  for (double x = -18.0; x <= x_max; x += 0.37) {
    const double Hm = cosmo.Hp_of_x(x - h), H = cosmo.Hp_of_x(x), Hp = cosmo.Hp_of_x(x + h);
    const double d1 = (Hp - Hm) / (2.0 * h);
    const double d2 = (Hp - 2.0 * H + Hm) / (h * h);
    max_err1 = std::max(max_err1, std::fabs(cosmo.dHpdx_of_x(x) - d1) / std::max(std::fabs(d1), H));
    max_err2 = std::max(max_err2, std::fabs(cosmo.ddHpddx_of_x(x) - d2) / std::max(std::fabs(d2), H));
  }
  check_close(label + ": dHp/dx vs finite difference", max_err1, 0.0, 1e-7);
  check_close(label + ": d2Hp/dx2 vs finite difference", max_err2, 0.0, 1e-5);
}

static void check_closure(const BackgroundCosmology& cosmo, const std::string& label) {
  double max_err = 0.0;
  for (double x = -20.0; x <= 5.0; x += 0.5) {
    const double sum = cosmo.OmegaB_of_x(x) + cosmo.OmegaCDM_of_x(x) + cosmo.OmegaGamma_of_x(x) +
                       cosmo.OmegaNu_of_x(x) + cosmo.OmegaK_of_x(x) + cosmo.OmegaDE_of_x(x);
    max_err = std::max(max_err, std::fabs(sum - 1.0));
  }
  check_close(label + ": E^2(0) = 1", cosmo.E2_of_x(0.0), 1.0, 1e-14);
  check_close(label + ": sum of Omega_i(x) = 1 for all x", max_err, 0.0, 1e-14);
}

// -----------------------------------------------------------------------------

static void test_fiducial() {
  std::cout << "Fiducial flat LCDM\n";
  const CosmologyParameters p = CosmologyParameters::fiducial();
  BackgroundCosmology cosmo(p);
  cosmo.solve();

  check_closure(cosmo, "LCDM");
  check_Hp_derivatives(cosmo, "LCDM");

  // Radiation era: Hp ~ e^{-x}  =>  Hp'/Hp = -1, Hp''/Hp = 1, eta Hp / c = 1
  const double xr = -18.0;
  check_close("RD: dHp/dx / Hp = -1",     cosmo.dHpdx_of_x(xr) / cosmo.Hp_of_x(xr), -1.0, 1e-4);
  check_close("RD: d2Hp/dx2 / Hp = 1",    cosmo.ddHpddx_of_x(xr) / cosmo.Hp_of_x(xr), 1.0, 1e-4);
  check_close("RD: eta Hp / c = 1",       cosmo.eta_of_x(xr) * cosmo.Hp_of_x(xr) / c, 1.0, 1e-4);
  // Far future, Lambda domination: Hp ~ e^{x}
  check_close("DE era (x=5): dHp/dx / Hp = 1", cosmo.dHpdx_of_x(5.0) / cosmo.Hp_of_x(5.0), 1.0, 1e-5);

  // Spline consistency: d eta/dx from the spline vs c/Hp
  {
    double max_err = 0.0;
    for (double x = -19.0; x <= 4.0; x += 0.73) {
      const double h = 1e-4;
      const double d = (cosmo.eta_of_x(x + h) - cosmo.eta_of_x(x - h)) / (2.0 * h);
      max_err = std::max(max_err, std::fabs(d / cosmo.detadx_of_x(x) - 1.0));
    }
    check_close("spline d(eta)/dx vs c/Hp", max_err, 0.0, 1e-6);
  }

  // Independent quadrature for eta and t, both today and at early times (tests the ICs)
  check_rel("eta0 vs quadrature",              cosmo.eta0(), eta_reference(p, 1.0), 1e-8);
  check_rel("t0 vs quadrature",                cosmo.t0(),   t_reference(p, 1.0),   1e-8);
  check_rel("eta(x=-15) vs quadrature",        cosmo.eta_of_x(-15.0), eta_reference(p, std::exp(-15.0)), 1e-8);
  check_rel("t(x=-15) vs quadrature",          cosmo.t_of_x(-15.0),   t_reference(p, std::exp(-15.0)),   1e-8);
  check_rel("t(x=-7) (recombination) vs quadrature", cosmo.t_of_x(-7.0), t_reference(p, std::exp(-7.0)), 1e-8);

  // Age without radiation: t0 = 2/(3 H0 sqrt(OmL)) asinh(sqrt(OmL/OmM)). Radiation shifts it by ~1e-4.
  const double OmM = p.OmegaM(), OmL = p.OmegaLambda(), OmR = p.OmegaR();
  check_rel("t0 vs matter+Lambda formula (radiation ~1e-4)", cosmo.t0(),
            2.0 / (3.0 * p.H0() * std::sqrt(OmL)) * std::asinh(std::sqrt(OmL / OmM)), 5e-4);

  // Epochs
  check_close("matter-radiation equality x = ln(OmR/OmM)", cosmo.x_matter_radiation_equality(), std::log(OmR / OmM), 1e-10);
  check_close("matter-Lambda equality x = ln(OmM/OmL)/3",  cosmo.x_matter_DE_equality(), std::log(OmM / OmL) / 3.0, 1e-10);
  // Acceleration onset with radiation: OmM a^-3 + 2 OmR a^-4 = 2 OmL  ->  solve numerically here
  const double x_acc = Utils::find_root([&](double x) {
    return OmM * std::exp(-3 * x) + 2 * OmR * std::exp(-4 * x) - 2 * OmL;
  }, -3.0, 0.0);
  check_close("acceleration onset", cosmo.x_acceleration_onset(), x_acc, 1e-9);

  // Low-z luminosity distance: d_L = (c z/H0) [1 + (1 - q0) z/2 + O(z^2)],  q0 = OmM/2 + OmR - OmL
  const double z = 1e-3, q0 = 0.5 * OmM + OmR - OmL;
  check_rel("d_L at z = 0.001 vs Hubble-law expansion", cosmo.luminosity_distance_of_x(-std::log1p(z)),
            c * z / p.H0() * (1.0 + 0.5 * (1.0 - q0) * z), 1e-6);
  const double xz = -std::log(1.0 + 2.0);
  check_rel("d_A = d_L / (1+z)^2", cosmo.angular_distance_of_x(xz),
            cosmo.luminosity_distance_of_x(xz) / 9.0, 1e-14);
  check_rel("T_CMB(z=1089) = 2.7255*1090 K", cosmo.TCMB_of_x(-std::log(1090.0)), 2.7255 * 1090.0, 1e-12);

  cosmo.info();
}

static void test_einstein_de_sitter() {
  std::cout << "Einstein-de Sitter (Omega_M = 1, negligible radiation)\n";
  CosmologyParameters p;
  p.OmegaB = 0.05; p.OmegaCDM = 0.95; p.Neff = 0.0; p.TCMB = 1e-4;  // Omega_gamma ~ 1e-22
  BackgroundCosmology cosmo(p);
  cosmo.solve();
  check_rel("eta0 = 2c/H0",      cosmo.eta0(), 2.0 * c / p.H0(), 1e-9);
  check_rel("t0 = 2/(3 H0)",     cosmo.t0(),   2.0 / (3.0 * p.H0()), 1e-9);
  check_close("dHp/dx / Hp = -1/2", cosmo.dHpdx_of_x(-5.0) / cosmo.Hp_of_x(-5.0), -0.5, 1e-10);
  check_close("d2Hp/dx2 / Hp = 1/4", cosmo.ddHpddx_of_x(-5.0) / cosmo.Hp_of_x(-5.0), 0.25, 1e-10);
  check_close("eta Hp / c = 2",     cosmo.eta_of_x(-5.0) * cosmo.Hp_of_x(-5.0) / c, 2.0, 1e-9);
  // d_L = (2c/H0) (1 + z) (1 - 1/sqrt(1+z))
  const double z = 2.0;
  check_rel("d_L(z=2) analytic", cosmo.luminosity_distance_of_x(-std::log(1.0 + z)),
            2.0 * c / p.H0() * (1.0 + z) * (1.0 - 1.0 / std::sqrt(1.0 + z)), 1e-9);
}

static void test_curvature() {
  std::cout << "Curved models\n";
  for (double OmK : {0.1, -0.1}) {
    CosmologyParameters p;
    p.OmegaK = OmK;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    const std::string label = std::string("OmegaK = ") + (OmK > 0 ? "+0.1" : "-0.1");
    check_closure(cosmo, label);
    check_Hp_derivatives(cosmo, label);
    check_rel(label + ": eta0 vs quadrature", cosmo.eta0(), eta_reference(p, 1.0), 1e-8);

    // d_L = (1+z) c/(H0 sqrt|OmK|) S(sqrt|OmK| H0 chi / c) with chi from quadrature
    const double z = 1.5, a = 1.0 / (1.0 + z);
    const double chi = eta_reference(p, 1.0) - eta_reference(p, a);
    const double s = std::sqrt(std::fabs(OmK)) * p.H0() * chi / c;
    const double S = OmK > 0 ? std::sinh(s) : std::sin(s);
    check_rel(label + ": d_L(z=1.5) vs quadrature", cosmo.luminosity_distance_of_x(std::log(a)),
              (1.0 + z) * c / (p.H0() * std::sqrt(std::fabs(OmK))) * S, 1e-8);
  }
}

// Geometry helpers sinK and cotK used by the perturbations and the line-of-sight integrals
static void test_geometry() {
  std::cout << "Curved geometry: sinK, cotK\n";
  {
    CosmologyParameters p;
    BackgroundCosmology flat(p);
    const double chi = 1e4 * Mpc;
    check_close("flat: sinK(chi) = chi", flat.sinK(chi) / chi, 1.0, 0.0);
    check_close("flat: cotK(chi) = 1/chi", flat.cotK(chi) * chi, 1.0, 0.0);
  }
  for (double OmK : {0.05, -0.05}) {
    CosmologyParameters p;
    p.OmegaK = OmK;
    BackgroundCosmology cosmo(p);
    const std::string label = std::string("OmegaK = ") + (OmK > 0 ? "+0.05: " : "-0.05: ");
    const double K = cosmo.get_K(), sK = std::sqrt(std::fabs(K));
    check_rel(label + "K = -OmegaK H0^2/c^2", K, -OmK * p.H0() * p.H0() / (c * c), 1e-14);

    // Closed form far from the series region
    const double chi = 1.0 / sK;
    check_rel(label + "sinK(1/sqrt|K|) closed form", cosmo.sinK(chi),
              (K > 0 ? std::sin(1.0) : std::sinh(1.0)) / sK, 1e-14);
    check_rel(label + "cotK(1/sqrt|K|) closed form", cosmo.cotK(chi),
              sK * (K > 0 ? 1.0 / std::tan(1.0) : 1.0 / std::tanh(1.0)), 1e-14);

    // Continuity across the switch to the series at |K| chi^2 = 1e-4
    const double chi_s = 1e-2 / sK;
    check_rel(label + "sinK continuous at series switch", cosmo.sinK(chi_s * (1 + 1e-9)) / cosmo.sinK(chi_s * (1 - 1e-9)),
              1.0 + 2e-9, 1e-12);
    check_rel(label + "cotK continuous at series switch", cosmo.cotK(chi_s * (1 + 1e-9)) / cosmo.cotK(chi_s * (1 - 1e-9)),
              1.0 - 2e-9, 1e-12);

    // cotK = d ln sinK / d chi and sinK'' = -K sinK (finite differences), at several chi
    double err_cot = 0.0, err_ode = 0.0;
    for (double y : {0.003, 0.05, 0.3, 1.0, 2.0}) {
      const double x0 = y / sK, h = 1e-4 * x0;
      const double sp = cosmo.sinK(x0 + h), s0 = cosmo.sinK(x0), sm = cosmo.sinK(x0 - h);
      err_cot = std::max(err_cot, std::fabs((sp - sm) / (2.0 * h) / s0 / cosmo.cotK(x0) - 1.0));
      // The second difference loses ~eps/(h^2 K) to roundoff, so only test where K chi^2 is not tiny
      if (y >= 0.3) err_ode = std::max(err_ode, std::fabs((sp - 2.0 * s0 + sm) / (h * h) / (-K * s0) - 1.0));
    }
    check_close(label + "cotK = d ln sinK / dchi", err_cot, 0.0, 1e-7);
    check_close(label + "sinK'' = -K sinK", err_ode, 0.0, 1e-4);

    if (K > 0) check_close(label + "sinK(pi/sqrt K) = 0", cosmo.sinK(M_PI / sK) * sK, 0.0, 1e-14);
    cosmo.solve();
    const double xz = -std::log(1.0 + 1.5);
    check_rel(label + "transverse distance = sinK(eta0 - eta)", cosmo.transverse_distance_of_x(xz),
              cosmo.sinK(cosmo.eta0() - cosmo.eta_of_x(xz)), 1e-15);
  }
}

// LCDM with Lambda replaced by CPL dark energy, assembled from the building blocks of Model.h
static BackgroundCosmology cpl_model(const CosmologyParameters& p, double w0, double wa) {
  Model m = Models::lcdm(p);
  m.name = "cpl";
  m.species.back() = std::make_shared<CPLDarkEnergy>(p.OmegaLambda(), w0, wa);
  return BackgroundCosmology(p, m);
}

static void test_wcdm() {
  std::cout << "Model with CPL dark energy\n";
  const CosmologyParameters p;
  BackgroundCosmology lcdm(p);
  BackgroundCosmology w_lcdm = cpl_model(p, -1.0, 0.0);
  lcdm.solve();
  w_lcdm.solve();
  double max_diff = 0.0;
  for (double x = -15.0; x <= 3.0; x += 0.5)
    max_diff = std::max(max_diff, std::fabs(w_lcdm.Hp_of_x(x) / lcdm.Hp_of_x(x) - 1.0));
  check_close("w0=-1, wa=0 reproduces LCDM: Hp(x)", max_diff, 0.0, 1e-14);
  check_rel("w0=-1, wa=0 reproduces LCDM: eta0", w_lcdm.eta0(), lcdm.eta0(), 1e-12);

  BackgroundCosmology wconst = cpl_model(p, -0.8, 0.0);
  check_rel("constant w=-0.8: rho_DE ~ a^{-0.6}", wconst.rhoDE(-1.0) / wconst.rhoDE(0.0), std::exp(0.6), 1e-14);

  BackgroundCosmology cpl = cpl_model(p, -0.9, 0.3);
  cpl.solve();
  check_closure(cpl, "CPL (w0=-0.9, wa=0.3)");
  // CPL is only meaningful for a <~ few: for a >> 1, w -> -wa a and rho_DE ~ exp(3 wa a),
  // which makes finite differences inaccurate. Check up to x = 2.
  check_Hp_derivatives(cpl, "CPL (w0=-0.9, wa=0.3)", 2.0);
  // Independent check of eta0 with the CPL density written out
  const double eta0_ref = Utils::integrate([&](double u) {
    const double a = u * u;
    const double E = std::sqrt(p.OmegaM() / (a * a * a) + p.OmegaR() / (a * a * a * a) +
                               p.OmegaLambda() * std::pow(a, -3.0 * (1.0 - 0.9 + 0.3)) * std::exp(-3.0 * 0.3 * (1.0 - a)));
    return 2.0 * u * c / (a * a * p.H0() * E);
  }, 0.0, 1.0, 0.0, 1e-12);
  check_rel("CPL: eta0 vs quadrature", cpl.eta0(), eta0_ref, 1e-8);
}

int main() {
  test_fiducial();
  test_einstein_de_sitter();
  test_curvature();
  test_geometry();
  test_wcdm();
  return Test::summary();
}
