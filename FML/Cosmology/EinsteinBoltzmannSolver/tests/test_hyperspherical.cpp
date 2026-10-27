// =============================================================================
// Tests for the hyperspherical Bessel functions Phi^nu_ell(chi). Run with `make test`.
// =============================================================================

#include <FML/Cosmology/Spectra/HypersphericalBessel.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Common.h>

#include <gsl/gsl_sf_gegenbauer.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace FML::COSMOLOGY;

using Test::check;
using Test::check_close;
using Test::check_rel;

static std::string geom(int sign) { return sign == 1 ? "closed" : (sign == -1 ? "open" : "flat"); }

// Residual of the radial equation Phi'' + 2 cot_K Phi' + [nu^2 - sign - l(l+1)/sin_K^2] Phi = 0,
// with Phi'' from finite differences and Phi' from the recurrence identity, relative to the
// size of the individual terms
static double ode_residual(const HypersphericalBessel& hb, int l, double chi) {
  const double nu = hb.nu();
  const double h = std::min(1e-2 / nu, 1e-3 * chi);
  std::vector<double> p, dp;
  hb.column_with_derivative(chi, l, p, dp);
  const double pp = hb(l, chi + h), pm = hb(l, chi - h);
  const double d2 = (pp - 2.0 * p[l] + pm) / (h * h);
  const double sK = hb.sinK(chi), cK = hb.cotK(chi);
  const double t1 = d2, t2 = 2.0 * cK * dp[l], t3 = (nu * nu - hb.sign() - l * (l + 1.0) / (sK * sK)) * p[l];
  // Normalise by the typical size of Phi'' (nu^2 times the amplitude in the oscillating region)
  const double amp = std::fabs(p[l]) + std::fabs(dp[l]) / nu + std::fabs(d2) / (nu * nu);
  return std::fabs(t1 + t2 + t3) / (nu * nu * amp + 1e-300);
}

int main() {
  std::cout << "Flat case: Phi^nu_ell(chi) = j_ell(nu chi)\n";
  {
    HypersphericalBessel hb(0, 1.0);
    double err = 0.0;
    std::vector<double> col;
    for (double z : {0.3, 3.0, 30.0, 300.0, 1500.0, 3000.0}) {
      const int lmax = 2500;
      hb.column(z, lmax, col);
      double amp = 0.0;
      for (int l = 0; l <= lmax; l++) amp = std::max(amp, std::fabs(Utils::j_ell(l, z)));
      for (int l = 0; l <= lmax; l += 7) err = std::max(err, std::fabs(col[l] - Utils::j_ell(l, z)) / amp);
    }
    check_close("recurrence vs j_ell, z = 0.3..3000, ell <= 2500", err, 0.0, 1e-10);
  }

  std::cout << "Flat limit of the curved functions (nu -> infinity, nu chi fixed)\n";
  for (int sign : {-1, 1}) {
    // Max deviation from j_ell(nu chi) over ell <= 400 and nu chi = 5, 50, 300 (relative to max |j_ell|)
    auto deviation = [sign](double nu) {
      HypersphericalBessel hb(sign, nu);
      double err = 0.0;
      std::vector<double> col;
      for (double z : {5.0, 50.0, 300.0}) {
        hb.column(z / nu, 400, col);
        double amp = 0.0;
        for (int l = 0; l <= 400; l++) amp = std::max(amp, std::fabs(Utils::j_ell(l, z)));
        for (int l = 0; l <= 400; l++) err = std::max(err, std::fabs(col[l] - Utils::j_ell(l, z)) / amp);
      }
      return err;
    };
    const double e5 = deviation(1e5), e6 = deviation(1e6);
    // The curvature corrections are O(chi^2) = O((nu chi)^2 / nu^2): ~1e-4 at nu = 1e5
    check_close(geom(sign) + ": Phi -> j_ell(nu chi), nu = 1e5, ell <= 400", e5, 0.0, 2e-4);
    check_rel(geom(sign) + ": deviation scales as 1/nu^2 (nu = 1e5 -> 1e6)", e6 / e5, 0.01, 0.05);
  }

  std::cout << "Radial equation (finite differences)\n";
  for (int sign : {-1, 1}) {
    for (double nu : {7.0, 60.0, 1000.0, 6000.0}) {
      HypersphericalBessel hb(sign, nu);
      double worst = 0.0;
      for (double chi : {0.05, 0.3, 0.7, 1.4}) {
        if (sign == 1 && chi > 3.0) continue;
        const double ltp = hb.turning_point(chi);
        // ell in the oscillating region, near the turning point and beyond it
        std::vector<int> ells = {0, 1, 2, 5};
        for (double f : {0.5, 0.9, 1.0, 1.05, 1.2}) ells.push_back(int(f * ltp));
        for (int l : ells) {
          if (l < 0 || l > 2500 || (sign == 1 && l >= nu)) continue;
          if (std::fabs(hb(l, chi)) < 1e-12 * std::fabs(hb(0, chi))) continue;  // too small to matter
          worst = std::max(worst, ode_residual(hb, l, chi));
        }
      }
      check_close(geom(sign) + ": ODE residual, nu = " + std::to_string(int(nu)), worst, 0.0, 1e-4);
    }
  }

  std::cout << "Derivative identity vs finite differences\n";
  for (int sign : {-1, 1}) {
    const double nu = 200.0, chi = 0.6;
    HypersphericalBessel hb(sign, nu);
    std::vector<double> p, dp;
    hb.column_with_derivative(chi, 150, p, dp);
    double err = 0.0, amp = 0.0;
    for (int l = 0; l <= 150; l++) amp = std::max(amp, std::fabs(dp[l]));
    for (int l = 0; l <= 150; l += 5) {
      const double h = 1e-5;
      err = std::max(err, std::fabs((hb(l, chi + h) - hb(l, chi - h)) / (2 * h) - dp[l]) / amp);
    }
    check_close(geom(sign) + ": dPhi/dchi = l cot_K Phi_l - sqrt(nu^2 - s(l+1)^2) Phi_{l+1}", err, 0.0, 1e-6);
  }

  std::cout << "Closed universe: exact properties\n";
  {
    // Phi^nu_l(chi) is proportional to sin^l(chi) C^(l+1)_{nu-l-1}(cos chi) (Gegenbauer polynomial)
    const int nu = 30;
    HypersphericalBessel hb(1, nu);
    double spread = 0.0;
    std::vector<double> col;
    for (int l : {0, 3, 10, 20, 28}) {
      std::vector<double> ratio;
      for (double chi : {0.4, 0.9, 1.3, 2.2}) {
        hb.column(chi, l, col);
        const double geg = std::pow(std::sin(chi), l) * gsl_sf_gegenpoly_n(nu - l - 1, l + 1.0, std::cos(chi));
        if (std::fabs(geg) > 1e-8) ratio.push_back(col[l] / geg);
      }
      for (double r : ratio) spread = std::max(spread, std::fabs(r / ratio[0] - 1.0));
    }
    check_close("Phi ~ sin^l C^(l+1)_{nu-l-1}(cos chi), nu = 30", spread, 0.0, 1e-10);

    // Symmetry Phi(pi - chi) = (-1)^(nu - l - 1) Phi(chi)
    double err = 0.0;
    std::vector<double> a, b;
    hb.column(0.8, 29, a);
    hb.column(M_PI - 0.8, 29, b);
    for (int l = 0; l < nu; l++) err = std::max(err, std::fabs(b[l] - ((nu - l - 1) % 2 == 0 ? 1.0 : -1.0) * a[l]));
    check_close("Phi(pi - chi) = (-1)^(nu-l-1) Phi(chi)", err, 0.0, 1e-12);

    hb.column(1.0, 40, col);
    double tail = 0.0;
    for (int l = nu; l <= 40; l++) tail = std::max(tail, std::fabs(col[l]));
    check_close("Phi_l = 0 for l >= nu", tail, 0.0, 0.0);

    bool threw = false;
    try { HypersphericalBessel bad(1, 30.5); } catch (const std::invalid_argument&) { threw = true; }
    check("closed universe rejects non-integer nu", threw);
  }

  std::cout << "Downward recurrence: independent of where it starts\n";
  for (int sign : {-1, 1}) {
    const double nu = 3000.0, chi = 0.2;  // turning point ell ~ 600
    HypersphericalBessel hb(sign, nu);
    std::vector<double> a, b;
    hb.column(chi, 700, a);
    hb.column(chi, 2500, b);
    double err = 0.0;
    for (int l = 0; l <= 700; l++) err = std::max(err, std::fabs(a[l] - b[l]) / (std::fabs(a[l]) + 1e-300));
    check_close(geom(sign) + ": column(lmax = 700) = column(lmax = 2500) for l <= 700", err, 0.0, 1e-8);
    check("decays beyond the turning point: |Phi_700| < 1e-10 |Phi_500|", std::fabs(b[700]) < 1e-10 * std::fabs(b[500]) ||
          std::fabs(b[500]) == 0.0);
  }

  std::cout << "Small chi\n";
  for (int sign : {-1, 1}) {
    HypersphericalBessel hb(sign, 50.0);
    std::vector<double> p, dp;
    hb.column_with_derivative(0.0, 3, p, dp);
    check_close(geom(sign) + ": Phi_0(0) = 1", p[0], 1.0, 0.0);
    // Phi_1 ~ sqrt(nu^2 - sign) chi / 3 for small chi
    const double chi = 1e-6;
    check_rel(geom(sign) + ": Phi_1 ~ sqrt(nu^2 - s) chi / 3", hb(1, chi), std::sqrt(2500.0 - sign) * chi / 3.0, 1e-6);
    check_rel(geom(sign) + ": dPhi_1/dchi(0) = sqrt(nu^2 - s)/3", dp[1], std::sqrt(2500.0 - sign) / 3.0, 1e-14);
  }

  {
    Utils::Timer t("column, nu = 6000, lmax = 2500, 1000 chi values");
    HypersphericalBessel hb(-1, 6000.0);
    std::vector<double> col;
    double sum = 0.0;
    for (int i = 1; i <= 1000; i++) {
      hb.column(0.7 * i / 1000.0, 2500, col);
      sum += col[100];
    }
    t.print();
    check("timing run finite", std::isfinite(sum));
  }
  return Test::summary();
}
