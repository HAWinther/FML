// =============================================================================
// Tests for the CMB lensing module (Wigner d-functions and the correlation-function method).
// The comparison of the full pipeline with CLASS is python/compare_class_lensing.py.
// Run with `make test`.
// =============================================================================

#include <FML/Cosmology/Spectra/Lensing.h>
#include "TestHelpers.h"

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace FML::COSMOLOGY;

using Test::check;
using Test::check_close;
using Test::check_rel;

// Synthetic spectra with acoustic-like oscillations and a damping tail, and a realistic-shaped
// lensing potential (amplitude ~ the LCDM deflection power)
static void synthetic(int L, double pp_scale, std::vector<double>& tt, std::vector<double>& ee,
                      std::vector<double>& te, std::vector<double>& pp) {
  tt.assign(L + 1, 0.0); ee.assign(L + 1, 0.0); te.assign(L + 1, 0.0); pp.assign(L + 1, 0.0);
  for (int l = 2; l <= L; l++) {
    const double ll = l * (l + 1.0), damp = std::exp(-std::pow(l / 1400.0, 2));
    tt[l] = 2.0 * M_PI / ll * 1e-10 * (1.0 + 0.6 * std::cos(l / 48.0)) * damp;
    ee[l] = 2.0 * M_PI / ll * 2e-12 * (l / 300.0) * (1.0 + 0.8 * std::sin(l / 48.0)) * damp;
    te[l] = 0.3 * std::sqrt(tt[l] * ee[l]) * std::cos(l / 30.0);
    pp[l] = pp_scale * 2.0 * M_PI / (ll * ll) * 1.4e-7 / (1.0 + std::pow(l / 60.0, 1.6));
  }
}

int main() {
  std::cout << "Wigner d-functions\n";
  {
    std::vector<double> d;
    double err = 0.0;
    for (double mu : {-0.9, -0.3, 0.2, 0.7, 0.99}) {
      Lensing::wigner_d(0, 0, mu, 300, d);
      for (int l : {0, 1, 2, 10, 100, 300}) err = std::max(err, std::fabs(d[l] - gsl_sf_legendre_Pl(l, mu)));
    }
    check_close("d^l_00 = P_l(mu)", err, 0.0, 1e-12);
    // l = 2 closed forms
    const double mu = 0.37, c2 = 0.5 * (1 + mu), s2 = 0.5 * (1 - mu), sb2 = 1 - mu * mu;
    Lensing::wigner_d(2, 2, mu, 2, d);
    check_close("d^2_22 = cos^4(beta/2)", d[2], c2 * c2, 1e-14);
    Lensing::wigner_d(2, -2, mu, 2, d);
    check_close("d^2_2-2 = sin^4(beta/2)", d[2], s2 * s2, 1e-14);
    Lensing::wigner_d(2, 0, mu, 2, d);
    check_close("d^2_20 = sqrt(3/8) sin^2(beta)", d[2], std::sqrt(3.0 / 8.0) * sb2, 1e-14);
    Lensing::wigner_d(1, -1, mu, 1, d);
    check_close("d^1_1-1 = sin^2(beta/2)", d[1], s2, 1e-14);
    // Orthogonality: int d^l_mm' d^l'_mm' dmu = 2/(2l+1) delta_ll'
    const int N = 1500;
    gsl_integration_glfixed_table* gl = gsl_integration_glfixed_table_alloc(N);
    double norm_err = 0.0, overlap = 0.0;
    const int pairs[5][2] = {{2, 2}, {2, -2}, {2, 0}, {3, -1}, {4, -4}};
    for (const auto& pr : pairs) {
      double n1 = 0.0, o = 0.0;
      for (int i = 0; i < N; i++) {
        double x, w;
        gsl_integration_glfixed_point(-1.0, 1.0, i, &x, &w, gl);
        Lensing::wigner_d(pr[0], pr[1], x, 1000, d);
        n1 += w * d[1000] * d[1000];
        o += w * d[1000] * d[999];
      }
      norm_err = std::max(norm_err, std::fabs(n1 * 2001.0 / 2.0 - 1.0));
      overlap = std::max(overlap, std::fabs(o));
    }
    gsl_integration_glfixed_table_free(gl);
    check_close("orthogonality: (2l+1)/2 int (d^l)^2 = 1, l = 1000", norm_err, 0.0, 1e-7);
    check_close("orthogonality: int d^l d^(l-1) = 0", overlap, 0.0, 1e-10);
  }

  const int L = 2600, lmax = 2000;
  std::vector<double> tt, ee, te, pp;

  std::cout << "No lensing potential\n";
  {
    synthetic(L, 0.0, tt, ee, te, pp);
    const auto r = Lensing::lens(tt, ee, te, {}, pp, lmax);
    double err = 0.0, bb = 0.0;
    for (int l = 2; l <= lmax; l++) {
      err = std::max({err, std::fabs(r.tt[l] / tt[l] - 1), std::fabs(r.ee[l] / ee[l] - 1)});
      bb = std::max(bb, std::fabs(r.bb[l]) / ee[l]);
    }
    check_close("C_phiphi = 0: lensed = unlensed", err, 0.0, 1e-11);
    check_close("C_phiphi = 0: no B-modes", bb, 0.0, 1e-11);
  }

  std::cout << "Physical behaviour\n";
  {
    synthetic(L, 1.0, tt, ee, te, pp);
    const auto r = Lensing::lens(tt, ee, te, {}, pp, lmax);
    // Lensing smooths the oscillations: the peak-to-trough contrast decreases at high ell
    auto contrast = [&](const std::vector<double>& c, int l0) {
      double mx = 0.0, mn = 1e300;
      for (int l = l0; l < l0 + 302; l++) {
        const double D = l * (l + 1.0) * c[l];
        mx = std::max(mx, D);
        mn = std::min(mn, D);
      }
      return (mx - mn) / (mx + mn);
    };
    check("lensing smooths the acoustic oscillations of TT (ell ~ 1500)", contrast(r.tt, 1400) < 0.95 * contrast(tt, 1400));
    double low = 0.0;
    for (int l = 2; l <= 30; l++) low = std::max(low, std::fabs(r.tt[l] / tt[l] - 1));
    check_close("large scales are hardly affected (ell <= 30)", low, 0.0, 2e-3);
    bool bb_pos = true;
    for (int l = 2; l <= lmax; l++) bb_pos &= r.bb[l] > 0.0;
    check("lensing turns E into positive C_l^BB", bb_pos);
    // Total TT variance sum (2l+1) C_l is nearly conserved (lensing remaps the sky)
    double s0 = 0.0, s1 = 0.0;
    for (int l = 2; l <= lmax; l++) { s0 += (2 * l + 1) * tt[l]; s1 += (2 * l + 1) * r.tt[l]; }
    check_close("sum (2l+1) C_l^TT nearly conserved", s1 / s0 - 1.0, 0.0, 2e-3);

    // Small deflections: the change is linear in C_phiphi
    std::vector<double> pph = pp, pp2 = pp;
    for (auto& v : pph) v *= 1e-3;
    for (auto& v : pp2) v *= 2e-3;
    const auto r1 = Lensing::lens(tt, ee, te, {}, pph, lmax);
    const auto r2 = Lensing::lens(tt, ee, te, {}, pp2, lmax);
    double lin = 0.0;
    for (int l : {300, 800, 1500}) lin = std::max(lin, std::fabs((r2.tt[l] - tt[l]) / (r1.tt[l] - tt[l]) - 2.0));
    check_close("weak deflections: Delta C_l linear in C_phiphi", lin, 0.0, 1e-2);
    check_rel("weak deflections: C_l^BB linear in C_phiphi", r2.bb[1000] / r1.bb[1000], 2.0, 1e-2);
  }
  return Test::summary();
}
