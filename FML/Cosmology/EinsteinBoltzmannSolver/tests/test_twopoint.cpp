// =============================================================================
// Tests for the two-point functions (TwoPoint.h and the PowerSpectrum wrappers).
// The comparison with CLASS is python/compare_class_twopoint.py. Run with `make test`.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Spectra/Lensing.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include "TestHelpers.h"
#include <FML/Cosmology/Spectra/TwoPoint.h>
#include <FML/Cosmology/Common.h>

#include <gsl/gsl_integration.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace FML::COSMOLOGY;

using namespace Constants;
using Test::check;
using Test::check_close;
using Test::check_rel;

int main() {
  std::cout << "Spherical Bessel functions for all ell\n";
  {
    std::vector<double> j(401);
    double err = 0.0;
    for (double x : {1e-3, 0.3, 4.0, 50.0, 199.5, 200.5, 399.0, 1500.0}) {
      Utils::j_ell_array(400, x, j.data());
      for (int l : {0, 1, 2, 7, 60, 150, 300, 400}) {
        const double r = Utils::j_ell(l, x);
        if (std::fabs(r) > 1e-250) err = std::max(err, std::fabs(j[l] / r - 1.0));
      }
    }
    check_close("j_ell_array vs j_ell (upward and Miller recurrences)", err, 0.0, 1e-9);
  }

  // Synthetic band-limited spectra
  const int L = 400;
  std::vector<double> tt(L + 1, 0.0), ee(L + 1, 0.0), te(L + 1, 0.0), bb(L + 1, 0.0);
  for (int l = 2; l <= L; l++) {
    tt[l] = 1.0 / (l * (l + 1.0)) * (1.0 + 0.5 * std::cos(l / 20.0)) * std::exp(-l / 150.0);
    ee[l] = 0.05 * tt[l] * (l / 50.0);
    te[l] = 0.3 * std::sqrt(tt[l] * ee[l]) * std::sin(l / 15.0);
    bb[l] = 0.1 * ee[l];
  }

  std::cout << "Angular correlation functions\n";
  {
    const auto th = Utils::linspace(0.0, M_PI, 7);
    const auto C = TwoPoint::cmb_correlation(tt, ee, te, bb, th);
    double var = 0.0, varP = 0.0;
    for (int l = 2; l <= L; l++) {
      var += (2 * l + 1) / (4 * M_PI) * tt[l];
      varP += (2 * l + 1) / (4 * M_PI) * (ee[l] + bb[l]);
    }
    check_rel("C^TT(0) = sum (2l+1)/(4 pi) C_l (variance)", C.TT[0], var, 1e-12);
    check_rel("xi_+(0) = <Q^2 + U^2>", C.plus[0], varP, 1e-12);
    check_close("xi_-(0) = 0, xi_X(0) = 0 (no preferred direction at zero lag)",
                std::fabs(C.minus[0]) + std::fabs(C.TQ[0]), 0.0, 1e-14);

    // Inversion with Gauss-Legendre quadrature recovers the C_l
    auto w = [&](double t) { return TwoPoint::angular_correlation(tt, {t})[0]; };
    const auto cl = TwoPoint::legendre_transform(w, L, L + 10);
    // (errors relative to the largest C_l; the Gauss-Legendre nodes of GSL limit this to ~1e-9)
    const double tmax = *std::max_element(tt.begin(), tt.end());
    double err = 0.0;
    for (int l = 2; l <= L; l++) err = std::max(err, std::fabs(cl[l] - tt[l]) / tmax);
    check_close("Legendre transform of C^TT(theta) returns C_l^TT", err, 0.0, 1e-8);
    check_close("... and no monopole/dipole", (std::fabs(cl[0]) + std::fabs(cl[1])) / tmax, 0.0, 1e-8);

    // Spin-2: C_l^EE + C_l^BB = 2 pi int xi_+ d^l_22 dmu, C_l^TE = 2 pi int xi_X d^l_20 dmu
    gsl_integration_glfixed_table* gl = gsl_integration_glfixed_table_alloc(L + 10);
    std::vector<double> sp(L + 1, 0.0), sx(L + 1, 0.0), d22, d20;
    for (int i = 0; i < L + 10; i++) {
      double mu, wt;
      gsl_integration_glfixed_point(-1.0, 1.0, i, &mu, &wt, gl);
      const auto c = TwoPoint::cmb_correlation(tt, ee, te, bb, {std::acos(mu)});
      Lensing::wigner_d(2, 2, mu, L, d22);
      Lensing::wigner_d(2, 0, mu, L, d20);
      for (int l = 2; l <= L; l++) {
        sp[l] += 2 * M_PI * wt * c.plus[0] * d22[l];
        sx[l] += 2 * M_PI * wt * c.TQ[0] * d20[l];
      }
    }
    gsl_integration_glfixed_table_free(gl);
    double ep = 0.0, ex = 0.0, temax = 0.0, pmax = 0.0;
    for (int l = 2; l <= L; l++) { temax = std::max(temax, std::fabs(te[l])); pmax = std::max(pmax, ee[l] + bb[l]); }
    for (int l = 2; l <= L; l++) {
      ep = std::max(ep, std::fabs(sp[l] - ee[l] - bb[l]) / pmax);
      ex = std::max(ex, std::fabs(sx[l] - te[l]) / temax);
    }
    check_close("spin-2 inversion: xi_+ -> C_l^EE + C_l^BB", ep, 0.0, 1e-8);
    check_close("spin-0/2 inversion: xi_X -> C_l^TE", ex, 0.0, 1e-8);

    // Gaussian beam: B_l and smoothing of the variance
    const double fwhm = 10.0 * M_PI / 180.0, sig = fwhm / std::sqrt(8 * std::log(2.0));
    check_rel("beam B_l = exp(-l(l+1) sigma^2/2)", TwoPoint::beam(100, fwhm), std::exp(-0.5 * 100 * 101 * sig * sig), 1e-14);
    check("beam lowers the variance", TwoPoint::angular_correlation(tt, {0.0}, fwhm)[0] < 0.9 * var);
  }

  std::cout << "3D correlation function and sigma_R\n";
  {
    // P = A exp(-k^2 R0^2): xi(r) = A exp(-r^2/(4 R0^2)) / (8 pi^{3/2} R0^3)
    const double A = 1.7, R0 = 2.0;
    const auto k = Utils::logspace(1e-4, 20.0, 20000);
    std::vector<double> P(k.size());
    for (size_t i = 0; i < k.size(); i++) P[i] = A * std::exp(-k[i] * k[i] * R0 * R0);
    const std::vector<double> r = {0.0, 0.5, 2.0, 5.0, 10.0, 15.0};
    auto exact = [&](double rr, double R2) { return A * std::exp(-rr * rr / (4 * R2)) / (8 * std::pow(M_PI, 1.5) * std::pow(R2, 1.5)); };
    const auto xi = TwoPoint::xi_of_r(k, P, r);
    double err = 0.0;
    for (size_t i = 0; i < r.size(); i++) err = std::max(err, std::fabs(xi[i] - exact(r[i], R0 * R0)) / exact(0, R0 * R0));
    check_close("xi(r) of a Gaussian P(k) (Filon)", err, 0.0, 1e-6);
    const auto xis = TwoPoint::xi_of_r(k, P, r, 1.5);   // smoothing adds in quadrature
    err = 0.0;
    for (size_t i = 0; i < r.size(); i++)
      err = std::max(err, std::fabs(xis[i] - exact(r[i], R0 * R0 + 2.25)) / exact(0, R0 * R0 + 2.25));
    check_close("Gaussian smoothing R: R0^2 -> R0^2 + R^2", err, 0.0, 1e-6);
    // Oscillating P(k): Filon handles large r on a coarse grid; compare with adaptive QAWO-like quadrature
    for (size_t i = 0; i < k.size(); i++) P[i] = A * std::exp(-k[i] * k[i]) * (1.0 + 0.2 * std::sin(150.0 * k[i]));
    const auto xo = TwoPoint::xi_of_r(k, P, {150.0});
    const double ref = Utils::integrate([&](double kk) {
      return kk * A * std::exp(-kk * kk) * (1.0 + 0.2 * std::sin(150.0 * kk)) * std::sin(150.0 * kk);
    }, 1e-4, 20.0, 0.0, 1e-11) / (2 * M_PI * M_PI * 150.0);
    check_rel("xi(r = 150) of an oscillating P(k) (BAO-like feature)", xo[0], ref, 1e-3);

    for (size_t i = 0; i < k.size(); i++) P[i] = A * std::exp(-k[i] * k[i] * R0 * R0);
    const double R = 3.0;
    const double s2 = Utils::integrate([&](double kk) {
      const double y = kk * R, W = 3 * (std::sin(y) - y * std::cos(y)) / (y * y * y);
      return kk * kk * A * std::exp(-kk * kk * R0 * R0) * W * W / (2 * M_PI * M_PI);
    }, 1e-4, 20.0, 0.0, 1e-12);
    check_rel("sigma_R (top hat) vs adaptive quadrature", TwoPoint::sigma_R(k, P, R), std::sqrt(s2), 1e-6);
  }

  std::cout << "Matter field of the fiducial model\n";
  {
    CosmologyParameters p;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationHistory rec(cosmo);
    rec.solve();
    PerturbationOptions o;
    o.n_k = 150;
    Perturbations pert(cosmo, rec, o);
    pert.solve();
    PowerSpectrum ps(cosmo, rec, pert);

    // sigma_8: CLASS gives 0.82068 for this model
    check_rel("sigma_8 = 0.8207 (CLASS)", ps.get_sigma8(), 0.82068, 2e-3);
    // Extrapolation beyond k_max is continuous
    const double kmax = pert.get_k_array().back();
    check_rel("Delta_M extrapolation continuous at k_max", ps.get_Delta_M_extrapolated(0.0, kmax * (1 + 1e-9)),
              ps.get_Delta_M(0.0, kmax), 1e-6);

    // BAO peak in r^2 xi(r) near r_d ~ 147 Mpc ~ 99 Mpc/h
    const double Mh = Mpc / p.h;
    std::vector<double> r;
    for (double rr = 80.0; rr <= 120.0; rr += 0.25) r.push_back(rr * Mh);
    const auto xi = ps.get_xi_matter(0.0, r);
    size_t imax = 0;
    for (size_t i = 0; i < r.size(); i++) if (r[i] * r[i] * xi[i] > r[imax] * r[imax] * xi[imax]) imax = i;
    std::cout << "    BAO peak of r^2 xi at r = " << r[imax] / Mh << " Mpc/h\n";
    check("BAO peak of r^2 xi(r) at 95-105 Mpc/h", r[imax] / Mh > 95.0 && r[imax] / Mh < 105.0);

    // Projected density: exact vs Limber, bias, window normalisation
    auto w = TwoPoint::RedshiftWindow::gaussian(1.0, 0.1);
    const std::vector<int> ells = {10, 300};
    const auto ce = ps.get_angular_cl_matter(w, ells, PowerSpectrum::Projection::Exact);
    const auto cl = ps.get_angular_cl_matter(w, ells, PowerSpectrum::Projection::Limber);
    check_rel("Limber = exact at ell = 300 (sigma_z = 0.1)", cl[1], ce[1], 2e-3);
    check("Limber fails at ell = 10 (> 2%)", std::fabs(cl[0] / ce[0] - 1.0) > 0.02);
    auto w2 = w;
    w2.bias = 2.0;
    const double n0 = 5.0;
    w2.n = [&w, n0](double z) { return n0 * w.n(z); };
    const auto c2 = ps.get_angular_cl_matter(w2, ells, PowerSpectrum::Projection::Limber);
    check_rel("C_ell ~ b^2 and independent of the normalisation of n(z)", c2[1], 4.0 * cl[1], 1e-12);
  }
  return Test::summary();
}
