#include <FML/Cosmology/Spectra/Lensing.h>

#include <gsl/gsl_integration.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FML::COSMOLOGY {

namespace Lensing {

namespace {
  double factorial(int n) {
    double f = 1.0;
    for (int i = 2; i <= n; i++) f *= i;
    return f;
  }

  // d^j_{a b}(beta) from the explicit (Wigner) sum, for small j
  double wigner_d_explicit(int j, int a, int b, double mu) {
    const double c = std::sqrt(std::max(0.0, 0.5 * (1.0 + mu)));  // cos(beta/2)
    const double s = std::sqrt(std::max(0.0, 0.5 * (1.0 - mu)));  // sin(beta/2)
    const double pre = std::sqrt(factorial(j + a) * factorial(j - a) * factorial(j + b) * factorial(j - b));
    double sum = 0.0;
    for (int k = 0; k <= 2 * j; k++) {
      if (j + b - k < 0 || a - b + k < 0 || j - a - k < 0) continue;
      const double sign = ((a - b + k) % 2 == 0) ? 1.0 : -1.0;
      const double den = factorial(j + b - k) * factorial(k) * factorial(a - b + k) * factorial(j - a - k);
      sum += sign / den * std::pow(c, 2 * j + b - a - 2 * k) * std::pow(s, a - b + 2 * k);
    }
    return pre * sum;
  }
}

void wigner_d(int m, int mp, double mu, int lmax, std::vector<double>& d) {
  d.assign(lmax + 1, 0.0);
  const int lmin = std::max(std::abs(m), std::abs(mp));
  if (lmax < lmin) return;
  d[lmin] = wigner_d_explicit(lmin, m, mp, mu);
  int l0 = lmin;
  if (lmin == 0) {  // Legendre polynomials: start the recurrence at l = 1
    if (lmax >= 1) d[1] = mu;
    l0 = 1;
  }
  const double mm = double(m) * mp, m2 = double(m) * m, mp2 = double(mp) * mp;
  for (int l = l0; l < lmax; l++) {
    const double L = l;
    const double prev = l - 1 >= lmin ? d[l - 1] : 0.0;
    const double c_prev = (L + 1.0) * std::sqrt(std::max(0.0, (L * L - m2) * (L * L - mp2)));
    const double c_next = L * std::sqrt(((L + 1.0) * (L + 1.0) - m2) * ((L + 1.0) * (L + 1.0) - mp2));
    d[l + 1] = ((2.0 * L + 1.0) * (L * (L + 1.0) * mu - mm) * d[l] - c_prev * prev) / c_next;
  }
}

Result lens(const std::vector<double>& cl_tt, const std::vector<double>& cl_ee, const std::vector<double>& cl_te,
            const std::vector<double>& cl_bb, const std::vector<double>& cl_pp, int lmax_out, const Options& opt) {
  const int L = int(cl_pp.size()) - 1;
  if (L <= lmax_out) throw std::invalid_argument("Lensing::lens: need C_l up to L > lmax_out");
  if (int(cl_tt.size()) < L + 1) throw std::invalid_argument("Lensing::lens: cl_tt shorter than cl_pp");
  const bool pol = !cl_ee.empty();
  if (pol && (int(cl_ee.size()) < L + 1 || int(cl_te.size()) < L + 1))
    throw std::invalid_argument("Lensing::lens: cl_ee/cl_te shorter than cl_pp");
  const bool has_bb = !cl_bb.empty();
  auto bb = [&](int l) { return has_bb ? cl_bb[l] : 0.0; };

  // Gauss-Legendre nodes in mu = cos(beta)
  const int N = L + opt.n_mu_extra;
  gsl_integration_glfixed_table* gl = gsl_integration_glfixed_table_alloc(N);
  std::vector<double> mu(N), wmu(N);
  for (int i = 0; i < N; i++) gsl_integration_glfixed_point(-1.0, 1.0, i, &mu[i], &wmu[i], gl);
  gsl_integration_glfixed_table_free(gl);

  // C_gl, C_gl,2 and sigma^2
  double Cgl_1 = 0.0;
  for (int l = 2; l <= L; l++) Cgl_1 += (2.0 * l + 1.0) * l * (l + 1.0) * cl_pp[l];
  Cgl_1 /= 4.0 * M_PI;
  std::vector<double> Cgl(N), Cgl2(N), sigma2(N);
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    std::vector<double> d11, d1m1;
    wigner_d(1, 1, mu[i], L, d11);
    wigner_d(1, -1, mu[i], L, d1m1);
    double a = 0.0, b = 0.0;
    for (int l = 2; l <= L; l++) {
      const double f = (2.0 * l + 1.0) * l * (l + 1.0) * cl_pp[l];
      a += f * d11[l];
      b += f * d1m1[l];
    }
    Cgl[i] = a / (4.0 * M_PI);
    Cgl2[i] = b / (4.0 * M_PI);
    sigma2[i] = Cgl_1 - Cgl[i];
  }

  // Lensed minus unlensed correlation functions
  std::vector<double> dxi(N, 0.0), dxip(N, 0.0), dxim(N, 0.0), dxix(N, 0.0);
  #pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < N; i++) {
    const double m = mu[i], s2 = sigma2[i], C2 = Cgl2[i], C0 = Cgl[i];
    std::vector<double> d00, d1m1, d2m2, d11, d20, d22, d31, d3m1, d3m3, d40, d4m2, d4m4, d33;
    wigner_d(0, 0, m, L, d00);
    wigner_d(1, -1, m, L, d1m1);
    wigner_d(2, -2, m, L, d2m2);
    if (opt.c0_order >= 1 || pol) wigner_d(1, 1, m, L, d11);
    if (pol) {
      wigner_d(2, 0, m, L, d20);
      wigner_d(2, 2, m, L, d22);
      wigner_d(3, 1, m, L, d31);
      wigner_d(3, -1, m, L, d3m1);
      wigner_d(3, -3, m, L, d3m3);
      wigner_d(4, 0, m, L, d40);
      wigner_d(4, -2, m, L, d4m2);
      wigner_d(4, -4, m, L, d4m4);
      if (opt.c0_order >= 1) wigner_d(3, 3, m, L, d33);
    }
    double xi = 0.0, xip = 0.0, xim = 0.0, xix = 0.0;
    for (int l = 2; l <= L; l++) {
      const double ll = l * (l + 1.0), fac = ll / 4.0, fac2 = 1.0 / ll;
      const double fac1 = (2.0 * l + 1.0) / (4.0 * M_PI);
      const double X000 = std::exp(-fac * s2);
      const double Xp000 = -fac * X000;
      const double sqrt1 = std::sqrt((l + 2.0) * (l + 1.0) * l * (l - 1.0));
      const double X220 = 0.25 * sqrt1 * X000 * std::exp(0.5 * s2);

      // TT
      double lens = X000 * X000 * d00[l];
      if (opt.c2_order >= 1) lens += Xp000 * Xp000 * d1m1[l] * C2 * 8.0 * fac2;
      if (opt.c2_order >= 2) lens += (Xp000 * Xp000 * d00[l] + X220 * X220 * d2m2[l]) * C2 * C2;
      if (opt.c0_order >= 1) lens += (2.0 * X000 * Xp000 * d00[l] + 8.0 * fac2 * Xp000 * Xp000 * d11[l]) * C0;
      xi += fac1 * cl_tt[l] * (lens - d00[l]);

      if (!pol) continue;
      const double X022 = X000 * std::exp(s2);                     // exp(-(fac - 1) sigma^2)
      const double Xp022 = -(fac - 1.0) * X022;
      const double sqrt2 = std::sqrt((l + 2.0) * (l - 1.0));
      const double sqrt3 = std::sqrt(std::max(0.0, (l + 3.0) * (l - 2.0)));
      const double sqrt4 = std::sqrt(std::max(0.0, (l + 4.0) * (l + 3.0) * (l - 2.0) * (l - 3.0)));
      const double X242 = 0.25 * sqrt4 * X000 * std::exp(2.5 * s2);
      const double X121 = -0.5 * sqrt2 * X000 * std::exp(2.0 / 3.0 * s2);
      const double X132 = -0.5 * sqrt3 * X000 * std::exp(5.0 / 3.0 * s2);
      const double isq = 1.0 / std::sqrt(ll);

      // TE
      double lx = X022 * X000 * d20[l];
      if (opt.c2_order >= 1) lx += C2 * 2.0 * Xp000 * isq * (X121 * d11[l] + X132 * d3m1[l]);
      if (opt.c2_order >= 2)
        lx += 0.5 * C2 * C2 * ((2.0 * Xp022 * Xp000 + X220 * X220) * d20[l] + X220 * X242 * d4m2[l]);
      if (opt.c0_order >= 1)
        lx += ((X022 * Xp000 + Xp022 * X000) * d20[l] + 2.0 * isq * (Xp000 * X132 * d31[l] + X121 * Xp000 * d1m1[l])) * C0;
      xix += fac1 * cl_te[l] * (lx - d20[l]);

      // EE +- BB
      double lp = X022 * X022 * d22[l], lm = X022 * X022 * d2m2[l];
      if (opt.c2_order >= 1) {
        lp += 2.0 * C2 * X132 * X121 * d31[l];
        lm += C2 * (X121 * X121 * d1m1[l] + X132 * X132 * d3m3[l]);
      }
      if (opt.c2_order >= 2) {
        lp += C2 * C2 * (Xp022 * Xp022 * d22[l] + X242 * X220 * d40[l]);
        lm += 0.5 * C2 * C2 * (2.0 * Xp022 * Xp022 * d2m2[l] + X220 * X220 * d00[l] + X242 * X242 * d4m4[l]);
      }
      if (opt.c0_order >= 1) {
        lp += (2.0 * X022 * Xp022 * d22[l] + X132 * X132 * d33[l] + X121 * X121 * d11[l]) * C0;
        lm += (2.0 * X022 * Xp022 * d2m2[l] + 2.0 * X121 * X132 * d3m1[l]) * C0;
      }
      xip += fac1 * (cl_ee[l] + bb(l)) * (lp - d22[l]);
      xim += fac1 * (cl_ee[l] - bb(l)) * (lm - d2m2[l]);
    }
    dxi[i] = xi;
    dxip[i] = xip;
    dxim[i] = xim;
    dxix[i] = xix;
  }

  // Transform back: Delta C_l = 2 pi int dmu Delta xi(mu) d^l(mu)
  std::vector<double> dtt(lmax_out + 1, 0.0), dp(lmax_out + 1, 0.0), dm(lmax_out + 1, 0.0), dte(lmax_out + 1, 0.0);
  #pragma omp parallel
  {
    std::vector<double> tt_loc(lmax_out + 1, 0.0), p_loc(lmax_out + 1, 0.0), m_loc(lmax_out + 1, 0.0),
                        te_loc(lmax_out + 1, 0.0);
    std::vector<double> d00, d22, d2m2, d20;
    #pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      wigner_d(0, 0, mu[i], lmax_out, d00);
      if (pol) {
        wigner_d(2, 2, mu[i], lmax_out, d22);
        wigner_d(2, -2, mu[i], lmax_out, d2m2);
        wigner_d(2, 0, mu[i], lmax_out, d20);
      }
      for (int l = 2; l <= lmax_out; l++) {
        tt_loc[l] += wmu[i] * dxi[i] * d00[l];
        if (pol) {
          p_loc[l] += wmu[i] * dxip[i] * d22[l];
          m_loc[l] += wmu[i] * dxim[i] * d2m2[l];
          te_loc[l] += wmu[i] * dxix[i] * d20[l];
        }
      }
    }
    #pragma omp critical
    for (int l = 0; l <= lmax_out; l++) {
      dtt[l] += tt_loc[l];
      dp[l] += p_loc[l];
      dm[l] += m_loc[l];
      dte[l] += te_loc[l];
    }
  }

  Result r;
  r.tt.assign(lmax_out + 1, 0.0);
  if (pol) {
    r.ee.assign(lmax_out + 1, 0.0);
    r.te.assign(lmax_out + 1, 0.0);
    r.bb.assign(lmax_out + 1, 0.0);
  }
  for (int l = 2; l <= lmax_out; l++) {
    r.tt[l] = cl_tt[l] + 2.0 * M_PI * dtt[l];
    if (pol) {
      const double P = 2.0 * M_PI * dp[l], M = 2.0 * M_PI * dm[l];
      r.ee[l] = cl_ee[l] + 0.5 * (P + M);
      r.bb[l] = bb(l) + 0.5 * (P - M);
      r.te[l] = cl_te[l] + 2.0 * M_PI * dte[l];
    }
  }
  return r;
}

}  // namespace Lensing

} // namespace FML::COSMOLOGY
