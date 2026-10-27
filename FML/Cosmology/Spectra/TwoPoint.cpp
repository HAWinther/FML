#include <FML/Cosmology/Spectra/TwoPoint.h>
#include <FML/Cosmology/Spectra/Lensing.h>

#include <gsl/gsl_integration.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY {

namespace TwoPoint {

  double beam(int ell, double fwhm) {
    if (fwhm <= 0.0) return 1.0;
    const double sigma = fwhm / std::sqrt(8.0 * std::log(2.0));
    return std::exp(-0.5 * ell * (ell + 1.0) * sigma * sigma);
  }

  std::vector<double> angular_correlation(const std::vector<double>& cl, const std::vector<double>& theta,
                                          double fwhm, int ell_min) {
    const int L = static_cast<int>(cl.size()) - 1;
    std::vector<double> c(L + 1, 0.0), w(theta.size(), 0.0);
    for (int l = std::max(ell_min, 0); l <= L; l++) {
      const double b = beam(l, fwhm);
      c[l] = (2.0 * l + 1.0) / (4.0 * M_PI) * cl[l] * b * b;
    }
    for (size_t i = 0; i < theta.size(); i++) {
      const double mu = std::cos(theta[i]);
      double p0 = 1.0, p1 = mu, s = c[0] + (L >= 1 ? c[1] * mu : 0.0);
      for (int l = 1; l < L; l++) {   // P_{l+1} = ((2l+1) mu P_l - l P_{l-1}) / (l+1)
        const double p2 = ((2.0 * l + 1.0) * mu * p1 - l * p0) / (l + 1.0);
        s += c[l + 1] * p2;
        p0 = p1;
        p1 = p2;
      }
      w[i] = s;
    }
    return w;
  }

  CMBCorrelation cmb_correlation(const std::vector<double>& tt, const std::vector<double>& ee,
                                 const std::vector<double>& te, const std::vector<double>& bb,
                                 const std::vector<double>& theta, double fwhm) {
    const int L = static_cast<int>(tt.size()) - 1;
    auto get = [](const std::vector<double>& v, int l) { return l < static_cast<int>(v.size()) ? v[l] : 0.0; };
    CMBCorrelation out;
    out.theta = theta;
    out.TT = angular_correlation(tt, theta, fwhm, 2);
    const size_t n = theta.size();
    out.plus.assign(n, 0.0);
    out.minus.assign(n, 0.0);
    out.TQ.assign(n, 0.0);
    if (ee.empty() && te.empty()) return out;
    std::vector<double> d22, d2m2, d20;
    for (size_t i = 0; i < n; i++) {
      const double mu = std::cos(theta[i]);
      Lensing::wigner_d(2, 2, mu, L, d22);
      Lensing::wigner_d(2, -2, mu, L, d2m2);
      Lensing::wigner_d(2, 0, mu, L, d20);
      double sp = 0.0, sm = 0.0, sx = 0.0;
      for (int l = 2; l <= L; l++) {
        const double b = beam(l, fwhm), f = (2.0 * l + 1.0) / (4.0 * M_PI) * b * b;
        const double e = get(ee, l), bm = get(bb, l);
        sp += f * (e + bm) * d22[l];
        sm += f * (e - bm) * d2m2[l];
        sx += f * get(te, l) * d20[l];
      }
      out.plus[i] = sp;
      out.minus[i] = sm;
      out.TQ[i] = sx;
    }
    return out;
  }

  std::vector<double> legendre_transform(const std::function<double(double)>& w, int l_max, int n) {
    gsl_integration_glfixed_table* gl = gsl_integration_glfixed_table_alloc(n);
    std::vector<double> cl(l_max + 1, 0.0);
    for (int i = 0; i < n; i++) {
      double mu, wt;
      gsl_integration_glfixed_point(-1.0, 1.0, i, &mu, &wt, gl);
      const double f = 2.0 * M_PI * wt * w(std::acos(mu));
      double p0 = 1.0, p1 = mu;
      cl[0] += f;
      if (l_max >= 1) cl[1] += f * mu;
      for (int l = 1; l < l_max; l++) {
        const double p2 = ((2.0 * l + 1.0) * mu * p1 - l * p0) / (l + 1.0);
        cl[l + 1] += f * p2;
        p0 = p1;
        p1 = p2;
      }
    }
    gsl_integration_glfixed_table_free(gl);
    return cl;
  }

  std::vector<double> xi_of_r(const std::vector<double>& k, const std::vector<double>& P,
                              const std::vector<double>& r, double R_smooth) {
    if (k.size() != P.size() || k.size() < 2) throw std::invalid_argument("TwoPoint::xi_of_r: bad table");
    const size_t n = k.size();
    std::vector<double> f(n);   // k P(k) e^{-k^2 R^2}
    for (size_t i = 0; i < n; i++) f[i] = k[i] * P[i] * std::exp(-k[i] * k[i] * R_smooth * R_smooth);
    std::vector<double> xi(r.size());
    for (size_t j = 0; j < r.size(); j++) {
      const double rr = r[j];
      double s = 0.0;
      if (rr <= 0.0) {   // xi(0) = int dk k^2 P / (2 pi^2): trapezoid in k of k f
        for (size_t i = 0; i + 1 < n; i++) s += 0.5 * (k[i + 1] - k[i]) * (k[i] * f[i] + k[i + 1] * f[i + 1]);
        xi[j] = s / (2.0 * M_PI * M_PI);
        continue;
      }
      for (size_t i = 0; i + 1 < n; i++) {
        const double a = k[i], b = k[i + 1], h = b - a, fa = f[i], fb = f[i + 1];
        if (h * rr < 0.05) {   // Simpson on the linear interpolant (Filon's formula loses digits here)
          const double m = 0.5 * (a + b);
          s += h / 6.0 * (fa * std::sin(a * rr) + 4.0 * 0.5 * (fa + fb) * std::sin(m * rr) + fb * std::sin(b * rr));
        } else {               // exact integral of (fa + slope (k - a)) sin(k r)
          const double sl = (fb - fa) / h, ca = std::cos(a * rr), cb = std::cos(b * rr);
          s += fa * (ca - cb) / rr + sl * (-h * cb / rr + (std::sin(b * rr) - std::sin(a * rr)) / (rr * rr));
        }
      }
      xi[j] = s / (2.0 * M_PI * M_PI * rr);
    }
    return xi;
  }

  double sigma_R(const std::vector<double>& k, const std::vector<double>& P, double R) {
    auto W = [](double y) {
      if (y < 1e-3) return 1.0 - y * y / 10.0;
      return 3.0 * (std::sin(y) - y * std::cos(y)) / (y * y * y);
    };
    double s = 0.0;
    for (size_t i = 0; i + 1 < k.size(); i++) {
      auto g = [&](size_t j) { const double w = W(k[j] * R); return k[j] * k[j] * k[j] * P[j] * w * w; };
      s += 0.5 * std::log(k[i + 1] / k[i]) * (g(i) + g(i + 1));   // trapezoid in ln k
    }
    return std::sqrt(s / (2.0 * M_PI * M_PI));
  }

  RedshiftWindow RedshiftWindow::gaussian(double z0, double sigma_z, double bias) {
    RedshiftWindow w;
    w.n = [z0, sigma_z](double z) { const double u = (z - z0) / sigma_z; return std::exp(-0.5 * u * u); };
    w.z_min = std::max(z0 - 6.0 * sigma_z, 1e-4);
    w.z_max = z0 + 6.0 * sigma_z;
    w.bias = bias;
    return w;
  }

  RedshiftWindow RedshiftWindow::tophat(double z1, double z2, double sigma_edge, double bias) {
    RedshiftWindow w;
    const double s = std::sqrt(2.0) * sigma_edge;
    w.n = [z1, z2, s](double z) { return 0.5 * (std::erf((z - z1) / s) - std::erf((z - z2) / s)); };
    w.z_min = std::max(z1 - 6.0 * sigma_edge, 1e-4);
    w.z_max = z2 + 6.0 * sigma_edge;
    w.bias = bias;
    return w;
  }
}

} // namespace FML::COSMOLOGY
