#include <FML/Cosmology/Spectra/HypersphericalBessel.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace FML::COSMOLOGY {

HypersphericalBessel::HypersphericalBessel(int sign, double nu, int lmax_hint) : sgn(sign), nu_(nu) {
  if (sign != -1 && sign != 0 && sign != 1) throw std::invalid_argument("HypersphericalBessel: sign must be -1, 0 or 1");
  if (!(nu > 0.0)) throw std::invalid_argument("HypersphericalBessel: need nu > 0");
  if (sign == 1 && (std::fabs(nu - std::round(nu)) > 1e-9 || nu < 2.5))
    throw std::invalid_argument("HypersphericalBessel: closed universe needs integer nu >= 3, got " + std::to_string(nu));
  if (sign == 1) nu_ = std::round(nu);
  if (lmax_hint > 0) {
    const int n = lmax_hint + 40 + int(std::sqrt(60.0 * lmax_hint)) + 2;
    coef_tab.resize(n);
    inv_coef_tab.resize(n);
    for (int l = 0; l < n; l++) {
      coef_tab[l] = coef(l);
      inv_coef_tab[l] = coef_tab[l] > 0.0 ? 1.0 / coef_tab[l] : 0.0;
    }
  }
}

double HypersphericalBessel::sinK(double chi) const {
  return sgn == 1 ? std::sin(chi) : (sgn == -1 ? std::sinh(chi) : chi);
}

double HypersphericalBessel::cotK(double chi) const {
  return sgn == 1 ? 1.0 / std::tan(chi) : (sgn == -1 ? 1.0 / std::tanh(chi) : 1.0 / chi);
}

double HypersphericalBessel::coef(int l) const {
  const double c2 = nu_ * nu_ - sgn * double(l) * double(l);
  return c2 > 0.0 ? std::sqrt(c2) : 0.0;
}

double HypersphericalBessel::turning_point(double chi) const {
  const double s = sinK(chi);
  return -0.5 + std::sqrt(0.25 + (nu_ * nu_ - sgn) * s * s);
}

void HypersphericalBessel::column(double chi, int lmax, std::vector<double>& out) const {
  if (lmax < 0) throw std::invalid_argument("HypersphericalBessel::column: lmax < 0");
  out.assign(lmax + 1, 0.0);
  if (chi < 0.0 || (sgn == 1 && chi >= M_PI))
    throw std::invalid_argument("HypersphericalBessel::column: chi out of range: " + std::to_string(chi));
  if (chi == 0.0) { out[0] = 1.0; return; }

  // Closed universe: Phi_ell = 0 for ell >= nu
  const int lmax_eff = (sgn == 1) ? std::min(lmax, int(nu_) - 1) : lmax;
  const double sK = sinK(chi), cK = cotK(chi);
  const double snu = std::sin(nu_ * chi), cnu = std::cos(nu_ * chi);
  out[0] = snu / (nu_ * sK);
  if (lmax_eff <= 0) return;

  // Upward recurrence in the oscillating region ell <= ell_tp (stable there)
  const int lf = std::min(lmax_eff, int(std::floor(turning_point(chi))));
  if (lf >= 1) {
    out[1] = (snu * cK / nu_ - cnu) / (sK * coef(1));
    for (int l = 1; l < lf; l++)
      out[l + 1] = ((2.0 * l + 1.0) * cK * out[l] - coef_fast(l) * out[l - 1]) * inv_coef_fast(l + 1);
  }
  if (lf >= lmax_eff) return;

  // Miller's downward recurrence from well above lmax_eff (exact start at nu - 1 when closed)
  int Ls = lmax_eff + 30 + int(std::sqrt(60.0 * lmax_eff));
  if (sgn == 1) Ls = std::min(Ls, int(nu_) - 1);
  const int lm = std::max(lf - 1, 0);  // lowest ell needed for matching
  thread_local std::vector<double> tmp;
  tmp.assign(lmax_eff + 1, 0.0);
  double p_next = 0.0, p = 1e-200;     // Phi_{l+1}, Phi_l (unnormalised)
  if (Ls <= lmax_eff) tmp[Ls] = p;
  for (int l = Ls; l > lm; l--) {
    const double p_prev = ((2.0 * l + 1.0) * cK * p - coef_fast(l + 1) * p_next) * inv_coef_fast(l);
    p_next = p;
    p = p_prev;
    if (l - 1 <= lmax_eff) tmp[l - 1] = p;
    if (std::fabs(p) > 1e200) {  // rescale to avoid overflow (deep in the decaying region)
      p *= 1e-200;
      p_next *= 1e-200;
      for (int j = l - 1; j <= std::min(lmax_eff, Ls); j++) tmp[j] *= 1e-200;
    }
  }

  // Match to the upward solution at the larger of Phi_lf, Phi_{lf-1} (or to Phi_0)
  int m = 0;
  if (lf >= 1) m = (std::fabs(out[lf]) >= std::fabs(out[lf - 1])) ? lf : lf - 1;
  if (tmp[m] == 0.0) return;  // cannot happen for a valid start; leave zeros
  const double scale = out[m] / tmp[m];
  for (int l = std::max(lf, 0) + 1; l <= lmax_eff; l++) out[l] = scale * tmp[l];
}

void HypersphericalBessel::column_with_derivative(double chi, int lmax, std::vector<double>& phi,
                                                  std::vector<double>& dphi) const {
  column(chi, lmax + 1, phi);
  dphi.assign(lmax + 1, 0.0);
  if (chi == 0.0) {  // only Phi_1 has a nonzero slope at the origin
    if (lmax >= 1) dphi[1] = coef(1) / 3.0;
    phi.resize(lmax + 1);
    return;
  }
  const double cK = cotK(chi);
  for (int l = 0; l <= lmax; l++) dphi[l] = l * cK * phi[l] - coef_fast(l + 1) * phi[l + 1];
  phi.resize(lmax + 1);
}

double HypersphericalBessel::operator()(int ell, double chi) const {
  std::vector<double> col;
  column(chi, ell, col);
  return col[ell];
}

} // namespace FML::COSMOLOGY
