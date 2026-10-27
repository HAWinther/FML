#include <FML/Cosmology/Models/Quintessence.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

QuintessenceSpecies::QuintessenceSpecies(double lambda_in, const std::vector<std::shared_ptr<const Species>>& others_in,
                                         double OmegaK_in, double Omega_phi)
  : ScalarFieldSpecies(others_in, OmegaK_in), lambda(lambda_in) {
  if (!(Omega_phi > 0.0)) throw std::invalid_argument("quintessence: need Omega_phi > 0");
  perturbed = lambda != 0.0;
  // Shooting on ln V0: rho_phi(0) increases with V0. Bracket, then Brent.
  auto f = [&](double lnV0) { return solve_V0(std::exp(lnV0)) / Omega_phi - 1.0; };
  double lo = std::log(Omega_phi), hi = lo;
  if (f(lo) > 0.0) throw std::runtime_error("quintessence: rho_phi(0) > V0 (should not happen)");
  int n = 0;
  while (f(hi) < 0.0) {
    hi += 1.0;
    if (++n > 60) throw std::runtime_error("quintessence: cannot reach Omega_phi (lambda too large?)");
  }
  const double lnV0 = (f(lo) == 0.0) ? lo : Utils::find_root(f, lo, hi, 1e-14, 1e-14);
  solve_V0(std::exp(lnV0));
}

double QuintessenceSpecies::solve_V0(double V0_in) {
  V0 = V0_in;
  solve(0.0, 14001, 10.0);
  return rho(0.0);
}

std::string QuintessenceSpecies::info() const {
  std::ostringstream os;
  os << "quintessence (V = V0 exp(-lambda phi), lambda = " << lambda << ", V0 = " << V0 << " rho_c0; Omega = "
     << rho(0.0) << ", w0 = " << p(0.0) / rho(0.0) << ", phi0 = " << phi(0.0) << " M_pl)";
  return os.str();
}

} // namespace FML::COSMOLOGY
