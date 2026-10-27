#include <FML/Cosmology/Models/DGP.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

DGPGravity::DGPGravity(double Omega_rc_in, int branch_in, bool lcdm_background)
  : MuSigmaGravity(0.0, 0.0), Omega_rc(Omega_rc_in), sq(std::sqrt(Omega_rc_in)), branch(branch_in),
    lcdm_bg(lcdm_background) {
  if (!(Omega_rc > 0.0)) throw std::invalid_argument("DGP: need Omega_rc > 0");
  if (branch != 1 && branch != -1) throw std::invalid_argument("DGP: branch must be +1 (self-accelerating) or -1 (normal)");
}

std::string DGPGravity::info() const {
  std::ostringstream os;
  os << name() << " (Omega_rc = " << Omega_rc << ", H0 r_c / c = " << 0.5 / sq
     << (lcdm_bg ? ", LCDM background" : "") << ")";
  return os.str();
}

void DGPGravity::init_background(const BackgroundCosmology& bg) {
  if (bg.get_params().OmegaK != 0.0) throw std::invalid_argument(name() + ": only implemented for a flat universe");
  if (std::fabs(E2(0.0, bg) - 1.0) > 1e-10) throw std::invalid_argument(name() + ": E(0) != 1 (inconsistent Omega_Lambda)");
}

// E = eps sq + s, s = sqrt(Omega_rc + rho):  E' = rho'/(2 s),  E'' = rho''/(2 s) - rho'^2/(4 s^3)
void DGPGravity::E_derivs(double x, const BackgroundCosmology& bg, double& E, double& dE, double& ddE) const {
  if (lcdm_bg) {
    const double e2 = GeneralRelativity::E2(x, bg), de2 = GeneralRelativity::dE2dx(x, bg);
    E = std::sqrt(e2);
    dE = de2 / (2.0 * E);
    ddE = GeneralRelativity::ddE2ddx(x, bg) / (2.0 * E) - de2 * de2 / (4.0 * E * E * E);
    return;
  }
  const double rho = bg.rho_species(x), drho = bg.drho_species(x), ddrho = bg.ddrho_species(x);
  const double s = std::sqrt(Omega_rc + rho);
  E = branch * sq + s;
  dE = drho / (2.0 * s);
  ddE = ddrho / (2.0 * s) - drho * drho / (4.0 * s * s * s);
}

double DGPGravity::E2(double x, const BackgroundCosmology& bg) const {
  if (lcdm_bg) return GeneralRelativity::E2(x, bg);
  const double s = std::sqrt(Omega_rc + bg.rho_species(x)), E = branch * sq + s;
  return E * E;
}
double DGPGravity::dE2dx(double x, const BackgroundCosmology& bg) const {
  if (lcdm_bg) return GeneralRelativity::dE2dx(x, bg);
  double E, dE, ddE;
  E_derivs(x, bg, E, dE, ddE);
  return 2.0 * E * dE;
}
double DGPGravity::ddE2ddx(double x, const BackgroundCosmology& bg) const {
  if (lcdm_bg) return GeneralRelativity::ddE2ddx(x, bg);
  double E, dE, ddE;
  E_derivs(x, bg, E, dE, ddE);
  return 2.0 * (dE * dE + E * ddE);
}

double DGPGravity::beta(double x, const BackgroundCosmology& bg) const {
  double E, dE, ddE;
  E_derivs(x, bg, E, dE, ddE);
  return 1.0 - branch * (E + dE / 3.0) / sq;   // E (1 + dlnE/dx / 3) = E + E'/3
}

// mu = 1 + 1/(3 beta), Sigma = 1, so 2 Sigma - mu = 1 - 1/(3 beta) and its derivative is beta'/(3 beta^2)
DGPGravity::MuSigma DGPGravity::mu_sigma(double x, double /*k*/, const BackgroundCosmology& bg) const {
  double E, dE, ddE;
  E_derivs(x, bg, E, dE, ddE);
  const double b = 1.0 - branch * (E + dE / 3.0) / sq;
  const double db = -branch * (dE + ddE / 3.0) / sq;
  if (b == 0.0) throw std::runtime_error(name() + ": beta = 0");
  MuSigma m;
  m.mu = 1.0 + 1.0 / (3.0 * b);
  m.Sigma = 1.0;
  m.dg = db / (3.0 * b * b);
  return m;
}

} // namespace FML::COSMOLOGY
