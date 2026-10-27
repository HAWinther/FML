#ifndef FML_COSMOLOGY_BACKGROUND_MASSIVENEUTRINOS_H
#define FML_COSMOLOGY_BACKGROUND_MASSIVENEUTRINOS_H

// =============================================================================
// Momentum integrals for one massive neutrino species (neutrino + antineutrino, g = 2) with a
// relativistic Fermi-Dirac distribution f0(q) = 1/(e^q + 1) frozen at decoupling.
//
// q = p / (k_B T_nu) is the comoving momentum in units of the neutrino temperature, and
// M(a) = m_nu c^2 a / (k_B T_nu0) the mass in the same units, so that the comoving energy is
// eps = sqrt(q^2 + M^2). Energy density and pressure are
//   rho = (g / 2 pi^2) (k_B T_nu0 / a)^4 / (hbar c)^3 * I_rho(M),   I_rho = int q^2 eps f0 dq
//   p   = ...                                         * I_p(M),     I_p   = int q^4 / (3 eps) f0 dq
// In units of the photon density, rho_nu / rho_gamma = (15 / pi^4) (T_nu / T_gamma)^4 I_rho(M).
// I_rho(0) = 7 pi^4 / 120, so a massless species has rho_nu / rho_gamma = (7/8) (T_nu / T_gamma)^4.
//
// The integrals use Gauss-Laguerre quadrature (weight e^{-q}): 48 nodes for the background
// (relative error < 1e-7) and a few nodes for the perturbations (see Quadrature).
// =============================================================================

#include <vector>

namespace FML::COSMOLOGY {

namespace MassiveNeutrinos {

  // Nodes q_i and weights w_i such that int_0^inf F(q) f0(q) dq ~ sum_i w_i F(q_i)
  struct Quadrature {
    std::vector<double> q, w;
    explicit Quadrature(int n);
    int size() const { return static_cast<int>(q.size()); }
  };

  // d ln f0 / d ln q = -q e^q / (e^q + 1)
  double dlnf0_dlnq(double q);

  // Background integrals with the 48-node quadrature
  double I_rho(double M);              // int q^2 eps f0
  double I_p(double M);                // int q^4 / (3 eps) f0
  double dI_p_dlnM(double M);          // M dI_p/dM = -int q^4 M^2 / (3 eps^3) f0 = I_pp - I_p
  double I_pp(double M);               // pseudo-pressure int q^6 / (3 eps^3) f0 (Lesgourgues & Tram 2011)

  // M today for mass m [eV] and neutrino temperature T_nu0 [K]
  double M0(double m_eV, double T_nu0);
}

} // namespace FML::COSMOLOGY

#endif
