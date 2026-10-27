#ifndef FML_COSMOLOGY_MODELS_QUINTESSENCE_H
#define FML_COSMOLOGY_MODELS_QUINTESSENCE_H

// =============================================================================
// Quintessence: a scalar field (models/ScalarField.h) with the exponential potential
//   V(phi) = 3 H0^2 M_pl^2 v(phi),   v = V0 exp(-lambda phi)
// starting at rest at phi = 0 deep in the radiation era (thawing: Hubble friction freezes it until the
// dark energy era). V0 is found by shooting so that Omega_phi today closes the universe (as Omega_Lambda
// in LCDM). lambda = 0 is LCDM (no perturbations).
// =============================================================================

#include <FML/Cosmology/Models/ScalarField.h>

namespace FML::COSMOLOGY {

class QuintessenceSpecies : public ScalarFieldSpecies {
  public:
    // others: all other species of the model; Omega_phi: the density today it must have
    QuintessenceSpecies(double lambda, const std::vector<std::shared_ptr<const Species>>& others, double OmegaK,
                        double Omega_phi);
    std::string name() const override { return "quintessence"; }
    std::string info() const override;

    double v(double phi) const override { return V0 * std::exp(-lambda * phi); }
    double v_phi(double phi) const override { return -lambda * v(phi); }
    double v_phiphi(double phi) const override { return lambda * lambda * v(phi); }
    double V0_value() const { return V0; }
    using ScalarFieldSpecies::E2;

  private:
    double lambda, V0 = 0.0;
    // Integrate with amplitude V0 and return rho_phi(0)
    double solve_V0(double V0);
};

} // namespace FML::COSMOLOGY

#endif
