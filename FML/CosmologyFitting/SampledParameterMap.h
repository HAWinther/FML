#ifndef FML_COSMOLOGYFITTING_SAMPLEDPARAMETERMAP_H
#define FML_COSMOLOGYFITTING_SAMPLEDPARAMETERMAP_H

// =============================================================================
// From a point in the sampled parameter space to the RunSettings of ./cmb.
//
// Sampled names are parameters of ./cmb (h, OmegaB, n_s, A_s, z_reion, m_nu, model parameters, ...), nuisance
// parameters of the likelihoods (ignored here, passed to the Theory), or the aliases
//   H0            = 100 h
//   omega_b       = OmegaB h^2,  omega_cdm = OmegaCDM h^2 (physical densities, kept fixed when h changes)
//   Omega_m       sets OmegaCDM = Omega_m - OmegaB - Omega_ncdm
//   logA          = ln(10^10 A_s)
//   theta_MC_100  sets h such that Theory::theta_MC() = theta_MC_100 at fixed omega_b, omega_cdm (Brent root in
//                 h in [0.4, 1]; ~10 background solutions)
//   tau_reio      sets z_reion such that the optical depth of the reionization model
//                 (RecombinationHistory::tau_reionization_model, to z_reion + 8 delta_z_reion) is tau_reio
//                 (the full tau of the run, with the residual ionization after recombination, is slightly higher)
//   log10_bd_omega   sets the Brans-Dicke bd_omega = 10^log10_bd_omega (model = jbd)
//   inv_bd_omega     sets bd_omega = 1/inv_bd_omega (0: GR, taken as bd_omega = 1e12); a flat prior on 1/omega
//                    reaches GR at the edge of its range
// The fixed parameters are "name=value" strings for ParameterFile::read (cosmology, model, precision).
// =============================================================================

#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>

#include <set>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::FITTING {

    class SampledParameterMap {
      public:
        SampledParameterMap(std::vector<std::string> fixed,
                            std::vector<std::string> sampled,
                            std::set<std::string> nuisance = {});
        RunSettings settings(const std::vector<double> & x) const; // throws for invalid points
        static const std::set<std::string> & aliases();

      private:
        std::vector<std::string> fixed, names;
        std::set<std::string> nuis;
        bool theta = false;
    };

} // namespace FML::COSMOLOGY::FITTING

#endif
