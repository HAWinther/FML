#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>
#include <FML/Cosmology/RecombinationHistory/RecombinationHistory.h>
#include <FML/Cosmology/LinearPerturbations/Perturbations.h>

int main(){
  // Parameter container
  FML::UTILS::ParameterMap p;
  auto&& param_map = p.get_map();

  // Add cosmological parameters
  const double Kelvin = FML::COSMOLOGY::Constants.K;
  param_map["CosmologyLabel"] = std::string("LCDM");
  param_map["PhysicalParameters"] = true;            // Supply physical parameters? i.e. Omegah^2 instead of Omega's
  param_map["OmegaBh2"]           = 0.0245;
  param_map["OmegaCDMh2"]         = 0.10976;
  param_map["OmegaKh2"]           = 0.0;
  param_map["OmegaLambdah2"]      = 0.35574;
  param_map["TCMB"]               = 2.7255 * Kelvin; // CMB temperature in your temperature units
  param_map["h"]                  = 0.7;             // Hubble constant H0 / (100km/s/Mpc)
  param_map["Neff"]               = 3.046;           // Effective number of neutrinos
 
  // Add recombination parameters
  param_map["userecfast"]           = false;         // Use recfast?
  param_map["RecFudgeFactor"]       = 1.14;          // Fudgefactor in Peebles equation in Recfast
  param_map["Yp"]                   = 0.24;          // Helium abundance
  param_map["reionization"]         = true;          // Include reionization?
  param_map["z_reion"]              = 11.0;          // Reionization redshift
  param_map["delta_z_reion"]        = 0.5;           // Reionization width
  param_map["helium_reionization"]  = true;          // Double Helium reionization?
  param_map["z_helium_reion"]       = 3.5;           // Double Helium reionization redshift
  param_map["delta_z_helium_reion"] = 0.5;           // Double Helium reionization width
  
  // Add perturbation parameters
  param_map["polarization"] = true;                  // Include photon polarization?
  param_map["neutrinos"]    = true;                  // Include massless neutrinos?
  param_map["n_ell_theta"]  = 12;                    // Number of temperature multipoles to keep in the Boltzmann hierarchy
  param_map["n_ell_nu"]     = 12;                    // Number of Nu multipoles to keep in the Boltzmann hierarchy
  param_map["keta_min"]     = 0.1;                   // Integrate k from k_min*eta0 -> k_max*eta0
  param_map["keta_max"]     = 3000.0;
  
  // Accuracy settings
  param_map["pert_integration_nk_per_logint"] = 25;  // Number of k-points per logarithmic interval, 100 for high accuracy
  param_map["pert_spline_all_ells"]  = false;        // Spline all the Theta_ell(k,x), Nu_ell(k,x), etc. multipoles or just the first 0,1,2
  param_map["pert_x_initial"]        = -15.0;        // When to start the integration x = log(aini)
  param_map["pert_delta_x"]          = 0.05;         // How many points to store in the integration till today, every deltax = deltalog(a)
  p.info();

  // Set up the cosmology and do all the solving and make splines
  auto lcdm = std::make_shared<FML::COSMOLOGY::BackgroundCosmology>(p);
  lcdm->solve();

  // Solve the recombination history (Saha+Peebles)
  auto rec = std::make_shared<FML::COSMOLOGY::RecombinationHistory>(lcdm, p);
  rec->solve();
  
  // Solve the perturbations
  auto pert = std::make_shared<FML::COSMOLOGY::Perturbations>(lcdm, rec, p);
  pert->solve();
}
