#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>
#include <FML/Cosmology/RecombinationHistory/RecombinationHistory.h>

int main(){
  
  // Parameter container
  FML::UTILS::ParameterMap p;
  auto&& param_map = p.get_map();

  // Add cosmological parameters
  param_map["CosmologyLabel"] = std::string("LCDM");
  param_map["PhysicalParameters"] = true;
  param_map["OmegaBh2"]           = 0.0245;
  param_map["OmegaCDMh2"]         = 0.10976;
  param_map["OmegaKh2"]           = 0.0;
  param_map["OmegaLambdah2"]      = 0.35574;
  param_map["TCMB"]               = 2.7255 * FML::COSMOLOGY::Constants.K;
  param_map["h"]                  = 0.7;
  param_map["Neff"]               = 3.046;
 
  // Add recombination parameters
  param_map["userecfast"]           = false;
  param_map["RecFudgeFactor"]       = 1.14;
  param_map["Yp"]                   = 0.24;
  param_map["reionization"]         = true;
  param_map["helium_reionization"]  = true;
  param_map["z_reion"]              = 11.0;
  param_map["delta_z_reion"]        = 0.5;
  param_map["z_helium_reion"]       = 3.5;
  param_map["delta_z_helium_reion"] = 0.5;
  param_map["pert_x_initial"]       = -15.0;
  p.info();

  // Set up the cosmology and do all the solving and make splines
  auto lcdm = std::make_shared<FML::COSMOLOGY::BackgroundCosmology>(p);
  lcdm->solve();

  // Solve the recombination history (Saha+Peebles)
  auto rec = std::make_shared<FML::COSMOLOGY::RecombinationHistory>(lcdm, p);
  rec->solve();
}
