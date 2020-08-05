#include <FML/Cosmology/BackgroundCosmology/BackgroundCosmology.h>

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
  p.info();
  
  // Set up the cosmology and do all the solving and make splines
  FML::COSMOLOGY::BackgroundCosmology lcdm(p);
  lcdm.solve();
}
