#include <iomanip>
#include <FML/Emulator/EuclidEmulator2.h>
        
// The EuclidEmulator2 code was written by Mischa Knabenhans 
// and was taken from https://github.com/miknab/EuclidEmulator2
// See headerfile EuclidEmulator2.h for more info
// For use cite the EuclidEmulator2 paper: https://arxiv.org/abs/2010.11288

using EuclidEmulator2 = FML::EMULATOR::EUCLIDEMULATOR2::EuclidEmulator;
using EuclidEmulator2Cosmology = FML::EMULATOR::EUCLIDEMULATOR2::Cosmology;

int main() {

    // The euclidemulator needs the EuclidEmulator2.dat file, if this is not
    // in the right folder set the path to it here. This path is computed automatically
    // at compiletime so as long as the file is in the same folder as EuclidEmulator2.h 
    // all should be good
    // FML::EMULATOR::EUCLIDEMULATOR2::set_path_to_ee2_data("../EuclidEmulator2.dat");

    // Set cosmological parameters
    double h = 0.67, OmegaB = 0.05, OmegaM = 0.3, Sum_m_nu = 0.15, n_s = 1.0, w_0 = -1.0, w_a = 0.0, A_s = 2e-9;

    // Check if cosmology is within bounds and the datafile exists
    EuclidEmulator2Cosmology ee2cosmo(OmegaB, OmegaM, Sum_m_nu, n_s, h, w_0, w_a, A_s);
    if (not ee2cosmo.is_good_to_use()) {
        ee2cosmo.print_errors();
        throw std::runtime_error("Cannot run the EuclidEmulator2 due to errors");
    }

    // Set up the emulator
    EuclidEmulator2 ee2(ee2cosmo);

    // Call it for a given redshift to get the boost B = P(k) / P_linear(k)
    double redshift = 0.0;
    auto result = ee2.compute_boost(redshift);
    auto k = result.first;
    auto boost = result.second;
    for (size_t i = 0; i < k.size(); i++)
        std::cout << std::setw(15) << k[i] << " " << std::setw(15) << boost[i] << "\n";
    return 0;
}
