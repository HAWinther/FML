#ifndef FML_COSMOLOGY_BACKGROUND_BBN_H
#define FML_COSMOLOGY_BACKGROUND_BBN_H

// =============================================================================
// Primordial helium from standard BBN (as CLASS with YHe = BBN): Y_He(omega_b, N_eff) interpolated (bicubic) in
// the PArthENoPE table external/bbn/sBBN_2025.dat (the CLASS default), with Delta N = N_eff - 3.044 as defined in
// the table (CLASS's code uses N_eff - 3.046; the difference is 3e-5 in Y_He). Y_He is the mass fraction used by
// RecombinationHistory. The table is read once (relative to the working directory, i.e. the project root, or
// BBN::set_table_file) and covers 0.0073 <= omega_b <= 0.0329 and -3 <= Delta N <= 3.
// =============================================================================

#include <string>

namespace FML::COSMOLOGY {

    namespace BBN {
        double helium_fraction(double omega_b, double N_eff); // throws outside the table
        void set_table_file(const std::string & file);        // before the first call
    } // namespace BBN

} // namespace FML::COSMOLOGY

#endif
