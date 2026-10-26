// Second translation unit that includes ComputePowerSpectrum.h (see test_pofk.cpp): the header (and the headers it
// includes) must not define non-inline functions or variables, otherwise linking fails with multiple definitions
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>

int pofk_odr_check() {
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<3> pofk(0.0, 1.0, 2, 0);
    return pofk.n == 2 ? 1 : 0;
}
