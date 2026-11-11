// Second translation unit that includes EuclidEmulator2.h and FofrFittingFunction.h (see test_emulator.cpp): the header must not define
// non-inline functions or variables, otherwise linking fails with multiple definitions
#include <FML/Emulator/EuclidEmulator2.h>
#include <FML/Emulator/FofrFittingFunction.h>

int emulator_odr_check() {
    return FML::EMULATOR::EUCLIDEMULATOR2::does_ee2_datafile_exist() and
                   FML::EMULATOR::FOFRFITTINGFUNCTION::nonlinear_boost(0.0, 1e-5, 1.0) > 1.0 ?
               1 :
               0;
}
