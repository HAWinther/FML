// Second translation unit that includes EuclidEmulator2.h (see test_emulator.cpp): the header must not define
// non-inline functions or variables, otherwise linking fails with multiple definitions
#include <FML/Emulator/EuclidEmulator2.h>

int emulator_odr_check() { return FML::EMULATOR::EUCLIDEMULATOR2::does_ee2_datafile_exist() ? 1 : 0; }
