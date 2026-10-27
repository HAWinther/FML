// Second translation unit that includes RamsesUtils.h (see test_ramses.cpp): the header must not define non-inline
// functions or variables, otherwise linking fails with multiple definitions
#include <FML/RamsesUtils/RamsesUtils.h>

int ramses_odr_check() { return sizeof(FML::FILEUTILS::RAMSES::ParticleFileHeader) > 0 ? 1 : 0; }
