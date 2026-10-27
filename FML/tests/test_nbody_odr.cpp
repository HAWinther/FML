// Second translation unit that includes NBody.h and PotentialGradientDecent.h (see test_nbody.cpp): the headers must
// not define non-inline functions or variables, otherwise linking fails with multiple definitions
#include <FML/NBody/NBody.h>
#include <FML/PGD/PotentialGradientDecent.h>

int nbody_odr_check() { return 1; }
