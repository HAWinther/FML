// Second translation unit that includes GaussianRandomField.h and DisplacementFields.h (see test_lpt.cpp): the
// headers (and the headers they include) must not define non-inline functions or variables (e.g. an explicit
// template specialisation without inline), otherwise linking fails with multiple definitions
#include <FML/LPT/DisplacementFields.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>

int lpt_odr_check() { return 1; }
