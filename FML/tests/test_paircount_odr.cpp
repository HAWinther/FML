// Second translation unit that includes PairCount.h (see test_paircount.cpp): the header must not define non-inline
// functions or variables, otherwise linking fails with multiple definitions
#include <FML/PairCounting/PairCount.h>

int paircount_odr_check() {
    return FML::CORRELATIONFUNCTIONS::PAIRCOUNTS::CorrelationFunctionEstimator(2.0, 1.0, 1.0, 1.0, "LZ") == 1.0 ? 1 : 0;
}
