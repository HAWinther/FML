#ifndef FML_HYBRIDBIASOPERATORS_HEADER
#define FML_HYBRIDBIASOPERATORS_HEADER

#include <string>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {
            /// The Lagrangian bias operators in the order they are stored (see HybridBiasSpectra.h)
            enum Operator : int { ONE = 0, DELTA = 1, DELTA2 = 2, S2 = 3, NABLA2DELTA = 4 };
            inline const std::vector<std::string> & operator_names() {
                static const std::vector<std::string> names{"1", "delta", "delta2", "s2", "nabla2delta"};
                return names;
            }
            constexpr int max_operators = 5;
        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
