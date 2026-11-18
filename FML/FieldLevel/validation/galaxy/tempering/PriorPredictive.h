#ifndef FML_VALIDATION_PRIOR_PREDICTIVE_H
#define FML_VALIDATION_PRIOR_PREDICTIVE_H
#include <FML/FieldLevel/Framework.h>
namespace MOCK {
    // Uniforms use a distinct Philox purpose and an independent coordinate counter.
    // This routine is deliberately separate from the sampler's beta=0 refresh.
    inline std::vector<double> prior_parameters(const std::vector<FML::FIELDLEVEL::FRAMEWORK::Parameter> & parameters,
                                                uint64_t seed) {
        namespace FF = FML::FIELDLEVEL::FRAMEWORK;
        std::vector<double> result;
        for (size_t j = 0; j < parameters.size(); ++j) {
            const auto & p = parameters[j];
            if (p.prior != FF::Prior::Uniform || !std::isfinite(p.lower) || !std::isfinite(p.upper) ||
                !(p.lower < p.upper) || !std::isfinite(p.upper - p.lower))
                throw std::invalid_argument("Prior-predictive scalars require finite bounded uniform priors");
            FML::RANDOM::COUNTER_BASED::Stream rng(seed, j, 0x53424353u);
            result.push_back(p.lower + (p.upper - p.lower) * rng.uniform());
        }
        return result;
    }
} // namespace MOCK
#endif
