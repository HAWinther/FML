#ifndef FML_FIELDLEVEL_STABILITY_PROBE_H
#define FML_FIELDLEVEL_STABILITY_PROBE_H
#include <FML/FieldLevel/Framework.h>
namespace STABILITY {
    struct Result {
        FML::MCMC::HMCResult hmc;
        double energy_error = HUGE_VAL;
        size_t evaluations = 0;
    };
    // Observe the real HMC kernel through its existing potential/mass callbacks.
    // The mass parameters stay fixed throughout this conditional field trajectory.
    template <class Target, class Work>
    Result trajectory(const Target & target,
                      const FML::FIELDLEVEL::Field & white,
                      const std::vector<double> & parameters,
                      const std::vector<double> & metric,
                      const FML::MCMC::HMCOptions & options,
                      Work & work) {
        namespace FF = FML::FIELDLEVEL::FRAMEWORK;
        Result result;
        double initial_U = 0, last_U = 0, initial_K = 0, last_K = 0;
        bool first_velocity = true;
        FML::MCMC::HMCMass mass;
        mass.n_custom = white.size();
        mass.apply_sqrt = [&](auto & v) { v = target.apply_field_metric(v, metric, work, .5); };
        mass.apply_inverse = [&](auto & v) {
            const auto momentum = v;
            v = target.apply_field_metric(v, metric, work, -1.);
            last_K = 0;
            for (size_t j = 0; j < v.size(); ++j)
                last_K += momentum[j] * v[j];
            last_K *= .5;
            if (first_velocity) {
                initial_K = last_K;
                first_velocity = false;
            }
        };
        auto potential = [&](const auto & x, auto & gradient) {
            std::vector<double> gp;
            last_U = target.evaluate(x, parameters, work, gradient, gp);
            if (!result.evaluations)
                initial_U = last_U;
            ++result.evaluations;
            return last_U;
        };
        using Model = std::decay_t<decltype(target.model())>;
        result.hmc = FML::MCMC::run_hmc(potential,
                                        white,
                                        std::move(mass),
                                        options,
                                        nullptr,
                                        FF::Execution<Model>::hmc(target.model()));
        result.energy_error = std::isfinite(last_U) ? (last_U + last_K) - (initial_U + initial_K) : HUGE_VAL;
        const double expected = std::isfinite(result.energy_error) ? std::min(1., std::exp(-result.energy_error)) : 0.;
        if (std::abs(expected - result.hmc.acceptance) > 1e-11 ||
            result.hmc.divergences != long(!(result.energy_error <= options.divergence_threshold)))
            throw std::runtime_error("Instrumented energy disagrees with actual HMC decision");
        return result;
    }
} // namespace STABILITY
#endif
