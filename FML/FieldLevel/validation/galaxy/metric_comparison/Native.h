#ifndef FML_VALIDATION_FROZEN_METRIC_NATIVE_H
#define FML_VALIDATION_FROZEN_METRIC_NATIVE_H
#include <FML/FieldLevel/validation/galaxy/gauss_newton/Metric.h>
#include <FML/FieldLevel/validation/galaxy/integrator_comparison/FieldHMC.h>
#include <FML/FieldLevel/EulerianCheckpoint.h>
namespace GN_FIELD {
    inline GN::Metric metric;
    inline bool legacy = false;
    inline std::string metric_identity;
    inline void load(const std::filesystem::path & path) {
        namespace CP = FML::FIELDLEVEL::FRAMEWORK::CHECKPOINT;
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            throw std::invalid_argument("Missing frozen field metric");
        const auto end = input.tellg();
        if (end < 0 || std::uintmax_t(end) > 65536)
            throw std::invalid_argument("Oversized frozen field metric");
        CP::Codec::Bytes raw(static_cast<size_t>(end));
        input.seekg(0);
        input.read(reinterpret_cast<char *>(raw.data()), std::streamsize(raw.size()));
        if (!input)
            throw std::invalid_argument("Cannot read frozen field metric");
        const std::string text(raw.begin(), raw.end());
        legacy = text == "FML_GN_LEGACY_1\n";
        if (!legacy)
            metric = GN::Metric::read(path.string());
        metric_identity = "FML frozen conditional GN 1:" + std::to_string(CP::Codec::checksum(raw));
    }
    inline const std::string & identity() {
        if (metric_identity.empty())
            throw std::invalid_argument("Missing frozen field metric");
        return metric_identity;
    }
    template <class Target>
    std::pair<FML::MCMC::HMCResult, uint64_t> counted(const Target & target,
                                                      const FML::FIELDLEVEL::Field & initial,
                                                      const std::vector<double> & parameters,
                                                      const std::vector<double> & frozen,
                                                      FML::MCMC::HMCOptions options) {
        if (legacy)
            return FIELD_HMC::counted(target, initial, parameters, frozen, options);
        namespace FF = FML::FIELDLEVEL::FRAMEWORK;
        using Model = std::decay_t<decltype(target.model())>;
        const auto & model = target.model();
        FF::Execution<Model>::validate(model, initial, parameters);
        FF::Execution<Model>::hmc_controls(model, false, options);
        const auto & noise = target.noise_policy();
        if (target.field_prior() != FF::InitialFieldPrior::WhitenedGaussian ||
            noise.mode != FF::GaussianNoise::Mode::Fixed || initial.size() != metric.dimension ||
            !model.in_support(parameters))
            throw std::invalid_argument("Unsupported frozen metric target");
        const double beta = std::exp(std::log(1000.) - noise.fixed_log_power);
        if (!std::isfinite(beta) || !(beta > 0) || beta > 1)
            throw std::invalid_argument("Invalid frozen metric temperature");
        options.lower.clear();
        options.upper.clear();
        auto work = model.workspace();
        uint64_t calls = 0;
        FML::MCMC::HMCMass mass;
        mass.n_custom = initial.size();
        mass.apply_sqrt = [&](auto & v) { v = metric.apply(v, beta, .5); };
        mass.apply_inverse = [&](auto & v) { v = metric.apply(v, beta, -1.); };
        auto potential = [&](const auto & v, auto & g) {
            ++calls;
            std::vector<double> gp;
            return target.evaluate(v, parameters, work, g, gp);
        };
        return {
            FML::MCMC::run_hmc(potential, initial, std::move(mass), options, nullptr, FF::Execution<Model>::hmc(model)),
            calls};
    }
} // namespace GN_FIELD
#endif
