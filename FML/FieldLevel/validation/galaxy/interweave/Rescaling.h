#ifndef FML_VALIDATION_AMPLITUDE_FIELD_RESCALING_H
#define FML_VALIDATION_AMPLITUDE_FIELD_RESCALING_H
#include <FML/FieldLevel/Framework.h>
// Serial validation kernel. Every local stored real coordinate is rescaled.
// A distributed version needs the GLOBAL dimension and collective validation.
namespace RESCALING {
    struct Proposal {
        FML::FIELDLEVEL::Field white;
        std::vector<double> parameters;
        double log_jacobian;
    };
    inline Proposal propose(const FML::FIELDLEVEL::Field & white,
                            const std::vector<double> & parameters,
                            size_t amplitude_index,
                            double alpha_new) {
        if (white.empty() || amplitude_index >= parameters.size() || !(parameters[amplitude_index] > 0) ||
            !std::isfinite(parameters[amplitude_index]) || !(alpha_new > 0) || !std::isfinite(alpha_new))
            throw std::invalid_argument("Invalid amplitude/field rescaling controls");
        const double alpha = parameters[amplitude_index], ratio = alpha / alpha_new;
        Proposal result{white, parameters, double(white.size()) * (std::log(alpha) - std::log(alpha_new))};
        if (!(ratio > 0) || !std::isfinite(ratio) || !std::isfinite(result.log_jacobian))
            throw std::invalid_argument("Unrepresentable amplitude/field rescaling");
        for (double & x : result.white) {
            x *= ratio;
            if (!std::isfinite(x))
                throw std::invalid_argument("Nonfinite rescaled field");
        }
        result.parameters[amplitude_index] = alpha_new;
        return result;
    }
    // Reflection preserves symmetry for a symmetric additive increment on [lower,upper].
    inline double reflected(double current, double lower, double upper, double increment) {
        const double span = upper - lower;
        if (!std::isfinite(current) || !(lower > 0) || !(span > 0) || !std::isfinite(2 * span) || current < lower ||
            current > upper || !std::isfinite(increment))
            throw std::invalid_argument("Invalid reflected amplitude proposal");
        double x = (current - lower) + increment;
        if (!std::isfinite(x))
            throw std::invalid_argument("Unrepresentable reflected amplitude proposal");
        x = std::fmod(x, 2 * span);
        if (x < 0)
            x += 2 * span;
        return lower + (x <= span ? x : 2 * span - x);
    }
    inline double log_acceptance(double current_u, double proposed_u, double log_jacobian) {
        if (!std::isfinite(current_u) || !std::isfinite(log_jacobian))
            throw std::invalid_argument("Invalid rescaling current target/Jacobian");
        if (!std::isfinite(proposed_u))
            return -HUGE_VAL;
        return current_u - proposed_u + log_jacobian;
    }
} // namespace RESCALING
#endif
