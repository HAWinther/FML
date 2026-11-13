#ifndef FML_HOD_ASSEMBLY_BIAS_H
#define FML_HOD_ASSEMBLY_BIAS_H
#include <FML/HOD/HOD.h>
#include <cmath>
#include <limits>

// Heaviside decorated HOD: Hearin et al. (2016), arXiv:1512.03050.
// Halotools prescription; marginal-occupation conventions and differences: README.md.
namespace FML::HOD {
    /// Heaviside decoration of central probability and MARGINAL satellite mean.
    /// At fixed mass, q is a caller-calibrated conditional percentile in [0,1].
    /// q > split is the high population; split is the low-population fraction.
    /// Population-weighted first moments are preserved. Realized counts need not be.
    struct HeavisideAssemblyBias {
        double split = .5;
        double central_strength = 0, satellite_strength = 0; // [-1,1], positive favors high q.

      private:
        // Evaluate a*b/c without avoidable overflow/underflow in intermediate products.
        static long double product_ratio(long double a, long double b, long double c) {
            if (a == 0 || b == 0)
                return 0;
            int ea, eb, ec;
            const auto ma = std::frexp(a, &ea), mb = std::frexp(b, &eb), mc = std::frexp(c, &ec);
            return std::scalbn((ma * mb) / mc, ea + eb - ec);
        }
        static double finite_result(long double value) {
            detail::require(std::isfinite(value) && value >= 0 && value <= std::numeric_limits<double>::max(),
                            "Decorated HOD mean cannot be represented as a finite double");
            return double(value);
        }

      public:
        MarginalOccupationMeans decorate_marginal(const MarginalOccupationMeans & baseline, double percentile) const {
            detail::require(std::isfinite(split) && split > 0 && split < 1 && std::isfinite(central_strength) &&
                                std::abs(central_strength) <= 1 && std::isfinite(satellite_strength) &&
                                std::abs(satellite_strength) <= 1 && std::isfinite(percentile) && percentile >= 0 &&
                                percentile <= 1,
                            "Invalid assembly-bias split, strength or conditional percentile");
            const long double p = baseline.central_probability, mu = baseline.satellite_mean;
            detail::require(std::isfinite(p) && p >= 0 && p <= 1 && std::isfinite(mu) && mu >= 0,
                            "Invalid assembly-bias baseline marginal means");
            const bool high = percentile > split;
            const long double low_fraction = split, high_fraction = 1.0L - low_fraction;
            long double probability = p, satellite = mu;
            if (central_strength != 0 && p > 0 && p < 1) {
                const bool prefers_high = central_strength > 0;
                const long double preferred_fraction = prefers_high ? high_fraction : low_fraction;
                const long double other_fraction = prefers_high ? low_fraction : high_fraction;
                // Extremal bounded occupations put centrals into the preferred population first.
                const long double preferred = p <= preferred_fraction ? p / preferred_fraction : 1;
                const long double other = p <= preferred_fraction ? 0 : (p - preferred_fraction) / other_fraction;
                const long double extreme = high == prefers_high ? preferred : other;
                const long double strength = std::abs(central_strength);
                probability = (1 - strength) * p + strength * extreme;
                // Bound only rounding of the analytic [0,1] expression, not a physical occupation draw.
                probability = std::max(0.0L, std::min(1.0L, probability));
            }
            if (satellite_strength != 0 && mu > 0) {
                const bool prefers_high = satellite_strength > 0;
                const long double preferred_fraction = prefers_high ? high_fraction : low_fraction;
                const long double strength = std::abs(satellite_strength);
                satellite = (1 - strength) * mu;
                if (high == prefers_high)
                    satellite += product_ratio(mu, strength, preferred_fraction);
            }
            return {finite_result(probability), finite_result(satellite)};
        }
        /// Adapt the existing p_c / conditional-rate convention before decorating marginal means.
        MarginalOccupationMeans operator()(const OccupationMeans & baseline, double percentile) const {
            detail::require(std::isfinite(baseline.central_probability) && baseline.central_probability >= 0 &&
                                baseline.central_probability <= 1 && std::isfinite(baseline.satellites_if_central) &&
                                baseline.satellites_if_central >= 0,
                            "Invalid assembly-bias baseline conditional means");
            return decorate_marginal(
                {baseline.central_probability, baseline.central_probability * baseline.satellites_if_central},
                percentile);
        }
    };
} // namespace FML::HOD
#endif
