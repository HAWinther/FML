#ifndef FML_GROWTH_HISTORY_H
#define FML_GROWTH_HISTORY_H
#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#ifdef USE_GSL
#include <FML/Spline/Spline.h>
#include <memory>
#endif
namespace FML::COSMOLOGY {
    /// Scale-independent time amplitudes for x = q + D1 Psi1 + D2 Psi2.
    /// Rates are f_i = (a / D_i) dD_i/da, including signed D2 (EdS: -3/7 D1^2).
    /// D1 and D2 must use the same normalization: D1 -> c D1 requires D2 -> c^2 D2.
    /// This interface supplies time dependence, not scale-dependent nonlinear kernels.
    class GrowthHistory {
      public:
        using Function = std::function<double(double)>;
        GrowthHistory(Function D1 = {}, Function f1 = {}, Function D2 = {}, Function f2 = {})
            : D1_(std::move(D1)), f1_(std::move(f1)), D2_(std::move(D2)), f2_(std::move(f2)) {}
        double linear(double a) const { return value(D1_, a); }
        double linear_rate(double a) const { return value(f1_, a); }
        double second_order(double a) const { return value(D2_, a); }
        double second_order_rate(double a) const { return value(f2_, a); }
        /// v = a^2 E(a) f1 D1 Psi1, where v = a^3 E(a) dx/da and E = H/H0.
        double linear_momentum(double a, double HoverH0) const {
            if (!(HoverH0 > 0.0) || !std::isfinite(HoverH0))
                throw std::invalid_argument("[GrowthHistory] H/H0 must be positive and finite");
            return a * a * HoverH0 * linear_rate(a) * linear(a);
        }
        double second_order_slope(double a) const {
            const double denominator = linear_rate(a) * linear(a);
            if (denominator == 0.0 || !std::isfinite(denominator))
                throw std::domain_error("[GrowthHistory] dD2/dD1 needs nonzero dD1/dln(a)");
            return second_order_rate(a) * second_order(a) / denominator;
        }

      private:
        Function D1_, f1_, D2_, f2_;
        static double value(const Function & function, double a) {
            if (!function)
                throw std::invalid_argument("[GrowthHistory] Required growth function is missing");
            if (!(a > 0.0) || !std::isfinite(a))
                throw std::invalid_argument("[GrowthHistory] Scale factor must be positive and finite");
            const double result = function(a);
            if (!std::isfinite(result))
                throw std::domain_error("[GrowthHistory] Growth function returned a nonfinite value");
            return result;
        }
    };
#ifdef USE_GSL
    /// Own copies of the log(a) splines returned by the LPT growth solver.
    inline GrowthHistory growth_history_from_log_a_splines(FML::INTERPOLATION::SPLINE::Spline D1,
                                                           FML::INTERPOLATION::SPLINE::Spline D2) {
        auto linear = std::make_shared<const FML::INTERPOLATION::SPLINE::Spline>(std::move(D1));
        auto second = std::make_shared<const FML::INTERPOLATION::SPLINE::Spline>(std::move(D2));
        return {[linear](double a) { return (*linear)(std::log(a)); },
                [linear](double a) { return linear->deriv_x(std::log(a)) / (*linear)(std::log(a)); },
                [second](double a) { return (*second)(std::log(a)); },
                [second](double a) { return second->deriv_x(std::log(a)) / (*second)(std::log(a)); }};
    }
#endif
} // namespace FML::COSMOLOGY
#endif
