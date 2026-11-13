#ifndef FML_HYDRODYNAMICS_EXACT_RIEMANN_HEADER
#define FML_HYDRODYNAMICS_EXACT_RIEMANN_HEADER
#include <FML/Hydrodynamics/Euler.h>
namespace FML::HYDRODYNAMICS {
    /// Exact ideal-gas 1D Riemann reference, including generated vacuum.
    /// sample(x/t) may return rho=p=0 in vacuum, unlike the positive-state FV API.
    class ExactRiemann {
        Primitive<1> left_, right_;
        IdealGas eos_;
        double cl_, cr_, pstar_ = 0, ustar_ = 0;
        bool vacuum_ = false;
        double wave(double p, const Primitive<1> & q, double c) const {
            const double gamma = eos_.gamma();
            if (p > q.pressure)
                return double((static_cast<long double>(p) - q.pressure) *
                              std::sqrt(2.L / (static_cast<long double>(gamma + 1) * q.density) /
                                        (p + (gamma - 1) / (gamma + 1) * q.pressure)));
            return 2 * c / (gamma - 1) * (std::pow(p / q.pressure, (gamma - 1) / (2 * gamma)) - 1);
        }
        Primitive<1> fan(double xi, bool left) const {
            const auto & q = left ? left_ : right_;
            const double c = left ? cl_ : cr_, g = eos_.gamma(), sign = left ? 1 : -1;
            const double velocity = 2 / (g + 1) * (sign * c + .5 * (g - 1) * q.velocity[0] + xi);
            const double sound = std::max(0., 2 / (g + 1) * (c + .5 * sign * (g - 1) * (q.velocity[0] - xi)));
            return {q.density * std::pow(sound / c, 2 / (g - 1)),
                    {velocity},
                    q.pressure * std::pow(sound / c, 2 * g / (g - 1))};
        }

      public:
        ExactRiemann(Primitive<1> left, Primitive<1> right, IdealGas eos = IdealGas{})
            : left_(left), right_(right), eos_(eos) {
            if (!valid(left, eos) || !valid(right, eos))
                throw std::invalid_argument("[ExactRiemann] Invalid positive states");
            cl_ = std::sqrt(eos.gamma() * (left.pressure / left.density));
            cr_ = std::sqrt(eos.gamma() * (right.pressure / right.density));
            const double dv = right.velocity[0] - left.velocity[0];
            vacuum_ = dv >= 2 * (cl_ + cr_) / (eos.gamma() - 1);
            if (vacuum_)
                return;
            auto f = [&](double p) { return wave(p, left_, cl_) + wave(p, right_, cr_) + dv; };
            double hi = std::max(left.pressure, right.pressure);
            for (int j = 0; f(hi) < 0; ++j) {
                if (j > 1024 || hi > std::numeric_limits<double>::max() / 2)
                    throw std::runtime_error("[ExactRiemann] Pressure bracket overflow");
                hi *= 2;
            }
            double lo = std::numeric_limits<double>::min();
            if (f(lo) > 0)
                throw std::runtime_error("[ExactRiemann] Pressure below representable normal range");
            double l = std::log(lo), h = std::log(hi);
            for (int j = 0; j < 220; ++j) {
                double m = .5 * l + .5 * h;
                double p = std::exp(m);
                if (f(p) > 0)
                    h = m;
                else
                    l = m;
            }
            pstar_ = std::exp(.5 * l + .5 * h);
            ustar_ = .5 * left.velocity[0] + .5 * right.velocity[0] +
                     .5 * (wave(pstar_, right_, cr_) - wave(pstar_, left_, cl_));
            if (!(pstar_ > 0 && std::isfinite(pstar_) && std::isfinite(ustar_)))
                throw std::runtime_error("[ExactRiemann] Unrepresentable star state");
        }
        bool has_vacuum() const { return vacuum_; }
        double star_pressure() const { return pstar_; }
        double star_velocity() const { return ustar_; }
        Primitive<1> sample(double xi) const {
            if (!std::isfinite(xi))
                throw std::invalid_argument("[ExactRiemann] Nonfinite similarity coordinate");
            const double g = eos_.gamma();
            if (vacuum_) {
                if (xi <= left_.velocity[0] - cl_)
                    return left_;
                if (xi < left_.velocity[0] + 2 * cl_ / (g - 1))
                    return fan(xi, true);
                if (xi < right_.velocity[0] - 2 * cr_ / (g - 1))
                    return {0, {xi}, 0};
                if (xi < right_.velocity[0] + cr_)
                    return fan(xi, false);
                return right_;
            }
            const bool isleft = xi < ustar_;
            const auto & q = isleft ? left_ : right_;
            const double c = isleft ? cl_ : cr_, sign = isleft ? -1 : 1;
            if (pstar_ > q.pressure) {
                const double speed =
                    q.velocity[0] +
                    sign * std::sqrt(double(((g + 1) * static_cast<long double>(pstar_) + (g - 1) * q.pressure) /
                                            (2.L * q.density)));
                if ((isleft && xi < speed) || (!isleft && xi > speed))
                    return q;
                const double ratio = q.pressure / pstar_, gg = (g - 1) / (g + 1);
                return {q.density * (1 + gg * ratio) / (gg + ratio), {ustar_}, pstar_};
            }
            const double sound = c * std::pow(pstar_ / q.pressure, (g - 1) / (2 * g));
            const double head = q.velocity[0] + sign * c, tail = ustar_ + sign * sound;
            if ((isleft && xi < head) || (!isleft && xi > head))
                return q;
            if ((isleft && xi > tail) || (!isleft && xi < tail))
                return {q.density * std::pow(pstar_ / q.pressure, 1 / g), {ustar_}, pstar_};
            return fan(xi, isleft);
        }
    };
} // namespace FML::HYDRODYNAMICS
#endif
