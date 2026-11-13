#ifndef FML_HYDRODYNAMICS_RIEMANN_HEADER
#define FML_HYDRODYNAMICS_RIEMANN_HEADER
#include <FML/Hydrodynamics/Euler.h>
namespace FML::HYDRODYNAMICS {
    enum class FluxMethod { Rusanov, HLLE, HLLC };
    enum class FluxFallback { None, Vacuum, Degenerate, StarPressure, StarEnergy, Nonfinite };
    template <int D>
    struct FluxResult {
        Conserved<D> flux{};
        double interface_velocity = 0, max_speed = 0;
        FluxMethod method = FluxMethod::Rusanov;
        FluxFallback fallback = FluxFallback::None;
        bool valid = false;
    };
    /// Positive left/right ideal-gas states. Invalid/unrepresentable inputs return valid=false.
    template <int D>
    FluxResult<D> riemann(const Primitive<D> & left,
                          const Primitive<D> & right,
                          const IdealGas & eos,
                          int axis = 0,
                          FluxMethod requested = FluxMethod::HLLC,
                          std::array<double, 2> auxiliary_density = {-1, -1}) {
        FluxResult<D> out;
        if (axis < 0 || axis >= D || int(requested) < 0 || int(requested) > 2)
            return out;
        Conserved<D> ul{}, ur{};
        if (!encode<D>(left, eos, ul) || !encode<D>(right, eos, ur))
            return out;
        if (auxiliary_density[0] != -1 || auxiliary_density[1] != -1) {
            if (!(auxiliary_density[0] > 0 && auxiliary_density[1] > 0 && std::isfinite(auxiliary_density[0]) &&
                  std::isfinite(auxiliary_density[1])))
                return out;
            ul[D + 2] = auxiliary_density[0];
            ur[D + 2] = auxiliary_density[1];
        }
        const auto fl = physical_flux<D>(left, ul, axis), fr = physical_flux<D>(right, ur, axis);
        const double vl = left.velocity[axis], vr = right.velocity[axis];
        const double cl = std::sqrt(eos.gamma() * (left.pressure / left.density)),
                     cr = std::sqrt(eos.gamma() * (right.pressure / right.density));
        const double alpha = std::max(std::abs(vl) + cl, std::abs(vr) + cr);
        auto finish = [&] {
            out.valid = std::isfinite(out.interface_velocity) && std::isfinite(out.max_speed) && out.max_speed > 0;
            for (auto v : out.flux)
                out.valid &= std::isfinite(v);
            return out;
        };
        auto llf = [&] {
            for (int j = 0; j < D + 3; ++j)
                out.flux[j] =
                    double(.5L * fl[j] + .5L * fr[j] - .5L * alpha * (static_cast<long double>(ur[j]) - ul[j]));
            out.interface_velocity = .5 * vl + .5 * vr;
            out.max_speed = alpha;
            out.method = FluxMethod::Rusanov;
        };
        if (requested == FluxMethod::Rusanov) {
            llf();
            return finish();
        }
        // Einfeldt/Roe bounds, widened to include both physical end-state cones.
        const long double rl = std::sqrt(left.density), rr = std::sqrt(right.density), weight = rl + rr;
        std::array<long double, D> roe{};
        long double kinetic = 0, enthalpy = 0;
        for (int d = 0; d < D; ++d) {
            roe[d] = (rl * left.velocity[d] + rr * right.velocity[d]) / weight;
            kinetic += .5L * roe[d] * roe[d];
        }
        auto h = [&](const Primitive<D> & q) {
            long double value = eos.gamma() / (eos.gamma() - 1) * (static_cast<long double>(q.pressure) / q.density);
            for (auto v : q.velocity)
                value += .5L * v * v;
            return value;
        };
        enthalpy = (rl * h(left) + rr * h(right)) / weight;
        const double croe = std::sqrt(double(std::max(0.L, (eos.gamma() - 1) * (enthalpy - kinetic))));
        const double sl = std::min({vl - cl, vr - cr, double(roe[axis]) - croe});
        const double sr = std::max({vl + cl, vr + cr, double(roe[axis]) + croe});
        out.max_speed = std::max(std::abs(sl), std::abs(sr));
        out.method = FluxMethod::HLLE;
        if (sl >= 0) {
            out.flux = fl;
            out.interface_velocity = vl;
            return finish();
        }
        if (sr <= 0) {
            out.flux = fr;
            out.interface_velocity = vr;
            return finish();
        }
        const double width = sr - sl;
        if (!(width > 0 && std::isfinite(width))) {
            out.fallback = FluxFallback::Degenerate;
            llf();
            return finish();
        }
        auto hlle = [&] {
            for (int j = 0; j < D + 3; ++j)
                out.flux[j] = double((static_cast<long double>(sr) * fl[j] - static_cast<long double>(sl) * fr[j] +
                                      static_cast<long double>(sl) * sr * (static_cast<long double>(ur[j]) - ul[j])) /
                                     width);
            const long double density =
                (static_cast<long double>(sr) * ur[0] - static_cast<long double>(sl) * ul[0] - fr[0] + fl[0]) / width;
            const long double momentum = (static_cast<long double>(sr) * ur[axis + 1] -
                                          static_cast<long double>(sl) * ul[axis + 1] - fr[axis + 1] + fl[axis + 1]) /
                                         width;
            out.interface_velocity =
                density > 0 ? double(momentum / density) : std::numeric_limits<double>::quiet_NaN();
            out.method = FluxMethod::HLLE;
        };
        hlle();
        if (requested == FluxMethod::HLLE)
            return finish();
        if (vr - vl >= 2 * (cl + cr) / (eos.gamma() - 1)) {
            out.fallback = FluxFallback::Vacuum;
            return finish();
        }
        const long double al = static_cast<long double>(left.density) * (sl - vl),
                          ar = static_cast<long double>(right.density) * (sr - vr);
        const long double denominator = al - ar;
        if (denominator == 0 || !std::isfinite(denominator)) {
            out.fallback = FluxFallback::Degenerate;
            return finish();
        }
        const double sm =
            double((static_cast<long double>(right.pressure) - left.pressure + al * vl - ar * vr) / denominator);
        if (!std::isfinite(sm) || sm <= sl || sm >= sr) {
            out.fallback = FluxFallback::Degenerate;
            return finish();
        }
        auto star = [&](const Primitive<D> & q, const Conserved<D> & u, double speed, Conserved<D> & value) {
            const long double v = q.velocity[axis], delta = sm - v;
            const long double pressure = q.pressure + static_cast<long double>(q.density) * (speed - v) * delta;
            if (!(pressure > 0 && std::isfinite(pressure)))
                return FluxFallback::StarPressure;
            const long double rho = static_cast<long double>(q.density) * (speed - v) / (speed - sm);
            const long double e = eos.specific_energy(q.density, q.pressure) + .5L * delta * delta +
                                  delta * q.pressure / (q.density * (speed - v));
            if (!(rho > 0 && e > 0 && std::isfinite(rho) && std::isfinite(e)))
                return FluxFallback::StarEnergy;
            value[0] = double(rho);
            long double ke = 0;
            for (int d = 0; d < D; ++d) {
                const long double vd = d == axis ? sm : q.velocity[d];
                value[d + 1] = double(rho * vd);
                ke += .5L * vd * vd;
            }
            value[D + 1] = double(rho * (e + ke));
            value[D + 2] = double(rho * (u[D + 2] / q.density));
            for (auto x : value)
                if (!std::isfinite(x))
                    return FluxFallback::Nonfinite;
            return FluxFallback::None;
        };
        Conserved<D> usl{}, usr{};
        auto reason = star(left, ul, sl, usl);
        if (reason == FluxFallback::None)
            reason = star(right, ur, sr, usr);
        if (reason != FluxFallback::None) {
            out.fallback = reason;
            return finish();
        }
        const auto & f = sm >= 0 ? fl : fr;
        const auto & u = sm >= 0 ? ul : ur;
        const auto & s = sm >= 0 ? usl : usr;
        const double speed = sm >= 0 ? sl : sr;
        Conserved<D> selected{};
        for (int j = 0; j < D + 3; ++j) {
            selected[j] = double(static_cast<long double>(f[j]) + speed * (static_cast<long double>(s[j]) - u[j]));
            if (!std::isfinite(selected[j])) {
                out.fallback = FluxFallback::Nonfinite;
                return finish();
            }
        }
        out.flux = selected;
        out.interface_velocity = sm;
        out.method = FluxMethod::HLLC;
        return finish();
    }
} // namespace FML::HYDRODYNAMICS
#endif
