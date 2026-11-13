#ifndef FML_HYDRODYNAMICS_EULER_HEADER
#define FML_HYDRODYNAMICS_EULER_HEADER
#include <FML/Hydrodynamics/EquationOfState.h>
#include <algorithm>
#include <array>
#include <limits>

namespace FML::HYDRODYNAMICS {
    template <int D>
    using Vector = std::array<double, D>;
    // density, momentum[D], total energy density, auxiliary internal energy density.
    template <int D>
    using Conserved = std::array<double, D + 3>;
    template <int D>
    struct Primitive {
        double density = 1;
        Vector<D> velocity{};
        double pressure = 1;
    };
    struct DualEnergy {
        bool enabled = false;
        double pressure_threshold = 1e-3, synchronization_threshold = 1e-2;
    };
    template <int D>
    bool valid(const Primitive<D> & q, const IdealGas & eos) {
        static_assert(D >= 1 && D <= 3);
        if (!(q.density > 0 && q.pressure > 0 && std::isfinite(q.density) && std::isfinite(q.pressure)))
            return false;
        for (auto x : q.velocity)
            if (!std::isfinite(x))
                return false;
        const double c2 = eos.gamma() * (q.pressure / q.density);
        return c2 > 0 && std::isfinite(c2);
    }
    template <int D>
    bool encode(const Primitive<D> & q, const IdealGas & eos, Conserved<D> & u) {
        if (!valid(q, eos))
            return false;
        u[0] = q.density;
        long double kinetic = 0;
        for (int d = 0; d < D; ++d) {
            u[1 + d] = q.density * q.velocity[d];
            kinetic += .5L * q.density * q.velocity[d] * q.velocity[d];
        }
        u[D + 2] = q.pressure / (eos.gamma() - 1);
        u[D + 1] = double(kinetic + u[D + 2]);
        for (auto x : u)
            if (!std::isfinite(x))
                return false;
        return u[D + 2] > 0 && u[D + 1] > 0;
    }
    template <int D>
    Conserved<D> conserved(const Primitive<D> & q, const IdealGas & eos) {
        Conserved<D> u{};
        if (!encode<D>(q, eos, u))
            throw std::invalid_argument("[Euler] Unrepresentable primitive state");
        return u;
    }
    template <int D>
    long double kinetic_energy(const Conserved<D> & u) {
        long double k = 0;
        for (int d = 0; d < D; ++d)
            k += .5L * (static_cast<long double>(u[d + 1]) / u[0]) * u[d + 1];
        return k;
    }
    template <int D>
    struct Recovery {
        Primitive<D> state;
        bool valid = false, used_auxiliary = false;
        long double thermal = 0;
    };
    template <int D>
    Recovery<D> recover(const Conserved<D> & u, const IdealGas & eos, DualEnergy policy = {}) {
        Recovery<D> out;
        for (auto v : u)
            if (!std::isfinite(v))
                return out;
        if (u[0] <= 0 || u[D + 1] <= 0)
            return out;
        const long double k = kinetic_energy<D>(u), thermal = static_cast<long double>(u[D + 1]) - k;
        out.thermal = thermal;
        const long double fraction = thermal / u[D + 1];
        double e = double(thermal);
        if (policy.enabled && fraction < policy.pressure_threshold) {
            if (u[D + 2] <= 0 || std::abs(fraction) > policy.pressure_threshold)
                return out;
            e = u[D + 2];
            out.used_auxiliary = true;
        }
        if (!(e > 0 && std::isfinite(e)))
            return out;
        out.state.density = u[0];
        out.state.pressure = (eos.gamma() - 1) * e;
        for (int d = 0; d < D; ++d)
            out.state.velocity[d] = u[d + 1] / u[0];
        out.valid = FML::HYDRODYNAMICS::valid(out.state, eos);
        return out;
    }
    template <int D>
    Primitive<D> primitive(const Conserved<D> & u, const IdealGas & eos, DualEnergy policy = {}) {
        auto r = recover<D>(u, eos, policy);
        if (!r.valid)
            throw std::invalid_argument("[Euler] Nonphysical conserved state");
        return r.state;
    }
    template <int D>
    bool normalize(Conserved<D> & u, const IdealGas & eos, DualEnergy policy, double & energy_change) {
        auto r = recover<D>(u, eos, policy);
        if (!r.valid)
            return false;
        energy_change = 0;
        if (r.used_auxiliary) {
            const double corrected = double(kinetic_energy<D>(u) + u[D + 2]);
            if (!std::isfinite(corrected))
                return false;
            energy_change = corrected - u[D + 1];
            u[D + 1] = corrected;
        } else if (!policy.enabled || r.thermal / u[D + 1] > policy.synchronization_threshold || u[D + 2] <= 0) {
            u[D + 2] = double(r.thermal);
        }
        return std::isfinite(energy_change) && u[D + 2] > 0;
    }
    template <int D>
    Conserved<D> physical_flux(const Primitive<D> & q, const Conserved<D> & u, int axis) {
        Conserved<D> f{};
        const double v = q.velocity[axis];
        f[0] = u[axis + 1];
        for (int d = 0; d < D; ++d)
            f[1 + d] = u[1 + d] * v + (d == axis ? q.pressure : 0);
        f[D + 1] = (u[D + 1] + q.pressure) * v;
        f[D + 2] = u[D + 2] * v;
        return f;
    }
} // namespace FML::HYDRODYNAMICS
#endif
