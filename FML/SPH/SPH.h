#ifndef FML_SPH_HEADER
#define FML_SPH_HEADER
#include <FML/Hydrodynamics/EquationOfState.h>
#include <FML/SPH/Cells.h>
#include <FML/SPH/Transport.h>
#include <functional>

namespace FML::SPH {
    template <int D>
    struct Particle {
        Vector<D> position{}, velocity{};
        double mass = 1, energy = 1, h = 0; // Specific internal energy; h=0 requests a bracket-midpoint initial guess.
        std::uint64_t id = 0;
    };
    struct Options {
        double boxsize = 1, gamma = 5. / 3, eta = 1.2;
        double h_min = .001, h_max = .1, density_tolerance = 1e-10;
        int density_iterations = 80;
        double viscosity_alpha = 1, viscosity_beta = 2, viscosity_epsilon = .01;
        bool balsara = true;
        double conductivity_alpha = 0, switch_epsilon = 1e-4;
        double courant = .25, acceleration_factor = .25, energy_factor = .2;
        size_t max_cells = 1000000, max_message_bytes = 8 * 1024 * 1024;
    };
    struct Background {
        double a = 1, H = 0;
    };
    struct Thermodynamics {
        double density = 0, omega = 1, pressure = 0, sound_speed = 0, divergence = 0, curl = 0, limiter = 1;
        int iterations = 0;
    };
    template <int D>
    struct Result {
        std::vector<Thermodynamics> gas;
        std::vector<Vector<D>> acceleration;
        std::vector<double> energy_rate, signal_speed;
        double timestep = std::numeric_limits<double>::infinity();
        size_t density_ghosts = 0, force_ghosts = 0;
    };
    template <int D>
    struct Diagnostics {
        double mass = 0, kinetic = 0, thermal = 0;
        Vector<D> momentum{};
        double gas_energy() const { return kinetic + thermal; }
    };
    /// Conservative density-energy SPH, periodic physical/comoving coordinates.
    /// All public operations except const accessors are collective on FML::communicator.
    template <int D>
    class Solver {
        static_assert(D >= 1 && D <= 3);

      public:
        using P = Particle<D>;
        using BackgroundFunction = std::function<Background(double)>;
        // Returns a contribution to dv_peculiar/dt, already in the evolution's units.
        // Called on every rank, including empty ones; it may use collective gravity solvers.
        using ExternalForce = std::function<void(double, const std::vector<P> &, std::vector<Vector<D>> &)>;

      private:
        Options options_;
        FML::HYDRODYNAMICS::IdealGas eos_;
        std::vector<P> particles_;
        std::vector<std::array<double, 2>> slabs_;
        BackgroundFunction background_;
        double time_ = 0;
        struct Source {
            P particle;
            Thermodynamics gas;
        };
        struct Stage {
            P particle, initial;
            Vector<D> acceleration0;
            double energy_rate0;
        };
        // Scratch is reused across evaluations; source records are rebuilt after every stage/migration.
        std::vector<P> density_sources_;
        std::vector<Source> force_sources_;
        detail::Cells<D> density_cells_, force_cells_;

        void validate_options() {
            detail::C::collective_try(
                [&] {
                    const auto pos = [](double x) { return std::isfinite(x) && x > 0; };
                    const auto nonneg = [](double x) { return std::isfinite(x) && x >= 0; };
                    if (!pos(options_.boxsize) || !pos(options_.gamma - 1) || !pos(options_.eta) ||
                        !pos(options_.h_min) || !pos(options_.h_max) || options_.h_min >= options_.h_max ||
                        2 * options_.h_max >= .5 * options_.boxsize || !pos(options_.density_tolerance) ||
                        options_.density_tolerance >= .1 || options_.density_iterations < 1 ||
                        !nonneg(options_.viscosity_alpha) || !nonneg(options_.viscosity_beta) ||
                        !pos(options_.viscosity_epsilon) || !nonneg(options_.conductivity_alpha) ||
                        !pos(options_.switch_epsilon) || !pos(options_.courant) || !pos(options_.acceleration_factor) ||
                        !pos(options_.energy_factor) || options_.max_cells < 1 || options_.max_cells > 100000000 ||
                        options_.max_message_bytes < sizeof(Stage) ||
                        !std::isfinite(CubicSpline<D>::normalization() / std::pow(options_.h_min, D + 1)) ||
                        !std::isfinite(std::pow(options_.eta, D)) ||
                        std::pow(options_.eta, D) <= CubicSpline<D>::normalization())
                        throw std::invalid_argument("[SPH] Invalid options/scales (require support 2h_max < L/2)");
                },
                "SPH options");
            detail::agree(std::array<double, 17>{options_.boxsize,
                                                 options_.gamma,
                                                 options_.eta,
                                                 options_.h_min,
                                                 options_.h_max,
                                                 options_.density_tolerance,
                                                 double(options_.density_iterations),
                                                 options_.viscosity_alpha,
                                                 options_.viscosity_beta,
                                                 options_.viscosity_epsilon,
                                                 double(options_.balsara),
                                                 options_.conductivity_alpha,
                                                 options_.switch_epsilon,
                                                 options_.courant,
                                                 options_.acceleration_factor,
                                                 options_.energy_factor,
                                                 double(options_.max_cells)},
                          "[SPH] Physics/options differ between ranks");
            eos_ = FML::HYDRODYNAMICS::IdealGas(options_.gamma);
        }
        void validate_particles(const std::vector<P> & p) const {
            detail::C::collective_try(
                [&] {
                    for (const auto & x : p) {
                        if (!std::isfinite(x.mass) || x.mass <= 0 || !std::isfinite(x.energy) || x.energy <= 0 ||
                            !std::isfinite(x.h) || x.h < 0)
                            throw std::invalid_argument(
                                "[SPH] Require finite positive mass/energy and nonnegative h guess");
                        for (int d = 0; d < D; ++d)
                            if (!std::isfinite(x.position[d]) || x.position[d] < 0 ||
                                x.position[d] >= options_.boxsize || !std::isfinite(x.velocity[d]))
                                throw std::invalid_argument("[SPH] Invalid position/velocity");
                    }
                },
                "SPH particle state");
        }
        template <class Record, class GetParticle>
        std::vector<Record> migrate(const std::vector<Record> & records, GetParticle get) const {
            const bool pencils = FML::domain.holds<FML::PencilDomain>();
            auto result = detail::route(
                records,
                [&](const Record & r) {
                    if constexpr (D >= 2) {
                        if (pencils) {
                            const double xy[2] = {
                                std::min(std::nextafter(1., 0.), get(r).position[0] / options_.boxsize),
                                std::min(std::nextafter(1., 0.), get(r).position[1] / options_.boxsize)};
                            return FML::domain.owner(xy);
                        }
                    }
                    const double x = std::min(std::nextafter(1., 0.), get(r).position[0] / options_.boxsize);
                    const auto it = std::upper_bound(slabs_.begin(), slabs_.end(), x, [](double y, const auto & slab) {
                        return y < slab[1];
                    });
                    return int(it - slabs_.begin());
                },
                options_.max_message_bytes);
            detail::C::collective_try(
                [&] {
                    std::sort(result.begin(), result.end(), [&](const auto & a, const auto & b) {
                        return get(a).id < get(b).id;
                    });
                },
                "SPH stable local ordering");
            return result;
        }
        template <class Record, class GetParticle>
        void ghosts(std::vector<Record> & records, double support, GetParticle get) const {
            if constexpr (D >= 2) {
                // Slabs: the same as append_periodic_slab_ghosts; pencils: the records near each pencil
                detail::C::append_periodic_domain_ghosts(
                    records,
                    support / options_.boxsize,
                    [&](const Record & p) {
                        return std::array<double, 2>{
                            std::min(std::nextafter(1., 0.), get(p).position[0] / options_.boxsize),
                            std::min(std::nextafter(1., 0.), get(p).position[1] / options_.boxsize)};
                    },
                    options_.max_message_bytes);
            } else {
                detail::C::append_periodic_slab_ghosts(
                    records,
                    support / options_.boxsize,
                    [&](const Record & p) {
                        return std::min(std::nextafter(1., 0.), get(p).position[0] / options_.boxsize);
                    },
                    options_.max_message_bytes);
            }
            detail::C::collective_try(
                [&] {
                    std::sort(records.begin(), records.end(), [&](const auto & a, const auto & b) {
                        return get(a).id < get(b).id;
                    });
                },
                "SPH ghost ordering");
        }
        Background background(double t) const {
            Background b;
            detail::C::collective_try(
                [&] {
                    if (!std::isfinite(t))
                        throw std::invalid_argument("[SPH] Nonfinite epoch");
                    b = background_ ? background_(t) : Background{};
                    if (!std::isfinite(b.a) || b.a <= 0 || !std::isfinite(b.H))
                        throw std::invalid_argument("[SPH] Invalid background");
                    if constexpr (D != 3)
                        if (b.a != 1 || b.H != 0)
                            throw std::invalid_argument("[SPH] Expanding background requires D=3");
                },
                "SPH background");
            detail::agree(std::array<double, 3>{t, b.a, b.H}, "[SPH] Background/epoch differs between ranks");
            return b;
        }
        struct Density {
            double rho = 0, dh = 0;
        };
        Density density(const P & p, double h) const {
            Density v;
            density_cells_.neighbors(p.position, [&](size_t j) {
                const auto & s = density_sources_[j];
                const auto dx = detail::separation<D>(p.position, s.position, options_.boxsize);
                const auto k = CubicSpline<D>{}(std::sqrt(detail::dot<D>(dx, dx)), h);
                v.rho += s.mass * k.w;
                v.dh += s.mass * k.dh;
            });
            return v;
        }
        bool solve_density(P & p, Thermodynamics & state) const {
            double lo = options_.h_min, hi = options_.h_max;
            const double target = std::pow(options_.eta, D) * p.mass;
            auto residual = [&](double h, Density v) { return std::pow(h, D) * v.rho - target; };
            const auto a = density(p, lo), b = density(p, hi);
            if (!(residual(lo, a) <= 0 && residual(hi, b) >= 0))
                return false;
            double h = p.h > lo && p.h < hi ? p.h : .5 * (lo + hi);
            for (int it = 0; it < options_.density_iterations; ++it) {
                const auto v = density(p, h);
                const double f = residual(h, v);
                const double omega = 1 + (h / D) * (v.dh / v.rho);
                if (std::abs(f) <= options_.density_tolerance * target) {
                    if (!(v.rho > 0 && std::isfinite(v.rho) && omega > 1e-8 && std::isfinite(omega)))
                        return false;
                    p.h = h;
                    state.density = v.rho;
                    state.omega = omega;
                    state.iterations = it + 1;
                    return true;
                }
                if (f > 0)
                    hi = h;
                else
                    lo = h;
                const double derivative = D * std::pow(h, D - 1) * v.rho * omega;
                const double next = h - f / derivative;
                h = derivative > 0 && std::isfinite(next) && next > lo && next < hi ? next : .5 * (lo + hi);
            }
            return false;
        }
        Result<D> rhs(std::vector<P> & owned, double epoch, const ExternalForce & external) {
            detail::agree(std::array<double, 1>{double(bool(external))},
                          "[SPH] External force presence differs between ranks");
            const auto bg = background(epoch);
            Result<D> out;
            detail::C::collective_try(
                [&] {
                    out.gas.resize(owned.size());
                    out.acceleration.resize(owned.size());
                    out.energy_rate.resize(owned.size());
                    out.signal_speed.resize(owned.size());
                    density_sources_ = owned;
                },
                "SPH density buffers");
            ghosts(density_sources_, 2 * options_.h_max, [](const P & p) -> const P & { return p; });
            out.density_ghosts = density_sources_.size() - owned.size();
            detail::C::collective_try(
                [&] {
                    density_cells_.build(
                        density_sources_,
                        [](const P & p) -> const auto & { return p.position; },
                        options_.boxsize,
                        2 * options_.h_max,
                        options_.max_cells);
                },
                "SPH density cells");
            int failed = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : failed)
#endif
            for (size_t i = 0; i < owned.size(); ++i)
                if (!solve_density(owned[i], out.gas[i]))
                    failed = 1;
            FML::MaxOverTasks(&failed);
            if (failed)
                throw std::runtime_error(
                    "[SPH] Adaptive density root unbracketed/nonconvergent; adjust h bounds or iterations");
            detail::C::collective_try(
                [&] {
                    for (size_t i = 0; i < owned.size(); ++i) {
                        auto & g = out.gas[i];
                        const auto & p = owned[i];
                        g.pressure = eos_.pressure(g.density, p.energy);
                        g.sound_speed = std::sqrt(eos_.sound_speed_squared(g.density, p.energy));
                        double div = 0;
                        std::array<double, 3> curl{};
                        density_cells_.neighbors(p.position, [&](size_t j) {
                            const auto & s = density_sources_[j];
                            const auto dx = detail::separation<D>(p.position, s.position, options_.boxsize);
                            const double r = std::sqrt(detail::dot<D>(dx, dx));
                            if (r == 0 || r >= 2 * p.h)
                                return;
                            const double factor = (s.mass / g.density) / g.omega * (CubicSpline<D>{}(r, p.h).dr / r);
                            std::array<double, 3> grad{}, dv{};
                            for (int d = 0; d < D; ++d) {
                                grad[d] = dx[d] * factor;
                                dv[d] = s.velocity[d] - p.velocity[d];
                                div += grad[d] * dv[d];
                            }
                            for (int d = 0; d < 3; ++d)
                                curl[d] += grad[(d + 1) % 3] * dv[(d + 2) % 3] - grad[(d + 2) % 3] * dv[(d + 1) % 3];
                        });
                        g.divergence = div / bg.a + D * bg.H;
                        g.curl = std::sqrt(curl[0] * curl[0] + curl[1] * curl[1] + curl[2] * curl[2]) / bg.a;
                        g.limiter = options_.balsara ? std::abs(g.divergence) /
                                                           (std::abs(g.divergence) + g.curl +
                                                            options_.switch_epsilon * g.sound_speed / (bg.a * p.h)) :
                                                       1;
                        if (!(g.pressure > 0) || !(g.sound_speed > 0) || !std::isfinite(g.pressure) ||
                            !std::isfinite(g.sound_speed) || !std::isfinite(g.limiter))
                            throw std::runtime_error("[SPH] Unrepresentable thermodynamics");
                    }
                },
                "SPH equation of state/velocity gradients");
            double hmax = 0;
            for (const auto & p : owned)
                hmax = std::max(hmax, p.h);
            FML::MaxOverTasks(&hmax);
            detail::C::collective_try(
                [&] {
                    force_sources_.resize(owned.size());
                    for (size_t i = 0; i < owned.size(); ++i)
                        force_sources_[i] = {owned[i], out.gas[i]};
                },
                "SPH force buffers");
            if (hmax > 0) {
                ghosts(force_sources_, 2 * hmax, [](const Source & s) -> const P & { return s.particle; });
                out.force_ghosts = force_sources_.size() - owned.size();
                detail::C::collective_try(
                    [&] {
                        force_cells_.build(
                            force_sources_,
                            [](const Source & s) -> const auto & { return s.particle.position; },
                            options_.boxsize,
                            2 * hmax,
                            options_.max_cells);
                    },
                    "SPH force cells");
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < owned.size(); ++i) {
                    const auto & p = owned[i];
                    const auto & g = out.gas[i];
                    double du = 0, signal = 2 * g.sound_speed;
                    Vector<D> acc{};
                    force_cells_.neighbors(p.position, [&](size_t j) {
                        const auto & s = force_sources_[j].particle;
                        const auto & q = force_sources_[j].gas;
                        if (s.id == p.id)
                            return;
                        auto dx = detail::separation<D>(p.position, s.position, options_.boxsize);
                        const double r2 = detail::dot<D>(dx, dx), r = std::sqrt(r2);
                        if (r == 0 || (r >= 2 * p.h && r >= 2 * s.h))
                            return;
                        Vector<D> dv{};
                        for (int d = 0; d < D; ++d)
                            dv[d] = p.velocity[d] - s.velocity[d];
                        const double vr = detail::dot<D>(dv, dx), hi = .5 * (p.h + s.h),
                                     rho = .5 * g.density + .5 * q.density;
                        const double physical_vr = vr + bg.H * bg.a * r2;
                        const double mu = std::min(0., hi * physical_vr / (r2 + options_.viscosity_epsilon * hi * hi));
                        const double viscosity =
                            (-options_.viscosity_alpha * .5 * (g.sound_speed + q.sound_speed) * mu +
                             options_.viscosity_beta * mu * mu) /
                            rho * .5 * (g.limiter + q.limiter);
                        const double wi = CubicSpline<D>{}(r, p.h).dr, wj = CubicSpline<D>{}(r, s.h).dr,
                                     wbar = .5 * (wi + wj);
                        const double fi = ((g.pressure / g.density) / g.density) / g.omega,
                                     fj = ((q.pressure / q.density) / q.density) / q.omega;
                        const double force = (fi * wi + fj * wj + viscosity * wbar) / r;
                        for (int d = 0; d < D; ++d)
                            acc[d] -= s.mass * force * dx[d];
                        du += s.mass * (fi * wi * vr + .5 * viscosity * wbar * physical_vr) / r;
                        const double heat_signal = std::sqrt(std::abs(g.pressure - q.pressure) / rho);
                        du += options_.conductivity_alpha * s.mass / rho * (p.energy - s.energy) * heat_signal * wbar;
                        signal = std::max(signal,
                                          g.sound_speed + q.sound_speed - options_.viscosity_beta * mu +
                                              options_.conductivity_alpha * heat_signal);
                    });
                    for (int d = 0; d < D; ++d)
                        out.acceleration[i][d] = acc[d] / bg.a - bg.H * p.velocity[d];
                    out.energy_rate[i] = du / bg.a - D * (options_.gamma - 1) * bg.H * p.energy;
                    out.signal_speed[i] = signal;
                }
            }
            if (external) {
                std::vector<Vector<D>> extra;
                detail::C::collective_try([&] { extra.resize(owned.size()); }, "SPH external force buffer");
                external(epoch, owned, extra); // Callback coordinates its own collective errors.
                detail::C::collective_try(
                    [&] {
                        if (extra.size() != owned.size())
                            throw std::invalid_argument("[SPH] External force layout mismatch");
                        for (size_t i = 0; i < extra.size(); ++i)
                            for (int d = 0; d < D; ++d) {
                                if (!std::isfinite(extra[i][d]))
                                    throw std::invalid_argument("[SPH] Nonfinite external acceleration");
                                out.acceleration[i][d] += extra[i][d];
                            }
                    },
                    "SPH external force values");
            }
            detail::C::collective_try(
                [&] {
                    for (size_t i = 0; i < owned.size(); ++i) {
                        const double a2 = detail::dot<D>(out.acceleration[i], out.acceleration[i]);
                        if (!std::isfinite(a2) || !std::isfinite(out.energy_rate[i]) ||
                            !std::isfinite(out.signal_speed[i]))
                            throw std::runtime_error("[SPH] Nonfinite hydrodynamic rates");
                        out.timestep =
                            std::min(out.timestep, options_.courant * bg.a * owned[i].h / out.signal_speed[i]);
                        if (a2 > 0)
                            out.timestep =
                                std::min(out.timestep,
                                         options_.acceleration_factor * std::sqrt(bg.a * owned[i].h / std::sqrt(a2)));
                        if (out.energy_rate[i] != 0)
                            out.timestep =
                                std::min(out.timestep,
                                         options_.energy_factor * owned[i].energy / std::abs(out.energy_rate[i]));
                    }
                },
                "SPH timestep");
            detail::C::collective_try(
                [&] {
                    if (!owned.empty() && !(out.timestep > 0))
                        throw std::runtime_error("[SPH] Unrepresentable timestep");
                },
                "SPH timestep range");
            FML::MinOverTasks(&out.timestep);
            return out;
        }
        void wrap(P & p) const {
            for (auto & x : p.position) {
                if (!std::isfinite(x))
                    throw std::runtime_error("[SPH] Nonfinite drift");
                x = std::fmod(x, options_.boxsize);
                if (x < 0)
                    x += options_.boxsize;
                if (x >= options_.boxsize)
                    x = 0;
            }
        }

      public:
        Solver(std::vector<P> particles, Options options = {}, BackgroundFunction bg = {})
            : options_(options), background_(std::move(bg)) {
            validate_options();
            if (FML::domain.holds<FML::PencilDomain>()) {
                if (D < 2)
                    throw std::invalid_argument("[SPH] A pencil FML::domain needs D >= 2");
            } else {
                slabs_ = detail::slabs();
            }
            validate_particles(particles);
            std::vector<std::uint64_t> ids;
            detail::C::collective_try(
                [&] {
                    ids.reserve(particles.size());
                    for (const auto & p : particles)
                        ids.push_back(p.id);
                },
                "SPH IDs");
            detail::unique_ids(ids, options_.max_message_bytes);
            particles_ = migrate(particles, [](const P & p) -> const P & { return p; });
        }
        const std::vector<P> & particles() const { return particles_; }
        const Options & options() const { return options_; }
        double time() const { return time_; }
        /// Density is a comoving density when background expansion is selected.
        Result<D> evaluate(const ExternalForce & external = {}) { return rhs(particles_, time_, external); }
        /// Global shared-timestep Heun RK2; commit only after all stages succeed.
        /// No positivity floor or smoothing-length clamp silently changes the equations.
      private:
        double advance_impl(double dt, bool limit, const ExternalForce & external) {
            detail::C::collective_try(
                [&] {
                    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(time_ + dt) || time_ + dt <= time_)
                        throw std::invalid_argument("[SPH] Invalid timestep");
                },
                "SPH step size");
            detail::agree(std::array<double, 2>{time_, dt}, "[SPH] Step differs between ranks");
            std::vector<P> initial;
            detail::C::collective_try([&] { initial = particles_; }, "SPH initial stage snapshot");
            auto first = rhs(initial, time_, external);
            if (limit)
                dt = std::min(dt, first.timestep);
            detail::C::collective_try(
                [&] {
                    if (time_ + dt <= time_)
                        throw std::runtime_error("[SPH] Timestep cannot advance epoch");
                },
                "SPH timestep resolution");
            const auto bg0 = background(time_), bg1 = background(time_ + dt);
            std::vector<Stage> stage;
            detail::C::collective_try(
                [&] {
                    stage.resize(initial.size());
                    for (size_t i = 0; i < initial.size(); ++i) {
                        auto & q = stage[i];
                        q = {initial[i], initial[i], first.acceleration[i], first.energy_rate[i]};
                        for (int d = 0; d < D; ++d) {
                            q.particle.position[d] += dt * initial[i].velocity[d] / bg0.a;
                            q.particle.velocity[d] += dt * first.acceleration[i][d];
                        }
                        q.particle.energy += dt * first.energy_rate[i];
                        wrap(q.particle);
                    }
                },
                "SPH predictor");
            stage = migrate(stage, [](const Stage & q) -> const P & { return q.particle; });
            std::vector<P> trial;
            detail::C::collective_try(
                [&] {
                    trial.reserve(stage.size());
                    for (const auto & q : stage)
                        trial.push_back(q.particle);
                },
                "SPH stage particles");
            validate_particles(trial);
            auto second = rhs(trial, time_ + dt, external);
            detail::C::collective_try(
                [&] {
                    for (size_t i = 0; i < trial.size(); ++i) {
                        const auto & q = stage[i];
                        P final = q.initial;
                        for (int d = 0; d < D; ++d) {
                            final.position[d] +=
                                .5 * dt * (q.initial.velocity[d] / bg0.a + q.particle.velocity[d] / bg1.a);
                            final.velocity[d] += .5 * dt * (q.acceleration0[d] + second.acceleration[i][d]);
                        }
                        final.energy += .5 * dt * (q.energy_rate0 + second.energy_rate[i]);
                        final.h = trial[i].h;
                        wrap(final);
                        trial[i] = final;
                    }
                },
                "SPH corrector");
            validate_particles(trial);
            auto final = migrate(trial, [](const P & p) -> const P & { return p; });
            particles_.swap(final);
            time_ += dt;
            return dt;
        }

      public:
        void step(double dt, const ExternalForce & external = {}) { (void)advance_impl(dt, false, external); }
        /// Choose min(max_dt, global recommendation) and advance with two RHS evaluations.
        double advance(double max_dt, const ExternalForce & external = {}) {
            return advance_impl(max_dt, true, external);
        }
        Diagnostics<D> diagnostics() const {
            Diagnostics<D> out;
            for (const auto & p : particles_) {
                out.mass += p.mass;
                out.thermal += p.mass * p.energy;
                out.kinetic += .5 * p.mass * detail::dot<D>(p.velocity, p.velocity);
                for (int d = 0; d < D; ++d)
                    out.momentum[d] += p.mass * p.velocity[d];
            }
            FML::SumOverTasks(&out.mass);
            FML::SumOverTasks(&out.thermal);
            FML::SumOverTasks(&out.kinetic);
            for (auto & x : out.momentum)
                FML::SumOverTasks(&x);
            return out;
        }
    };
} // namespace FML::SPH
#endif
