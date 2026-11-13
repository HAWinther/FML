#ifndef FML_HYDRODYNAMICS_FINITE_VOLUME_HEADER
#define FML_HYDRODYNAMICS_FINITE_VOLUME_HEADER
#include <FML/Hydrodynamics/CartesianGrid.h>
#include <FML/Hydrodynamics/Riemann.h>
#include <functional>
#include <memory>
namespace FML::HYDRODYNAMICS {
    template <int D>
    struct Options {
        std::array<int, D> shape;
        Vector<D> lengths;
        std::array<std::array<Boundary, 2>, D> boundaries{};
        double gamma = 5. / 3, courant = .35, slope_theta = 1.5, acceleration_factor = .2;
        int order = 2, max_retries = 16;
        FluxMethod flux = FluxMethod::HLLC;
        DualEnergy dual{};
        size_t max_local_cells = 10000000, max_message_bytes = 8 * 1024 * 1024;
        Options() {
            shape.fill(32);
            lengths.fill(1);
            for (auto & b : boundaries)
                b.fill(Boundary::Periodic);
        }
    };
    struct Expansion {
        double a = 1, H = 0;
    };
    struct Statistics {
        unsigned long long low_order_stages = 0, rejected_attempts = 0, riemann_fallbacks = 0,
                           reconstructed_faces_reset = 0, dual_cells = 0;
        double energy_repair = 0, absolute_energy_repair = 0;
        double initial_energy_repair = 0, initial_absolute_energy_repair = 0;
    };
    template <int D>
    struct Diagnostics {
        double mass = 0, energy = 0, kinetic = 0, thermal = 0;
        Vector<D> momentum{};
    };
    template <int D>
    class FiniteVolume {
        static_assert(D >= 1 && D <= 3);

      public:
        using U = Conserved<D>;
        using Initializer = std::function<Primitive<D>(const Vector<D> &)>;
        using Background = std::function<Expansion(double)>;
        using Gravity =
            std::function<void(double, const Layout<D> &, const std::vector<Primitive<D>> &, std::vector<Vector<D>> &)>;

      private:
        Options<D> options_;
        IdealGas eos_;
        std::unique_ptr<CartesianGrid<D>> grid_;
        std::vector<U> cells_;
        std::vector<Primitive<D>> padded_primitive_;
        std::array<std::vector<FluxResult<D>>, D> faces_;
        std::array<std::array<int, D>, D> face_shapes_{};
        Background background_;
        double time_ = 0;
        Statistics stats_;
        struct RHS {
            std::vector<U> value;
            double timestep = std::numeric_limits<double>::infinity();
            unsigned long long fallbacks = 0, resets = 0;
            bool valid = true;
        };
        struct Repairs {
            double signed_energy = 0, absolute_energy = 0;
            unsigned long long cells = 0;
        };
        Expansion expansion(double t) const {
            Expansion b;
            detail::C::collective_try(
                [&] {
                    if (!std::isfinite(t))
                        throw std::invalid_argument("[Hydro] Invalid epoch");
                    b = background_ ? background_(t) : Expansion{};
                    if (!(b.a > 0 && std::isfinite(b.a) && std::isfinite(b.H)))
                        throw std::invalid_argument("[Hydro] Invalid background");
                    if constexpr (D != 3)
                        if (b.a != 1 || b.H != 0)
                            throw std::invalid_argument("[Hydro] Expansion requires three dimensions");
                },
                "Hydro background");
            detail::agree(std::array<double, 3>{t, b.a, b.H}, "[Hydro] Background/epoch differs between ranks");
            return b;
        }
        void validate_options() {
            detail::C::collective_try(
                [&] {
                    if (!(std::isfinite(options_.gamma) && options_.gamma > 1 && options_.courant > 0 &&
                          options_.courant <= .45 && std::isfinite(options_.courant) && options_.slope_theta >= 1 &&
                          options_.slope_theta <= 2 && std::isfinite(options_.slope_theta) &&
                          options_.acceleration_factor > 0 && std::isfinite(options_.acceleration_factor)) ||
                        (options_.order != 1 && options_.order != 2) || options_.max_retries < 0 ||
                        options_.max_retries > 60 || options_.max_local_cells < 1 ||
                        options_.max_local_cells > 1000000000 ||
                        options_.max_message_bytes < sizeof(U) + sizeof(size_t) || int(options_.flux) < 0 ||
                        int(options_.flux) > 2 ||
                        !(options_.dual.pressure_threshold > 0 &&
                          options_.dual.pressure_threshold <= options_.dual.synchronization_threshold &&
                          options_.dual.synchronization_threshold < 1))
                        throw std::invalid_argument("[Hydro] Invalid solver options");
                    size_t count = 1;
                    for (int d = 0; d < D; ++d) {
                        if (options_.shape[d] < 1 || options_.shape[d] > std::numeric_limits<int>::max() - 4 ||
                            !std::isfinite(options_.lengths[d]) || options_.lengths[d] <= 0 ||
                            !(options_.lengths[d] / options_.shape[d] > 0))
                            throw std::invalid_argument("[Hydro] Invalid mesh");
                        count = detail::product(count, size_t(options_.shape[d]));
                        for (auto b : options_.boundaries[d])
                            if (int(b) < 0 || int(b) > 2)
                                throw std::invalid_argument("[Hydro] Invalid boundary");
                        if ((options_.boundaries[d][0] == Boundary::Periodic) !=
                            (options_.boundaries[d][1] == Boundary::Periodic))
                            throw std::invalid_argument("[Hydro] Periodic boundaries must be paired");
                    }
                    (void)count;
                },
                "Hydro configuration");
            std::array<double, 13 + 4 * D> values{};
            size_t j = 0;
            for (int d = 0; d < D; ++d) {
                values[j++] = options_.shape[d];
                values[j++] = options_.lengths[d];
                for (auto b : options_.boundaries[d])
                    values[j++] = int(b);
            }
            for (double x : {options_.gamma,
                             options_.courant,
                             options_.slope_theta,
                             options_.acceleration_factor,
                             double(options_.order),
                             double(options_.max_retries),
                             double(int(options_.flux)),
                             double(options_.dual.enabled),
                             options_.dual.pressure_threshold,
                             options_.dual.synchronization_threshold,
                             double(options_.max_local_cells),
                             double(D),
                             1.})
                values[j++] = x;
            detail::agree(values, "[Hydro] Configuration differs between ranks");
            eos_ = IdealGas(options_.gamma);
        }
        static double minmod(double a, double b, double c) {
            if (a > 0 && b > 0 && c > 0)
                return std::min({a, b, c});
            if (a < 0 && b < 0 && c < 0)
                return std::max({a, b, c});
            return 0;
        }
        Primitive<D> reconstructed(std::array<int, D> q, int axis, double sign, int order, bool & reset) const {
            const auto center = padded_primitive_[grid_->padded_index(q)];
            if (order == 1)
                return center;
            auto l = q, r = q;
            --l[axis];
            ++r[axis];
            const auto & a = padded_primitive_[grid_->padded_index(l)];
            const auto & b = padded_primitive_[grid_->padded_index(r)];
            auto value = [&](double left, double middle, double right) {
                return middle + sign * .5 *
                                    minmod(options_.slope_theta * (middle - left),
                                           .5 * (right - left),
                                           options_.slope_theta * (right - middle));
            };
            Primitive<D> result;
            result.density = value(a.density, center.density, b.density);
            result.pressure = value(a.pressure, center.pressure, b.pressure);
            for (int d = 0; d < D; ++d)
                result.velocity[d] = value(a.velocity[d], center.velocity[d], b.velocity[d]);
            if (!FML::HYDRODYNAMICS::valid(result, eos_)) {
                reset = true;
                return center;
            }
            return result;
        }
        double reconstructed_auxiliary(std::array<int, D> q, int axis, double sign, int order, bool & reset) const {
            auto value = [&](std::array<int, D> p) { return grid_->padded()[grid_->padded_index(p)][D + 2]; };
            double center = value(q);
            if (order == 1)
                return center;
            auto left = q, right = q;
            --left[axis];
            ++right[axis];
            double a = value(left), b = value(right);
            double face =
                center +
                sign * .5 *
                    minmod(options_.slope_theta * (center - a), .5 * (b - a), options_.slope_theta * (b - center));
            if (!(face > 0 && std::isfinite(face))) {
                reset = true;
                return center;
            }
            return face;
        }
        RHS rhs(const std::vector<U> & state, double epoch, int order, FluxMethod method, const Gravity & gravity) {
            detail::agree(std::array<double, 1>{double(bool(gravity))},
                          "[Hydro] Gravity callback presence differs between ranks");
            const auto bg = expansion(epoch);
            grid_->exchange(state);
            const auto & layout = grid_->layout();
            RHS out;
            std::vector<Primitive<D>> primitive_owned;
            std::vector<Vector<D>> extra;
            detail::C::collective_try(
                [&] {
                    padded_primitive_.resize(grid_->padded().size());
                    for (size_t i = 0; i < padded_primitive_.size(); ++i) {
                        auto q = recover<D>(grid_->padded()[i], eos_, options_.dual);
                        if (!q.valid)
                            throw std::runtime_error("[Hydro] Invalid halo primitive");
                        padded_primitive_[i] = q.state;
                    }
                    out.value.resize(state.size());
                    primitive_owned.resize(state.size());
                    extra.resize(state.size());
                    for (size_t i = 0; i < state.size(); ++i)
                        primitive_owned[i] = recover<D>(state[i], eos_, options_.dual).state;
                    for (int axis = 0; axis < D; ++axis) {
                        face_shapes_[axis] = layout.owned_shape;
                        ++face_shapes_[axis][axis];
                        faces_[axis].resize(layout.width ? detail::count<D>(face_shapes_[axis]) : 0);
                    }
                },
                "Hydro RHS buffers");
            std::array<double, D> speed{};
            int bad = 0;
            for (int axis = 0; axis < D; ++axis) {
                double maxspeed = 0;
                unsigned long long fallbacks = 0, resets = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : maxspeed, bad) reduction(+ : fallbacks, resets)
#endif
                for (size_t face = 0; face < faces_[axis].size(); ++face) {
                    auto right = detail::coordinates<D>(face, face_shapes_[axis]), left = right;
                    --left[axis];
                    bool reset = false;
                    auto ql = reconstructed(left, axis, 1, order, reset),
                         qr = reconstructed(right, axis, -1, order, reset);
                    std::array<double, 2> auxiliary = {-1, -1};
                    if (options_.dual.enabled)
                        auxiliary = {reconstructed_auxiliary(left, axis, 1, order, reset),
                                     reconstructed_auxiliary(right, axis, -1, order, reset)};
                    auto f = riemann<D>(ql, qr, eos_, axis, method, auxiliary);
                    faces_[axis][face] = f;
                    if (!f.valid)
                        bad = 1;
                    maxspeed = std::max(maxspeed, f.max_speed);
                    fallbacks += f.fallback != FluxFallback::None;
                    resets += reset;
                }
                speed[axis] = maxspeed;
                out.fallbacks += fallbacks;
                out.resets += resets;
            }
            FML::MaxOverTasks(&bad);
            out.valid = !bad;
            for (auto & v : speed)
                FML::MaxOverTasks(&v);
            if (!out.valid)
                return out;
            if (gravity)
                gravity(epoch, layout, primitive_owned, extra);
            detail::C::collective_try(
                [&] {
                    if (extra.size() != state.size())
                        throw std::invalid_argument("[Hydro] Gravity output shape mismatch");
                    for (const auto & v : extra)
                        for (auto x : v)
                            if (!std::isfinite(x))
                                throw std::invalid_argument("[Hydro] Nonfinite external acceleration");
                },
                "Hydro gravity output");
            double inverse = 0;
            for (int d = 0; d < D; ++d)
                inverse += speed[d] / (bg.a * layout.spacing[d]);
            if (inverse > 0)
                out.timestep = options_.courant / inverse;
            if (bg.H != 0)
                out.timestep = std::min(out.timestep, .1 / (std::abs(bg.H) * std::max(1., D * (options_.gamma - 1))));
            double acceleration_step = std::numeric_limits<double>::infinity();
#ifdef USE_OMP
#pragma omp parallel for reduction(max : bad) reduction(min : acceleration_step)
#endif
            for (size_t i = 0; i < state.size(); ++i) {
                auto q = detail::coordinates<D>(i, layout.owned_shape);
                double divergence = 0;
                U derivative{};
                for (int axis = 0; axis < D; ++axis) {
                    auto right = q;
                    ++right[axis];
                    const auto & fl = faces_[axis][detail::index<D>(q, face_shapes_[axis])];
                    const auto & fr = faces_[axis][detail::index<D>(right, face_shapes_[axis])];
                    const double factor = 1 / (bg.a * layout.spacing[axis]);
                    for (int j = 0; j < D + 3; ++j)
                        derivative[j] -= (fr.flux[j] - fl.flux[j]) * factor;
                    divergence += (fr.interface_velocity - fl.interface_velocity) * factor;
                }
                const auto & p = primitive_owned[i];
                const double thermal = p.pressure / (options_.gamma - 1);
                double work = 0, g2 = 0, min_dx = layout.spacing[0];
                for (int d = 0; d < D; ++d) {
                    derivative[1 + d] += -bg.H * state[i][1 + d] + p.density * extra[i][d];
                    work += state[i][1 + d] * extra[i][d];
                    g2 += extra[i][d] * extra[i][d];
                    min_dx = std::min(min_dx, layout.spacing[d]);
                }
                derivative[D + 1] +=
                    work - 2 * bg.H * double(kinetic_energy<D>(state[i])) - D * (options_.gamma - 1) * bg.H * thermal;
                derivative[D + 2] += -p.pressure * (divergence + D * bg.H);
                if (g2 > 0)
                    acceleration_step =
                        std::min(acceleration_step,
                                 options_.acceleration_factor * std::sqrt(bg.a * min_dx / std::sqrt(g2)));
                for (auto v : derivative)
                    if (!std::isfinite(v))
                        bad = 1;
                out.value[i] = derivative;
            }
            FML::MaxOverTasks(&bad);
            FML::MinOverTasks(&acceleration_step);
            out.valid = !bad;
            out.timestep = std::min(out.timestep, acceleration_step);
            return out;
        }
        bool normalize_cells(std::vector<U> & state, Repairs & repairs) const {
            int bad = 0;
            double change = 0, absolute = 0;
            unsigned long long number = 0;
            for (auto & u : state) {
                double delta = 0;
                const auto q = recover<D>(u, eos_, options_.dual);
                number += q.used_auxiliary;
                if (!normalize<D>(u, eos_, options_.dual, delta))
                    bad = 1;
                change += delta;
                absolute += std::abs(delta);
            }
            FML::MaxOverTasks(&bad);
            FML::SumOverTasks(&change);
            FML::SumOverTasks(&absolute);
            FML::SumOverTasks(&number);
            repairs = {change * grid_->layout().cell_volume(), absolute * grid_->layout().cell_volume(), number};
            return !bad;
        }
        void account(const RHS & r) {
            auto f = r.fallbacks, n = r.resets;
            FML::SumOverTasks(&f);
            FML::SumOverTasks(&n);
            stats_.riemann_fallbacks += f;
            stats_.reconstructed_faces_reset += n;
        }
        bool euler_stage(const std::vector<U> & base,
                         double t,
                         double dt,
                         const Gravity & gravity,
                         RHS & high,
                         std::vector<U> & result,
                         Repairs & repair) {
            auto trial = [&](const RHS & r) {
                if (!r.valid)
                    return false;
                detail::C::collective_try(
                    [&] {
                        result.resize(base.size());
                        for (size_t i = 0; i < base.size(); ++i)
                            for (int j = 0; j < D + 3; ++j)
                                result[i][j] = base[i][j] + dt * r.value[i][j];
                    },
                    "Hydro Euler trial");
                return normalize_cells(result, repair);
            };
            if (trial(high))
                return true;
            ++stats_.low_order_stages;
            auto low = rhs(base, t, 1, FluxMethod::Rusanov, gravity);
            account(low);
            return trial(low);
        }
        double advance_impl(double requested, bool adaptive, const Gravity & gravity) {
            detail::C::collective_try(
                [&] {
                    if (!(requested > 0 && std::isfinite(requested) && std::isfinite(time_ + requested) &&
                          time_ + requested > time_))
                        throw std::invalid_argument("[Hydro] Invalid timestep");
                },
                "Hydro timestep request");
            detail::agree(std::array<double, 2>{time_, requested}, "[Hydro] Timestep/epoch differs between ranks");
            auto first = rhs(cells_, time_, options_.order, options_.flux, gravity);
            account(first);
            if (!first.valid) {
                ++stats_.low_order_stages;
                first = rhs(cells_, time_, 1, FluxMethod::Rusanov, gravity);
                account(first);
                if (!first.valid)
                    throw std::runtime_error("[Hydro] Initial flux/rates are unrepresentable");
            }
            double dt = adaptive ? std::min(requested, first.timestep) : requested;
            for (int retry = 0; retry <= (adaptive ? options_.max_retries : 0); ++retry) {
                if (!(dt > 0 && std::isfinite(dt) && time_ + dt > time_))
                    throw std::runtime_error("[Hydro] Timestep cannot advance epoch");
                std::vector<U> stage, next;
                Repairs r1, r2, rf;
                bool good = euler_stage(cells_, time_, dt, gravity, first, stage, r1);
                if (good) {
                    auto second = rhs(stage, time_ + dt, options_.order, options_.flux, gravity);
                    account(second);
                    good = euler_stage(stage, time_ + dt, dt, gravity, second, next, r2);
                }
                if (good) {
                    for (size_t i = 0; i < next.size(); ++i)
                        for (int j = 0; j < D + 3; ++j)
                            next[i][j] = .5 * cells_[i][j] + .5 * next[i][j];
                    good = normalize_cells(next, rf);
                }
                if (good) {
                    cells_.swap(next);
                    time_ += dt;
                    stats_.energy_repair += .5 * r1.signed_energy + .5 * r2.signed_energy + rf.signed_energy;
                    stats_.absolute_energy_repair +=
                        .5 * r1.absolute_energy + .5 * r2.absolute_energy + rf.absolute_energy;
                    stats_.dual_cells += r1.cells + r2.cells + rf.cells;
                    return dt;
                }
                ++stats_.rejected_attempts;
                dt *= .5;
            }
            throw std::runtime_error("[Hydro] Positivity retry exhausted; state and epoch unchanged");
        }

      public:
        FiniteVolume(Options<D> options = {}, Initializer initial = {}, Background background = {})
            : options_(options), background_(std::move(background)) {
            validate_options();
            grid_ = std::make_unique<CartesianGrid<D>>(options_.shape,
                                                       options_.lengths,
                                                       options_.boundaries,
                                                       options_.max_local_cells,
                                                       options_.max_message_bytes);
            detail::C::collective_try(
                [&] {
                    cells_.resize(grid_->layout().cells);
                    for (size_t i = 0; i < cells_.size(); ++i) {
                        auto p = initial ? initial(grid_->layout().center(i)) : Primitive<D>{};
                        if (!encode<D>(p, eos_, cells_[i]))
                            throw std::invalid_argument("[Hydro] Initial primitive is unrepresentable");
                    }
                },
                "Hydro initial state");
            Repairs r;
            if (!normalize_cells(cells_, r))
                throw std::invalid_argument("[Hydro] Initial conserved state nonphysical; dual energy may be required");
            stats_.initial_energy_repair = r.signed_energy;
            stats_.initial_absolute_energy_repair = r.absolute_energy;
        }
        const Layout<D> & layout() const { return grid_->layout(); }
        const std::vector<U> & cells() const { return cells_; }
        const Options<D> & options() const { return options_; }
        const Statistics & statistics() const { return stats_; }
        double time() const { return time_; }
        std::vector<Primitive<D>> primitive_cells() const {
            std::vector<Primitive<D>> q;
            q.reserve(cells_.size());
            for (const auto & u : cells_)
                q.push_back(primitive<D>(u, eos_, options_.dual));
            return q;
        }
        void set_state(std::vector<U> state, double epoch = 0) {
            detail::C::collective_try(
                [&] {
                    if (state.size() != cells_.size() || !std::isfinite(epoch))
                        throw std::invalid_argument("[Hydro] Initial cell shape/epoch mismatch");
                },
                "Hydro replacement state");
            detail::agree(std::array<double, 1>{epoch}, "[Hydro] Replacement epochs differ");
            Repairs r;
            if (!normalize_cells(state, r))
                throw std::invalid_argument("[Hydro] Invalid replacement cells");
            cells_.swap(state);
            time_ = epoch;
            stats_ = {};
            stats_.initial_energy_repair = r.signed_energy;
            stats_.initial_absolute_energy_repair = r.absolute_energy;
        }
        void step(double dt, const Gravity & gravity = {}) { (void)advance_impl(dt, false, gravity); }
        double advance(double max_dt, const Gravity & gravity = {}) { return advance_impl(max_dt, true, gravity); }
        Diagnostics<D> diagnostics() const {
            Diagnostics<D> result;
            const double volume = layout().cell_volume();
            for (const auto & u : cells_) {
                result.mass += u[0] * volume;
                result.energy += u[D + 1] * volume;
                result.kinetic += double(kinetic_energy<D>(u)) * volume;
                result.thermal += primitive<D>(u, eos_, options_.dual).pressure / (options_.gamma - 1) * volume;
                for (int d = 0; d < D; ++d)
                    result.momentum[d] += u[d + 1] * volume;
            }
            FML::SumOverTasks(&result.mass);
            FML::SumOverTasks(&result.energy);
            FML::SumOverTasks(&result.kinetic);
            FML::SumOverTasks(&result.thermal);
            for (auto & p : result.momentum)
                FML::SumOverTasks(&p);
            return result;
        }
    };
} // namespace FML::HYDRODYNAMICS
#endif
