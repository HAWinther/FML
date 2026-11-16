#ifndef FML_FIELD_LEVEL_PARTICLE_MESH_H
#define FML_FIELD_LEVEL_PARTICLE_MESH_H
#include <FML/FieldLevel/DistributedLinear.h>
#include <FML/FieldLevel/DistributedAssignment.h>
#include <FML/FieldLevel/PMAssignment.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    template <class GridType> class ParticleMeshAdjointT;
    // Fixed flat matter + Lambda background; a=1 is the spectrum/growth reference.
    // Coordinates are fractions of the box; p=a^2 dx/d(H0 t).
    struct PMBackground {
        double omega_m{.3};
        double E(double a) const { return std::sqrt(omega_m / (a * a * a) + 1 - omega_m); }
        template <class Function>
        static double integrate(Function f, double a, double b, int panels = 128) {
            const double h = (b - a) / panels;
            double sum = f(a) + f(b);
            for (int j = 1; j < panels; ++j)
                sum += (j % 2 ? 4 : 2) * f(a + j * h);
            return sum * h / 3;
        }
        // x=a*t^2 removes the early matter-era endpoint singular derivatives.
        double growth_integral(double a) const {
            return integrate([&](double t) {
                if (t == 0) return 0.;
                const double x = a * t * t, e = E(x);
                return 2 * a * t / (x * x * x * e * e * e);
            }, 0., 1.);
        }
        double growth(double a) const {
            return omega_m == 1 ? a : E(a) * growth_integral(a) / growth_integral(1.);
        }
        double growth_rate(double a) const {
            if (omega_m == 1) return 1;
            const double e = E(a);
            return -1.5 * omega_m / (a * a * a * e * e) + 1 / (a * a * e * e * e * growth_integral(a));
        }
        double kick(double a, double b) const {
            if (omega_m == 1) return 3 * (std::sqrt(b) - std::sqrt(a));
            return 1.5 * omega_m * integrate([&](double x) {
                const double scale = std::exp(x);
                return 1 / (scale * E(scale));
            }, std::log(a), std::log(b));
        }
        double drift(double a, double b) const {
            if (omega_m == 1) return 2 * (1 / std::sqrt(a) - 1 / std::sqrt(b));
            return integrate([&](double x) {
                const double scale = std::exp(x);
                return 1 / (scale * scale * E(scale));
            }, std::log(a), std::log(b));
        }
    };
    // Compensation removes the selected deposit and gather windows in the force,
    // while the reported density retains its assignment window.
    enum class PMForceWindow { None = 0, CICCompensated = 1, TSCCompensated = 2 };
    struct PMSettings {
        PMBackground background;
        std::vector<double> scale_factors{.05, .1, .2, .4, .7, 1.};
        PMForceWindow force_window{PMForceWindow::None};
        PMAssignment assignment{PMAssignment::CIC};
    };
    struct PMDiagnostic {
        double a{}, mass_error{}, density_rms{};
        std::array<double, 3> mean_momentum{};
    };

    // Periodic forward map. ParticleMeshAdjoint.h adds its discrete pullback;
    // particles retain their Lagrangian owners. No PM posterior API is implied.
    template <class GridType = FML::GRID::FFTWGrid<3>>
    class ParticleMeshForwardT {
        friend class ParticleMeshAdjointT<GridType>;
      public:
        using Grid = GridType;
        using Linear = DistributedLinearModelT<Grid>;
        struct Result {
            Model::Layout particle_layout, density_layout;
            Field initial_density, matter_density;
            std::array<Field, 3> displacement, momentum;
            std::vector<PMDiagnostic> diagnostics;
            double a{};
            std::array<double, 3> position(size_t q) const {
                const auto c = particle_layout.global_coordinate(q);
                std::array<double, 3> x;
                for (int d = 0; d < 3; ++d) {
                    x[d] = double(c[d]) / particle_layout.global_mesh + displacement[d].at(q);
                    x[d] -= std::floor(x[d]);
                }
                return x;
            }
        };
        struct Workspace {
            friend class ParticleMeshForwardT;
            friend class ParticleMeshAdjointT<Grid>;
          private:
            const ParticleMeshForwardT * owner;
            typename Linear::Workspace initial;
            Grid density;
            std::array<Grid, 3> force;
            DETAIL::DistributedAssignmentT<Grid> assignment;
            Result output;
            Field input_white;
            std::vector<double> input_parameters;
            bool valid{false};
            explicit Workspace(const ParticleMeshForwardT * m)
                : owner(m), initial(m->linear_.workspace()),
                  density(FML::GRID::make_grid<Grid>(m->configuration().density_mesh, 0, 0, m->domain())),
                  assignment(density) {
                for (auto & g : force)
                    g = FML::GRID::make_grid<Grid>(m->configuration().density_mesh, 0, 0, m->domain());
            }
          public:
            Workspace(Workspace &&) = default;
            Workspace & operator=(Workspace &&) = default;
            Workspace(const Workspace &) = delete;
            Workspace & operator=(const Workspace &) = delete;
            const Result & result() const {
                if (!valid) throw std::logic_error("[Field PM] No successful evolution in workspace");
                return output;
            }
        };
      private:
        Linear linear_;
        PMSettings settings_;
        std::vector<Parameter> parameters_;
        struct Step { double first_kick, drift, last_kick; };
        std::vector<Step> steps_;
        void require(bool ok, const char * message) const {
            if (linear_.maximum(ok ? 0 : 1) != 0) throw std::invalid_argument(message);
        }
        template <class Visit>
        void stencil(const Result & r, size_t q, Visit visit) const {
            const int n = configuration().density_mesh;
            const auto x = r.position(q);
            if (settings_.assignment == PMAssignment::TSC) {
                DETAIL::tsc_stencil(n, x, [&](auto node, double weight, const auto &) { visit(node, weight); });
                return;
            }
            int base[3];
            double f[3];
            for (int d = 0; d < 3; ++d) {
                const double u = x[d] * n;
                base[d] = int(std::floor(u));
                f[d] = u - base[d];
            }
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    for (int k = 0; k < 2; ++k) {
                        const auto node = (std::uint64_t((base[0] + i) % n) * n + (base[1] + j) % n) * n +
                                          (base[2] + k) % n;
                        visit(node, (i ? f[0] : 1 - f[0]) * (j ? f[1] : 1 - f[1]) * (k ? f[2] : 1 - f[2]));
                    }
        }
        void density(Workspace & w) const {
            const double ratio = double(configuration().density_mesh) / configuration().mesh;
            w.assignment.deposit(w.density, [&](auto add) {
                for (size_t q = 0; q < cells(); ++q)
                    stencil(w.output, q, [&](auto node, double weight) { add(node, weight * ratio * ratio * ratio); });
            });
            for (auto i : w.density.get_real_range())
                w.density.set_real_from_index(i, w.density.get_real_from_index(i) - 1.);
            w.density.set_grid_status_real(true);
        }
        template <class Wave>
        double force_coefficient(const Wave & wave, double square, int component, bool transpose = false) const {
            double value = (transpose ? -wave[component] : wave[component]) / (2 * M_PI * square);
            if (settings_.force_window != PMForceWindow::None) {
                // Each deposit/gather window is product sinc^order, order 2 or 3.
                // The real, even compensation preserves the skew-adjoint force.
                double window = 1.;
                for (int k : wave) if (k != 0) {
                    const double x = M_PI * k / configuration().density_mesh;
                    const double sinc = std::sin(x) / x;
                    window *= sinc * sinc;
                    if (settings_.assignment == PMAssignment::TSC) window *= sinc;
                }
                value /= window * window;
            }
            return value;
        }
        void force(Workspace & w) const {
            w.density.fftw_r2c();
            // Same continuous i*k/k^2 kernel as NBody, without importing its
            // unrelated GSL/ODE dependencies. Optional matched assignment compensation. Each even
            // Nyquist derivative component is zero, preserving Hermitian symmetry.
            for (int d = 0; d < 3; ++d) {
                w.force[d].fill_fourier_grid(0.);
                w.force[d].set_grid_status_real(false);
            }
            const int n = configuration().density_mesh;
            for (auto i : w.density.get_fourier_range()) {
                const auto wave = w.density.get_fourier_integer_wavevector_from_index(i);
                double square = 0;
                for (int k : wave) square += double(k) * k;
                if (square == 0) continue;
                for (int d = 0; d < 3; ++d)
                    if (std::abs(wave[d]) != n / 2)
                        w.force[d].set_fourier_from_index(i, w.density.get_fourier_from_index(i) *
                            FML::GRID::ComplexType(0, force_coefficient(wave, square, d)));
            }
            for (auto & g : w.force) g.fftw_c2r();
        }
        void kick(Workspace & w, double integral) const {
            for (int d = 0; d < 3; ++d) {
                const auto nodes = w.assignment.reverse(w.force[d]);
                for (size_t q = 0; q < cells(); ++q) {
                    double acceleration = 0;
                    stencil(w.output, q, [&](auto node, double weight) {
                        acceleration += weight * w.assignment.at(nodes, node);
                    });
                    w.output.momentum[d][q] += integral * acceleration;
                }
            }
        }
        void finite(const Result & r) const {
            bool ok = true;
            for (int d = 0; d < 3; ++d)
                for (size_t q = 0; q < cells(); ++q)
                    ok &= std::isfinite(r.displacement[d][q]) && std::isfinite(r.momentum[d][q]);
            if (linear_.maximum(ok ? 0 : 1) != 0)
                throw UnrepresentableState("[Field PM] Nonfinite particle state");
        }
        void diagnostic(Workspace & w, double a) const {
            PMDiagnostic d;
            d.a = a;
            double mass = 0, square = 0;
            bool ok = true;
            for (auto i : w.density.get_real_range()) {
                const double delta = w.density.get_real_from_index(i);
                mass += delta + 1;
                square += delta * delta;
                ok &= std::isfinite(delta);
            }
            if (linear_.maximum(ok ? 0 : 1) != 0)
                throw UnrepresentableState("[Field PM] Nonfinite density");
            const double n = configuration().density_mesh;
            d.mass_error = linear_.sum(mass) / (n * n * n) - 1;
            d.density_rms = std::sqrt(linear_.sum(square) / (n * n * n));
            require(std::isfinite(d.mass_error) && std::isfinite(d.density_rms),
                    "[Field PM] Unrepresentable density diagnostics");
            w.output.diagnostics.push_back(d);
        }
        void momentum_diagnostic(Workspace & w) const {
            auto & d = w.output.diagnostics.back();
            for (int c = 0; c < 3; ++c) {
                double p = 0;
                for (double x : w.output.momentum[c]) p += x;
                d.mean_momentum[c] = linear_.sum(p) / linear_.layout().global_cells;
            }
            require(std::isfinite(d.mean_momentum[0]) && std::isfinite(d.mean_momentum[1]) &&
                        std::isfinite(d.mean_momentum[2]), "[Field PM] Unrepresentable momentum diagnostics");
        }
      public:
        ParticleMeshForwardT(Configuration initial, Spectrum spectrum, PMSettings settings = {},
                             const FML::Domain & domain = FML::domain)
            : linear_(initial, std::move(spectrum), {}, {}, domain), settings_(std::move(settings)) {
            // The supplied configuration describes the initial-field generator.
            // A denser particle/IC lattice may sample a coarser force mesh. Keep
            // the initial band below BOTH Nyquists; linear_ checks the IC one.
            require(initial.growth_rate == 0 && initial.density_mesh >= 4 && initial.density_mesh % 2 == 0 &&
                        initial.initial_cutoff < M_PI * initial.density_mesh / initial.boxsize &&
                        (!FML::GRID::is_fftwgrid_v<Grid> || initial.density_mesh % FML::NTasks == 0),
                    "[Field PM] Requires real-space band-limited ICs and a supported even force mesh >= 4");
            const size_t n = size_t(initial.density_mesh);
            require(n <= Field().max_size() / n / n, "[Field PM] Force mesh storage overflow");
            linear_.controls({settings_.background.omega_m, double(settings_.scale_factors.size()),
                              double(static_cast<int>(settings_.force_window)), double(static_cast<int>(settings_.assignment))});
            require(settings_.assignment == PMAssignment::CIC || settings_.assignment == PMAssignment::TSC,
                    "[Field PM] Unsupported assignment");
            require(settings_.force_window == PMForceWindow::None ||
                        (settings_.force_window == PMForceWindow::CICCompensated && settings_.assignment == PMAssignment::CIC) ||
                        (settings_.force_window == PMForceWindow::TSCCompensated && settings_.assignment == PMAssignment::TSC),
                    "[Field PM] Force compensation must match assignment");
            require(settings_.background.omega_m > 0 && settings_.background.omega_m <= 1 &&
                        settings_.scale_factors.size() >= 2 && settings_.scale_factors.size() <= 100001,
                    "[Field PM] Requires 0 < Omega_m <= 1 and 2..100001 scale-factor nodes");
            linear_.controls(settings_.scale_factors);
            bool valid = true;
            for (size_t j = 0; j < settings_.scale_factors.size(); ++j) {
                const double a = settings_.scale_factors[j];
                valid &= a >= 1e-4 && a <= 1 && (j == 0 || a > settings_.scale_factors[j - 1]);
            }
            require(valid, "[Field PM] Scale factors must increase in [1e-4,1]");
            // Check common cell edges before sparse mesh ownership is constructed.
            if (!domain.is_slab()) {
                const auto & p = domain.template get<FML::PencilDomain>();
                const auto dims = p.process_grid();
                valid &= dims[0] <= initial.density_mesh && dims[1] <= initial.density_mesh / 2 + 1;
                for (const auto & edges : {p.x_edges(), p.y_edges()})
                    for (double edge : edges)
                        valid &= std::abs(edge * initial.density_mesh - std::round(edge * initial.density_mesh)) < 1e-9;
            }
            require(valid, "[Field PM] Process grid must fit the force mesh and domain edges must align");
            FML::COMMUNICATION::collective_try([&] {
                parameters_.assign(linear_.parameters().begin(), linear_.parameters().end() - 2);
                for (size_t j = 1; j < settings_.scale_factors.size(); ++j) {
                    const double a = settings_.scale_factors[j - 1], b = settings_.scale_factors[j], mid = .5 * (a + b);
                    steps_.push_back({settings_.background.kick(a, mid), settings_.background.drift(a, b),
                                      settings_.background.kick(mid, b)});
                }
            }, "Field PM fixed schedule");
        }
        const Configuration & configuration() const { return linear_.configuration(); }
        const PMSettings & settings() const { return settings_; }
        const FML::Domain & domain() const { return linear_.domain(); }
        auto layout() const { return linear_.layout(); }
        size_t cells() const { return linear_.cells(); }
        const auto & parameters() const { return parameters_; }
        double sum(double x) const { return linear_.sum(x); }
        double maximum(double x) const { return linear_.maximum(x); }
        Field white_noise(std::uint64_t seed) const { return linear_.white_noise(seed); }
        std::vector<double> initial_parameters() const {
            auto p = linear_.initial_parameters(); p.resize(parameters_.size()); return p;
        }
        Workspace workspace() const { return Workspace(this); }
        struct NoObserver {};
        const Result & evolve(const Field & white, const std::vector<double> & theta, Workspace & w) const {
            return evolve_observed(white,theta,w,NoObserver{});
        }
        // Read-only local callback at each position/force node, before deposition.
        // All ranks participate; callbacks must not call collectives. NoObserver
        // compiles away, preserving the default arithmetic and communication path.
        template<class Observer>
        const Result & evolve_observed(const Field & white, const std::vector<double> & theta, Workspace & w,
                                       const Observer & observer) const {
            require(w.owner == this, "[Field PM] Foreign/moved workspace");
            w.valid = false;
            require(theta.size() == parameters_.size(), "[Field PM] Spectrum parameter dimension mismatch");
            auto p = linear_.initial_parameters();
            std::copy(theta.begin(), theta.end(), p.begin());
            linear_.validate_state(white, p);
            FML::COMMUNICATION::collective_try([&] {
                // Cache before mutating output: inputs may refer to an earlier result.
                w.input_white = white;
                w.input_parameters = theta;
            }, "Field PM cached input");
            {
                auto initial = linear_.reconstruct(w.input_white, p, w.initial, 0.);
                w.output.initial_density = std::move(initial.initial_density);
                w.output.displacement = std::move(initial.displacement);
            }
            auto & r = w.output;
            r.particle_layout = layout();
            r.density_layout = {};
            r.density_layout.global_mesh = configuration().density_mesh;
            const size_t n = configuration().density_mesh;
            r.density_layout.global_cells = n * n * n;
            r.density_layout.owned_cells = 1;
            for (int d = 0; d < 3; ++d) {
                const auto range = w.density.get_local_range(d);
                r.density_layout.local_starts[d] = int(range[0]);
                r.density_layout.local_counts[d] = int(range[1]);
                r.density_layout.owned_cells *= size_t(range[1]);
            }
            r.density_layout.local_start = r.density_layout.local_starts[0];
            r.density_layout.local_width = r.density_layout.local_counts[0];
            const double a0 = settings_.scale_factors.front(), D = settings_.background.growth(a0);
            const double velocity = a0 * a0 * settings_.background.E(a0) * settings_.background.growth_rate(a0);
            FML::COMMUNICATION::collective_try([&] {
                r.diagnostics.clear(); r.diagnostics.reserve(settings_.scale_factors.size());
                for (int d = 0; d < 3; ++d) {
                    r.momentum[d].resize(cells());
                    for (size_t q = 0; q < cells(); ++q) {
                        r.displacement[d][q] *= D;
                        r.momentum[d][q] = velocity * r.displacement[d][q];
                    }
                }
            }, "Field PM particle state");
            finite(r);
            auto observe=[&](size_t node) {
                if constexpr(!std::is_same_v<Observer,NoObserver>)
                    FML::COMMUNICATION::collective_try([&]{observer(node,static_cast<const Result &>(r));},"PM read-only observer");
            };
            observe(0);
            density(w); diagnostic(w, a0); momentum_diagnostic(w); force(w);
            for (size_t j = 0; j < steps_.size(); ++j) {
                kick(w, steps_[j].first_kick);
                for (int d = 0; d < 3; ++d)
                    for (size_t q = 0; q < cells(); ++q)
                        r.displacement[d][q] += steps_[j].drift * r.momentum[d][q];
                finite(r);
                observe(j+1);
                density(w); diagnostic(w, settings_.scale_factors[j + 1]);
                force(w); kick(w, steps_[j].last_kick);
                finite(r);
                momentum_diagnostic(w);
            }
            // Only the final output needs an inverse density FFT; diagnostics
            // were accumulated before each density transform, at unchanged positions.
            w.density.fftw_c2r();
            FML::COMMUNICATION::collective_try([&] {
                r.matter_density.resize(r.density_layout.owned_cells);
                size_t q = 0;
                for (auto i : w.density.get_real_range()) r.matter_density[q++] = w.density.get_real_from_index(i);
            }, "Field PM output");
            r.a = settings_.scale_factors.back();
            w.valid = true;
            return r;
        }
    };
    using ParticleMeshForward = ParticleMeshForwardT<>;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
