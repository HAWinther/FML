#ifndef FML_FIELD_LEVEL_DISTRIBUTED_LPT_H
#define FML_FIELD_LEVEL_DISTRIBUTED_LPT_H
#include <FML/FieldLevel/DistributedLinear.h>
#include <FML/FieldLevel/DistributedAssignment.h>
#include <FML/FFTWGrid/FourierResampling.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    // Distributed 1LPT/2LPT evolution with the periodic framework's smoothed
    // Lagrangian bias, plane-parallel RSD and smooth/deconvolved mesh assignment.
    class DistributedLPTModel {
        Configuration config_;
        DistributedLinearModel linear_;
        std::vector<Parameter> parameters_;
        std::vector<double> kernel_ft_;
        static constexpr int pair_[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
        static Configuration linear_config(Configuration c) {
            c.evolution = Evolution::Linear;
            return c;
        }
        static std::vector<Parameter> linear_priors(const std::vector<Parameter> & priors) {
            std::vector<Parameter> out;
            for (const auto & p : priors)
                if (p.name != "b2" && p.name != "b_s2" && p.name != "b_laplacian")
                    out.push_back(p);
            return out;
        }
        void require(bool ok, const char * message) const { linear_.require(ok, message); }
        std::vector<double> base_parameters(const std::vector<double> & p) const {
            auto out = p;
            out.erase(out.end() - 4, out.end() - 1);
            return out;
        }
        double kernel(double z) const {
            const double r = 1 - z * z;
            return r <= 0 ? 0 : r * std::exp(2.3 * config_.kernel_width * (std::sqrt(r) - 1));
        }
        double kernel_derivative(double z) const {
            const double r = 1 - z * z;
            return r <= 0 ? 0 :
                            -z * std::exp(2.3 * config_.kernel_width * (std::sqrt(r) - 1)) *
                                (2 + 2.3 * config_.kernel_width * std::sqrt(r));
        }
        double assignment_factor(size_t i, bool observed = true) const {
            const auto & m = modes()[i];
            double f = observed ? linear_.response_[i] : 1.;
            for (int c = 0; c < 3; ++c)
                f /= kernel_ft_[std::abs(m.n[c])];
            return f;
        }
        static Complex displacement_kernel(const FieldLevelBAO::Mode & m, int c) {
            const double n2 = double(m.n[0]) * m.n[0] + double(m.n[1]) * m.n[1] + double(m.n[2]) * m.n[2];
            return {0, m.n[c] / (2 * M_PI * n2)};
        }
        static double tidal_kernel(const FieldLevelBAO::Mode & m, int a) {
            const double n2 = double(m.n[0]) * m.n[0] + double(m.n[1]) * m.n[1] + double(m.n[2]) * m.n[2];
            return double(m.n[pair_[a][0]]) * m.n[pair_[a][1]] / n2;
        }
        struct Taps {
            int node[9];
            double w[9], dw[9];
        };
        void taps(double u, Taps & t) const {
            const int n = config_.density_mesh;
            u -= n * std::floor(u / n);
            const double h = .5 * config_.kernel_width;
            const int start = int(std::floor(u - h)) + 1;
            for (int j = 0; j <= config_.kernel_width; ++j) {
                const int node = start + j;
                const double z = (node - u) / h;
                t.node[j] = ((node % n) + n) % n;
                t.w[j] = kernel(z);
                t.dw[j] = -kernel_derivative(z) / h;
            }
        }

      public:
        struct Workspace {
            friend class DistributedLPTModel;

          private:
            const DistributedLPTModel * owner;
            DistributedLinearModel::Workspace base;
            FML::GRID::FFTWGrid<3> density;
            FML::GRID::FourierResamplingPlan<3> up, down;
            DETAIL::DistributedAssignment assignment;
            std::array<Field, 3> psi;
            std::array<Field, 6> tidal, matter_tidal;
            Field delta, lap, weight, matter_delta;
            double mean_d2{}, mean_s2{};
            Modes prediction, cotangent;
            std::vector<double> parameters;
            bool valid{false};
            explicit Workspace(const DistributedLPTModel * m)
                : owner(m), base(m->linear_.workspace()), density(m->config_.density_mesh), up(base.grid, density),
                  down(density, base.grid), assignment(density) {}

          public:
            Workspace(Workspace &&) = default;
            Workspace & operator=(Workspace &&) = default;
            Workspace(const Workspace &) = delete;
            Workspace & operator=(const Workspace &) = delete;
            Modes & cotangent_buffer() { return cotangent; }
        };

      private:
        void check(const Workspace & w) const {
            require(w.owner == this && w.density.get_nmesh() == config_.density_mesh,
                    "[Distributed LPT] Foreign/moved workspace");
            linear_.check(w.base);
        }
        void particle_taps(size_t q, const Workspace & w, Taps (&t)[3]) const {
            const int n = config_.mesh;
            const int pos[3] = {int(q / (size_t(n) * n)) + layout().local_start, int(q / n % n), int(q % n)};
            for (int c = 0; c < 3; ++c) {
                // Wrap displacement before scaling: even very large finite shifts cannot overflow grid coordinates.
                const double shift = w.psi[c][q] - std::floor(w.psi[c][q]);
                taps((double(pos[c]) / n + shift) * config_.density_mesh, t[c]);
            }
        }
        template <class Visit>
        void stencil(size_t q, const Workspace & w, Visit visit) const {
            Taps t[3];
            particle_taps(q, w, t);
            const size_t n = config_.density_mesh;
            for (int a = 0; a <= config_.kernel_width; ++a)
                for (int b = 0; b <= config_.kernel_width; ++b)
                    for (int c = 0; c <= config_.kernel_width; ++c) {
                        const std::uint64_t node = (std::uint64_t(t[0].node[a]) * n + t[1].node[b]) * n + t[2].node[c];
                        visit(node,
                              t[0].w[a] * t[1].w[b] * t[2].w[c],
                              t[0].dw[a] * t[1].w[b] * t[2].w[c],
                              t[0].w[a] * t[1].dw[b] * t[2].w[c],
                              t[0].w[a] * t[1].w[b] * t[2].dw[c]);
                    }
        }
        template <class Kernel>
        Field field(const Modes & modes, Workspace & w, Kernel kernel) const {
            Modes coeff(modes.size());
            for (size_t i = 0; i < coeff.size(); ++i)
                coeff[i] = modes[i] * kernel(linear_.modes_[i]);
            return linear_.from_modes(coeff, w.base);
        }
        // Full initial-grid D = i n/(2 pi n^2), not truncated to the input band:
        // the quadratic source generates additional modes. D^T=-D on real cells.
        // Zero and every Nyquist plane are removed to preserve real/Hermitian fields.
        Field gradient_full(const Field & values, Workspace & w, int axis, int sign) const {
            auto & grid = w.base.grid;
            linear_.to_grid(values, w.base);
            grid.fftw_r2c();
            const int n = config_.mesh, half = n / 2;
            for (int ix = 0; ix < layout().local_width; ++ix)
                for (int iy = 0; iy < n; ++iy)
                    for (int iz = 0; iz <= half; ++iz) {
                        const int gx = ix + layout().local_start;
                        const int wave[3] = {gx <= half ? gx : gx - n, iy <= half ? iy : iy - n, iz};
                        const double n2 =
                            double(wave[0]) * wave[0] + double(wave[1]) * wave[1] + double(wave[2]) * wave[2];
                        const auto index = grid.get_index_fourier({ix, iy, iz});
                        const bool zero =
                            n2 == 0 || std::abs(wave[0]) == half || std::abs(wave[1]) == half || iz == half;
                        grid.set_fourier_from_index(
                            index,
                            zero ? FML::GRID::ComplexType(0) :
                                   grid.get_fourier_from_index(index) *
                                       FML::GRID::ComplexType(0, sign * wave[axis] / (2 * M_PI * n2)));
                    }
            grid.fftw_c2r();
            Field result(cells());
            size_t q = 0;
            for (auto i : grid.get_real_range())
                result[q++] = grid.get_real_from_index(i);
            return result;
        }
        Field second_order_source(const Workspace & w) const {
            Field source(cells());
            for (size_t q = 0; q < cells(); ++q) {
                double tt = 0;
                for (int a = 0; a < 6; ++a)
                    tt += (a < 3 ? 1 : 2) * w.matter_tidal[a][q] * w.matter_tidal[a][q];
                source[q] = .5 * (w.matter_delta[q] * w.matter_delta[q] - tt);
            }
            return source;
        }
        void deposit(Workspace & w, Modes & result, bool observed) const {
            const double fill = std::pow(double(config_.density_mesh), 3) / layout().global_cells;
            w.assignment.deposit(w.density, [&](auto add) {
                for (size_t q = 0; q < cells(); ++q)
                    stencil(q, w, [&](auto node, double weight, double, double, double) {
                        add(node, fill * w.weight[q] * weight);
                    });
            });
            w.density.fftw_r2c();
            w.down.apply(w.density, w.base.grid);
            result.resize(modes().size());
            for (size_t i = 0; i < modes().size(); ++i)
                result[i] = assignment_factor(i, observed) *
                            Complex(w.base.grid.get_fourier_from_index(modes()[i].fourier_index));
        }
        template <class Kernel>
        void add_adjoint(const Field & values, Workspace & w, Modes & result, Kernel kernel) const {
            linear_.to_grid(values, w.base);
            w.base.grid.fftw_r2c();
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += Complex(w.base.grid.get_fourier_from_index(modes()[i].fourier_index)) * kernel(modes()[i]);
        }

      public:
        DistributedLPTModel(Configuration c,
                            Spectrum spectrum,
                            FourierResponse response = {},
                            std::vector<Parameter> priors = {})
            : config_(c), linear_(linear_config(c), std::move(spectrum), std::move(response), linear_priors(priors)) {
            linear_.controls({double(c.evolution), double(c.density_mesh), double(c.kernel_width)});
            const size_t nd = c.density_mesh > 0 ? size_t(c.density_mesh) : 0;
            require((c.evolution == Evolution::Zeldovich || c.evolution == Evolution::LPT2) &&
                        c.density_mesh >= c.mesh && c.density_mesh % 2 == 0 && c.density_mesh % FML::NTasks == 0 &&
                        (c.kernel_width == 4 || c.kernel_width == 6 || c.kernel_width == 8) &&
                        c.initial_cutoff < .5 * M_PI * c.mesh / c.boxsize && nd > 0 &&
                        nd <= Field().max_size() / nd / nd,
                    "[Distributed LPT] Requires Zeldovich/2LPT, even divisible density mesh, width 4/6/8, and initial "
                    "cutoff below half Nyquist");
            parameters_ = linear_.parameters_;
            parameters_.insert(parameters_.end() - 1, {{"b2", 0, 0, 1}, {"b_s2", 0, 0, 1}, {"b_laplacian", 0, 0, 100}});
            bool valid = true;
            std::set<std::string> overridden, names;
            for (const auto & prior : priors) {
                if (!overridden.insert(prior.name).second)
                    valid = false;
                for (auto & parameter : parameters_)
                    if (parameter.name == prior.name)
                        parameter = prior;
            }
            std::vector<double> specs;
            for (const auto & p : parameters_) {
                valid &= !p.name.empty() && names.insert(p.name).second && std::isfinite(p.initial) &&
                         std::isfinite(p.mean) && std::isfinite(p.sigma) && p.sigma > 0 && p.lower < p.upper &&
                         p.initial >= p.lower && p.initial <= p.upper &&
                         (p.prior == Prior::Gaussian || (p.prior == Prior::Uniform && std::isfinite(p.lower) &&
                                                         std::isfinite(p.upper) && std::isfinite(p.upper - p.lower)));
                specs.insert(specs.end(),
                             {p.initial,
                              p.mean,
                              p.sigma,
                              double(p.prior),
                              std::isfinite(p.lower) ? p.lower : 0.,
                              std::isfinite(p.upper) ? p.upper : 0.,
                              double(std::isfinite(p.lower)),
                              double(std::isfinite(p.upper))});
            }
            require(valid, "[Distributed LPT] Invalid/duplicate parameter specification");
            linear_.controls(specs);
            kernel_ft_.resize(c.mesh / 2 + 1);
            const double h = .5 * c.kernel_width;
            for (size_t mode = 0; mode < kernel_ft_.size(); ++mode) {
                double integral = 0;
                const int nq = 4000;
                for (int j = 0; j <= nq; ++j) {
                    const double t = h * j / nq;
                    integral += (j == 0 || j == nq ? 1 :
                                 j % 2             ? 4 :
                                                     2) *
                                kernel(t / h) * std::cos(2 * M_PI * mode * t / c.density_mesh);
                }
                kernel_ft_[mode] = 2 * h * integral / (3 * nq);
                require(std::isfinite(kernel_ft_[mode]) && kernel_ft_[mode] != 0,
                        "[Distributed LPT] Singular assignment response");
            }
        }
        const Configuration & configuration() const { return config_; }
        const auto & parameters() const { return parameters_; }
        const std::vector<FieldLevelBAO::Mode> & modes() const { return linear_.modes(); }
        const auto & response() const { return linear_.response(); }
        Model::Layout layout() const { return linear_.layout(); }
        size_t cells() const { return linear_.cells(); }
        double volume() const { return linear_.volume(); }
        double sum(double x) const { return linear_.sum(x); }
        double maximum(double x) const { return linear_.maximum(x); }
        void controls(const std::vector<double> & values) const { linear_.controls(values); }
        void validate_modes(const Modes & data) const { linear_.validate_modes(data); }
        Workspace workspace() const { return Workspace(this); }
        Field white_noise(std::uint64_t seed) const { return linear_.white_noise(seed); }
        std::vector<double> initial_parameters() const {
            std::vector<double> p;
            for (auto & x : parameters_)
                p.push_back(x.initial);
            return p;
        }
        void validate_state(const Field & s, const std::vector<double> & p) const {
            require(s.size() == cells() && p.size() == parameters_.size(),
                    "[Distributed LPT] State dimension mismatch");
        }
        bool in_support(const std::vector<double> & p) const {
            require(p.size() == parameters_.size(), "[Distributed LPT] Parameter dimension mismatch");
            bool ok = true;
            for (size_t j = 0; j < p.size(); ++j)
                ok &= std::isfinite(p[j]) && p[j] >= parameters_[j].lower && p[j] <= parameters_[j].upper;
            if (maximum(ok ? 0 : 1) != 0)
                return false;
            controls(p);
            return true;
        }
        const Modes & predict(const Field & s, const std::vector<double> & p, Workspace & w) const {
            validate_state(s, p);
            check(w);
            w.valid = false;
            require(in_support(p), "[Distributed LPT] Parameters outside support");
            bool finite = true;
            for (double x : s)
                finite &= std::isfinite(x);
            if (maximum(finite ? 0 : 1) != 0)
                throw UnrepresentableState("[Distributed LPT] Nonfinite initial field");
            linear_.predict(s, base_parameters(p), w.base);
            const size_t offset = linear_.spectrum_.parameters.size();
            Modes dk(modes().size());
            for (size_t i = 0; i < dk.size(); ++i)
                dk[i] = std::sqrt(w.base.power[i].value * layout().global_cells / volume()) * w.base.white_modes[i];
            for (int c = 0; c < 3; ++c)
                w.psi[c] = field(dk, w, [&](const auto & m) {
                    return displacement_kernel(m, c) * (c == 2 ? 1 + config_.growth_rate : 1);
                });
            if (config_.evolution == Evolution::LPT2) {
                w.matter_delta = field(dk, w, [](const auto &) { return Complex(1); });
                for (int a = 0; a < 6; ++a)
                    w.matter_tidal[a] = field(dk, w, [&](const auto & m) { return Complex(tidal_kernel(m, a)); });
                const auto source = second_order_source(w);
                // EdS Psi2 = (3/7) D[S2]; RSD adds 2 f Psi2_z.
                for (int c = 0; c < 3; ++c) {
                    const auto psi2 = gradient_full(source, w, c, +1);
                    const double factor = 3. / 7 * (c == 2 ? 1 + 2 * config_.growth_rate : 1);
                    for (size_t q = 0; q < cells(); ++q)
                        w.psi[c][q] += factor * psi2[q];
                }
            }
            const auto keep = [&](const auto & m) { return m.k < config_.bias_cutoff; };
            w.delta = field(dk, w, [&](const auto & m) { return Complex(keep(m) ? 1 : 0); });
            w.lap = field(dk, w, [&](const auto & m) { return Complex(keep(m) ? -m.k * m.k : 0); });
            for (int a = 0; a < 6; ++a)
                w.tidal[a] = field(dk, w, [&](const auto & m) { return Complex(keep(m) ? tidal_kernel(m, a) : 0); });
            double d2 = 0, s2 = 0;
            for (size_t q = 0; q < cells(); ++q) {
                double tt = 0;
                for (int a = 0; a < 6; ++a)
                    tt += (a < 3 ? 1 : 2) * w.tidal[a][q] * w.tidal[a][q];
                d2 += w.delta[q] * w.delta[q];
                s2 += tt - w.delta[q] * w.delta[q] / 3;
            }
            w.mean_d2 = sum(d2) / layout().global_cells;
            w.mean_s2 = sum(s2) / layout().global_cells;
            w.weight.resize(cells());
            for (size_t q = 0; q < cells(); ++q) {
                double tt = 0;
                for (int a = 0; a < 6; ++a)
                    tt += (a < 3 ? 1 : 2) * w.tidal[a][q] * w.tidal[a][q];
                w.weight[q] = 1 + (p[offset] - 1) * w.delta[q] + p[offset + 1] * (w.delta[q] * w.delta[q] - w.mean_d2) +
                              p[offset + 2] * (tt - w.delta[q] * w.delta[q] / 3 - w.mean_s2) + p[offset + 3] * w.lap[q];
                finite &= std::isfinite(w.weight[q]);
                for (int c = 0; c < 3; ++c)
                    finite &= std::isfinite(w.psi[c][q]);
            }
            if (maximum(finite ? 0 : 1) != 0)
                throw UnrepresentableState("[Distributed LPT] Unrepresentable particle state");
            deposit(w, w.prediction, true);
            w.parameters = p;
            w.valid = true;
            return w.prediction;
        }
        void pullback(const std::vector<double> & p,
                      const Modes & g,
                      Workspace & w,
                      Field & gs,
                      std::vector<double> & gp) const {
            check(w);
            require(w.valid && w.parameters == p && g.size() == modes().size(),
                    "[Distributed LPT] Pullback needs matching forward");
            const double nd3 = std::pow(double(config_.density_mesh), 3), fill = nd3 / layout().global_cells;
            w.base.grid.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t i = 0; i < g.size(); ++i)
                w.base.grid.set_fourier_from_index(modes()[i].fourier_index,
                                                   FML::GRID::ComplexType(assignment_factor(i) * g[i] / nd3));
            w.base.grid.set_grid_status_real(false);
            w.up.apply(w.base.grid, w.density);
            w.density.fftw_c2r();
            const auto reversed = w.assignment.reverse(w.density);
            std::array<Field, 3> xbar;
            for (auto & v : xbar)
                v.assign(cells(), 0);
            Field wbar(cells(), 0);
            for (size_t q = 0; q < cells(); ++q)
                stencil(q, w, [&](auto node, double weight, double dx, double dy, double dz) {
                    const double bar = fill * w.assignment.at(reversed, node);
                    wbar[q] += bar * weight;
                    const double factor = bar * w.weight[q] * config_.density_mesh;
                    xbar[0][q] += factor * dx;
                    xbar[1][q] += factor * dy;
                    xbar[2][q] += factor * dz;
                });
            Modes Q(modes().size(), 0);
            for (int c = 0; c < 3; ++c)
                add_adjoint(xbar[c], w, Q, [&](const auto & m) {
                    return std::conj(displacement_kernel(m, c)) * (c == 2 ? 1 + config_.growth_rate : 1);
                });
            if (config_.evolution == Evolution::LPT2) {
                Field source_bar(cells(), 0);
                for (int c = 0; c < 3; ++c) {
                    const auto back = gradient_full(xbar[c], w, c, -1);
                    const double factor = 3. / 7 * (c == 2 ? 1 + 2 * config_.growth_rate : 1);
                    for (size_t q = 0; q < cells(); ++q)
                        source_bar[q] += factor * back[q];
                }
                Field back(cells());
                for (size_t q = 0; q < cells(); ++q)
                    back[q] = source_bar[q] * w.matter_delta[q];
                add_adjoint(back, w, Q, [](const auto &) { return Complex(1); });
                for (int a = 0; a < 6; ++a) {
                    for (size_t q = 0; q < cells(); ++q)
                        back[q] = -(a < 3 ? 1 : 2) * source_bar[q] * w.matter_tidal[a][q];
                    add_adjoint(back, w, Q, [&](const auto & m) { return Complex(tidal_kernel(m, a)); });
                }
            }
            const size_t offset = linear_.spectrum_.parameters.size();
            gp.assign(p.size(), 0);
            const double meanw = sum(std::accumulate(wbar.begin(), wbar.end(), 0.)) / layout().global_cells;
            Field db(cells());
            std::array<Field, 6> tb;
            for (auto & v : tb)
                v.resize(cells());
            for (size_t q = 0; q < cells(); ++q) {
                const double d = w.delta[q], centered = wbar[q] - meanw;
                double tt = 0;
                db[q] = (p[offset] - 1) * wbar[q] + 2 * (p[offset + 1] - p[offset + 2] / 3) * d * centered;
                for (int a = 0; a < 6; ++a) {
                    tb[a][q] = (a < 3 ? 2 : 4) * p[offset + 2] * w.tidal[a][q] * centered;
                    tt += (a < 3 ? 1 : 2) * w.tidal[a][q] * w.tidal[a][q];
                }
                gp[offset] += wbar[q] * d;
                gp[offset + 1] += wbar[q] * (d * d - w.mean_d2);
                gp[offset + 2] += wbar[q] * (tt - d * d / 3 - w.mean_s2);
                gp[offset + 3] += wbar[q] * w.lap[q];
            }
            const auto keep = [&](const auto & m) { return m.k < config_.bias_cutoff; };
            add_adjoint(db, w, Q, [&](const auto & m) { return Complex(keep(m) ? 1 : 0); });
            for (int a = 0; a < 6; ++a)
                add_adjoint(tb[a], w, Q, [&](const auto & m) { return Complex(keep(m) ? tidal_kernel(m, a) : 0); });
            add_adjoint(wbar, w, Q, [&](const auto & m) { return Complex(keep(m) ? -p[offset + 3] * m.k * m.k : 0); });
            Modes coeff(Q.size());
            for (size_t i = 0; i < Q.size(); ++i) {
                const double T = std::sqrt(w.base.power[i].value * layout().global_cells / volume());
                coeff[i] = T * Q[i];
                if (T != 0)
                    for (size_t j = 0; j < offset; ++j)
                        gp[j] += .5 * layout().global_cells * modes()[i].weight *
                                 std::real(std::conj(Q[i]) * T * w.base.white_modes[i]) * w.base.power[i].dlog[j];
            }
            gs = linear_.from_modes(coeff, w.base); // gp contains local contributions; posterior reduces once.
        }
        Field apply_field_metric(const Field & s, const std::vector<double> & p, Workspace & w, double exponent) const {
            validate_state(s, p);
            check(w);
            require(in_support(p), "[Distributed LPT] Invalid metric parameters");
            return linear_.apply_field_metric(s, base_parameters(p), w.base, exponent);
        }
        // Physical real-space outputs, independent of tracer bias, observation response and RSD.
        // velocity_factor=a H f L; Lagrangian displacement samples are in box units.
        Model::Reconstruction
        reconstruct(const Field & s, const std::vector<double> & p, Workspace & w, double velocity_factor) const {
            require(std::isfinite(velocity_factor), "[Distributed LPT] Invalid velocity factor");
            controls({velocity_factor});
            Model::Reconstruction result;
            result.tracer = predict(s, p, w);
            w.valid = false; // Subsequent scratch/deposition uses real-space, unbiased particles.
            Modes dk(modes().size());
            for (size_t i = 0; i < dk.size(); ++i)
                dk[i] = std::sqrt(w.base.power[i].value * layout().global_cells / volume()) * w.base.white_modes[i];
            result.initial_density = config_.evolution == Evolution::LPT2 ?
                                         w.matter_delta :
                                         field(dk, w, [](const auto &) { return Complex(1); });
            Field source;
            if (config_.evolution == Evolution::LPT2)
                source = second_order_source(w);
            bool finite = true;
            for (auto z : result.tracer)
                finite &= std::isfinite(z.real()) && std::isfinite(z.imag());
            for (double x : result.initial_density)
                finite &= std::isfinite(x);
            for (int c = 0; c < 3; ++c) {
                const auto first = field(dk, w, [&](const auto & m) { return displacement_kernel(m, c); });
                Field second;
                if (config_.evolution == Evolution::LPT2)
                    second = gradient_full(source, w, c, +1);
                result.displacement[c].resize(cells());
                result.velocity[c].resize(cells());
                for (size_t q = 0; q < cells(); ++q) {
                    const double psi2 = second.empty() ? 0 : 3. / 7 * second[q];
                    result.displacement[c][q] = first[q] + psi2;
                    result.velocity[c][q] = velocity_factor * (first[q] + 2 * psi2);
                    finite &= std::isfinite(result.displacement[c][q]) && std::isfinite(result.velocity[c][q]);
                }
                w.psi[c] = result.displacement[c];
            }
            if (maximum(finite ? 0 : 1) != 0)
                throw UnrepresentableState("[Distributed LPT] Unrepresentable reconstruction");
            w.weight.assign(cells(), 1.);
            Modes matter;
            deposit(w, matter, false);
            result.matter_density = linear_.from_modes(matter, w.base);
            for (double x : result.matter_density)
                finite &= std::isfinite(x);
            if (maximum(finite ? 0 : 1) != 0)
                throw UnrepresentableState("[Distributed LPT] Unrepresentable matter reconstruction");
            return result;
        }
        Modes mock(const Field & s, const std::vector<double> & p, std::uint64_t seed, Workspace & w) const {
            auto result = predict(s, p, w);
            linear_.to_grid(white_noise(seed), w.base);
            w.base.grid.fftw_r2c();
            const double amplitude = std::sqrt(std::exp(p.back()) * layout().global_cells / volume());
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += amplitude * Complex(w.base.grid.get_fourier_from_index(modes()[i].fourier_index));
            return result;
        }
        FML::MCMC::HMCExecution hmc_execution() const { return Execution<DistributedLinearModel>::hmc(linear_); }
    };
    template <>
    struct Execution<DistributedLPTModel> {
        static void validate(const DistributedLPTModel & m, const Field & s, const std::vector<double> & p) {
            m.validate_state(s, p);
        }
        static double sum(const DistributedLPTModel & m, double x) { return m.sum(x); }
        static double maximum(const DistributedLPTModel & m, double x) { return m.maximum(x); }
        static void map_controls(const DistributedLPTModel & m, bool vary, int iterations, double tolerance) {
            m.controls({double(vary), double(iterations), tolerance});
        }
        static void hmc_controls(const DistributedLPTModel & m, bool vary, const FML::MCMC::HMCOptions & o) {
            m.controls({double(vary),
                        double(o.warmup),
                        double(o.samples),
                        double(o.n_leapfrog),
                        double(o.jitter),
                        o.step_size,
                        double(o.adapt_step_size),
                        o.target_acceptance,
                        double(o.adapt_mass),
                        o.divergence_threshold,
                        double(o.seed >> 32),
                        double(o.seed & 0xffffffffu),
                        double(o.verbose_every)});
        }
        template <class Likelihood>
        static Likelihood likelihood(const DistributedLPTModel & m, Modes data) {
            if constexpr (std::is_same_v<Likelihood, GaussianFieldLikelihood>)
                return {m.modes(),
                        m.configuration().likelihood_cutoff,
                        m.volume(),
                        std::move(data),
                        [&](const Modes & d) { m.validate_modes(d); }};
            else
                return {m.modes(), m.configuration().likelihood_cutoff, m.volume(), std::move(data)};
        }
        static FML::MCMC::HMCExecution hmc(const DistributedLPTModel & m) { return m.hmc_execution(); }
    };
    using DistributedLPTPosterior = BasicPosterior<DistributedLPTModel>;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
