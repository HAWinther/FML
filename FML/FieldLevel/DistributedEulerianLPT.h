#ifndef FML_FIELD_LEVEL_DISTRIBUTED_EULERIAN_LPT_H
#define FML_FIELD_LEVEL_DISTRIBUTED_EULERIAN_LPT_H
#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/FieldLevel/EulerianBias.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    // Amplitude-only spectrum at the analysis epoch. The reference shape and all other cosmology are fixed.
    inline Spectrum sigma8_spectrum(std::function<double(double)> reference_power) {
        return {{{"sigma8_ratio", 1, 1, 1, .5, 1.5, Prior::Uniform}},
                [reference_power = std::move(reference_power)](double k, const std::vector<double> & p) {
                    if (p.size() != 1 || !(p[0] > 0) || !std::isfinite(p[0]))
                        throw std::invalid_argument("[EulerianLPT] Invalid sigma8 ratio");
                    return Power{p[0] * p[0] * reference_power(k), {2 / p[0]}};
                }};
    }
    struct EulerianLPTConfiguration {
        int mesh{16}, density_mesh{32}, kernel_width{6};
        double boxsize{400}, initial_cutoff{.04}; // Initial band is spherical in this first implementation.
        Evolution evolution{Evolution::LPT2};
        MatterAssignment matter_assignment{MatterAssignment::Smooth};
        double nufft_epsilon{1e-8};
        EULERIAN::SharpKBand matter_band{.035, EULERIAN::BandGeometry::Sphere};
        EULERIAN::SharpKBand likelihood_band{.03, EULERIAN::BandGeometry::Cube};
        int product_mesh{0}; // 0 keeps the original shared mesh; otherwise padded LPT/operator mesh.
    };

    // Real-space unbiased LPT matter -> Eulerian quadratic bias -> explicitly selected Fourier modes.
    // Select smooth assignment or PSWF NUFFT with its matched discrete position adjoint.
    template <class GridType = FML::GRID::FFTWGrid<3>>
    class DistributedEulerianLPTModelT {
      public:
        using Grid = GridType;
        using Matter = DistributedLPTModelT<Grid>;
        using Bias = EULERIAN::QuadraticBias<Grid>;

      private:
        const EulerianLPTConfiguration config_;
        const size_t spectrum_size_;
        const Spectrum spectrum_;
        Matter matter_;
        std::unique_ptr<DistributedLinearModelT<Grid>> compact_;
        Bias bias_;
        std::vector<Parameter> parameters_;
        std::vector<FieldLevelBAO::Mode> modes_;
        static void require(bool ok, const char * message) {
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        static int product_size(const EulerianLPTConfiguration & c) {
            return c.product_mesh == 0 ? c.mesh : c.product_mesh;
        }
        static Configuration gravity_config(const EulerianLPTConfiguration & c) {
            auto valid_band = [](const auto & b) {
                return std::isfinite(b.cutoff) && b.cutoff > 0 &&
                       (b.geometry == EULERIAN::BandGeometry::Sphere || b.geometry == EULERIAN::BandGeometry::Cube);
            };
            require(valid_band(c.matter_band) && valid_band(c.likelihood_band) &&
                        c.likelihood_band.cutoff <= c.matter_band.cutoff,
                    "[EulerianLPT] Invalid matter/likelihood bands; require likelihood cutoff <= matter cutoff");
            require(std::isfinite(c.boxsize) && c.boxsize > 0 && c.mesh >= 4 && c.mesh % 2 == 0 &&
                        product_size(c) >= c.mesh && product_size(c) % 2 == 0 &&
                        std::isfinite(c.initial_cutoff) && c.initial_cutoff > 0 &&
                        c.initial_cutoff < M_PI * c.mesh / c.boxsize &&
                        c.likelihood_band.cutoff < M_PI * c.mesh / c.boxsize &&
                        c.matter_band.cutoff <= M_PI * product_size(c) / (2 * c.boxsize),
                    "[EulerianLPT] Matter band requires a larger product mesh");
            Configuration g;
            g.mesh = product_size(c);
            g.density_mesh = c.density_mesh;
            g.kernel_width = c.kernel_width;
            g.matter_assignment = c.matter_assignment;
            g.nufft_epsilon = c.nufft_epsilon;
            g.boxsize = c.boxsize;
            g.initial_cutoff = c.initial_cutoff;
            g.bias_cutoff = c.initial_cutoff;
            g.growth_rate = 0;
            g.evolution = c.evolution;
            // A radial envelope includes every requested matter-band corner, then the explicit band is applied.
            const double reach =
                c.matter_band.cutoff * (c.matter_band.geometry == EULERIAN::BandGeometry::Cube ? std::sqrt(3.) : 1.);
            g.likelihood_cutoff = std::nextafter(std::max(c.initial_cutoff, reach), HUGE_VAL);
            return g;
        }
        template <class F>
        static void local(F && f) {
            FML::COMMUNICATION::collective_try(std::forward<F>(f), "[EulerianLPT] Local preparation");
        }
        std::vector<double> matter_parameters(const std::vector<double> & p) const {
            std::vector<double> q;
            local([&] {
                q = matter_.initial_parameters();
                std::copy(p.begin(), p.begin() + spectrum_size_, q.begin());
                q[spectrum_size_] = 1.; // Unbiased matter: every particle carries unit weight.
                for (size_t j = spectrum_size_ + 1; j < q.size(); ++j)
                    q[j] = 0.;
            });
            return q;
        }
        void finite_modes(const Modes & x, size_t size) const {
            bool ok = x.size() == size;
            for (auto z : x)
                ok &= std::isfinite(z.real()) && std::isfinite(z.imag());
            require(ok, "[EulerianLPT] Invalid local Fourier dimensions/values");
        }
        Field real_field(Grid & grid, size_t count = 0) const {
            Field f;
            local([&] {
                f.resize(count ? count : matter_.cells());
                size_t q = 0;
                for (auto i : grid.get_real_range())
                    f[q++] = grid.get_real_from_index(i);
            });
            return f;
        }
        void to_grid(const Field & f, Grid & grid) const {
            size_t q = 0;
            for (auto i : grid.get_real_range())
                grid.set_real_from_index(i, f[q++]);
            grid.set_grid_status_real(true);
        }

      public:
        struct Workspace {
            friend class DistributedEulerianLPTModelT;

          private:
            const DistributedEulerianLPTModelT * owner;
            typename Matter::Workspace matter;
            Grid grid, compact_grid;
            std::unique_ptr<DETAIL::LPTResampling<Grid>> lift_plan, restrict_plan;
            Field lifted;
            typename Bias::Workspace bias;
            Modes prediction, cotangent;
            std::vector<double> parameters;
            std::vector<std::pair<size_t, double>> metric_log_precision;
            std::vector<double> metric_parameters, metric_weights;
            double metric_log_noise{};
            bool metric_valid{false};
            bool valid{false};
            explicit Workspace(const DistributedEulerianLPTModelT * m)
                : owner(m), matter(m->matter_.workspace()),
                  grid(FML::GRID::make_grid<Grid>(m->product_mesh(), 0, 0, m->domain())), bias(m->bias_.workspace(grid)) {
                if (m->compact_) {
                    compact_grid = FML::GRID::make_grid<Grid>(m->config_.mesh, 0, 0, m->domain());
                    lift_plan = std::make_unique<DETAIL::LPTResampling<Grid>>(compact_grid, grid);
                    restrict_plan = std::make_unique<DETAIL::LPTResampling<Grid>>(grid, compact_grid);
                }
            }

          public:
            Workspace(Workspace && other) noexcept
                : owner(std::exchange(other.owner, nullptr)), matter(std::move(other.matter)),
                  grid(std::move(other.grid)), compact_grid(std::move(other.compact_grid)),
                  lift_plan(std::move(other.lift_plan)), restrict_plan(std::move(other.restrict_plan)),
                  lifted(std::move(other.lifted)), bias(std::move(other.bias)), prediction(std::move(other.prediction)),
                  cotangent(std::move(other.cotangent)), parameters(std::move(other.parameters)),
                  metric_log_precision(std::move(other.metric_log_precision)),
                  metric_parameters(std::move(other.metric_parameters)), metric_weights(std::move(other.metric_weights)), metric_log_noise(other.metric_log_noise),
                  metric_valid(std::exchange(other.metric_valid, false)),
                  valid(std::exchange(other.valid, false)) {}
            Workspace(const Workspace &) = delete;
            Modes & cotangent_buffer() { return cotangent; }
        };

      private:
        void check(const Workspace & w) const { require(w.owner == this, "[EulerianLPT] Foreign/moved workspace"); }
        double white_gain() const { return std::pow(double(config_.mesh) / product_mesh(), 1.5); }
        const Field & lift_white(const Field & s, Workspace & w) const {
            if (!compact_) return s;
            to_grid(s, w.compact_grid);
            w.compact_grid.fftw_r2c();
            w.lift_plan->apply(w.compact_grid, w.grid);
            w.grid.fftw_c2r();
            w.lifted = real_field(w.grid);
            const double gain = white_gain();
            for (auto & v : w.lifted) v *= gain;
            return w.lifted;
        }
        Field restrict_white_bar(const Field & bar, Workspace & w) const {
            to_grid(bar, w.grid);
            w.grid.fftw_r2c();
            w.restrict_plan->apply(w.grid, w.compact_grid);
            w.compact_grid.fftw_c2r();
            auto result = real_field(w.compact_grid, cells());
            // Euclidean transpose: N_product/N_compact times the forward gain.
            const double inverse_gain = 1 / white_gain();
            for (auto & v : result) v *= inverse_gain;
            return result;
        }
        const Grid & observation_fourier(Workspace & w) const {
            if (!compact_) return w.grid;
            w.restrict_plan->apply(w.grid, w.compact_grid);
            return w.compact_grid;
        }

      public:
        // Supply all four bias priors explicitly, including physical units for the Laplacian coefficient.
        DistributedEulerianLPTModelT(EulerianLPTConfiguration c,
                                     Spectrum spectrum,
                                     std::array<Parameter, 4> bias_parameters,
                                     const FML::Domain & domain = FML::domain)
            : config_(c), spectrum_size_(spectrum.parameters.size()), spectrum_(spectrum),
              matter_(gravity_config(c), spectrum, {}, {}, domain),
              bias_(FML::GRID::make_grid<Grid>(product_size(c), 0, 0, domain), c.boxsize, c.matter_band), parameters_() {
            controls({double(c.mesh), double(c.product_mesh), c.likelihood_band.cutoff,
                      double(c.likelihood_band.geometry)});
            if (c.mesh != product_mesh()) {
                auto lc = gravity_config(c);
                lc.mesh = c.mesh;
                lc.evolution = Evolution::Linear;
                lc.matter_assignment = MatterAssignment::Smooth;
                lc.likelihood_cutoff = c.initial_cutoff;
                compact_ = std::make_unique<DistributedLinearModelT<Grid>>(lc, spectrum, FourierResponse{},
                                                                         std::vector<Parameter>{}, domain);
            }
            local([&] { parameters_ = spectrum.parameters; });
            const std::array<std::string, 4> names = {"b_delta", "b_delta2", "b_K2", "b_laplacian"};
            bool valid = true;
            std::vector<double> specs;
            local([&] {
                parameters_.insert(parameters_.end(), bias_parameters.begin(), bias_parameters.end());
                std::set<std::string> unique;
                for (size_t j = 0; j < parameters_.size(); ++j) {
                    const auto & p = parameters_[j];
                    valid &=
                        unique.insert(p.name).second && !p.name.empty() && std::isfinite(p.initial) &&
                        std::isfinite(p.mean) && std::isfinite(p.sigma) && p.sigma > 0 && p.lower < p.upper &&
                        p.initial >= p.lower && p.initial <= p.upper &&
                        (p.prior == Prior::Gaussian || (p.prior == Prior::Uniform && std::isfinite(p.lower) &&
                                                        std::isfinite(p.upper) && std::isfinite(p.upper - p.lower)));
                    if (j >= spectrum_size_)
                        valid &= p.name == names[j - spectrum_size_];
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
            });
            require(valid, "[EulerianLPT] Invalid bias/spectrum parameter specification");
            controls(specs);
            auto grid = FML::GRID::make_grid<Grid>(c.mesh, 0, 0, domain);
            local([&] {
                for (auto i : grid.get_fourier_range()) {
                    const auto n = grid.get_fourier_integer_wavevector_from_index(i);
                    if (!c.likelihood_band.contains(n, c.boxsize, c.mesh))
                        continue;
                    double n2 = 0;
                    for (int v : n)
                        n2 += double(v) * v;
                    modes_.push_back({{n[0], n[1], n[2]},
                                      2 * M_PI * std::sqrt(n2) / c.boxsize,
                                      double(FML::GRID::fourier_mode_multiplicity(n[2], c.mesh)),
                                      size_t(i),
                                      0,
                                      0,
                                      0});
                }
            });
        }
        const EulerianLPTConfiguration & configuration() const { return config_; }
        int assignment_width() const { return matter_.assignment_width(); }
        int product_mesh() const { return product_size(config_); }
        const auto & parameters() const { return parameters_; }
        const auto & modes() const { return modes_; }
        const auto & domain() const { return matter_.domain(); }
        auto layout() const { return compact_ ? compact_->layout() : matter_.layout(); }
        size_t cells() const { return compact_ ? compact_->cells() : matter_.cells(); }
        double volume() const { return matter_.volume(); }
        double sum(double x) const { return matter_.sum(x); }
        double maximum(double x) const { return matter_.maximum(x); }
        void controls(const std::vector<double> & x) const { matter_.controls(x); }
        Workspace workspace() const { return Workspace(this); }
        Field white_noise(uint64_t seed) const {
            return compact_ ? compact_->white_noise(seed) : matter_.white_noise(seed);
        }
        std::vector<double> initial_parameters() const {
            std::vector<double> p;
            local([&] {
                for (const auto & v : parameters_)
                    p.push_back(v.initial);
            });
            return p;
        }
        void validate_state(const Field & s, const std::vector<double> & p) const {
            require(s.size() == cells() && p.size() == parameters_.size(), "[EulerianLPT] State dimension mismatch");
        }
        bool in_support(const std::vector<double> & p) const {
            require(p.size() == parameters_.size(), "[EulerianLPT] Scalar dimension mismatch");
            bool valid = true;
            for (size_t j = 0; j < p.size(); ++j)
                valid &= std::isfinite(p[j]) && p[j] >= parameters_[j].lower && p[j] <= parameters_[j].upper;
            if (maximum(valid ? 0 : 1))
                return false;
            controls(p);
            return true;
        }
        void validate_modes(const Modes & data) const {
            finite_modes(data, modes_.size());
            auto grid = FML::GRID::make_grid<Grid>(config_.mesh, 0, 0, domain());
            grid.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t j = 0; j < data.size(); ++j)
                grid.set_fourier_from_index(modes_[j].fourier_index, data[j]);
            grid.set_grid_status_real(false);
            grid.fftw_c2r();
            grid.fftw_r2c();
            bool valid = true;
            for (size_t j = 0; j < data.size(); ++j)
                valid &= std::abs(Complex(grid.get_fourier_from_index(modes_[j].fourier_index)) - data[j]) <=
                         1e-10 * std::max(1., std::abs(data[j]));
            require(valid, "[EulerianLPT] Modes must preserve Hermitian symmetry across ranks");
        }
        const Modes & predict(const Field & s, const std::vector<double> & p, Workspace & w) const {
            check(w);
            w.valid = false;
            validate_state(s, p);
            require(in_support(p), "[EulerianLPT] Scalars outside support");
            const auto & mk = matter_.predict(lift_white(s, w), matter_parameters(p), w.matter);
            w.grid.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t j = 0; j < mk.size(); ++j)
                w.grid.set_fourier_from_index(matter_.modes()[j].fourier_index, mk[j]);
            w.grid.set_grid_status_real(false);
            w.grid.fftw_c2r();
            const auto & columns = bias_.evaluate(real_field(w.grid), w.bias);
            Field tracer;
            local([&] {
                tracer.assign(matter_.cells(), 0.);
                for (size_t q = 0; q < matter_.cells(); ++q)
                    for (size_t a = 0; a < 4; ++a)
                        tracer[q] += p[spectrum_size_ + a] * columns[a][q];
                w.prediction.resize(modes_.size());
                w.parameters = p;
            });
            to_grid(tracer, w.grid);
            w.grid.fftw_r2c();
            const auto & observed = observation_fourier(w);
            for (size_t j = 0; j < modes_.size(); ++j)
                w.prediction[j] = observed.get_fourier_from_index(modes_[j].fourier_index);
            bool finite = true;
            for (auto z : w.prediction)
                finite &= std::isfinite(z.real()) && std::isfinite(z.imag());
            if (maximum(finite ? 0 : 1))
                throw UnrepresentableState("[EulerianLPT] Unrepresentable tracer prediction");
            w.valid = true;
            return w.prediction;
        }
        // Fourier columns at the latest forward state, for exact fixed-IC bias likelihood caching.
        // Scratch FFTs leave the matter/operator reverse tape intact.
        std::array<Modes, 4> bias_modes(Workspace & w) const {
            check(w);
            require(w.valid, "[EulerianLPT] Bias columns require a successful forward");
            const auto & columns = bias_.columns(w.bias);
            std::array<Modes, 4> result;
            local([&] {
                for (auto & c : result)
                    c.resize(modes_.size());
            });
            for (size_t a = 0; a < 4; ++a) {
                to_grid(columns[a], w.grid);
                w.grid.fftw_r2c();
                const auto & observed = observation_fourier(w);
                for (size_t j = 0; j < modes_.size(); ++j)
                    result[a][j] = observed.get_fourier_from_index(modes_[j].fourier_index);
            }
            return result;
        }
        // g uses the weighted half-complex dot product sum w_k Re(conj(g_k) dm_k).
        // Scalars returned here are local contributions; the posterior reduces them once.
        void pullback(const std::vector<double> & p,
                      const Modes & g,
                      Workspace & w,
                      Field & gs,
                      std::vector<double> & gp) const {
            check(w);
            require(w.valid && w.parameters == p, "[EulerianLPT] Pullback needs matching forward");
            finite_modes(g, modes_.size());
            const double n3 = double(matter_.layout().global_cells);
            auto & observation = compact_ ? w.compact_grid : w.grid;
            observation.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t j = 0; j < g.size(); ++j)
                observation.set_fourier_from_index(modes_[j].fourier_index, g[j] / n3);
            observation.set_grid_status_real(false);
            if (compact_) w.lift_plan->apply(observation, w.grid);
            w.grid.fftw_c2r();
            const Field tracer_bar = real_field(w.grid);
            const auto & columns = bias_.columns(w.bias);
            typename Bias::Columns cb;
            local([&] {
                gp.assign(p.size(), 0.);
                for (size_t a = 0; a < 4; ++a) {
                    cb[a].resize(matter_.cells());
                    for (size_t q = 0; q < matter_.cells(); ++q) {
                        cb[a][q] = p[spectrum_size_ + a] * tracer_bar[q];
                        gp[spectrum_size_ + a] += tracer_bar[q] * columns[a][q];
                    }
                }
            });
            const auto matter_bar = bias_.pullback(cb, w.bias);
            to_grid(matter_bar, w.grid);
            w.grid.fftw_r2c();
            Modes gm;
            local([&] {
                gm.resize(matter_.modes().size());
                for (size_t j = 0; j < gm.size(); ++j)
                    gm[j] = n3 * Complex(w.grid.get_fourier_from_index(matter_.modes()[j].fourier_index));
            });
            std::vector<double> pm;
            if (compact_) {
                Field padded_bar;
                matter_.pullback(matter_parameters(p), gm, w.matter, padded_bar, pm);
                gs = restrict_white_bar(padded_bar, w);
            } else {
                matter_.pullback(matter_parameters(p), gm, w.matter, gs, pm);
            }
            std::copy(pm.begin(), pm.begin() + spectrum_size_, gp.begin());
        }
        // Diagonal precision of the zero-field Gaussian linearization. Quadratic bias
        // has no linear contribution. The operator acts on the sampled mesh and uses
        // the posterior's noise; parameters/noise are frozen by the HMC driver.
        Field apply_gaussian_field_metric(const Field & input,
                                          const std::vector<double> & p,
                                          Workspace & w,
                                          double exponent,
                                          const GaussianFieldLikelihood & likelihood,
                                          double log_noise_power) const {
            check(w);
            validate_state(input, p);
            require(in_support(p), "[EulerianLPT] Metric scalars outside support");
            const auto & weights = likelihood.weights();
            require(weights.size() == modes_.size() && std::isfinite(likelihood.volume()) && likelihood.volume() > 0,
                    "[EulerianLPT] Invalid metric likelihood layout/volume");
            bool compatible = true;
            for (size_t i = 0; i < weights.size(); ++i)
                compatible &= weights[i] == 0 || weights[i] == modes_[i].weight;
            require(compatible, "[EulerianLPT] Unsupported Gaussian mode weighting");
            bool finite = std::isfinite(exponent) && std::isfinite(log_noise_power);
            for (double x : input)
                finite &= std::isfinite(x);
            require(finite, "[EulerianLPT] Nonfinite metric input or controls");
            controls({exponent, log_noise_power, likelihood.volume()});
            const double effective_noise = log_noise_power + (std::log(volume()) - std::log(likelihood.volume()));
            if (exponent == 0)
                return input;
            const bool rebuild = !w.metric_valid || w.metric_parameters != p || w.metric_log_noise != effective_noise ||
                                 w.metric_weights != weights;
            // Masks/layouts are rank-local. A change on one owner (including an
            // empty-mode rank) must rebuild collectively before any FFT calls.
            auto & grid = compact_ ? w.compact_grid : w.grid;
            if (maximum(rebuild ? 1. : 0.) != 0) {
                const bool check_mask = !w.metric_valid || w.metric_weights != weights;
                w.metric_valid = false;
                if (maximum(check_mask ? 1. : 0.) != 0) {
                    // A real-field FFT metric needs Hermitian masks. Otherwise
                    // separately applying its square root and inverse is inconsistent.
                    grid.fill_fourier_grid(FML::GRID::ComplexType(0));
                    for (size_t i = 0; i < modes_.size(); ++i)
                        grid.set_fourier_from_index(modes_[i].fourier_index, weights[i] == 0 ? 0. : 1.);
                    grid.set_grid_status_real(false);
                    grid.fftw_c2r();
                    grid.fftw_r2c();
                    bool hermitian = true;
                    for (size_t i = 0; i < modes_.size(); ++i)
                        hermitian &= std::abs(Complex(grid.get_fourier_from_index(modes_[i].fourier_index)) -
                                              Complex(weights[i] == 0 ? 0. : 1.)) < 1e-10;
                    require(hermitian, "[EulerianLPT] Gaussian metric mask must preserve Hermitian symmetry");
                }
                local([&] {
                    w.metric_log_precision.clear();
                    const std::vector<double> theta(p.begin(), p.begin() + spectrum_size_);
                    for (size_t i = 0; i < modes_.size(); ++i) {
                        const auto & mode = modes_[i];
                        if (weights[i] == 0 || !(mode.k < config_.initial_cutoff) ||
                            !config_.matter_band.contains({mode.n[0], mode.n[1], mode.n[2]},
                                                          config_.boxsize,
                                                          product_mesh()))
                            continue;
                        const auto power = spectrum_.evaluate(mode.k, theta);
                        if (!(power.value > 0) || !std::isfinite(power.value) || power.dlog.size() != spectrum_size_)
                            throw std::runtime_error("[EulerianLPT] Invalid metric spectrum");
                        for (double x : power.dlog)
                            if (!std::isfinite(x))
                                throw std::runtime_error("[EulerianLPT] Nonfinite metric spectrum derivative");
                        const double response = p[spectrum_size_] - p[spectrum_size_ + 3] * mode.k * mode.k;
                        if (!std::isfinite(response))
                            throw std::runtime_error("[EulerianLPT] Unrepresentable metric response");
                        if (response == 0)
                            continue;
                        const double log_ratio =
                            std::log(power.value) + 2 * std::log(std::abs(response)) - effective_noise;
                        const double log_precision =
                            std::max(0., log_ratio) + std::log1p(std::exp(-std::abs(log_ratio)));
                        if (!std::isfinite(log_precision))
                            throw std::runtime_error("[EulerianLPT] Unrepresentable metric precision");
                        w.metric_log_precision.emplace_back(mode.fourier_index, log_precision);
                    }
                    w.metric_parameters = p;
                    w.metric_log_noise = effective_noise;
                    w.metric_weights = weights;
                });
                w.metric_valid = true;
            }
            to_grid(input, grid);
            grid.fftw_r2c();
            local([&] {
                for (const auto & mode : w.metric_log_precision) {
                    const double multiplier = std::exp(exponent * mode.second);
                    if (!(multiplier > 0) || !std::isfinite(multiplier))
                        throw std::runtime_error("[EulerianLPT] Unrepresentable metric power");
                    grid.set_fourier_from_index(
                        mode.first,
                        FML::GRID::ComplexType(Complex(grid.get_fourier_from_index(mode.first)) * multiplier));
                }
                for (auto i : grid.get_fourier_range()) {
                    const auto value = grid.get_fourier_from_index(i);
                    if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                        throw std::runtime_error("[EulerianLPT] Nonfinite metric transform");
                }
            });
            grid.fftw_c2r();
            auto result = real_field(grid, cells());
            bool valid = true;
            for (double x : result)
                valid &= std::isfinite(x);
            require(valid, "[EulerianLPT] Nonfinite metric output");
            return result;
        }
        Modes
        mock(const Field & s, const std::vector<double> & p, double noise_power, uint64_t seed, Workspace & w) const {
            require(std::isfinite(noise_power) && noise_power > 0, "[EulerianLPT] Invalid mock noise power");
            controls({noise_power});
            Modes data;
            const auto & prediction = predict(s, p, w);
            Field noise;
            local([&] {
                data = prediction;
                noise = white_noise(seed);
            });
            auto & observation = compact_ ? w.compact_grid : w.grid;
            to_grid(noise, observation);
            observation.fftw_r2c();
            const double amplitude = std::sqrt(noise_power * double(layout().global_cells) / volume());
            for (size_t j = 0; j < data.size(); ++j)
                data[j] += amplitude * Complex(observation.get_fourier_from_index(modes_[j].fourier_index));
            finite_modes(data, modes_.size());
            return data;
        }
        FML::MCMC::HMCExecution hmc_execution() const {
            return compact_ ? Execution<DistributedLinearModelT<Grid>>::hmc(*compact_) : matter_.hmc_execution();
        }
    };
    template <class Grid>
    struct PosteriorFieldMetric<DistributedEulerianLPTModelT<Grid>, GaussianFieldLikelihood> {
        template <class Work>
        static Field apply(const DistributedEulerianLPTModelT<Grid> & model,
                           const GaussianFieldLikelihood & likelihood,
                           const GaussianNoise & noise,
                           const Field & input,
                           const std::vector<double> & parameters,
                           Work & work,
                           double exponent) {
            return model.apply_gaussian_field_metric(input,
                                                     parameters,
                                                     work,
                                                     exponent,
                                                     likelihood,
                                                     noise.log_power(parameters));
        }
    };
    template <class Grid>
    struct Execution<DistributedEulerianLPTModelT<Grid>> {
        using Forward = DistributedEulerianLPTModelT<Grid>;
        static void validate(const Forward & m, const Field & s, const std::vector<double> & p) {
            m.validate_state(s, p);
        }
        static double sum(const Forward & m, double x) { return m.sum(x); }
        static double maximum(const Forward & m, double x) { return m.maximum(x); }
        static void map_controls(const Forward & m, bool vary, int n, double tol) {
            m.controls({double(vary), double(n), tol});
        }
        static void hmc_controls(const Forward & m, bool vary, const FML::MCMC::HMCOptions & o) {
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
        static Likelihood likelihood(const Forward & m, Modes data) {
            if constexpr (std::is_same_v<Likelihood, GaussianFieldLikelihood>)
                return {m.modes(), HUGE_VAL, m.volume(), std::move(data), [&](const Modes & d) {
                            m.validate_modes(d);
                        }};
            else
                return {m.modes(), HUGE_VAL, m.volume(), std::move(data)};
        }
        static FML::MCMC::HMCExecution hmc(const Forward & m) { return m.hmc_execution(); }
    };
    using DistributedEulerianLPTModel = DistributedEulerianLPTModelT<>;
    // No noise coordinate is present in this model's scalar vector. Use an explicit fixed-noise posterior.
    template <class Grid>
    BasicPosterior<DistributedEulerianLPTModelT<Grid>>
    fixed_noise_posterior(const DistributedEulerianLPTModelT<Grid> & model,
                          Modes data,
                          double noise_power,
                          InitialFieldPrior field_prior = InitialFieldPrior::WhitenedGaussian) {
        return {model, std::move(data), GaussianNoise::fixed_power(noise_power), field_prior};
    }
    template <class Grid>
    BasicPosterior<DistributedEulerianLPTModelT<Grid>>
    fixed_noise_posterior(DistributedEulerianLPTModelT<Grid> &&,
                          Modes,
                          double,
                          InitialFieldPrior = InitialFieldPrior::WhitenedGaussian) = delete;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
