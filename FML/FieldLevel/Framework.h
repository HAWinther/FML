#ifndef FML_FIELD_LEVEL_FRAMEWORK_H
#define FML_FIELD_LEVEL_FRAMEWORK_H
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <memory>
#include <set>
#include <map>
namespace FML::FIELDLEVEL::FRAMEWORK {
    using Modes = std::vector<Complex>;
    enum class Prior { Gaussian, Uniform };
    struct Parameter {
        std::string name;
        double initial{}, mean{}, sigma{1}, lower{-HUGE_VAL}, upper{HUGE_VAL};
        Prior prior{Prior::Gaussian};
        double scale() const { return prior == Prior::Uniform ? (upper - lower) / std::sqrt(12.) : sigma; }
        double penalty(double x) const { return prior == Prior::Uniform ? 0 : .5 * std::pow((x - mean) / sigma, 2); }
        double derivative(double x) const { return prior == Prior::Uniform ? 0 : (x - mean) / sigma / sigma; }
    };
    // Derivatives are of ln P, with k in h/Mpc and P in (Mpc/h)^3.
    struct Power {
        double value{};
        std::vector<double> dlog;
    };
    struct Spectrum {
        std::vector<Parameter> parameters;
        std::function<Power(double, const std::vector<double> &)> evaluate;
        static Spectrum fixed(std::function<double(double)> P) {
            return {{}, [P = std::move(P)](double k, const auto &) { return Power{P(k), {}}; }};
        }
        static Spectrum bao(BAOSpectrum spectrum) {
            return {{{"alpha_BAO", 1, 1, .1, .8, 1.2}}, [spectrum = std::move(spectrum)](double k, const auto & p) {
                        return Power{spectrum(k, p[0]), {spectrum.dlnP_dbeta(k, p[0])}};
                    }};
        }
    };
    enum class Evolution { Linear, Zeldovich, LPT2 };
    struct Configuration {
        int mesh{16}, density_mesh{32}, kernel_width{6};
        double boxsize{400}, initial_cutoff{.04}, bias_cutoff{.035}, likelihood_cutoff{.03}, growth_rate{0};
        Evolution evolution{Evolution::LPT2};
    };
    // Fixed periodic Fourier observation response. A separate component, not a noise/selection probability.
    struct FourierResponse {
        std::vector<double> multiplier; // Empty means identity; otherwise exactly one real value per retained mode.
    };
    // Compatibility kernel: shares the verified particle/adjoint numerics with the BAO application.
    // One instance belongs to one chain/workspace; it must not be shared by concurrent evaluations.
    class PeriodicBackend {
        friend class Model;
        Configuration config_;
        std::unique_ptr<FieldLevelBAO> engine_;
        FieldLevelBAO::Workspace cache_;
        Field delta_;
        Modes white_modes_, delta_modes_;
        std::vector<double> transfer_;
        std::vector<std::vector<double>> derivatives_;
        static Settings settings(const Configuration & c) {
            Settings s;
            s.nmesh = c.mesh;
            s.nmesh_density = c.density_mesh;
            s.boxsize = c.boxsize;
            s.kernel_width = c.kernel_width;
            s.model = c.evolution == Evolution::Linear ? FML::FIELDLEVEL::Model::Linear : FML::FIELDLEVEL::Model::LPT;
            s.lpt_order = c.evolution == Evolution::Zeldovich ? 1 : 2;
            s.kmax = std::max(c.initial_cutoff, c.likelihood_cutoff);
            s.bias_cutoff = c.bias_cutoff;
            s.growth_rate = c.growth_rate;
            s.bias = 1;
            s.b2 = 0;
            s.bG2 = 0;
            s.centering = 0;
            return s;
        }
        static BAOSpectrum identity(const Configuration & c) {
            BAOSpectrum p;
            p.amplitude = 0;
            p.smooth = [c](double) { return std::pow(c.boxsize, 3) / (double(c.mesh) * c.mesh * c.mesh); };
            return p;
        }
        void correlate(const Field & white, const Spectrum & spectrum, const std::vector<double> & params) {
            engine_->check_field(white);
            auto & grid = engine_->grid_;
            engine_->to_grid(white, grid);
            grid.fftw_r2c();
            const auto & modes = engine_->modes_;
            const auto n = modes.size();
            white_modes_.resize(n);
            delta_modes_.resize(n);
            transfer_.resize(n);
            derivatives_.assign(spectrum.parameters.size(), std::vector<double>(n, 0));
            std::vector<double> theta(params.begin(), params.begin() + spectrum.parameters.size());
            for (size_t i = 0; i < n; ++i) {
                white_modes_[i] = Complex(grid.get_fourier_from_index(modes[i].fourier_index));
                transfer_[i] = 0;
                if (modes[i].k < config_.initial_cutoff) {
                    const auto p = spectrum.evaluate(modes[i].k, theta);
                    if (!(p.value > 0) || !std::isfinite(p.value) || p.dlog.size() != theta.size())
                        throw std::domain_error("[FieldFramework] Invalid power or derivative dimensions");
                    transfer_[i] = std::sqrt(p.value * engine_->ncells_ / engine_->volume_);
                    for (size_t j = 0; j < theta.size(); ++j) {
                        if (!std::isfinite(p.dlog[j]))
                            throw std::domain_error("[FieldFramework] Nonfinite spectrum derivative");
                        derivatives_[j][i] = .5 * transfer_[i] * p.dlog[j];
                    }
                }
                delta_modes_[i] = transfer_[i] * white_modes_[i];
            }
            delta_ = engine_->from_modes(delta_modes_, [](const auto &) { return Complex(1); });
        }
        double mu2(size_t i) const {
            const auto & m = engine_->modes_[i];
            const double n2 = double(m.n[0] * m.n[0] + m.n[1] * m.n[1] + m.n[2] * m.n[2]);
            return m.n[2] * m.n[2] / n2;
        }

      public:
        explicit PeriodicBackend(Configuration c)
            : config_(c), engine_(std::make_unique<FieldLevelBAO>(settings(c), identity(c))) {}
    };
    class Workspace {
        friend class Model;
        const void * owner{};
        bool valid{false};
        std::vector<double> evaluation_parameters;
        std::unique_ptr<PeriodicBackend> backend;
        Modes prediction, observed, cotangent;
        Field gradient;
        std::vector<double> parameter_gradient;

      public:
        Modes & cotangent_buffer() { return cotangent; }
        Workspace() = default;
        Workspace(Workspace &&) = default;
        Workspace & operator=(Workspace &&) = default;
        Workspace(const Workspace &) = delete;
        Workspace & operator=(const Workspace &) = delete;
    };
    class Model {
        Configuration config_;
        Spectrum spectrum_;
        std::vector<Parameter> parameters_;
        std::vector<FieldLevelBAO::Mode> modes_;
        std::vector<double> response_;
        size_t cells_{};
        double volume_{};
        void dimensions(const Field & s, const std::vector<double> & p) const {
            if (s.size() != cells_ || p.size() != parameters_.size())
                throw std::invalid_argument("[FieldFramework] State dimension mismatch");
        }
        void check_workspace(const Workspace & w) const {
            if (w.owner != this || !w.backend || w.backend->config_.mesh != config_.mesh ||
                w.backend->config_.boxsize != config_.boxsize ||
                w.backend->config_.initial_cutoff != config_.initial_cutoff ||
                w.backend->config_.bias_cutoff != config_.bias_cutoff ||
                w.backend->config_.likelihood_cutoff != config_.likelihood_cutoff ||
                w.backend->config_.evolution != config_.evolution ||
                w.backend->config_.density_mesh != config_.density_mesh ||
                w.backend->config_.kernel_width != config_.kernel_width ||
                w.backend->config_.growth_rate != config_.growth_rate)
                throw std::invalid_argument("[FieldFramework] Incompatible chain workspace");
        }

      public:
        Model(Configuration config,
              Spectrum spectrum,
              FourierResponse response = {},
              std::vector<Parameter> priors = {})
            : config_(config), spectrum_(std::move(spectrum)) {
            if ((config.evolution != Evolution::Linear && config.evolution != Evolution::Zeldovich &&
                 config.evolution != Evolution::LPT2) ||
                config.mesh < 4 || config.mesh % 2 != 0 || config.density_mesh % 2 != 0 ||
                config.density_mesh < config.mesh || !spectrum_.evaluate || !std::isfinite(config.boxsize) ||
                config.boxsize <= 0 || !(config.initial_cutoff > 0) || !(config.bias_cutoff > 0) ||
                config.bias_cutoff > config.initial_cutoff || !(config.likelihood_cutoff > 0) ||
                !std::isfinite(config.initial_cutoff) || !std::isfinite(config.bias_cutoff) ||
                !std::isfinite(config.likelihood_cutoff) || !std::isfinite(config.growth_rate))
                throw std::invalid_argument("[FieldFramework] Invalid forward configuration");
            const double nyquist = 2 * M_PI * (config.mesh / 2) / config.boxsize;
            if (config.evolution != Evolution::Linear && config.initial_cutoff >= nyquist * .5)
                throw std::invalid_argument("[FieldFramework] Initial 1/2LPT scope requires initial_cutoff < half "
                                            "Nyquist; increase the mesh to resolve quadratic products");
            PeriodicBackend metadata(config);
            modes_ = metadata.engine_->modes();
            cells_ = metadata.engine_->ncells();
            volume_ = std::pow(config.boxsize, 3);
            parameters_ = spectrum_.parameters;
            parameters_.push_back({"b1", 1.5, 1.5, 1});
            if (config.evolution != Evolution::Linear) {
                parameters_.push_back({"b2", 0, 0, 1});
                parameters_.push_back({"b_s2", 0, 0, 1});
                parameters_.push_back({"b_laplacian", 0, 0, 100});
            }
            parameters_.push_back({"log_noise_power", std::log(1000.), std::log(1000.), 2, -30, 40});
            std::set<std::string> overridden;
            for (const auto & prior : priors) {
                if (!overridden.insert(prior.name).second)
                    throw std::invalid_argument("[FieldFramework] Duplicate prior override");
                auto it = std::find_if(parameters_.begin(), parameters_.end(), [&](const auto & p) {
                    return p.name == prior.name;
                });
                if (it == parameters_.end())
                    throw std::invalid_argument("[FieldFramework] Unknown prior override");
                *it = prior;
            }
            std::set<std::string> names;
            for (const auto & p : parameters_)
                if (p.name.empty() || !names.insert(p.name).second || !std::isfinite(p.initial) ||
                    !std::isfinite(p.mean) || !std::isfinite(p.sigma) || p.sigma <= 0 || !(p.lower < p.upper) ||
                    p.initial < p.lower || p.initial > p.upper ||
                    (p.prior != Prior::Gaussian && p.prior != Prior::Uniform) ||
                    (p.prior == Prior::Uniform &&
                     (!std::isfinite(p.lower) || !std::isfinite(p.upper) || !std::isfinite(p.upper - p.lower))))
                    throw std::invalid_argument("[FieldFramework] Invalid/duplicate parameter specification");
            response_ = std::move(response.multiplier);
            if (response_.empty())
                response_.assign(modes_.size(), 1);
            if (response_.size() != modes_.size())
                throw std::invalid_argument("[FieldFramework] Fourier response dimension mismatch");
            for (auto x : response_)
                if (!std::isfinite(x))
                    throw std::invalid_argument("[FieldFramework] Nonfinite Fourier response");
            std::map<std::array<int, 3>, size_t> lookup;
            for (size_t i = 0; i < modes_.size(); ++i)
                lookup[{modes_[i].n[0], modes_[i].n[1], modes_[i].n[2]}] = i;
            for (size_t i = 0; i < modes_.size(); ++i)
                if (modes_[i].n[2] == 0 && response_[i] != response_[lookup.at({-modes_[i].n[0], -modes_[i].n[1], 0})])
                    throw std::invalid_argument("[FieldFramework] Response must preserve Hermitian symmetry");
        }
        const Configuration & configuration() const { return config_; }
        const std::vector<Parameter> & parameters() const { return parameters_; }
        const auto & modes() const { return modes_; }
        struct Layout {
            int global_mesh, local_start, local_width;
            size_t global_cells, owned_cells;
        };
        Layout layout() const { return {config_.mesh, 0, config_.mesh, cells_, cells_}; }
        size_t cells() const { return cells_; }
        double volume() const { return volume_; }
        std::vector<double> initial_parameters() const {
            std::vector<double> p;
            for (const auto & v : parameters_)
                p.push_back(v.initial);
            return p;
        }
        Workspace workspace() const {
            Workspace w;
            w.owner = this;
            w.backend = std::make_unique<PeriodicBackend>(config_);
            return w;
        }
        Field white_noise(std::uint64_t seed) const {
            Field s(cells_);
            for (size_t i = 0; i < cells_; ++i) {
                FML::RANDOM::COUNTER_BASED::Stream rng(seed, i, 0x4649454cu);
                s[i] = rng.normal();
            }
            return s;
        }
        bool in_support(const std::vector<double> & p) const {
            if (p.size() != parameters_.size())
                throw std::invalid_argument("[FieldFramework] Parameter dimension mismatch");
            for (size_t j = 0; j < p.size(); ++j)
                if (!std::isfinite(p[j]) || p[j] < parameters_[j].lower || p[j] > parameters_[j].upper)
                    return false;
            return true;
        }
        const Modes & predict(const Field & white, const std::vector<double> & p, Workspace & w) const {
            dimensions(white, p);
            check_workspace(w);
            w.valid = false;
            if (!in_support(p))
                throw std::domain_error("[FieldFramework] Parameters outside support");
            for (double x : white)
                if (!std::isfinite(x))
                    throw std::domain_error("[FieldFramework] Nonfinite initial field");
            auto & b = *w.backend;
            b.correlate(white, spectrum_, p);
            const size_t offset = spectrum_.parameters.size();
            if (config_.evolution == Evolution::Linear) {
                w.prediction.resize(modes_.size());
                for (size_t i = 0; i < modes_.size(); ++i)
                    w.prediction[i] = (p[offset] + config_.growth_rate * b.mu2(i)) * b.delta_modes_[i];
            } else {
                b.engine_->s_.bias = p[offset];
                b.engine_->s_.b2 = p[offset + 1] + 2. / 3 * p[offset + 2];
                b.engine_->s_.bG2 = p[offset + 2];
                b.engine_->s_.bLap = p[offset + 3];
                w.prediction = b.engine_->forward(b.delta_, 1., b.cache_);
            }
            w.observed.resize(modes_.size());
            for (size_t i = 0; i < modes_.size(); ++i)
                w.observed[i] = response_[i] * w.prediction[i];
            w.evaluation_parameters = p;
            w.valid = true;
            return w.observed;
        }
        // gbar uses the full-Hermitian inner product sum weight*Re(conj(gbar)*delta_prediction).
        // Called immediately after predict in the same workspace/state; posterior evaluation enforces that order.
        void pullback(const std::vector<double> & p,
                      const Modes & gbar,
                      Workspace & w,
                      Field & gradient,
                      std::vector<double> & gp) const {
            check_workspace(w);
            if (!w.valid || w.evaluation_parameters != p)
                throw std::logic_error("[FieldFramework] Pullback requires the matching forward evaluation");
            if (p.size() != parameters_.size() || gbar.size() != modes_.size())
                throw std::invalid_argument("[FieldFramework] Pullback dimension mismatch");
            auto & b = *w.backend;
            auto & engine = *b.engine_;
            const auto offset = spectrum_.parameters.size();
            Modes g(gbar.size()), Q(gbar.size(), 0);
            gp.assign(p.size(), 0);
            for (size_t i = 0; i < g.size(); ++i)
                g[i] = response_[i] * gbar[i];
            if (config_.evolution == Evolution::Linear)
                for (size_t i = 0; i < g.size(); ++i) {
                    Q[i] = (p[offset] + config_.growth_rate * b.mu2(i)) * g[i] / double(cells_);
                    gp[offset] += modes_[i].weight * std::real(std::conj(g[i]) * b.delta_modes_[i]);
                }
            else {
                std::array<double, 4> bias{};
                engine.lpt_adjoint(g, b.cache_, Q, &bias);
                gp[offset] = bias[0];
                gp[offset + 1] = bias[1];
                gp[offset + 2] = bias[2] + 2. / 3 * bias[1];
                gp[offset + 3] = bias[3];
            }
            auto delta_bar = engine.from_modes(Q, [](const auto &) { return Complex(1); });
            engine.to_grid(delta_bar, engine.grid_);
            engine.grid_.fftw_r2c();
            Modes barred(modes_.size());
            for (size_t i = 0; i < barred.size(); ++i)
                barred[i] = Complex(engine.grid_.get_fourier_from_index(modes_[i].fourier_index));
            Modes coeff(barred.size());
            for (size_t i = 0; i < coeff.size(); ++i)
                coeff[i] = barred[i] * b.transfer_[i];
            gradient = engine.from_modes(coeff, [](const auto &) { return Complex(1); });
            for (size_t j = 0; j < offset; ++j) {
                for (size_t i = 0; i < coeff.size(); ++i)
                    coeff[i] = b.white_modes_[i] * b.derivatives_[j][i];
                const auto derivative = engine.from_modes(coeff, [](const auto &) { return Complex(1); });
                for (size_t q = 0; q < cells_; ++q)
                    gp[j] += delta_bar[q] * derivative[q];
            }
        }
        Field
        apply_field_metric(const Field & input, const std::vector<double> & p, Workspace & w, double exponent) const {
            dimensions(input, p);
            check_workspace(w);
            auto & b = *w.backend;
            auto & e = *b.engine_;
            e.to_grid(input, e.grid_);
            e.grid_.fftw_r2c();
            Modes coeff(modes_.size());
            const size_t offset = spectrum_.parameters.size();
            std::vector<double> theta(p.begin(), p.begin() + offset);
            const double noise = std::exp(p.back());
            for (size_t i = 0; i < coeff.size(); ++i) {
                double metric = 1;
                if (modes_[i].k < config_.initial_cutoff && modes_[i].k < config_.likelihood_cutoff) {
                    const double B = (p[offset] + config_.growth_rate * b.mu2(i)) * response_[i];
                    const double P = spectrum_.evaluate(modes_[i].k, theta).value;
                    metric += B * B * P / noise;
                }
                coeff[i] =
                    Complex(e.grid_.get_fourier_from_index(modes_[i].fourier_index)) * (std::pow(metric, exponent) - 1);
            }
            auto correction = e.from_modes(coeff, [](const auto &) { return Complex(1); });
            for (size_t i = 0; i < cells_; ++i)
                correction[i] += input[i];
            return correction;
        }
        struct LinearConditional {
            Field mean;
            std::vector<double> covariance;
            double log_marginal{};
        };
        LinearConditional linear_conditional(const Modes & data, const std::vector<double> & p, Workspace & w) const {
            if (config_.evolution != Evolution::Linear)
                throw std::invalid_argument("[FieldFramework] Exact conditional requires linear evolution");
            check_workspace(w);
            if (data.size() != modes_.size() || !in_support(p))
                throw std::invalid_argument("[FieldFramework] Invalid conditional input");
            auto & b = *w.backend;
            const size_t offset = spectrum_.parameters.size();
            std::vector<double> theta(p.begin(), p.begin() + offset);
            const double noise = std::exp(p.back());
            Modes mean(modes_.size(), 0);
            LinearConditional result;
            result.covariance.assign(modes_.size(), 1);
            for (size_t i = 0; i < modes_.size(); ++i)
                if (modes_[i].k < config_.likelihood_cutoff) {
                    const double P =
                        modes_[i].k < config_.initial_cutoff ? spectrum_.evaluate(modes_[i].k, theta).value : 0;
                    const double B = (p[offset] + config_.growth_rate * b.mu2(i)) * response_[i],
                                 T = std::sqrt(P * cells_ / volume_), sigma2 = noise / volume_;
                    const double precision = 1 + B * B * P / noise;
                    result.covariance[i] = 1 / precision;
                    mean[i] = B * T * data[i] / (double(cells_) * sigma2 * precision);
                    const double C = (B * B * P + noise) / volume_;
                    result.log_marginal -= .5 * modes_[i].weight * (std::norm(data[i]) / C + std::log(C));
                }
            result.mean = b.engine_->from_modes(mean, [](const auto &) { return Complex(1); });
            return result;
        }
        double linear_log_marginal(const Modes & data, const std::vector<double> & p) const {
            if (config_.evolution != Evolution::Linear || data.size() != modes_.size())
                throw std::invalid_argument("[FieldFramework] Invalid linear marginal request");
            if (!in_support(p))
                return -HUGE_VAL;
            const size_t offset = spectrum_.parameters.size();
            std::vector<double> theta(p.begin(), p.begin() + offset);
            double result = 0;
            for (size_t i = 0; i < modes_.size(); ++i)
                if (modes_[i].k < config_.likelihood_cutoff) {
                    const auto & m = modes_[i];
                    const double n2 = double(m.n[0] * m.n[0] + m.n[1] * m.n[1] + m.n[2] * m.n[2]);
                    const double B = (p[offset] + config_.growth_rate * m.n[2] * m.n[2] / n2) * response_[i];
                    const double P = m.k < config_.initial_cutoff ? spectrum_.evaluate(m.k, theta).value : 0;
                    const double C = (B * B * P + std::exp(p.back())) / volume_;
                    result -= .5 * m.weight * (std::norm(data[i]) / C + std::log(C));
                }
            return result;
        }
        struct Reconstruction {
            Field initial_density, matter_density;
            std::array<Field, 3> displacement, velocity; // At Lagrangian q; displacement in box lengths.
            Modes tracer;
        };
        // velocity_factor = a*H*f*boxsize gives peculiar velocity units for compatible H/length units.
        // EdS-factorized 2LPT: velocity proportional to Psi1+2 Psi2; output matter/tracer modes use the declared model
        // band.
        Reconstruction
        reconstruct(const Field & white, const std::vector<double> & p, Workspace & w, double velocity_factor) const {
            if (!std::isfinite(velocity_factor))
                throw std::invalid_argument("[FieldFramework] Invalid velocity factor");
            Reconstruction result;
            result.tracer = predict(white, p, w);
            auto & b = *w.backend;
            auto & e = *b.engine_;
            result.initial_density = b.delta_;
            std::array<Field, 3> first;
            for (int c = 0; c < 3; ++c)
                first[c] =
                    e.from_modes(b.delta_modes_, [&](const auto & m) { return Complex(0, e.kc(m.n, c) / e.k2(m.n)); });
            if (config_.evolution == Evolution::Linear) {
                result.displacement = first;
                result.matter_density = b.delta_;
            } else {
                w.valid = false;
                const auto saved = e.s_;
                try {
                    e.s_.growth_rate = 0;
                    e.s_.bias = 1;
                    e.s_.b2 = e.s_.bG2 = e.s_.bLap = 0;
                    const auto matter = e.forward(b.delta_, 1., b.cache_);
                    result.displacement = b.cache_.psi;
                    result.matter_density = e.from_modes(matter, [](const auto &) { return Complex(1); });
                    e.s_ = saved;
                } catch (...) {
                    e.s_ = saved;
                    throw;
                }
            }
            for (int c = 0; c < 3; ++c) {
                result.velocity[c].resize(cells_);
                for (size_t q = 0; q < cells_; ++q)
                    result.velocity[c][q] = velocity_factor * (config_.evolution == Evolution::LPT2 ?
                                                                   2 * result.displacement[c][q] - first[c][q] :
                                                                   first[c][q]);
            }
            return result;
        }
        Modes mock(const Field & white, const std::vector<double> & p, std::uint64_t seed, Workspace & w) const {
            auto result = predict(white, p, w);
            auto noise = white_noise(seed);
            const double amplitude = std::sqrt(std::exp(p.back()) * cells_ / volume_);
            auto & e = *w.backend->engine_;
            e.to_grid(noise, e.grid_);
            e.grid_.fftw_r2c();
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += amplitude * Complex(e.grid_.get_fourier_from_index(modes_[i].fourier_index));
            return result;
        }
    };
    // Explicit likelihood component: full-Hermitian Gaussian fields, not a positive count-intensity model.
    class GaussianFieldLikelihood {
        Modes data_;
        std::vector<double> weights_;
        double volume_;

      public:
        template <class ModeList>
        GaussianFieldLikelihood(const ModeList & modes, double cutoff, double volume, Modes data)
            : data_(std::move(data)), weights_(modes.size()), volume_(volume) {
            if (data_.size() != modes.size())
                throw std::invalid_argument("[FieldFramework] Data dimension mismatch");
            std::map<std::array<int, 3>, size_t> lookup;
            for (size_t i = 0; i < data_.size(); ++i) {
                const auto & m = modes[i];
                lookup[{m.n[0], m.n[1], m.n[2]}] = i;
                if (!std::isfinite(data_[i].real()) || !std::isfinite(data_[i].imag()))
                    throw std::invalid_argument("[FieldFramework] Nonfinite data");
                weights_[i] = m.k < cutoff ? m.weight : 0;
            }
            for (size_t i = 0; i < data_.size(); ++i) {
                const auto & m = modes[i];
                if (m.n[2] == 0) {
                    const auto z = std::conj(data_[lookup.at({-m.n[0], -m.n[1], 0})]);
                    if (std::abs(data_[i] - z) > 1e-10 * std::max(1., std::abs(z)))
                        throw std::invalid_argument("[FieldFramework] Data must preserve Hermitian symmetry");
                }
            }
        }
        double evaluate(const Modes & prediction, double log_noise, Modes & g, double & noise_gradient) const {
            if (prediction.size() != data_.size())
                throw std::invalid_argument("[FieldFramework] Likelihood prediction dimension mismatch");
            const double variance = std::exp(log_noise) / volume_;
            if (!(variance > 0) || !std::isfinite(variance))
                throw std::domain_error("[FieldFramework] Unrepresentable noise variance");
            double U = 0;
            noise_gradient = 0;
            g.assign(prediction.size(), 0);
            for (size_t i = 0; i < g.size(); ++i)
                if (weights_[i] != 0) {
                    const auto r = prediction[i] - data_[i];
                    const double chi = std::norm(r) / variance;
                    U += .5 * weights_[i] * (chi + std::log(variance));
                    noise_gradient += .5 * weights_[i] * (1 - chi);
                    g[i] = r / variance;
                }
            return U;
        }
    };
    // A forward model satisfying this interface can replace Model without changing the posterior or sampler.
    template <class Forward, class Likelihood = GaussianFieldLikelihood>
    class BasicPosterior {
        const Forward & model_;
        Likelihood likelihood_;

      public:
        BasicPosterior(const Forward & model, Modes data)
            : model_(model),
              likelihood_(model.modes(), model.configuration().likelihood_cutoff, model.volume(), std::move(data)) {}
        BasicPosterior(const Forward & model, Likelihood likelihood)
            : model_(model), likelihood_(std::move(likelihood)) {}
        BasicPosterior(Forward &&, Modes) = delete;
        BasicPosterior(Forward &&, Likelihood) = delete;
        const Forward & model() const { return model_; }
        template <class Work>
        double evaluate(const Field & white,
                        const std::vector<double> & p,
                        Work & w,
                        Field & gs,
                        std::vector<double> & gp) const {
            if (white.size() != model_.cells() || p.size() != model_.parameters().size())
                throw std::invalid_argument("[FieldFramework] Posterior dimension mismatch");
            if (!model_.in_support(p)) {
                gs.assign(white.size(), 0);
                gp.assign(p.size(), 0);
                return HUGE_VAL;
            }
            const auto & prediction = model_.predict(white, p, w);
            double noise_gradient = 0;
            auto & g = w.cotangent_buffer();
            double U = likelihood_.evaluate(prediction, p.back(), g, noise_gradient);
            model_.pullback(p, g, w, gs, gp);
            gp.back() += noise_gradient;
            for (size_t i = 0; i < white.size(); ++i) {
                U += .5 * white[i] * white[i];
                gs[i] += white[i];
            }
            for (size_t j = 0; j < p.size(); ++j) {
                const auto & prior = model_.parameters()[j];
                U += prior.penalty(p[j]);
                gp[j] += prior.derivative(p[j]);
            }
            return U;
        }
    };
    using Posterior = BasicPosterior<Model>;
    enum class MAPStatus { Converged, MaximumIterations, LineSearchFailed };
    struct MAPResult {
        Field white;
        std::vector<double> parameters;
        double potential{}, gradient_norm{};
        int iterations{};
        MAPStatus status{MAPStatus::MaximumIterations};
    };
    template <class Forward, class Likelihood>
    inline MAPResult map(const BasicPosterior<Forward, Likelihood> & posterior,
                         Field white,
                         std::vector<double> p,
                         bool vary_parameters = true,
                         int max_iterations = 200,
                         double tolerance = 1e-6) {
        if (max_iterations < 0 || tolerance <= 0 || !std::isfinite(tolerance))
            throw std::invalid_argument("[FieldFramework] Invalid MAP controls");
        const auto & model = posterior.model();
        auto w = model.workspace();
        MAPResult result;
        Field grad;
        std::vector<double> gp;
        double U = posterior.evaluate(white, p, w, grad, gp);
        if (!std::isfinite(U))
            throw std::invalid_argument("[FieldFramework] MAP initial state outside support");
        for (int iteration = 0;; ++iteration) {
            auto direction = model.apply_field_metric(grad, p, w, -1);
            std::vector<double> dp(p.size(), 0);
            double slope = 0, norm = 0;
            for (size_t q = 0; q < white.size(); ++q) {
                slope += grad[q] * direction[q];
                norm = std::max(norm, std::abs(grad[q]));
            }
            if (vary_parameters)
                for (size_t j = 0; j < p.size(); ++j) {
                    dp[j] = gp[j] * std::pow(model.parameters()[j].scale(), 2);
                    slope += gp[j] * dp[j];
                    norm = std::max(norm, std::abs(gp[j] * model.parameters()[j].scale()));
                }
            result = {white, p, U, norm, iteration, MAPStatus::MaximumIterations};
            if (norm < tolerance) {
                result.status = MAPStatus::Converged;
                return result;
            }
            if (iteration == max_iterations)
                return result;
            double step = 1;
            bool accepted = false;
            Field candidate, trialgrad;
            std::vector<double> params, trialgp;
            for (int line = 0; line < 40; ++line) {
                candidate = white;
                params = p;
                for (size_t q = 0; q < white.size(); ++q)
                    candidate[q] -= step * direction[q];
                for (size_t j = 0; j < p.size(); ++j)
                    params[j] -= step * dp[j];
                const double next = posterior.evaluate(candidate, params, w, trialgrad, trialgp);
                if (std::isfinite(next) && next <= U - 1e-4 * step * slope) {
                    white.swap(candidate);
                    p.swap(params);
                    grad.swap(trialgrad);
                    gp.swap(trialgp);
                    U = next;
                    accepted = true;
                    break;
                }
                step *= .5;
            }
            if (!accepted) {
                result.status = MAPStatus::LineSearchFailed;
                return result;
            }
        }
    }
    using FieldSink = std::function<void(const Field &, const std::vector<double> &, std::size_t)>;
    template <class Forward, class Likelihood>
    inline FML::MCMC::HMCResult sample(const BasicPosterior<Forward, Likelihood> & posterior,
                                       Field white,
                                       std::vector<double> p,
                                       FML::MCMC::HMCOptions options,
                                       bool vary_parameters = true,
                                       FieldSink sink = {}) {
        const auto & model = posterior.model();
        auto w = model.workspace();
        const auto frozen = p;
        const size_t n = model.cells(), np = p.size();
        if (white.size() != n || !model.in_support(p))
            throw std::invalid_argument("[FieldFramework] Invalid HMC initial state");
        FML::MCMC::HMCMass mass;
        mass.n_custom = n;
        mass.apply_sqrt = [&](auto & x) { x = model.apply_field_metric(x, frozen, w, .5); };
        mass.apply_inverse = [&](auto & x) { x = model.apply_field_metric(x, frozen, w, -1); };
        options.lower.clear();
        options.upper.clear();
        if (vary_parameters)
            for (const auto & v : model.parameters()) {
                mass.diagonal.push_back(1 / (v.scale() * v.scale()));
                options.lower.push_back(v.lower);
                options.upper.push_back(v.upper);
            }
        Field field(n), gs;
        std::vector<double> gp, params = p;
        auto potential = [&](const auto & state, auto & gradient) {
            std::copy_n(state.begin(), n, field.begin());
            if (vary_parameters)
                std::copy_n(state.begin() + n, np, params.begin());
            const double U = posterior.evaluate(field, params, w, gs, gp);
            std::copy(gs.begin(), gs.end(), gradient.begin());
            if (vary_parameters)
                std::copy(gp.begin(), gp.end(), gradient.begin() + n);
            return U;
        };
        size_t draw = 0;
        auto recorder = [&](const auto & state) {
            std::copy_n(state.begin(), n, field.begin());
            if (vary_parameters)
                std::copy_n(state.begin() + n, np, params.begin());
            if (sink)
                sink(field, params, draw);
            ++draw;
            return params;
        };
        if (vary_parameters)
            white.insert(white.end(), p.begin(), p.end());
        return FML::MCMC::run_hmc(potential, std::move(white), std::move(mass), options, recorder);
    }
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
