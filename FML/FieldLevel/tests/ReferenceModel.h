#ifndef FML_FIELD_LEVEL_TESTS_REFERENCE_MODEL_H
#define FML_FIELD_LEVEL_TESTS_REFERENCE_MODEL_H
// Test-only independent serial implementation of the periodic linear/1LPT/2LPT forward on the
// FieldLevelBAO engine: the library Model before 22 November 2026. The library Model now wraps the
// distributed models; tests compare those against this separately written reference.
#include <FML/FieldLevel/Framework.h>
#include <map>
#include <memory>
#include <set>

namespace FML::FIELDLEVEL::FRAMEWORK::TESTING {
    // Shares the particle/adjoint numerics of the BAO prototype. One instance per chain/workspace.
    // One instance belongs to one chain/workspace; it must not be shared by concurrent evaluations.
    class PeriodicBackend {
        friend class ReferenceModel;
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
    class ReferenceWorkspace {
        friend class ReferenceModel;
        const void * owner{};
        bool valid{false};
        std::vector<double> evaluation_parameters;
        std::unique_ptr<PeriodicBackend> backend;
        Modes prediction, observed, cotangent;
        Field gradient;
        std::vector<double> parameter_gradient;

      public:
        Modes & cotangent_buffer() { return cotangent; }
        ReferenceWorkspace() = default;
        ReferenceWorkspace(ReferenceWorkspace &&) = default;
        ReferenceWorkspace & operator=(ReferenceWorkspace &&) = default;
        ReferenceWorkspace(const ReferenceWorkspace &) = delete;
        ReferenceWorkspace & operator=(const ReferenceWorkspace &) = delete;
    };
    class ReferenceModel {
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
        void check_workspace(const ReferenceWorkspace & w) const {
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
        ReferenceModel(Configuration config,
              Spectrum spectrum,
              FourierResponse response = {},
              std::vector<Parameter> priors = {})
            : config_(config), spectrum_(std::move(spectrum)) {
            int unsupported_assignment = config.matter_assignment != MatterAssignment::Smooth;
            FML::MaxOverTasks(&unsupported_assignment);
            if (unsupported_assignment)
                throw std::invalid_argument("[FieldFramework] PSWF assignment requires DistributedLPTModelT");
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
        const auto & response() const { return response_; }
        using Layout = FieldLayout;
        Layout layout() const { return {config_.mesh, 0, config_.mesh, cells_, cells_}; }
        size_t cells() const { return cells_; }
        double volume() const { return volume_; }
        std::vector<double> initial_parameters() const {
            std::vector<double> p;
            for (const auto & v : parameters_)
                p.push_back(v.initial);
            return p;
        }
        ReferenceWorkspace workspace() const {
            ReferenceWorkspace w;
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
        const Modes & predict(const Field & white, const std::vector<double> & p, ReferenceWorkspace & w) const {
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
                      ReferenceWorkspace & w,
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
        apply_field_metric(const Field & input, const std::vector<double> & p, ReferenceWorkspace & w, double exponent) const {
            return apply_field_metric(input, p, w, exponent, p.back());
        }
        // Same metric with the noise log-power the posterior actually uses.
        Field apply_field_metric(const Field & input, const std::vector<double> & p, ReferenceWorkspace & w, double exponent,
                                 double log_noise_power) const {
            dimensions(input, p);
            if (!std::isfinite(log_noise_power))
                throw std::invalid_argument("[FieldFramework] Nonfinite metric noise");
            check_workspace(w);
            auto & b = *w.backend;
            auto & e = *b.engine_;
            e.to_grid(input, e.grid_);
            e.grid_.fftw_r2c();
            Modes coeff(modes_.size());
            const size_t offset = spectrum_.parameters.size();
            std::vector<double> theta(p.begin(), p.begin() + offset);
            const double noise = std::exp(log_noise_power);
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
        using LinearConditional = FRAMEWORK::LinearConditional;
        LinearConditional linear_conditional(const Modes & data, const std::vector<double> & p, ReferenceWorkspace & w) const {
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
        using Reconstruction = FRAMEWORK::Reconstruction;
        // velocity_factor = a*H*f*boxsize gives peculiar velocity units for compatible H/length units.
        // EdS-factorized 2LPT: velocity proportional to Psi1+2 Psi2; output matter/tracer modes use the declared model
        // band.
        Reconstruction
        reconstruct(const Field & white, const std::vector<double> & p, ReferenceWorkspace & w, double velocity_factor) const {
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
        Modes mock(const Field & white, const std::vector<double> & p, std::uint64_t seed, ReferenceWorkspace & w) const {
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
} // namespace FML::FIELDLEVEL::FRAMEWORK::TESTING
#endif
