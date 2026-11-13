#ifndef FML_FIELD_LEVEL_DISTRIBUTED_LINEAR_H
#define FML_FIELD_LEVEL_DISTRIBUTED_LINEAR_H
#include <FML/FieldLevel/Framework.h>
#include <type_traits>

namespace FML::FIELDLEVEL::FRAMEWORK {
    // Collective, periodic linear-Gaussian model. Only local slabs/modes are stored.
    // All ranks of FML::communicator must call methods in the same order, with the
    // same configuration, scalar parameters and inference controls. Spectrum callbacks
    // must be deterministic and identical across ranks. No particle/LPT MPI claim.
    class DistributedLinearModel {
        friend class DistributedLPTModel;
        Configuration config_;
        Spectrum spectrum_;
        std::vector<Parameter> parameters_;
        std::vector<FieldLevelBAO::Mode> modes_;
        std::vector<double> response_;
        Model::Layout layout_{};
        double volume_{};
        static double reduce(double x, bool maximum = false) {
#ifdef USE_MPI
            double out;
            MPI_Allreduce(&x, &out, 1, MPI_DOUBLE, maximum ? MPI_MAX : MPI_SUM, FML::communicator);
            return out;
#else
            (void)maximum;
            return x;
#endif
        }
        static void require(bool ok, const char * message) {
            if (reduce(ok ? 0. : 1., true) != 0)
                throw std::invalid_argument(message);
        }
        static void replicated(const std::vector<double> & values) {
#ifdef USE_MPI
            // Size first: a mismatched parameter count must not give mismatched collectives.
            double n = double(values.size());
            require(reduce(n, true) == -reduce(-n, true), "[DistributedField] Replicated dimensions differ");
            bool finite = true;
            for (double x : values)
                finite &= std::isfinite(x);
            require(finite, "[DistributedField] Nonfinite replicated input");
            std::vector<double> lo(values.size()), hi(values.size());
            require(values.size() <= size_t(std::numeric_limits<int>::max()),
                    "[DistributedField] Too many replicated values");
            MPI_Allreduce(values.data(), lo.data(), int(values.size()), MPI_DOUBLE, MPI_MIN, FML::communicator);
            MPI_Allreduce(values.data(), hi.data(), int(values.size()), MPI_DOUBLE, MPI_MAX, FML::communicator);
            require(lo == hi, "[DistributedField] Replicated values differ");
#else
            (void)values;
#endif
        }
        double factor(size_t i, const std::vector<double> & p) const {
            const auto & m = modes_[i];
            const double n2 = double(m.n[0]) * m.n[0] + double(m.n[1]) * m.n[1] + double(m.n[2]) * m.n[2];
            return response_[i] * (p[spectrum_.parameters.size()] + config_.growth_rate * m.n[2] * m.n[2] / n2);
        }
        std::vector<Power> powers(const std::vector<double> & p) const {
            std::vector<double> theta(p.begin(), p.begin() + spectrum_.parameters.size());
            std::vector<Power> result(modes_.size());
            bool ok = true, callback_ok = true, shape_ok = true;
            try {
                for (size_t i = 0; i < result.size(); ++i)
                    if (modes_[i].k < config_.initial_cutoff) {
                        auto & v = result[i];
                        v = spectrum_.evaluate(modes_[i].k, theta);
                        shape_ok &= v.dlog.size() == theta.size();
                        ok &= v.value > 0 && std::isfinite(v.value);
                        for (double x : v.dlog)
                            ok &= std::isfinite(x);
                    }
            } catch (...) {
                callback_ok = false;
            }
            require(callback_ok, "[DistributedField] Spectrum callback failed on a rank");
            require(shape_ok, "[DistributedField] Spectrum derivative dimension mismatch");
            if (reduce(ok ? 0. : 1., true) != 0)
                throw UnrepresentableState("[DistributedField] Unrepresentable spectrum value/derivatives");
            return result;
        }

      public:
        struct Workspace {
            friend class DistributedLinearModel;
            friend class DistributedLPTModel;

          private:
            const DistributedLinearModel * owner;
            FML::GRID::FFTWGrid<3> grid;
            Modes prediction, white_modes, cotangent;
            std::vector<Power> power;
            std::vector<double> parameters;
            bool valid{false};
            explicit Workspace(const DistributedLinearModel * model) : owner(model), grid(model->config_.mesh) {}

          public:
            Workspace(Workspace &&) = default;
            Workspace & operator=(Workspace &&) = default;
            Workspace(const Workspace &) = delete;
            Workspace & operator=(const Workspace &) = delete;
            Modes & cotangent_buffer() { return cotangent; }
        };

      private:
        void check(const Workspace & w) const {
            require(w.owner == this && w.grid.get_nmesh() == config_.mesh,
                    "[DistributedField] Foreign/moved workspace");
        }
        void to_grid(const Field & field, Workspace & w) const {
            size_t q = 0;
            w.grid.set_grid_status_real(true);
            for (auto i : w.grid.get_real_range())
                w.grid.set_real_from_index(i, field[q++]);
        }
        Field from_modes(const Modes & modes, Workspace & w) const {
            w.grid.fill_fourier_grid(FML::GRID::ComplexType(0));
            for (size_t i = 0; i < modes.size(); ++i)
                w.grid.set_fourier_from_index(modes_[i].fourier_index, FML::GRID::ComplexType(modes[i]));
            w.grid.set_grid_status_real(false);
            w.grid.fftw_c2r();
            Field result(cells());
            size_t q = 0;
            for (auto i : w.grid.get_real_range())
                result[q++] = w.grid.get_real_from_index(i);
            return result;
        }

      public:
        DistributedLinearModel(Configuration config,
                               Spectrum spectrum,
                               FourierResponse response = {},
                               std::vector<Parameter> priors = {})
            : config_(config), spectrum_(std::move(spectrum)) {
            int tasks = FML::NTasks;
#ifdef USE_MPI
            int initialized = 0, finalized = 0;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            if (!initialized || finalized)
                throw std::logic_error("[DistributedField] Initialize FML/MPI first");
            MPI_Comm_size(FML::communicator, &tasks);
#endif
            replicated({double(config.mesh),
                        double(config.density_mesh),
                        config.boxsize,
                        config.initial_cutoff,
                        config.bias_cutoff,
                        config.likelihood_cutoff,
                        config.growth_rate,
                        double(config.evolution)});
            require(tasks == FML::NTasks && config.mesh >= 4 && config.mesh % 2 == 0 && config.mesh % tasks == 0 &&
                        config.evolution == Evolution::Linear && config.boxsize > 0 && std::isfinite(config.boxsize) &&
                        config.initial_cutoff > 0 && config.likelihood_cutoff > 0 && config.bias_cutoff > 0 &&
                        config.bias_cutoff <= config.initial_cutoff && std::isfinite(config.growth_rate) &&
                        config.initial_cutoff < M_PI * config.mesh / config.boxsize &&
                        config.likelihood_cutoff < M_PI * config.mesh / config.boxsize && bool(spectrum_.evaluate),
                    "[DistributedField] Invalid configuration; linear evolution and nonempty divisible slabs required");
            const size_t n = size_t(config.mesh);
            require(n <= Field().max_size() / n / n, "[DistributedField] Mesh storage overflow");
            volume_ = std::pow(config.boxsize, 3);
            require(volume_ > 0 && std::isfinite(volume_), "[DistributedField] Invalid volume");
            parameters_ = spectrum_.parameters;
            parameters_.push_back({"b1", 1.5, 1.5, 1});
            parameters_.push_back({"log_noise_power", std::log(1000.), std::log(1000.), 2, -30, 40});
            bool valid = true;
            std::set<std::string> overridden, names;
            for (const auto & prior : priors) {
                auto it = std::find_if(parameters_.begin(), parameters_.end(), [&](const auto & p) {
                    return p.name == prior.name;
                });
                if (!overridden.insert(prior.name).second || it == parameters_.end())
                    valid = false;
                else
                    *it = prior;
            }
            for (const auto & p : parameters_)
                valid &= !p.name.empty() && names.insert(p.name).second && std::isfinite(p.initial) &&
                         std::isfinite(p.mean) && std::isfinite(p.sigma) && p.sigma > 0 && p.lower < p.upper &&
                         p.initial >= p.lower && p.initial <= p.upper &&
                         (p.prior == Prior::Gaussian || (p.prior == Prior::Uniform && std::isfinite(p.lower) &&
                                                         std::isfinite(p.upper) && std::isfinite(p.upper - p.lower)));
            require(valid, "[DistributedField] Invalid/duplicate parameter specification or override");
            // Priors may have infinite bounds; compare them through finite sentinel flags.
            std::vector<double> specs;
            for (const auto & p : parameters_) {
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
            replicated(specs);
            const auto slab = FML::GRID::FFTWGrid<3>::local_slab(config.mesh);
            require(slab[0] > 0, "[DistributedField] Empty FFTW slabs unsupported");
            layout_ = {config.mesh, int(slab[1]), int(slab[0]), n * n * n, size_t(slab[0]) * n * n};
            const double kf = 2 * M_PI / config.boxsize;
            for (int ix = 0; ix < layout_.local_width; ++ix)
                for (int iy = 0; iy < config.mesh; ++iy)
                    for (int iz = 0; iz <= config.mesh / 2; ++iz) {
                        const int gx = ix + layout_.local_start;
                        const int x = gx <= config.mesh / 2 ? gx : gx - config.mesh;
                        const int y = iy <= config.mesh / 2 ? iy : iy - config.mesh;
                        const double k = kf * std::sqrt(double(x) * x + double(y) * y + double(iz) * iz);
                        if (k == 0 || k >= std::max(config.initial_cutoff, config.likelihood_cutoff))
                            continue;
                        modes_.push_back(
                            {{x, y, iz}, k, iz == 0 ? 1. : 2., (size_t(ix) * n + iy) * (n / 2 + 1) + iz, 0, 0, 0});
                    }
            response_ = std::move(response.multiplier);
            if (response_.empty())
                response_.assign(modes_.size(), 1.);
            valid = response_.size() == modes_.size();
            for (double x : response_)
                valid &= std::isfinite(x);
            require(valid, "[DistributedField] Invalid local Fourier response");
            Modes response_modes(response_.begin(), response_.end());
            validate_modes(response_modes); // Cross-rank Hermitian planes, without all-gathering modes.
        }
        const Configuration & configuration() const { return config_; }
        const auto & parameters() const { return parameters_; }
        const auto & modes() const { return modes_; }
        const auto & response() const { return response_; }
        auto layout() const { return layout_; }
        size_t cells() const { return layout_.owned_cells; }
        double volume() const { return volume_; }
        double sum(double x) const { return reduce(x); }
        double maximum(double x) const { return reduce(x, true); }
        Workspace workspace() const { return Workspace(this); }
        std::vector<double> initial_parameters() const {
            std::vector<double> p;
            for (const auto & v : parameters_)
                p.push_back(v.initial);
            return p;
        }
        void validate_state(const Field & s, const std::vector<double> & p) const {
            require(s.size() == cells() && p.size() == parameters_.size(),
                    "[DistributedField] State dimension mismatch");
            // Scalar synchronization is checked by in_support before a forward transform.
        }
        bool in_support(const std::vector<double> & p) const {
            require(p.size() == parameters_.size(), "[DistributedField] Parameter dimension mismatch");
            bool ok = true;
            for (size_t j = 0; j < p.size(); ++j)
                ok &= std::isfinite(p[j]) && p[j] >= parameters_[j].lower && p[j] <= parameters_[j].upper;
            if (reduce(ok ? 0. : 1., true) != 0)
                return false;
            replicated(p);
            return true;
        }
        void controls(const std::vector<double> & values) const { replicated(values); }
        Field white_noise(std::uint64_t seed, std::uint32_t stream = 0) const {
            Field s(cells());
            const size_t offset = size_t(layout_.local_start) * config_.mesh * config_.mesh;
            for (size_t q = 0; q < s.size(); ++q) {
                FML::RANDOM::COUNTER_BASED::Stream rng(seed, offset + q, 0x4649454cu, stream);
                s[q] = rng.normal();
            }
            return s;
        }
        void validate_modes(const Modes & data) const {
            bool ok = data.size() == modes_.size();
            for (auto z : data)
                ok &= std::isfinite(z.real()) && std::isfinite(z.imag());
            require(ok, "[DistributedField] Invalid local data/cotangent dimensions or values");
            auto w = workspace();
            auto real = from_modes(data, w);
            to_grid(real, w);
            w.grid.fftw_r2c();
            for (size_t i = 0; i < data.size(); ++i)
                ok &= std::abs(Complex(w.grid.get_fourier_from_index(modes_[i].fourier_index)) - data[i]) <=
                      1e-10 * std::max(1., std::abs(data[i]));
            require(ok, "[DistributedField] Data/response must preserve Hermitian symmetry across ranks");
        }
        const Modes & predict(const Field & s, const std::vector<double> & p, Workspace & w) const {
            validate_state(s, p);
            check(w);
            w.valid = false;
            require(in_support(p), "[DistributedField] Parameters outside support");
            w.power = powers(p);
            to_grid(s, w);
            w.grid.fftw_r2c();
            w.prediction.resize(modes_.size());
            w.white_modes.resize(modes_.size());
            for (size_t i = 0; i < modes_.size(); ++i) {
                w.white_modes[i] = Complex(w.grid.get_fourier_from_index(modes_[i].fourier_index));
                w.prediction[i] =
                    factor(i, p) * std::sqrt(w.power[i].value * layout_.global_cells / volume_) * w.white_modes[i];
            }
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
            require(w.valid && w.parameters == p && g.size() == modes_.size(),
                    "[DistributedField] Pullback needs matching forward evaluation");
            Modes coeff(g.size());
            gp.assign(p.size(), 0);
            const size_t offset = spectrum_.parameters.size();
            for (size_t i = 0; i < g.size(); ++i) {
                const double T = std::sqrt(w.power[i].value * layout_.global_cells / volume_);
                coeff[i] = factor(i, p) * T * g[i] / double(layout_.global_cells);
                const double product = modes_[i].weight * std::real(std::conj(g[i]) * w.white_modes[i]);
                gp[offset] += product * response_[i] * T;
                if (T != 0)
                    for (size_t j = 0; j < offset; ++j)
                        gp[j] += .5 * product * factor(i, p) * T * w.power[i].dlog[j];
            }
            gs = from_modes(coeff, w); // Parameter derivatives are LOCAL; posterior reduces once.
        }
        Field apply_field_metric(const Field & s, const std::vector<double> & p, Workspace & w, double exponent) const {
            validate_state(s, p);
            check(w);
            require(in_support(p), "[DistributedField] Metric parameters outside support");
            const auto power = powers(p);
            to_grid(s, w);
            w.grid.fftw_r2c();
            Modes correction(modes_.size());
            for (size_t i = 0; i < modes_.size(); ++i) {
                const double precision = modes_[i].k < config_.likelihood_cutoff ?
                                             1 + std::pow(factor(i, p), 2) * power[i].value / std::exp(p.back()) :
                                             1;
                correction[i] = Complex(w.grid.get_fourier_from_index(modes_[i].fourier_index)) *
                                (std::pow(precision, exponent) - 1);
            }
            auto result = from_modes(correction, w);
            for (size_t q = 0; q < s.size(); ++q)
                result[q] += s[q];
            return result;
        }
        Model::LinearConditional
        linear_conditional(const Modes & data, const std::vector<double> & p, Workspace & w) const {
            check(w);
            validate_modes(data);
            require(in_support(p), "[DistributedField] Invalid conditional parameters");
            const auto power = powers(p);
            Model::LinearConditional result;
            result.covariance.assign(data.size(), 1);
            Modes mean(data.size(), 0);
            for (size_t i = 0; i < data.size(); ++i)
                if (modes_[i].k < config_.likelihood_cutoff) {
                    const double B = factor(i, p), P = power[i].value, noise = std::exp(p.back());
                    const double precision = 1 + B * B * P / noise, C = (B * B * P + noise) / volume_;
                    result.covariance[i] = 1 / precision;
                    mean[i] = B * std::sqrt(P * layout_.global_cells / volume_) * data[i] /
                              (double(layout_.global_cells) * noise / volume_ * precision);
                    result.log_marginal -= .5 * modes_[i].weight * (std::norm(data[i]) / C + std::log(C));
                }
            result.log_marginal = sum(result.log_marginal);
            result.mean = from_modes(mean, w);
            return result;
        }
        // Owning local fields; displacements in box units, velocities at Lagrangian q.
        // velocity_factor=a H f L for compatible H/length units. Invalidates the pullback cache.
        Model::Reconstruction
        reconstruct(const Field & s, const std::vector<double> & p, Workspace & w, double velocity_factor) const {
            require(std::isfinite(velocity_factor), "[DistributedField] Invalid velocity factor");
            controls({velocity_factor});
            Model::Reconstruction result;
            result.tracer = predict(s, p, w);
            w.valid = false;
            Modes dk(modes_.size());
            for (size_t i = 0; i < dk.size(); ++i)
                dk[i] = std::sqrt(w.power[i].value * layout_.global_cells / volume_) * w.white_modes[i];
            result.initial_density = from_modes(dk, w);
            result.matter_density = result.initial_density;
            bool finite = true;
            for (auto z : result.tracer)
                finite &= std::isfinite(z.real()) && std::isfinite(z.imag());
            for (double x : result.initial_density)
                finite &= std::isfinite(x);
            for (int c = 0; c < 3; ++c) {
                Modes psi(dk.size());
                for (size_t i = 0; i < dk.size(); ++i) {
                    const auto & m = modes_[i];
                    const double n2 = double(m.n[0]) * m.n[0] + double(m.n[1]) * m.n[1] + double(m.n[2]) * m.n[2];
                    psi[i] = dk[i] * Complex(0, m.n[c] / (2 * M_PI * n2));
                }
                result.displacement[c] = from_modes(psi, w);
                result.velocity[c].resize(cells());
                for (size_t q = 0; q < cells(); ++q) {
                    result.velocity[c][q] = velocity_factor * result.displacement[c][q];
                    finite &= std::isfinite(result.velocity[c][q]) && std::isfinite(result.initial_density[q]);
                }
            }
            if (maximum(finite ? 0 : 1) != 0)
                throw UnrepresentableState("[DistributedField] Unrepresentable reconstruction");
            return result;
        }
        Modes mock(const Field & s, const std::vector<double> & p, std::uint64_t seed, Workspace & w) const {
            auto result = predict(s, p, w);
            to_grid(white_noise(seed), w);
            w.grid.fftw_r2c();
            const double amplitude = std::sqrt(std::exp(p.back()) * layout_.global_cells / volume_);
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += amplitude * Complex(w.grid.get_fourier_from_index(modes_[i].fourier_index));
            return result;
        }
    };
    template <>
    struct Execution<DistributedLinearModel> {
        static void validate(const DistributedLinearModel & m, const Field & s, const std::vector<double> & p) {
            m.validate_state(s, p);
        }
        static void map_controls(const DistributedLinearModel & m, bool vary, int iterations, double tolerance) {
            m.controls({double(vary), double(iterations), tolerance});
        }
        static void hmc_controls(const DistributedLinearModel & m, bool vary, const FML::MCMC::HMCOptions & o) {
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
        static double sum(const DistributedLinearModel & m, double x) { return m.sum(x); }
        static double maximum(const DistributedLinearModel & m, double x) { return m.maximum(x); }
        template <class Likelihood>
        static Likelihood likelihood(const DistributedLinearModel & m, Modes data) {
            if constexpr (std::is_same_v<Likelihood, GaussianFieldLikelihood>)
                return {m.modes(),
                        m.configuration().likelihood_cutoff,
                        m.volume(),
                        std::move(data),
                        [&](const Modes & d) { m.validate_modes(d); }};
            else
                return {m.modes(), m.configuration().likelihood_cutoff, m.volume(), std::move(data)};
        }
        static FML::MCMC::HMCExecution hmc(const DistributedLinearModel & m) {
            return {[&m](double x) { return m.sum(x); },
                    [&m](Field & x, std::uint64_t seed, std::uint64_t iteration) {
                        // Distinct purpose from IC/mock draws; global cell + iteration, independent of rank count.
                        const size_t offset =
                            size_t(m.layout().local_start) * m.configuration().mesh * m.configuration().mesh;
                        for (size_t q = 0; q < x.size(); ++q) {
                            FML::RANDOM::COUNTER_BASED::Stream rng(seed, offset + q, 0x484d434du);
                            // Incorporate all 64 iteration bits in the key/object mapping.
                            FML::RANDOM::COUNTER_BASED::Stream step(rng.next64(), iteration, 0x53544550u);
                            x[q] = step.normal();
                        }
                    },
                    [&m](bool valid) { return m.maximum(valid ? 0 : 1) == 0; }};
        }
    };
    using DistributedLinearPosterior = BasicPosterior<DistributedLinearModel>;
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
