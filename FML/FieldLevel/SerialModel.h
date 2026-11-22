#ifndef FML_FIELD_LEVEL_SERIAL_MODEL_H
#define FML_FIELD_LEVEL_SERIAL_MODEL_H
// Included at the end of Framework.h. The serial periodic forward (linear, Zel'dovich, 2LPT with
// Lagrangian bias and plane-parallel RSD) is a single-rank front end over the distributed models,
// so the library has one implementation of this physics. Before 22 November 2026 it ran on the
// FieldLevelBAO engine; that version is kept as an independent test reference
// (tests/ReferenceModel.h).
#include <FML/FieldLevel/DistributedLPT.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    class Model {
      public:
        using Grid = FML::GRID::FFTWGrid<3>;
        using Linear = DistributedLinearModelT<Grid>;
        using LPT = DistributedLPTModelT<Grid>;
        using Layout = FieldLayout;
        using Reconstruction = FRAMEWORK::Reconstruction;
        using LinearConditional = FRAMEWORK::LinearConditional;
        class Workspace {
            friend class Model;
            const Model * owner{};
            std::unique_ptr<typename Linear::Workspace> linear;
            std::unique_ptr<typename LPT::Workspace> lpt;
            Modes cotangent;

          public:
            Modes & cotangent_buffer() { return cotangent; }
            Workspace() = default;
            Workspace(Workspace &&) = default;
            Workspace & operator=(Workspace &&) = default;
            Workspace(const Workspace &) = delete;
            Workspace & operator=(const Workspace &) = delete;
        };

      private:
        Configuration config_;
        Spectrum spectrum_;
        std::shared_ptr<const Linear> linear_; // Immutable after construction; copies share it.
        std::shared_ptr<const LPT> lpt_;
        static Configuration checked(const Configuration & c) {
            int tasks = FML::NTasks;
#ifdef USE_MPI
            MPI_Comm_size(FML::communicator, &tasks);
#endif
            if (tasks != 1)
                throw std::invalid_argument("[FieldFramework] Model is serial; use the distributed models with MPI");
            if (c.matter_assignment != MatterAssignment::Smooth)
                throw std::invalid_argument("[FieldFramework] PSWF assignment requires DistributedLPTModelT");
            if ((c.evolution != Evolution::Linear && c.evolution != Evolution::Zeldovich &&
                 c.evolution != Evolution::LPT2) ||
                c.mesh < 4 || c.mesh % 2 != 0 || c.density_mesh % 2 != 0 || c.density_mesh < c.mesh ||
                !std::isfinite(c.boxsize) || c.boxsize <= 0 || !(c.initial_cutoff > 0) || !(c.bias_cutoff > 0) ||
                c.bias_cutoff > c.initial_cutoff || !(c.likelihood_cutoff > 0) || !std::isfinite(c.initial_cutoff) ||
                !std::isfinite(c.bias_cutoff) || !std::isfinite(c.likelihood_cutoff) || !std::isfinite(c.growth_rate))
                throw std::invalid_argument("[FieldFramework] Invalid forward configuration");
            const double nyquist = 2 * M_PI * (c.mesh / 2) / c.boxsize;
            if (c.evolution != Evolution::Linear && c.initial_cutoff >= nyquist * .5)
                throw std::invalid_argument("[FieldFramework] Initial 1/2LPT scope requires initial_cutoff < half "
                                            "Nyquist; increase the mesh to resolve quadratic products");
            return c;
        }
        void check(const Workspace & w) const {
            if (w.owner != this || (linear_ && !w.linear) || (lpt_ && !w.lpt))
                throw std::invalid_argument("[FieldFramework] Incompatible chain workspace");
        }

      public:
        Model(Configuration config, Spectrum spectrum, FourierResponse response = {},
              std::vector<Parameter> priors = {})
            : config_(checked(config)), spectrum_(spectrum) {
            if (!spectrum_.evaluate)
                throw std::invalid_argument("[FieldFramework] Invalid forward configuration");
            if (config_.evolution == Evolution::Linear)
                linear_ = std::make_shared<const Linear>(config_, std::move(spectrum), std::move(response), std::move(priors));
            else
                lpt_ = std::make_shared<const LPT>(config_, std::move(spectrum), std::move(response), std::move(priors));
        }
        const Configuration & configuration() const { return config_; }
        const std::vector<Parameter> & parameters() const { return linear_ ? linear_->parameters() : lpt_->parameters(); }
        const std::vector<FieldLevelBAO::Mode> & modes() const { return linear_ ? linear_->modes() : lpt_->modes(); }
        const std::vector<double> & response() const { return linear_ ? linear_->response() : lpt_->response(); }
        Layout layout() const { return linear_ ? linear_->layout() : lpt_->layout(); }
        size_t cells() const { return linear_ ? linear_->cells() : lpt_->cells(); }
        double volume() const { return linear_ ? linear_->volume() : lpt_->volume(); }
        std::vector<double> initial_parameters() const {
            return linear_ ? linear_->initial_parameters() : lpt_->initial_parameters();
        }
        Workspace workspace() const {
            Workspace w;
            w.owner = this;
            if (linear_)
                w.linear = std::make_unique<typename Linear::Workspace>(linear_->workspace());
            else
                w.lpt = std::make_unique<typename LPT::Workspace>(lpt_->workspace());
            return w;
        }
        Field white_noise(std::uint64_t seed) const { return linear_ ? linear_->white_noise(seed) : lpt_->white_noise(seed); }
        bool in_support(const std::vector<double> & p) const {
            if (p.size() != parameters().size())
                throw std::invalid_argument("[FieldFramework] Parameter dimension mismatch");
            return linear_ ? linear_->in_support(p) : lpt_->in_support(p);
        }
        const Modes & predict(const Field & white, const std::vector<double> & p, Workspace & w) const {
            check(w);
            if (white.size() != cells() || p.size() != parameters().size())
                throw std::invalid_argument("[FieldFramework] State dimension mismatch");
            if (!in_support(p))
                throw std::domain_error("[FieldFramework] Parameters outside support");
            for (double x : white)
                if (!std::isfinite(x))
                    throw std::domain_error("[FieldFramework] Nonfinite initial field");
            return linear_ ? linear_->predict(white, p, *w.linear) : lpt_->predict(white, p, *w.lpt);
        }
        // gbar uses the full-Hermitian inner product sum weight*Re(conj(gbar)*delta_prediction).
        void pullback(const std::vector<double> & p, const Modes & gbar, Workspace & w, Field & gradient,
                      std::vector<double> & gp) const {
            check(w);
            if (linear_)
                linear_->pullback(p, gbar, *w.linear, gradient, gp);
            else
                lpt_->pullback(p, gbar, *w.lpt, gradient, gp);
        }
        Field apply_field_metric(const Field & input, const std::vector<double> & p, Workspace & w,
                                 double exponent) const {
            check(w);
            return linear_ ? linear_->apply_field_metric(input, p, *w.linear, exponent)
                           : lpt_->apply_field_metric(input, p, *w.lpt, exponent);
        }
        // Same metric with the noise log-power the posterior actually uses.
        Field apply_field_metric(const Field & input, const std::vector<double> & p, Workspace & w, double exponent,
                                 double log_noise_power) const {
            check(w);
            if (!std::isfinite(log_noise_power))
                throw std::invalid_argument("[FieldFramework] Nonfinite metric noise");
            return linear_ ? linear_->apply_field_metric(input, p, *w.linear, exponent, log_noise_power)
                           : lpt_->apply_field_metric(input, p, *w.lpt, exponent, log_noise_power);
        }
        LinearConditional linear_conditional(const Modes & data, const std::vector<double> & p, Workspace & w) const {
            if (!linear_)
                throw std::invalid_argument("[FieldFramework] Exact conditional requires linear evolution");
            check(w);
            if (data.size() != modes().size() || !in_support(p))
                throw std::invalid_argument("[FieldFramework] Invalid conditional input");
            return linear_->linear_conditional(data, p, *w.linear);
        }
        double linear_log_marginal(const Modes & data, const std::vector<double> & p) const {
            if (!linear_ || data.size() != modes().size())
                throw std::invalid_argument("[FieldFramework] Invalid linear marginal request");
            if (!in_support(p))
                return -HUGE_VAL;
            const size_t offset = spectrum_.parameters.size();
            const std::vector<double> theta(p.begin(), p.begin() + offset);
            double result = 0;
            for (size_t i = 0; i < modes().size(); ++i)
                if (modes()[i].k < config_.likelihood_cutoff) {
                    const auto & m = modes()[i];
                    const double n2 = double(m.n[0] * m.n[0] + m.n[1] * m.n[1] + m.n[2] * m.n[2]);
                    const double B = (p[offset] + config_.growth_rate * m.n[2] * m.n[2] / n2) * response()[i];
                    const double P = m.k < config_.initial_cutoff ? spectrum_.evaluate(m.k, theta).value : 0;
                    const double C = (B * B * P + std::exp(p.back())) / volume();
                    result -= .5 * m.weight * (std::norm(data[i]) / C + std::log(C));
                }
            return result;
        }
        // velocity_factor = a*H*f*boxsize; EdS-factorized 2LPT velocities Psi1 + 2 Psi2.
        Reconstruction reconstruct(const Field & white, const std::vector<double> & p, Workspace & w,
                                   double velocity_factor) const {
            check(w);
            if (!std::isfinite(velocity_factor))
                throw std::invalid_argument("[FieldFramework] Invalid velocity factor");
            return linear_ ? linear_->reconstruct(white, p, *w.linear, velocity_factor)
                           : lpt_->reconstruct(white, p, *w.lpt, velocity_factor);
        }
        Modes mock(const Field & white, const std::vector<double> & p, std::uint64_t seed, Workspace & w) const {
            check(w);
            return linear_ ? linear_->mock(white, p, seed, *w.linear) : lpt_->mock(white, p, seed, *w.lpt);
        }
    };
    // Historical spelling of the serial workspace type.
    using Workspace = Model::Workspace;
    using Posterior = BasicPosterior<Model>;
    template <>
    struct NoiseAwareFieldMetric<Model> : std::true_type {};
    template <>
    struct PosteriorFieldMetric<Model, GaussianFieldLikelihood> : DETAIL::ExplicitNoiseFieldMetric<Model> {};
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
