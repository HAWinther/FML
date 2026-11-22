#ifndef FML_FIELD_LEVEL_FRAMEWORK_CORE_H
#define FML_FIELD_LEVEL_FRAMEWORK_CORE_H
// Framework core: parameters, spectra, likelihood, posterior, legacy MAP and HMC drivers.
// Include <FML/FieldLevel/Framework.h> for the complete framework including the serial Model.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <memory>
#include <set>
#include <map>
namespace FML::FIELDLEVEL::FRAMEWORK {
    using Modes = std::vector<Complex>;
    // Numerical proposal failure, distinct from a malformed model/callback contract.
    struct UnrepresentableState : std::domain_error {
        using std::domain_error::domain_error;
    };
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
    enum class MatterAssignment { Smooth, PSWF };
    struct Configuration {
        int mesh{16}, density_mesh{32}, kernel_width{6};
        double boxsize{400}, initial_cutoff{.04}, bias_cutoff{.035}, likelihood_cutoff{.03}, growth_rate{0};
        Evolution evolution{Evolution::LPT2};
        MatterAssignment matter_assignment{MatterAssignment::Smooth};
        double nufft_epsilon{1e-8};
    };
    // Fixed periodic Fourier observation response. A separate component, not a noise/selection probability.
    struct FourierResponse {
        std::vector<double> multiplier; // Empty means identity; otherwise exactly one real value per retained mode.
    };
    // Owned-field layout of a forward model (all forwards; serial models own every cell).
    struct FieldLayout {
        int global_mesh, local_start, local_width; // x aliases retained for slab callers
        size_t global_cells, owned_cells;
        std::array<int, 3> local_starts{}, local_counts{};
        FieldLayout(int mesh = 0, int start = 0, int width = 0, size_t total = 0, size_t owned = 0)
            : global_mesh(mesh), local_start(start), local_width(width), global_cells(total), owned_cells(owned),
              local_starts{start, 0, 0}, local_counts{width, mesh, mesh} {}
        // Compact owned fields follow real-range iteration: x, then y, then z (no FFT padding).
        std::array<int, 3> global_coordinate(size_t q) const {
            if (q >= owned_cells)
                throw std::out_of_range("[Field layout] Cell outside owned field");
            const size_t row = size_t(local_counts[1]) * local_counts[2];
            return {local_starts[0] + int(q / row),
                    local_starts[1] + int(q / local_counts[2] % local_counts[1]),
                    local_starts[2] + int(q % local_counts[2])};
        }
        size_t global_cell_id(size_t q) const {
            const auto c = global_coordinate(q);
            return (size_t(c[0]) * global_mesh + c[1]) * global_mesh + c[2];
        }
    };
    struct LinearConditional {
        Field mean;
        std::vector<double> covariance;
        double log_marginal{};
    };
    struct Reconstruction {
        Field initial_density, matter_density;
        std::array<Field, 3> displacement, velocity; // At Lagrangian q; displacement in box lengths.
        Modes tracer;
    };
    // Explicit likelihood component: full-Hermitian Gaussian fields, not a positive count-intensity model.
    class GaussianFieldLikelihood {
        Modes data_;
        std::vector<double> weights_;
        double volume_;

      public:
        template <class ModeList>
        GaussianFieldLikelihood(const ModeList & modes,
                                double cutoff,
                                double volume,
                                Modes data,
                                std::function<void(const Modes &)> distributed_check = {})
            : data_(std::move(data)), weights_(modes.size()), volume_(volume) {
            if (distributed_check)
                distributed_check(data_);
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
            for (size_t i = 0; !distributed_check && i < data_.size(); ++i) {
                const auto & m = modes[i];
                if (m.n[2] == 0) {
                    const auto z = std::conj(data_[lookup.at({-m.n[0], -m.n[1], 0})]);
                    if (std::abs(data_[i] - z) > 1e-10 * std::max(1., std::abs(z)))
                        throw std::invalid_argument("[FieldFramework] Data must preserve Hermitian symmetry");
                }
            }
        }
        const Modes & data() const { return data_; }
        const std::vector<double> & weights() const { return weights_; }
        double volume() const { return volume_; }
        // Local mathematical target change, trial minus reference. Evaluate the
        // small residual/noise changes directly, avoiding subtraction of full NLLs.
        long double difference(const Modes & trial, double noise,
                               const Modes & reference, double reference_noise) const {
            if(trial.size()!=data_.size() || reference.size()!=data_.size())
                throw std::invalid_argument("[FieldFramework] Likelihood difference dimension mismatch");
            for(double n:{noise,reference_noise}) {
                const double variance=std::exp(n)/volume_;
                if(!(variance>0) || !std::isfinite(variance))
                    throw UnrepresentableState("[FieldFramework] Unrepresentable difference variance");
            }
            const long double v=std::exp((long double)noise)/volume_,v0=std::exp((long double)reference_noise)/volume_;
            const long double dn=(long double)noise-reference_noise;
            long double result=0;
            for(size_t i=0;i<data_.size();++i) if(weights_[i]!=0) {
                const long double rr=(long double)reference[i].real()-data_[i].real(),ri=(long double)reference[i].imag()-data_[i].imag();
                const long double dr=(long double)trial[i].real()-reference[i].real(),di=(long double)trial[i].imag()-reference[i].imag();
                const long double chi0=(rr*rr+ri*ri)/v0;
                const long double dc=std::abs(dn)<=1 ? (2*(rr*dr+ri*di)+dr*dr+di*di)/v+chi0*std::expm1(-dn) :
                    ((rr+dr)*(rr+dr)+(ri+di)*(ri+di))/v-chi0;
                result+=.5L*weights_[i]*(dc+dn);
            }
            return result;
        }
        double evaluate(const Modes & prediction, double log_noise, Modes & g, double & noise_gradient) const {
            if (prediction.size() != data_.size())
                throw std::invalid_argument("[FieldFramework] Likelihood prediction dimension mismatch");
            const double variance = std::exp(log_noise) / volume_;
            if (!(variance > 0) || !std::isfinite(variance))
                throw UnrepresentableState("[FieldFramework] Unrepresentable noise variance");
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
    // Specialize for distributed ownership. Default preserves the serial/custom-forward interface.
    template <class Forward>
    struct Execution {
        static void validate(const Forward & m, const Field & s, const std::vector<double> & p) {
            if (s.size() != m.cells() || p.size() != m.parameters().size())
                throw std::invalid_argument("[FieldFramework] State dimension mismatch");
        }
        static void map_controls(const Forward &, bool, int, double) {}
        static void hmc_controls(const Forward &, bool, const FML::MCMC::HMCOptions &) {}
        static double sum(const Forward &, double x) { return x; }
        static double maximum(const Forward &, double x) { return x; }
        template <class Likelihood>
        static Likelihood likelihood(const Forward & m, Modes data) {
            return {m.modes(), m.configuration().likelihood_cutoff, m.volume(), std::move(data)};
        }
        static FML::MCMC::HMCExecution hmc(const Forward &) { return {}; }
    };
    namespace DETAIL {
        // Shared Execution members of the collective (distributed) forwards: replicated
        // controls checked through Forward::controls, reductions through the model, and
        // HMC momenta from a counter-based stream keyed by global cell and iteration
        // (independent of the rank count).
        template <class Forward>
        struct DistributedExecution {
            static void validate(const Forward & m, const Field & s, const std::vector<double> & p) {
                m.validate_state(s, p);
            }
            static double sum(const Forward & m, double x) { return m.sum(x); }
            static double maximum(const Forward & m, double x) { return m.maximum(x); }
            static void map_controls(const Forward & m, bool vary, int iterations, double tolerance) {
                m.controls({double(vary), double(iterations), tolerance});
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
            static FML::MCMC::HMCExecution counter_based_hmc(const Forward & m) {
                return {[&m](double x) { return m.sum(x); },
                        [&m](Field & x, std::uint64_t seed, std::uint64_t iteration) {
                            // Distinct purpose from IC/mock draws; global cell + iteration.
                            for (size_t q = 0; q < x.size(); ++q) {
                                FML::RANDOM::COUNTER_BASED::Stream rng(seed, m.layout().global_cell_id(q), 0x484d434du);
                                // Incorporate all 64 iteration bits in the key/object mapping.
                                FML::RANDOM::COUNTER_BASED::Stream step(rng.next64(), iteration, 0x53544550u);
                                x[q] = step.normal();
                            }
                        },
                        [&m](bool valid) { return m.maximum(valid ? 0 : 1) == 0; }};
            }
        };
    } // namespace DETAIL
    template<class Likelihood,class=void> struct HasLikelihoodDifference : std::false_type {};
    template<class Likelihood> struct HasLikelihoodDifference<Likelihood,std::void_t<decltype(
        std::declval<const Likelihood &>().difference(std::declval<const Modes &>(),double{},std::declval<const Modes &>(),double{}))>> : std::true_type {};
    // Explicit noise layout. Defaults preserve the historical last-scalar log-power convention.
    struct GaussianNoise {
        enum class Mode { LastParameter, Parameter, Fixed };
        Mode mode{Mode::LastParameter};
        size_t parameter_index{};
        double fixed_log_power{};
        static GaussianNoise parameter(size_t index) { return {Mode::Parameter, index, 0}; }
        static GaussianNoise fixed_power(double power) { return {Mode::Fixed, 0, std::log(power)}; }
        size_t index(size_t size) const { return mode == Mode::LastParameter ? size - 1 : parameter_index; }
        double log_power(const std::vector<double> & p) const {
            return mode == Mode::Fixed ? fixed_log_power : p[index(p.size())];
        }
        void add_gradient(std::vector<double> & gp, double gradient) const {
            if (mode != Mode::Fixed) gp[index(gp.size())] += gradient;
        }
    };
    // ConditionOnField gives a scalar target conditional on the supplied whitened realization.
    // It is not a prior for sampling the field itself.
    enum class InitialFieldPrior { WhitenedGaussian, ConditionOnField };
    // A forward model satisfying this interface can replace Model without changing the posterior or sampler.
    // Posterior-aware metric dispatch. Existing forwards retain their exact numerical path.
    // Likelihood-specific adapters can consume the actual noise policy without dummy scalars.
    template <class Forward, class Likelihood>
    struct PosteriorFieldMetric {
        template <class Work>
        static Field apply(const Forward & model, const Likelihood &, const GaussianNoise &,
                           const Field & input, const std::vector<double> & parameters,
                           Work & work, double exponent) {
            return model.apply_field_metric(input, parameters, work, exponent);
        }
    };
    // True for forwards whose generic linear-response metric takes the posterior's
    // Gaussian noise explicitly (checkpoint identities version this behaviour).
    template <class Forward>
    struct NoiseAwareFieldMetric : std::false_type {};
    namespace DETAIL {
        // The historical metric read the noise from the last scalar, which is wrong for
        // fixed or non-last noise and for a likelihood volume differing from the model's.
        // The default policy keeps the exact original call (and arithmetic).
        template <class Forward>
        struct ExplicitNoiseFieldMetric {
            template <class Likelihood, class Work>
            static Field apply(const Forward & model, const Likelihood & likelihood, const GaussianNoise & noise,
                               const Field & input, const std::vector<double> & parameters,
                               Work & work, double exponent) {
                if (noise.mode == GaussianNoise::Mode::LastParameter && likelihood.volume() == model.volume())
                    return model.apply_field_metric(input, parameters, work, exponent);
                const double log_noise = noise.log_power(parameters) + std::log(model.volume()) -
                                         std::log(likelihood.volume());
                return model.apply_field_metric(input, parameters, work, exponent, log_noise);
            }
        };
    } // namespace DETAIL
    template <class Forward, class Likelihood = GaussianFieldLikelihood>
    class BasicPosterior {
        const Forward & model_;
        Likelihood likelihood_;
        GaussianNoise noise_;
        InitialFieldPrior field_prior_;
        void validate_policy() const {
            bool valid = (noise_.mode == GaussianNoise::Mode::Fixed && std::isfinite(noise_.fixed_log_power) &&
                          std::isfinite(std::exp(noise_.fixed_log_power)) && std::exp(noise_.fixed_log_power) > 0) ||
                         ((noise_.mode == GaussianNoise::Mode::Parameter || noise_.mode == GaussianNoise::Mode::LastParameter) &&
                          !model_.parameters().empty() && noise_.index(model_.parameters().size()) < model_.parameters().size());
            valid &= field_prior_ == InitialFieldPrior::WhitenedGaussian || field_prior_ == InitialFieldPrior::ConditionOnField;
            if (Execution<Forward>::maximum(model_, valid ? 0 : 1) != 0)
                throw std::invalid_argument("[FieldFramework] Invalid noise layout or field-prior policy");
            for (double x : {double(noise_.mode), double(noise_.parameter_index), noise_.fixed_log_power, double(field_prior_)})
                if (Execution<Forward>::maximum(model_, x) != -Execution<Forward>::maximum(model_, -x))
                    throw std::invalid_argument("[FieldFramework] Posterior policy differs across ranks");
        }

      public:
        BasicPosterior(const Forward & model, Modes data, GaussianNoise noise = {},
                       InitialFieldPrior field_prior = InitialFieldPrior::WhitenedGaussian)
            : BasicPosterior(model, Execution<Forward>::template likelihood<Likelihood>(model, std::move(data)), noise, field_prior) {}
        BasicPosterior(const Forward & model, Likelihood likelihood, GaussianNoise noise = {},
                       InitialFieldPrior field_prior = InitialFieldPrior::WhitenedGaussian)
            : model_(model), likelihood_(std::move(likelihood)), noise_(noise), field_prior_(field_prior) { validate_policy(); }
        BasicPosterior(Forward &&, Modes, GaussianNoise = {}, InitialFieldPrior = InitialFieldPrior::WhitenedGaussian) = delete;
        BasicPosterior(Forward &&, Likelihood, GaussianNoise = {}, InitialFieldPrior = InitialFieldPrior::WhitenedGaussian) = delete;
        const Forward & model() const { return model_; }
        const Likelihood & likelihood() const { return likelihood_; }
        const GaussianNoise & noise_policy() const { return noise_; }
        InitialFieldPrior field_prior() const { return field_prior_; }
        template <class Work>
        Field apply_field_metric(const Field & input, const std::vector<double> & parameters,
                                 Work & work, double exponent) const {
            return PosteriorFieldMetric<Forward, Likelihood>::apply(model_, likelihood_, noise_, input,
                                                                    parameters, work, exponent);
        }
        // Same normalized target as evaluate(), without the forward-model pullback.
        // Custom likelihoods retain their existing evaluate interface/cotangent work.
        template <class Work>
        double value(const Field & white, const std::vector<double> & p, Work & w) const {
            Execution<Forward>::validate(model_, white, p);
            if (!model_.in_support(p)) return HUGE_VAL;
            double U = 0, unused_noise = 0;
            try {
                const auto & prediction = model_.predict(white, p, w);
                U = likelihood_.evaluate(prediction, noise_.log_power(p), w.cotangent_buffer(), unused_noise);
            } catch (const UnrepresentableState &) { return HUGE_VAL; }
            if (field_prior_ == InitialFieldPrior::WhitenedGaussian)
                for (double x : white) U += .5 * x * x;
            U = Execution<Forward>::sum(model_, U);
            for (size_t j = 0; j < p.size(); ++j) U += model_.parameters()[j].penalty(p[j]);
            return U;
        }
        static constexpr bool supports_stable_changes=HasLikelihoodDifference<Likelihood>::value;
        struct ValueReference {
          private:
            friend class BasicPosterior;
            const BasicPosterior * owner{};
            Modes prediction;
            std::vector<double> parameters;
        };
        struct ValueChange {double value{},change{};};
        // The supplied base white field must remain consistent with this reference
        // until value_change() returns. It is supplied separately to avoid copying
        // another potentially large field into every reference.
        template<class Work>
        ValueReference value_reference(const Field & white,const std::vector<double> & p,Work & work) const {
            if constexpr(!supports_stable_changes) {
                throw std::invalid_argument("[FieldFramework] Likelihood has no stable difference interface");
            } else {
                Execution<Forward>::validate(model_,white,p);
                if(Execution<Forward>::maximum(model_,model_.in_support(p)?0:1)!=0)
                    throw std::invalid_argument("[FieldFramework] Reference outside support");
                const auto & prediction=model_.predict(white,p,work);
                ValueReference result;bool failed=false;
                try {result.prediction=prediction;result.parameters=p;result.owner=this;}catch(...){failed=true;}
                if(Execution<Forward>::maximum(model_,failed?1:0)!=0)
                    throw std::runtime_error("[FieldFramework] Collective value-reference allocation failed");
                return result;
            }
        }
        template<class Work>
        ValueChange value_change(const Field & white,const std::vector<double> & p,Work & work,
                                 const Field & base_white,const ValueReference & reference) const {
            if constexpr(!supports_stable_changes) {
                throw std::invalid_argument("[FieldFramework] Likelihood has no stable difference interface");
            } else {
                if(Execution<Forward>::maximum(model_,reference.owner==this?0:1)!=0)
                    throw std::invalid_argument("[FieldFramework] Foreign/empty value reference");
                Execution<Forward>::validate(model_,base_white,reference.parameters);
                Execution<Forward>::validate(model_,white,p);
                if(!model_.in_support(p))return {HUGE_VAL,HUGE_VAL};
                double U=0,unused=0;long double delta=0;
                try {
                    const auto & prediction=model_.predict(white,p,work);
                    U=likelihood_.evaluate(prediction,noise_.log_power(p),work.cotangent_buffer(),unused);
                    delta=likelihood_.difference(prediction,noise_.log_power(p),reference.prediction,noise_.log_power(reference.parameters));
                }catch(const UnrepresentableState &){return {HUGE_VAL,HUGE_VAL};}
                if (field_prior_ == InitialFieldPrior::WhitenedGaussian) for(size_t i=0;i<white.size();++i) {
                    U+=.5*white[i]*white[i];
                    const long double dx=(long double)white[i]-base_white[i];
                    delta+=(long double)base_white[i]*dx+.5L*dx*dx;
                }
                U=Execution<Forward>::sum(model_,U);
                delta=Execution<Forward>::sum(model_,double(delta));
                for(size_t j=0;j<p.size();++j) {
                    const auto & prior=model_.parameters()[j];U+=prior.penalty(p[j]);
                    if(prior.prior==Prior::Gaussian) {
                        const long double dx=((long double)p[j]-reference.parameters[j])/prior.sigma;
                        const long double x0=((long double)reference.parameters[j]-prior.mean)/prior.sigma;
                        delta+=x0*dx+.5L*dx*dx;
                    }
                }
                return {U,double(delta)};
            }
        }
        template <class Work>
        double evaluate(const Field & white,
                        const std::vector<double> & p,
                        Work & w,
                        Field & gs,
                        std::vector<double> & gp) const {
            Execution<Forward>::validate(model_, white, p);
            if (!model_.in_support(p)) {
                gs.assign(white.size(), 0);
                gp.assign(p.size(), 0);
                return HUGE_VAL;
            }
            double noise_gradient = 0, U = 0;
            try {
                const auto & prediction = model_.predict(white, p, w);
                auto & g = w.cotangent_buffer();
                U = likelihood_.evaluate(prediction, noise_.log_power(p), g, noise_gradient);
                model_.pullback(p, g, w, gs, gp);
            } catch (const UnrepresentableState &) {
                // A collectively detected numerical domain failure rejects the proposal.
                // Programming/configuration errors still propagate to the caller.
                gs.assign(white.size(), 0);
                gp.assign(p.size(), 0);
                return HUGE_VAL;
            }
            noise_.add_gradient(gp, noise_gradient);
            if (field_prior_ == InitialFieldPrior::WhitenedGaussian) for (size_t i = 0; i < white.size(); ++i) {
                U += .5 * white[i] * white[i];
                gs[i] += white[i];
            }
            U = Execution<Forward>::sum(model_, U);
            for (auto & x : gp)
                x = Execution<Forward>::sum(model_, x);
            for (size_t j = 0; j < p.size(); ++j) {
                const auto & prior = model_.parameters()[j];
                U += prior.penalty(p[j]);
                gp[j] += prior.derivative(p[j]);
            }
            return U;
        }
    };
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
        Execution<Forward>::map_controls(posterior.model(), vary_parameters, max_iterations, tolerance);
        if (posterior.field_prior() != InitialFieldPrior::WhitenedGaussian)
            throw std::invalid_argument("[FieldFramework] Joint MAP requires a field prior; use a scalar optimizer for fixed ICs");
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
            auto direction = posterior.apply_field_metric(grad, p, w, -1);
            std::vector<double> dp(p.size(), 0);
            double slope = 0, norm = 0;
            for (size_t q = 0; q < white.size(); ++q) {
                slope += grad[q] * direction[q];
                norm = std::max(norm, std::abs(grad[q]));
            }
            slope = Execution<Forward>::sum(model, slope);
            norm = Execution<Forward>::maximum(model, norm);
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
    namespace DETAIL {
        template <class Forward, class Likelihood>
        inline FML::MCMC::HMCResult sample_run(const BasicPosterior<Forward, Likelihood> & posterior,
                                               Field white,
                                               std::vector<double> p,
                                               FML::MCMC::HMCOptions options,
                                               bool vary_parameters,
                                               FieldSink sink,
                                               const std::vector<double> & frozen,
                                               const FML::MCMC::HMCRunControl & run = {}) {
            const auto & model = posterior.model();
            Execution<Forward>::validate(model, white, p);
            Execution<Forward>::hmc_controls(model, vary_parameters, options);
            if (posterior.field_prior() != InitialFieldPrior::WhitenedGaussian)
                throw std::invalid_argument("[FieldFramework] Field HMC requires a field prior; use a scalar sampler for fixed ICs");
            auto w = model.workspace();
            const size_t n = model.cells(), np = p.size();
            if (white.size() != n || !model.in_support(p))
                throw std::invalid_argument("[FieldFramework] Invalid HMC initial state");
            FML::MCMC::HMCMass mass;
            mass.n_custom = n;
            mass.apply_sqrt = [&](auto & x) { x = posterior.apply_field_metric(x, frozen, w, .5); };
            mass.apply_inverse = [&](auto & x) { x = posterior.apply_field_metric(x, frozen, w, -1); };
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
            size_t draw = run.state && run.state->initialized ?
                              size_t(std::max(0, run.state->next_iteration - options.warmup)) :
                              0;
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
            return FML::MCMC::run_hmc(potential,
                                      std::move(white),
                                      std::move(mass),
                                      options,
                                      recorder,
                                      Execution<Forward>::hmc(model),
                                      run);
        }
    } // namespace DETAIL
    template <class Forward, class Likelihood>
    inline FML::MCMC::HMCResult sample(const BasicPosterior<Forward, Likelihood> & posterior,
                                       Field white,
                                       std::vector<double> parameters,
                                       FML::MCMC::HMCOptions options,
                                       bool vary_parameters = true,
                                       FieldSink sink = {}) {
        const auto frozen = parameters;
        return DETAIL::sample_run(posterior,
                                  std::move(white),
                                  std::move(parameters),
                                  options,
                                  vary_parameters,
                                  std::move(sink),
                                  frozen);
    }
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
