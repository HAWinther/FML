#ifndef FML_FIELD_LEVEL_MARGINALIZED_BIAS_H
#define FML_FIELD_LEVEL_MARGINALIZED_BIAS_H
// Exact analytic marginalization of the Eulerian bias coefficients.
//
// The Eulerian tracer is linear in the four bias coefficients, d_pred = sum_a b_a C_a(s, theta), with the
// columns C_a computed from the evolved matter. With fixed Gaussian noise and independent Gaussian priors
// b_a ~ N(mu_a, sigma_a^2) (unbounded), the integral over b is Gaussian:
//
//   -ln p(d | s, theta) = U(b_hat) + P(b_hat) + 1/2 ln det A - 1/2 ln det Lambda,
//   A = G + Lambda,  G_ab = <C_a, C_b>,  Lambda = diag(1/sigma_a^2),  b_hat = argmin (U + P),
//
// with <x, y> = sum_k w_k Re(conj(x_k) y_k) / v the likelihood inner product (v = P_noise / V). By the
// envelope theorem the gradient of the first two terms is the likelihood gradient at fixed b = b_hat, and
// 1/2 d ln det A = sum_ab (A^-1)_ab <C_b, dC_a>. Both act on the columns, so one column pullback gives the
// exact gradient: G_a = [b_hat_a (C b_hat - d) + sum_b (A^-1)_ab C_b] / v.
//
// The sampled target is (white field s, spectrum parameters theta); b given (s, theta, d) is Gaussian with
// mean b_hat and covariance A^-1 (conditional_bias), so exact bias draws can be added to every sample.
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/FixedIC.h>

namespace FML::FIELDLEVEL::FRAMEWORK {
    template <class Grid = FML::GRID::FFTWGrid<3>>
    class BiasMarginalizedPosterior {
      public:
        using Model = DistributedEulerianLPTModelT<Grid>;
        using Workspace = typename Model::Workspace;
        struct Conditional {
            std::array<double, 4> mean{};
            std::array<double, 16> covariance{}; // row-major A^-1
            std::array<double, 16> cholesky{};   // lower L with L L^T = A^-1
        };

      private:
        const Model & model_;
        Modes data_;
        double noise_power_, variance_;
        size_t spectrum_size_;
        std::array<double, 4> mean_{}, precision_{};
        std::vector<double> weights_;
        GaussianFieldLikelihood metric_likelihood_;
        mutable std::size_t evaluations_{};
        void require(bool ok, const char * message) const {
            if (model_.maximum(ok ? 0 : 1) != 0)
                throw std::invalid_argument(message);
        }
        struct Solve {
            std::array<double, 4> b_hat{};
            std::array<double, 16> inverse{}, lower{};
            double log_det{}, value{};
        };
        // A = G + Lambda (4x4, SPD): Cholesky, inverse and log determinant; b_hat from the cached quadratic.
        Solve solve(const FixedNoiseLinearBiasLikelihood<4> & like) const {
            const auto g0 = like.gradient(mean_); // dU/db at b = mu (the cache's reference point)
            std::array<double, 16> A = like.gram();
            for (int a = 0; a < 4; ++a)
                A[a * 4 + a] += precision_[a];
            Solve r;
            auto & L = r.lower;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j <= i; ++j) {
                    long double v = A[i * 4 + j];
                    for (int k = 0; k < j; ++k)
                        v -= (long double)L[i * 4 + k] * L[j * 4 + k];
                    if (i == j) {
                        if (!(v > 0) || !std::isfinite(double(v)))
                            throw UnrepresentableState("[MarginalizedBias] Bias precision not positive definite");
                        L[i * 4 + i] = std::sqrt(double(v));
                        r.log_det += 2 * std::log(L[i * 4 + i]);
                    } else
                        L[i * 4 + j] = double(v) / L[j * 4 + j];
                }
            auto solve_vector = [&](std::array<double, 4> x) {
                for (int i = 0; i < 4; ++i) {
                    for (int k = 0; k < i; ++k)
                        x[i] -= L[i * 4 + k] * x[k];
                    x[i] /= L[i * 4 + i];
                }
                for (int i = 3; i >= 0; --i) {
                    for (int k = i + 1; k < 4; ++k)
                        x[i] -= L[k * 4 + i] * x[k];
                    x[i] /= L[i * 4 + i];
                }
                return x;
            };
            const auto step = solve_vector(g0);
            for (int a = 0; a < 4; ++a)
                r.b_hat[a] = mean_[a] - step[a];
            for (int b = 0; b < 4; ++b) {
                std::array<double, 4> e{};
                e[b] = 1;
                const auto column = solve_vector(e);
                for (int a = 0; a < 4; ++a)
                    r.inverse[a * 4 + b] = column[a];
            }
            r.value = like.value(r.b_hat);
            for (int a = 0; a < 4; ++a)
                r.value += .5 * precision_[a] * (r.b_hat[a] - mean_[a]) * (r.b_hat[a] - mean_[a]);
            return r;
        }
        std::vector<double> parameters_for(const std::vector<double> & theta, const std::array<double, 4> & b) const {
            std::vector<double> p(theta);
            p.insert(p.end(), b.begin(), b.end());
            return p;
        }

      public:
        // The model's four bias priors must be Gaussian and unbounded (the marginal is then exact).
        BiasMarginalizedPosterior(const Model & model, Modes data, double noise_power)
            : model_(model), data_(std::move(data)), noise_power_(noise_power),
              variance_(noise_power / model.volume()),
              spectrum_size_(model.parameters().size() - 4),
              metric_likelihood_(Execution<Model>::template likelihood<GaussianFieldLikelihood>(model, data_)) {
            require(std::isfinite(noise_power) && noise_power > 0, "[MarginalizedBias] Invalid noise power");
            model.controls({noise_power});
            bool gaussian = true;
            for (int a = 0; a < 4; ++a) {
                const auto & prior = model.parameters()[spectrum_size_ + a];
                gaussian &= prior.prior == Prior::Gaussian && std::isinf(prior.lower) && std::isinf(prior.upper);
                mean_[a] = prior.mean;
                precision_[a] = 1 / (prior.sigma * prior.sigma);
            }
            require(gaussian, "[MarginalizedBias] Exact marginalization needs unbounded Gaussian bias priors");
            FML::COMMUNICATION::collective_try(
                [&] {
                    weights_.clear();
                    for (const auto & m : model.modes())
                        weights_.push_back(m.weight);
                },
                "[MarginalizedBias] Weights");
        }
        BiasMarginalizedPosterior(Model &&, Modes, double) = delete;
        const Model & model() const { return model_; }
        const Modes & data() const { return data_; }
        double noise_power() const { return noise_power_; }
        size_t parameter_count() const { return spectrum_size_; }
        // Number of evaluate() calls (forward + adjoint) made through this object.
        std::size_t evaluations() const { return evaluations_; }
        const std::vector<Parameter> spectrum_parameters() const {
            return {model_.parameters().begin(), model_.parameters().begin() + spectrum_size_};
        }
        // Exact Gaussian conditional of the bias coefficients (replicated on all ranks).
        Conditional conditional_bias(const Field & white, const std::vector<double> & theta, Workspace & w) const {
            require(theta.size() == spectrum_size_, "[MarginalizedBias] Spectrum parameter dimension");
            model_.predict(white, parameters_for(theta, mean_), w);
            const FixedNoiseLinearBiasLikelihood<4> like(model_.bias_modes(w), data_, weights_, variance_, mean_);
            const auto r = solve(like);
            Conditional c;
            c.mean = r.b_hat;
            c.covariance = r.inverse;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j <= i; ++j) {
                    double v = c.covariance[i * 4 + j];
                    for (int k = 0; k < j; ++k)
                        v -= c.cholesky[i * 4 + k] * c.cholesky[j * 4 + k];
                    c.cholesky[i * 4 + j] = i == j ? std::sqrt(v) : v / c.cholesky[j * 4 + j];
                }
            return c;
        }
        // -ln of the bias-marginalized posterior of (white, theta) up to a constant, and its gradient.
        double evaluate(const Field & white, const std::vector<double> & theta, Workspace & w, Field & gs,
                        std::vector<double> & gtheta) const {
            require(white.size() == model_.cells() && theta.size() == spectrum_size_,
                    "[MarginalizedBias] State dimension mismatch");
            ++evaluations_;
            auto p = parameters_for(theta, mean_);
            if (!model_.in_support(p)) {
                gs.assign(white.size(), 0);
                gtheta.assign(theta.size(), 0);
                return HUGE_VAL;
            }
            double U = 0;
            try {
                model_.predict(white, p, w);
                const auto columns = model_.bias_modes(w);
                const FixedNoiseLinearBiasLikelihood<4> like(columns, data_, weights_, variance_, mean_);
                const auto r = solve(like);
                U = r.value + .5 * r.log_det; // replicated (statistics are reduced)
                std::array<Modes, 4> g;
                FML::COMMUNICATION::collective_try(
                    [&] {
                        for (auto & column : g)
                            column.assign(data_.size(), Complex(0));
                        for (size_t k = 0; k < data_.size(); ++k) {
                            Complex residual = -data_[k];
                            for (int a = 0; a < 4; ++a)
                                residual += r.b_hat[a] * columns[a][k];
                            for (int a = 0; a < 4; ++a) {
                                Complex value = r.b_hat[a] * residual;
                                for (int b = 0; b < 4; ++b)
                                    value += r.inverse[a * 4 + b] * columns[b][k];
                                g[a][k] = value / variance_;
                            }
                        }
                    },
                    "[MarginalizedBias] Column cotangents");
                std::vector<double> gp;
                model_.pullback_columns(p, g, w, gs, gp);
                gtheta.assign(gp.begin(), gp.begin() + spectrum_size_);
            } catch (const UnrepresentableState &) {
                gs.assign(white.size(), 0);
                gtheta.assign(theta.size(), 0);
                return HUGE_VAL;
            }
            double prior = 0;
            for (size_t q = 0; q < white.size(); ++q) {
                prior += .5 * white[q] * white[q];
                gs[q] += white[q];
            }
            U += model_.sum(prior);
            for (auto & x : gtheta)
                x = model_.sum(x);
            for (size_t j = 0; j < spectrum_size_; ++j) {
                const auto & spec = model_.parameters()[j];
                U += spec.penalty(theta[j]);
                gtheta[j] += spec.derivative(theta[j]);
            }
            return U;
        }
        // The Eulerian posterior-aware field metric at the given bias values (frozen by the sampler).
        Field apply_field_metric(const Field & input, const std::vector<double> & theta,
                                 const std::array<double, 4> & bias, Workspace & w, double exponent) const {
            return model_.apply_gaussian_field_metric(input, parameters_for(theta, bias), w, exponent,
                                                      metric_likelihood_, std::log(noise_power_));
        }
    };

    // HMC on (white field, spectrum parameters) of the bias-marginalized posterior. The field metric is the
    // Eulerian one at the initial point with the bias at its conditional mean (frozen, as for sample()).
    // The sink receives (white, theta, draw); draw exact bias samples with conditional_bias if needed.
    template <class Grid>
    FML::MCMC::HMCResult sample_bias_marginalized(const BiasMarginalizedPosterior<Grid> & posterior, Field white,
                                                  std::vector<double> theta, FML::MCMC::HMCOptions options,
                                                  FieldSink sink = {}) {
        const auto & model = posterior.model();
        using E = Execution<typename BiasMarginalizedPosterior<Grid>::Model>;
        E::hmc_controls(model, true, options);
        auto w = model.workspace();
        const size_t n = model.cells(), np = theta.size();
        if (white.size() != n || np != posterior.parameter_count())
            throw std::invalid_argument("[MarginalizedBias] Invalid HMC initial state");
        const auto frozen_theta = theta;
        const auto frozen_bias = posterior.conditional_bias(white, theta, w).mean;
        FML::MCMC::HMCMass mass;
        mass.n_custom = n;
        mass.apply_sqrt = [&](auto & x) { x = posterior.apply_field_metric(x, frozen_theta, frozen_bias, w, .5); };
        mass.apply_inverse = [&](auto & x) { x = posterior.apply_field_metric(x, frozen_theta, frozen_bias, w, -1); };
        options.lower.clear();
        options.upper.clear();
        for (const auto & v : posterior.spectrum_parameters()) {
            mass.diagonal.push_back(1 / (v.scale() * v.scale()));
            options.lower.push_back(v.lower);
            options.upper.push_back(v.upper);
        }
        Field field(n), gs;
        std::vector<double> gt, params = theta;
        auto potential = [&](const auto & state, auto & gradient) {
            std::copy_n(state.begin(), n, field.begin());
            std::copy_n(state.begin() + n, np, params.begin());
            const double U = posterior.evaluate(field, params, w, gs, gt);
            std::copy(gs.begin(), gs.end(), gradient.begin());
            std::copy(gt.begin(), gt.end(), gradient.begin() + n);
            return U;
        };
        size_t draw = 0;
        auto recorder = [&](const auto & state) {
            std::copy_n(state.begin(), n, field.begin());
            std::copy_n(state.begin() + n, np, params.begin());
            if (sink)
                sink(field, params, draw);
            ++draw;
            return params;
        };
        white.insert(white.end(), theta.begin(), theta.end());
        return FML::MCMC::run_hmc(potential, std::move(white), std::move(mass), options, recorder, E::hmc(model));
    }
} // namespace FML::FIELDLEVEL::FRAMEWORK
#endif
