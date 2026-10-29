#ifndef FML_MCMC_HMC_HEADER
#define FML_MCMC_HMC_HEADER
// Hamiltonian Monte Carlo for gradient-based posteriors of any dimension, e.g. field-level inference with 1e5-1e7
// parameters. Header only, C++17, no FML dependencies.
//
// The state is one vector split into two blocks:
//   [0, n_custom)   a block whose mass matrix is given as operators (M^{1/2} applied to white noise, and M^{-1}),
//                   e.g. a Fourier-diagonal filter for a field; nothing of it is stored per step;
//   [n_custom, n)   the parameters, with a diagonal mass adapted during warmup; they may have bounds (handled by
//                   reflection) and are recorded every step by default.
// The sampler is leapfrog HMC with a jittered number of steps, dual-averaging step-size adaptation (Hoffman & Gelman
// 2014) and Stan-like windowed adaptation of the diagonal mass. Use several chains (different seeds) and
// diagnose_hmc_chains on recorded quantities; legacy helpers remain available.
#include "Diagnostics.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace FML::MCMC {

    struct HMCMass {
        using Vector = std::vector<double>;
        std::size_t n_custom = 0;
        std::function<void(Vector &)> apply_sqrt;    // in place on the custom block: white noise -> M^{1/2} noise
        std::function<void(Vector &)> apply_inverse; // in place on the custom block: p -> M^{-1} p
        Vector diagonal; // mass of the parameter block (empty: 1); the starting point of the adaptation
    };

    struct HMCOptions {
        int warmup = 500;
        int samples = 1000;
        int n_leapfrog = 16;        // jittered uniformly in [n_leapfrog / 2, n_leapfrog] (fixed if jitter = false)
        bool jitter = true;
        double step_size = 0.1;     // initial; adapted during warmup (unless adapt_step_size = false)
        bool adapt_step_size = true;
        double target_acceptance = 0.8;
        bool adapt_mass = true;     // windowed estimate of the diagonal mass of the parameter block
        std::vector<double> lower, upper; // bounds of the parameter block (empty: unbounded; +-inf allowed)
        double divergence_threshold = 1000; // H1 - H0 above this counts as a divergence
        std::uint64_t seed = 1;
        int verbose_every = 0; // print progress every this many iterations (0: silent)
    };

    struct HMCResult {
        std::vector<std::vector<double>> recorded; // one row per sample (default: the parameter block)
        std::vector<double> potential;             // U per sample
        double acceptance = 0;                     // mean acceptance probability in the sampling phase
        long divergences = 0;                      // in the sampling phase
        double step_size = 0;                      // after adaptation
        std::vector<double> diagonal;              // parameter-block mass after adaptation
        std::vector<double> last;                  // final state
    };

    /// U(x) = -ln posterior and its gradient. Return +infinity (or NaN) for a point outside the support.
    using HMCPotential = std::function<double(const std::vector<double> & x, std::vector<double> & gradient)>;
    using HMCRecorder = std::function<std::vector<double>(const std::vector<double> & x)>;

    namespace detail {
        // Dual averaging of log(step size), Hoffman & Gelman (2014) section 3.2
        struct DualAveraging {
            double mu = 0, log_eps = 0, log_eps_bar = 0, h_bar = 0, gamma = .05, t0 = 10, kappa = .75;
            long t = 0;
            void restart(double eps) {
                mu = std::log(10 * eps);
                log_eps = std::log(eps);
                log_eps_bar = 0;
                h_bar = 0;
                t = 0;
            }
            double update(double accept_prob, double target) {
                ++t;
                const double w = 1. / (double(t) + t0);
                h_bar = (1 - w) * h_bar + w * (target - accept_prob);
                log_eps = mu - std::sqrt(double(t)) / gamma * h_bar;
                const double eta = std::pow(double(t), -kappa);
                log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar;
                return std::exp(log_eps);
            }
            double final_step() const { return std::exp(log_eps_bar); }
        };
        // Ends of the slow mass-adaptation windows (Stan's schedule: 15% initial buffer, windows of 25, 50, 100, ...
        // and a 10% final buffer; the last window absorbs the remainder)
        inline std::vector<int> mass_windows(int warmup) {
            std::vector<int> ends;
            if (warmup < 20)
                return ends;
            const int start = int(.15 * warmup), stop = warmup - int(.1 * warmup);
            int end = start, size = 25;
            while (end + size <= stop) {
                const int next = end + size;
                end = (next + 2 * size > stop) ? stop : next;
                ends.push_back(end);
                size *= 2;
            }
            if (ends.empty())
                ends.push_back(stop);
            return ends;
        }
    } // namespace detail

    inline HMCResult run_hmc(const HMCPotential & potential,
                             std::vector<double> x,
                             HMCMass mass,
                             const HMCOptions & opt,
                             const HMCRecorder & record = nullptr) {
        using Vector = std::vector<double>;
        const std::size_t n = x.size(), nc = mass.n_custom, np = n - std::min(n, nc);
        if (nc > n)
            throw std::invalid_argument("[run_hmc] n_custom exceeds the dimension");
        if (nc > 0 && (!mass.apply_sqrt || !mass.apply_inverse))
            throw std::invalid_argument("[run_hmc] The custom mass block needs apply_sqrt and apply_inverse");
        if (mass.diagonal.empty())
            mass.diagonal.assign(np, 1.);
        if (mass.diagonal.size() != np)
            throw std::invalid_argument("[run_hmc] diagonal must have one entry per parameter");
        for (double m : mass.diagonal)
            if (!(m > 0) || !std::isfinite(m))
                throw std::invalid_argument("[run_hmc] Diagonal masses must be positive");
        Vector lower = opt.lower, upper = opt.upper;
        if (lower.empty())
            lower.assign(np, -std::numeric_limits<double>::infinity());
        if (upper.empty())
            upper.assign(np, std::numeric_limits<double>::infinity());
        if (lower.size() != np || upper.size() != np)
            throw std::invalid_argument("[run_hmc] Bounds must have one entry per parameter");
        for (std::size_t i = 0; i < np; ++i)
            if (!(lower[i] < upper[i]) || x[nc + i] < lower[i] || x[nc + i] > upper[i])
                throw std::invalid_argument("[run_hmc] Invalid bounds or initial point outside them");
        if (opt.warmup < 0 || opt.samples < 0 || opt.n_leapfrog < 1 || !(opt.step_size > 0) ||
            !(opt.target_acceptance > 0 && opt.target_acceptance < 1))
            throw std::invalid_argument("[run_hmc] Invalid options");

        std::mt19937_64 rng(opt.seed);
        std::normal_distribution<double> gauss;
        std::uniform_real_distribution<double> uniform;

        Vector block(nc);
        // velocity v = M^{-1} p
        const auto velocity = [&](const Vector & p, Vector & v) {
            v.resize(n);
            if (nc) {
                std::copy(p.begin(), p.begin() + nc, block.begin());
                mass.apply_inverse(block);
                std::copy(block.begin(), block.end(), v.begin());
            }
            for (std::size_t i = 0; i < np; ++i)
                v[nc + i] = p[nc + i] / mass.diagonal[i];
        };
        const auto kinetic = [&](const Vector & p, const Vector & v) {
            double K = 0;
            for (std::size_t i = 0; i < n; ++i)
                K += p[i] * v[i];
            return .5 * K;
        };

        Vector grad(n), grad_new(n), x_new, p(n), v(n);
        double U = potential(x, grad);
        if (!std::isfinite(U))
            throw std::invalid_argument("[run_hmc] The potential is not finite at the initial point");

        double eps = opt.step_size;
        detail::DualAveraging da;
        da.restart(eps);
        const auto windows = detail::mass_windows(opt.warmup);
        std::size_t window = 0;
        // Running moments of the parameter block in the current window (Welford)
        Vector mean(np, 0), m2(np, 0);
        long count = 0;

        HMCResult result;
        double accept_sum = 0;
        for (int it = 0; it < opt.warmup + opt.samples; ++it) {
            const bool warm = it < opt.warmup;
            // Momentum p ~ N(0, M)
            for (std::size_t i = 0; i < nc; ++i)
                block[i] = gauss(rng);
            if (nc) {
                mass.apply_sqrt(block);
                std::copy(block.begin(), block.end(), p.begin());
            }
            for (std::size_t i = 0; i < np; ++i)
                p[nc + i] = std::sqrt(mass.diagonal[i]) * gauss(rng);
            velocity(p, v);
            const double H0 = U + kinetic(p, v);

            x_new = x;
            grad_new = grad;
            double U_new = U;
            const int nlf = opt.jitter ? opt.n_leapfrog / 2 + int(uniform(rng) * (opt.n_leapfrog - opt.n_leapfrog / 2 + 1))
                                       : opt.n_leapfrog;
            bool finite = true;
            for (int l = 0; l < std::max(1, nlf) && finite; ++l) {
                for (std::size_t i = 0; i < n; ++i)
                    p[i] -= .5 * eps * grad_new[i];
                velocity(p, v);
                for (std::size_t i = 0; i < n; ++i)
                    x_new[i] += eps * v[i];
                // Reflect bounded parameters (diagonal mass, so flipping p_i flips v_i): volume preserving, reversible
                for (std::size_t i = 0; i < np; ++i) {
                    double & y = x_new[nc + i];
                    for (int r = 0; r < 100 && (y < lower[i] || y > upper[i]); ++r) {
                        y = y < lower[i] ? 2 * lower[i] - y : 2 * upper[i] - y;
                        p[nc + i] = -p[nc + i];
                    }
                    if (y < lower[i] || y > upper[i])
                        finite = false;
                }
                if (!finite)
                    break;
                U_new = potential(x_new, grad_new);
                if (!std::isfinite(U_new)) {
                    finite = false;
                    break;
                }
                for (std::size_t i = 0; i < n; ++i)
                    p[i] -= .5 * eps * grad_new[i];
            }
            double H1 = std::numeric_limits<double>::infinity();
            if (finite) {
                velocity(p, v);
                H1 = U_new + kinetic(p, v);
            }
            const double accept_prob = std::isfinite(H1) ? std::min(1., std::exp(H0 - H1)) : 0.;
            if (!warm && !(H1 - H0 <= opt.divergence_threshold))
                ++result.divergences;
            if (uniform(rng) < accept_prob) {
                x.swap(x_new);
                grad.swap(grad_new);
                U = U_new;
            }

            if (warm) {
                if (opt.adapt_step_size)
                    eps = da.update(accept_prob, opt.target_acceptance);
                if (opt.adapt_mass && window < windows.size() && it >= int(.15 * opt.warmup)) {
                    ++count;
                    for (std::size_t i = 0; i < np; ++i) {
                        const double d = x[nc + i] - mean[i];
                        mean[i] += d / double(count);
                        m2[i] += d * (x[nc + i] - mean[i]);
                    }
                    if (it + 1 == windows[window]) {
                        // Regularised variance (as Stan) -> mass = 1 / variance; then restart the step adaptation
                        for (std::size_t i = 0; i < np; ++i) {
                            const double var = count > 1 ? m2[i] / double(count - 1) : 0;
                            const double reg = (double(count) / (count + 5.)) * var + 1e-3 * 5. / (count + 5.);
                            if (reg > 0 && std::isfinite(reg))
                                mass.diagonal[i] = 1 / reg;
                        }
                        std::fill(mean.begin(), mean.end(), 0.);
                        std::fill(m2.begin(), m2.end(), 0.);
                        count = 0;
                        ++window;
                        if (opt.adapt_step_size)
                            da.restart(eps);
                    }
                }
                if (it + 1 == opt.warmup && opt.adapt_step_size)
                    eps = da.final_step();
            } else {
                accept_sum += accept_prob;
                result.recorded.push_back(record ? record(x) : Vector(x.begin() + nc, x.end()));
                result.potential.push_back(U);
            }
            if (opt.verbose_every > 0 && it % opt.verbose_every == 0) {
                std::cout << "[run_hmc] " << (warm ? "warmup " : "sample ") << it << " U " << U << " step " << eps
                          << " accept " << accept_prob;
                for (std::size_t i = 0; i < std::min<std::size_t>(np, 4); ++i)
                    std::cout << " p" << i << " " << x[nc + i];
                std::cout << "\n";
            }
        }
        result.acceptance = opt.samples ? accept_sum / opt.samples : 0;
        result.step_size = eps;
        result.diagonal = mass.diagonal;
        result.last = std::move(x);
        return result;
    }

    /// Legacy scalar Sokal-window IAT; insufficient, constant or nonfinite draws return infinity (ESS=0).
    inline double integrated_autocorrelation_time(const std::vector<double> & x) {
        const std::size_t n = x.size();
        if (n < 4)
            return std::numeric_limits<double>::infinity();
        for (double value : x)
            if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
        if (std::all_of(x.begin(), x.end(), [&](double value) { return value == x.front(); }))
            return std::numeric_limits<double>::infinity();
        double mean = 0, var = 0;
        for (double v : x)
            mean += v;
        mean /= double(n);
        for (double v : x)
            var += (v - mean) * (v - mean);
        var /= double(n);
        if (!(var > 0) || !std::isfinite(var))
            return std::numeric_limits<double>::infinity();
        double tau = 1;
        for (std::size_t lag = 1; lag < n / 2; ++lag) {
            double c = 0;
            for (std::size_t i = 0; i + lag < n; ++i)
                c += (x[i] - mean) * (x[i + lag] - mean);
            tau += 2 * c / (double(n - lag) * var);
            if (double(lag) >= 5 * tau)
                break;
        }
        return std::max(tau, 1.);
    }

    /// Column j of the recorded rows
    inline std::vector<double> recorded_column(const HMCResult & r, std::size_t j) {
        std::vector<double> c;
        c.reserve(r.recorded.size());
        for (const auto & row : r.recorded)
            c.push_back(row.at(j));
        return c;
    }

    /// Post-warmup recorded scalar across independent HMC runs (not successive chunks of one run).
    inline ChainDiagnostics diagnose_hmc_chains(const std::vector<HMCResult> & runs, std::size_t column = 0) {
        std::vector<std::vector<double>> chains;
        for (const auto & run : runs) chains.push_back(recorded_column(run, column));
        return diagnose_chains(chains);
    }

    /// Gelman-Rubin R - 1 of a scalar from several chains of equal length
    inline double gelman_rubin(const std::vector<std::vector<double>> & chains) {
        const std::size_t m = chains.size();
        if (m < 2)
            throw std::invalid_argument("[gelman_rubin] Needs at least two chains");
        const std::size_t n = chains[0].size();
        for (const auto & c : chains) {
            if (c.size() != n || n < 2)
                throw std::invalid_argument("[gelman_rubin] Chains must have equal length >= 2");
        }
        for (const auto & c : chains) {
            if (std::any_of(c.begin(), c.end(), [](double v) { return !std::isfinite(v); }) ||
                std::all_of(c.begin(), c.end(), [&](double v) { return v == c.front(); }))
                return std::numeric_limits<double>::infinity();
        }
        std::vector<double> means(m);
        double grand = 0, W = 0;
        for (std::size_t j = 0; j < m; ++j) {
            for (double v : chains[j])
                means[j] += v;
            means[j] /= double(n);
            grand += means[j] / double(m);
            double s2 = 0;
            for (double v : chains[j])
                s2 += (v - means[j]) * (v - means[j]);
            W += s2 / double(n - 1) / double(m);
        }
        double B = 0;
        for (double mj : means)
            B += (mj - grand) * (mj - grand);
        B *= double(n) / double(m - 1);
        const double var_plus = (double(n - 1) / double(n)) * W + B / double(n);
        if (!(W > 0) || !std::isfinite(W) || !std::isfinite(var_plus))
            return std::numeric_limits<double>::infinity();
        return std::sqrt(var_plus / W) - 1;
    }
} // namespace FML::MCMC
#endif
