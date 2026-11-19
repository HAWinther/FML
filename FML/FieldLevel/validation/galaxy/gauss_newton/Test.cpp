#include "Metric.h"
#include <iostream>
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        const double c = std::cos(.37), s = std::sin(.37);
        GN::Metric metric;
        metric.dimension = 2;
        metric.eigenvalues = {3, 99};
        metric.vectors = {{c, s}, {-s, c}};
        metric.validate();
        for (double beta : {0., .25, 1.}) {
            const double a = 1 + beta * (3 * c * c + 99 * s * s), b = beta * (3 - 99) * c * s,
                         d = 1 + beta * (3 * s * s + 99 * c * c), det = a * d - b * b;
            for (auto v : {std::vector<double>{.2, -.3}, std::vector<double>{1., 0.}, std::vector<double>{0., 1.}}) {
                const auto inv = metric.apply(v, beta, -1.), full = metric.apply(v, beta, 1.),
                           root = metric.apply(metric.apply(v, beta, .5), beta, .5);
                const std::vector<double> expected_inv = {(d * v[0] - b * v[1]) / det, (-b * v[0] + a * v[1]) / det},
                                          expected_full = {a * v[0] + b * v[1], b * v[0] + d * v[1]};
                for (size_t j = 0; j < 2; ++j)
                    if (std::abs(inv[j] - expected_inv[j]) > 1e-12 || std::abs(full[j] - expected_full[j]) > 1e-11 ||
                        std::abs(root[j] - full[j]) > 1e-11)
                        throw std::runtime_error("Independent dense inverse/root control failed");
            }
            for (unsigned seed = 1; seed <= 16; ++seed) {
                FML::MCMC::HMCOptions o;
                o.warmup = 0;
                o.samples = 1;
                o.step_size = .075;
                o.n_leapfrog = 16;
                o.jitter = false;
                o.adapt_step_size = false;
                o.adapt_mass = false;
                o.seed = seed;
                FML::MCMC::HMCMass mass;
                mass.n_custom = 2;
                mass.apply_sqrt = [&](auto & v) { v = metric.apply(v, beta, .5); };
                mass.apply_inverse = [&](auto & v) { v = metric.apply(v, beta, -1.); };
                auto potential = [&](const auto & v, auto & g) {
                    g = {a * v[0] + b * v[1], b * v[0] + d * v[1]};
                    return .5 * (v[0] * g[0] + v[1] * g[1]);
                };
                std::vector<double> initial = {.02, -.015};
                const auto result = FML::MCMC::run_hmc(potential, initial, mass, o);
                std::mt19937_64 rng(seed);
                std::normal_distribution<double> normal;
                std::uniform_real_distribution<double> uniform;
                const double n0 = normal(rng), n1 = normal(rng), m0 = 1 + 3 * beta, m1 = 1 + 99 * beta;
                std::array<double, 2> q = {std::sqrt(m0) * (c * initial[0] + s * initial[1]),
                                           std::sqrt(m1) * (-s * initial[0] + c * initial[1])},
                                      p = {c * n0 + s * n1, -s * n0 + c * n1};
                const double h0 = .5 * (q[0] * q[0] + q[1] * q[1] + p[0] * p[0] + p[1] * p[1]);
                for (int l = 0; l < 16; ++l)
                    for (size_t j = 0; j < 2; ++j) {
                        const double h = .075, x = (1 - h * h / 2) * q[j] + h * p[j];
                        p[j] = -h * (1 - h * h / 4) * q[j] + (1 - h * h / 2) * p[j];
                        q[j] = x;
                    }
                const double h1 = .5 * (q[0] * q[0] + q[1] * q[1] + p[0] * p[0] + p[1] * p[1]),
                             prob = std::min(1., std::exp(h0 - h1));
                std::vector<double> expected = {c * q[0] / std::sqrt(m0) - s * q[1] / std::sqrt(m1),
                                                s * q[0] / std::sqrt(m0) + c * q[1] / std::sqrt(m1)};
                if (uniform(rng) >= prob)
                    expected = initial;
                if (std::abs(prob - result.acceptance) > 1e-11)
                    throw std::runtime_error("Independent harmonic acceptance failed");
                for (size_t j = 0; j < 2; ++j)
                    if (std::abs(expected[j] - result.last[j]) > 1e-12)
                        throw std::runtime_error("Independent harmonic trajectory failed");
            }
        }
        auto with_null = metric;
        with_null.dimension = 3;
        for (auto & v : with_null.vectors)
            v.push_back(0);
        with_null.validate();
        for (double beta : {0., .1, 1.})
            for (double power : {-1., -.5, 0., .5, 1.}) {
                const auto x = with_null.apply({0, 0, 2.5}, beta, power);
                if (x != std::vector<double>({0, 0, 2.5}))
                    throw std::runtime_error("Prior-only direction changed");
            }
        for (unsigned variant = 0; variant < 4; ++variant) {
            auto bad = metric;
            if (variant == 0)
                bad.eigenvalues[0] = -1;
            if (variant == 1)
                bad.vectors[0][0] = HUGE_VAL;
            if (variant == 2)
                bad.vectors[1] = bad.vectors[0];
            if (variant == 3)
                bad.dimension = 0;
            bool rejected = false;
            try {
                bad.validate();
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Invalid metric accepted");
        }
        std::cout << "PASS: independent dense inverses/roots, 48 harmonic HMC controls, exact prior null action, "
                     "invalid metric rejection\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
