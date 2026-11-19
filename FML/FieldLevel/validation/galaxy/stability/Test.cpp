#include "Probe.h"
#include <iostream>
struct GaussianModel {};
namespace FML::FIELDLEVEL::FRAMEWORK {
    template <>
    struct Execution<GaussianModel> {
        static FML::MCMC::HMCExecution hmc(const GaussianModel &) { return {}; }
    };
} // namespace FML::FIELDLEVEL::FRAMEWORK
struct GaussianTarget {
    GaussianModel m;
    const GaussianModel & model() const { return m; }
    double evaluate(const std::vector<double> & x,
                    const std::vector<double> &,
                    int &,
                    std::vector<double> & g,
                    std::vector<double> & gp) const {
        g = {10000 * x[0], 4000 * x[1]};
        gp.clear();
        return .5 * (x[0] * g[0] + x[1] * g[1]);
    }
    std::vector<double>
    apply_field_metric(std::vector<double> x, const std::vector<double> & metric, int &, double exponent) const {
        for (size_t j = 0; j < x.size(); ++j)
            x[j] *= std::pow(metric[j], exponent);
        return x;
    }
};
int main() {
    try {
        GaussianTarget target;
        int work = 0;
        std::vector<double> x = {.01, -.015};
        double accepted[3] = {0, 0, 0};
        long divergences[3] = {0, 0, 0};
        for (int option = 0; option < 3; ++option)
            for (unsigned seed = 1; seed <= 32; ++seed) {
                FML::MCMC::HMCOptions o;
                o.warmup = 0;
                o.samples = 1;
                o.n_leapfrog = option == 1 ? 800 : 8;
                o.step_size = option == 1 ? .0015 : .15;
                o.jitter = false;
                o.adapt_step_size = false;
                o.adapt_mass = false;
                o.seed = seed;
                const std::vector<double> metric =
                    option == 2 ? std::vector<double>{10000, 4000} : std::vector<double>{1, 1};
                auto result = STABILITY::trajectory(target, x, {}, metric, o, work);
                FML::MCMC::HMCMass mass;
                mass.n_custom = 2;
                mass.apply_sqrt = [&](auto & v) { v = target.apply_field_metric(v, metric, work, .5); };
                mass.apply_inverse = [&](auto & v) { v = target.apply_field_metric(v, metric, work, -1.); };
                auto potential = [&](const auto & v, auto & g) {
                    std::vector<double> gp;
                    return target.evaluate(v, {}, work, g, gp);
                };
                const auto native = FML::MCMC::run_hmc(potential, x, mass, o);
                if (native.last != result.hmc.last || native.acceptance != result.hmc.acceptance ||
                    native.divergences != result.hmc.divergences)
                    throw std::runtime_error("Gaussian native/probe mismatch");
                accepted[option] += result.hmc.acceptance;
                divergences[option] += result.hmc.divergences;
            }
        if (divergences[0] != 32 || divergences[1] != 0 || divergences[2] != 0 || accepted[1] / 32 < .98 ||
            accepted[2] / 32 < .98)
            throw std::runtime_error("Analytic stiff-Gaussian control failed");
        std::cout
            << "PASS: 96 exact native/probe Gaussian matches; instability, small-step and matched-metric controls\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
