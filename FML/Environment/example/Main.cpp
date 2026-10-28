#include <FML/Environment/EnvironmentClustering.h>
#include <iomanip>
#include <iostream>
using namespace FML::ENVIRONMENT;
int main() {
    const double L = 200, radius = 20;
    auto uniform = [&](size_t number, std::uint64_t seed) {
        auto queries = FML::COUNTS_IN_CELLS::uniform_queries(number, {{0, 0, 0}, {L, L, L}}, seed);
        std::vector<Sample> out;
        for (size_t i = 0; i < number; ++i)
            out.push_back({queries[i].position, i});
        return out;
    };
    // Uniform Poisson toy, replicated identically on every rank. Replace tracers
    // with an existing halo/HOD catalogue for a cosmological measurement.
    auto tracers = uniform(1000, 127), centers = uniform(2000, 251);
    auto env = top_hat_environment(tracers, centers, L, radius);
    auto split = density_split_correlation(centers, env.delta, tracers, 5, L, 5, 80, 8);
    auto tracer_env = top_hat_environment(tracers, tracers, L, radius, true);
    auto marks = inverse_density_marks(tracer_env.delta, 1, 1);
    auto marked = marked_correlation(tracers, marks, L, 5, 80, 8);
    if (!FML::ThisTask) {
        std::cout << std::setprecision(9) << "# periodic Poisson toy; L=" << L << " top-hat R=" << radius
                  << "\n# quantile sizes";
        for (auto n : split.quantiles.counts)
            std::cout << ' ' << n;
        std::cout << "\n# r xi_DS1 xi_DS2 xi_DS3 xi_DS4 xi_DS5 xi_parent M M_pair_normalized\n";
        for (size_t b = 0; b < split.parent.radius.size(); ++b) {
            std::cout << split.parent.radius[b];
            for (const auto& q : split.cross)
                std::cout << ' ' << q.xi[b];
            std::cout << ' ' << split.parent.xi[b] << ' ' << marked.mark_ratio[b] << ' '
                      << marked.pair_normalized_ratio[b] << '\n';
        }
    }
}
