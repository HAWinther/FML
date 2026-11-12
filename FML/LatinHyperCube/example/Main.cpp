// A Latin hypercube design for a 3-parameter cosmology emulator: draw the points in [0,1)^3, compare the
// optimizations by their centered discrepancy and scale the best design to the parameter ranges.

#include <FML/LatinHyperCube/LatinHyperCube.h>

#include <cstdio>

using namespace FML::SAMPLING;

int main() {
    const int npoints = 25; // 5^2 so strength 2 can be used as well
    const int ndim = 3;
    const uint64_t seed = 1234;

    struct Design {
        const char * name;
        int strength;
        LHSOptimization optimization;
    };
    const Design designs[] = {{"strength 1", 1, LHSOptimization::none},
                              {"strength 1 + random-cd", 1, LHSOptimization::random_cd},
                              {"strength 1 + lloyd", 1, LHSOptimization::lloyd},
                              {"strength 2", 2, LHSOptimization::none}};

    Sample best;
    double best_cd = 1e100;
    for (const auto & design : designs) {
        LatinHyperCube lhs(ndim, true, design.strength, design.optimization, seed);
        const Sample sample = lhs.random(npoints);
        const double cd = centered_discrepancy(sample);
        std::printf("%-24s centered discrepancy = %.6f\n", design.name, cd);
        if (cd < best_cd) {
            best_cd = cd;
            best = sample;
        }
    }

    // Omega_m in [0.1, 0.5), sigma8 in [0.6, 1.0), h in [0.55, 0.85)
    const Sample params = scale(best, {0.1, 0.6, 0.55}, {0.5, 1.0, 0.85});
    std::printf("\n# The design with the lowest discrepancy\n# %8s %10s %10s\n", "Omega_m", "sigma8", "h");
    for (const auto & x : params)
        std::printf("%10.5f %10.5f %10.5f\n", x[0], x[1], x[2]);
}
