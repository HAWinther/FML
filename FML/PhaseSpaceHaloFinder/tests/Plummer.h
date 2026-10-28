#ifndef FML_PHASE_TEST_PLUMMER_HEADER
#define FML_PHASE_TEST_PLUMMER_HEADER
#include <FML/PhaseSpaceHaloFinder/Types.h>
#include <random>
namespace phase_test {
    /// Finite realization of an isotropic Plummer distribution, conditional on radius<r_cut.
    /// M=1, G=1, a=.015. Particle masses sum to 1; truncation slightly modifies the equilibrium potential.
    inline std::vector<FML::PHASE_SPACE_HALOS::Particle> plummer(std::size_t count, std::uint64_t seed = 867) {
        using namespace FML::PHASE_SPACE_HALOS;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform(0, 1);
        std::normal_distribution<double> normal(0, 1);
        std::vector<Particle> p(count);
        const double scale = .015, cut = .15, umax = std::pow(cut / std::sqrt(cut * cut + scale * scale), 3);
        for (std::size_t i = 0; i < count; ++i) {
            p[i].id = i;
            p[i].mass = 1.0 / count;
            const double u = std::max(1e-15, uniform(rng)) * umax;
            const double r = scale / std::sqrt(std::pow(u, -2.0 / 3) - 1);
            Vector direction{normal(rng), normal(rng), normal(rng)};
            double magnitude =
                std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
            for (int d = 0; d < 3; ++d)
                p[i].position[d] = .5 + r * direction[d] / magnitude;
            double q;
            do {
                q = uniform(rng);
            } while (.1 * uniform(rng) > q * q * std::pow(1 - q * q, 3.5));
            const double velocity = q * std::sqrt(2 / std::sqrt(r * r + scale * scale));
            direction = {normal(rng), normal(rng), normal(rng)};
            magnitude =
                std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
            for (int d = 0; d < 3; ++d)
                p[i].velocity[d] = velocity * direction[d] / magnitude;
        }
        return p;
    }
} // namespace phase_test
#endif
