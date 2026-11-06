#ifndef SIMCODE_DYNAMICS_FIXTURES_H
#define SIMCODE_DYNAMICS_FIXTURES_H
#include "ClusteredForceReference.h"
#include <stdexcept>
#include <string>
namespace SIMCODE::DYNAMICS_TEST {
    using Vector = CLUSTER_REFERENCE::Vector;
    struct Particle {
        Vector x{}, v{};
        double mass{1};
    };
    struct Fixture {
        std::vector<Particle> particles;
        double begin{.5}, end{.65}, softening{1.25};
    };
    inline Fixture fixture(const std::string & name, double omega_m) {
        Fixture f;
        const double pi = std::acos(-1.);
        if (name == "circular" || name == "eccentric") {
            f.particles.resize(2);
            const double r = .06;
            const double speed = std::sqrt(1.5 * omega_m * f.begin / (4 * pi * r)) * (name == "eccentric" ? .65 : 1.);
            for (int i = 0; i < 2; ++i) {
                auto & p = f.particles[i];
                p.x = {(name == "eccentric" ? .995 : .5) + (i == 0 ? -.5 : .5) * r, .5, .5};
                p.v = {name == "eccentric" ? .08 : 0., (i == 0 ? -.5 : .5) * speed, 0.};
            }
        } else if (name == "cluster" || name == "merger") {
            f.particles.resize(16);
            f.end = name == "cluster" ? .54 : .53;
            for (int i = 0; i < 16; ++i) {
                auto & p = f.particles[i];
                const int group = name == "merger" ? i % 2 : 0, j = name == "merger" ? i / 2 : i,
                          n = name == "merger" ? 8 : 16;
                const double z = 1 - 2 * (j + .5) / n, phi = pi * (3 - std::sqrt(5.)) * j;
                Vector direction{std::sqrt(1 - z * z) * std::cos(phi), std::sqrt(1 - z * z) * std::sin(phi), z};
                const double radius = (name == "merger" ? .011 : .022) * (1 + .18 * (j % 3));
                for (int d = 0; d < 3; ++d)
                    p.x[d] = .5 + radius * direction[d];
                if (name == "merger")
                    p.x[0] += group == 0 ? -.04 : .04;
                // Deterministic tangent directions, not a sampled equilibrium distribution.
                Vector axis{std::sin(j + 1.), std::cos(2 * j + 1.), .7}, tangent;
                for (int d = 0; d < 3; ++d)
                    tangent[d] =
                        axis[(d + 1) % 3] * direction[(d + 2) % 3] - axis[(d + 2) % 3] * direction[(d + 1) % 3];
                const double speed = name == "merger" ? .25 : .45, norm = CLUSTER_REFERENCE::norm(tangent);
                for (int d = 0; d < 3; ++d)
                    p.v[d] = speed * tangent[d] / norm;
                if (name == "merger")
                    p.v[0] += group == 0 ? .2 : -.2;
                p.mass = .8 + .4 * (i + .5) / 16;
            }
            double mass = 0;
            Vector mean{};
            for (const auto & p : f.particles) {
                mass += p.mass;
                for (int d = 0; d < 3; ++d)
                    mean[d] += p.mass * p.v[d];
            }
            for (auto & p : f.particles)
                for (int d = 0; d < 3; ++d)
                    p.v[d] -= mean[d] / mass;
        } else
            throw std::invalid_argument("Unknown dynamics fixture: " + name);
        return f;
    }
} // namespace SIMCODE::DYNAMICS_TEST
#endif
