// SPH (routing of the particles to their task and the ghosts within the kernel support) on a pencil
// FML::domain evolves the gas as on the slabs (2D and 3D)
#include <FML/Global/Global.h>
#include <FML/SPH/SPH.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <cmath>
#include <string>

using namespace FML::SPH;
using FML::TESTS::check;
using FML::TESTS::num;
namespace C = FML::COMMUNICATION;

namespace {
    template <int D>
    std::vector<Particle<D>> lattice(int n) {
        size_t count = 1;
        for (int d = 0; d < D; ++d)
            count *= n;
        std::vector<Particle<D>> p;
        for (size_t id = 0; id < count; ++id) {
            if (int(id % FML::NTasks) != FML::ThisTask)
                continue;
            Particle<D> q;
            q.id = id;
            q.mass = (1. + .25 * std::cos(.67 * id)) / count;
            q.energy = 1 + .15 * std::sin(.79 * id);
            size_t k = id;
            for (int d = 0; d < D; ++d) {
                q.position[d] = (k % n + .5) / n + .08 / n * std::sin(.53 * (id + 1) * (d + 1));
                k /= n;
                q.velocity[d] = .1 * std::sin(.37 * (id + 1) * (d + 2));
            }
            p.push_back(q);
        }
        return p;
    }

    // The particles of all tasks sorted by ID: id, position, velocity, energy, h
    template <int D>
    std::vector<std::array<double, 2 * D + 3>> evolve(int n, int steps) {
        Options o;
        o.h_min = .45 / n;
        o.h_max = 1.7 / n;
        o.conductivity_alpha = .2;
        o.max_message_bytes = 4096;
        Solver<D> gas(lattice<D>(n), o);
        for (int s = 0; s < steps; s++)
            gas.advance(0.05);
        std::vector<std::array<double, 2 * D + 3>> local;
        for (auto & p : gas.particles()) {
            std::array<double, 2 * D + 3> r;
            r[0] = double(p.id);
            for (int d = 0; d < D; d++) {
                r[1 + d] = p.position[d];
                r[1 + D + d] = p.velocity[d];
            }
            r[1 + 2 * D] = p.energy;
            r[2 + 2 * D] = p.h;
            local.push_back(r);
        }
        auto all = C::gather_records_checked(local, true);
        std::sort(all.begin(), all.end());
        return all;
    }

    template <int D>
    void compare(int n, int steps) {
        const auto ref = evolve<D>(n, steps);
        const FML::Domain slab_domain = FML::domain;
        for (int p0 = 1; p0 <= FML::NTasks; p0++) {
            if (FML::NTasks % p0)
                continue;
            FML::set_domain(FML::Domain::uniform_pencil(p0, FML::NTasks / p0));
            const auto res = evolve<D>(n, steps);
            FML::set_domain(slab_domain);
            double diff = res.size() == ref.size() ? 0.0 : 1e100;
            for (size_t i = 0; diff < 1e100 and i < res.size(); i++)
                for (size_t j = 0; j < res[i].size(); j++) {
                    double d = std::abs(res[i][j] - ref[i][j]);
                    if (j >= 1 and j <= size_t(D))
                        d = std::min(d, 1.0 - d); // periodic positions
                    diff = std::max(diff, d);
                }
            check(diff < 1e-12, std::to_string(D) + "D, pencils " + std::to_string(p0) + "x" +
                                    std::to_string(FML::NTasks / p0) + ": " + std::to_string(steps) +
                                    " steps as on slabs (max diff " + num(diff, 2) + ")");
        }
    }
} // namespace

int main() {
    compare<2>(24, 4);
    compare<3>(10, 3);
    return FML::TESTS::summary("SPH on pencils");
}
