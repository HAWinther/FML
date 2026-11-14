// Glass initial conditions on a pencil FML::domain (long-range force on PencilGrids, short-range ghosts
// from the domain, migration and ownership by the pencils): the relaxation, the displacement of the glass by IC
// fields and save/load give what the slabs give
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/GlassIC/GlassIC.h>
#include <FML/GlassIC/LoadIO.h>
#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace GL = FML::GLASSIC;

namespace {
    const int mesh = 24;
    // All the particles (sorted by ID): id, position, velocity
    std::vector<std::array<double, 7>> all(GL::Load & load) {
        std::vector<std::array<double, 7>> local;
        for (size_t i = 0; i < load.particles.get_npart(); i++) {
            const auto & p = load.particles.get_particles_ptr()[i];
            local.push_back({double(p.id), p.position[0], p.position[1], p.position[2], p.velocity[0], p.velocity[1],
                             p.velocity[2]});
        }
        auto r = FML::COMMUNICATION::gather_records_checked(local, true);
        std::sort(r.begin(), r.end());
        return r;
    }
    double diff(const std::vector<std::array<double, 7>> & a, const std::vector<std::array<double, 7>> & b) {
        if (a.size() != b.size())
            return 1e100;
        double d = 0;
        for (size_t i = 0; i < a.size(); i++)
            for (int j = 0; j < 7; j++) {
                double x = std::abs(a[i][j] - b[i][j]);
                if (j >= 1 and j <= 3)
                    x = std::min(x, 1 - x);
                d = std::max(d, x);
            }
        return d;
    }
    double field(int c, const std::array<double, 3> & x) {
        return 0.01 * std::sin(2 * M_PI * (x[(c + 1) % 3] + 0.1 * c)) + 0.005 * std::cos(2 * M_PI * x[c]);
    }
    // Relax a few steps, then displace by the fields (on grids of type G)
    template <class G>
    std::vector<std::array<double, 7>> run(std::vector<std::array<double, 7>> & relaxed, const std::string & save) {
        auto load = GL::poisson_load(4096, 17);
        GL::Options o;
        o.mesh = mesh;
        o.max_steps = 4;
        o.min_steps = 10;
        GL::relax<G>(load, o);
        relaxed = all(load);
        if (not save.empty())
            GL::save_load(load, save);
        std::array<G, 3> disp, vel;
        for (int c = 0; c < 3; c++) {
            disp[c] = FML::GRID::make_grid<G>(mesh, 1, 1, FML::domain);
            vel[c] = FML::GRID::make_grid<G>(mesh, 1, 1, FML::domain);
            fill<3>(disp[c], [&](auto x) { return field(c, x); });
            fill<3>(vel[c], [&](auto x) { return 2.0 * field(c, x); });
        }
        GL::apply_fields<G>(load, disp, vel);
        return all(load);
    }
} // namespace

int main() {
    std::vector<std::array<double, 7>> ref_relaxed;
    const auto ref = run<FML::GRID::FFTWGrid<3>>(ref_relaxed, "");
    const FML::Domain slab = FML::domain;
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or mesh % p0 or mesh % p1)
            continue;
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(p1);
        const std::string prefix = "test_glass_pencil_" + std::to_string(p0);
        FML::set_domain(FML::Domain::uniform_pencil(p0, p1));
        std::vector<std::array<double, 7>> relaxed;
        const auto res = run<FML::GRID::PencilGrid<3>>(relaxed, prefix);
        // The glass saved on pencils, loaded on slabs
        FML::set_domain(slab);
        auto back = GL::load_load(prefix);
        auto loaded = all(back);
        const double d1 = diff(relaxed, ref_relaxed), d2 = diff(res, ref);
        double d3 = loaded.size() == relaxed.size() ? 0.0 : 1e100;
        for (size_t i = 0; d3 < 1e100 and i < loaded.size(); i++)
            for (int j = 0; j < 4; j++)
                d3 = std::max(d3, std::abs(loaded[i][j] - relaxed[i][j]));
        check(d1 < 1e-10, name + ": 4 relaxation steps as on the slabs (max diff " + num(d1, 2) + ")");
        check(d2 < 1e-10, name + ": the glass displaced by IC fields on PencilGrids as on the slabs (max diff " +
                              num(d2, 2) + ")");
        check(d3 < 1e-15, name + ": the glass saved on pencils and loaded on slabs (max diff " + num(d3, 2) + ")");
#ifdef USE_MPI
        MPI_Barrier(FML::communicator);
#endif
        if (FML::ThisTask == 0) {
            std::remove((prefix + ".manifest").c_str());
            for (int t = 0; t < FML::NTasks; t++)
                std::remove((prefix + ".rank" + std::to_string(t) + ".glass").c_str());
        }
    }
    return FML::TESTS::summary("glass ICs on pencils");
}
