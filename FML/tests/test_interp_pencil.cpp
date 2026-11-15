// Density assignment with interlacing (the half-cell shift and the higher orders), offset deposition
// and the deconvolution of the window function on pencils give the same density modes as on the FFTWGrid slabs.
// Bad input that must stop with a clear message:
//   mpirun -np 2 ./test_interp_pencil outside : a particle far outside the pencil of its task along y (1 x 2 pencils)
//   mpirun -np 2 ./test_interp_pencil zero    : no particles (NumPartTot = 0)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace I = FML::INTERPOLATION;

namespace {
    struct Part {
        double pos[3];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 3; }
    };

    std::vector<Part> mine(int npart) {
        std::mt19937 gen(99);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<Part> v;
        for (int i = 0; i < npart; i++) {
            Part p;
            for (int d = 0; d < 3; d++)
                p.pos[d] = U(gen);
            // (some exactly on cell faces)
            if (i % 50 == 0)
                p.pos[0] = (i / 50 % 16) / 16.0;
            if (i % FML::NTasks == FML::ThisTask)
                v.push_back(p);
        }
        return v;
    }

    template <template <int> class Grid>
    std::vector<double> densities(FML::PARTICLE::MPIParticles<Part> & p, const std::function<Grid<3>(int, int, int)> & make,
                                  int nmesh, size_t ntot) {
        std::vector<double> r;
        auto add = [&](const Grid<3> & g) {
            for (auto z : global_fourier(g)) {
                r.push_back(std::real(z));
                r.push_back(std::imag(z));
            }
        };
        for (std::string method : {"CIC", "TSC", "PCS"}) {
            for (bool interlacing : {false, true}) {
                auto g = make(nmesh, 2, 3);
                I::particles_to_fourier_grid<3>(p.get_particles_ptr(), p.get_npart(), ntot, g, method, interlacing);
                add(g);
                I::deconvolve_window_function_fourier<3>(g, method);
                add(g);
            }
            for (int order : {2, 4, 8}) {
                auto g = make(nmesh, 2, 3);
                I::particles_to_fourier_grid_interlaced<3>(p.get_particles_ptr(), p.get_npart(), ntot, g, method, order);
                add(g);
            }
        }
        return r;
    }
} // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "outside" or mode == "zero") {
        const auto layout = FML::PencilDomain::uniform(1, FML::NTasks);
        FML::GRID::PencilGrid<3> g(16, 1, 2, layout);
        // (the pencils split y: the particle of task 0 is 0.1 * 16 = 1.6 cells into the pencil of task 1)
        std::vector<Part> part(1, Part{{0.5, FML::ThisTask == 0 ? 0.5 + 0.1 : 0.6, 0.5}});
        if (mode == "zero")
            part.clear();
        I::particles_to_grid<3>(part.data(), part.size(), mode == "zero" ? 0 : FML::NTasks, g, "CIC");
        return 0;
    }
    const int nmesh = FML::NTasks % 3 == 0 ? 24 : 32, npart = 20000;
    auto v = mine(npart);
    FML::PARTICLE::MPIParticles<Part> pslab;
    {
        auto c = v;
        pslab.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
    }
    const auto ref = densities<FML::GRID::FFTWGrid>(
        pslab, [](int n, int l, int r) { return FML::GRID::FFTWGrid<3>(n, l, r); }, nmesh, npart);
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> proto(nmesh, 0, 0, layout);
        FML::PARTICLE::MPIParticles<Part> p(FML::GRID::particle_domain_of(proto));
        {
            auto c = v;
            p.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        const auto res = densities<FML::GRID::PencilGrid>(
            p, [&](int n, int l, int r) { return FML::GRID::PencilGrid<3>(n, l, r, layout); }, nmesh, npart);
        // Particles moved by a shift of less than a cell that stay on their task (not wrapped along the split axes,
        // as the force interlacing of the SimulationCode does): the grid with one more right extra cell than the
        // kernel needs takes them, and gives the deposition at the offset
        if (FML::NTasks > 1) {
            const std::array<double, 3> shift{0.37 / nmesh, 0.81 / nmesh, 0.5 / nmesh};
            const int wrap_type = FML::NTasks / p0 > 1 ? 3 : 2;
            double diff_shift = 0.0;
            for (std::string method : {"CIC", "TSC", "PCS"}) {
                const auto nlr = I::get_extra_slices(method);
                FML::GRID::PencilGrid<3> offset(nmesh, nlr.first, nlr.second + 1, layout);
                I::particles_to_grid<3>(p.get_particles_ptr(), p.get_npart(), npart, offset, method, shift, false);
                FML::GRID::PencilGrid<3> moved(nmesh, nlr.first, nlr.second + 1, layout);
                const auto saved = FML::PARTICLE::save_particle_positions<Part, 3>(p.get_particles_ptr(), p.get_npart());
                FML::PARTICLE::shift_particles<Part, 3>(p.get_particles_ptr(), p.get_npart(), shift, +1.0, wrap_type);
                I::particles_to_grid<3>(p.get_particles_ptr(), p.get_npart(), npart, moved, method);
                FML::PARTICLE::restore_particle_positions<Part, 3>(p.get_particles_ptr(), p.get_npart(), saved);
                std::vector<double> a, b;
                for (auto && index : offset.get_real_range()) {
                    a.push_back(offset.get_real_from_index(index));
                    b.push_back(moved.get_real_from_index(index));
                }
                diff_shift = std::max(diff_shift, max_rel_diff(b, a));
            }
            check(diff_shift < 1e-12,
                  "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                      ": particles shifted beyond the right edge (not wrapped) = deposition at the offset (rel. diff " +
                      num(diff_shift, 2) + ")");
        }
        const double diff = max_rel_diff(res, ref);
        check(diff < 1e-12,
              "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                  ": CIC/TSC/PCS with and without interlacing (orders 2, 4, 8) and deconvolved: the modes of the "
                  "slabs (rel. diff " +
                  num(diff, 2) + ")");
    }
    return FML::TESTS::summary("interlacing on pencils");
}
