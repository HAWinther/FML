// Spectral sheet interpolation (the displacement field on the lattice from the owners of the lattice
// nodes, the tracers on the task of their position) on pencils gives the density modes of the slabs
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/SheetInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;

namespace {
    struct Part {
        double pos[3];
        double q[3];
        double * get_pos() { return pos; }
        double * get_q() { return q; }
        constexpr int get_ndim() const { return 3; }
    };
    const int nlattice = 12;

    std::vector<Part> mine() {
        std::vector<Part> v;
        long long id = 0;
        for (int i = 0; i < nlattice; i++)
            for (int j = 0; j < nlattice; j++)
                for (int k = 0; k < nlattice; k++, id++) {
                    Part p;
                    const int c[3] = {i, j, k};
                    for (int d = 0; d < 3; d++)
                        p.q[d] = (c[d] + 0.5) / nlattice;
                    for (int d = 0; d < 3; d++) {
                        double x = p.q[d] + 0.03 * std::sin(2 * M_PI * (p.q[(d + 1) % 3] + 0.2 * d)) +
                                   0.01 * std::cos(4 * M_PI * p.q[(d + 2) % 3]);
                        x -= std::floor(x);
                        p.pos[d] = x >= 1.0 ? 0.0 : x;
                    }
                    if (id % FML::NTasks == FML::ThisTask)
                        v.push_back(p);
                }
        return v;
    }
} // namespace

int main() {
    const int nmesh = 24;
    auto v = mine();
    FML::PARTICLE::MPIParticles<Part> pslab;
    {
        auto c = v;
        pslab.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
    }
    std::vector<std::string> methods{"CIC", "TSC", "NUFFT"};
    std::vector<std::vector<ComplexType>> ref;
    for (auto & m : methods) {
        FML::GRID::FFTWGrid<3> g(nmesh, 1, 2);
        FML::INTERPOLATION::particles_to_fourier_grid_sheet<3>(
            pslab.get_particles_ptr(), pslab.get_npart(), nlattice, 2, g, m, 1e-8);
        ref.push_back(global_fourier(g));
    }
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        const int p1 = FML::NTasks / p0;
        if (FML::NTasks % p0 or nlattice % p0 or nlattice % p1)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, p1);
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(p1);
        FML::GRID::PencilGrid<3> proto(nmesh, 0, 0, layout);
        FML::PARTICLE::MPIParticles<Part> p(FML::GRID::particle_domain_of(proto));
        {
            auto c = v;
            p.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        for (size_t i = 0; i < methods.size(); i++) {
            FML::GRID::PencilGrid<3> g(nmesh, 1, 2, layout);
            FML::INTERPOLATION::particles_to_fourier_grid_sheet<3>(
                p.get_particles_ptr(), p.get_npart(), nlattice, 2, g, methods[i], 1e-8);
            const double d = max_rel_diff(global_fourier(g), ref[i]);
            check(d < 1e-10, name + ": sheet density (refinement 2, " + methods[i] + ") = slabs (max diff " +
                                 num(d, 2) + ")");
        }
    }
    return FML::TESTS::summary("sheet interpolation on pencils");
}
