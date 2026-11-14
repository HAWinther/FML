// The DTFE (Triangulation/DTFE.h) on a pencil domain interpolates the density, fields with their
// gradients and the velocity (divergence, vorticity, shear) to a PencilGrid as on the slabs (USE_CGAL=true)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/Triangulation/DTFE.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace T = FML::TRIANGULATION;

namespace {
    struct Part {
        double pos[3];
        double vel[3];
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        constexpr int get_ndim() const { return 3; }
    };

    std::vector<Part> mine(int npart) {
        std::mt19937 gen(5);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<Part> v;
        for (int i = 0; i < npart; i++) {
            Part p;
            for (int d = 0; d < 3; d++)
                p.pos[d] = U(gen);
            for (int d = 0; d < 3; d++)
                p.vel[d] = std::sin(2 * M_PI * p.pos[(d + 1) % 3]) + 0.3 * std::cos(2 * M_PI * p.pos[d]);
            if (i % FML::NTasks == FML::ThisTask)
                v.push_back(p);
        }
        return v;
    }

    // Everything on the grids of one DTFE, gathered by global cell
    template <class G>
    std::vector<double> grids(T::DTFE<Part> & dtfe, Part * p, size_t n, int nmesh) {
        std::vector<double> r;
        auto add = [&](const G & g) {
            auto x = global_real(g);
            r.insert(r.end(), x.begin(), x.end());
        };
        G density;
        dtfe.template density_to_grid<G>(nmesh, density);
        add(density);
        std::vector<double> values(n);
        for (size_t i = 0; i < n; i++)
            values[i] = p[i].pos[0] * p[i].pos[1] + std::cos(2 * M_PI * p[i].pos[2]);
        std::vector<G> fields, gradients;
        dtfe.template fields_to_grid<G>(nmesh, values, 1, fields, &gradients, T::DTFESampling::CellAverage,
                                        T::DTFEWeighting::Mass);
        add(fields[0]);
        for (auto & g : gradients)
            add(g);
        T::DTFEVelocityGridsT<G> vel;
        dtfe.template velocity_to_grid<G>(p, nmesh, vel, T::DTFESampling::GridPoint);
        for (auto & g : vel.velocity)
            add(g);
        add(vel.divergence);
        for (auto & g : vel.vorticity)
            add(g);
        for (auto & g : vel.shear)
            add(g);
        return r;
    }
} // namespace

int main() {
    const int nmesh = FML::NTasks % 3 == 0 ? 12 : 16, npart = 6000;
    auto v = mine(npart);
    FML::PARTICLE::MPIParticles<Part> pslab;
    {
        auto c = v;
        pslab.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
    }
    T::DTFE<Part> dslab;
    // (thin slabs need a buffer of more than the neighbouring slab)
    dslab.create(pslab.get_particles_ptr(), pslab.get_npart(), FML::NTasks >= 5 ? 2.0 : 0.9);
    const auto ref = grids<FML::GRID::FFTWGrid<3>>(dslab, pslab.get_particles_ptr(), pslab.get_npart(), nmesh);
    dslab.free();

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> proto(nmesh, 0, 0, layout);
        const auto domain = FML::GRID::particle_domain_of(proto);
        FML::PARTICLE::MPIParticles<Part> p(domain);
        {
            auto c = v;
            p.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        T::DTFE<Part> d(domain);
        d.create(p.get_particles_ptr(), p.get_npart(), FML::NTasks >= 5 and (p0 == 1 or p0 == FML::NTasks) ? 2.0 : 0.9);
        const auto res = grids<FML::GRID::PencilGrid<3>>(d, p.get_particles_ptr(), p.get_npart(), nmesh);
        double diff = res.size() == ref.size() ? 0.0 : 1e100;
        bool finite = true;
        for (size_t i = 0; diff < 1e100 and i < res.size(); i++) {
            finite = finite and std::isfinite(res[i]);
            diff = std::max(diff, std::abs(res[i] - ref[i]) / std::max(1.0, std::abs(ref[i])));
        }
        check(diff < 1e-6 and finite, "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                                          ": DTFE density, field + gradients, velocity + derivatives on a PencilGrid "
                                          "= on the slabs (max diff " +
                                          num(diff, 2) + ")");
    }
    return FML::TESTS::summary("DTFE on pencils");
}
