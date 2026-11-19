// The DTFE (Triangulation/DTFE.h) on FML's own tessellation (DTFETessellation::Native, no CGAL needed):
//   - the density, a field with its gradient and the velocity with its derivatives on the grid are finite, the mean
//     density is 1 and a linear field (and its gradient) is reproduced exactly
//   - the same grids whether the particles are on the slabs or spread over the tasks in any other way, and (MPI) on
//     a PencilGrid as on the slabs
//   - with USE_CGAL: the same grids as with CGAL's MPIPeriodicDelaunay
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <FML/Triangulation/DTFE.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
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

    // Uniform particles and clumps (also on the slab boundaries and box corners), round robin over the tasks
    std::vector<Part> robin_particles(int nuniform) {
        std::mt19937 gen(5);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.03);
        std::vector<Part> v;
        int n = 0;
        auto add = [&](double x, double y, double z) {
            Part p;
            const double q[3] = {x, y, z};
            for (int d = 0; d < 3; d++) {
                p.pos[d] = q[d] - std::floor(q[d]);
                if (p.pos[d] >= 1.0)
                    p.pos[d] = 0.0;
            }
            for (int d = 0; d < 3; d++)
                p.vel[d] = std::sin(2 * M_PI * p.pos[(d + 1) % 3]) + 0.3 * std::cos(2 * M_PI * p.pos[d]);
            if (n++ % FML::NTasks == FML::ThisTask)
                v.push_back(p);
        };
        for (int i = 0; i < nuniform; i++)
            add(U(gen), U(gen), U(gen));
        for (double c : {0.0, 0.5})
            for (int i = 0; i < nuniform / 10; i++)
                add(c + G(gen), 0.5 + G(gen), c + G(gen));
        return v;
    }

    // The value of the field we interpolate at a particle
    double field(const Part & p) { return p.pos[0] * p.pos[1] + std::cos(2 * M_PI * p.pos[2]); }

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
            values[i] = field(p[i]);
        std::vector<G> fields, gradients;
        dtfe.template fields_to_grid<G>(nmesh,
                                        values,
                                        1,
                                        fields,
                                        &gradients,
                                        T::DTFESampling::CellAverage,
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

    // Max difference relative to max(1, |b|); 1e100 if the sizes differ or a value is not finite
    double max_diff(const std::vector<double> & a, const std::vector<double> & b) {
        if (a.size() != b.size())
            return 1e100;
        double d = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            if (!std::isfinite(a[i]) or !std::isfinite(b[i]))
                return 1e100;
            d = std::max(d, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
        }
        return d;
    }
} // namespace

int main() {
    const int nmesh = FML::NTasks % 3 == 0 ? 12 : 16;
    auto robin = robin_particles(6000);
    FML::PARTICLE::MPIParticles<Part> slabs;
    {
        auto c = robin;
        slabs.create(c, 3 * robin.size() + 1000, FML::PARTICLE::Input::Distributed);
    }

    // On the slabs
    T::DTFE<Part> dtfe(T::DTFETessellation::Native);
    dtfe.create(slabs.get_particles_ptr(), slabs.get_npart());
    const auto ref = grids<FML::GRID::FFTWGrid<3>>(dtfe, slabs.get_particles_ptr(), slabs.get_npart(), nmesh);
    const size_t ncells = size_t(nmesh) * nmesh * nmesh;
    double mean = 0.0;
    bool finite = ref.size() == 18 * ncells;
    for (size_t i = 0; finite and i < ref.size(); i++)
        finite = std::isfinite(ref[i]);
    for (size_t i = 0; finite and i < ncells; i++)
        mean += ref[i] / double(ncells);
    check(finite and std::abs(mean - 1.0) < 0.02,
          "native on the slabs: every grid value is finite, the mean density is " + num(mean, 4));

    // A linear field f = 2 x + 3 y is reproduced exactly (value and gradient) away from x, y = 0 (where the
    // periodic wrap makes it jump)
    {
        std::vector<double> values(slabs.get_npart());
        for (size_t i = 0; i < values.size(); i++)
            values[i] = 2.0 * slabs[i].pos[0] + 3.0 * slabs[i].pos[1];
        std::vector<FML::GRID::FFTWGrid<3>> f, g;
        dtfe.fields_to_grid(nmesh, values, 1, f, &g, T::DTFESampling::GridPoint);
        const auto fv = global_real(f[0]);
        const auto gx = global_real(g[0]), gy = global_real(g[1]), gz = global_real(g[2]);
        double err = 0.0;
        for (int ix = 2; ix < nmesh - 2; ix++)
            for (int iy = 2; iy < nmesh - 2; iy++)
                for (int iz = 0; iz < nmesh; iz++) {
                    const size_t k = (size_t(ix) * nmesh + iy) * nmesh + iz;
                    const double x = double(ix) / nmesh, y = double(iy) / nmesh;
                    err = std::max({err,
                                    std::abs(fv[k] - (2 * x + 3 * y)),
                                    std::abs(gx[k] - 2.0),
                                    std::abs(gy[k] - 3.0),
                                    std::abs(gz[k])});
                }
        check(err < 1e-9, "native: a linear field and its gradient are exact (max error " + num(err, 2) + ")");
    }
    dtfe.free();

    // Round robin (not on any domain)
    {
        T::DTFE<Part> d(T::DTFETessellation::Native);
        d.create(robin.data(), robin.size());
        const auto r = grids<FML::GRID::FFTWGrid<3>>(d, robin.data(), robin.size(), nmesh);
        const double diff = max_diff(r, ref);
        check(diff < 1e-10, "native: the same grids for particles on any task (max diff " + num(diff, 2) + ")");
    }

#ifdef USE_MPI
    // Pencil grids (the particles on the pencil domain, and round robin)
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> proto(nmesh, 0, 0, layout);
        const auto domain = FML::GRID::particle_domain_of(proto);
        FML::PARTICLE::MPIParticles<Part> p(domain);
        {
            auto c = robin;
            p.create(c, 3 * robin.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        T::DTFE<Part> d(domain, T::DTFETessellation::Native);
        d.create(p.get_particles_ptr(), p.get_npart());
        const auto r = grids<FML::GRID::PencilGrid<3>>(d, p.get_particles_ptr(), p.get_npart(), nmesh);
        T::DTFE<Part> d2(domain, T::DTFETessellation::Native);
        d2.create(robin.data(), robin.size());
        const auto r2 = grids<FML::GRID::PencilGrid<3>>(d2, robin.data(), robin.size(), nmesh);
        const double diff = std::max(max_diff(r, ref), max_diff(r2, ref));
        check(diff < 1e-10,
              "native, pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                  ": the same grids as on the slabs (max diff " + num(diff, 2) + ")");
    }

#endif

#ifdef USE_CGAL
    {
        T::DTFE<Part> c(T::DTFETessellation::CGAL);
        c.create(slabs.get_particles_ptr(), slabs.get_npart(), FML::NTasks >= 5 ? 2.0 : 0.9);
        const auto r = grids<FML::GRID::FFTWGrid<3>>(c, slabs.get_particles_ptr(), slabs.get_npart(), nmesh);
        // CGAL uses the exact positions, the native tessellation rounds them to 2^-48 of the box
        const double diff = max_diff(ref, r);
        check(diff < 1e-7, "native and CGAL: the same grids (max diff " + num(diff, 2) + ")");
    }
#endif
    return FML::TESTS::summary("DTFE on the native tessellation");
}
