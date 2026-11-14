// The voxel void finder (density field, smoothing and watershed) and the spherical void finder (candidate
// grid, particle buffers and cell lists) on pencils give the catalogues of the slabs (the zone IDs follow the task
// order, so the catalogues are compared sorted)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/VoidFinders/SphericalVoidFinder.h>
#include <FML/VoidFinders/VoxelVoidFinder.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
namespace V = FML::VOIDFINDER;

namespace {
    struct Part {
        double pos[3];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 3; }
    };

    // Particles in clumps (so there are voids), every task makes all and keeps its share
    std::vector<Part> mine(int npart) {
        std::mt19937 gen(7);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.04);
        std::vector<std::array<double, 3>> centers(40);
        for (auto & c : centers)
            for (auto & x : c)
                x = U(gen);
        std::vector<Part> v;
        for (int i = 0; i < npart; i++) {
            Part p;
            const auto & c = centers[i % centers.size()];
            for (int d = 0; d < 3; d++) {
                double x = i % 4 == 0 ? U(gen) : c[d] + G(gen);
                x -= std::floor(x);
                p.pos[d] = x >= 1.0 ? 0.0 : x;
            }
            if (i % FML::NTasks == FML::ThisTask)
                v.push_back(p);
        }
        return v;
    }

    std::vector<std::vector<double>> rows(std::vector<V::Void> voids) {
        std::vector<std::vector<double>> r;
        for (auto & v : voids) {
            std::vector<double> x{v.pos_min[0], v.pos_min[1], v.pos_min[2], v.pos_barycenter[0], v.pos_barycenter[1],
                                  v.pos_barycenter[2], v.volume, v.radius, v.density_min, v.density_avg,
                                  v.density_ratio, double(v.nmembers), double(v.nzones), double(v.parent >= 0)};
            r.push_back(x);
        }
        std::sort(r.begin(), r.end());
        return r;
    }

    double max_diff(const std::vector<std::vector<double>> & a, const std::vector<std::vector<double>> & b) {
        if (a.size() != b.size())
            return 1e100;
        double d = 0.0;
        for (size_t i = 0; i < a.size(); i++)
            for (size_t j = 0; j < a[i].size(); j++)
                d = std::max(d, std::abs(a[i][j] - b[i][j]) / std::max(1.0, std::abs(a[i][j])));
        return d;
    }
} // namespace

int main() {
    const int npart = 30000;
    auto v = mine(npart);
    std::vector<std::pair<std::string, V::VoxelOptions>> cases;
    for (int smooth = 0; smooth < 2; smooth++)
        for (int clusters = 0; clusters < 2; clusters++) {
            V::VoxelOptions o;
            o.Nmesh = FML::NTasks % 3 == 0 ? 24 : 32;
            o.smoothing_scale = smooth ? -1.0 : 0.0;
            o.find_clusters = clusters;
            if (clusters) {
                o.merging = V::ZoneMerging::Threshold;
                o.link_density_threshold = 3.0;
            }
            cases.push_back({std::string(clusters ? "clusters (merged)" : "voids") +
                                 (smooth ? ", smoothed" : ", not smoothed"),
                             o});
        }

    FML::PARTICLE::MPIParticles<Part> pslab;
    {
        auto c = v;
        pslab.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
    }
    std::vector<std::vector<std::vector<double>>> ref;
    for (auto & c : cases)
        ref.push_back(rows(V::VoxelVoidFinder(pslab.get_particles_ptr(), pslab.get_npart(), c.second)));
    // The spherical void finder (with and without recentering)
    std::vector<V::SphericalOptions> sph(2);
    for (int i = 0; i < 2; i++) {
        sph[i].grid.Nmesh = FML::NTasks % 3 == 0 ? 24 : 32;
        sph[i].recentering_steps = 4 * i;
        sph[i].max_overlap = 0.2 * i;
    }
    std::vector<std::vector<std::vector<double>>> ref_sph;
    for (auto & o : sph)
        ref_sph.push_back(rows(V::SphericalVoidFinder(pslab.get_particles_ptr(), pslab.get_npart(), o)));
    const FML::Domain slab_domain = FML::domain;

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> proto(FML::NTasks % 3 == 0 ? 24 : 32, 0, 0, layout);
        const auto domain = FML::GRID::particle_domain_of(proto);
        FML::PARTICLE::MPIParticles<Part> p(domain);
        {
            auto c = v;
            p.create(c, 3 * v.size() + 1000, FML::PARTICLE::Input::Distributed);
        }
        for (size_t i = 0; i < cases.size(); i++) {
            const auto res = rows(V::VoxelVoidFinder<Part, FML::GRID::PencilGrid>(
                p.get_particles_ptr(), p.get_npart(), cases[i].second, nullptr, domain));
            const double diff = max_diff(res, ref[i]);
            check(diff < 1e-9 and not res.empty(),
                  "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) + ": " + cases[i].first +
                      ": " + std::to_string(res.size()) + " as on slabs (max diff " + num(diff, 2) + ")");
        }
        // The spherical void finder uses the global domain
        FML::set_domain(domain);
        for (size_t i = 0; i < sph.size(); i++) {
            const auto res =
                rows(V::SphericalVoidFinder<Part, FML::GRID::PencilGrid>(p.get_particles_ptr(), p.get_npart(), sph[i]));
            const double diff = max_diff(res, ref_sph[i]);
            check(diff < 1e-12 and not res.empty(),
                  "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                      ": spherical voids (recentering steps " + std::to_string(sph[i].recentering_steps) + "): " +
                      std::to_string(res.size()) + " as on slabs (max diff " + num(diff, 2) + ")");
        }
        FML::set_domain(slab_domain);
    }
    return FML::TESTS::summary("voxel void finder on pencils");
}
