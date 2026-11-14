// The periodic Delaunay tessellation on a domain (MPIPeriodicDelaunay(domain), Global/DomainBuffer.h) gives
// the same tessellation as on the global slabs, for particles on slabs and on every pencil layout: the same Voronoi
// volume of every particle, no unreliable cells, the boundary values, and the same watershed basins
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/PeriodicDelaunay.h>
#include <FML/Triangulation/WatershedBinning.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using namespace FML::TRIANGULATION;

namespace {

    struct Part {
        double pos[CGAL_NDIM];
        long long id;
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return CGAL_NDIM; }
    };
    using Basin = WatershedBasin<Part, CGAL_NDIM>;

    // The same particles on every task: uniform and clumps (also at the task boundaries and corners)
    std::vector<Part> all_particles() {
        std::mt19937 gen(77);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<Part> part;
        long long id = 0;
        const int nuniform = CGAL_NDIM == 3 ? 12000 : 6000;
        for (int i = 0; i < nuniform; i++) {
            Part p{};
            for (int d = 0; d < CGAL_NDIM; d++)
                p.pos[d] = U(gen);
            p.id = id++;
            part.push_back(p);
        }
        for (double x : {0.0, 1.0 / 3, 0.5})
            for (double y : {0.0, 0.5, 1.0 / 3}) {
                std::normal_distribution<double> G(0.0, 0.02);
                for (int i = 0; i < 150; i++) {
                    Part p{};
                    p.pos[0] = x + G(gen);
                    p.pos[1] = y + G(gen);
                    if (CGAL_NDIM == 3)
                        p.pos[CGAL_NDIM - 1] = 0.5 + G(gen);
                    for (int d = 0; d < CGAL_NDIM; d++) {
                        p.pos[d] -= std::floor(p.pos[d]);
                        if (p.pos[d] >= 1.0)
                            p.pos[d] = 0.0;
                    }
                    p.id = id++;
                    part.push_back(p);
                }
            }
        return part;
    }

    std::vector<double> gather(const std::vector<double> & local) {
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks), displacements(FML::NTasks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displacements[i] = displacements[i - 1] + counts[i - 1];
        std::vector<double> all(displacements.back() + counts.back());
        MPI_Allgatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displacements.data(), MPI_DOUBLE,
                       FML::communicator);
        return all;
    }

    struct Result {
        std::map<long long, double> volume;       // particle id -> Voronoi volume
        std::vector<std::array<double, 3>> basins; // number of particles, volume, leak (sorted), on every task
        long long nbad{0};
        bool boundary_values_ok{true};
    };

    Result tessellate(MPIPeriodicDelaunay<Part, VertexDataWatershed> & D, FML::PARTICLE::MPIParticles<Part> & mp) {
        Result r;
        // The buffer must hold the neighbours of the boundary particles: thin slabs need a larger fraction
        const double buffer_fraction = FML::NTasks >= 5 ? 0.9 : 0.5, random_fraction = 0.3;
        D.create(mp.get_particles_ptr(), mp.get_npart(), buffer_fraction, random_fraction,
                 [](VertexDataWatershed * v, Part * p) { v->part_ptr = p; });
        r.nbad = D.num_bad_cells();
        FML::SumOverTasks(&r.nbad);

        // The boundary values: the ids of the copies from their owners
        std::vector<long long> ids;
        for (auto & p : mp)
            ids.push_back(p.id);
        const auto copy_ids = D.communicate_boundary_values(ids);
        const auto & copies = D.get_boundary_particles();
        r.boundary_values_ok = copy_ids.size() == copies.size();
        for (size_t i = 0; r.boundary_values_ok and i < copies.size(); i++)
            r.boundary_values_ok = copy_ids[i] == copies[i].id;
        int bad = r.boundary_values_ok ? 0 : 1;
        FML::SumOverTasks(&bad);
        r.boundary_values_ok = bad == 0;

        // The Voronoi volumes
        std::vector<double> volumes;
        D.VoronoiVolume(volumes);
        std::vector<double> local;
        size_t i = 0;
        for (auto & p : mp) {
            local.push_back(double(p.id));
            local.push_back(volumes[i++]);
        }
        const auto all = gather(local);
        for (size_t k = 0; k < all.size(); k += 2)
            r.volume[(long long)all[k]] = all[k + 1];

        // The watershed basins of the volumes (on task 0)
        std::vector<Basin> basins;
        WatershedGeneral(D, mp.get_particles_ptr(), mp.get_npart(), volumes, basins);
        std::vector<double> b;
        for (auto & basin : basins)
            b.insert(b.end(), {double(basin.ningroup), basin.volume, basin.quantity_leak});
        const auto allb = gather(b);
        for (size_t k = 0; k < allb.size(); k += 3)
            r.basins.push_back({allb[k], allb[k + 1], allb[k + 2]});
        std::sort(r.basins.begin(), r.basins.end());
        return r;
    }

    std::string compare(const Result & a, const Result & b) {
        if (a.volume.size() != b.volume.size())
            return "different number of particles";
        double dv = 0.0;
        for (auto & [id, v] : a.volume) {
            auto it = b.volume.find(id);
            if (it == b.volume.end())
                return "a particle is missing";
            dv = std::max(dv, std::abs(v - it->second) / it->second);
        }
        if (dv > 1e-8)
            return "different volumes (max rel. diff " + FML::TESTS::num(dv, 3) + ")";
        if (a.basins.size() != b.basins.size())
            return "different number of basins (" + std::to_string(a.basins.size()) + " vs " +
                   std::to_string(b.basins.size()) + ")";
        for (size_t i = 0; i < a.basins.size(); i++)
            if (a.basins[i][0] != b.basins[i][0] or std::abs(a.basins[i][1] - b.basins[i][1]) > 1e-8 * b.basins[i][1])
                return "different basins";
        return "";
    }

} // namespace

int main() {
    const auto all = all_particles();
    std::vector<Part> mine;
    for (size_t i = FML::ThisTask; i < all.size(); i += FML::NTasks)
        mine.push_back(all[i]);

    // The reference: the tessellation on the global slabs
    FML::PARTICLE::MPIParticles<Part> slabs;
    slabs.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
    MPIPeriodicDelaunay<Part, VertexDataWatershed> D0;
    const auto ref = tessellate(D0, slabs);
    double total = 0.0;
    for (auto & [id, v] : ref.volume)
        total += v;
    check(std::abs(total - 1.0) < 1e-10 and ref.nbad == 0 and ref.basins.size() > 10,
          "the reference: the volumes fill the box, no unreliable cells, " + std::to_string(ref.basins.size()) +
              " basins");

    std::vector<FML::Domain> domains{FML::domain};
    for (int p0 = 1; p0 <= FML::NTasks; p0++)
        if (FML::NTasks % p0 == 0)
            domains.push_back(FML::Domain::uniform_pencil(p0, FML::NTasks / p0));
    for (auto & domain : domains) {
        FML::PARTICLE::MPIParticles<Part> particles(domain);
        particles.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
        MPIPeriodicDelaunay<Part, VertexDataWatershed> D(particles.get_domain());
        const auto r = tessellate(D, particles);
        const std::string name = domain.is_slab() ? "Slab" :
                                                    "Pencil " + std::to_string(domain.pencil_px()) + "x" +
                                                        std::to_string(domain.pencil_py());
        const auto why = compare(r, ref);
        check(why.empty() and r.nbad == 0 and r.boundary_values_ok,
              name + ": the same volumes and basins, no unreliable cells, the boundary values" +
                  (why.empty() ? "" : " (" + why + ")") + (r.nbad ? " (" + std::to_string(r.nbad) + " bad cells)" : "") +
                  (r.boundary_values_ok ? "" : " (boundary values)"));
    }
    return FML::TESTS::summary("Delaunay on a domain (" + std::to_string(CGAL_NDIM) + "D)");
}
