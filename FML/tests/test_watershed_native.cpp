// The watershed and ZOBOV on FML's own tessellation (Triangulation/VoronoiTessellation.h, no CGAL):
//   - the Voronoi volumes fill the box and every particle has neighbours
//   - the basins, links and ZOBOV voids are the same whether the particles are on the slabs (MPIParticles) or
//     spread over the tasks in any other way (the native tessellation does not need a domain)
//   - with USE_CGAL: the same volumes, basins (number of particles, volume, leak quantity, circumcenter), links and
//     ZOBOV voids as with CGAL's MPIPeriodicDelaunay (the watershed itself is the same code, WatershedOnGraph)
// 3D by default; TEST_NDIM=2 (with CGAL: also CGAL_NDIM=2) for 2D (no ZOBOV)
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/VoronoiTessellation.h>
#ifdef USE_CGAL
#include <FML/Triangulation/PeriodicDelaunay.h>
#endif
#include <FML/VoidFinders/ZobovVoidFinder.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

#ifndef TEST_NDIM
#define TEST_NDIM 3
#endif

using FML::TESTS::check;
using namespace FML::TRIANGULATION;
constexpr int NDIM = TEST_NDIM;
#ifdef USE_CGAL
static_assert(CGAL_NDIM == NDIM, "Compile with CGAL_NDIM = TEST_NDIM");
#endif

namespace {

    struct Part {
        double pos[NDIM];
        long long id;
        double volume{0.0};
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return NDIM; }
        double get_volume() const { return volume; }
        void set_volume(double v) { volume = v; }
    };
    using Basin = WatershedBasin<Part, NDIM>;

    // The same particles on every task: uniform and clumps (also at the slab boundaries and box corners)
    std::vector<Part> all_particles() {
        std::mt19937 gen(77);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<Part> part;
        long long id = 0;
        const int nuniform = NDIM == 3 ? 12000 : 6000;
        for (int i = 0; i < nuniform; i++) {
            Part p{};
            for (int d = 0; d < NDIM; d++)
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
                    if (NDIM == 3)
                        p.pos[NDIM - 1] = 0.5 + G(gen);
                    for (int d = 0; d < NDIM; d++) {
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

    struct IdValue {
        long long id;
        double value;
    };

    template <class R>
    std::vector<R> gather(const std::vector<R> & local) {
        return FML::COMMUNICATION::gather_records_checked(local, true);
    }

    struct BasinSummary {
        double n, volume, leak;
        std::array<double, NDIM> circ;
        bool operator<(const BasinSummary & o) const {
            if (n != o.n)
                return n < o.n;
            if (volume != o.volume)
                return volume < o.volume;
            return leak < o.leak;
        }
    };

    struct Result {
        std::map<long long, double> volume; // particle id -> Voronoi volume
        std::vector<BasinSummary> basins;   // on task 0, sorted
        std::vector<std::array<double, 3>> links;
        size_t nlonely{0}; // particles without neighbours
    };

    // Basins and links in a form that does not depend on the basin numbering: links as (n_a, n_b, quantity) with
    // the basin sizes (enough to identify them here)
    void summarize(const std::vector<Basin> & basins, const std::vector<WatershedLink> & links, Result & r) {
        for (auto & b : basins) {
            BasinSummary s{double(b.ningroup), b.volume, b.quantity_leak, {}};
            for (int d = 0; d < NDIM; d++)
                s.circ[d] = b.pos_circ[d];
            r.basins.push_back(s);
        }
        for (auto & l : links) {
            double na = basins[l.a].ningroup, nb = basins[l.b].ningroup;
            r.links.push_back({std::min(na, nb), std::max(na, nb), l.quantity});
        }
        std::sort(r.basins.begin(), r.basins.end());
        std::sort(r.links.begin(), r.links.end());
    }

    Result native(std::vector<Part> & part) {
        Result r;
        PeriodicVoronoiTessellation<NDIM> D;
        D.create(part.data(), part.size());
        std::vector<double> volumes;
        D.VoronoiVolume(volumes);
        std::vector<IdValue> local;
        for (size_t i = 0; i < part.size(); i++) {
            local.push_back({part[i].id, volumes[i]});
            size_t n = 0;
            D.for_each_neighbor(i, [&](size_t) { n++; });
            r.nlonely += n == 0;
        }
        for (auto & [id, v] : gather(local))
            r.volume[id] = v;
        std::vector<double> density = watershed_density_quantity(part.data(), volumes, false);
        std::vector<Basin> basins;
        std::vector<WatershedLink> links;
        WatershedGeneral(D, part.data(), part.size(), density, basins, &links);
        summarize(basins, links, r);
        return r;
    }

#ifdef USE_CGAL
    Result cgal(FML::PARTICLE::MPIParticles<Part> & mp) {
        Result r;
        MPIPeriodicDelaunay<Part, VertexDataWatershed> D;
        const double buffer_fraction = FML::NTasks >= 4 ? 0.9 : 0.5;
        D.create(mp.get_particles_ptr(), mp.get_npart(), buffer_fraction, 0.3, [](VertexDataWatershed * v, Part * p) {
            v->part_ptr = p;
        });
        std::vector<double> volumes;
        D.VoronoiVolume(volumes);
        std::vector<IdValue> local;
        for (size_t i = 0; i < mp.get_npart(); i++)
            local.push_back({mp.get_particles_ptr()[i].id, volumes[i]});
        for (auto & [id, v] : gather(local))
            r.volume[id] = v;
        std::vector<double> density = watershed_density_quantity(mp.get_particles_ptr(), volumes, false);
        std::vector<Basin> basins;
        std::vector<WatershedLink> links;
        WatershedGeneral(D, mp.get_particles_ptr(), mp.get_npart(), density, basins, &links);
        summarize(basins, links, r);
        return r;
    }
#endif

    // "" if the same (volumes to rel. tolerance tol, the rest to tol as well; basin sizes exactly)
    std::string compare(const Result & a, const Result & b, double tol) {
        if (a.volume.size() != b.volume.size())
            return "different number of particles";
        double dv = 0.0;
        for (auto & [id, v] : a.volume) {
            auto it = b.volume.find(id);
            if (it == b.volume.end())
                return "a particle is missing";
            dv = std::max(dv, std::abs(v - it->second) / it->second);
        }
        if (dv > tol)
            return "different volumes (max rel. diff " + FML::TESTS::num(dv, 3) + ")";
        if (FML::ThisTask != 0)
            return "";
        if (a.basins.size() != b.basins.size())
            return "different number of basins (" + std::to_string(a.basins.size()) + " vs " +
                   std::to_string(b.basins.size()) + ")";
        for (size_t i = 0; i < a.basins.size(); i++) {
            const auto &x = a.basins[i], &y = b.basins[i];
            if (x.n != y.n or std::abs(x.volume - y.volume) > tol * y.volume or
                std::abs(x.leak - y.leak) > tol * std::abs(y.leak))
                return "different basins";
            for (int d = 0; d < NDIM; d++) {
                double dx = x.circ[d] - y.circ[d];
                dx -= std::round(dx);
                if (std::abs(dx) > 1e-6)
                    return "different basin circumcenters";
            }
        }
        if (a.links.size() != b.links.size())
            return "different number of links (" + std::to_string(a.links.size()) + " vs " +
                   std::to_string(b.links.size()) + ")";
        for (size_t i = 0; i < a.links.size(); i++)
            if (a.links[i][0] != b.links[i][0] or a.links[i][1] != b.links[i][1] or
                std::abs(a.links[i][2] - b.links[i][2]) > tol * b.links[i][2])
                return "different links";
        return "";
    }

    std::string compare_voids(const std::vector<FML::VOIDFINDER::Void> & a,
                              const std::vector<FML::VOIDFINDER::Void> & b,
                              double tol) {
        if (FML::ThisTask != 0)
            return "";
        if (a.size() != b.size())
            return "different number of voids (" + std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")";
        auto key = [](const FML::VOIDFINDER::Void & v) { return std::make_pair(v.nmembers, v.volume); };
        auto sa = a, sb = b;
        auto less = [&](const auto & x, const auto & y) { return key(x) < key(y); };
        std::sort(sa.begin(), sa.end(), less);
        std::sort(sb.begin(), sb.end(), less);
        for (size_t i = 0; i < sa.size(); i++) {
            if (sa[i].nmembers != sb[i].nmembers or std::abs(sa[i].volume - sb[i].volume) > tol * sb[i].volume or
                std::abs(sa[i].radius - sb[i].radius) > tol * sb[i].radius or
                std::abs(sa[i].density_min - sb[i].density_min) > tol * sb[i].density_min)
                return "different voids";
            for (int d = 0; d < 3; d++) {
                double dx = sa[i].pos[d] - sb[i].pos[d];
                dx -= std::round(dx);
                if (std::abs(dx) > 1e-6)
                    return "different void centres";
            }
        }
        return "";
    }

} // namespace

int main() {
    const auto all = all_particles();
    const size_t ntotal = all.size();

    // The particles on the slabs, and round robin over the tasks (not on any domain)
    std::vector<Part> robin;
    for (size_t i = FML::ThisTask; i < ntotal; i += FML::NTasks)
        robin.push_back(all[i]);
    FML::PARTICLE::MPIParticles<Part> slabs;
    slabs.create(robin, 3 * robin.size() + 1000, FML::PARTICLE::Input::Distributed);
    std::vector<Part> on_slabs(slabs.get_particles_ptr(), slabs.get_particles_ptr() + slabs.get_npart());

    const Result ref = native(on_slabs);
    double total = 0.0;
    for (auto & [id, v] : ref.volume)
        total += v;
    long long nlonely = (long long)ref.nlonely;
    FML::SumOverTasks(&nlonely);
    double nbasin = 0.0;
    for (auto & b : ref.basins)
        nbasin += b.n;
    FML::SumOverTasks(&nbasin); // only task 0 has basins
    check(ref.volume.size() == ntotal and std::abs(total - 1.0) < 1e-10 and nlonely == 0 and nbasin == double(ntotal),
          "native on the slabs: the volumes fill the box, every particle has neighbours, the " +
              std::to_string(ref.basins.size()) + " basins hold every particle");

    const Result r = native(robin);
    const auto why = compare(r, ref, 1e-12);
    check(why.empty(),
          "native: the same volumes, basins and links for particles on any task" +
              (why.empty() ? "" : " (" + why + ")"));

#ifdef USE_CGAL
    {
        const Result c = cgal(slabs);
        // CGAL uses the exact positions, the native tessellation rounds them to 2^-48 of the box
        const auto why_c = compare(ref, c, 1e-8);
        check(why_c.empty(),
              "native and CGAL: the same volumes, basins, circumcenters and links" +
                  (why_c.empty() ? "" : " (" + why_c + ")"));
    }
#endif

    if constexpr (NDIM == 3) {
        FML::VOIDFINDER::ZobovOptions o;
        o.tessellation = FML::VOIDFINDER::ZobovTessellation::Native;
        o.merging = FML::VOIDFINDER::ZoneMerging::Full;
        const auto v_slabs = FML::VOIDFINDER::ZobovVoidFinder(on_slabs.data(), on_slabs.size(), o);
        const auto v_robin = FML::VOIDFINDER::ZobovVoidFinder(robin.data(), robin.size(), o);
        const auto why_v = compare_voids(v_robin, v_slabs, 1e-12);
        check(why_v.empty() and (FML::ThisTask != 0 or v_slabs.size() > 10),
              "ZOBOV (native): " + std::to_string(v_slabs.size()) + " voids, the same for particles on any task" +
                  (why_v.empty() ? "" : " (" + why_v + ")"));
#ifdef USE_CGAL
        o.tessellation = FML::VOIDFINDER::ZobovTessellation::CGAL;
        const auto v_cgal = FML::VOIDFINDER::ZobovVoidFinder(on_slabs.data(), on_slabs.size(), o);
        const auto why_c = compare_voids(v_slabs, v_cgal, 1e-8);
        check(why_c.empty(),
              "ZOBOV: the same voids with the native and the CGAL tessellation" +
                  (why_c.empty() ? "" : " (" + why_c + ")"));
#endif
    }
    return FML::TESTS::summary("Watershed on the native tessellation (" + std::to_string(NDIM) + "D)");
}
