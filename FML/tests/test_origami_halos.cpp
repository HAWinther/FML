// ORIGAMIHalos (CosmicWeb/ORIGAMIHalos.h) on FML's own tessellation (no CGAL needed):
//   - halos are found, every halo particle has the ORIGAMI halo morphology and the catalogue counts them all
//   - the same halos (catalogue and membership of every particle) whether the particles are on the slabs or spread
//     over the tasks in any other way
//   - with USE_CGAL: the same halos as with CGAL's MPIPeriodicDelaunay
// Particles on a lattice displaced by plane waves strong enough to shell cross (as in CosmicWeb/example)
#include <FML/CosmicWeb/ORIGAMI.h>
#include <FML/CosmicWeb/ORIGAMIHalos.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;

namespace {

    const int Nlattice = 32;

    struct Part {
        double pos[3];
        long long int id;
        int morphology{0};
        constexpr int get_ndim() const { return 3; }
        double * get_pos() { return pos; }
        long long int get_id() const { return id; }
        void set_id(long long int i) { id = i; }
    };

    std::vector<Part> lattice_particles() {
        const double twopi = 2.0 * M_PI;
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const int nwaves = 12;
        std::vector<std::array<double, 5>> waves;
        for (int w = 0; w < nwaves; w++) {
            std::array<int, 3> n{int(uniform(rng) * 5) - 2, int(uniform(rng) * 5) - 2, int(uniform(rng) * 5) - 2};
            if (n[0] == 0 and n[1] == 0 and n[2] == 0)
                n[0] = 1;
            waves.push_back({double(n[0]), double(n[1]), double(n[2]), twopi * uniform(rng), 2.0 + 4.0 * uniform(rng)});
        }
        std::vector<Part> part;
        for (int ix = 0; ix < Nlattice; ix++)
            for (int iy = 0; iy < Nlattice; iy++)
                for (int iz = 0; iz < Nlattice; iz++) {
                    Part p;
                    const double q[3] = {double(ix) / Nlattice, double(iy) / Nlattice, double(iz) / Nlattice};
                    for (int d = 0; d < 3; d++)
                        p.pos[d] = q[d];
                    for (auto & w : waves) {
                        const double k2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
                        const double phase = twopi * (w[0] * q[0] + w[1] * q[1] + w[2] * q[2]) + w[3];
                        for (int d = 0; d < 3; d++)
                            p.pos[d] += w[4] * w[d] / (twopi * k2) * std::sin(phase) / std::sqrt(double(nwaves));
                    }
                    for (int d = 0; d < 3; d++) {
                        p.pos[d] -= std::floor(p.pos[d]);
                        if (p.pos[d] >= 1.0)
                            p.pos[d] = 0.0;
                    }
                    p.id = ((long long int)(ix)*Nlattice + iy) * Nlattice + iz;
                    part.push_back(p);
                }
        return part;
    }

    struct Membership {
        long long int id;
        long long int npart_of_halo; // 0 if not in a halo
        int morphology;
    };

    struct Result {
        std::vector<FML::COSMICWEB::OrigamiHalo> halos;
        std::map<long long int, Membership> members; // particle id -> membership (all particles)
    };

    Result run(std::vector<Part> & part, FML::COSMICWEB::OrigamiTessellation tessellation) {
        Result r;
        std::vector<int> morphology;
        for (auto & p : part)
            morphology.push_back(p.morphology);
        FML::COSMICWEB::OrigamiHaloOptions options;
        options.core_density_threshold = 20.0;
        options.tessellation = tessellation;
        std::vector<long long int> halo_id;
        FML::COSMICWEB::ORIGAMIHalos(part.data(), part.size(), Nlattice, morphology, halo_id, r.halos, options);
        std::vector<Membership> local;
        for (size_t i = 0; i < part.size(); i++)
            local.push_back({part[i].id, halo_id[i] >= 0 ? r.halos[halo_id[i]].npart : 0, part[i].morphology});
        for (auto & m : FML::COMMUNICATION::gather_records_checked(local, true))
            r.members[m.id] = m;
        return r;
    }

    std::string compare(const Result & a, const Result & b, double tol) {
        if (a.halos.size() != b.halos.size())
            return "different number of halos (" + std::to_string(a.halos.size()) + " vs " +
                   std::to_string(b.halos.size()) + ")";
        for (size_t h = 0; h < a.halos.size(); h++) {
            const auto &x = a.halos[h], &y = b.halos[h];
            if (x.npart != y.npart or x.ncore != y.ncore or std::abs(x.mass - y.mass) > tol * y.mass or
                std::abs(x.density_max - y.density_max) > tol * y.density_max)
                return "different halo " + std::to_string(h);
            for (int d = 0; d < 3; d++) {
                double dx = x.pos[d] - y.pos[d], dy = x.pos_densest[d] - y.pos_densest[d];
                dx -= std::round(dx);
                dy -= std::round(dy);
                if (std::abs(dx) > 1e-10 or std::abs(dy) > 1e-10)
                    return "different position of halo " + std::to_string(h);
            }
        }
        if (a.members.size() != b.members.size())
            return "different number of particles";
        for (auto & [id, m] : a.members) {
            auto it = b.members.find(id);
            if (it == b.members.end() or it->second.npart_of_halo != m.npart_of_halo)
                return "particle " + std::to_string(id) + " is in a different halo";
        }
        return "";
    }

} // namespace

int main() {
    using FML::COSMICWEB::OrigamiTessellation;
    const auto all = lattice_particles();

    // The ORIGAMI morphology on the slabs (it needs the particles in the local domain)
    std::vector<Part> mine;
    for (size_t i = FML::ThisTask; i < all.size(); i += FML::NTasks)
        mine.push_back(all[i]);
    FML::PARTICLE::MPIParticles<Part> slabs;
    slabs.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
    std::vector<int> morphology;
    FML::COSMICWEB::ORIGAMI(slabs.get_particles_ptr(), slabs.get_npart(), Nlattice, morphology);
    std::vector<Part> on_slabs(slabs.get_particles_ptr(), slabs.get_particles_ptr() + slabs.get_npart());
    for (size_t i = 0; i < on_slabs.size(); i++)
        on_slabs[i].morphology = morphology[i];

    // The same particles (with their morphology) round robin over the tasks
    const auto every = FML::COMMUNICATION::gather_records_checked(on_slabs, true);
    std::vector<Part> robin;
    for (size_t i = FML::ThisTask; i < every.size(); i += FML::NTasks)
        robin.push_back(every[i]);

    const Result ref = run(on_slabs, OrigamiTessellation::Native);
    long long int nin = 0, nwithcore = 0, nbad = 0, ninhalo = 0;
    for (auto & h : ref.halos) {
        nin += h.npart;
        nwithcore += h.ncore > 0;
        nbad += h.npart < 20;
    }
    for (auto & [id, m] : ref.members)
        if (m.npart_of_halo > 0) {
            ninhalo++;
            nbad += m.morphology != FML::COSMICWEB::ORIGAMI_HALO;
        }
    check(ref.halos.size() > 20 and nwithcore > 20 and nbad == 0 and nin == ninhalo,
          "native: " + std::to_string(ref.halos.size()) + " halos (" + std::to_string(nwithcore) +
              " with a core) of halo particles with >= 20 particles, holding " + std::to_string(nin) + " particles");

    const Result r = run(robin, OrigamiTessellation::Native);
    const auto why = compare(r, ref, 1e-12);
    check(why.empty(),
          "native: the same halos and members for particles on any task" + (why.empty() ? "" : " (" + why + ")"));

#ifdef USE_CGAL
    std::vector<Part> copy = on_slabs;
    const Result c = run(copy, OrigamiTessellation::CGAL);
    // CGAL uses the exact positions, the native tessellation rounds them to 2^-48 of the box
    const auto why_c = compare(ref, c, 1e-8);
    check(why_c.empty(), "native and CGAL: the same halos and members" + (why_c.empty() ? "" : " (" + why_c + ")"));
#endif
    return FML::TESTS::summary("ORIGAMIHalos on the native tessellation");
}
