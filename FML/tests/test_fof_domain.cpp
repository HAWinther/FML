// Friends-of-friends on a domain (FriendsOfFriends/FoFDomain.h) finds the same groups as FoF.h, for
// particles on slabs and on every pencil layout: the same members (independent of the labels) and the same catalogue
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/FriendsOfFriends/FoFDomain.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <map>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;

namespace {

    struct Part {
        double pos[3];
        long long id;
        size_t fofid{FML::FOF::no_FoF_ID};
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return 3; }
        void set_fofid(size_t f) { fofid = f; }
    };

    // The same particles on every task: clumps (at the periodic edge, at slab boundaries, at the pencil corners
    // where 2 or 4 tasks meet, and inside) and a uniform background
    std::vector<Part> all_particles() {
        std::mt19937 gen(2024);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::vector<std::array<double, 4>> clumps; // x, y, z, radius
        for (double x : {0.0, 1.0 / 6, 1.0 / 4, 1.0 / 3, 0.5, 2.0 / 3, 3.0 / 4})
            for (double y : {0.0, 0.5, 1.0 / 3})
                clumps.push_back({x, y, U(gen), 0.004 + 0.012 * U(gen)});
        for (int i = 0; i < 20; i++)
            clumps.push_back({U(gen), U(gen), U(gen), 0.003 + 0.01 * U(gen)});
        std::vector<Part> part;
        long long id = 0;
        for (auto & c : clumps) {
            std::normal_distribution<double> G(0.0, c[3]);
            const int n = 40 + int(300 * U(gen));
            for (int i = 0; i < n; i++) {
                Part p{};
                for (int d = 0; d < 3; d++) {
                    double x = c[d] + G(gen);
                    x -= std::floor(x);
                    p.pos[d] = x < 1.0 ? x : 0.0;
                }
                p.id = id++;
                part.push_back(p);
            }
        }
        for (int i = 0; i < 20000; i++) {
            Part p{};
            for (int d = 0; d < 3; d++)
                p.pos[d] = U(gen);
            p.id = id++;
            part.push_back(p);
        }
        return part;
    }

    // Gather pairs of doubles from all tasks
    std::vector<double> gather(const std::vector<double> & local) {
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks), displacements(FML::NTasks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displacements[i] = displacements[i - 1] + counts[i - 1];
        std::vector<double> all(displacements.back() + counts.back());
        MPI_Allgatherv(local.data(),
                       n,
                       MPI_DOUBLE,
                       all.data(),
                       counts.data(),
                       displacements.data(),
                       MPI_DOUBLE,
                       FML::communicator);
        return all;
    }

    struct Result {
        std::map<long long, long long> label;     // particle id -> smallest particle id of its group (or -1)
        std::vector<std::array<double, 4>> halos; // np, center (sorted)
        size_t ngroups{0};
    };

    Result result_of(FML::PARTICLE::MPIParticles<Part> & particles, std::vector<FML::FOF::FoFHalo<Part, 3>> & halos) {
        Result r;
        std::vector<double> local;
        for (auto & p : particles) {
            local.push_back(double(p.id));
            local.push_back(p.fofid == FML::FOF::no_FoF_ID ? -1.0 : double(p.fofid));
        }
        const auto all = gather(local);
        std::map<long long, long long> smallest;
        for (size_t i = 0; i < all.size(); i += 2)
            if (all[i + 1] >= 0) {
                const auto g = (long long)all[i + 1];
                const auto id = (long long)all[i];
                smallest[g] = smallest.count(g) ? std::min(smallest[g], id) : id;
            }
        for (size_t i = 0; i < all.size(); i += 2)
            r.label[(long long)all[i]] = all[i + 1] >= 0 ? smallest[(long long)all[i + 1]] : -1;
        r.ngroups = smallest.size();
        std::vector<double> h;
        for (auto & halo : halos) {
            h.push_back(double(halo.np));
            for (int d = 0; d < 3; d++)
                h.push_back(halo.pos[d]);
        }
        const auto allh = gather(h);
        for (size_t i = 0; i < allh.size(); i += 4)
            r.halos.push_back({allh[i], allh[i + 1], allh[i + 2], allh[i + 3]});
        std::sort(r.halos.begin(), r.halos.end());
        return r;
    }

    // The same members, and the same catalogue (centers to round-off, periodic)
    std::string compare(const Result & a, const Result & b) {
        if (a.label != b.label)
            return "different members";
        if (a.halos.size() != b.halos.size())
            return "different number of halos (" + std::to_string(a.halos.size()) + " vs " +
                   std::to_string(b.halos.size()) + ")";
        for (size_t i = 0; i < a.halos.size(); i++) {
            if (a.halos[i][0] != b.halos[i][0])
                return "different halo sizes";
            for (int d = 1; d < 4; d++) {
                double dx = std::abs(a.halos[i][d] - b.halos[i][d]);
                if (std::min(dx, 1.0 - dx) > 1e-10)
                    return "different halo centers";
            }
        }
        return "";
    }

} // namespace

int main() {
    const auto all = all_particles();
    std::vector<Part> mine;
    for (size_t i = FML::ThisTask; i < all.size(); i += FML::NTasks)
        mine.push_back(all[i]);
    const double linking_length = 0.2, buffer = 0.08;
    const int nmin = 20;
    FML::FOF::FoF_verbose = false;

    for (bool periodic : {true, false}) {
        // The reference: FoF.h with the particles on the global slabs
        FML::PARTICLE::MPIParticles<Part> slabs;
        slabs.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
        std::vector<FML::FOF::FoFHalo<Part, 3>> halos;
        FML::FOF::FriendsOfFriends<Part, 3>(slabs.get_particles_ptr(),
                                            slabs.get_npart(),
                                            linking_length,
                                            nmin,
                                            periodic,
                                            buffer,
                                            halos);
        const auto ref = result_of(slabs, halos);
        check(ref.halos.size() > 30 and ref.ngroups == ref.halos.size(),
              "the reference finds the groups (" + std::to_string(ref.halos.size()) + " halos" +
                  (periodic ? "" : ", not periodic") + ")");

        // The domain version on the slabs and on every pencil layout
        std::vector<FML::Domain> domains{FML::domain};
        for (int p0 = 1; p0 <= FML::NTasks; p0++)
            if (FML::NTasks % p0 == 0)
                domains.push_back(FML::Domain::uniform_pencil(p0, FML::NTasks / p0));
        for (auto & domain : domains) {
            FML::PARTICLE::MPIParticles<Part> particles(domain);
            particles.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);
            std::vector<FML::FOF::FoFHalo<Part, 3>> h;
            FML::FOF::FriendsOfFriends<Part, 3>(particles.get_domain(),
                                                particles.get_particles_ptr(),
                                                particles.get_npart(),
                                                linking_length,
                                                nmin,
                                                periodic,
                                                buffer,
                                                h);
            const auto r = result_of(particles, h);
            std::string name = domain.is_slab() ? "Slab" :
                                                  "Pencil " + std::to_string(domain.pencil_px()) + "x" +
                                                      std::to_string(domain.pencil_py());
            const auto why = compare(r, ref);
            check(why.empty(),
                  name + (periodic ? "" : " (not periodic)") + ": the same groups as FoF.h on slabs" +
                      (why.empty() ? "" : " (" + why + ")"));
        }
    }
    return FML::TESTS::summary("FoF on a domain");
}
