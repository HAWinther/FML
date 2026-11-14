//==========================================================
// Tests of the particle domain (Global/Domain.h, the global FML::domain) and of MPIParticles with
// a domain:
//  - the global domain is a Slab with the ranges xmin_domain, xmax_domain (also after changing them)
//  - owner() and owner_near() of Slab (with empty slabs), Pencil and PeanoHilbert against brute force (the key of the
//    PeanoHilbertDomain module; the rank layout ix * py + iy of PencilDomain)
//  - MPIParticles: the default (global Slab) and an explicit Slab give the same as before (create,
//    create_particle_grid); with every domain type, communicate_particles and create (Input::Distributed,
//    Input::Replicated) put every particle on its owner and keep them all (count and sum of ids), and
//    dump_to_file / load_from_file keep the particles and the domain
// Run with 1, 2 and 4 MPI tasks (OMP_NUM_THREADS=2).
//==========================================================
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <stdexcept>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check;
namespace PH = FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT;

struct Particle3 {
    double pos[3];
    long long id;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
struct Particle2 {
    double pos[2];
    long long id;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 2; }
};

namespace {
    // Random particles on every task (different on each), ids unique over the tasks
    template <class P>
    std::vector<P> random_particles(size_t n, unsigned seed) {
        std::mt19937 rng(seed + 1000u * unsigned(FML::ThisTask));
        std::uniform_real_distribution<double> u(0.0, 1.0);
        std::vector<P> v(n);
        for (size_t i = 0; i < n; i++) {
            for (int d = 0; d < P{}.get_ndim(); d++)
                v[i].pos[d] = u(rng);
            v[i].id = (long long)(FML::ThisTask) * (long long)(n) + (long long)(i);
        }
        // Points on the boundaries
        if (n > 2) {
            for (int d = 0; d < P{}.get_ndim(); d++) {
                v[0].pos[d] = 0.0;
                v[1].pos[d] = 1.0;
            }
        }
        return v;
    }

    // Communicate the particles in the domain: all kept, all owned by this task
    template <class P>
    void check_communication(const FML::Domain & domain, const std::string & label) {
        const size_t n = 2000;
        auto part = random_particles<P>(n, 7);
        long long idsum = 0;
        for (auto & q : part)
            idsum += q.id;
        FML::SumOverTasks(&idsum);

        FML::PARTICLE::MPIParticles<P> mp(domain);
        mp.create(part, 4 * n * size_t(FML::NTasks), FML::PARTICLE::Input::Local);
        mp.communicate_particles();

        bool owned = true;
        long long idsum_after = 0;
        for (size_t i = 0; i < mp.get_npart(); i++) {
            // (with one task communicate_particles returns at once: x = 1.0 is not moved just below 1)
            double pos[3];
            for (int d = 0; d < P{}.get_ndim(); d++)
                pos[d] = std::min(mp[i].get_pos()[d], std::nextafter(1.0, 0.0));
            owned =
                owned and mp.get_domain().owner(pos) == FML::ThisTask and mp.get_domain().contains(FML::ThisTask, pos);
            idsum_after += mp[i].id;
        }
        FML::SumOverTasks(&idsum_after);
        size_t ntot = mp.get_npart();
        FML::SumOverTasks(&ntot);
        check(owned, label + ": every particle is on the task that owns it");
        check(ntot == n * size_t(FML::NTasks) and idsum_after == idsum,
              label + ": no particle lost or duplicated (count and sum of ids)");
        check(mp.get_domain() == domain, label + ": MPIParticles keeps its domain");

        // create with Input::Distributed: the same, sent to the owners while creating
        {
            FML::PARTICLE::MPIParticles<P> md(domain);
            md.create(part, 4 * n * size_t(FML::NTasks), FML::PARTICLE::Input::Distributed);
            bool ok = true;
            long long ids = 0;
            for (size_t i = 0; i < md.get_npart(); i++) {
                double pos[3];
                for (int d = 0; d < P{}.get_ndim(); d++)
                    pos[d] = std::min(md[i].get_pos()[d], std::nextafter(1.0, 0.0));
                ok = ok and md.get_domain().owner(pos) == FML::ThisTask;
                ids += md[i].id;
            }
            FML::SumOverTasks(&ids);
            check(ok and md.get_npart_total() == n * size_t(FML::NTasks) and ids == idsum,
                  label + ": create(Input::Distributed) puts every particle on its owner");
        }
        // create with Input::Replicated: every task has the same particles and keeps those it owns
        {
            auto same = random_particles<P>(n, 7);
            std::mt19937 rng(99);
            std::uniform_real_distribution<double> u(0.0, 1.0);
            for (size_t i = 0; i < n; i++) {
                for (int d = 0; d < P{}.get_ndim(); d++)
                    same[i].pos[d] = u(rng);
                same[i].id = (long long)(i);
            }
            FML::PARTICLE::MPIParticles<P> mr(domain);
            mr.create(same, n, FML::PARTICLE::Input::Replicated);
            bool ok = true;
            for (size_t i = 0; i < mr.get_npart(); i++)
                ok = ok and mr.get_domain().owner(mr[i].get_pos()) == FML::ThisTask;
            check(ok and mr.get_npart_total() == n, label + ": create(Input::Replicated) keeps each particle once");
        }
        // dump_to_file / load_from_file keep the particles and the domain
        {
            const std::string prefix = "test_domain_dump";
            mp.dump_to_file(prefix);
            FML::PARTICLE::MPIParticles<P> ml;
            ml.load_from_file(prefix);
            std::remove((prefix + "." + std::to_string(FML::ThisTask)).c_str());
            bool ok = ml.get_domain() == domain and ml.get_npart() == mp.get_npart() and
                      ml.get_npart_total() == mp.get_npart_total();
            for (size_t i = 0; ok and i < ml.get_npart(); i++)
                ok = ml[i].id == mp[i].id and ml[i].pos[0] == mp[i].pos[0];
            check(ok, label + ": dump_to_file / load_from_file keep the particles and the domain");
        }
    }
} // namespace

int main() {
    const int N = FML::NTasks;

    // The global domain
    {
        check(FML::domain.is_slab() and FML::domain.ntasks() == N, "the global particle domain is a Slab for NTasks");
        const auto d = FML::get_domain();
        check(d.x_min(FML::ThisTask) == FML::xmin_domain and d.x_max(FML::ThisTask) == FML::xmax_domain and
                  d == FML::Domain::uniform_slab(N) and d.valid_slab_partition(),
              "the global Slab has the ranges xmin_domain, xmax_domain (uniform)");
        // Changing xmin_domain, xmax_domain (as some tests do) changes the global Slab's ranges
        const double lo = FML::xmin_domain, hi = FML::xmax_domain;
        std::vector<double> edges(N + 1);
        for (int i = 0; i <= N; i++)
            edges[i] = std::pow(i / double(N), 2);
        FML::xmin_domain = edges[FML::ThisTask];
        FML::xmax_domain = edges[FML::ThisTask + 1];
        const auto d2 = FML::get_domain();
        check(d2.x_min(FML::ThisTask) == edges[FML::ThisTask] and d2.valid_slab_partition(),
              "the global Slab follows xmin_domain, xmax_domain");
        FML::xmin_domain = lo;
        FML::xmax_domain = hi;
        // set_domain with a Slab sets xmin_domain, xmax_domain
        FML::set_domain(FML::Domain::slab(std::vector<double>(edges.begin(), edges.end() - 1),
                                          std::vector<double>(edges.begin() + 1, edges.end())));
        check(FML::xmin_domain == edges[FML::ThisTask] and FML::xmax_domain == edges[FML::ThisTask + 1],
              "set_domain (Slab) sets xmin_domain, xmax_domain");
        FML::set_domain(FML::Domain::uniform_slab(N));
        check(FML::xmin_domain == lo and FML::xmax_domain == hi, "set_domain (uniform Slab) restores them");
    }

    // owner() against brute force
    {
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        // Slab with empty slabs
        const auto slab = FML::Domain::slab({0.0, 0.2, 0.2, 0.7}, {0.2, 0.2, 0.7, 1.0});
        bool ok = slab.valid_slab_partition();
        for (int i = 0; i < 10000; i++) {
            double x[1] = {u(rng)};
            int brute = -1;
            for (int t = 0; t < 4; t++)
                if (x[0] >= slab.x_min(t) and x[0] < slab.x_max(t))
                    brute = t;
            ok = ok and slab.owner(x) == brute and slab.contains(brute, x);
        }
        check(ok, "Slab owner (with an empty slab) = brute force");
        check(not FML::Domain::slab({0.0, 0.5}, {0.4, 1.0}).valid_slab_partition(),
              "a Slab with a gap is not a valid partition");

        // Pencil 3 x 2 with uneven edges: rank ix * py + iy
        const auto pen = FML::Domain::pencil({0.0, 0.25, 0.5, 1.0}, {0.0, 0.6, 1.0});
        ok = pen.ntasks() == 6 and pen.pencil_px() == 3 and pen.pencil_py() == 2;
        for (int i = 0; i < 10000; i++) {
            double x[3] = {u(rng), u(rng), u(rng)};
            const int ix = x[0] < 0.25 ? 0 : (x[0] < 0.5 ? 1 : 2), iy = x[1] < 0.6 ? 0 : 1;
            ok = ok and pen.owner(x) == ix * 2 + iy and pen.contains(ix * 2 + iy, x);
        }
        check(ok, "Pencil owner = brute force (rank ix * py + iy)");

        // PeanoHilbert: the key = that of the module, owner = the range of the key
        for (unsigned bits : {4u, 21u}) {
            const auto ph = FML::Domain::uniform_peano_hilbert(5, bits);
            ok = ph.ntasks() == 5;
            for (int i = 0; i < 10000; i++) {
                double x[3] = {u(rng), u(rng), u(rng)};
                const auto key = PH::key_from_position({x[0], x[1], x[2]}, bits);
                int brute = -1;
                for (int t = 0; t < 5; t++)
                    if (key >= ph.key_boundaries()[t] and key < ph.key_boundaries()[t + 1])
                        brute = t;
                ok = ok and ph.key_of(x) == key and ph.owner(x) == brute and ph.contains(brute, x);
            }
            check(ok, "PeanoHilbert key and owner = the module's (bits = " + std::to_string(bits) + ")");
        }
        // owner_near (the neighbours of a task first) = owner, for every type and task
        ok = true;
        const std::vector<FML::Domain> all{slab,
                                           pen,
                                           FML::Domain::uniform_peano_hilbert(5, 6),
                                           FML::Domain::uniform_slab(1),
                                           FML::Domain::uniform_pencil(1, 1)};
        for (const auto & d : all)
            for (int task = 0; task < d.ntasks(); task++)
                for (int i = 0; i < 2000; i++) {
                    double x[3] = {u(rng), u(rng), u(rng)};
                    ok = ok and d.owner_near(task, x) == d.owner(x);
                }
        check(ok, "owner_near = owner (every type, every task)");
        bool threw = false;
        try {
            FML::Domain::uniform_pencil(2, 2).check_ndim(1);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "Pencil refuses NDIM = 1");
    }

    // MPIParticles with the (default) Slab: the domain of create(xmin, xmax) and create_particle_grid
    {
        const size_t n = 1000;
        auto part = random_particles<Particle3>(n, 11);
        // Every task has the same particles here
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (size_t i = 0; i < n; i++)
            for (int d = 0; d < 3; d++)
                part[i].pos[d] = u(rng);
        FML::PARTICLE::MPIParticles<Particle3> mp;
        mp.create(part.data(), n, 2 * n, FML::PARTICLE::Input::Replicated);
        size_t ntot = mp.get_npart();
        FML::SumOverTasks(&ntot);
        check(ntot == n and mp.get_domain() == FML::Domain::uniform_slab(N),
              "create(xmin, xmax): the global slabs, all particles kept");
        FML::PARTICLE::MPIParticles<Particle3> grid;
        grid.create_particle_grid(8, 2.0);
        check(grid.get_npart_total() == 512 and grid.get_domain() == FML::Domain::uniform_slab(N),
              "create_particle_grid: the global slabs");
        FML::PARTICLE::MPIParticles<Particle3> explicit_slab(FML::Domain::uniform_slab(N));
        auto part2 = random_particles<Particle3>(n, 13);
        explicit_slab.create(part2, 4 * n * size_t(N), FML::PARTICLE::Input::Local);
        check(explicit_slab.get_domain() == FML::Domain::uniform_slab(N),
              "an explicit Slab domain is kept by create(vector, ...)");
    }

    // Communication with every type of domain
    check_communication<Particle3>(FML::Domain::uniform_slab(N), "Slab (3D)");
    {
        std::vector<double> lo(N), hi(N);
        for (int i = 0; i < N; i++) {
            lo[i] = std::pow(i / double(N), 2);
            hi[i] = std::pow((i + 1) / double(N), 2);
        }
        check_communication<Particle3>(FML::Domain::slab(lo, hi), "uneven Slab (3D)");
    }
    check_communication<Particle3>(FML::pencil_domain(), "Pencil (3D, MPI_Dims_create)");
    check_communication<Particle2>(FML::pencil_domain(N, 1), "Pencil (2D, N x 1)");
    check_communication<Particle2>(FML::pencil_domain(1, N), "Pencil (2D, 1 x N)");
    check_communication<Particle3>(FML::Domain::uniform_peano_hilbert(N), "PeanoHilbert (3D, 21 bits)");
    check_communication<Particle3>(FML::Domain::uniform_peano_hilbert(N, 2), "PeanoHilbert (3D, 2 bits)");
    {
        // PeanoHilbert from a load-balanced decomposition of the particles
        auto part = random_particles<Particle3>(3000, 17);
        std::vector<PH::Key> keys;
        // (the module requires [0,1): the boundary particle at 1.0 is mapped just below)
        auto below1 = [](double x) { return std::min(x, std::nextafter(1.0, 0.0)); };
        for (auto & q : part)
            keys.push_back(PH::key_from_position({below1(q.pos[0]), below1(q.pos[1]), below1(q.pos[2])}, 10));
#ifdef USE_MPI
        const auto dec = PH::PeanoHilbertDomain::build(keys, 10, 8);
        check_communication<Particle3>(FML::Domain::peano_hilbert(dec), "PeanoHilbert (load-balanced)");
#endif
    }
    // PeanoHilbert: create_particle_grid throws (on every task); the lattice made on slabs and moved there with
    // set_domain is complete and owned
    {
        const auto ph = FML::Domain::uniform_peano_hilbert(N);
        FML::PARTICLE::MPIParticles<Particle3> lattice(ph);
        bool threw = false;
        try {
            lattice.create_particle_grid(8, 2.0);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        int all = threw ? 1 : 0;
        FML::MinOverTasks(&all);
        check(all == 1, "create_particle_grid on a PeanoHilbert domain throws on every task");
        FML::PARTICLE::MPIParticles<Particle3> moved(FML::Domain::uniform_slab(N));
        moved.create_particle_grid(8, 2.0 * N);
        moved.set_domain(ph);
        bool owned = true;
        double sum = 0;
        for (auto & q : moved) {
            owned = owned and ph.owner(q.pos) == FML::ThisTask;
            sum += q.pos[0] + 2 * q.pos[1] + 3 * q.pos[2];
        }
        int ok = owned ? 1 : 0;
        FML::MinOverTasks(&ok);
        size_t ntot = moved.get_npart();
        FML::SumOverTasks(&ntot);
        FML::SumOverTasks(&sum);
        // (the sum over the lattice of x + 2y + 3z = 6 * 512 * mean(i/8) = 6 * 512 * 7/16)
        check(ok == 1 and ntot == 512 and std::abs(sum - 6 * 512 * 7.0 / 16) < 1e-9 and moved.get_domain() == ph,
              "a slab lattice moved to PeanoHilbert with set_domain: complete and owned");
    }
    return FML::TESTS::summary("domain");
}
