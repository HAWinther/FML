// The particle buffers on pencils. With FML::domain the pencils (every layout of the tasks):
//   - ShortRangeGravity::exchange_domain_ghosts: the short-range force of the particles with the ghosts is the force
//     from all the particles (also with a cutoff wider than a pencil), no ghost twice
//   - COMMUNICATION::append_periodic_domain_ghosts: the ghosts are exactly the records within the buffer of the
//     pencil (periodic, the corners included)
//   - FOF::FriendsOfFriends (the particles on the pencils) and SphericalOverdensity around the FoF centers: the same
//     halos as on the slabs
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <FML/Global/Global.h>
#include <FML/Global/PeriodicSlabExchange.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ShortRangeGravity/SlabGhosts.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
namespace SR = FML::SHORT_RANGE_GRAVITY;

namespace {

    // The same clustered particles on every task
    std::vector<std::array<double, 3>> positions(int n) {
        std::mt19937 gen(5);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.01);
        std::vector<std::array<double, 3>> x(n);
        std::array<std::array<double, 3>, 6> centres{
            {{0.1, 0.2, 0.3}, {0.5, 0.5, 0.5}, {0.98, 0.02, 0.7}, {0.25, 0.75, 0.1}, {0.5, 0.0, 0.9}, {0.75, 0.5, 0.2}}};
        for (int i = 0; i < n; i++)
            for (int d = 0; d < 3; d++) {
                double v = i % 3 == 0 ? U(gen) : centres[i % 6][d] + G(gen);
                v -= std::floor(v);
                x[i][d] = v >= 1.0 ? 0.0 : v;
            }
        return x;
    }

    struct FoFPart {
        double pos[3];
        double vel[3];
        double mass{1.0};
        double * get_pos() { return pos; }
        double * get_vel() { return vel; }
        double get_mass() const { return mass; }
        constexpr int get_ndim() const { return 3; }
    };

    void short_range_ghosts(const std::string & name) {
        const auto x = positions(1500);
        std::vector<SR::Particle> all(x.size()), local;
        for (size_t i = 0; i < x.size(); i++) {
            all[i] = {{x[i][0], x[i][1], x[i][2]}, 1.0, i};
            if (FML::domain.owner(x[i].data()) == FML::ThisTask)
                local.push_back(all[i]);
        }
        double worst = 0.0;
        bool unique = true;
        for (double rs : {0.02, 0.045}) {
            SR::Options o;
            o.split_scale = rs;
            o.cutoff_tolerance = 1e-8;
            SR::Kernel k(o);
            auto sources = SR::exchange_domain_ghosts(local, FML::domain, k);
            std::set<std::uint64_t> ids;
            for (auto & p : sources.particles)
                ids.insert(p.id);
            unique = unique and ids.size() == sources.particles.size();
            const auto actual = SR::CellList(sources.particles, k).compute(local);
            const auto expected = SR::CellList(all, k).compute(local);
            for (size_t i = 0; i < local.size(); i++)
                for (int d = 0; d < 3; d++)
                    worst = std::max(worst, std::abs(actual.acceleration[i][d] - expected.acceleration[i][d]) /
                                                (1e-10 + std::abs(expected.acceleration[i][d])));
        }
        int bad = (worst < 1e-10 and unique) ? 0 : 1;
        FML::SumOverTasks(&bad);
        FML::MaxOverTasks(&worst);
        check(bad == 0, name + ": the short-range force with the ghosts is the force from all particles (rel. diff " +
                            num(worst, 2) + ", no ghost twice)");
    }

    void record_ghosts(const std::string & name) {
        struct Record {
            double x[3];
            long long id;
        };
        const auto x = positions(3000);
        std::vector<Record> mine;
        for (size_t i = 0; i < x.size(); i++)
            if (FML::domain.owner(x[i].data()) == FML::ThisTask)
                mine.push_back({{x[i][0], x[i][1], x[i][2]}, (long long)i});
        const double buffer = 0.07;
        const size_t nown = mine.size();
        FML::COMMUNICATION::append_periodic_domain_ghosts(
            mine, buffer, [](const Record & r) { return std::array<double, 2>{r.x[0], r.x[1]}; });
        std::set<long long> got;
        for (size_t i = nown; i < mine.size(); i++)
            got.insert(mine[i].id);
        // The records within the buffer of the pencil (not ours)
        const auto & pencils = FML::domain.get<FML::PencilDomain>();
        const auto c = pencils.coords_of(FML::ThisTask);
        auto dist = [](double v, double lo, double hi) {
            double best = 1.0;
            for (int image = -1; image <= 1; image++)
                best = std::min(best, std::max({lo - (v + image), v + image - hi, 0.0}));
            return best;
        };
        std::set<long long> expected;
        for (size_t i = 0; i < x.size(); i++) {
            if (FML::domain.owner(x[i].data()) == FML::ThisTask)
                continue;
            const double dx = dist(x[i][0], pencils.x_edges()[c[0]], pencils.x_edges()[c[0] + 1]);
            const double dy = dist(x[i][1], pencils.y_edges()[c[1]], pencils.y_edges()[c[1] + 1]);
            if (std::hypot(dx, dy) <= buffer)
                expected.insert(i);
        }
        int bad = (got == expected and got.size() == mine.size() - nown) ? 0 : 1;
        FML::SumOverTasks(&bad);
        check(bad == 0, name + ": append_periodic_domain_ghosts gives the records within the buffer of the pencil");
    }

    // The FoF halos (np, mass, centre) and the SO masses on every task, sorted
    std::vector<std::array<double, 6>> halos() {
        const auto x = positions(6000);
        std::vector<FoFPart> part;
        for (size_t i = 0; i < x.size(); i++)
            if (i % FML::NTasks == size_t(FML::ThisTask)) {
                FoFPart p{};
                for (int d = 0; d < 3; d++) {
                    p.pos[d] = x[i][d];
                    p.vel[d] = 0.01 * d;
                }
                part.push_back(p);
            }
        FML::PARTICLE::MPIParticles<FoFPart> mp;
        mp.create(part, 4 * part.size() + 1000, FML::PARTICLE::Input::Distributed);
        std::vector<FML::FOF::FoFHalo<FoFPart, 3>> groups;
        FML::FOF::FriendsOfFriends<FoFPart, 3>(mp.get_particles_ptr(), mp.get_npart(), 0.2, 20, true, 0.1, groups);
        std::vector<std::array<double, 3>> centers;
        std::vector<double> radius;
        for (auto & g : groups) {
            centers.push_back({g.pos[0], g.pos[1], g.pos[2]});
            radius.push_back(0.05);
        }
        auto so = FML::FOF::SphericalOverdensity<FoFPart, 3>(
            mp.get_particles_ptr(), mp.get_npart(), double(x.size()), centers, radius, {200.0}, 0.2);
        std::vector<double> local;
        for (size_t i = 0; i < groups.size(); i++) {
            auto & g = groups[i];
            local.insert(local.end(), {double(g.np), g.mass, g.pos[0], g.pos[1], g.pos[2], so[i].mass[0]});
        }
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks), displ(FML::NTasks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displ[i] = displ[i - 1] + counts[i - 1];
        std::vector<double> all(displ.back() + counts.back());
        MPI_Allgatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displ.data(), MPI_DOUBLE,
                       FML::communicator);
        std::vector<std::array<double, 6>> h;
        for (size_t k = 0; k < all.size(); k += 6)
            h.push_back({all[k], all[k + 1], all[k + 2], all[k + 3], all[k + 4], all[k + 5]});
        std::sort(h.begin(), h.end());
        return h;
    }

} // namespace

int main() {
    const auto slabs = FML::domain;
    const auto ref = halos();
    check(ref.size() >= 4, "the slabs: " + std::to_string(ref.size()) + " FoF halos");
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        FML::set_domain(FML::Domain::uniform_pencil(p0, FML::NTasks / p0));
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0);
        short_range_ghosts(name);
        record_ghosts(name);
        const auto h = halos();
        double diff = h.size() == ref.size() ? 0.0 : 1.0;
        for (size_t i = 0; diff == 0.0 and i < h.size(); i++)
            for (int k = 0; k < 6; k++)
                diff = std::max(diff, std::abs(h[i][k] - ref[i][k]) / (1e-12 + std::abs(ref[i][k])));
        check(diff < 1e-10, name + ": FoF and SO give the halos of the slabs (" + std::to_string(h.size()) +
                                " halos, rel. diff " + num(diff, 2) + ")");
        FML::set_domain(slabs);
    }
    return FML::TESTS::summary("buffers on pencils");
}
