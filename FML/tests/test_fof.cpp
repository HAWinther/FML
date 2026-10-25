//==========================================================
// Tests for the friends-of-friends halo finder (FriendsOfFriends/FoF.h): the groups, their centers and the FoF IDs
// stored in the particles must be the same as from a simple serial FoF, for any number of tasks, and groups that
// reach the edge of the buffer copied from the neighbor tasks must be reported.
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FriendsOfFriends/FoF.h>
#include <FML/Global/Global.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <vector>

constexpr int NDIM = 3;

struct Particle {
    double pos[NDIM];
    long long int index;
    size_t fofid;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return NDIM; }
    void set_fofid(size_t id) { fofid = id; }
    size_t get_fofid() const { return fofid; }
};

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}

double wrap(double x) { return x - std::floor(x); }

// The same particles on all tasks: a uniform background plus gaussian blobs, some of them on the boundaries of the
// task domains for 2 and 4 tasks (and on the periodic boundary). Optionally a long filament along x through x = 0.5
std::vector<Particle> make_particles(bool filament) {
    std::mt19937 gen(1234);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<Particle> part;
    auto add = [&](double x, double y, double z) {
        Particle p;
        p.pos[0] = wrap(x);
        p.pos[1] = wrap(y);
        p.pos[2] = wrap(z);
        p.index = (long long int)part.size();
        p.fofid = 0;
        part.push_back(p);
    };
    for (int i = 0; i < 20000; i++)
        add(uni(gen), uni(gen), uni(gen));
    const std::vector<double> xblob{0.0, 0.25, 0.5, 0.75, 0.1, 0.37, 0.62, 0.9, 0.999, 0.501};
    for (size_t b = 0; b < xblob.size(); b++) {
        const double y = uni(gen), z = uni(gen);
        const int n = 50 + 40 * int(b);
        const double sigma = 0.002 + 0.0005 * b;
        for (int i = 0; i < n; i++)
            add(xblob[b] + sigma * gauss(gen), y + sigma * gauss(gen), z + sigma * gauss(gen));
    }
    if (filament)
        for (int i = 0; i < 400; i++)
            add(0.4 + 0.2 * i / 400.0, 0.3, 0.3);
    return part;
}

// A simple serial FoF (union-find on a grid of cells larger than the linking length): the root of each particle
std::vector<size_t> serial_fof(const std::vector<Particle> & part, double linking_distance) {
    const size_t n = part.size();
    std::vector<size_t> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<size_t(size_t)> find = [&](size_t i) {
        while (parent[i] != i)
            i = parent[i] = parent[parent[i]];
        return i;
    };
    const int ngrid = int(1.0 / linking_distance);
    auto cell_of = [&](const Particle & p, int idim) { return std::min(int(p.pos[idim] * ngrid), ngrid - 1); };
    std::map<long long int, std::vector<size_t>> cells;
    auto key = [&](int i, int j, int k) {
        i = (i + ngrid) % ngrid;
        j = (j + ngrid) % ngrid;
        k = (k + ngrid) % ngrid;
        return (long long int)(i * ngrid + j) * ngrid + k;
    };
    for (size_t i = 0; i < n; i++)
        cells[key(cell_of(part[i], 0), cell_of(part[i], 1), cell_of(part[i], 2))].push_back(i);
    const double d2 = linking_distance * linking_distance;
    for (size_t i = 0; i < n; i++) {
        const int ci = cell_of(part[i], 0), cj = cell_of(part[i], 1), ck = cell_of(part[i], 2);
        for (int a = -1; a <= 1; a++)
            for (int b = -1; b <= 1; b++)
                for (int c = -1; c <= 1; c++) {
                    auto it = cells.find(key(ci + a, cj + b, ck + c));
                    if (it == cells.end())
                        continue;
                    for (auto j : it->second) {
                        if (j <= i)
                            continue;
                        double r2 = 0.0;
                        for (int idim = 0; idim < NDIM; idim++) {
                            double dx = std::fabs(part[i].pos[idim] - part[j].pos[idim]);
                            dx = std::min(dx, 1.0 - dx);
                            r2 += dx * dx;
                        }
                        if (r2 < d2)
                            parent[find(i)] = find(j);
                    }
                }
    }
    std::vector<size_t> root(n);
    for (size_t i = 0; i < n; i++)
        root[i] = find(i);
    return root;
}

void test(bool filament, double buffer) {
    const double linking_length = 0.2;
    const int nmin = 20;
    auto all = make_particles(filament);
    std::vector<Particle> local;
    for (auto & p : all)
        if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
            local.push_back(p);

    std::vector<FML::FOF::FoFHalo<Particle, NDIM>> groups;
    FML::FOF::FriendsOfFriends<Particle, NDIM>(
        local.data(), local.size(), linking_length, nmin, true, buffer, groups, 0);
    const size_t nedge = FML::FOF::FoF_number_of_groups_reaching_buffer_edge;
    const std::string label = std::string(filament ? "filament longer than the buffer" : "blobs") + ", " +
                              std::to_string(FML::NTasks) + " task(s): ";

    if (filament) {
        // The filament (0.2 of the box) is longer than the buffer: this must be reported with more than one task
        check(FML::NTasks == 1 ? nedge == 0 : nedge > 0,
              label + "groups reaching the buffer edge reported (" + std::to_string(nedge) + ")");
        return;
    }
    check(nedge == 0, label + "no group reaches the buffer edge");

    // The reference groups
    const double linking_distance = linking_length / std::cbrt(double(all.size()));
    auto root = serial_fof(all, linking_distance);
    std::map<size_t, std::vector<size_t>> members;
    for (size_t i = 0; i < all.size(); i++)
        members[root[i]].push_back(i);
    std::vector<size_t> ref_sizes;
    for (auto & m : members)
        if (int(m.second.size()) >= nmin)
            ref_sizes.push_back(m.second.size());
    std::sort(ref_sizes.begin(), ref_sizes.end());

    // The FoF groups from all tasks (size and center)
    std::vector<double> mine;
    for (auto & g : groups) {
        mine.push_back(double(g.np));
        for (int idim = 0; idim < NDIM; idim++)
            mine.push_back(g.pos[idim]);
    }
    std::vector<double> allgroups;
    {
        int n = int(mine.size());
        std::vector<int> counts(FML::NTasks), displs(FML::NTasks, 0);
#ifdef USE_MPI
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
#else
        counts[0] = n;
#endif
        for (int i = 1; i < FML::NTasks; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
        allgroups.resize(displs.back() + counts.back());
#ifdef USE_MPI
        MPI_Allgatherv(mine.data(), n, MPI_DOUBLE, allgroups.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       FML::communicator);
#else
        allgroups = mine;
#endif
    }
    std::vector<size_t> sizes;
    for (size_t i = 0; i < allgroups.size(); i += 4)
        sizes.push_back(size_t(allgroups[i]));
    std::sort(sizes.begin(), sizes.end());
    check(sizes == ref_sizes, label + "the same groups as the serial FoF (" + std::to_string(sizes.size()) + " vs " +
                                  std::to_string(ref_sizes.size()) + ")");

    // Centers: compare each group with the reference group of the same size (all sizes here are distinct)
    double maxerr = 0.0;
    for (size_t i = 0; i < allgroups.size(); i += 4) {
        for (auto & m : members) {
            if (m.second.size() != size_t(allgroups[i]))
                continue;
            // Periodic mean relative to the first particle
            std::array<double, NDIM> c{};
            const auto & p0 = all[m.second[0]];
            for (auto j : m.second)
                for (int idim = 0; idim < NDIM; idim++) {
                    double dx = all[j].pos[idim] - p0.pos[idim];
                    dx -= std::round(dx);
                    c[idim] += dx / double(m.second.size());
                }
            for (int idim = 0; idim < NDIM; idim++) {
                double d = wrap(p0.pos[idim] + c[idim]) - allgroups[i + 1 + idim];
                d -= std::round(d);
                maxerr = std::max(maxerr, std::fabs(d));
            }
        }
    }
    check(maxerr < 1e-10, label + "the same group centers (max difference " + std::to_string(maxerr) + ")");

    // FoF IDs in the particles: one-to-one with the reference groups (no_FoF_ID outside groups with >= nmin)
    std::vector<long long int> pairs;
    bool ok = true;
    for (auto & p : local) {
        const size_t r = root[p.index];
        const bool in_group = int(members[r].size()) >= nmin;
        if (in_group != (p.fofid != FML::FOF::no_FoF_ID))
            ok = false;
        if (in_group) {
            pairs.push_back((long long int)r);
            pairs.push_back((long long int)p.fofid);
        }
    }
    int ok_int = ok ? 1 : 0;
    FML::MinOverTasks(&ok_int);
    ok = ok_int == 1;
    std::vector<long long int> allpairs;
    {
        int n = int(pairs.size());
        std::vector<int> counts(FML::NTasks), displs(FML::NTasks, 0);
#ifdef USE_MPI
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
#else
        counts[0] = n;
#endif
        for (int i = 1; i < FML::NTasks; i++)
            displs[i] = displs[i - 1] + counts[i - 1];
        allpairs.resize(displs.back() + counts.back());
#ifdef USE_MPI
        MPI_Allgatherv(pairs.data(), n, MPI_LONG_LONG, allpairs.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                       FML::communicator);
#else
        allpairs = pairs;
#endif
    }
    std::map<long long int, std::set<long long int>> root_to_id, id_to_root;
    for (size_t i = 0; i < allpairs.size(); i += 2) {
        root_to_id[allpairs[i]].insert(allpairs[i + 1]);
        id_to_root[allpairs[i + 1]].insert(allpairs[i]);
    }
    for (auto & r : root_to_id)
        ok = ok and r.second.size() == 1;
    for (auto & r : id_to_root)
        ok = ok and r.second.size() == 1;
    ok = ok and root_to_id.size() == ref_sizes.size();
    check(ok, label + "the FoF IDs in the particles are one-to-one with the groups");
}

int main() {
    test(false, 0.05);
    test(true, 0.05);
    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All FoF tests passed\n" : std::to_string(fails) + " FoF test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
