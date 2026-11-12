// =============================================================================
// Tests of relaxed friends-of-friends (RelaxedFoF.h) and of the FoF options it needs. Run with `make run` in this
// folder (on 1, 2 and 4 MPI tasks).
//
//   * The halos (members, pass, bin) of fastpm and paper mode at z = 0, 1, 1.5 equal those of an independent
//     implementation (make_rfof_reference.py: scipy KD-tree FoF) on a catalogue with clumps of 8-600 particles,
//     diffuse and fake (too fast) clumps and cold cores in hot envelopes
//   * One bin without the velocity cut is standard FoF; FriendsOfFriends equals the reference
//   * FriendsOfFriends with a given mean particle separation on a subset (empty tasks with 2 or 4 tasks)
//   * The linking lengths, r0 and the settings checks; the halo IDs stored in the particles
// =============================================================================

#include <FML/FriendsOfFriends/RelaxedFoF.h>
#include <FML/tests/TestHelpers.h>

#include <fstream>
#include <map>
#include <set>
#include <sstream>

using namespace FML::FOF;
using FML::TESTS::check;
using FML::TESTS::check_local;

constexpr int NDIM = 3;
const double mass_unit = 1e10; // Msun/h per particle (as make_rfof_reference.py)
const double buffer = 0.12;

struct Particle {
    double pos[NDIM];
    double vel[NDIM];
    size_t gid;
    size_t fofid{no_FoF_ID};
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    void set_fofid(size_t id) { fofid = id; }
    size_t get_fofid() const { return fofid; }
    constexpr int get_ndim() const { return NDIM; }
};

/// Gather a vector from all the tasks to all the tasks (concatenated in task order)
static std::vector<long long> all_gather(const std::vector<long long> & mine) {
#ifdef USE_MPI
    int n = int(mine.size());
    std::vector<int> counts(size_t(FML::NTasks)), offsets(size_t(FML::NTasks), 0);
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
    for (int t = 1; t < FML::NTasks; t++)
        offsets[size_t(t)] = offsets[size_t(t) - 1] + counts[size_t(t) - 1];
    std::vector<long long> all(size_t(offsets.back() + counts.back()));
    MPI_Allgatherv(mine.data(), n, MPI_LONG_LONG, all.data(), counts.data(), offsets.data(), MPI_LONG_LONG,
                   FML::communicator);
    return all;
#else
    return mine;
#endif
}

struct RefHalo {
    int pass{0}, bin{0};
    std::vector<size_t> members;
};

struct Reference {
    size_t npart{0};
    std::map<std::string, std::vector<RefHalo>> cases; // "fastpm 0" ... and "plain", "subset"
    std::vector<std::tuple<std::string, double, double>> case_list;
};

static Reference read_reference(size_t npart) {
    Reference ref;
    ref.npart = npart;
    std::ifstream in("data/rfof_reference.txt");
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() or line[0] == '#')
            continue;
        std::stringstream ss(line);
        std::string word;
        ss >> word;
        std::string key;
        size_t n = 0;
        bool with_pass = true;
        if (word == "case") {
            std::string mode;
            double z, E;
            ss >> mode >> z >> E >> n;
            key = mode + " " + std::to_string(z);
            ref.case_list.emplace_back(mode, z, E);
        } else {
            key = word;
            ss >> n;
            with_pass = false;
        }
        auto & halos = ref.cases[key];
        for (size_t h = 0; h < n; h++) {
            std::getline(in, line);
            std::stringstream hs(line);
            RefHalo r;
            size_t np;
            if (with_pass)
                hs >> r.pass >> r.bin;
            hs >> np;
            r.members.resize(np);
            for (auto & m : r.members)
                hs >> m;
            halos.push_back(r);
        }
    }
    return ref;
}

/// The local particles (x in this task's domain) of the test catalogue; total in npart
static std::vector<Particle> read_particles(size_t & npart) {
    std::ifstream in("data/rfof_particles.txt");
    std::vector<Particle> part;
    std::string line;
    npart = 0;
    while (std::getline(in, line)) {
        if (line.empty() or line[0] == '#')
            continue;
        std::stringstream ss(line);
        Particle p{};
        for (auto & x : p.pos)
            ss >> x;
        for (auto & v : p.vel)
            ss >> v;
        p.gid = npart++;
        if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
            part.push_back(p);
    }
    return part;
}

/// Compare a labelling of all the particles (label per global index, -1: none) with reference groups: the same
/// partition. Returns an empty string if equal, otherwise what differs
static std::string compare_partition(const std::vector<long long> & label, const std::vector<RefHalo> & ref) {
    std::map<long long, size_t> size_of_label;
    for (auto l : label)
        if (l >= 0)
            size_of_label[l]++;
    if (size_of_label.size() != ref.size())
        return std::to_string(size_of_label.size()) + " halos, expected " + std::to_string(ref.size());
    size_t assigned = 0;
    for (auto & r : ref) {
        const long long l = label[r.members[0]];
        if (l < 0)
            return "a reference halo of " + std::to_string(r.members.size()) + " particles is missing";
        for (auto m : r.members)
            if (label[m] != l)
                return "a reference halo of " + std::to_string(r.members.size()) + " particles is split or merged";
        if (size_of_label[l] != r.members.size())
            return "a halo has " + std::to_string(size_of_label[l]) + " particles, expected " +
                   std::to_string(r.members.size());
        assigned += r.members.size();
    }
    return "";
}

/// The labels of all the particles from (gid, label) of the local particles
static std::vector<long long> gather_labels(const std::vector<Particle> & part,
                                            const std::vector<size_t> & ids,
                                            size_t npart) {
    std::vector<long long> mine;
    for (size_t i = 0; i < part.size(); i++) {
        mine.push_back((long long)part[i].gid);
        mine.push_back(ids[i] == no_FoF_ID ? -1 : (long long)ids[i]);
    }
    const auto all = all_gather(mine);
    std::vector<long long> label(npart, -1);
    for (size_t k = 0; k + 1 < all.size(); k += 2)
        label[size_t(all[k])] = all[k + 1];
    return label;
}

int main() {
    size_t npart = 0;
    const auto particles = read_particles(npart);
    const auto ref = read_reference(npart);
    const double sep = 1.0 / std::cbrt(double(npart));
    check(npart == 9016 and ref.case_list.size() == 5, "read the test catalogue and the reference");

    //-------------------------------------------------------------------------
    // The parameters
    //-------------------------------------------------------------------------
    {
        RelaxedFoFParameters p;
        const bool ll_ok = std::abs(p.linking_length(1, 0.0) - 0.238) < 1e-14 and
                           std::abs(p.linking_length(6, 0.0) - 0.2) < 1e-14 and
                           std::abs(p.linking_length(6, 2.0) - 0.22) < 1e-14 and
                           std::abs(p.linking_length(3, 0.0) - (3 * 0.238 + 2 * 0.2) / 5) < 1e-14;
        check(ll_ok, "linking lengths: eqs. 2.3-2.5 with FastPM's defaults");
        check(std::abs(p.r0(1.0) - (7.02 - 6.025 * std::log(2.0))) < 1e-14 and p.bin_upper_edge(1) == 20 and
                  p.bin_upper_edge(6) == std::numeric_limits<size_t>::max(),
              "r0 (natural log), bin edges");
        p.r0_min = 0.5;
        check(p.r0(3.0) == 0.5, "r0_min floors r0");
        RelaxedFoFParameters bad;
        bad.bin_edges = {20, 10};
        check(FML::TESTS::throws<std::invalid_argument>([&] { bad.validate(); }), "decreasing bin edges rejected");
    }

    //-------------------------------------------------------------------------
    // The reference cases
    //-------------------------------------------------------------------------
    for (auto & [mode, z, E] : ref.case_list) {
        auto part = particles;
        RelaxedFoFParameters p;
        p.mode = mode == "paper" ? RelaxedFoFParameters::Mode::paper : RelaxedFoFParameters::Mode::fastpm;
        p.verbose = false;
        std::vector<size_t> ids;
        std::vector<RelaxedFoFPass> passes;
        const auto halos = RelaxedFriendsOfFriends<Particle, NDIM>(
            part.data(), part.size(), p, z, E, mass_unit, 1.0, true, buffer, &ids, &passes);
        const auto & expected = ref.cases.at(mode + " " + std::to_string(z));
        const std::string tag = mode + " z = " + FML::TESTS::num(z);

        const auto label = gather_labels(part, ids, npart);
        const std::string diff = compare_partition(label, expected);
        check(diff.empty(), tag + ": the same halos as the reference" + (diff.empty() ? "" : " (" + diff + ")"));

        // pass, bin and np of each halo; the IDs in the particles
        std::vector<long long> mine;
        for (auto & h : halos) {
            mine.push_back((long long)h.id);
            mine.push_back(h.pass);
            mine.push_back(h.bin);
            mine.push_back((long long)h.np);
        }
        const auto all = all_gather(mine);
        std::map<long long, std::array<long long, 3>> info;
        for (size_t k = 0; k + 3 < all.size(); k += 4)
            info[all[k]] = {all[k + 1], all[k + 2], all[k + 3]};
        bool same = info.size() == expected.size();
        for (auto & r : expected) {
            const auto it = info.find(label[r.members[0]]);
            same = same and it != info.end() and it->second[0] == r.pass and it->second[1] == r.bin and
                   size_t(it->second[2]) == r.members.size();
        }
        check(same, tag + ": pass, bin and size of every halo");
        bool fofid_ok = true;
        for (size_t i = 0; i < part.size(); i++)
            fofid_ok = fofid_ok and part[i].fofid == ids[i];
        check(fofid_ok, tag + ": the halo IDs are stored in the particles (set_fofid)");

        size_t saved = 0;
        for (auto & ps : passes)
            saved += ps.saved;
        check(saved == expected.size() and passes.size() >= 6, tag + ": the pass summary counts the halos");
    }

    //-------------------------------------------------------------------------
    // Standard FoF: FriendsOfFriends, and RelaxedFoF with one bin without the velocity cut
    //-------------------------------------------------------------------------
    {
        auto part = particles;
        std::vector<FoFHalo<Particle, NDIM>> groups;
        FoF_verbose = false;
        FriendsOfFriends<Particle, NDIM>(part.data(), part.size(), 0.2, 8, true, buffer, groups);
        std::vector<size_t> ids;
        for (auto & p : part)
            ids.push_back(p.fofid);
        const std::string diff = compare_partition(gather_labels(part, ids, npart), ref.cases.at("plain"));
        check(diff.empty(), "FriendsOfFriends equals the reference FoF" + (diff.empty() ? "" : " (" + diff + ")"));

        auto part2 = particles;
        RelaxedFoFParameters p;
        p.bin_edges = {};
        p.l1 = 0.2;
        p.a1 = 0.0;
        p.velocity_rejection = false;
        p.verbose = false;
        std::vector<size_t> ids2;
        RelaxedFriendsOfFriends<Particle, NDIM>(part2.data(), part2.size(), p, 0.0, 1.0, mass_unit, 1.0, true, buffer,
                                                &ids2);
        const std::string diff2 = compare_partition(gather_labels(part2, ids2, npart), ref.cases.at("plain"));
        check(diff2.empty(), "RelaxedFoF with one bin and no velocity cut is standard FoF" +
                                 (diff2.empty() ? "" : " (" + diff2 + ")"));
    }

    //-------------------------------------------------------------------------
    // FoF of a subset (x > 0.75) with the mean separation of all the particles: tasks without particles
    //-------------------------------------------------------------------------
    {
        std::vector<Particle> part;
        for (auto & p : particles)
            if (p.pos[0] > 0.75)
                part.push_back(p);
        size_t nlocal = part.size(), empty_tasks = nlocal == 0 ? 1 : 0;
        FML::SumOverTasks(&empty_tasks);
        std::vector<FoFHalo<Particle, NDIM>> groups;
        FriendsOfFriends<Particle, NDIM>(part.data(), part.size(), 0.2, 8, true, buffer, groups, FoF_Ngrid_max, sep);
        std::vector<size_t> ids;
        for (auto & p : part)
            ids.push_back(p.fofid);
        const std::string diff = compare_partition(gather_labels(part, ids, npart), ref.cases.at("subset"));
        check(diff.empty(), "FoF of a subset with the full mean separation (" + std::to_string(empty_tasks) +
                                " tasks without particles)" + (diff.empty() ? "" : " (" + diff + ")"));
    }

    return FML::TESTS::summary("RelaxedFoF");
}
