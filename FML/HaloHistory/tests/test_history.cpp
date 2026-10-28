#include <FML/Global/Global.h>
#include <FML/HaloHistory/CatalogueIO.h>
#include <iostream>
#include <sstream>
using namespace FML::HALO_HISTORY;
int failures = 0;
void check(bool ok, const std::string& label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
Halo halo(ID id, std::vector<ID> members, std::optional<ID> host = {}) {
    Halo h;
    h.id = id;
    h.particles = std::move(members);
    h.host = host;
    h.mass = 1e12;
    h.radius = .1;
    h.vmax = 150;
    h.position = {.2, .3, .4};
    h.velocity = {10, 20, 30};
    return h;
}
Snapshot snapshot(ID id, double a, std::vector<Halo> halos) {
    Snapshot s;
    s.id = id;
    s.metadata.mass_definition = "M200m_host_bound_sub";
    s.metadata.id_namespace = "test-run";
    s.scale_factor = a;
    s.halos = std::move(halos);
    return s;
}
bool same_links(const History& a, const History& b) {
    if (a.links.size() != b.links.size() || a.nodes.size() != b.nodes.size())
        return false;
    for (size_t i = 0; i < a.links.size(); ++i)
        if (!(a.links[i].source == b.links[i].source && a.links[i].target == b.links[i].target &&
              a.links[i].shared == b.links[i].shared && a.links[i].merit == b.links[i].merit))
            return false;
    return true;
}
int main() {
    auto a = snapshot(10, .4, {halo(1, {10, 11, 12, 13}), halo(2, {20, 21}), halo(3, {30, 31})});
    auto b = snapshot(20, .5, {halo(101, {10, 11, 12, 13, 20, 21}), halo(102, {30, 31})});
    auto m = match(a, b);
    check(m.descendants.size() == 3 && m.descendants.at(1).target == 101 && m.descendants.at(1).shared == 4 &&
              m.descendants.at(1).source_fraction == 1 && std::abs(m.descendants.at(1).merit - 2. / 3) < 1e-15 &&
              m.descendants.at(2).target == 101,
          "particle-overlap merit and many-to-one merger match a hand-computed catalogue");
    bool direct = true;
    for (const auto& c : m.candidates) {
        const Halo *from = nullptr, *to = nullptr;
        for (const auto& h : a.halos)
            if (h.id == c.source)
                from = &h;
        for (const auto& h : b.halos)
            if (h.id == c.target)
                to = &h;
        size_t shared = 0;
        for (ID x : from->particles)
            for (ID y : to->particles)
                shared += x == y;
        direct &= shared == c.shared;
    }
    check(direct, "inverted ID index agrees with independent all-halo/all-particle intersection counting");
    auto tree = build({a, b});
    const Node* merged = nullptr;
    for (const auto& n : tree.nodes)
        if (n.key == NodeKey{20, 101})
            merged = &n;
    check(merged && merged->progenitors.size() == 2 && merged->main_progenitor == std::optional<NodeKey>{{10, 1}} &&
              merged->mass == 1e12 && tree.roots.size() == 2,
          "history stores properties, all accepted progenitors and a main progenitor among actual incoming links");
#ifdef USE_OMP
    int concurrent_bad = 0;
#pragma omp parallel for reduction(+ : concurrent_bad)
    for (int i = 0; i < 8; ++i)
        if (!same_links(tree, build({a, b})))
            ++concurrent_bad;
    check(concurrent_bad == 0, "independent history construction is thread-safe");
#endif
    auto reversed_a = a, reversed_b = b;
    std::reverse(reversed_a.halos.begin(), reversed_a.halos.end());
    std::reverse(reversed_b.halos.begin(), reversed_b.halos.end());
    for (auto& h : reversed_a.halos)
        std::reverse(h.particles.begin(), h.particles.end());
    check(same_links(tree, build({reversed_a, reversed_b})),
          "halo and member ordering do not change graph identities or links");
    auto split = snapshot(20, .5, {halo(200, {10, 11}), halo(100, {12, 13})});
    auto tied = match(snapshot(10, .4, {a.halos[0]}), split);
    check(tied.descendants.at(1).target == 100, "equal-merit descendant ties use shared count then stable halo ID");
    MatchOptions threshold;
    threshold.minimum_shared = 3;
    auto cut = match(a, b, threshold);
    check(cut.descendants.size() == 1 && cut.descendants.count(1),
          "shared-count thresholds reject unresolved candidate links");
    auto gap0 = snapshot(1, .3, {halo(1, {100, 101})}), gap1 = snapshot(2, .4, {}),
         gap2 = snapshot(3, .5, {halo(9, {100, 101})});
    auto adjacent = build({gap0, gap1, gap2});
    auto skipped = build({gap0, gap1, gap2}, {}, 1);
    check(adjacent.links.empty() && skipped.links.size() == 1 && skipped.links[0].skipped_snapshots == 1 &&
              skipped.links[0].target == NodeKey{3, 9},
          "bounded lookahead recovers a temporarily absent halo without changing the adjacent-only default");
    auto late = snapshot(30, .6, {halo(500, {10, 11, 12, 13})});
    auto earliest = build({snapshot(10, .4, {a.halos[0]}), snapshot(20, .5, {halo(300, {10, 11})}), late}, {}, 1);
    check(earliest.links[0].target == NodeKey{20, 300},
          "lookahead preserves an eligible earlier descendant instead of choosing a later stronger overlap");
    auto inclusive = snapshot(10, .4, {halo(1, {1, 2, 3, 4}), halo(2, {1, 2}, 1)});
    inclusive.membership = Membership::Inclusive;
    auto next = inclusive;
    next.id = 20;
    next.scale_factor = .5;
    next.halos[0].id = 11;
    next.halos[1].id = 12;
    next.halos[1].host = 11;
    auto sub = build({inclusive, next});
    bool host = false;
    for (const auto& n : sub.nodes)
        if (n.key == NodeKey{10, 2})
            host = n.host == std::optional<NodeKey>{{10, 1}};
    check(host && sub.links.size() == 2,
          "inclusive host/subhalo memberships remain distinct graph nodes with explicit host relationships");
    auto exclusive = inclusive;
    exclusive.membership = Membership::Exclusive;
    bool rejected = false;
    try {
        validate(exclusive);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "overlapping memberships require an explicit inclusive convention");
    auto cyclic = inclusive;
    cyclic.halos[0].host = 2;
    rejected = false;
    try {
        validate(cyclic);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "cyclic host hierarchies are rejected");
    auto duplicate = a;
    duplicate.halos[0].particles.push_back(10);
    rejected = false;
    try {
        validate(duplicate);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "duplicate IDs within one halo are rejected rather than double-counted");
    MatchOptions budget;
    budget.max_overlap_work = 1;
    rejected = false;
    try {
        match(a, b, budget);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "overlap work budget fails explicitly without truncating links");
    budget = {};
    budget.max_candidates = 1;
    rejected = false;
    try {
        match(a, b, budget);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "candidate memory budget fails explicitly without truncating links");
    auto incompatible = b;
    incompatible.metadata.id_namespace = "another-run";
    rejected = false;
    try {
        match(a, incompatible);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "inconsistent particle ID namespaces/units are rejected");
    auto exact = inclusive;
    exact.id = std::numeric_limits<ID>::max();
    exact.halos[0].id = std::numeric_limits<ID>::max();
    exact.halos[1].host = exact.halos[0].id;
    exact.halos[0].particles = {0, (ID(1) << 63) + 17, std::numeric_limits<ID>::max()};
    exact.halos[1].particles = {std::numeric_limits<ID>::max()};
    std::stringstream stream;
    write_catalogue(stream, exact);
    auto loaded = read_catalogue(stream);
    check(loaded.id == exact.id && loaded.halos[0].id == exact.halos[0].id &&
              loaded.halos[0].particles == exact.halos[0].particles && loaded.halos[1].host == exact.halos[1].host &&
              loaded.metadata == exact.metadata,
          "neutral catalogue IO preserves full uint64 IDs, memberships, host relations and metadata");
    std::stringstream formatted;
    formatted << std::hex << std::scientific << std::setprecision(3);
    auto original_flags = formatted.flags();
    auto original_precision = formatted.precision();
    write_catalogue(formatted, exact);
    auto formatted_loaded = read_catalogue(formatted);
    check(formatted_loaded.halos[0].id == exact.halos[0].id && formatted.flags() == original_flags &&
              formatted.precision() == original_precision,
          "catalogue IO uses canonical decimal formatting and restores caller stream settings");
    auto high_source = snapshot(1, .3, {halo((ID(1) << 63) + 9, {std::numeric_limits<ID>::max(), (ID(1) << 63) + 10})});
    auto high_target = high_source;
    high_target.id = 2;
    high_target.scale_factor = .4;
    high_target.halos[0].id = std::numeric_limits<ID>::max();
    check(match(high_source, high_target).descendants.at((ID(1) << 63) + 9).target == std::numeric_limits<ID>::max(),
          "large particle and halo IDs stay exact during graph matching");
    auto text = stream.str();
    rejected = false;
    try {
        std::istringstream bad(text.substr(0, text.size() - 6));
        read_catalogue(bad);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "truncated catalogue membership input is rejected");
    rejected = false;
    try {
        std::istringstream bad(text + "junk");
        read_catalogue(bad);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "unexpected trailing content is rejected");
    rejected = false;
    try {
        std::istringstream too_large(text);
        read_catalogue(too_large, 1, 100);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "external catalogue read budgets are enforced before allocation");
    auto hod = to_hod(inclusive, 6, [](const Halo&) { return 150.; });
    auto all_hod = to_hod(
        inclusive, 6, [](const Halo&) { return 150.; }, true);
    check(hod.size() == 1 && all_hod.size() == 2 && hod[0].id == 1 && hod[0].mass == 1e12 &&
              hod[0].concentration == 6 && hod[0].velocity_dispersion == 150 && hod[0].members.empty(),
          "HOD adapter preserves units/IDs and requires explicit concentration/velocity-dispersion assumptions");
    if (!FML::ThisTask)
        std::cout << (failures ? "Halo history tests FAILED" : "All halo history tests passed") << '\n';
    return failures ? 1 : 0;
}
