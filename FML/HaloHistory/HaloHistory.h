#ifndef FML_HALO_HISTORY_HEADER
#define FML_HALO_HISTORY_HEADER
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace FML::HALO_HISTORY {
    using ID = std::uint64_t;
    enum class Membership { Exclusive, Inclusive };
    struct Metadata {
        std::string mass_definition, length_unit = "Mpc/h", mass_unit = "Msun/h", velocity_unit = "km/s",
                                     length_convention = "comoving", velocity_convention = "peculiar", id_namespace;
        bool operator==(const Metadata& m) const {
            return std::tie(mass_definition, length_unit, mass_unit, velocity_unit, length_convention,
                            velocity_convention,
                            id_namespace) == std::tie(m.mass_definition, m.length_unit, m.mass_unit, m.velocity_unit,
                                                      m.length_convention, m.velocity_convention, m.id_namespace);
        }
    };
    struct Halo {
        ID id = 0;
        std::optional<ID> host;
        double mass = 1, radius = 1, vmax = 0;
        std::array<double, 3> position{}, velocity{};
        std::vector<ID> particles;
    };
    struct Snapshot {
        ID id = 0;
        double scale_factor = 1, boxsize = 1;
        Membership membership = Membership::Exclusive;
        Metadata metadata;
        std::vector<Halo> halos;
    };
    struct NodeKey {
        ID snapshot = 0, halo = 0;
        bool operator<(const NodeKey& k) const { return std::tie(snapshot, halo) < std::tie(k.snapshot, k.halo); }
        bool operator==(const NodeKey& k) const { return snapshot == k.snapshot && halo == k.halo; }
    };
    struct MatchOptions {
        ID minimum_shared = 2;
        double minimum_source_fraction = 0, minimum_target_fraction = 0, minimum_merit = 0;
        size_t max_candidates = 10000000, max_overlap_work = 100000000;
    };
    struct Candidate {
        ID source = 0, target = 0, shared = 0;
        double source_fraction = 0, target_fraction = 0, merit = 0;
    };
    struct Matching {
        ID source_snapshot = 0, target_snapshot = 0;
        std::vector<Candidate> candidates;
        std::map<ID, Candidate> descendants;
    };
    struct Link {
        NodeKey source, target;
        ID shared = 0;
        double merit = 0, source_fraction = 0, target_fraction = 0;
        size_t skipped_snapshots = 0;
    };
    struct Node {
        NodeKey key;
        double scale_factor = 0, mass = 0, radius = 0, vmax = 0;
        std::array<double, 3> position{}, velocity{};
        std::optional<NodeKey> host, descendant, main_progenitor;
        std::vector<NodeKey> progenitors;
    };
    struct History {
        Metadata metadata;
        double boxsize = 0;
        Membership membership = Membership::Exclusive;
        MatchOptions match_options;
        size_t max_skipped_snapshots = 0;
        std::vector<Node> nodes;
        std::vector<Link> links;
        std::vector<NodeKey> roots;
    };
    namespace detail {
        inline void require(bool valid, const char* text) {
            if (!valid)
                throw std::invalid_argument(text);
        }
        inline bool token(const std::string& s) {
            if (s.empty())
                return false;
            for (unsigned char c : s)
                if (c <= 32 || c > 126)
                    return false;
            return true;
        }
        inline bool better(const Candidate& a, const Candidate& b) {
            if (a.merit != b.merit)
                return a.merit > b.merit;
            if (a.shared != b.shared)
                return a.shared > b.shared;
            return a.target < b.target;
        }
        inline void options(const MatchOptions& o) {
            require(o.minimum_shared > 0 && o.max_candidates > 0 && o.max_overlap_work > 0,
                    "Invalid halo match count/budget");
            for (double x : {o.minimum_source_fraction, o.minimum_target_fraction, o.minimum_merit})
                require(std::isfinite(x) && x >= 0 && x <= 1, "Halo match thresholds must be in [0,1]");
        }
        using Entry = std::pair<ID, size_t>;
        inline std::vector<Entry> index(const Snapshot& s) {
            std::vector<Entry> out;
            size_t count = 0;
            for (const auto& h : s.halos) {
                require(h.particles.size() <= out.max_size() - count,
                        "Halo membership index exceeds addressable storage");
                count += h.particles.size();
            }
            out.reserve(count);
            for (size_t i = 0; i < s.halos.size(); ++i)
                for (ID pid : s.halos[i].particles)
                    out.emplace_back(pid, i);
            std::sort(out.begin(), out.end());
            return out;
        }
    } // namespace detail
    /// Local validation. Inclusive memberships permit shared IDs between halos,
    /// never duplicate IDs within a halo. Host relationships must form a forest.
    inline void validate(const Snapshot& s) {
        detail::require(s.scale_factor > 0 && std::isfinite(s.scale_factor) && s.boxsize > 0 &&
                            std::isfinite(s.boxsize),
                        "Invalid snapshot epoch/boxsize");
        detail::require(s.membership == Membership::Exclusive || s.membership == Membership::Inclusive,
                        "Invalid halo membership policy");
        for (const auto& token :
             {s.metadata.mass_definition, s.metadata.length_unit, s.metadata.mass_unit, s.metadata.velocity_unit,
              s.metadata.length_convention, s.metadata.velocity_convention, s.metadata.id_namespace})
            detail::require(detail::token(token), "Halo metadata must use nonempty ASCII tokens");
        std::map<ID, size_t> ids;
        std::set<ID> global_particles;
        for (size_t i = 0; i < s.halos.size(); ++i) {
            const auto& h = s.halos[i];
            detail::require(ids.emplace(h.id, i).second, "Duplicate halo ID in snapshot");
            detail::require(h.mass > 0 && h.radius > 0 && h.vmax >= 0 && std::isfinite(h.mass) &&
                                std::isfinite(h.radius) && std::isfinite(h.vmax),
                            "Invalid halo mass/radius/Vmax");
            for (double x : h.position)
                detail::require(std::isfinite(x) && x >= 0 && x < s.boxsize, "Halo positions must lie in [0,L)");
            for (double v : h.velocity)
                detail::require(std::isfinite(v), "Invalid halo velocity");
            std::set<ID> members;
            for (ID p : h.particles) {
                detail::require(members.insert(p).second, "Duplicate particle ID within halo");
                if (s.membership == Membership::Exclusive)
                    detail::require(global_particles.insert(p).second,
                                    "Exclusive catalogue assigns a particle to multiple halos");
            }
        }
        std::vector<unsigned char> state(s.halos.size());
        for (size_t i = 0; i < s.halos.size(); ++i) {
            size_t current = i;
            std::vector<size_t> path;
            while (state[current] == 0) {
                state[current] = 1;
                path.push_back(current);
                const auto& host = s.halos[current].host;
                if (!host)
                    break;
                auto found = ids.find(*host);
                detail::require(found != ids.end() && found->second != current, "Missing/self halo host");
                current = found->second;
                detail::require(state[current] != 1, "Cyclic halo host hierarchy");
            }
            for (size_t j : path)
                state[j] = 2;
        }
    }
    /// Sorted inverted particle-ID indices replace an all-halo-pairs scan.
    inline Matching match(const Snapshot& source, const Snapshot& target, const MatchOptions& options = {}) {
        validate(source);
        validate(target);
        detail::options(options);
        detail::require(
            source.id < target.id && source.scale_factor < target.scale_factor && source.boxsize == target.boxsize &&
                source.metadata == target.metadata && source.membership == target.membership,
            "Halo matches require increasing snapshots/epochs and identical units/ID namespace/box metadata");
        const auto a = detail::index(source), b = detail::index(target);
        std::map<std::pair<size_t, size_t>, ID> shared;
        size_t work = 0, i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i].first < b[j].first) {
                ++i;
                continue;
            }
            if (b[j].first < a[i].first) {
                ++j;
                continue;
            }
            size_t ai = i + 1, bj = j + 1;
            while (ai < a.size() && a[ai].first == a[i].first)
                ++ai;
            while (bj < b.size() && b[bj].first == b[j].first)
                ++bj;
            detail::require(bj - j <= options.max_overlap_work / (ai - i),
                            "Inclusive membership overlap work exceeds budget");
            size_t more = (ai - i) * (bj - j);
            detail::require(more <= options.max_overlap_work - work, "Membership overlap work exceeds budget");
            work += more;
            for (size_t x = i; x < ai; ++x)
                for (size_t y = j; y < bj; ++y) {
                    auto key = std::make_pair(a[x].second, b[y].second);
                    auto found = shared.find(key);
                    if (found == shared.end()) {
                        detail::require(shared.size() < options.max_candidates, "Halo candidate budget exceeded");
                        found = shared.emplace(key, 0).first;
                    }
                    detail::require(found->second < std::numeric_limits<ID>::max(), "Shared particle count overflow");
                    ++found->second;
                }
            i = ai;
            j = bj;
        }
        Matching out;
        out.source_snapshot = source.id;
        out.target_snapshot = target.id;
        for (const auto& item : shared) {
            const auto& from = source.halos[item.first.first];
            const auto& to = target.halos[item.first.second];
            Candidate c;
            c.source = from.id;
            c.target = to.id;
            c.shared = item.second;
            c.source_fraction = double(c.shared) / from.particles.size();
            c.target_fraction = double(c.shared) / to.particles.size();
            c.merit = c.source_fraction * c.target_fraction;
            if (c.shared < options.minimum_shared || c.source_fraction < options.minimum_source_fraction ||
                c.target_fraction < options.minimum_target_fraction || c.merit < options.minimum_merit)
                continue;
            out.candidates.push_back(c);
            auto chosen = out.descendants.find(c.source);
            if (chosen == out.descendants.end() || detail::better(c, chosen->second))
                out.descendants[c.source] = c;
        }
        std::sort(out.candidates.begin(), out.candidates.end(), [](const Candidate& a, const Candidate& b) {
            return std::tie(a.source, a.target) < std::tie(b.source, b.target);
        });
        return out;
    }
    /// Earliest accepted descendant, optional bounded forward snapshot skipping.
    /// No trajectories, binding-rank weighting, or mass-history repair are inferred.
    inline History build(const std::vector<Snapshot>& snapshots, const MatchOptions& options = {},
                         size_t max_skipped_snapshots = 0) {
        detail::options(options);
        for (size_t s = 0; s < snapshots.size(); ++s) {
            validate(snapshots[s]);
            if (s)
                detail::require(snapshots[s - 1].id < snapshots[s].id &&
                                    snapshots[s - 1].scale_factor < snapshots[s].scale_factor &&
                                    snapshots[s - 1].metadata == snapshots[s].metadata &&
                                    snapshots[s - 1].boxsize == snapshots[s].boxsize &&
                                    snapshots[s - 1].membership == snapshots[s].membership,
                                "History snapshots must increase in ID/epoch with consistent metadata");
        }
        History out;
        out.match_options = options;
        out.max_skipped_snapshots = max_skipped_snapshots;
        if (!snapshots.empty()) {
            out.metadata = snapshots.front().metadata;
            out.boxsize = snapshots.front().boxsize;
            out.membership = snapshots.front().membership;
        }
        std::map<NodeKey, size_t> node_index;
        for (const auto& s : snapshots) {
            std::vector<const Halo*> ordered;
            for (const auto& h : s.halos)
                ordered.push_back(&h);
            std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return a->id < b->id; });
            for (const auto* h : ordered) {
                Node node;
                node.key = {s.id, h->id};
                node.scale_factor = s.scale_factor;
                node.mass = h->mass;
                node.radius = h->radius;
                node.vmax = h->vmax;
                node.position = h->position;
                node.velocity = h->velocity;
                if (h->host)
                    node.host = NodeKey{s.id, *h->host};
                node_index.emplace(node.key, out.nodes.size());
                out.nodes.push_back(std::move(node));
            }
        }
        for (size_t s = 0; s < snapshots.size(); ++s) {
            const size_t remaining = snapshots.size() - s - 1,
                         reach = max_skipped_snapshots >= remaining ? remaining : max_skipped_snapshots + 1;
            size_t unresolved = snapshots[s].halos.size();
            for (size_t offset = 1; offset <= reach && unresolved; ++offset) {
                auto candidates = match(snapshots[s], snapshots[s + offset], options);
                for (const auto& item : candidates.descendants) {
                    auto& node = out.nodes[node_index.at({snapshots[s].id, item.first})];
                    if (node.descendant)
                        continue;
                    const auto& c = item.second;
                    NodeKey target{snapshots[s + offset].id, c.target};
                    node.descendant = target;
                    --unresolved;
                    out.links.push_back(
                        {node.key, target, c.shared, c.merit, c.source_fraction, c.target_fraction, offset - 1});
                }
            }
        }
        std::map<NodeKey, Link> main;
        for (const auto& link : out.links) {
            auto& target = out.nodes[node_index.at(link.target)];
            target.progenitors.push_back(link.source);
            auto old = main.find(link.target);
            if (old == main.end() || link.merit > old->second.merit ||
                (link.merit == old->second.merit &&
                 (link.shared > old->second.shared ||
                  (link.shared == old->second.shared && link.source < old->second.source))))
                main[link.target] = link;
        }
        for (auto& node : out.nodes) {
            std::sort(node.progenitors.begin(), node.progenitors.end());
            auto found = main.find(node.key);
            if (found != main.end())
                node.main_progenitor = found->second.source;
            if (!node.descendant)
                out.roots.push_back(node.key);
        }
        std::sort(out.links.begin(), out.links.end(), [](const Link& a, const Link& b) { return a.source < b.source; });
        return out;
    }
} // namespace FML::HALO_HISTORY
#endif
