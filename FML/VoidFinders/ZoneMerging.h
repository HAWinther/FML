#ifndef FML_VOIDFINDERS_ZONEMERGING_HEADER
#define FML_VOIDFINDERS_ZONEMERGING_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <FML/Global/Global.h>

namespace FML {
    namespace VOIDFINDER {

        /// How the watershed zones are merged into voids
        enum class ZoneMerging {
            None,      ///< Every zone is a void (as in Revolver)
            Threshold, ///< Zones are merged through links with link density below a threshold (as in VIDE)
            Full       ///< The full ZOBOV hierarchy (no threshold)
        };

        /// What the zone merging needs to know about a zone (sums so that zones can be added together)
        struct Zone {
            /// The quantity (density for voids, 1/density for clusters) of the minimum
            double quantity_core{0.0};
            /// The position of the minimum
            double pos_min[3]{};
            /// The center (e.g. the circumcenter for ZOBOV)
            double pos_center[3]{};
            double volume{0.0};
            double mass{0.0};
            double sum_density{0.0};
            /// The barycenter is pos_min + weighted_dx / weight
            double weight{0.0};
            double weighted_dx[3]{};
            long long int n{0};
        };

        /// The link between two zones: the lowest quantity at which they connect (for a pair of neighboring
        /// particles/voxels in the two zones the link has the larger of the two quantities)
        struct ZoneLink {
            long long int a{0};
            long long int b{0};
            double quantity{0.0};
        };

        /// A void from merging zones: the sum of its zones (relative to the minimum of the deepest zone)
        struct MergedVoid {
            Zone sum{};
            /// The quantity at which the void merges with a deeper void (or links to another void)
            double quantity_leak{std::numeric_limits<double>::infinity()};
            /// The deepest zone (the void is labeled by this)
            long long int root_zone{0};
            /// The void (labeled by its deepest zone) this void merges into, -1 for a top-level void
            long long int parent_zone{-1};
            long long int nzones{1};
        };

        /// Combine links between the same pair of zones (keep the lowest) and drop links within a zone
        inline void reduce_zone_links(std::vector<ZoneLink> & links) {
            for (auto & l : links)
                if (l.a > l.b)
                    std::swap(l.a, l.b);
            links.erase(std::remove_if(links.begin(), links.end(), [](const ZoneLink & l) { return l.a == l.b; }),
                        links.end());
            std::sort(links.begin(), links.end(), [](const ZoneLink & x, const ZoneLink & y) {
                if (x.a != y.a)
                    return x.a < y.a;
                if (x.b != y.b)
                    return x.b < y.b;
                return x.quantity < y.quantity;
            });
            links.erase(std::unique(links.begin(),
                                    links.end(),
                                    [](const ZoneLink & x, const ZoneLink & y) { return x.a == y.a and x.b == y.b; }),
                        links.end());
        }

        //========================================================================================
        /// Merge zones into voids (the ZOBOV hierarchy). We go through the links from the lowest quantity up
        /// (union-find). When two voids join at a link, the shallower one (the one whose deepest zone has the
        /// larger quantity) is finished: it is the set of zones it had just before, its leak quantity is the link
        /// and its parent is the deeper one, which grows to include it. This is the same as growing a void from
        /// each zone by raising the "water level" until it reaches a zone deeper than its own (jozov in ZOBOV).
        ///
        /// Every zone gives one void (labeled by the zone), so the result has the same size as zones. With
        /// ZoneMerging::None every zone is a void (leak = its lowest link). With Threshold only links below
        /// threshold_quantity are used, and the leak of a top-level void is its lowest link to another void.
        ///
        /// Ties are broken by the position of the minimum (not the zone number) so the result does not depend on
        /// how the zones are numbered. Positions are in [0,1) (periodic). The barycenter of a merged void is computed
        /// relative to its minimum and is only meaningful for voids that extend less than half the box.
        ///
        /// @param[in] zones The zones
        /// @param[in] links The links between zones (any order, duplicates allowed)
        /// @param[in] merging How to merge
        /// @param[in] threshold_quantity The threshold for ZoneMerging::Threshold
        ///
        //========================================================================================
        inline std::vector<MergedVoid>
        merge_zones(const std::vector<Zone> & zones,
                    std::vector<ZoneLink> links,
                    ZoneMerging merging,
                    double threshold_quantity = std::numeric_limits<double>::infinity()) {
            const long long int nzones = (long long int)(zones.size());
            reduce_zone_links(links);

            // Order of the zones: by the quantity of the minimum, then its position
            auto deeper = [&](long long int a, long long int b) {
                const auto & za = zones[a];
                const auto & zb = zones[b];
                if (za.quantity_core != zb.quantity_core)
                    return za.quantity_core < zb.quantity_core;
                for (int j = 0; j < 3; j++)
                    if (za.pos_min[j] != zb.pos_min[j])
                        return za.pos_min[j] < zb.pos_min[j];
                return a < b;
            };

            std::vector<MergedVoid> result(nzones);
            for (long long int a = 0; a < nzones; a++) {
                result[a].sum = zones[a];
                result[a].root_zone = a;
            }

            if (merging == ZoneMerging::None) {
                for (auto & l : links) {
                    result[l.a].quantity_leak = std::min(result[l.a].quantity_leak, l.quantity);
                    result[l.b].quantity_leak = std::min(result[l.b].quantity_leak, l.quantity);
                }
                return result;
            }

            // Links in increasing order (ties: by the deeper and then the shallower zone of the link)
            std::sort(links.begin(), links.end(), [&](const ZoneLink & x, const ZoneLink & y) {
                if (x.quantity != y.quantity)
                    return x.quantity < y.quantity;
                const long long int xd = deeper(x.a, x.b) ? x.a : x.b, xs = xd == x.a ? x.b : x.a;
                const long long int yd = deeper(y.a, y.b) ? y.a : y.b, ys = yd == y.a ? y.b : y.a;
                if (xd != yd)
                    return deeper(xd, yd);
                if (xs != ys)
                    return deeper(xs, ys);
                return false;
            });

            // Union-find: the root of a set is its deepest zone, sums[root] is the sum of the set
            std::vector<long long int> uf(nzones);
            std::iota(uf.begin(), uf.end(), 0LL);
            auto find = [&](long long int a) {
                while (uf[a] != a) {
                    uf[a] = uf[uf[a]];
                    a = uf[a];
                }
                return a;
            };
            std::vector<Zone> sums(zones);
            std::vector<long long int> count(nzones, 1);

            size_t ilink = 0;
            for (; ilink < links.size(); ilink++) {
                const auto & l = links[ilink];
                if (merging == ZoneMerging::Threshold and not(l.quantity < threshold_quantity))
                    break;
                long long int ra = find(l.a), rb = find(l.b);
                if (ra == rb)
                    continue;
                const long long int d = deeper(ra, rb) ? ra : rb;
                const long long int s = d == ra ? rb : ra;

                // The shallow void is finished
                result[s].sum = sums[s];
                result[s].quantity_leak = l.quantity;
                result[s].parent_zone = d;
                result[s].nzones = count[s];

                // Add it to the deep one (moving its barycenter sum to be relative to the minimum of d)
                Zone & zd = sums[d];
                const Zone & zs = sums[s];
                for (int j = 0; j < 3; j++) {
                    double shift = zs.pos_min[j] - zd.pos_min[j];
                    shift -= std::round(shift);
                    zd.weighted_dx[j] += zs.weighted_dx[j] + zs.weight * shift;
                }
                zd.volume += zs.volume;
                zd.mass += zs.mass;
                zd.sum_density += zs.sum_density;
                zd.weight += zs.weight;
                zd.n += zs.n;
                count[d] += count[s];
                uf[s] = d;
            }

            // The top-level voids. Their leak is the lowest remaining link to another void
            for (long long int a = 0; a < nzones; a++) {
                if (find(a) != a)
                    continue;
                result[a].sum = sums[a];
                result[a].nzones = count[a];
                result[a].parent_zone = -1;
                result[a].quantity_leak = std::numeric_limits<double>::infinity();
            }
            for (; ilink < links.size(); ilink++) {
                const auto & l = links[ilink];
                const long long int ra = find(l.a), rb = find(l.b);
                if (ra == rb)
                    continue;
                result[ra].quantity_leak = std::min(result[ra].quantity_leak, l.quantity);
                result[rb].quantity_leak = std::min(result[rb].quantity_leak, l.quantity);
            }
            return result;
        }

    } // namespace VOIDFINDER
} // namespace FML

#endif
