#ifndef FML_PHASE_SPACE_HALO_FINDER_HEADER
#define FML_PHASE_SPACE_HALO_FINDER_HEADER
#include <FML/KDTree/KDTree.h>
#include <FML/PhaseSpaceHaloFinder/BindingPotential.h>
#include <map>
#include <numeric>
#include <set>
namespace FML::PHASE_SPACE_HALOS {
    namespace detail {
        struct Summary {
            Vector x{}, v{};
            double mass{0}, sx{0}, sv{0}, effective_count{0};
        };
        inline Summary summarize(const std::vector<Particle> & p, const std::vector<std::size_t> & members) {
            Summary s;
            if (members.empty())
                return s;
            const auto & reference = p[members.front()];
            for (auto i : members) {
                s.mass += p[i].mass;
                for (int d = 0; d < 3; ++d) {
                    s.x[d] += p[i].mass * (p[i].position[d] - reference.position[d]);
                    s.v[d] += p[i].mass * (p[i].velocity[d] - reference.velocity[d]);
                }
            }
            if (!std::isfinite(s.mass) || s.mass <= 0)
                throw std::runtime_error("[PhaseSpaceHalos] Aggregate mass overflow");
            for (int d = 0; d < 3; ++d) {
                s.x[d] = reference.position[d] + s.x[d] / s.mass;
                s.v[d] = reference.velocity[d] + s.v[d] / s.mass;
            }
            for (auto i : members) {
                const auto dx = difference(p[i].position, s.x), dv = difference(p[i].velocity, s.v);
                s.sx += p[i].mass * dot(dx, dx);
                s.sv += p[i].mass * dot(dv, dv);
            }
            double squared_weights = 0;
            for (auto i : members) {
                const double weight = p[i].mass / s.mass;
                squared_weights += weight * weight;
            }
            s.effective_count = 1 / squared_weights;
            s.sx = std::sqrt(s.sx / s.mass);
            s.sv = std::sqrt(s.sv / s.mass);
            if (!std::isfinite(s.sx) || !std::isfinite(s.sv))
                throw std::runtime_error("[PhaseSpaceHalos] Phase-space dispersion overflow");
            return s;
        }
        inline std::pair<double, double> circular_velocity(const std::vector<Particle> & p,
                                                           const std::vector<std::size_t> & members,
                                                           const Vector & centre,
                                                           const Options & o,
                                                           double density = 0) {
            std::vector<std::pair<double, double>> radial;
            radial.reserve(members.size());
            for (auto i : members) {
                auto dx = difference(p[i].position, centre);
                radial.push_back({std::sqrt(dot(dx, dx)), p[i].mass});
            }
            std::sort(radial.begin(), radial.end());
            double mass = 0, vmax2 = 0, rmax = 0;
            for (std::size_t i = 0; i < radial.size();) {
                const auto r = radial[i].first;
                do {
                    mass += radial[i++].second;
                } while (i < radial.size() && radial[i].first == r);
                // density > 0: as ROCKSTAR (radii floored at the softening, inside the density threshold)
                const double re = density > 0 ? std::max(r, o.softening_support) : r;
                const bool use =
                    density > 0 ? mass > density * (4 * std::acos(-1.0) / 3) * re * re * re : r >= o.softening_support;
                if (use) {
                    const double v2 = o.gravitational_constant / o.position_to_physical * mass / re;
                    if (v2 > vmax2) {
                        vmax2 = v2;
                        rmax = re;
                    }
                }
            }
            return {std::sqrt(vmax2), rmax};
        }
        inline std::map<std::size_t, std::vector<std::size_t>> components(DisjointSet & sets, std::size_t n) {
            std::map<std::size_t, std::vector<std::size_t>> groups;
            for (std::size_t i = 0; i < n; ++i)
                groups[sets.find(i)].push_back(i);
            return groups;
        }
        inline void count_visits(std::uint64_t & visits, std::size_t n, const Options & o) {
            if (n > o.max_neighbor_visits - visits)
                throw std::length_error("[PhaseSpaceHalos] Neighbour-work budget exceeded");
            visits += n;
        }
        struct Seed {
            ID id{0};
            std::vector<std::size_t> members, core;
            Summary centre;
            double xscale{0}, vscale{0};
        };
        class ParentFinder {
            std::vector<Particle> p_;
            const Options & o_;
            double density_;
            Statistics & stats_;
            void update(Seed & s) {
                s.centre = summarize(p_, s.core);
                const auto all = summarize(p_, s.members);
                const auto circular = circular_velocity(p_, s.members, s.centre.x, o_);
                const double frequency = std::sqrt((4 * std::acos(-1.0) / 3) * o_.gravitational_constant * density_ /
                                                   o_.position_to_physical);
                s.xscale = std::max(o_.softening_support, circular.first / frequency);
                s.vscale = std::max(o_.velocity_floor, all.sv);
            }
            void merge_seeds(std::vector<Seed> & seeds,
                             const FML::KDTREE::KDTree & tree,
                             const Summary & mean,
                             double xs,
                             double vs) {
                // Poisson-consistent seeds: each seed merges into at most one LARGER seed (the most consistent one),
                // measured with its own core uncertainties. A symmetric union of all consistent pairs was transitive:
                // a small, poorly measured host fragment between a host and a subhalo is consistent with both and
                // merged a 800-particle subhalo (55 sigma from the host centre) into its host.
                DisjointSet set(seeds.size());
                const auto larger = [&](std::size_t a, std::size_t b) {
                    return seeds[a].core.size() != seeds[b].core.size() ? seeds[a].core.size() > seeds[b].core.size() :
                                                                          seeds[a].id < seeds[b].id;
                };
                for (std::size_t i = 0; i < seeds.size(); ++i) {
                    const auto & small = seeds[i];
                    const double n = small.centre.effective_count,
                                 xvar = std::pow(std::max(o_.softening_support, small.centre.sx), 2) / n,
                                 vvar = std::pow(std::max(o_.velocity_floor, small.centre.sv), 2) / n;
                    std::size_t target = seeds.size();
                    double best = 2 * o_.seed_significance * o_.seed_significance;
                    for (std::size_t j = 0; j < seeds.size(); ++j) {
                        if (j == i || !larger(j, i))
                            continue;
                        const auto dx = difference(seeds[i].centre.x, seeds[j].centre.x),
                                   dv = difference(seeds[i].centre.v, seeds[j].centre.v);
                        const double chi2 = dot(dx, dx) / xvar + dot(dv, dv) / vvar;
                        if (chi2 < best) {
                            best = chi2;
                            target = j;
                        }
                    }
                    if (target < seeds.size())
                        set.join(i, target);
                }
                // Attach each weak peak to at most one denser peak, reducing transitive merging through
                // a low-density bridge between two genuine peaks. A 6D kNN density is proportional to radius^-6.
                if (o_.minimum_peak_contrast > 1 && seeds.size() > 1) {
                    const std::size_t k =
                        std::min(tree.size(), std::max(std::size_t(20), 2 * o_.minimum_seed_particles));
                    std::vector<FML::KDTREE::point_t> centres;
                    std::vector<double> radius2;
                    for (const auto & seed : seeds) {
                        FML::KDTREE::point_t x(6);
                        for (int d = 0; d < 3; ++d) {
                            x[d] = (seed.centre.x[d] - mean.x[d]) / xs;
                            x[d + 3] = (seed.centre.v[d] - mean.v[d]) / vs;
                        }
                        std::vector<double> distances;
                        tree.knearest_indices(x, k, &distances);
                        centres.push_back(x);
                        radius2.push_back(std::max(1e-24, distances.back()));
                    }
                    for (std::size_t i = 0; i < seeds.size(); ++i) {
                        std::size_t nearest = seeds.size();
                        double best = std::numeric_limits<double>::infinity();
                        for (std::size_t j = 0; j < seeds.size(); ++j) {
                            if (!(radius2[j] < radius2[i] || (radius2[j] == radius2[i] && seeds[j].id < seeds[i].id)))
                                continue;
                            const double d = FML::KDTREE::dist2(centres[i], centres[j]);
                            if (d < best) {
                                best = d;
                                nearest = j;
                            }
                        }
                        if (nearest == seeds.size())
                            continue;
                        double saddle = radius2[i];
                        for (int step = 1; step < o_.saddle_samples; ++step) {
                            auto x = centres[i];
                            const double t = double(step) / o_.saddle_samples;
                            for (int d = 0; d < 6; ++d)
                                x[d] = (1 - t) * x[d] + t * centres[nearest][d];
                            std::vector<double> distances;
                            tree.knearest_indices(x, k, &distances);
                            saddle = std::max(saddle, distances.back());
                        }
                        if (3 * std::log(saddle / radius2[i]) < std::log(o_.minimum_peak_contrast) &&
                            set.find(i) != set.find(nearest)) {
                            set.join(i, nearest);
                            ++stats_.saddle_merges;
                        }
                    }
                }
                auto groups = components(set, seeds.size());
                std::vector<Seed> result;
                for (const auto & item : groups) {
                    Seed merged;
                    merged.id = no_halo;
                    for (auto i : item.second) {
                        merged.id = std::min(merged.id, seeds[i].id);
                        merged.members.insert(merged.members.end(), seeds[i].members.begin(), seeds[i].members.end());
                        merged.core.insert(merged.core.end(), seeds[i].core.begin(), seeds[i].core.end());
                    }
                    std::sort(merged.members.begin(), merged.members.end());
                    std::sort(merged.core.begin(), merged.core.end());
                    update(merged);
                    result.push_back(std::move(merged));
                }
                stats_.seed_merges += seeds.size() - result.size();
                seeds = std::move(result);
                std::sort(seeds.begin(), seeds.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
            }
            std::vector<Seed> refine(const std::vector<std::size_t> & members, int depth) {
                ++stats_.refinement_nodes;
                const auto leaf = [&]() {
                    Seed s;
                    s.id = p_[members.front()].id;
                    s.members = s.core = members;
                    update(s);
                    return std::vector<Seed>{std::move(s)};
                };
                if (members.size() <= o_.minimum_seed_particles)
                    return leaf();
                if (depth == o_.max_refinement_depth) {
                    ++stats_.depth_limited;
                    return leaf();
                }
                const auto mean = summarize(p_, members);
                const double xs = std::max(mean.sx, o_.softening_support), vs = std::max(mean.sv, o_.velocity_floor);
                FML::KDTREE::pointVec points;
                points.reserve(members.size());
                for (auto i : members) {
                    FML::KDTREE::point_t point(6);
                    for (int d = 0; d < 3; ++d) {
                        point[d] = (p_[i].position[d] - mean.x[d]) / xs;
                        point[d + 3] = (p_[i].velocity[d] - mean.v[d]) / vs;
                    }
                    points.push_back(std::move(point));
                }
                FML::KDTREE::KDTree tree(points);
                std::vector<double> distances;
                distances.reserve(members.size());
                for (std::size_t i = 0; i < members.size(); ++i) {
                    std::vector<double> r2;
                    const auto nearest = tree.knearest_indices(points[i], 2, &r2);
                    double distance = 0;
                    for (std::size_t j = 0; j < nearest.size(); ++j)
                        if (nearest[j] != i) {
                            distance = std::sqrt(r2[j]);
                            break;
                        }
                    distances.push_back(distance);
                }
                const auto quantile =
                    std::min(distances.size() - 1, std::size_t(std::ceil(o_.retained_fraction * distances.size()) - 1));
                std::nth_element(distances.begin(), distances.begin() + quantile, distances.end());
                const double radius = distances[quantile];
                DisjointSet sets(members.size());
                std::uint64_t visits = 0;
                for (std::size_t i = 0; i < members.size(); ++i) {
                    const auto neighbors = tree.neighborhood_indices(points[i], radius);
                    count_visits(visits, neighbors.size(), o_);
                    for (auto j : neighbors)
                        sets.join(i, j);
                }
                auto grouped = components(sets, members.size());
                std::vector<std::vector<std::size_t>> children;
                for (const auto & group : grouped)
                    if (group.second.size() >= o_.minimum_seed_particles) {
                        std::vector<std::size_t> child;
                        for (auto i : group.second)
                            child.push_back(members[i]);
                        children.push_back(std::move(child));
                    }
                if (children.empty() || (children.size() == 1 && children[0].size() == members.size()))
                    return leaf();
                std::vector<Seed> seeds;
                for (const auto & child : children) {
                    auto found = refine(child, depth + 1);
                    if (found.size() > o_.max_seeds - seeds.size())
                        throw std::length_error("[PhaseSpaceHalos] Seed budget exceeded");
                    seeds.insert(seeds.end(),
                                 std::make_move_iterator(found.begin()),
                                 std::make_move_iterator(found.end()));
                }
                merge_seeds(seeds, tree, mean, xs, vs);
                std::vector<bool> assigned(p_.size());
                for (const auto & s : seeds)
                    for (auto i : s.members)
                        assigned[i] = true;
                // Fixed seed metrics during each assignment pass make particle traversal order immaterial.
                for (auto i : members)
                    if (!assigned[i]) {
                        std::size_t best = 0;
                        double best_distance = std::numeric_limits<double>::infinity();
                        for (std::size_t j = 0; j < seeds.size(); ++j) {
                            const auto dx = difference(p_[i].position, seeds[j].centre.x),
                                       dv = difference(p_[i].velocity, seeds[j].centre.v);
                            const double d = dot(dx, dx) / (seeds[j].xscale * seeds[j].xscale) +
                                             dot(dv, dv) / (seeds[j].vscale * seeds[j].vscale);
                            if (d < best_distance) {
                                best_distance = d;
                                best = j;
                            }
                        }
                        seeds[best].members.push_back(i);
                    }
                for (auto & s : seeds) {
                    std::sort(s.members.begin(), s.members.end());
                    update(s);
                }
                return seeds;
            }
            std::vector<std::size_t> shrink_core(const std::vector<std::size_t> & members, Vector centre) const {
                auto core = members;
                auto best = core;
                const auto initial = summarize(p_, core);
                double best_error = initial.sx / std::sqrt(initial.effective_count);
                while (core.size() > o_.minimum_seed_particles) {
                    std::sort(core.begin(), core.end(), [&](auto a, auto b) {
                        const auto da = difference(p_[a].position, centre), db = difference(p_[b].position, centre);
                        const double a2 = dot(da, da), b2 = dot(db, db);
                        return a2 != b2 ? a2 < b2 : p_[a].id < p_[b].id;
                    });
                    const auto keep = std::max(o_.minimum_seed_particles, std::size_t(.7 * core.size()));
                    core.resize(keep);
                    const auto summary = summarize(p_, core);
                    centre = summary.x;
                    const double error = summary.sx / std::sqrt(summary.effective_count);
                    if (error < best_error) {
                        best_error = error;
                        best = core;
                    }
                }
                std::sort(best.begin(), best.end());
                return best;
            }
            Halo unbind(Seed seed, ID fof) {
                Halo halo;
                halo.id = seed.id;
                halo.binding_performed = o_.unbind;
                halo.fof_id = fof;
                const double original_mass = summarize(p_, seed.members).mass;
                bool converged = !o_.unbind;
                for (int iteration = 0; o_.unbind && iteration < o_.max_unbinding_iterations &&
                                        seed.members.size() >= o_.minimum_particles;
                     ++iteration) {
                    std::vector<Particle> current;
                    current.reserve(seed.members.size());
                    for (auto i : seed.members)
                        current.push_back(p_[i]);
                    const auto bulk = summarize(p_, seed.members);
                    std::uint64_t visits = 0;
                    const auto phi = BindingPotentialTree(current,
                                                          o_.softening_support,
                                                          o_.gravitational_constant / o_.position_to_physical,
                                                          o_.potential_opening_angle)
                                         .evaluate(&visits);
                    stats_.potential_pair_visits += visits;
                    std::vector<std::size_t> kept;
                    for (std::size_t j = 0; j < seed.members.size(); ++j) {
                        const auto i = seed.members[j];
                        auto dv = difference(p_[i].velocity, bulk.v);
                        const auto dx = difference(p_[i].position, bulk.x);
                        for (int d = 0; d < 3; ++d)
                            dv[d] += o_.hubble_velocity_gradient * dx[d];
                        const double energy = .5 * dot(dv, dv) + phi[j];
                        if (!std::isfinite(energy))
                            throw std::runtime_error("[PhaseSpaceHalos] Nonfinite binding energy");
                        if (energy < 0)
                            kept.push_back(i);
                    }
                    stats_.unbound_particles += seed.members.size() - kept.size();
                    if (kept.size() == seed.members.size()) {
                        converged = true;
                        break;
                    }
                    seed.members = std::move(kept);
                }
                if (seed.members.size() < o_.minimum_particles)
                    return halo;
                const auto bulk = summarize(p_, seed.members);
                halo.bound_fraction = bulk.mass / original_mass;
                if (halo.bound_fraction < o_.minimum_bound_fraction)
                    return halo;
                halo.unbinding_converged = converged;
                // An iteration-limited object is not advertised as bound: the caller must increase the limit.
                if (!converged)
                    throw std::runtime_error("[PhaseSpaceHalos] Unbinding did not converge");
                const auto core = shrink_core(seed.members, seed.centre.x);
                const auto center = summarize(p_, core);
                halo.position = center.x;
                halo.velocity = center.v;
                halo.bulk_velocity = bulk.v;
                halo.bound_mass = halo.inclusive_bound_mass = bulk.mass;
                for (auto i : seed.members) {
                    halo.particle_ids.push_back(p_[i].id);
                    const auto dx = difference(p_[i].position, center.x), dv = difference(p_[i].velocity, bulk.v);
                    halo.bound_radius = std::max(halo.bound_radius, std::sqrt(dot(dx, dx)));
                    for (int d = 0; d < 3; ++d)
                        halo.velocity_dispersion[d] += p_[i].mass * dv[d] * dv[d];
                }
                for (int d = 0; d < 3; ++d) {
                    halo.position[d] = wrap(halo.position[d], o_.boxsize);
                    halo.velocity_dispersion[d] = std::sqrt(halo.velocity_dispersion[d] / bulk.mass);
                }
                if (!o_.vmax_inclusive) {
                    const auto circular = circular_velocity(p_, seed.members, center.x, o_);
                    halo.vmax = circular.first;
                    halo.rmax = circular.second;
                }
                std::sort(halo.particle_ids.begin(), halo.particle_ids.end());
                return halo;
            }

          public:
            ParentFinder(std::vector<Particle> particles, const Options & options, double density, Statistics & stats)
                : p_(std::move(particles)), o_(options), density_(density), stats_(stats) {
                if (p_.size() > o_.max_parent_particles)
                    throw std::length_error("[PhaseSpaceHalos] Parent particle budget exceeded");
                std::sort(p_.begin(), p_.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
                const auto anchor = p_.front().position;
                Vector low{}, high{};
                for (auto & p : p_) {
                    p.position = periodic_difference(p.position, anchor, o_.boxsize);
                    for (int d = 0; d < 3; ++d) {
                        low[d] = std::min(low[d], p.position[d]);
                        high[d] = std::max(high[d], p.position[d]);
                    }
                }
                for (int d = 0; d < 3; ++d)
                    if (high[d] - low[d] >= .5 * o_.boxsize)
                        throw std::runtime_error("[PhaseSpaceHalos] Noncompact/percolating parent spans half the box");
                // Restore the anchor after analysis when wrapping halo positions back to the base box.
                anchor_ = anchor;
            }
            Vector anchor_{};
            std::vector<Halo> run(ID fof) {
                ++stats_.parent_groups;
                std::vector<std::size_t> all(p_.size());
                std::iota(all.begin(), all.end(), 0);
                const auto seeds = refine(all, 0);
                std::vector<Halo> halos;
                for (const auto & seed : seeds) {
                    auto halo = unbind(seed, fof);
                    if (halo.particle_ids.empty())
                        continue;
                    for (int d = 0; d < 3; ++d)
                        halo.position[d] = wrap(halo.position[d] + anchor_[d], o_.boxsize);
                    halos.push_back(std::move(halo));
                }
                for (auto & h : halos) {
                    double closest = std::numeric_limits<double>::infinity();
                    for (const auto & larger : halos) {
                        if (!(larger.bound_mass > h.bound_mass ||
                              (larger.bound_mass == h.bound_mass && larger.id < h.id)))
                            continue;
                        const auto dx = periodic_difference(h.position, larger.position, o_.boxsize);
                        const double radius = std::max(o_.softening_support, larger.bound_radius);
                        const double r2 = dot(dx, dx) / (radius * radius);
                        if (r2 < 1 && (r2 < closest || (r2 == closest && larger.id < h.host))) {
                            h.host = larger.id;
                            closest = r2;
                        }
                    }
                }
                // Strict mass/ID ordering prevents host cycles; accumulate each exclusive mass up its ancestor chain.
                std::map<ID, std::size_t> lookup;
                for (std::size_t i = 0; i < halos.size(); ++i)
                    lookup[halos[i].id] = i;
                for (const auto & h : halos) {
                    ID host = h.host;
                    while (host != no_halo) {
                        auto & parent = halos[lookup.at(host)];
                        parent.inclusive_bound_mass += h.bound_mass;
                        host = parent.host;
                    }
                }
                if (o_.vmax_inclusive) {
                    // Members of every object and all its descendants (p_ is sorted by ID)
                    std::vector<std::vector<std::size_t>> members(halos.size());
                    for (std::size_t k = 0; k < halos.size(); ++k)
                        for (auto id : halos[k].particle_ids) {
                            const auto it = std::lower_bound(p_.begin(), p_.end(), id, [](const auto & a, ID b) {
                                return a.id < b;
                            });
                            for (std::size_t j = k;; j = lookup.at(halos[j].host)) {
                                members[j].push_back(std::size_t(it - p_.begin()));
                                if (halos[j].host == no_halo)
                                    break;
                            }
                        }
                    for (std::size_t k = 0; k < halos.size(); ++k) {
                        const auto centre = periodic_difference(halos[k].position, anchor_, o_.boxsize);
                        const auto circular = circular_velocity(p_, members[k], centre, o_, density_);
                        halos[k].vmax = circular.first;
                        halos[k].rmax = circular.second;
                    }
                }
                return halos;
            }
        };
        inline std::vector<ID> spatial_labels(const std::vector<Particle> & p, double length, const Options & o) {
            FML::KDTREE::pointVec points;
            for (const auto & x : p)
                points.push_back({x.position[0], x.position[1], x.position[2]});
            FML::KDTREE::KDTree tree(points, o.boxsize);
            DisjointSet groups(p.size());
            std::uint64_t visits = 0;
            for (std::size_t i = 0; i < p.size(); ++i) {
                const auto neighbors = tree.neighborhood_indices(points[i], length);
                count_visits(visits, neighbors.size(), o);
                for (auto j : neighbors)
                    groups.join(i, j);
            }
            std::map<std::size_t, ID> minimum;
            for (std::size_t i = 0; i < p.size(); ++i) {
                const auto root = groups.find(i);
                auto it = minimum.emplace(root, p[i].id).first;
                it->second = std::min(it->second, p[i].id);
            }
            std::vector<ID> labels;
            for (std::size_t i = 0; i < p.size(); ++i)
                labels.push_back(minimum.at(groups.find(i)));
            return labels;
        }
        inline Catalogue
        analyze(std::vector<Particle> p, const std::vector<ID> & labels, const Options & o, double density) {
            Catalogue out;
            out.options = o;
            out.resolved_assignment_density = density;
            std::map<ID, std::vector<Particle>> parents;
            for (std::size_t i = 0; i < p.size(); ++i) {
                parents[labels[i]].push_back(p[i]);
                out.assignments.push_back({p[i].id, no_halo, labels[i]});
            }
            for (auto & parent : parents)
                if (parent.second.size() >= o.minimum_particles) {
                    auto found = ParentFinder(std::move(parent.second), o, density, out.statistics).run(parent.first);
                    out.halos.insert(out.halos.end(),
                                     std::make_move_iterator(found.begin()),
                                     std::make_move_iterator(found.end()));
                }
            std::sort(out.halos.begin(), out.halos.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
            std::map<ID, ID> membership;
            for (const auto & h : out.halos)
                for (auto id : h.particle_ids)
                    membership[id] = h.id;
            for (auto & assignment : out.assignments) {
                const auto found = membership.find(assignment.particle);
                if (found != membership.end())
                    assignment.halo = found->second;
            }
            std::sort(out.assignments.begin(), out.assignments.end(), [](const auto & a, const auto & b) {
                return a.particle < b.particle;
            });
            return out;
        }
    } // namespace detail
    /// Serial complete-catalogue finder, including in MPI builds. Does not reorder or modify caller input.
    inline Catalogue find(const std::vector<Particle> & particles, const Options & options = {}) {
        options.validate();
        detail::validate_particles(particles, options);
        if (particles.empty()) {
            Catalogue empty;
            empty.options = options;
            return empty;
        }
        auto sorted = particles;
        std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) { return a.id < b.id; });
        double mass = 0;
        for (const auto & p : sorted)
            mass += p.mass;
        const double density =
            options.assignment_density > 0 ? options.assignment_density : 200 * mass / std::pow(options.boxsize, 3);
        if (!std::isfinite(density) || density <= 0)
            throw std::invalid_argument("[PhaseSpaceHalos] Invalid mean density");
        const double length = options.linking_length * options.boxsize / std::cbrt(double(sorted.size()));
        if (length >= .5 * options.boxsize)
            throw std::invalid_argument("[PhaseSpaceHalos] Linking radius must be smaller than half the box");
        auto result = detail::analyze(sorted, detail::spatial_labels(sorted, length, options), options, density);
        result.spatial_linking_radius = length;
        return result;
    }
} // namespace FML::PHASE_SPACE_HALOS
#endif
