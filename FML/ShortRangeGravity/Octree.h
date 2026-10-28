#ifndef FML_SHORT_RANGE_OCTREE_HEADER
#define FML_SHORT_RANGE_OCTREE_HEADER
#include <FML/ShortRangeGravity/ShortRangeGravity.h>

namespace FML::SHORT_RANGE_GRAVITY {
    struct TreeOptions {
        double opening_angle{0.3}; // 0: exact traversal; geometric criterion, not a guaranteed relative force error.
        std::size_t leaf_capacity{16};
        int max_depth{40};
    };
    struct TreeResult : Result {
        std::uint64_t accepted_nodes{0}, visited_nodes{0};
        std::size_t tree_nodes{0};
    };

    /// Flat monopole octree. Source vector must outlive the tree and remain unchanged. No quadrupole approximation.
    class Octree {
        struct Node {
            Vector centre{}, centre_of_mass{};
            double halfwidth{0}, mass{0};
            std::size_t begin{0}, end{0};
            std::array<std::size_t, 8> children{}; // 0 denotes absent; root is never a child.
            bool leaf{true};
        };
        const std::vector<Particle> & sources_;
        Kernel kernel_;
        TreeOptions options_;
        std::vector<Node> nodes_;
        std::vector<std::size_t> indices_, scratch_;
        std::vector<std::pair<std::uint64_t, std::size_t>> identity_order_;

        std::size_t build(std::size_t begin, std::size_t end, Vector centre, double halfwidth, int depth) {
            const auto node_index = nodes_.size();
            nodes_.emplace_back();
            Node node;
            node.centre = centre;
            node.halfwidth = halfwidth;
            node.begin = begin;
            node.end = end;
            for (auto i = begin; i < end; ++i) {
                const auto & p = sources_[indices_[i]];
                node.mass += p.mass;
                for (int d = 0; d < 3; ++d)
                    node.centre_of_mass[d] += p.mass * p.position[d];
            }
            if (!std::isfinite(node.mass))
                throw std::invalid_argument("[ShortRangeGravity] Tree mass overflow");
            if (node.mass > 0)
                for (auto & x : node.centre_of_mass)
                    x /= node.mass;
            else
                node.centre_of_mass = centre;
            if (end - begin > options_.leaf_capacity && depth < options_.max_depth) {
                const auto octant = [&](const Particle & p) {
                    int index = 0;
                    for (int d = 0; d < 3; ++d)
                        index |= int(p.position[d] >= centre[d]) << d;
                    return index;
                };
                std::array<std::size_t, 9> offsets{};
                for (auto i = begin; i < end; ++i)
                    ++offsets[octant(sources_[indices_[i]]) + 1];
                std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
                auto cursor = offsets;
                for (auto i = begin; i < end; ++i)
                    scratch_[begin + cursor[octant(sources_[indices_[i]])]++] = indices_[i];
                std::copy(scratch_.begin() + begin, scratch_.begin() + end, indices_.begin() + begin);
                node.leaf = false;
                for (int child = 0; child < 8; ++child) {
                    if (offsets[child] == offsets[child + 1])
                        continue;
                    auto child_centre = centre;
                    for (int d = 0; d < 3; ++d)
                        child_centre[d] += ((child >> d) & 1 ? .5 : -.5) * halfwidth;
                    node.children[child] = build(begin + offsets[child],
                                                 begin + offsets[child + 1],
                                                 child_centre,
                                                 halfwidth * .5,
                                                 depth + 1);
                }
            }
            nodes_[node_index] = node; // Recursion can reallocate nodes_.
            return node_index;
        }
        void walk(std::size_t index,
                  std::size_t excluded_index,
                  const Particle & target,
                  Vector & acceleration,
                  std::uint64_t & candidates,
                  std::uint64_t & accepted,
                  std::uint64_t & visited,
                  std::uint64_t & approximated) const {
            ++visited;
            const auto & node = nodes_[index];
            if (node.mass == 0)
                return;
            const double box = kernel_.options().boxsize;
            const auto dx = detail::displacement(target.position, node.centre, box);
            double min2 = 0, max2 = 0;
            bool contains = true, single_image = true;
            for (int d = 0; d < 3; ++d) {
                const double near = std::max(0.0, std::abs(dx[d]) - node.halfwidth);
                const double far = std::abs(dx[d]) + node.halfwidth;
                min2 += near * near;
                max2 += far * far;
                contains &= std::abs(target.position[d] - node.centre[d]) <= node.halfwidth;
                single_image &= far < .5 * box;
            }
            const double cutoff2 = kernel_.cutoff() * kernel_.cutoff();
            if (min2 >= cutoff2)
                return;
            const auto com_dx = detail::displacement(target.position, node.centre_of_mass, box);
            const double r2 = com_dx[0] * com_dx[0] + com_dx[1] * com_dx[1] + com_dx[2] * com_dx[2];
            // Never approximate a node containing the target, crossing a periodic image/cutoff, or touching softening.
            // Also resolve the Gaussian variation: a Newtonian opening angle alone is insufficient near the cutoff.
            const double diameter = 2 * std::sqrt(3.0) * node.halfwidth;
            if (!node.leaf && !(excluded_index >= node.begin && excluded_index < node.end) && !contains &&
                single_image && max2 < cutoff2 &&
                min2 >= kernel_.options().softening_support * kernel_.options().softening_support &&
                diameter < options_.opening_angle * std::sqrt(r2) &&
                diameter < options_.opening_angle * kernel_.options().split_scale) {
                const double factor = node.mass * kernel_.coefficient(std::sqrt(r2));
                for (int d = 0; d < 3; ++d)
                    acceleration[d] += factor * com_dx[d];
                ++approximated;
                return;
            }
            if (node.leaf) {
                for (auto i = node.begin; i < node.end; ++i) {
                    const auto & source = sources_[indices_[i]];
                    if (target.id == source.id)
                        continue;
                    ++candidates;
                    accepted += detail::add_pair(acceleration, target, source, kernel_);
                }
            } else {
                for (auto child : node.children)
                    if (child)
                        walk(child, excluded_index, target, acceleration, candidates, accepted, visited, approximated);
            }
        }

      public:
        Octree(const std::vector<Particle> & sources, const Kernel & kernel, TreeOptions options = {})
            : sources_(sources), kernel_(kernel), options_(options) {
            detail::validate_particles(sources, kernel.options().boxsize, true);
            if (!std::isfinite(options.opening_angle) || options.opening_angle < 0 || options.opening_angle > 1 ||
                options.leaf_capacity == 0 || options.max_depth < 1 || options.max_depth > 60)
                throw std::invalid_argument("[ShortRangeGravity] Invalid tree options");
            indices_.resize(sources.size());
            scratch_.resize(sources.size());
            std::iota(indices_.begin(), indices_.end(), 0);
            const double half = kernel.options().boxsize * .5;
            build(0, sources.size(), {half, half, half}, half, 0);
            for (std::size_t i = 0; i < indices_.size(); ++i)
                identity_order_.emplace_back(sources_[indices_[i]].id, i);
            std::sort(identity_order_.begin(), identity_order_.end());
            scratch_.clear();
            scratch_.shrink_to_fit();
        }
        Octree(std::vector<Particle> &&, const Kernel &, TreeOptions = {}) = delete;
        TreeResult compute(const std::vector<Particle> & targets) const {
            detail::validate_particles(targets, kernel_.options().boxsize, false);
            TreeResult result;
            result.acceleration.resize(targets.size());
            result.tree_nodes = nodes_.size();
            std::uint64_t candidates = 0, accepted = 0, visited = 0, approximated = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : candidates, accepted, visited, approximated)
#endif
            for (std::size_t i = 0; i < targets.size(); ++i) {
                const auto match = std::lower_bound(identity_order_.begin(),
                                                    identity_order_.end(),
                                                    std::pair<std::uint64_t, std::size_t>{targets[i].id, 0});
                const auto excluded =
                    match != identity_order_.end() && match->first == targets[i].id ? match->second : sources_.size();
                walk(0, excluded, targets[i], result.acceleration[i], candidates, accepted, visited, approximated);
            }
            result.statistics.candidate_pairs = candidates;
            result.statistics.accepted_pairs = accepted;
            result.visited_nodes = visited;
            result.accepted_nodes = approximated;
            return result;
        }
    };
} // namespace FML::SHORT_RANGE_GRAVITY
#endif
