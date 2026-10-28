#ifndef FML_HALO_BINDING_POTENTIAL_HEADER
#define FML_HALO_BINDING_POTENTIAL_HEADER
#include <FML/PhaseSpaceHaloFinder/Types.h>
#include <FML/ShortRangeGravity/Softening.h>
#include <numeric>
namespace FML::PHASE_SPACE_HALOS {
    /// Isolated unsplit softened Newtonian potential. Sources must be consistently unwrapped and remain unchanged.
    class BindingPotentialTree {
        struct Node {
            Vector low{}, high{}, centre{};
            double mass{0};
            std::size_t begin{0}, end{0}, left{0}, right{0};
        };
        const std::vector<Particle> & p_;
        double h_, coupling_, theta_;
        std::vector<Node> nodes_;
        std::vector<std::size_t> order_, inverse_;
        std::size_t build(std::size_t begin, std::size_t end) {
            const auto index = nodes_.size();
            nodes_.emplace_back();
            Node node;
            node.begin = begin;
            node.end = end;
            node.low = node.high = p_[order_[begin]].position;
            for (auto i = begin; i < end; ++i) {
                const auto & p = p_[order_[i]];
                node.mass += p.mass;
                for (int d = 0; d < 3; ++d) {
                    node.centre[d] += p.mass * p.position[d];
                    node.low[d] = std::min(node.low[d], p.position[d]);
                    node.high[d] = std::max(node.high[d], p.position[d]);
                }
            }
            if (!std::isfinite(node.mass) || node.mass <= 0)
                throw std::invalid_argument("[BindingPotential] Invalid aggregate mass");
            for (auto & x : node.centre) {
                x /= node.mass;
                if (!std::isfinite(x))
                    throw std::invalid_argument("[BindingPotential] Centre overflow");
            }
            double extent2 = 0;
            for (int d = 0; d < 3; ++d) {
                const double side = node.high[d] - node.low[d];
                extent2 += side * side;
            }
            if (!std::isfinite(extent2) || !std::isfinite(coupling_ * node.mass / h_))
                throw std::invalid_argument("[BindingPotential] Length or potential outside representable range");
            if (end - begin > 16) {
                int axis = 0;
                for (int d = 1; d < 3; ++d)
                    if (node.high[d] - node.low[d] > node.high[axis] - node.low[axis])
                        axis = d;
                std::stable_sort(order_.begin() + begin, order_.begin() + end, [&](auto a, auto b) {
                    return p_[a].position[axis] != p_[b].position[axis] ? p_[a].position[axis] < p_[b].position[axis] :
                                                                          p_[a].id < p_[b].id;
                });
                const auto mid = begin + (end - begin) / 2;
                node.left = build(begin, mid);
                node.right = build(mid, end);
            }
            nodes_[index] = node;
            return index;
        }
        double walk(std::size_t node_index, std::size_t target, std::uint64_t & visits) const {
            const auto & n = nodes_[node_index];
            const auto dx = detail::difference(p_[target].position, n.centre);
            const double r2 = detail::dot(dx, dx);
            double size2 = 0, min2 = 0;
            for (int d = 0; d < 3; ++d) {
                const double side = n.high[d] - n.low[d];
                size2 += side * side;
                const double distance =
                    std::max({n.low[d] - p_[target].position[d], 0., p_[target].position[d] - n.high[d]});
                min2 += distance * distance;
            }
            const bool contains = inverse_[target] >= n.begin && inverse_[target] < n.end;
            if (n.left && !contains && min2 >= h_ * h_ && size2 < theta_ * theta_ * r2) {
                ++visits;
                return -coupling_ * n.mass / std::sqrt(r2);
            }
            if (n.left)
                return walk(n.left, target, visits) + walk(n.right, target, visits);
            double phi = 0;
            for (auto i = n.begin; i < n.end; ++i) {
                const auto source = order_[i];
                if (source == target)
                    continue;
                const auto delta = detail::difference(p_[target].position, p_[source].position);
                ++visits;
                phi -= coupling_ * p_[source].mass *
                       FML::SHORT_RANGE_GRAVITY::newtonian_spline_green(std::sqrt(detail::dot(delta, delta)), h_);
            }
            return phi;
        }

      public:
        BindingPotentialTree(const std::vector<Particle> & p,
                             double softening,
                             double coupling,
                             double opening_angle = .3)
            : p_(p), h_(softening), coupling_(coupling), theta_(opening_angle) {
            if (!std::isfinite(h_) || h_ <= 0 || !std::isfinite(coupling_) || coupling_ <= 0 ||
                !std::isfinite(theta_) || theta_ < 0 || theta_ > 1)
                throw std::invalid_argument("[BindingPotential] Invalid softening, coupling or opening angle");
            for (const auto & x : p) {
                if (!std::isfinite(x.mass) || x.mass <= 0)
                    throw std::invalid_argument("[BindingPotential] Invalid mass");
                for (double v : x.position)
                    if (!std::isfinite(v))
                        throw std::invalid_argument("[BindingPotential] Invalid position");
            }
            order_.resize(p.size());
            inverse_.resize(p.size());
            std::iota(order_.begin(), order_.end(), 0);
            if (!p.empty())
                build(0, p.size());
            for (std::size_t i = 0; i < p.size(); ++i)
                inverse_[order_[i]] = i;
        }
        BindingPotentialTree(std::vector<Particle> &&, double, double, double = .3) = delete;
        std::vector<double> evaluate(std::uint64_t * interaction_visits = nullptr) const {
            std::vector<double> phi(p_.size());
            std::uint64_t visits = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : visits)
#endif
            for (std::size_t i = 0; i < p_.size(); ++i)
                phi[i] = walk(0, i, visits);
            if (interaction_visits)
                *interaction_visits = visits;
            return phi;
        }
    };
} // namespace FML::PHASE_SPACE_HALOS
#endif
