#ifndef FML_PHASE_SPACE_HALO_TYPES_HEADER
#define FML_PHASE_SPACE_HALO_TYPES_HEADER
#include <FML/Global/Global.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace FML::PHASE_SPACE_HALOS {
    using ID = std::uint64_t;
    using Vector = std::array<double, 3>;
    inline constexpr ID no_halo = std::numeric_limits<ID>::max();
    struct Particle {
        ID id{0};
        Vector position{}, velocity{}; // Position in [0,L); peculiar velocity, not canonical momentum.
        double mass{1};
    };
    struct Options {
        double boxsize{1}, linking_length{.28}; // Linking length in global mean particle separation units.
        double retained_fraction{.7}, seed_significance{10}, minimum_peak_contrast{2};
        double softening_support{.001};     // Coordinate length, compact spline support h, not Plummer epsilon.
        double velocity_floor{1e-8};        // Velocity unit, for cold groups.
        double gravitational_constant{1};   // Velocity^2 * physical length / mass.
        double position_to_physical{1};     // a for comoving lengths, 1 for physical lengths.
        double hubble_velocity_gradient{0}; // Velocity / coordinate length; binding energies only.
        double assignment_density{0};       // Mass / coordinate length^3; 0 -> 200 times global mean mass density.
        double potential_opening_angle{.3}, minimum_bound_fraction{.5};
        std::size_t minimum_particles{20}, minimum_seed_particles{10};
        std::size_t max_parent_particles{1000000}, max_local_particles{10000000}, max_seeds{4096};
        std::uint64_t max_neighbor_visits{100000000};
        int max_refinement_depth{32}, max_unbinding_iterations{50}, max_label_iterations{10000}, saddle_samples{8};
        std::size_t max_message_bytes{8 * 1024 * 1024};
        bool unbind{true};
        void validate() const {
            auto positive = [](double x) { return std::isfinite(x) && x > 0; };
            if (!positive(boxsize) || !positive(linking_length) || !positive(softening_support) ||
                softening_support >= .25 * boxsize || !std::isfinite(1 / softening_support) ||
                !positive(velocity_floor) || !positive(gravitational_constant) || !positive(position_to_physical) ||
                !positive(gravitational_constant / position_to_physical) || !std::isfinite(hubble_velocity_gradient) ||
                !std::isfinite(assignment_density) || assignment_density < 0 || !std::isfinite(retained_fraction) ||
                retained_fraction < .5 || retained_fraction >= .95 || !std::isfinite(seed_significance) ||
                seed_significance < 0 || !std::isfinite(minimum_peak_contrast) || minimum_peak_contrast < 1 ||
                !std::isfinite(potential_opening_angle) || potential_opening_angle < 0 || potential_opening_angle > 1 ||
                !std::isfinite(minimum_bound_fraction) || minimum_bound_fraction < 0 || minimum_bound_fraction > 1 ||
                minimum_seed_particles < 2 || minimum_particles < minimum_seed_particles ||
                max_parent_particles < minimum_particles || max_local_particles < max_parent_particles ||
                max_seeds == 0 || max_neighbor_visits == 0 || max_refinement_depth < 1 || max_refinement_depth > 64 ||
                max_unbinding_iterations < 1 || max_label_iterations < 1 || saddle_samples < 2 || saddle_samples > 64 ||
                max_message_bytes < 128)
                throw std::invalid_argument("[PhaseSpaceHalos] Invalid options");
        }
    };
    struct Halo {
        ID id{no_halo}, fof_id{no_halo}, host{no_halo};
        Vector position{}, velocity{}, bulk_velocity{}, velocity_dispersion{};
        double bound_mass{0}, inclusive_bound_mass{0}, bound_radius{0}, vmax{0}, rmax{0}, bound_fraction{0};
        bool unbinding_converged{true}, binding_performed{false};
        std::vector<ID> particle_ids; // Exclusive surviving membership, sorted; descendants are not duplicated.
    };
    struct Assignment {
        ID particle{0}, halo{no_halo}, fof{no_halo};
    };
    struct Statistics {
        std::uint64_t parent_groups{0}, refinement_nodes{0}, depth_limited{0}, seed_merges{0}, saddle_merges{0},
            unbound_particles{0};
        std::uint64_t label_iterations{0}, ghost_particles{0}, routed_particles{0}, potential_pair_visits{0};
    };
    struct Catalogue {
        Options options;
        double resolved_assignment_density{0}, spatial_linking_radius{0};
        std::vector<Halo> halos;             // Local group-owner output in MPI mode, never a global catalogue gather.
        std::vector<Assignment> assignments; // Returned to each particle's input rank, sorted by particle ID.
        Statistics statistics;
    };
    namespace detail {
        inline double dot(const Vector & a, const Vector & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
        inline Vector difference(const Vector & a, const Vector & b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
        inline Vector periodic_difference(const Vector & a, const Vector & b, double box) {
            auto dx = difference(a, b);
            for (auto & x : dx) {
                if (x > .5 * box)
                    x -= box;
                else if (x < -.5 * box)
                    x += box;
            }
            return dx;
        }
        inline double wrap(double x, double box) {
            x = std::fmod(x, box);
            return x < 0 ? x + box : x;
        }
        inline void validate_particles(const std::vector<Particle> & p, const Options & o) {
            if (p.size() > o.max_local_particles)
                throw std::length_error("[PhaseSpaceHalos] Local particle budget exceeded");
            std::vector<ID> ids;
            ids.reserve(p.size());
            for (const auto & x : p) {
                if (x.id == no_halo || !std::isfinite(x.mass) || x.mass <= 0)
                    throw std::invalid_argument("[PhaseSpaceHalos] Invalid particle ID/mass");
                for (int d = 0; d < 3; ++d)
                    if (!std::isfinite(x.position[d]) || x.position[d] < 0 || x.position[d] >= o.boxsize ||
                        !std::isfinite(x.velocity[d]))
                        throw std::invalid_argument("[PhaseSpaceHalos] Invalid position or peculiar velocity");
                ids.push_back(x.id);
            }
            std::sort(ids.begin(), ids.end());
            if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
                throw std::invalid_argument("[PhaseSpaceHalos] Duplicate IDs");
        }
        struct DisjointSet {
            std::vector<std::size_t> root;
            explicit DisjointSet(std::size_t n) : root(n) {
                for (std::size_t i = 0; i < n; ++i)
                    root[i] = i;
            }
            std::size_t find(std::size_t i) {
                while (root[i] != i) {
                    root[i] = root[root[i]];
                    i = root[i];
                }
                return i;
            }
            void join(std::size_t a, std::size_t b) {
                a = find(a);
                b = find(b);
                if (a > b)
                    std::swap(a, b);
                root[b] = a;
            }
        };
    } // namespace detail
} // namespace FML::PHASE_SPACE_HALOS
#endif
