#ifndef FML_BARYONIFICATION_H
#define FML_BARYONIFICATION_H

#include "Profiles.h"
#include <FML/KDTree/KDTree.h>
#include <array>
#include <type_traits>

namespace FML::BARYONIFICATION {

    struct HaloDisplacement {
        std::array<double, 3> center;
        RadialMapping mapping;
    };

    /// A read-only displacement field for one snapshot. Halo contributions are evaluated at the ORIGINAL
    /// positions, added, and applied once. Overlap therefore does not depend on halo processing order except
    /// for floating-point summation. This superposition is approximate, not an exact multi-halo mass transport.
    ///
    /// All relevant halos (including halos across MPI domain boundaries) must be supplied on every rank.
    /// No MPI collectives are performed. After apply(), call MPIParticles::communicate_particles() if needed.
    /// A periodic halo's support must be smaller than half the box: only its nearest image contributes.
    class Baryonifier {
      public:
        Baryonifier(std::vector<HaloDisplacement> halos, double boxsize, bool periodic = true)
            : halos(std::move(halos)), boxsize(boxsize), periodic(periodic) {
            detail::require(detail::positive(boxsize), "Baryonifier: boxsize must be finite and positive");
            for (auto & halo : this->halos) {
                for (double & x : halo.center) {
                    detail::require(std::isfinite(x), "Baryonifier: non-finite halo center");
                    if (periodic)
                        x = FML::KDTREE::periodic_wrap(x, boxsize);
                }
                const double support = halo.mapping.maximum_radius();
                detail::require(!periodic || support < 0.5 * boxsize,
                                "Baryonifier: periodic support must be smaller than half the box");
                max_support = std::max(max_support, support);
            }
            // The halos are grouped by their support (within a factor of 2 in each group) with one tree per group,
            // queried with the largest support of the group: querying all halos with the largest support of the
            // catalogue returned ~18x more halos than contribute for a realistic mass function
            std::vector<std::vector<size_t>> members;
            for (size_t i = 0; i < this->halos.size(); ++i) {
                const double support = this->halos[i].mapping.maximum_radius();
                const int bin = std::min(max_support_bins - 1, int(std::floor(std::log2(max_support / support))));
                if (int(members.size()) <= bin)
                    members.resize(bin + 1);
                members[bin].push_back(i);
            }
            for (auto & indices : members) {
                if (indices.empty())
                    continue;
                SupportGroup group;
                FML::KDTREE::pointVec centers;
                for (size_t i : indices) {
                    group.support = std::max(group.support, this->halos[i].mapping.maximum_radius());
                    centers.emplace_back(this->halos[i].center.begin(), this->halos[i].center.end());
                }
                group.halo_index = std::move(indices);
                group.tree = FML::KDTREE::KDTree(centers, periodic ? boxsize : 0.0);
                groups.push_back(std::move(group));
            }
        }

        std::array<double, 3> displacement_at(std::array<double, 3> position) const {
            for (double & x : position) {
                detail::require(std::isfinite(x), "Baryonifier: non-finite particle position");
                if (periodic)
                    x = FML::KDTREE::periodic_wrap(x, boxsize);
            }
            std::array<double, 3> result{};
            if (halos.empty())
                return result;
            std::vector<size_t> neighbors;
            const FML::KDTREE::point_t point(position.begin(), position.end());
            for (const auto & group : groups)
                for (size_t local : group.tree.neighborhood_indices(point, group.support))
                    neighbors.push_back(group.halo_index[local]);
            std::sort(neighbors.begin(), neighbors.end()); // stable summation for repeated/threaded queries
            for (size_t index : neighbors) {
                const auto & halo = halos[index];
                std::array<double, 3> dx;
                double r2 = 0.0;
                for (size_t d = 0; d < 3; ++d) {
                    dx[d] = position[d] - halo.center[d];
                    if (periodic)
                        dx[d] -= boxsize * std::round(dx[d] / boxsize);
                    r2 += dx[d] * dx[d];
                }
                const double r = std::sqrt(r2);
                if (r == 0.0 || r >= halo.mapping.maximum_radius())
                    continue;
                const double factor = halo.mapping.displacement(r) / r;
                for (size_t d = 0; d < 3; ++d)
                    result[d] += factor * dx[d];
            }
            return result;
        }

        /// Compute a separate displacement array without altering the snapshot. Particle types need
        /// get_pos() and get_ndim(), as in the rest of FML. This module is specifically three-dimensional.
        template <class T>
        std::vector<std::array<double, 3>> displacements(T * particles, size_t count) const {
            validate_particles(particles, count);
            std::vector<std::array<double, 3>> shifts(count);
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < count; ++i) {
                const auto * x = particles[i].get_pos();
                shifts[i] = displacement_at({double(x[0]), double(x[1]), double(x[2])});
            }
            return shifts;
        }

        /// Position-only post-processing. IDs, masses, velocities, and all other particle fields are untouched.
        template <class T>
        void apply(T * particles, size_t count) const {
            validate_particles(particles, count);
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < count; ++i) {
                auto * x = particles[i].get_pos();
                // Complete the vector displacement at the original position before
                // modifying any component. The field is independent of other particles.
                const auto shift = displacement_at({double(x[0]), double(x[1]), double(x[2])});
                using Position = std::remove_reference_t<decltype(x[0])>;
                for (size_t d = 0; d < 3; ++d) {
                    const double moved = double(x[d]) + shift[d];
                    x[d] = Position(periodic ? FML::KDTREE::periodic_wrap(moved, boxsize) : moved);
                    // Casting to float can round a wrapped position back to boxsize.
                    if (periodic && double(x[d]) >= boxsize)
                        x[d] = Position(0);
                }
            }
        }

      private:
        template <class T>
        static void validate_particles(T * particles, size_t count) {
            detail::require(count == 0 || particles != nullptr, "Baryonifier: null particle array");
            // Validate the whole input before OpenMP or any in-place changes.
            for (size_t i = 0; i < count; ++i) {
                detail::require(particles[i].get_ndim() == 3, "Baryonifier: particles must be three-dimensional");
                for (size_t d = 0; d < 3; ++d)
                    detail::require(std::isfinite(particles[i].get_pos()[d]),
                                    "Baryonifier: non-finite particle position");
            }
        }

        std::vector<HaloDisplacement> halos;
        double boxsize, max_support = 0.0;
        bool periodic;
        // Halos with support in (max_support / 2^(b+1), max_support / 2^b] (the last group: all smaller ones)
        struct SupportGroup {
            double support = 0.0;
            std::vector<size_t> halo_index; // index in halos of the tree's points
            FML::KDTREE::KDTree tree;
        };
        static constexpr int max_support_bins = 16;
        std::vector<SupportGroup> groups;
    };

} // namespace FML::BARYONIFICATION
#endif
