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
            FML::KDTREE::pointVec centers;
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
                centers.emplace_back(halo.center.begin(), halo.center.end());
            }
            tree = FML::KDTREE::KDTree(centers, periodic ? boxsize : 0.0);
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
            auto neighbors = tree.neighborhood_indices({position.begin(), position.end()}, max_support);
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
            detail::require(count == 0 || particles != nullptr, "Baryonifier: null particle array");
            // Validate before entering OpenMP; ordinary input errors must remain catchable exceptions.
            for (size_t i = 0; i < count; ++i) {
                detail::require(particles[i].get_ndim() == 3, "Baryonifier: particles must be three-dimensional");
                for (size_t d = 0; d < 3; ++d)
                    detail::require(std::isfinite(particles[i].get_pos()[d]),
                                    "Baryonifier: non-finite particle position");
            }
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
            const auto shifts = displacements(particles, count);
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < count; ++i) {
                auto * x = particles[i].get_pos();
                using Position = std::remove_reference_t<decltype(x[0])>;
                for (size_t d = 0; d < 3; ++d) {
                    const double moved = double(x[d]) + shifts[i][d];
                    x[d] = Position(periodic ? FML::KDTREE::periodic_wrap(moved, boxsize) : moved);
                    // Casting to float can round a wrapped position back to boxsize.
                    if (periodic && double(x[d]) >= boxsize)
                        x[d] = Position(0);
                }
            }
        }

      private:
        std::vector<HaloDisplacement> halos;
        double boxsize, max_support = 0.0;
        bool periodic;
        FML::KDTREE::KDTree tree;
    };

} // namespace FML::BARYONIFICATION
#endif
