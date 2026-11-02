#ifndef SIMCODE_INITIAL_DENSITY_FIELD_HEADER
#define SIMCODE_INITIAL_DENSITY_FIELD_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Global/Global.h>

#include <array>
#include <string>

namespace SIMCODE {

    //=============================================================================
    /// The initial CDM+baryon density field delta_cb(k, zini) of the particles on the force grid, for the sources
    /// that are added to the density field later in terms of it (linear massive neutrinos, the N-body gauge source).
    /// It is computed from the particles at the first force evaluation so that it has the same window as the density
    /// fields it is added to:
    ///   late    : with the force settings of the late-time steps (params.force)
    ///   early   : with the early-time force settings (force_early_method), only while the early-time force is used
    ///   shifted : the field in the frame of the particles when they are shifted relative to the grid (force
    ///             interlacing: delta_s(k) = exp(-ik.s) delta(k)), only during one density computation
    //=============================================================================
    template <int NDIM>
    class InitialDensityField {
      public:
        using FFTWGrid = FML::GRID::FFTWGrid<NDIM>;

        /// Is the field needed by the current step stored (the late one, and the early one during the early force)?
        bool stored(bool early_force) const { return late and (not early_force or early); }

        /// Store the field from density, the density field of the first step (computed with the current force
        /// settings). With the early-time force it is the early field and the late field is computed by
        /// assign_late(grid) (the particles to the allocated grid with the late-time settings)
        template <class F>
        void store(const FFTWGrid & density, bool early_force, int nmesh, int nleft, int nright, F && assign_late) {
            if (not early_force) {
                if (not late) {
                    late = density;
                    late.add_memory_label("density(k,zini)");
                }
                return;
            }
            early = density;
            early.add_memory_label("density(k,zini) early-time force");
            if (not late) {
                late = FFTWGrid(nmesh, nleft, nright);
                late.add_memory_label("density(k,zini)");
                assign_late(late);
            }
        }

        /// The field of the current step (with its density assignment during the early-time force), in the frame of
        /// the particles shifted by shift relative to the grid (computed once per density computation)
        const FFTWGrid & in_particle_frame(bool early_force, const std::array<double, NDIM> & shift) {
            const auto & delta_ini = early_force and early ? early : late;
            if (not is_shifted(shift))
                return delta_ini;
            if (not shifted) {
                shifted = delta_ini;
                FML::GRID::shift_grid_fourier<NDIM>(shifted, shift, -1.0);
            }
            return shifted;
        }

        /// The end of a density computation: free the shifted field
        void end_of_density_computation() { shifted.free(); }
        /// The early-time force is no longer used: free its field
        void end_of_early_force() { early.free(); }

        static bool is_shifted(const std::array<double, NDIM> & shift) {
            for (double s : shift)
                if (s != 0.0)
                    return true;
            return false;
        }

      private:
        FFTWGrid late;
        FFTWGrid early;
        FFTWGrid shifted;
    };

} // namespace SIMCODE

#endif
