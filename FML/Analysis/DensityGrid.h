#ifndef FML_ANALYSIS_DENSITYGRID_H
#define FML_ANALYSIS_DENSITYGRID_H

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/Smoothing/SmoothingFourier.h>

#include <cmath>
#include <stdexcept>
#include <string>

//=============================================================================
// The density contrast of particles on a grid (mass weighted if the particles have masses) and Gaussian smoothing:
// the input of the cosmic-web classification and the Minkowski functionals
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        /// The density contrast delta of the particles on an nmesh^3 grid (real space). Collective
        template <class P>
        FML::GRID::FFTWGrid<3> density_contrast_grid(P * part, size_t npart, int nmesh, const std::string & method) {
            size_t ntot = npart;
            FML::SumOverTasks(&ntot);
            if (ntot == 0)
                throw std::runtime_error("density_contrast_grid: no particles");
            const auto nextra = FML::INTERPOLATION::get_extra_slices(method);
            FML::GRID::FFTWGrid<3> delta(nmesh, nextra.first, nextra.second);
            FML::INTERPOLATION::particles_to_grid(part, npart, ntot, delta, method);
            return delta;
        }

        /// Gaussian smoothing exp(-k^2 R^2 / 2) of a real-space grid in place (R in units of the box; R <= 0: none)
        inline void gaussian_smooth(FML::GRID::FFTWGrid<3> & grid, double R_box) {
            if (not(R_box > 0.0))
                return;
            grid.fftw_r2c();
            FML::GRID::multiply_fourier_by_function_of_k<3>(grid,
                                                            [R_box](double k) { return std::exp(-0.5 * k * k * R_box * R_box); });
            grid.fftw_c2r();
        }

        namespace detail {
            inline void validate_grid_settings(const std::string & prefix,
                                               int nmesh,
                                               double smoothing_scale_mpch,
                                               const std::string & method) {
                if (nmesh < 2 or nmesh % FML::NTasks != 0)
                    throw std::runtime_error(prefix + "nmesh must be >= 2 and divisible by the number of MPI tasks");
                if (not std::isfinite(smoothing_scale_mpch) or smoothing_scale_mpch < 0.0)
                    throw std::runtime_error(prefix + "smoothing_scale_mpch must be >= 0");
                if (method != "NGP" and method != "CIC" and method != "TSC" and method != "PCS" and method != "PQS")
                    throw std::runtime_error(prefix + "density_assignment_method must be NGP, CIC, TSC, PCS or PQS");
            }
        } // namespace detail

    } // namespace ANALYSIS
} // namespace FML
#endif
