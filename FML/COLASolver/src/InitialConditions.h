#ifndef INITIALCONDITIONS_HEADER
#define INITIALCONDITIONS_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <memory>

//=============================================================================
// Generating the initial density field delta(k,zini) that the particles are
// displaced from (with LPT) to create the initial conditions
//=============================================================================

namespace COLASOLVER {

    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using RandomGenerator = FML::RANDOM::RandomGenerator;
    using GSLRandomGenerator = FML::RANDOM::GSLRandomGenerator;

    /// The random number generator to use for the initial conditions
    inline std::shared_ptr<RandomGenerator> make_random_generator(const ICSettings & ic) {
        std::shared_ptr<RandomGenerator> rng;
        if (ic.random_generator == RandomGeneratorType::GSL)
            rng = std::make_shared<GSLRandomGenerator>();
        else
            rng = std::make_shared<RandomGenerator>();
        rng->set_seed(ic.random_seed);
        return rng;
    }

    /// Apply func(kmag, value) -> new value to every mode of a grid in fourier space
    template <int NDIM>
    void transform_fourier_modes(
        FFTWGrid<NDIM> & grid,
        const std::function<std::complex<FML::GRID::FloatType>(double, std::complex<FML::GRID::FloatType>)> & func) {
        const auto Local_nx = grid.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            [[maybe_unused]] double kmag;
            [[maybe_unused]] std::array<double, NDIM> kvec;
            for (auto && fourier_index : grid.get_fourier_range(islice, islice + 1)) {
                grid.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                grid.set_fourier_from_index(fourier_index, func(kmag, grid.get_fourier_from_index(fourier_index)));
            }
        }
    }

    //=============================================================================
    /// Generate a gaussian or non-gaussian (non-local fNL) random field delta(k,zini)
    /// in fourier space with the power-spectrum power_initial_spline
    //=============================================================================
    template <int NDIM>
    void generate_initial_density_field(FFTWGrid<NDIM> & delta_ini_fourier,
                                        const ICSettings & ic,
                                        double boxsize,
                                        const Spline & power_initial_spline,
                                        const Spline & power_primordial_spline,
                                        std::shared_ptr<Cosmology> cosmo,
                                        std::shared_ptr<GravityModel<NDIM>> grav_ic,
                                        RandomGenerator * rng) {

        if (ic.random_field_type == ICRandomFieldType::Gaussian) {

            auto Pofk_of_kBox_over_volume = [&](double kBox) {
                return power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM);
            };
            FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(delta_ini_fourier,
                                                                                rng,
                                                                                Pofk_of_kBox_over_volume,
                                                                                ic.fix_amplitude);

        } else if (ic.random_field_type == ICRandomFieldType::NonGaussian) {

            // The power-spectrum of the Bardeen potential (-Phi) at the given redshift
            const double afnl = 1.0 / (1.0 + ic.fnl_redshift);
            const double H0Box = boxsize * grav_ic->H0_hmpc;
            const double factor = 1.5 * cosmo->get_OmegaM() / afnl;
            auto Pofk_of_kBox_over_volume_primordial = [&](double kBox) {
                return power_primordial_spline(kBox / boxsize) / std::pow(boxsize, NDIM) *
                       std::pow(factor * grav_ic->get_D_1LPT(afnl, kBox / H0Box) /
                                    grav_ic->get_D_1LPT(1.0, kBox / H0Box),
                                2);
            };
            // The product of this and the function above is the initial power-spectrum for delta
            auto Pofk_of_kBox_over_Pofk_primordal = [&](double kBox) {
                return power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM) /
                       Pofk_of_kBox_over_volume_primordial(kBox);
            };
            FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier_cosmology(
                delta_ini_fourier,
                rng,
                Pofk_of_kBox_over_Pofk_primordal,
                Pofk_of_kBox_over_volume_primordial,
                ic.fix_amplitude,
                ic.fnl,
                ic.fnl_type);
        }
    }

    /// Keep the phases of delta(k), but set the amplitudes to sqrt(P(k,zini)): |delta(k)| -> sqrt(P(k))
    template <int NDIM>
    void
    whiten_and_set_power_spectrum(FFTWGrid<NDIM> & delta_fourier, const Spline & power_initial_spline, double boxsize) {
        auto sqrt_pofk = [&](double kBox) {
            return std::sqrt(power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM));
        };
        transform_fourier_modes<NDIM>(delta_fourier, [&](double kmag, std::complex<FML::GRID::FloatType> value) {
            // Whiten the field
            auto absvalue = std::abs(value);
            if (absvalue > 0.0)
                value /= absvalue;
            // Multiply by power-spectrum
            value *= sqrt_pofk(kmag);
            return value;
        });
    }

    /// delta(k) -> -delta(k) (for paired simulations)
    template <int NDIM>
    void reverse_phases(FFTWGrid<NDIM> & delta_fourier) {
        transform_fourier_modes<NDIM>(delta_fourier, [](double, std::complex<FML::GRID::FloatType> value) {
            return FML::GRID::FloatType(-1.0) * value;
        });
    }

    /// The 1LPT potential phi_1LPT(k) = delta(k) / k^2  (i.e. D^2 phi_1LPT = -delta)
    template <int NDIM>
    FFTWGrid<NDIM> compute_phi_1LPT_from_delta(const FFTWGrid<NDIM> & delta_fourier) {
        FFTWGrid<NDIM> phi_1LPT_fourier = delta_fourier;
        auto Local_x_start = phi_1LPT_fourier.get_local_x_start();
        auto Local_nx = phi_1LPT_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            double kmag2;
            std::array<double, NDIM> kvec;
            for (auto && fourier_index : phi_1LPT_fourier.get_fourier_range(islice, islice + 1)) {
                if (Local_x_start == 0 and fourier_index == 0)
                    continue; // DC mode k = 0
                phi_1LPT_fourier.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                auto value = phi_1LPT_fourier.get_fourier_from_index(fourier_index) / FML::GRID::FloatType(kmag2);
                phi_1LPT_fourier.set_fourier_from_index(fourier_index, value);
            }
        }
        // Deal with DC mode
        if (Local_x_start == 0)
            phi_1LPT_fourier.set_fourier_from_index(0, 0.0);
        return phi_1LPT_fourier;
    }

} // namespace COLASOLVER

#endif
