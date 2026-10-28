#ifndef INITIALCONDITIONS_HEADER
#define INITIALCONDITIONS_HEADER
#include <FML/InitialConditions/SpeciesIC.h>

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <memory>
#include <string>
#include <vector>

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

    // The library helpers for loops over the fourier modes
    using FML::GRID::transform_fourier_modes;

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

        FML::RANDOM::GAUSSIAN::RandomFieldOptions random_options;
        random_options.backend = ic.random_field_backend == RandomFieldBackend::Philox ?
                                     FML::RANDOM::GAUSSIAN::RandomFieldBackend::Philox :
                                     FML::RANDOM::GAUSSIAN::RandomFieldBackend::NGenIC;
        random_options.seed = static_cast<uint32_t>(ic.random_seed);
        if (ic.random_field_type == ICRandomFieldType::Gaussian) {

            auto Pofk_of_kBox_over_volume = [&](double kBox) {
                return power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM);
            };
            FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(delta_ini_fourier,
                                                                                rng,
                                                                                Pofk_of_kBox_over_volume,
                                                                                ic.fix_amplitude,
                                                                                random_options);

        } else if (ic.random_field_type == ICRandomFieldType::NonGaussian) {

            // The power-spectrum of the Bardeen potential (-Phi) at the given redshift. On large scales the
            // potential is Phi = (3/5) g(a) zeta with zeta the primordial curvature perturbation (power_primordial
            // uses As) and g(a) = D(a)/a the growth suppression of the potential (1 in the matter era). The scale
            // dependence D(afnl, k) / D(1, k) is that of the gravity model. With ic_fnl_redshift = 0 fNL is defined
            // with the potential extrapolated to z = 0 with linear growth (the usual large-scale-structure
            // convention), in the matter era (e.g. ic_fnl_redshift = 30, after radiation and before Lambda)
            // it is the primordial (CMB) convention Phi = (3/5) zeta. The factor used to be 1.5 OmegaM, which is only
            // (3/5) g(a = 1) for OmegaM ~ 0.32 (the effective fNL was 0.84 fNL for OmegaM = 0.25 and 2.5 fNL in EdS)
            const double afnl = 1.0 / (1.0 + ic.fnl_redshift);
            const double H0Box = boxsize * grav_ic->H0_hmpc;
            const double g_today = cosmo->get_potential_growth_suppression_today();
            const double factor = 0.6 * g_today / afnl;
            if (FML::ThisTask == 0)
                std::cout << "# Non-gaussian IC: Phi = (3/5) g zeta on large scales with g(z = 0) = D(1)/D_EdS(1) = "
                          << g_today << ", fNL defined at z = " << ic.fnl_redshift << "\n";
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
                ic.fnl_type,
                0.0,
                {},
                random_options);
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


    //=============================================================================
    /// Make the particles of a species i from the CDM+baryon density field delta_cb(k,zini): delta_i(k) =
    /// delta_cb(k) ratio(kB) with ratio = T_i / T_cb, LPT displacements and velocities (1LPT velocity
    /// velocity_norm_1LPT(kB) Psi, the rest from velocity_norms; see FML::NBODY::NBodyInitialConditions). With
    /// LPT_order >= 2 the 2LPT term is that of the total field delta_cb. The particles start from the lattice
    /// offset by lattice_shift (in units of the box) in each direction: the fields are evaluated at the offset
    /// positions and the particles moved there. id_offset is added to the IDs. With store_cb_LPT_fields the LPT
    /// displacement fields stored in the particles (for COLA) are those of delta_cb, i.e. all the species share the
    /// COLA frame of the total CDM+baryon field. With mass_contrast_of_kBox the particles get the masses
    /// m = 1 + eps(q) with eps(k) = delta_cb(k) mass_contrast(kB) (the mass-encoded baryon-CDM relative mode).
    /// With velocity_norm_2LPT_of_kBox the 2LPT velocity is velocity_norm_2LPT(kB) Psi_2LPT (instead of
    /// velocity_norms[1] Psi_2LPT). Returns delta_i(k) (shifted).
    //=============================================================================
    template <int NDIM, class P>
    FFTWGrid<NDIM> generate_species_from_delta_cb(FML::PARTICLE::MPIParticles<P> & particles,
                                                  const FFTWGrid<NDIM> & delta_cb_fourier,
                                                  const std::function<double(double)> & ratio_of_kBox,
                                                  std::function<double(double)> velocity_norm_1LPT_of_kBox,
                                                  const std::vector<double> & velocity_norms,
                                                  int LPT_order,
                                                  int Npart_1D,
                                                  double allocation_factor,
                                                  double boxsize,
                                                  double zini,
                                                  double lattice_shift,
                                                  long long int id_offset,
                                                  bool store_cb_LPT_fields = false,
                                                  std::function<double(double)> mass_contrast_of_kBox = nullptr,
                                                  std::function<double(double)> velocity_norm_2LPT_of_kBox = nullptr) {
        std::array<double, NDIM> shift_vec;
        shift_vec.fill(lattice_shift);

        // delta_i(k) on the (offset) lattice: f(q) -> f(q + shift) so that the particle created at q gets the
        // displacement at q + shift (and is then moved there)
        FFTWGrid<NDIM> delta = delta_cb_fourier;
        const bool need_total = LPT_order >= 2 or store_cb_LPT_fields;
        FFTWGrid<NDIM> delta_total;
        if (need_total)
            delta_total = delta_cb_fourier;
        FML::GRID::multiply_fourier_by_function_of_k<NDIM>(delta, ratio_of_kBox);
        if (lattice_shift != 0.0) {
            FML::GRID::shift_grid_fourier<NDIM>(delta, shift_vec, +1.0);
            if (need_total)
                FML::GRID::shift_grid_fourier<NDIM>(delta_total, shift_vec, +1.0);
        }

        // The mass contrast eps(k) = delta_cb(k) mass_contrast(kB) (on the same offset lattice)
        FFTWGrid<NDIM> mass_contrast;
        if (mass_contrast_of_kBox) {
            mass_contrast = delta_cb_fourier;
            FML::GRID::multiply_fourier_by_function_of_k<NDIM>(mass_contrast, mass_contrast_of_kBox);
            if (lattice_shift != 0.0)
                FML::GRID::shift_grid_fourier<NDIM>(mass_contrast, shift_vec, +1.0);
        }

        std::vector<FFTWGrid<NDIM>> no_potentials;
        FML::NBODY::NBodyInitialConditions<NDIM, P>(particles,
                                                    Npart_1D,
                                                    allocation_factor,
                                                    delta,
                                                    no_potentials,
                                                    LPT_order,
                                                    boxsize,
                                                    zini,
                                                    velocity_norms,
                                                    velocity_norm_1LPT_of_kBox,
                                                    LPT_order >= 2 ? &delta_total : nullptr,
                                                    store_cb_LPT_fields ? &delta_total : nullptr,
                                                    mass_contrast_of_kBox ? &mass_contrast : nullptr,
                                                    velocity_norm_2LPT_of_kBox);

        for (auto & p : particles) {
            auto * pos = FML::PARTICLE::GetPos(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += lattice_shift;
                if (pos[idim] >= 1.0)
                    pos[idim] -= 1.0;
            }
            if constexpr (FML::PARTICLE::has_get_q<P>()) {
                auto * q = FML::PARTICLE::GetLagrangianPos(p);
                for (int idim = 0; idim < NDIM; idim++) {
                    q[idim] += lattice_shift;
                    if (q[idim] >= 1.0)
                        q[idim] -= 1.0;
                }
            }
            if constexpr (FML::PARTICLE::has_set_id<P>())
                FML::PARTICLE::SetID(p, FML::PARTICLE::GetID(p) + id_offset);
        }
        particles.communicate_particles();
        return delta;
    }


    using TwoSpeciesICSplines = FML::INITIALCONDITIONS::TwoSpeciesICSplines;
    template <int NDIM>
    TwoSpeciesICSplines two_species_ic_splines(const FML::INITIALCONDITIONS::TransferProvider & provider,
                                               const GravityModel<NDIM> & grav_ic,
                                               double zini,
                                               double weight_cdm,
                                               double weight_baryon) {
        return FML::INITIALCONDITIONS::two_species_ic_splines(
            provider,
            [&grav_ic](double k_hmpc, double a) { return grav_ic.get_f_1LPT(a, k_hmpc / grav_ic.H0_hmpc); },
            zini,
            weight_cdm,
            weight_baryon);
    }
    using FML::INITIALCONDITIONS::two_species_mass_contrast_today;

} // namespace COLASOLVER

#endif
