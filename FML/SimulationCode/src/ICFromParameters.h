#ifndef ICFROMPARAMETERS_HEADER
#define ICFROMPARAMETERS_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Global/Global.h>
#include <FML/InitialConditions/LinearInput.h>
#include <FML/InitialConditions/SpeciesIC.h>
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
#include "TransferInput.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// The initial conditions from the parameters (ic_*): the linear power-spectrum and transfer function at the initial
// redshift that the initial conditions are generated from (read from file: P(k), T(k) or a set of transfer functions
// from CAMB/CLASS, at some redshift, and brought to the initial redshift with the growth factors of the gravity
// model), and the initial density field delta(k,zini) made from it (gaussian or non-gaussian, the random generator).
// The library does the rest: FML/InitialConditions (the linear input, the CDM/baryon transfer ratios),
// FML/RandomFields (the random fields, fixed amplitudes and reversed phases) and FML/NBody (the particles of a species
// displaced from the density field with LPT). How they are combined for a simulation is in SimulationInit.h
//=============================================================================

namespace SIMCODE {


    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    using LinearPowerSpectrumTable = FML::INITIALCONDITIONS::LinearPowerSpectrumTable;

    //=============================================================================
    /// Read the input power-spectrum / transfer function (as given by ic.type_of_input)
    /// and bring it to the initial redshift with the growth factors of grav_ic.
    ///
    /// For ic.type_of_input = camb, class, internal or tables the transfer functions T_i(k,z) are returned in transfers
    /// (the object the gravity model loaded)
    //=============================================================================
    template <int NDIM>
    LinearPowerSpectrumTable read_initial_power_spectrum(const ICSettings & ic,
                                                         std::shared_ptr<Cosmology> cosmo,
                                                         std::shared_ptr<GravityModel<NDIM>> grav,
                                                         std::shared_ptr<GravityModel<NDIM>> grav_ic,
                                                         std::shared_ptr<TransferProvider> & transfers) {

        const double ainput = 1.0 / (1.0 + ic.input_redshift);
        const double aini = 1.0 / (1.0 + ic.initial_redshift);

        // Growth factor from the input redshift to the initial redshift
        auto D_ini_over_D_input = [&](double k_hmpc) {
            return grav_ic->get_D_1LPT(aini, k_hmpc / grav_ic->H0_hmpc) /
                   grav_ic->get_D_1LPT(ainput, k_hmpc / grav_ic->H0_hmpc);
        };

        // The transfer functions: the ones the gravity model loaded (GravityModel::read_parameters; it needs them for
        // the growth factors with massive neutrinos)
        transfers = nullptr;
        if (ic.has_transfer_tables()) {
            transfers = grav->get_transfers();
            if (not transfers)
                throw std::runtime_error("read_initial_power_spectrum: the gravity model has not loaded the transfer "
                                         "functions of ic_type_of_input = " +
                                         ic.type_of_input_name());
        }
        auto type = ic.type_of_input == ICInputType::PowerSpectrum ?
                        FML::INITIALCONDITIONS::LinearInputType::PowerSpectrum :
                    ic.type_of_input == ICInputType::TransferFunction ?
                        FML::INITIALCONDITIONS::LinearInputType::TransferFunction :
                        FML::INITIALCONDITIONS::LinearInputType::Provider;
        auto table = FML::INITIALCONDITIONS::prepare_linear_input(
            {type, ic.input_filename, ainput, cosmo->get_h()},
            D_ini_over_D_input,
            [cosmo](double k) { return cosmo->get_primordial_pofk(k); },
            transfers.get());

        table.print_sample("Read [" + ic.input_filename + "] at z = " + std::to_string(ic.input_redshift) +
                           " and scaled to z = " + std::to_string(ic.initial_redshift));
        return table;
    }

    //=============================================================================
    /// sigma(R) at redshift z computed from P(k,zini) using the growth factors of grav_ic
    //=============================================================================
    template <int NDIM>
    double sigma_of_R(const Spline & power_initial_spline,
                      std::shared_ptr<GravityModel<NDIM>> grav_ic,
                      double boxsize,
                      double initial_redshift,
                      double redshift,
                      double R_mpch) {
        auto pofk_over_volume = [&](double kBox) {
            const double D = grav_ic->get_D_1LPT(1.0 / (1.0 + redshift), kBox / boxsize / grav_ic->H0_hmpc);
            const double D_initial =
                grav_ic->get_D_1LPT(1.0 / (1.0 + initial_redshift), kBox / boxsize / grav_ic->H0_hmpc);
            return std::pow(D / D_initial, 2) * power_initial_spline(kBox / boxsize) / std::pow(boxsize, NDIM);
        };
        return FML::NBODY::compute_sigma_of_R(pofk_over_volume, R_mpch, boxsize);
    }

    //=============================================================================
    // The initial density field delta(k,zini)
    //=============================================================================

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

    /// The CDM/baryon transfer ratios and growth rates at zini (FML::INITIALCONDITIONS::two_species_ic_splines) with
    /// the growth rate of the gravity model of the IC
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

} // namespace SIMCODE

#endif
