#ifndef INITIALPOWERSPECTRUM_HEADER
#define INITIALPOWERSPECTRUM_HEADER

#include <FML/InitialConditions/LinearInput.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Global/Global.h>
#include <FML/NBody/NBody.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"
#include "TransferInput.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

//=============================================================================
// The linear power-spectrum and transfer function at the initial redshift that
// the initial conditions are generated from. These are read in from file (P(k),
// T(k) or a set of transfer functions from CAMB/CLASS) at some redshift and
// brought to the initial redshift using the growth factors of the gravity model
//=============================================================================

namespace SIMCODE {

    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    using LinearPowerSpectrumTable = FML::INITIALCONDITIONS::LinearPowerSpectrumTable;

    //=============================================================================
    /// Read the input power-spectrum / transfer function (as given by ic.type_of_input)
    /// and bring it to the initial redshift with the growth factors of grav_ic.
    ///
    /// For ic.type_of_input = camb, class or internal the transfer functions T_i(k,z) are returned in transfers (the
    /// object the gravity model loaded)
    //=============================================================================
    template <int NDIM>
    LinearPowerSpectrumTable
    read_initial_power_spectrum(const ICSettings & ic,
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
                                         "functions of ic_type_of_input = " + ic.type_of_input_name());
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

} // namespace SIMCODE

#endif
