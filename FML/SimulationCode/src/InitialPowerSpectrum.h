#ifndef INITIALPOWERSPECTRUM_HEADER
#define INITIALPOWERSPECTRUM_HEADER

#include <FML/InitialConditions/CAMBTransferProvider.h>
#include <FML/InitialConditions/LinearInput.h>
#include <FML/InitialConditions/InternalTransferLoader.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Global/Global.h>
#include <FML/NBody/NBody.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

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
    using LinearTransferData = FML::FILEUTILS::LinearTransferData;

    using LinearPowerSpectrumTable = FML::INITIALCONDITIONS::LinearPowerSpectrumTable;

    //=============================================================================
    /// Read the input power-spectrum / transfer function (as given by ic.type_of_input)
    /// and bring it to the initial redshift with the growth factors of grav_ic.
    ///
    /// For ic.type_of_input = transferinfofile we read in all the transfer functions
    /// T_i(k,z) and store them in transferdata (which is shared with the gravity model)
    //=============================================================================
    template <int NDIM>
    LinearPowerSpectrumTable
    read_initial_power_spectrum(const ICSettings & ic,
                                std::shared_ptr<Cosmology> cosmo,
                                std::shared_ptr<GravityModel<NDIM>> grav,
                                std::shared_ptr<GravityModel<NDIM>> grav_ic,
                                std::shared_ptr<LinearTransferData> & transferdata,
                                std::shared_ptr<FML::INITIALCONDITIONS::TransferProvider> * provider_out = nullptr,
                                double boxsize_mpch = 0.0) {

        const double ainput = 1.0 / (1.0 + ic.input_redshift);
        const double aini = 1.0 / (1.0 + ic.initial_redshift);

        // Growth factor from the input redshift to the initial redshift
        auto D_ini_over_D_input = [&](double k_hmpc) {
            return grav_ic->get_D_1LPT(aini, k_hmpc / grav_ic->H0_hmpc) /
                   grav_ic->get_D_1LPT(ainput, k_hmpc / grav_ic->H0_hmpc);
        };

        std::shared_ptr<FML::INITIALCONDITIONS::TransferProvider> provider;
        if (ic.type_of_input == ICInputType::TransferInfoFile) {
            transferdata = grav->get_transferdata();
            if (not transferdata) {
                transferdata = std::make_shared<LinearTransferData>(cosmo->get_Omegab(),
                                                                    cosmo->get_OmegaCDM(),
                                                                    cosmo->get_OmegaMNu(),
                                                                    cosmo->get_kpivot_mpc(),
                                                                    cosmo->get_As(),
                                                                    cosmo->get_ns(),
                                                                    cosmo->get_h(),
                                                                    ic.type_of_input_fileformat);
                transferdata->read_transfer(ic.input_filename);

                // Make sure the gravity model also gets a pointer to this
                grav->set_transferdata(transferdata);
            }

            provider = std::make_shared<FML::INITIALCONDITIONS::CAMBTransferProvider>(transferdata);
        }
        if (ic.type_of_input == ICInputType::InternalTransfer) {
            if (cosmo->get_name() != "LCDM" || grav->get_name() != "GR" || cosmo->get_OmegaK() != 0 ||
                cosmo->get_OmegaMNu() != 0)
                throw std::invalid_argument(
                    "internal_transfer currently requires flat LCDM, GR and no massive neutrinos");
            if (!(boxsize_mpch > 0))
                throw std::invalid_argument("internal_transfer needs a positive box size");
            provider = FML::INITIALCONDITIONS::load_internal_transfers(
                {ic.input_filename,
                 cosmo->get_h(),
                 cosmo->get_Omegab(),
                 cosmo->get_OmegaCDM(),
                 cosmo->get_TCMB_kelvin(),
                 cosmo->get_Neff(),
                 cosmo->get_As(),
                 cosmo->get_ns(),
                 cosmo->get_kpivot_mpc(),
                 {2 * M_PI / boxsize_mpch,
                  std::sqrt(double(NDIM)) * M_PI * ic.nmesh / boxsize_mpch,
                  std::min(aini, ainput),
                  std::max({1.0, aini, ainput})},
                 ic.sigma8_normalization});
        }
        auto type = ic.type_of_input == ICInputType::PowerSpectrum ?
                        FML::INITIALCONDITIONS::LinearInputType::PowerSpectrum :
                    ic.type_of_input == ICInputType::TransferFunction ?
                        FML::INITIALCONDITIONS::LinearInputType::TransferFunction :
                        FML::INITIALCONDITIONS::LinearInputType::TransferInfoFile;
        auto table = FML::INITIALCONDITIONS::prepare_linear_input(
            {type, ic.input_filename, ainput, cosmo->get_h()},
            D_ini_over_D_input,
            [cosmo](double k) { return cosmo->get_primordial_pofk(k); },
            provider.get());
        if (provider_out)
            *provider_out = provider;

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
