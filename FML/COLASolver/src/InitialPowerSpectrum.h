#ifndef INITIALPOWERSPECTRUM_HEADER
#define INITIALPOWERSPECTRUM_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
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

namespace COLASOLVER {

    using DVector = FML::INTERPOLATION::SPLINE::DVector;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using LinearTransferData = FML::FILEUTILS::LinearTransferData;

    /// P(k), P_primordial(k) and T(k) of baryons+CDM at the initial redshift.
    /// Units: k in h/Mpc, P in (Mpc/h)^3 and T in Mpc^2
    struct LinearPowerSpectrumTable {
        DVector k;
        DVector power;
        DVector power_primordial;
        DVector transfer;

        void resize(size_t n) {
            k.resize(n);
            power.resize(n);
            power_primordial.resize(n);
            transfer.resize(n);
        }

        /// Multiply the power-spectra by factor (and the transfer function by sqrt(factor))
        void scale(double factor) {
            for (auto & p : power)
                p *= factor;
            for (auto & p : power_primordial)
                p *= factor;
            for (auto & t : transfer)
                t *= std::sqrt(factor);
        }

        void print_sample(const std::string & description) const {
            if (FML::ThisTask != 0)
                return;
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# " << description << "\n";
            std::cout << "# Found n: " << k.size() << " k-values (sample of it below):\n";
            std::cout << "#=====================================================\n";
            std::cout << "# k (h/Mpc)      P(k,zini)  (Mpc/h)^3    T(k,zini)  (Mpc)^2\n";
            const size_t every = std::max(size_t(1), k.size() / 10);
            for (size_t i = 0; i < k.size(); i += every) {
                std::cout << std::setw(15) << k[i] << " ";
                std::cout << std::setw(15) << power[i] << " ";
                std::cout << std::setw(15) << transfer[i] << "\n";
            }
        }
    };

    //=============================================================================
    /// Read the input power-spectrum / transfer function (as given by ic.type_of_input)
    /// and bring it to the initial redshift with the growth factors of grav_ic.
    ///
    /// For ic.type_of_input = transferinfofile we read in all the transfer functions
    /// T_i(k,z) and store them in transferdata (which is shared with the gravity model)
    //=============================================================================
    template <int NDIM>
    LinearPowerSpectrumTable read_initial_power_spectrum(const ICSettings & ic,
                                                         std::shared_ptr<Cosmology> cosmo,
                                                         std::shared_ptr<GravityModel<NDIM>> grav,
                                                         std::shared_ptr<GravityModel<NDIM>> grav_ic,
                                                         std::shared_ptr<LinearTransferData> & transferdata) {

        const double ainput = 1.0 / (1.0 + ic.input_redshift);
        const double aini = 1.0 / (1.0 + ic.initial_redshift);

        // Growth factor from the input redshift to the initial redshift
        auto D_ini_over_D_input = [&](double k_hmpc) {
            return grav_ic->get_D_1LPT(aini, k_hmpc / grav_ic->H0_hmpc) /
                   grav_ic->get_D_1LPT(ainput, k_hmpc / grav_ic->H0_hmpc);
        };

        // Read a two column ascii file with a single header line
        auto read_two_columns = [&]() {
            const int ncols = 2;
            std::vector<int> cols_to_keep{0, 1};
            const int nheaderlines = 1;
            return FML::FILEUTILS::read_regular_ascii(ic.input_filename, ncols, cols_to_keep, nheaderlines);
        };

        LinearPowerSpectrumTable table;
        switch (ic.type_of_input) {
            case ICInputType::PowerSpectrum: {
                // Fileformat: [k (h/Mpc)  P(k)  (Mpc/h)^3]
                auto pofkdata = read_two_columns();
                table.resize(pofkdata.size());
                for (size_t i = 0; i < table.k.size(); i++) {
                    table.k[i] = pofkdata[i][0];
                    const double k_hmpc = table.k[i];
                    const double k_mpc = table.k[i] * cosmo->get_h();
                    const double DinioverDinput = D_ini_over_D_input(k_hmpc);
                    table.power_primordial[i] = cosmo->get_primordial_pofk(k_hmpc);
                    table.power[i] = DinioverDinput * DinioverDinput * pofkdata[i][1];
                    table.transfer[i] = std::sqrt(table.power[i] / table.power_primordial[i]) / (k_mpc * k_mpc);
                }
                break;
            }
            case ICInputType::TransferFunction: {
                // Fileformat: [k (h/Mpc)  T(k)  (Mpc)^2]
                auto tofkdata = read_two_columns();
                table.resize(tofkdata.size());
                for (size_t i = 0; i < table.k.size(); i++) {
                    table.k[i] = tofkdata[i][0];
                    const double k_hmpc = table.k[i];
                    const double k_mpc = table.k[i] * cosmo->get_h();
                    const double DinioverDinput = D_ini_over_D_input(k_hmpc);
                    table.power_primordial[i] = cosmo->get_primordial_pofk(k_hmpc);
                    table.transfer[i] = DinioverDinput * tofkdata[i][1];
                    table.power[i] = table.power_primordial[i] * std::pow(table.transfer[i] * k_mpc * k_mpc, 2);
                }
                break;
            }
            case ICInputType::TransferInfoFile: {
                // Read in all transfer functions T_i(k,z) unless the gravity model already has done so
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

                // For consistency we generate the arrays at the input redshift and use growth-factors
                // to bring them to the initial time (this way we can pick backscaling or not by
                // selecting ic_input_redshift)
                const int nk = 1000;
                const double kmin = transferdata->get_kmin_hmpc_splines();
                const double kmax = transferdata->get_kmax_hmpc_splines();
                table.resize(nk);
                for (size_t i = 0; i < table.k.size(); i++) {
                    const double k_hmpc = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(nk));
                    table.k[i] = k_hmpc;
                    table.transfer[i] = transferdata->get_cdm_baryon_transfer_function(table.k[i], ainput);
                    table.power[i] = transferdata->get_cdm_baryon_power_spectrum(table.k[i], ainput);
                    const double DinioverDinput = D_ini_over_D_input(k_hmpc);
                    table.power_primordial[i] = cosmo->get_primordial_pofk(k_hmpc);
                    table.transfer[i] *= DinioverDinput;
                    table.power[i] *= DinioverDinput * DinioverDinput;
                }
                break;
            }
        }

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

} // namespace COLASOLVER

#endif
