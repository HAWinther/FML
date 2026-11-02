#ifndef FML_MAKE_TRANSFER_PROVIDER_H
#define FML_MAKE_TRANSFER_PROVIDER_H

#include <FML/InitialConditions/TransferProvider.h>
#include <FML/TransferProviders/CAMBFileTransferProvider.h>
#include <FML/TransferProviders/CLASSTransferProvider.h>
#include <FML/TransferProviders/GaugeView.h>
#ifdef USE_INTERNAL_COSMOLOGY
#include <FML/InitialConditions/InternalTransferProvider.h>
#endif

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

//=====================================================================================================================
// One way to make a TransferProvider from CAMB files, CLASS files or FML's own Einstein-Boltzmann solver, with the
// gauge of the input declared and converted to the gauge that is asked for (see GaugeView.h).
//
//   source = "camb":     a CAMB transfer info file (CAMBFileTransferProvider.h). CAMB's density columns are always
//                        CDM-synchronous, so input_gauge must be SynchronousCDM.
//   source = "class":    native CLASS output (CLASSTransferProvider.h), gauge = synchronous or newtonian: declare it
//                        in input_gauge. Newtonian input is converted (at the tabulated redshifts) when output_gauge
//                        is SynchronousCDM.
//   source = "internal": FML's Einstein-Boltzmann solver (needs USE_INTERNAL_COSMOLOGY and linking the Cosmology
//                        modules, see tests/Makefile). It computes both gauges itself. By default the cosmology is
//                        taken from h, Omegab, OmegaCDM, OmegaMNu, A_s, n_s and k_pivot_mpc below (the same fields as
//                        for the other sources). InternalTransferProvider supports any model of the solver (with
//                        curvature) and its massive neutrinos: OmegaMNu > 0 is internal_settings.cosmo.deg_ncdm
//                        (default 1) degenerate species (their mass is found from OmegaMNu, with the solver's Neff =
//                        3.046 in total and T_ncdm/TCMB = 0.71611, as CLASS with N_ur = 2.0328).
//=====================================================================================================================

namespace FML::INITIALCONDITIONS {

    struct TransferSourceSettings {
        /// camb, class or internal
        std::string source{"camb"};
        /// The gauge of the input files (camb: always SynchronousCDM; class: what CLASS was run with)
        Gauge input_gauge{Gauge::SynchronousCDM};
        /// The density gauge of the returned provider (the particle initial conditions need SynchronousCDM)
        Gauge output_gauge{Gauge::SynchronousCDM};
        RangePolicy range_policy{RangePolicy::SourceExtrapolation};

        /// The primordial spectrum (camb and class): P_R(k) = A_s (k / k_pivot)^(n_s - 1)
        double A_s{0.0};
        double n_s{1.0};
        double k_pivot_mpc{0.05};

        /// camb: the transfer info file (the columns are read by name from the header, or by position: see
        /// CAMBFileTransferProvider.h). h, Omegab and OmegaCDM are needed (OmegaMNu only if the files have no total
        /// column)
        std::string camb_infofile;
        double Omegab{0.0};
        double OmegaCDM{0.0};
        double OmegaMNu{0.0};
        double h{0.0};

        /// class: either the root of the run (the files <root>z*_tk.dat and <root>background.dat) or the files
        std::string class_root;
        std::vector<std::string> class_transfer_files;
        std::string class_background_file;

        /// internal: the k-range of the solver in h/Mpc and the number of k-modes (0: keep internal_settings). The
        /// solver's default k_max (0.3/Mpc) is too small for particle initial conditions
        double internal_kmin_hmpc{1e-5};
        double internal_kmax_hmpc{10.0};
        int internal_nk{400};
        /// internal: the perturbations are tabulated up to this scale factor. Past a = 1 so that a = 1 is not at the
        /// edge of the table, where the spline end distorts the growth rate (by 2.6e-3 with the table ending at a = 1)
        double internal_amax{1.1};
        /// internal: take the cosmology (h, Omegab, OmegaCDM, OmegaMNu, A_s, n_s, k_pivot_mpc) from the fields above.
        /// false: use internal_settings.cosmo as it is
        bool internal_cosmology_from_settings{true};
#ifdef USE_INTERNAL_COSMOLOGY
        /// internal: the settings of the solver (precision, recombination, ...; the cosmology and k-range are set
        /// from the fields above unless switched off)
        FML::COSMOLOGY::RunSettings internal_settings;
#endif
    };

    inline std::shared_ptr<TransferProvider> make_transfer_provider(const TransferSourceSettings & s) {
        const auto gauge_name = [](Gauge g) { return g == Gauge::Newtonian ? "Newtonian" : "SynchronousCDM"; };
        if (s.output_gauge != Gauge::SynchronousCDM and s.output_gauge != Gauge::Newtonian)
            throw std::invalid_argument("[make_transfer_provider] Unknown output gauge");

        if (s.source == "camb") {
            if (s.input_gauge != Gauge::SynchronousCDM)
                throw std::invalid_argument("[make_transfer_provider] CAMB density transfers are CDM-synchronous: "
                                            "input_gauge must be SynchronousCDM");
            if (s.output_gauge != Gauge::SynchronousCDM)
                throw std::invalid_argument("[make_transfer_provider] Converting CAMB (synchronous) densities to the "
                                            "Newtonian gauge is not supported");
            CAMBFileTransferProvider::Settings cs;
            cs.infofile = s.camb_infofile;
            cs.h = s.h;
            cs.Omegab = s.Omegab;
            cs.OmegaCDM = s.OmegaCDM;
            cs.OmegaMNu = s.OmegaMNu;
            cs.A_s = s.A_s;
            cs.n_s = s.n_s;
            cs.k_pivot_mpc = s.k_pivot_mpc;
            cs.range_policy = s.range_policy;
            return std::make_shared<CAMBFileTransferProvider>(cs);
        }

        if (s.source == "class") {
            CLASSTransferProvider::Settings cs;
            if (not s.class_root.empty()) {
                auto files = find_class_output_files(s.class_root);
                cs.transfer_files = files.first;
                cs.background_file = files.second;
            }
            if (not s.class_transfer_files.empty())
                cs.transfer_files = s.class_transfer_files;
            if (not s.class_background_file.empty())
                cs.background_file = s.class_background_file;
            cs.gauge = s.input_gauge;
            cs.output_gauge = s.output_gauge;
            cs.A_s = s.A_s;
            cs.n_s = s.n_s;
            cs.k_pivot_mpc = s.k_pivot_mpc;
            cs.range_policy = s.range_policy;
            if (s.input_gauge != s.output_gauge and s.input_gauge == Gauge::SynchronousCDM)
                throw std::invalid_argument(std::string("[make_transfer_provider] Converting CLASS ") +
                                            gauge_name(s.input_gauge) + " output to " + gauge_name(s.output_gauge) +
                                            " is not supported: run CLASS with gauge = newtonian");
            // A Newtonian run is converted by the provider at the redshifts of the files (more accurate growth
            // rates than a GaugeView on top of the interpolated Newtonian tables)
            return std::make_shared<CLASSTransferProvider>(cs);
        }

        if (s.source == "internal") {
#ifdef USE_INTERNAL_COSMOLOGY
            auto settings = s.internal_settings;
            if (s.internal_cosmology_from_settings) {
                if (not(s.OmegaMNu >= 0.0))
                    throw std::invalid_argument("[make_transfer_provider] source = internal: OmegaMNu must be >= 0");
                if (not(s.h > 0.0) or not(s.Omegab > 0.0) or not(s.OmegaCDM > 0.0) or not(s.A_s > 0.0))
                    throw std::invalid_argument("[make_transfer_provider] source = internal: set h, Omegab, OmegaCDM "
                                                "and A_s (or internal_cosmology_from_settings = false)");
                settings.cosmo.h = s.h;
                settings.cosmo.OmegaB = s.Omegab;
                settings.cosmo.OmegaCDM = s.OmegaCDM;
                settings.cosmo.m_nu = 0.0;
                if (s.OmegaMNu > 0.0) {
                    // The mass of the one massive species with Omega_ncdm = OmegaMNu (increasing in m_nu: bisect in
                    // log m_nu)
                    auto cosmo = settings.cosmo;
                    double lo = std::log(1e-5), hi = std::log(100.0);
                    for (int i = 0; i < 200 and hi - lo > 1e-14; i++) {
                        cosmo.m_nu = std::exp(0.5 * (lo + hi));
                        (cosmo.OmegaNcdm() < s.OmegaMNu ? lo : hi) = std::log(cosmo.m_nu);
                    }
                    settings.cosmo.m_nu = std::exp(0.5 * (lo + hi));
                    if (not(std::abs(settings.cosmo.OmegaNcdm() / s.OmegaMNu - 1.0) < 1e-8))
                        throw std::invalid_argument("[make_transfer_provider] source = internal: no neutrino mass "
                                                    "below 100 eV gives OmegaMNu = " + std::to_string(s.OmegaMNu));
                }
                settings.cosmo.A_s = s.A_s;
                settings.cosmo.n_s = s.n_s;
                settings.cosmo.k_pivot = s.k_pivot_mpc;
            }
            // k in h/Mpc -> 1/m
            const double hmpc_to_si = settings.cosmo.h / FML::COSMOLOGY::Constants::Mpc;
            if (s.internal_kmin_hmpc > 0.0)
                settings.pert.k_min = s.internal_kmin_hmpc * hmpc_to_si;
            if (s.internal_kmax_hmpc > 0.0)
                settings.pert.k_max = s.internal_kmax_hmpc * hmpc_to_si;
            if (s.internal_nk > 0)
                settings.pert.n_k = s.internal_nk;
            if (s.internal_amax > 0.0) {
                settings.pert.x_end = std::log(s.internal_amax);
                settings.bg_x_end = std::max(settings.bg_x_end, settings.pert.x_end);
            }
            return std::make_shared<InternalTransferProvider>(settings, s.output_gauge);
#else
            throw std::invalid_argument("[make_transfer_provider] source = internal needs USE_INTERNAL_COSMOLOGY");
#endif
        }

        throw std::invalid_argument("[make_transfer_provider] Unknown source [" + s.source +
                                    "]: choose camb, class or internal");
    }

} // namespace FML::INITIALCONDITIONS
#endif
