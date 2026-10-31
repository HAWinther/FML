#include "TransferInput.h"

#include <FML/TransferProviders/MakeTransferProvider.h>
#ifdef USE_INTERNAL_COSMOLOGY
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#endif

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace SIMCODE {

    namespace {
        bool matches(double a, double b, double rtol) {
            return std::abs(a - b) <= rtol * std::max({std::abs(a), std::abs(b), 1e-30});
        }
    } // namespace

    //=============================================================================
    /// camb: CAMB transfer files (the columns read by name from the header, or by position), with the
    ///   weights of CDM+baryons from the cosmology of the simulation. The densities are CDM-synchronous.
    /// class: native CLASS output (format = class, output = mTk, vTk, write_background = yes) with the root
    ///   ic_input_filename, run in the gauge ic_input_gauge (Newtonian is converted to CDM-synchronous). h and
    ///   whether there are massive neutrinos are checked against the simulation.
    /// internal_transfer: FML's Einstein-Boltzmann solver run with the .ini file ic_input_filename (needs a build with
    ///   USE_INTERNAL_COSMOLOGY=true). Its cosmology must be that of the simulation, and its k- and a-range must cover
    ///   the request (no extrapolation).
    /// The CAMB and CLASS tables are extrapolated outside their range as the old CAMB reader did (RangePolicy::
    /// SourceExtrapolation: a power law above k_max, the end values elsewhere)
    //=============================================================================
    std::shared_ptr<TransferProvider> load_transfers(const TransferInputRequest & r) {
        using namespace FML::INITIALCONDITIONS;
        TransferSourceSettings s;
        s.h = r.h;
        s.Omegab = r.Omegab;
        s.OmegaCDM = r.OmegaCDM;
        s.OmegaMNu = r.OmegaMNu;
        s.A_s = r.As;
        s.n_s = r.ns;
        s.k_pivot_mpc = r.kpivot_mpc;
        s.output_gauge = Gauge::SynchronousCDM;
        s.range_policy = RangePolicy::SourceExtrapolation;

        if (canonical_input_type(r.type) == "camb") {
            s.source = "camb";
            s.camb_infofile = r.filename;
            return make_transfer_provider(s);
        }

        if (r.type == "class") {
            if (r.gauge != "synchronous" and r.gauge != "newtonian")
                throw std::invalid_argument("ic_input_gauge must be synchronous or newtonian (the gauge of the CLASS "
                                            "run), not [" + r.gauge + "]");
            s.source = "class";
            s.class_root = r.filename;
            s.input_gauge = r.gauge == "newtonian" ? Gauge::Newtonian : Gauge::SynchronousCDM;
            auto provider = make_transfer_provider(s);
            if (not matches(provider->h(), r.h, 1e-4))
                throw std::invalid_argument("The CLASS run [" + r.filename + "] has h = " +
                                            std::to_string(provider->h()) + ", the simulation " + std::to_string(r.h));
            if (provider->has_density(Species::MassiveNeutrino) != (r.OmegaMNu > 0.0))
                throw std::invalid_argument("The CLASS run [" + r.filename + "] " +
                                            (r.OmegaMNu > 0.0 ? "has no" : "has") +
                                            " massive neutrinos, the simulation has cosmology_OmegaMNu = " +
                                            std::to_string(r.OmegaMNu));
            return provider;
        }

        if (r.type == "internal_transfer") {
#ifdef USE_INTERNAL_COSMOLOGY
            auto settings = FML::COSMOLOGY::ParameterFile::read(r.filename);
            const auto & p = settings.cosmo;
            if (not matches(p.h, r.h, 1e-8) or not matches(p.OmegaB, r.Omegab, 1e-8) or
                not matches(p.OmegaCDM, r.OmegaCDM, 1e-8) or not matches(p.TCMB, r.TCMB, 1e-8) or
                not matches(p.Neff, r.Neff, 1e-8) or not matches(p.A_s, r.As, 1e-8) or not matches(p.n_s, r.ns, 1e-8) or
                not matches(p.k_pivot, r.kpivot_mpc, 1e-8) or p.m_nu != 0.0 or r.OmegaMNu != 0.0)
                throw std::invalid_argument("[internal_transfer] Solver and SimulationCode cosmological/primordial "
                                            "parameters differ (and massive neutrinos are not supported here yet)");
            // The tables run past a = 1 (growth rates at a = 1 are then not at the end of the table)
            const double amax = std::max(1.1, std::exp(settings.pert.x_end));
            const double kmin = settings.pert.k_min * FML::COSMOLOGY::Constants::Mpc / p.h;
            const double kmax = settings.pert.k_max * FML::COSMOLOGY::Constants::Mpc / p.h;
            if (r.require_sigma8_coverage and (kmin > 1e-5 or kmax < 100.0))
                throw std::out_of_range("[internal_transfer] sigma8 normalization requires transfer coverage from 1e-5 "
                                        "to 100 h/Mpc; use As normalization or supply wider-range inputs");
            if (kmin > r.required.kmin_hmpc or kmax < r.required.kmax_hmpc or
                std::exp(settings.pert.x_start) > r.required.amin or amax < r.required.amax)
                throw std::out_of_range(
                    "[internal_transfer] Solver k/a range does not cover the IC mesh, input epoch and today");
            s.source = "internal";
            s.internal_cosmology_from_settings = false;
            s.internal_settings = settings;
            s.internal_kmin_hmpc = s.internal_kmax_hmpc = 0.0; // the k-range of the .ini file
            s.internal_nk = 0;
            s.internal_amax = amax;
            s.range_policy = RangePolicy::Reject;
            return make_transfer_provider(s);
#else
            throw std::runtime_error("internal_transfer needs a SimulationCode build with USE_INTERNAL_COSMOLOGY=true");
#endif
        }

        throw std::invalid_argument("[load_transfers] Unknown ic_type_of_input [" + r.type +
                                    "] (camb, class or internal_transfer)");
    }

} // namespace SIMCODE
