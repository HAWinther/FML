#ifndef SIMCODE_TRANSFERINPUT_HEADER
#define SIMCODE_TRANSFERINPUT_HEADER

#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/ParameterMap/ParameterMap.h>

#include "ParameterDefaults.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>

//=============================================================================
// The transfer functions of the simulation (ic_type_of_input = camb, class or internal), made with
// FML/TransferProviders (make_transfer_provider) or, for internal, FML's Einstein-Boltzmann solver run with the
// cosmology of the simulation. The provider is shared by the gravity model (scale-dependent growth with massive
// neutrinos), the initial conditions and the analysis. Only TransferInput.cpp includes MakeTransferProvider.h (and
// with USE_INTERNAL_COSMOLOGY the Einstein-Boltzmann solver).
//=============================================================================

namespace SIMCODE {

    using TransferProvider = FML::INITIALCONDITIONS::TransferProvider;
    using TransferSpecies = FML::INITIALCONDITIONS::Species;

    struct TransferInputRequest {
        /// camb (a CAMB transfer info file), class (the root of a native CLASS run) or internal (FML's
        /// Einstein-Boltzmann solver with the cosmology below; filename: an optional .ini of the solver with its
        /// precision, recombination and reionization settings, empty for the defaults)
        std::string type;
        std::string filename;
        /// class: the gauge CLASS was run in (synchronous or newtonian; converted to CDM-synchronous)
        std::string gauge{"synchronous"};
        /// The cosmology of the simulation (CAMB: the weights and h; CLASS: h and the massive neutrinos checked)
        double h{0.0}, Omegab{0.0}, OmegaCDM{0.0}, OmegaMNu{0.0}, TCMB{0.0}, Neff{0.0};
        double As{0.0}, ns{1.0}, kpivot_mpc{0.05};
        /// internal: the background of the simulation (all its parameters, with the neutrino masses) and its model
        /// in the background/perturbation registry with the model parameters
        FML::COSMOLOGY::BACKGROUND::BackgroundParameters background;
        std::string model;
        std::map<std::string, double> model_parameters;
        /// internal: the k-range (h/Mpc) and scale factors the tables must cover (the solver is run on a wider range;
        /// outside the tables the values are extrapolated as for CAMB and CLASS)
        FML::INITIALCONDITIONS::TransferRange required{0.0, 0.0, 0.0, 0.0};
    };

    /// ic_type_of_input with the old names (transferinfofile, internal_transfer) replaced by camb and internal
    inline std::string canonical_input_type(const std::string & type) {
        if (type == "transferinfofile")
            return "camb";
        if (type == "internal_transfer")
            return "internal";
        return type;
    }

    /// Whether ic_type_of_input gives transfer functions of all species at all redshifts (the gravity model needs
    /// them with massive neutrinos, and so do neutrino particles)
    inline bool input_has_transfer_tables(const std::string & type) {
        const auto t = canonical_input_type(type);
        return t == "camb" or t == "class" or t == "internal";
    }

    /// The transfer functions of the input (see TransferInput.cpp)
    std::shared_ptr<TransferProvider> load_transfers(const TransferInputRequest & request);

    /// The request of the simulation from its parameters (ic_*, simulation_boxsize) and cosmology
    template <class CosmologyType>
    TransferInputRequest
    make_transfer_request(const FML::UTILS::ParameterMap & param, const CosmologyType & cosmo, int ndim) {
        TransferInputRequest r;
        r.type = canonical_input_type(param.get<std::string>("ic_type_of_input"));
        r.filename = SIMCODE::get_parameter<std::string>(param, "ic_input_filename");
        r.gauge = SIMCODE::get_parameter<std::string>(param, "ic_input_gauge");
        r.h = cosmo.get_h();
        r.Omegab = cosmo.get_Omegab();
        r.OmegaCDM = cosmo.get_OmegaCDM();
        r.OmegaMNu = cosmo.get_OmegaMNu();
        r.TCMB = cosmo.get_TCMB_kelvin();
        r.Neff = cosmo.get_Neff();
        r.As = cosmo.get_As();
        r.ns = cosmo.get_ns();
        r.kpivot_mpc = cosmo.get_kpivot_mpc();
        if (r.type == "internal") {
            r.background = cosmo.background().parameters();
            r.model = cosmo.get_registry_name();
            r.model_parameters = cosmo.get_model_parameters();
            // The modes of the IC grid, the input and initial redshifts and today
            const double boxsize = param.get<double>("simulation_boxsize");
            const int nmesh = param.get<int>("ic_nmesh");
            const double aini = 1.0 / (1.0 + param.get<double>("ic_initial_redshift"));
            const double ainput = 1.0 / (1.0 + param.get<double>("ic_input_redshift"));
            r.required = {2.0 * M_PI / boxsize,
                          std::sqrt(double(ndim)) * M_PI * nmesh / boxsize,
                          std::min(aini, ainput),
                          std::max({1.0, aini, ainput})};
        }
        return r;
    }

    /// The highest redshift of the transfer tables
    inline double transfer_zmax(const TransferProvider & transfers) { return 1.0 / transfers.range().amin - 1.0; }

} // namespace SIMCODE

#endif
