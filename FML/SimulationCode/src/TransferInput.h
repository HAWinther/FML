#ifndef SIMCODE_TRANSFERINPUT_HEADER
#define SIMCODE_TRANSFERINPUT_HEADER

#include <FML/InitialConditions/TransferProvider.h>

#include <memory>
#include <string>

//=============================================================================
// The transfer functions of the simulation (ic_type_of_input = camb, class or internal_transfer), made
// with FML/TransferProviders (make_transfer_provider). The provider is shared by the gravity model (scale-dependent
// growth with massive neutrinos), the initial conditions and the analysis. Only TransferInput.cpp includes
// MakeTransferProvider.h (and with USE_INTERNAL_COSMOLOGY the Einstein-Boltzmann solver).
//=============================================================================

namespace SIMCODE {

    using TransferProvider = FML::INITIALCONDITIONS::TransferProvider;
    using TransferSpecies = FML::INITIALCONDITIONS::Species;

    struct TransferInputRequest {
        /// camb (a CAMB transfer info file), class (the root of a native CLASS run) or internal_transfer
        /// (an .ini file of FML's Einstein-Boltzmann solver)
        std::string type;
        std::string filename;
        /// class: the gauge CLASS was run in (synchronous or newtonian; converted to CDM-synchronous)
        std::string gauge{"synchronous"};
        /// The cosmology of the simulation (CAMB: the weights and h; CLASS and internal: checked against the input)
        double h{0.0}, Omegab{0.0}, OmegaCDM{0.0}, OmegaMNu{0.0}, TCMB{0.0}, Neff{0.0};
        double As{0.0}, ns{1.0}, kpivot_mpc{0.05};
        /// internal_transfer: the k- and a-range the solver must cover, and whether sigma8 normalization is used (it
        /// needs k from 1e-5 to 100 h/Mpc)
        FML::INITIALCONDITIONS::TransferRange required{0.0, 0.0, 0.0, 0.0};
        bool require_sigma8_coverage{false};
    };

    /// ic_type_of_input with the old name transferinfofile replaced by camb
    inline std::string canonical_input_type(const std::string & type) {
        return type == "transferinfofile" ? "camb" : type;
    }

    /// Whether ic_type_of_input gives transfer tables of all species at many redshifts (the gravity model needs them
    /// with massive neutrinos, and so do neutrino particles and the N-body gauge)
    inline bool input_has_transfer_tables(const std::string & type) {
        return canonical_input_type(type) == "camb" or type == "class";
    }

    /// The transfer functions of the input (see TransferInput.cpp)
    std::shared_ptr<TransferProvider> load_transfers(const TransferInputRequest & request);

    /// A request with the cosmology of the simulation
    template <class CosmologyType>
    TransferInputRequest make_transfer_request(const std::string & type,
                                               const std::string & filename,
                                               const std::string & gauge,
                                               CosmologyType & cosmo) {
        TransferInputRequest r;
        r.type = canonical_input_type(type);
        r.filename = filename;
        r.gauge = gauge;
        r.h = cosmo.get_h();
        r.Omegab = cosmo.get_Omegab();
        r.OmegaCDM = cosmo.get_OmegaCDM();
        r.OmegaMNu = cosmo.get_OmegaMNu();
        r.TCMB = cosmo.get_TCMB_kelvin();
        r.Neff = cosmo.get_Neff();
        r.As = cosmo.get_As();
        r.ns = cosmo.get_ns();
        r.kpivot_mpc = cosmo.get_kpivot_mpc();
        return r;
    }

    /// The highest redshift of the transfer tables
    inline double transfer_zmax(const TransferProvider & transfers) { return 1.0 / transfers.range().amin - 1.0; }

} // namespace SIMCODE

#endif
