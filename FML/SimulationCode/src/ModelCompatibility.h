#ifndef MODELCOMPATIBILITY_HEADER
#define MODELCOMPATIBILITY_HEADER

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

#include <stdexcept>
#include <string>

namespace SIMCODE {

    //=============================================================================
    /// The checks of the parameters (SimulationParameters, already read and validated on their own) that depend on
    /// the cosmology and gravity model: what is only implemented for some models. Throws if not. Called once in
    /// Main.cpp before the simulation starts (the checks that only need the parameters are in
    /// SimulationParameters::validate)
    //=============================================================================
    template <int NDIM>
    void check_parameters_against_models(const SimulationParameters & params,
                                         const Cosmology & cosmo,
                                         const GravityModel<NDIM> & grav,
                                         bool has_transfers) {
        auto fail = [](const std::string & msg) { throw std::runtime_error(msg); };

        if (params.baryonification.enabled) {
            if (NDIM != 3 || grav.get_name() != "GR" || cosmo.get_OmegaMNu() != 0.0)
                fail("baryonification currently requires NDIM=3, gravity_model=GR and cosmology_OmegaMNu=0");
            params.baryonification.profile_model(cosmo.get_Omegab() / cosmo.get_OmegaM()).validate();
        }

        // The particle species (CDM+baryons as one, or CDM and baryons, and massive neutrinos)
        if (params.particle.neutrinos and not(cosmo.get_OmegaMNu() > 0.0 and cosmo.get_Mnu_eV() > 0.0))
            fail("particle_neutrinos = true needs massive neutrinos (cosmology_OmegaMNu > 0 or "
                 "cosmology_neutrino_masses_eV)");

        // The N-body gauge needs GR: with the internal solver any cosmology with the GR field equations (dark energy
        // that is a fluid or a scalar field); the tables of scripts/make_nbody_gauge_transfer.py are for LCDM
        if (params.force.nbody_gauge) {
            const auto model = cosmo.get_name();
            const bool internal = params.ic.from_internal_solver();
            if (grav.get_name() != "GR")
                fail("force_nbody_gauge needs gravity_model = GR");
            if (internal and not(model == "LCDM" or model == "w0waCDM" or model == "Quintessence" or model == "EDE"))
                fail("force_nbody_gauge with ic_type_of_input = internal or tables needs a cosmology with the GR field "
                     "equations (LCDM, w0waCDM, Quintessence or EDE), not " +
                     model);
            if (not internal and model != "LCDM")
                fail("force_nbody_gauge with ic_type_of_input = camb is only implemented for cosmology_model = LCDM "
                     "(the source is computed with CLASS)");
            if (not has_transfers)
                fail("force_nbody_gauge needs the transfer functions");
        }
    }

} // namespace SIMCODE

#endif
