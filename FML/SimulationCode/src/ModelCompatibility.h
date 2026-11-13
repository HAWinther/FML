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

        if (params.force.short_range.enabled) {
            if (cosmo.get_OmegaMNu() != 0.0)
                fail("force_short_range currently excludes massive neutrinos");
            grav.validate_short_range_support();
            grav.short_range_interaction(params.ic.a_initial());
        }

        if (params.baryonification.enabled) {
            if (NDIM != 3 || grav.get_name() != "GR" || cosmo.get_OmegaMNu() != 0.0)
                fail("baryonification currently requires NDIM=3, gravity_model=GR and cosmology_OmegaMNu=0");
            params.baryonification.profile_model(cosmo.get_Omegab() / cosmo.get_OmegaM()).validate();
        }

        // Coupled quintessence: the background is the cosmology's and the force and velocity term of the CDM the gravity
        // model's (GravityModel_CoupledQuintessence: one fluid, or with particle_baryons a second kick of the CDM,
        // kick_cdm_fifth_force, and its own steps). The two go together
        if (cosmo.is_coupled_quintessence() != (grav.get_name() == "CoupledQuintessence"))
            fail("cosmology_model = CoupledQuintessence and gravity_model = CoupledQuintessence go together (the "
                 "background of the coupled field and the force and velocity term of the coupled CDM)");
        if (cosmo.is_coupled_quintessence()) {
            if (cosmo.get_OmegaMNu() != 0.0)
                fail("cosmology_model = CoupledQuintessence is not implemented with massive neutrinos");
            if (params.particle.baryons and params.force.short_range.enabled)
                fail("cosmology_model = CoupledQuintessence with particle_baryons = true is not implemented with "
                     "force_short_range");
            if (params.particle.baryons and params.timestep.method == FML::NBODY::TimestepMethod::BullFrog)
                fail("cosmology_model = CoupledQuintessence with particle_baryons = true needs timestep_method = "
                     "Quinn, Tassev or FastPM (BullFrog follows the growth of one fluid)");
        }

        // The cubic Galileon force needs the background of the Galileon (the cosmology CubicGalileon with gravity_model
        // = GR is the background alone)
        if (grav.get_name() == "CubicGalileon" and cosmo.get_name() != "CubicGalileon")
            fail(
                "gravity_model = CubicGalileon needs cosmology_model = CubicGalileon (the background of the Galileon)");

        // Generic Horndeski: the background of the frontend's table
        if (grav.get_name() == "Horndeski" and cosmo.get_name() != "Horndeski" and cosmo.get_name() != "Tabulated")
            fail("gravity_model = Horndeski needs cosmology_model = Horndeski (or Tabulated with the frontend's "
                 "_background.txt)");

        // Interacting dark energy: the CDM particles carry their own masses (Simulation::update_ide_masses)
        if (cosmo.is_ide()) {
            if (cosmo.get_OmegaMNu() != 0.0)
                fail("cosmology_model = IDE is not implemented with massive neutrinos");
            if (params.particle.baryons and params.particle.baryon_relative_mode != "displacement")
                fail("cosmology_model = IDE needs particle_baryon_relative_mode = displacement (the particle masses "
                     "are those of the energy exchange)");
            if (params.particle.baryons and params.timestep.method == FML::NBODY::TimestepMethod::BullFrog)
                fail("cosmology_model = IDE with particle_baryons = true needs timestep_method = Quinn, Tassev or "
                     "FastPM (BullFrog follows the growth of one fluid)");
        }

        // Velocity friction (GravityModel::velocity_friction): on the CDM+baryon fluid, or with particle_baryons on the
        // CDM particles alone (their own steps, SpeciesTimeStepping.h; not with BullFrog). Not with neutrino particles
        if (grav.has_velocity_friction() and params.particle.neutrinos)
            fail("a cosmology with velocity friction needs particle_neutrinos = false");
        if (grav.has_velocity_friction() and params.particle.baryons and
            params.timestep.method == FML::NBODY::TimestepMethod::BullFrog)
            fail("a cosmology with velocity friction and particle_baryons = true needs timestep_method = Quinn, Tassev "
                 "or FastPM (BullFrog follows the growth of one fluid)");

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
