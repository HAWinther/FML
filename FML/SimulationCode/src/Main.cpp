#include "ModelCompatibility.h"
#include "Models.h"
#include "ParameterSchema.h"
#include "Particles.h"
#include "RunSimulation.h"
#include "TransferInput.h"

#include <FML/ParameterMap/ParameterMap.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

//=============================================================================
// Usage:
//   ./nbody parameterfile.lua    Run a simulation
//   ./nbody --parameters [file]  Print the reference of all parameters (markdown) [to file]
//   ./nbody --features           Print the optional features compiled in (hdf5, healpix)
//
// The cosmologies and gravity models are listed in Models.h, the parameters in
// ParameterSchema.h and the particle types in Particles.h
//=============================================================================

using ParameterMap = FML::UTILS::ParameterMap;
constexpr int NDIM = NDIM_SPACE;

int run(int argc, char ** argv);

int main(int argc, char ** argv) {
    // Errors (invalid parameters, files that cannot be read, ...) are exceptions: print the message and stop all
    // tasks (an uncaught exception printed a backtrace, and could leave the other tasks waiting)
    try {
        return run(argc, argv);
    } catch (std::exception & e) {
        std::cout << std::flush;
        std::cerr << "ERROR (task " << FML::ThisTask << "): " << e.what() << std::endl;
        FML::abort_mpi(1);
    }
    return 1;
}

int run(int argc, char ** argv) {
    if (argc == 1) {
        if (FML::ThisTask == 0) {
            std::cout << "Missing parameterfile. Run as: ./nbody parameterfile.lua\n";
            std::cout << "To see all the parameters run: ./nbody --parameters\n";
        }
        return 1;
    }
    if (std::strcmp(argv[1], "--features") == 0) {
        // The optional features this executable was compiled with (used by the tests)
        if (FML::ThisTask == 0) {
            std::cout << "FEATURES:";
#ifdef USE_HDF5
            std::cout << " hdf5";
#endif
#ifdef USE_HEALPIX
            std::cout << " healpix";
#endif
            std::cout << "\n";
        }
        return 0;
    }
    if (std::strcmp(argv[1], "--parameters") == 0) {
        if (FML::ThisTask == 0) {
            if (argc > 2) {
                std::ofstream file(argv[2]);
                SIMCODE::print_parameter_reference<NDIM>(file);
                file.close();
                if (not file) {
                    std::cerr << "ERROR: could not write the parameter reference to " << argv[2] << "\n";
                    return 1;
                }
                std::cout << "Wrote the parameter reference to " << argv[2] << "\n";
            } else {
                SIMCODE::print_parameter_reference<NDIM>(std::cout);
            }
        }
        return 0;
    }
    if (std::strncmp(argv[1], "--", 2) == 0) {
        if (FML::ThisTask == 0)
            std::cerr << "Unknown option " << argv[1] << ". Run as: ./nbody parameterfile.lua, ./nbody --parameters "
                      << "[file] or ./nbody --features\n";
        return 1;
    }

    //=============================================================
    // Parse the parameterfile
    //=============================================================
    ParameterMap param;
    SIMCODE::read_parameterfile<NDIM>(param, argv[1]);
    if (FML::ThisTask == 0)
        param.info();

    //=============================================================
    // The cosmology (background) and the gravity model (growth and forces)
    //=============================================================
    auto cosmo = SIMCODE::make_cosmology(param.get<std::string>("cosmology_model"));
    cosmo->read_parameters(param);
    cosmo->init();
    cosmo->info();

    // The transfer functions of the input (CAMB, CLASS or the internal Einstein-Boltzmann solver), loaded once here:
    // the gravity model (growth with massive neutrinos), the initial conditions and the analysis share them
    std::shared_ptr<SIMCODE::TransferProvider> transfers;
    if (SIMCODE::input_has_transfer_tables(param.get<std::string>("ic_type_of_input")))
        transfers = SIMCODE::load_transfers(SIMCODE::make_transfer_request(param, *cosmo, NDIM));

    auto grav = SIMCODE::make_gravity_model<NDIM>(param.get<std::string>("gravity_model"), cosmo);
    grav->set_transfers(transfers);
    grav->read_parameters(param);
    grav->init();
    grav->info();

    //=============================================================
    // The parameters of the simulation: read and checked once here,
    // also against the cosmology and gravity model
    //=============================================================
    SIMCODE::SimulationParameters params;
    params.read(param, grav->is_growth_scaledependent());
    SIMCODE::check_parameters_against_models<NDIM>(params, *cosmo, *grav, transfers != nullptr);
    // The value of simulation_use_scaledependent_cola that is used (see SimulationParameters::read), so that the rest
    // of the code (and the output) sees it
    if (param.contains("simulation_use_scaledependent_cola"))
        param["simulation_use_scaledependent_cola"] = params.simulation.use_scaledependent_cola;

    //=============================================================
    // Run the simulation. By default with ParticleFiducial (all the
    // fields: COLA, q, mass). With USE_MINIMAL_PARTICLES with the
    // smallest particle type that has what we need (see Particles.h).
    // COLA with LPT order < 1 is turned off (see SimulationParameters.h)
    //=============================================================
#ifdef USE_MINIMAL_PARTICLES
    const bool use_cola = params.simulation.use_cola;
    const bool scaledependent_cola = params.simulation.use_scaledependent_cola;
    // Sheet interpolation needs the Lagrangian position q
    const bool need_q = params.force.density_method == "sheet" or params.force.density_method == "sheet_nufft";
    // The mass-encoded baryon-CDM relative mode needs a mass in the particles (density assignment only, so no q)
    const bool need_mass = params.particle.baryons and params.particle.baryon_relative_mode != "displacement";
    if (need_mass) {
        if (scaledependent_cola)
            run_simulation<ParticleWithMass<ParticleCOLAScaleDep<NDIM>>>(param, params, cosmo, grav);
        else if (use_cola)
            run_simulation<ParticleWithMass<ParticleCOLA<NDIM>>>(param, params, cosmo, grav);
        else
            run_simulation<ParticleWithMass<ParticleBasic<NDIM>>>(param, params, cosmo, grav);
    } else if (scaledependent_cola or (use_cola and need_q))
        run_simulation<ParticleCOLAScaleDep<NDIM>>(param, params, cosmo, grav);
    else if (use_cola)
        run_simulation<ParticleCOLA<NDIM>>(param, params, cosmo, grav);
    else if (need_q)
        run_simulation<ParticleBasicQ<NDIM>>(param, params, cosmo, grav);
    else
        run_simulation<ParticleBasic<NDIM>>(param, params, cosmo, grav);
#else
    run_simulation<ParticleFiducial<NDIM>>(param, params, cosmo, grav);
#endif
    return 0;
}
