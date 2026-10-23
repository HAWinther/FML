#include "Models.h"
#include "ParameterSchema.h"
#include "Particles.h"
#include "Simulation.h"

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

/// Run the simulation with the particle type T
template <class T>
void run_simulation(const ParameterMap & param,
                    std::shared_ptr<Cosmology> cosmo,
                    std::shared_ptr<GravityModel<NDIM>> grav) {
    FML::PARTICLE::info<T>();
    NBodySimulation<NDIM, T> sim(cosmo, grav);
    ParameterMap p = param;
    sim.read_parameters(p);
    sim.init();
    sim.run();
}

int main(int argc, char ** argv) {
    if (argc == 1) {
        if (FML::ThisTask == 0) {
            std::cout << "Missing parameterfile. Run as: ./nbody parameterfile.lua\n";
            std::cout << "To see all the parameters run: ./nbody --parameters\n";
        }
        return 0;
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
                COLASOLVER::print_parameter_reference<NDIM>(file);
                std::cout << "Wrote the parameter reference to " << argv[2] << "\n";
            } else {
                COLASOLVER::print_parameter_reference<NDIM>(std::cout);
            }
        }
        return 0;
    }

    //=============================================================
    // Parse the parameterfile
    //=============================================================
    ParameterMap param;
    COLASOLVER::read_parameterfile<NDIM>(param, argv[1]);
    if (FML::ThisTask == 0)
        param.info();

    //=============================================================
    // The cosmology (background) and the gravity model (growth and forces)
    //=============================================================
    auto cosmo = COLASOLVER::make_cosmology(param.get<std::string>("cosmology_model"));
    cosmo->read_parameters(param);
    cosmo->init();
    cosmo->info();

    auto grav = COLASOLVER::make_gravity_model<NDIM>(param.get<std::string>("gravity_model"), cosmo);
    grav->read_parameters(param);
    grav->init();
    grav->info();

    //=============================================================
    // Run the simulation with the smallest particle type that has
    // what we need (see Particles.h). COLA with LPT order < 1 is
    // turned off (see SimulationParameters.h)
    //=============================================================
    const bool use_cola =
        param.get<bool>("simulation_use_cola") and param.get<int>("simulation_cola_LPT_order", 2) >= 1;
    const bool scaledependent_cola = use_cola and param.get<bool>("simulation_use_scaledependent_cola");
    if (scaledependent_cola)
        run_simulation<ParticleCOLAScaleDep<NDIM>>(param, cosmo, grav);
    else if (use_cola)
        run_simulation<ParticleCOLA<NDIM>>(param, cosmo, grav);
    else
        run_simulation<ParticleBasic<NDIM>>(param, cosmo, grav);
}
