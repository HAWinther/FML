#ifndef RUNSIMULATIONIMPL_HEADER
#define RUNSIMULATIONIMPL_HEADER

//=============================================================================
// The definition of run_simulation (included by the RunSimulation_*.cpp files)
//=============================================================================

#include "RunSimulation.h"
#include "Simulation.h"

#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

template <class T>
void run_simulation(const FML::UTILS::ParameterMap & param,
                    std::shared_ptr<Cosmology> cosmo,
                    std::shared_ptr<GravityModel<NDIM_SPACE>> grav) {
    FML::PARTICLE::info<T>();
    NBodySimulation<NDIM_SPACE, T> sim(cosmo, grav);
    FML::UTILS::ParameterMap p = param;
    sim.read_parameters(p);
    sim.init();
    sim.run();
}

#define COLASOLVER_INSTANTIATE_RUN_SIMULATION(T)                                                                       \
    template void run_simulation<T>(const FML::UTILS::ParameterMap &,                                                  \
                                    std::shared_ptr<Cosmology>,                                                        \
                                    std::shared_ptr<GravityModel<NDIM_SPACE>>);

#endif
