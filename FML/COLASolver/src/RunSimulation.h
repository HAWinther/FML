#ifndef RUNSIMULATION_HEADER
#define RUNSIMULATION_HEADER

//=============================================================================
// Running the simulation with a given particle type. The simulation is compiled
// once for each particle type in its own translation unit (RunSimulation_*.cpp)
// so that these can be compiled in parallel; Main.cpp only declares them.
//=============================================================================

#include "Cosmology.h"
#include "GravityModel.h"
#include "Particles.h"

#include <FML/ParameterMap/ParameterMap.h>

#include <memory>

#ifndef NDIM_SPACE
#define NDIM_SPACE 3
#endif

/// Read the parameters, initialize and run the simulation with particles of type T
template <class T>
void run_simulation(const FML::UTILS::ParameterMap & param,
                    std::shared_ptr<Cosmology> cosmo,
                    std::shared_ptr<GravityModel<NDIM_SPACE>> grav);

/// The particle types the simulation is compiled for (see Particles.h and the choice in Main.cpp)
#define COLASOLVER_PARTICLE_TYPES(X)                                                                                   \
    X(ParticleWithMass<ParticleCOLAScaleDep<NDIM_SPACE>>)                                                              \
    X(ParticleWithMass<ParticleCOLA<NDIM_SPACE>>)                                                                      \
    X(ParticleWithMass<ParticleBasic<NDIM_SPACE>>)                                                                     \
    X(ParticleCOLAScaleDep<NDIM_SPACE>)                                                                                \
    X(ParticleCOLA<NDIM_SPACE>)                                                                                        \
    X(ParticleBasicQ<NDIM_SPACE>)                                                                                      \
    X(ParticleBasic<NDIM_SPACE>)

#define COLASOLVER_DECLARE_RUN_SIMULATION(T)                                                                           \
    extern template void run_simulation<T>(const FML::UTILS::ParameterMap &,                                           \
                                           std::shared_ptr<Cosmology>,                                                 \
                                           std::shared_ptr<GravityModel<NDIM_SPACE>>);
COLASOLVER_PARTICLE_TYPES(COLASOLVER_DECLARE_RUN_SIMULATION)
#undef COLASOLVER_DECLARE_RUN_SIMULATION

#endif
