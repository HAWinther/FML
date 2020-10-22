
//=============================================================
// How to process the parameters from the input file
//=============================================================
#include "ReadParameters.h"

//=============================================================
// The availiable cosmologies
//=============================================================
#include "Cosmology.h"
#include "Cosmology_LCDM.h"
#include "Cosmology_DGP.h"
#include "Cosmology_w0wa.h"

//=============================================================
// The availiable gravity models
//=============================================================
#include "GravityModel.h"
#include "GravityModel_DGP.h"
#include "GravityModel_GR.h"
#include "GravityModel_fofr.h"

#include "Simulation.h"

#include <FML/ParameterMap/ParameterMap.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

using ParameterMap = FML::UTILS::ParameterMap;

//=============================================================
// Design the particle you want to use here. Free choice of types
//=============================================================
constexpr int NDIM = 3;
class Particle {
  public:

    // Things that any particle must have
    double pos[NDIM];
    double * get_pos() { return pos; }
    double vel[NDIM];
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
   
    // --- Optional things ---

    // 1LPT displacement field Psi (needed for 1LPT COLA)
    float Psi_1LPT[NDIM];
    float * get_D_1LPT() { return Psi_1LPT; }
    
    // 2LPT displacement field Psi (needed for 2LPT COLA)
    float Psi_2LPT[NDIM];
    float * get_D_2LPT() { return Psi_2LPT; }

    // Lagrangian position
    // Needed for COLA with scaledependent growth
    double q[NDIM];
    double * get_q() { return q; }

    // Derivative of 1LPT displacement field Psi 
    // This is only needed if you want only 1LPT COLA with scaledependent growth
    // float dPsidloga_1LPT[NDIM];
    // float * get_dDdloga_1LPT() { return dPsidloga_1LPT; }
    
    // ... other things like ID, mass, etc. can be added
};

int main(int argc, char ** argv) {
    if (argc == 1) {
        std::cout << "Missing parameterfile. Run as: ./code input.lua\n";
        return 0;
    }

    //=============================================================
    // Show info about the particle we are using
    //=============================================================
    FML::PARTICLE::info<Particle>();

    //=============================================================
    // Parse the parameterfile
    //=============================================================
    ParameterMap param;
    std::string filename = std::string(argv[1]);
    read_parameterfile(param, filename);
    if (FML::ThisTask == 0)
        param.info();

    //=============================================================
    // Choose the cosmology
    //=============================================================
    auto cosmology_model = param.get<std::string>("cosmology_model");
    std::shared_ptr<Cosmology> cosmo;
    if (cosmology_model == "LCDM")
        cosmo = std::make_shared<CosmologyLCDM>();
    else if (cosmology_model == "w0waCDM")
        cosmo = std::make_shared<Cosmologyw0waCDM>();
    else if (cosmology_model == "DGP")
        cosmo = std::make_shared<CosmologyDGP>();
    else
        throw std::runtime_error("Unknown cosmology [" + cosmology_model + "]");
    cosmo->read_parameters(param);
    cosmo->init();
    cosmo->info();

    //=============================================================
    // Choose the gravity model
    //=============================================================
    auto gravity_model = param.get<std::string>("gravity_model");
    std::shared_ptr<GravityModel<NDIM>> grav;
    if (gravity_model == "GR")
        grav = std::make_shared<GravityModelGR<NDIM>>(cosmo);
    else if (gravity_model == "f(R)")
        grav = std::make_shared<GravityModelFofR<NDIM>>(cosmo);
    else if (gravity_model == "DGP")
        grav = std::make_shared<GravityModelDGP<NDIM>>(cosmo);
    else
        throw std::runtime_error("Unknown gravitymodel [" + gravity_model + "]");
    grav->read_parameters(param);
    grav->init();
    grav->info();

    //=============================================================
    // Run the simulation
    //=============================================================
    NBodySimulation<NDIM, Particle> sim(cosmo, grav);
    sim.read_parameters(param);
    sim.init();
    sim.run();
}

