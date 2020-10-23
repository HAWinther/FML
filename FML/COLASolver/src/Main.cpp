
//=============================================================
// How to process the parameters from the input file
// Gravity model parameters are read by the GravityModel,
// cosmology parameters are read by the Cosmology model
// and the rest are read by the Simulation
//=============================================================
#include "ReadParameters.h"

//=============================================================
// The availiable cosmologies
//=============================================================
#include "Cosmology.h"
#include "Cosmology_DGP.h"
#include "Cosmology_LCDM.h"
#include "Cosmology_w0wa.h"
#include "Cosmology_JBD.h"

//=============================================================
// The availiable gravity models
//=============================================================
#include "GravityModel.h"
#include "GravityModel_DGP.h"
#include "GravityModel_GR.h"
#include "GravityModel_fofr.h"
#include "GravityModel_JBD.h"
#include "GravityModel_symmetron.h"

#include "Simulation.h"

#include <FML/ParameterMap/ParameterMap.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

using ParameterMap = FML::UTILS::ParameterMap;

//=============================================================
// Design the particle you want to use here. Free choice of
// types so if you for example want to save memory you can change to floats
// or comment out fields if you, say, don't want scaledependent COLA
//
// See src/ExampleParticles.h for more examples
// See FML/ParticleTypes/ReflectOnParticleMethods.h for standard methods
//=============================================================

constexpr int NDIM = 3;

class Particle {
  public:
    //=============================================================
    // Things that any particle must have
    //=============================================================
    double pos[NDIM];
    double * get_pos() { return pos; }
    double vel[NDIM];
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }

    //=============================================================
    // Optional things below:
    //=============================================================

    //=============================================================
    // Add ID to particles (ok to skip this, but if so this will not be
    // present in GADGET output files)
    //=============================================================
    long long int id;
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }

    //=============================================================
    // Initial Lagrangian position
    // Only needed for COLA with scaledependent growth
    // NB: should ideally have same type as [pos] to avoid truncating the
    // precision of pos (these are temporarily swapped by some algorithms)
    //=============================================================
    double q[NDIM];
    double * get_q() { return q; }

    //=============================================================
    // 1LPT displacement field Psi (needed if you want >= 1LPT COLA)
    //=============================================================
    float Psi_1LPT[NDIM];
    float * get_D_1LPT() { return Psi_1LPT; }

    //=============================================================
    // 2LPT displacement field Psi (needed for >= 2LPT COLA)
    //=============================================================
    float Psi_2LPT[NDIM];
    float * get_D_2LPT() { return Psi_2LPT; }

    //=============================================================
    // 3LPT displacement field Psi (needed for >= 3LPT COLA)
    //=============================================================
    // float Psi_3LPTa[NDIM];
    // float Psi_3LPTb[NDIM];
    // float * get_D_3LPTa() { return Psi_3LPTa; }
    // float * get_D_3LPTb() { return Psi_3LPTb; }
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
    else if (cosmology_model == "JBD")
        cosmo = std::make_shared<CosmologyJBD>();
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
    else if (gravity_model == "JBD")
        grav = std::make_shared<GravityModelJBD<NDIM>>(cosmo);
    else if (gravity_model == "Symmetron")
        grav = std::make_shared<GravityModelSymmetron<NDIM>>(cosmo);
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

