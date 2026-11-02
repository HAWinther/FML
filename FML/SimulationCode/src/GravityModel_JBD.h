#ifndef GRAVITYMODEL_JBD_HEADER
#define GRAVITYMODEL_JBD_HEADER

#include "Cosmology.h"
#include "GravityModel.h"

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_BransDicke.h>
#include <FML/Global/Global.h>

#include <memory>

/// Jordan-Brans-Dicke: G_eff/G = g phi_*/phi(a) from the shared quasi-static theory
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_BransDicke.h, phi(a) of the JBD background). The parameter
/// cosmology_JBD_GeffG_today = phi_*/phi(a = 1) sets the strength of gravity today (1 for the Newtonian limit). The
/// force and the growth are the defaults of GravityModel
template <int NDIM>
class GravityModelJBD final : public GravityModel<NDIM> {
  public:
    explicit GravityModelJBD(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "JBD") {
        FML::assert_mpi(
            cosmo->get_name() == "JBD",
            "As currently written the JBD gravity model requires a JBD cosmology as we need phi(a) from there");
    }

    void init() override {
        this->quasi_static = std::make_shared<const FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_BransDicke>(
            this->cosmo->background());
        GravityModel<NDIM>::init();
    }
};

#endif
