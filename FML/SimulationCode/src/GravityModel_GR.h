#ifndef GRAVITYMODEL_GR_HEADER
#define GRAVITYMODEL_GR_HEADER

#include "GravityModel.h"

#include <memory>

/// General Relativity: G_eff/G = 1 and the force from 1.5 OmegaM a delta (the defaults of GravityModel)
template <int NDIM>
class GravityModelGR final : public GravityModel<NDIM> {
  public:
    double GeffOverG_highk(double /*a*/) const override { return 1.; }
    double short_range_comoving_range(double /*a*/) const override { return 0.; }
    explicit GravityModelGR(std::shared_ptr<Cosmology> cosmo) : GravityModel<NDIM>(cosmo, "GR") {}
};

#endif
