#include <FML/Cosmology/Models/GeneralRelativity.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>

namespace FML::COSMOLOGY {

using Constants::c;

double GeneralRelativity::E2(double x, const BackgroundCosmology& bg) const {
  return bg.rho_species(x) + bg.rhoK(x);
}
double GeneralRelativity::dE2dx(double x, const BackgroundCosmology& bg) const {
  return bg.drho_species(x) + bg.drhoKdx(x);
}
double GeneralRelativity::ddE2ddx(double x, const BackgroundCosmology& bg) const {
  return bg.ddrho_species(x) + bg.ddrhoKddx(x);
}

MetricAtX GeneralRelativity::metric(const ModeAtX& b, const double*, const double* y, const Moments& total,
                                    double* dydx) const {
  const double Phi = y[0];
  MetricAtX m;
  m.Phi = Phi;
  m.Psi = -Phi - 3.0 * total.shear / (c * c * b.k * b.k * b.s2);
  m.dPhi = m.Psi - b.ckH * b.ckH * b.curv * Phi / 3.0 + total.drho / (2.0 * b.Hp * b.Hp);
  dydx[0] = m.dPhi;
  return m;
}

} // namespace FML::COSMOLOGY
