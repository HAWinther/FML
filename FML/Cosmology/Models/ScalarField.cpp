#include <FML/Cosmology/Models/ScalarField.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY {

namespace {
  // Field perturbation dphi and w = dphi_x - phi_x Psi (see ScalarField.h)
  class ScalarFieldPerturbations : public SpeciesPerturbations {
    public:
      explicit ScalarFieldPerturbations(const ScalarFieldSpecies& s) : q(s) {}
      std::string name() const override { return q.name(); }
      int n_vars(int) const override { return 2; }

      enum { DPHI, VPHI, VPP, E2INV3, RHO, N_CACHE };   // phi_x, v_phi, v_phiphi, 3/E^2, H0^2 a^2 rho_phi
      int cache_size() const override { return N_CACHE; }
      void update_cache(const ModeAtX& b, double* cc) const override {
        const double phi = q.phi(b.x);
        cc[DPHI] = q.dphidx(b.x);
        cc[VPHI] = q.v_phi(phi);
        cc[VPP] = q.v_phiphi(phi);
        cc[E2INV3] = 3.0 * b.H0a2 / (b.Hp * b.Hp);
        cc[RHO] = b.H0a2 * q.rho(b.x);
      }
      void initial_conditions(const ModeAtX&, const InitialData&, double* y) const override { y[0] = y[1] = 0.0; }

      Moments moments(const ModeAtX& b, const double* cc, int, const double* y) const override {
        Moments m;
        const double kin = b.Hp * b.Hp * cc[DPHI] * y[1] / 3.0, pot = b.H0a2 * cc[VPHI] * y[0];
        m.drho = kin + pot;
        m.dp = kin - pot;
        m.flux = b.ckH * b.Hp * b.Hp * cc[DPHI] * y[0] / 3.0;
        return m;
      }
      void rhs(const ModeAtX& b, const double* cc, int, const MetricAtX& m, const double* y, double* dydx) const override {
        const double dphi = y[0], w = y[1];
        dydx[0] = w + cc[DPHI] * m.Psi;
        dydx[1] = -(2.0 + b.dHpH) * w - (b.ckH * b.ckH + cc[E2INV3] * cc[VPP]) * dphi - 3.0 * cc[DPHI] * m.dPhi -
                  cc[E2INV3] * cc[VPHI] * m.Psi;
      }
      std::vector<std::string> output_names() const override { return {"delta_phi", "delta_de"}; }
      void outputs(const ModeAtX& b, const double* cc, int mode, const double* y, double* out) const override {
        out[0] = y[0];
        out[1] = moments(b, cc, mode, y).drho / cc[RHO];
      }

    private:
      const ScalarFieldSpecies& q;
  };
}

ScalarFieldSpecies::ScalarFieldSpecies(const std::vector<std::shared_ptr<const Species>>& others_in, double OmegaK_in,
                                       double OmegaL_in)
  : others(others_in), OmegaK(OmegaK_in), OmegaL_(OmegaL_in) {}

double ScalarFieldSpecies::rho_others(double x) const {
  double r = OmegaK * std::exp(-2.0 * x) + OmegaL_;
  for (const auto& s : others) r += s->rho(x);
  return r;
}
double ScalarFieldSpecies::drho_others(double x) const {
  double r = -2.0 * OmegaK * std::exp(-2.0 * x);
  for (const auto& s : others) r += s->drhodx(x);
  return r;
}
double ScalarFieldSpecies::ddrho_others(double x) const {
  double r = 4.0 * OmegaK * std::exp(-2.0 * x);
  for (const auto& s : others) r += s->ddrhoddx(x);
  return r;
}

double ScalarFieldSpecies::E2(double x, double phi, double dphi) const {
  return (rho_others(x) + v(phi)) / (1.0 - dphi * dphi / 6.0);
}

double ScalarFieldSpecies::ddphi(double x, double phi, double dphi) const {
  const double e2 = E2(x, phi, dphi);
  const double dlnE = (drho_others(x) - e2 * dphi * dphi) / (2.0 * e2);
  return -(3.0 + dlnE) * dphi - 3.0 * v_phi(phi) / e2;
}

void ScalarFieldSpecies::solve(double phi_ini, int n, double x_end, double abserr, double relerr) {
  phi_ini_ = phi_ini;
  x_fin = x_end;
  ODEFunction rhs = [&](double x, const double* y, double* dydx) {
    dydx[0] = y[1];
    dydx[1] = ddphi(x, y[0], y[1]);
    return GSL_SUCCESS;
  };
  const auto xs = Utils::linspace(x_ini, x_end, n);
  ODESolver ode(1e-4, abserr, relerr);
  ode.solve(rhs, xs, {phi_ini, 0.0}, gsl_odeiv2_step_rk8pd);
  phi_spline.create(xs, ode.get_data_by_component(0), "phi");
  dphi_spline.create(xs, ode.get_data_by_component(1), "dphidx");
}

double ScalarFieldSpecies::phi(double x) const {
  if (x < x_ini) return phi_ini_;
  if (x > x_fin) throw std::out_of_range(name() + ": x beyond the end of the field solution");
  return phi_spline(x);
}
double ScalarFieldSpecies::dphidx(double x) const {
  if (x < x_ini) return 0.0;
  if (x > x_fin) throw std::out_of_range(name() + ": x beyond the end of the field solution");
  return dphi_spline(x);
}

double ScalarFieldSpecies::E2(double x) const { return E2(x, phi(x), dphidx(x)); }

double ScalarFieldSpecies::rho(double x) const {
  const double f = phi(x), df = dphidx(x);
  return E2(x, f, df) * df * df / 6.0 + v(f);
}
double ScalarFieldSpecies::p(double x) const {
  const double f = phi(x), df = dphidx(x);
  return E2(x, f, df) * df * df / 6.0 - v(f);
}
// rho' = -3 (rho + p) = -E^2 phi_x^2
double ScalarFieldSpecies::drhodx(double x) const {
  const double f = phi(x), df = dphidx(x);
  return -E2(x, f, df) * df * df;
}
// rho'' = -(E2' phi_x^2 + 2 E^2 phi_x phi_xx), E2' = rho_o' + rho_phi'
double ScalarFieldSpecies::ddrhoddx(double x) const {
  const double f = phi(x), df = dphidx(x), e2 = E2(x, f, df);
  const double de2 = drho_others(x) - e2 * df * df;
  return -(de2 * df * df + 2.0 * e2 * df * ddphi(x, f, df));
}

std::unique_ptr<SpeciesPerturbations> ScalarFieldSpecies::perturbations(const Perturbations&) const {
  if (!perturbed) return nullptr;
  return std::make_unique<ScalarFieldPerturbations>(*this);
}

} // namespace FML::COSMOLOGY
