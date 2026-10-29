#include <FML/Cosmology/Models/CoupledQuintessence.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    // =============================================================================
    // Background
    // =============================================================================

    CoupledDarkSector::CoupledDarkSector(double lambda_in,
                                         double beta_in,
                                         const std::vector<std::shared_ptr<const Species>> & others_in,
                                         double OmegaK_in,
                                         double Omega_c,
                                         double Omega_phi)
        : lambda(lambda_in), beta(beta_in), OmegaK(OmegaK_in), others(others_in) {
        if (!(Omega_phi > 0.0) || !(Omega_c > 0.0))
            throw std::invalid_argument("coupled quintessence: need Omega_c, Omega_phi > 0");
        C = Omega_c;
        for (int it = 0;; it++) {
            // V0 such that rho_phi(0) = Omega_phi (rho_phi(0) increases with V0)
            auto f = [&](double lnV0) {
                V0 = std::exp(lnV0);
                solve();
                return rho_phi(0.0) / Omega_phi - 1.0;
            };
            double lo = std::log(Omega_phi), hi = lo;
            int n = 0;
            while (f(lo) > 0.0) {
                lo -= 1.0;
                if (++n > 60)
                    throw std::runtime_error("coupled quintessence: no V0");
            }
            while (f(hi) < 0.0) {
                hi += 1.0;
                if (++n > 120)
                    throw std::runtime_error("coupled quintessence: no V0");
            }
            V0 = std::exp(Utils::find_root(f, lo, hi, 1e-14, 1e-14));
            solve();
            // CDM normalisation (rho_c(0) depends on phi(0))
            const double r = Omega_c / rho_c(0.0);
            C *= r;
            if (std::fabs(r - 1.0) < 1e-13)
                break;
            if (it > 100)
                throw std::runtime_error("coupled quintessence: background iteration did not converge");
        }
        solve();
    }

    double CoupledDarkSector::rho_o(double x) const {
        double r = OmegaK * std::exp(-2.0 * x);
        for (const auto & s : others)
            r += s->rho(x);
        return r;
    }
    double CoupledDarkSector::drho_o(double x) const {
        double r = -2.0 * OmegaK * std::exp(-2.0 * x);
        for (const auto & s : others)
            r += s->drhodx(x);
        return r;
    }

    double CoupledDarkSector::E2(double x, double f, double df) const {
        return (rho_o(x) + C * std::exp(-3.0 * x - beta * f) + v(f)) / (1.0 - df * df / 6.0);
    }
    // E2' = rho_o' + rho_c' + rho_phi' = rho_o' - 3 rho_c - E^2 phi_x^2 (the exchange cancels)
    double CoupledDarkSector::dlnE(double x, double f, double df) const {
        const double e2 = E2(x, f, df), rc = C * std::exp(-3.0 * x - beta * f);
        return (drho_o(x) - 3.0 * rc - e2 * df * df) / (2.0 * e2);
    }
    double CoupledDarkSector::ddphi(double x, double f, double df) const {
        const double e2 = E2(x, f, df), rc = C * std::exp(-3.0 * x - beta * f);
        return -(3.0 + dlnE(x, f, df)) * df - 3.0 * v_phi(f) / e2 + 3.0 * beta * rc / e2;
    }

    void CoupledDarkSector::solve() {
        ODEFunction rhs = [&](double x, const double * y, double * dydx) {
            dydx[0] = y[1];
            dydx[1] = ddphi(x, y[0], y[1]);
            return GSL_SUCCESS;
        };
        const auto xs = Utils::linspace(x_ini, x_fin, 14001);
        ODESolver ode(1e-4, 1e-14, 1e-12);
        ode.solve(rhs, xs, {0.0, 0.0}, gsl_odeiv2_step_rk8pd);
        phi_spline.create(xs, ode.get_data_by_component(0), "phi");
        dphi_spline.create(xs, ode.get_data_by_component(1), "dphidx");
    }

    double CoupledDarkSector::phi(double x) const {
        if (x < x_ini)
            return 0.0;
        if (x > x_fin)
            throw std::out_of_range("coupled quintessence: x beyond the end of the field solution");
        return phi_spline(x);
    }
    double CoupledDarkSector::dphi(double x) const {
        if (x < x_ini)
            return 0.0;
        if (x > x_fin)
            throw std::out_of_range("coupled quintessence: x beyond the end of the field solution");
        return dphi_spline(x);
    }
    double CoupledDarkSector::ddphi(double x) const { return ddphi(x, phi(x), dphi(x)); }
    double CoupledDarkSector::rho_c(double x) const { return C * std::exp(-3.0 * x - beta * phi(x)); }
    double CoupledDarkSector::E2(double x) const { return E2(x, phi(x), dphi(x)); }
    double CoupledDarkSector::dlnE(double x) const { return dlnE(x, phi(x), dphi(x)); }
    double CoupledDarkSector::rho_phi(double x) const {
        const double f = phi(x), df = dphi(x);
        return E2(x, f, df) * df * df / 6.0 + v(f);
    }
    double CoupledDarkSector::p_phi(double x) const {
        const double f = phi(x), df = dphi(x);
        return E2(x, f, df) * df * df / 6.0 - v(f);
    }

    // rho_c' = -(3 + beta phi_x) rho_c,  rho_c'' = [(3 + beta phi_x)^2 - beta phi_xx] rho_c
    double CoupledCDM::drhodx(double x) const { return -(3.0 + d->beta * d->dphi(x)) * d->rho_c(x); }
    double CoupledCDM::ddrhoddx(double x) const {
        const double a = 3.0 + d->beta * d->dphi(x);
        return (a * a - d->beta * d->ddphi(x)) * d->rho_c(x);
    }
    // rho_phi' = -E^2 phi_x^2 + beta phi_x rho_c
    double CoupledField::drhodx(double x) const {
        const double df = d->dphi(x);
        return -d->E2(x) * df * df + d->beta * df * d->rho_c(x);
    }
    // rho_phi'' = -(E2' phi_x^2 + 2 E^2 phi_x phi_xx) + beta (phi_xx rho_c + phi_x rho_c')
    double CoupledField::ddrhoddx(double x) const {
        const double df = d->dphi(x), ddf = d->ddphi(x), e2 = d->E2(x), de2 = 2.0 * d->dlnE(x) * e2;
        const double rc = d->rho_c(x), drc = -(3.0 + d->beta * df) * rc;
        return -(de2 * df * df + 2.0 * e2 * df * ddf) + d->beta * (ddf * rc + df * drc);
    }

    std::string CoupledField::info() const {
        std::ostringstream os;
        os << "coupled quintessence (lambda = " << d->lambda << ", beta = " << d->beta << ", V0 = " << d->V0
           << "; Omega = " << rho(0.0) << ", w0 = " << p(0.0) / rho(0.0) << ", phi0 = " << d->phi(0.0) << " M_pl)";
        return os.str();
    }

    // =============================================================================
    // Perturbations: CDM and field, (delta_c, v_c, dphi, w)
    // =============================================================================

    namespace {
        class CoupledPerturbations : public SpeciesPerturbations {
          public:
            explicit CoupledPerturbations(const CoupledDarkSector & d) : d(d) {}
            std::string name() const override { return "coupled CDM + quintessence"; }
            int n_vars(int) const override { return 4; }

            enum { RC, DPHI, VPHI, VPP, G3, BRC, RPHI, N_CACHE };
            int cache_size() const override { return N_CACHE; }
            void update_cache(const ModeAtX & b, double * cc) const override {
                const double f = d.phi(b.x), rc = d.rho_c(b.x);
                cc[RC] = b.H0a2 * rc;
                cc[DPHI] = d.dphi(b.x);
                cc[VPHI] = d.v_phi(f);
                cc[VPP] = d.v_phiphi(f);
                cc[G3] = 3.0 * b.H0a2 / (b.Hp * b.Hp); // 3/E^2
                cc[BRC] = d.beta * rc;
                cc[RPHI] = b.H0a2 * d.rho_phi(b.x);
            }
            void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
                // CDM: the isocurvature values if set (the dark energy has no isocurvature mode of its own and starts
                // from the adiabatic form with the Psi of the mode, as the other species not in IsoValues)
                y[0] = ic.iso ? ic.iso->delta_c : -1.5 * ic.Psi;
                y[1] = ic.iso ? ic.iso->v_c : -0.5 * b.ckH * ic.Psi;
                y[2] = y[3] = 0.0;
            }
            Moments moments(const ModeAtX & b, const double * cc, int, const double * y) const override {
                Moments m;
                const double kin = b.Hp * b.Hp * cc[DPHI] * y[3] / 3.0, pot = b.H0a2 * cc[VPHI] * y[2];
                m.drho = cc[RC] * y[0] + kin + pot;
                m.dp = kin - pot;
                m.flux = -cc[RC] * y[1] + b.ckH * b.Hp * b.Hp * cc[DPHI] * y[2] / 3.0;
                return m;
            }
            void rhs(const ModeAtX & b, const double * cc, int, const MetricAtX & m, const double * y, double * dydx)
                const override {
                const double beta = d.beta, ckH = b.ckH, Fx = cc[DPHI];
                const double dc = y[0], vc = y[1], dphi = y[2], w = y[3];
                const double dphi_x = w + Fx * m.Psi;
                dydx[0] = ckH * vc - 3.0 * m.dPhi - beta * dphi_x;
                dydx[1] = -(1.0 - beta * Fx) * vc - ckH * m.Psi + beta * ckH * dphi;
                dydx[2] = dphi_x;
                dydx[3] = -(2.0 + b.dHpH) * w - (ckH * ckH + cc[G3] * cc[VPP]) * dphi - 3.0 * Fx * m.dPhi -
                          cc[G3] * (cc[VPHI] - cc[BRC]) * m.Psi + cc[G3] * cc[BRC] * dc;
            }
            std::vector<std::string> output_names() const override {
                return {"delta_cdm", "v_cdm", "delta_phi", "delta_de"};
            }
            void outputs(const ModeAtX & b, const double * cc, int, const double * y, double * out) const override {
                out[0] = y[0];
                out[1] = y[1];
                out[2] = y[2];
                out[3] = (b.Hp * b.Hp * cc[DPHI] * y[3] / 3.0 + b.H0a2 * cc[VPHI] * y[2]) / cc[RPHI];
            }

          private:
            const CoupledDarkSector & d;
        };
    } // namespace

    std::unique_ptr<SpeciesPerturbations> CoupledField::perturbations(const Perturbations &) const {
        return std::make_unique<CoupledPerturbations>(*d);
    }

} // namespace FML::COSMOLOGY
