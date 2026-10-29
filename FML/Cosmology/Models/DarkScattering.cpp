#include <FML/Cosmology/Models/DarkScattering.h>
#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using Constants::c;

    FluidDarkEnergy::FluidDarkEnergy(double Omega0_in, double w_in, double cs2_in, double xi_in, double rho_crit0_in)
        : Omega0(Omega0_in), w(w_in), cs2(cs2_in), xi(xi_in), rho_crit0(rho_crit0_in) {
        if (std::fabs(1.0 + w) < 1e-6)
            throw std::invalid_argument("dark energy fluid: w = -1 has no perturbations (use lcdm)");
        if (!(cs2 > 0.0))
            throw std::invalid_argument("dark energy fluid: need cs2 > 0");
        if (!(xi >= 0.0))
            throw std::invalid_argument("dark scattering: need xi >= 0");
    }

    std::string FluidDarkEnergy::info() const {
        std::ostringstream os;
        os << "dark energy fluid (Omega = " << Omega0 << ", w = " << w << ", cs2 = " << cs2
           << ", xi = " << xi / 0.0560958865 << " b/GeV)";
        return os.str();
    }

    namespace {
        // CDM and dark-energy fluid, (delta_c, v_c, delta_DE[, u_DE])
        class DarkScatteringPerturbations : public SpeciesPerturbations {
          public:
            DarkScatteringPerturbations(const FluidDarkEnergy & de, const Perturbations & p) : de(de), pert(p) {
                const auto & o = p.get_options();
                auto r = [&](double x) { return rate_DE(x) / p.get_cosmo().Hp_of_x(x) - lock_ratio; };
                if (de.xi == 0.0 || r(o.x_start) <= 0.0)
                    locked_start = false;
                else {
                    locked_start = true;
                    x_unlock = r(o.x_end) > 0.0 ? 1e30 : Utils::find_root(r, o.x_start, o.x_end);
                }
            }
            std::string name() const override { return "dark scattering (CDM + dark energy)"; }
            std::string info() const override {
                std::ostringstream os;
                os << name();
                if (locked_start)
                    os << " (velocities locked until z = " << std::exp(-x_unlock) - 1.0 << ")";
                return os.str();
            }
            std::vector<double> switch_points(double) const override {
                if (locked_start && x_unlock < 1e29)
                    return {x_unlock};
                return {};
            }
            bool locked(int mode) const { return locked_start && mode == 0; }
            int n_vars(int mode) const override { return locked(mode) ? 3 : 4; }

            // Gamma_DE = a c xi rho_c [1/s]
            double rate_DE(double x) const {
                return std::exp(x) * c * de.xi * pert.get_cosmo().rhoCDM(x) * de.rho_crit0;
            }

            enum { RC, RDE, GDE, GC, N_CACHE }; // H0^2 a^2 rho_c, H0^2 a^2 rho_DE, Gamma_DE/Hp, Gamma_c/Hp
            int cache_size() const override { return N_CACHE; }
            void update_cache(const ModeAtX & b, double * cc) const override {
                const double rc = pert.get_cosmo().rhoCDM(b.x), rde = de.rho(b.x);
                cc[RC] = b.H0a2 * rc;
                cc[RDE] = b.H0a2 * rde;
                cc[GDE] = rate_DE(b.x) / b.Hp;
                cc[GC] = (1.0 + de.w) * rde / rc * cc[GDE];
            }
            void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
                // CDM: the isocurvature values if set (the dark energy has no isocurvature mode of its own and starts
                // from the adiabatic form with the Psi of the mode, as the other species not in IsoValues)
                y[0] = ic.iso ? ic.iso->delta_c : -1.5 * ic.Psi;
                y[1] = ic.iso ? ic.iso->v_c : -0.5 * b.ckH * ic.Psi;
                y[2] = -1.5 * (1.0 + de.w) * ic.Psi; // adiabatic: delta_DE/(1 + w) = delta_c
                if (!locked(0))
                    y[3] = 0.5 * b.ckH * ic.Psi; // u = theta/(ck) = -v of the dark matter
            }
            void switch_mode(const ModeAtX &, const double *, int, const double * yo, double * yn) const override {
                yn[0] = yo[0];
                yn[1] = yo[1];
                yn[2] = yo[2];
                yn[3] = -yo[1];
            }
            Moments moments(const ModeAtX & b, const double * cc, int mode, const double * y) const override {
                const double w = de.w, cs2 = de.cs2, u = locked(mode) ? -y[1] : y[3];
                Moments m;
                m.drho = cc[RC] * y[0] + cc[RDE] * y[2];
                m.dp = cc[RDE] * (cs2 * y[2] + 3.0 * (1.0 + w) * (cs2 - w) * u / b.ckH);
                m.flux = -cc[RC] * y[1] + (1.0 + w) * cc[RDE] * u;
                return m;
            }
            void rhs(const ModeAtX & b,
                     const double * cc,
                     int mode,
                     const MetricAtX & m,
                     const double * y,
                     double * dydx) const override {
                const double w = de.w, cs2 = de.cs2, ckH = b.ckH;
                const bool lk = locked(mode);
                const double vc = y[1], u = lk ? -vc : y[3];
                dydx[0] = ckH * vc - 3.0 * m.dPhi;
                dydx[1] = -vc - ckH * m.Psi + (lk ? 0.0 : cc[GC] * (-u - vc));
                dydx[2] = -(1.0 + w) * (ckH * u + 3.0 * m.dPhi) - 3.0 * (cs2 - w) * (y[2] + 3.0 * (1.0 + w) * u / ckH);
                if (!lk)
                    dydx[3] = -(1.0 - 3.0 * cs2) * u + ckH * (cs2 * y[2] / (1.0 + w) + m.Psi) - cc[GDE] * (u + vc);
            }
            std::vector<std::string> output_names() const override {
                return {"delta_cdm", "v_cdm", "delta_de", "v_de"};
            }
            void outputs(const ModeAtX &, const double *, int mode, const double * y, double * out) const override {
                out[0] = y[0];
                out[1] = y[1];
                out[2] = y[2];
                out[3] = locked(mode) ? y[1] : -y[3];
            }

          private:
            const FluidDarkEnergy & de;
            const Perturbations & pert;
            const double lock_ratio = 200.0;
            bool locked_start = false;
            double x_unlock = 1e30;
        };
    } // namespace

    std::unique_ptr<SpeciesPerturbations> FluidDarkEnergy::perturbations(const Perturbations & pert) const {
        return std::make_unique<DarkScatteringPerturbations>(*this, pert);
    }

} // namespace FML::COSMOLOGY
