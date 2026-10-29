#include <FML/Cosmology/Models/InteractingDarkEnergy.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    InteractingDarkEnergy::InteractingDarkEnergy(double Omega_DE,
                                                 double Omega_c_in,
                                                 double w_in,
                                                 double xi_in,
                                                 double cs2_in)
        : Omega0(Omega_DE), Omega_c(Omega_c_in), w(w_in), xi(xi_in), cs2(cs2_in), L(3.0 * (1.0 + w_in) + xi_in) {
        if (std::fabs(1.0 + w) < 1e-6)
            throw std::invalid_argument("interacting dark energy: need w != -1");
        if (!(cs2 > 0.0))
            throw std::invalid_argument("interacting dark energy: need cs2 > 0");
        // the dark matter density must be positive until today (with xi < 0 it vanishes in the future)
        InteractingCDM c(std::shared_ptr<const InteractingDarkEnergy>(this, [](const InteractingDarkEnergy *) {}));
        for (double x = -25.0; x <= 1e-12; x += 0.01)
            if (!(c.rho(x) > 0.0))
                throw std::invalid_argument("interacting dark energy: rho_c < 0 at a = " + std::to_string(std::exp(x)) +
                                            " (too much energy flows out of the dark matter)");
    }

    // rho_c a^3 = Omega_c + xi Omega_DE g(x),  g = int_0^x e^{-s x'} dx' = -expm1(-s x)/s,  s = 3w + xi
    double InteractingCDM::rho(double x) const {
        const double s = 3.0 * de->w + de->xi;
        const double g = (s == 0.0) ? x : -std::expm1(-s * x) / s;
        return std::exp(-3.0 * x) * (de->Omega_c + de->xi * de->Omega0 * g);
    }

    std::string InteractingDarkEnergy::info() const {
        std::ostringstream os;
        os << "interacting dark energy (Q = xi H rho_DE: xi = " << xi << ", w = " << w << ", cs2 = " << cs2
           << ", doom factor d = " << doom() << ")";
        return os.str();
    }

    namespace {
        // (delta_c, v_c, delta_DE, u_DE), see InteractingDarkEnergy.h
        class IDEPerturbations : public SpeciesPerturbations {
          public:
            IDEPerturbations(const InteractingDarkEnergy & de, const Perturbations & p) : de(de), pert(p) {}
            std::string name() const override { return "interacting dark energy (CDM + dark energy)"; }
            int n_vars(int) const override { return 4; }

            enum { RC, RDE, N_CACHE }; // H0^2 a^2 rho_c, H0^2 a^2 rho_DE
            int cache_size() const override { return N_CACHE; }
            void update_cache(const ModeAtX & b, double * cc) const override {
                cc[RC] = b.H0a2 * pert.get_cosmo().rhoCDM(b.x);
                cc[RDE] = b.H0a2 * de.rho(b.x);
            }
            void initial_conditions(const ModeAtX & b, const InitialData & ic, double * y) const override {
                // CDM: the isocurvature values if set (the dark energy has no isocurvature mode of its own and starts
                // from the adiabatic form with the Psi of the mode, as the other species not in IsoValues)
                y[0] = ic.iso ? ic.iso->delta_c : -1.5 * ic.Psi;
                y[1] = ic.iso ? ic.iso->v_c : -0.5 * b.ckH * ic.Psi;
                y[2] = -1.5 * de.L / 3.0 * ic.Psi; // adiabatic: delta_DE / delta_c = rho_DE'/rho_DE / (rho_c'/rho_c)
                y[3] = 0.5 * b.ckH * ic.Psi;       // u = -v_c
            }
            double dp_over_rho(const ModeAtX & b, const double * y) const {
                const double w = de.w, cs2 = de.cs2, d = de.doom();
                return cs2 * y[2] + 3.0 * (1.0 + w) * (1.0 + d) * (cs2 - w) * y[3] / b.ckH;
            }
            Moments moments(const ModeAtX & b, const double * cc, int, const double * y) const override {
                Moments m;
                m.drho = cc[RC] * y[0] + cc[RDE] * y[2];
                m.dp = cc[RDE] * dp_over_rho(b, y);
                m.flux = -cc[RC] * y[1] + (1.0 + de.w) * cc[RDE] * y[3];
                return m;
            }
            void rhs(const ModeAtX & b, const double * cc, int, const MetricAtX & m, const double * y, double * dydx)
                const override {
                const double w = de.w, cs2 = de.cs2, xi = de.xi, d = de.doom(), ckH = b.ckH;
                const double dc = y[0], vc = y[1], dde = y[2], u = y[3];
                const double E = -ckH * vc / 3.0 + m.dPhi; // (theta_c - 3 Phi_MB')/(3 Hp)
                dydx[0] = ckH * vc - 3.0 * m.dPhi + xi * cc[RDE] / cc[RC] * (dde - dc + E);
                dydx[1] = -vc - ckH * m.Psi;
                dydx[2] = -(1.0 + w) * (ckH * u + 3.0 * m.dPhi) -
                          3.0 * (cs2 - w) * (dde + 3.0 * (1.0 + w) * (1.0 + d) * u / ckH) - xi * E;
                dydx[3] = (3.0 * cs2 - 1.0 + 3.0 * d * (cs2 - w)) * u + ckH * (cs2 * dde / (1.0 + w) + m.Psi) + xi * u +
                          3.0 * d * vc;
            }
            std::vector<std::string> output_names() const override {
                return {"delta_cdm", "v_cdm", "delta_de", "v_de"};
            }
            void outputs(const ModeAtX &, const double *, int, const double * y, double * out) const override {
                out[0] = y[0];
                out[1] = y[1];
                out[2] = y[2];
                out[3] = -y[3];
            }

          private:
            const InteractingDarkEnergy & de;
            const Perturbations & pert;
        };
    } // namespace

    std::unique_ptr<SpeciesPerturbations> InteractingDarkEnergy::perturbations(const Perturbations & pert) const {
        return std::make_unique<IDEPerturbations>(*this, pert);
    }

} // namespace FML::COSMOLOGY
