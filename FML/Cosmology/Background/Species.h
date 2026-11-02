#ifndef FML_COSMOLOGY_BACKGROUND_SPECIES_H
#define FML_COSMOLOGY_BACKGROUND_SPECIES_H

// =====================================================================================================================
// Species: one component of a ComponentBackground, by its energy density and pressure as functions of x = ln a in
// units of today's critical density rho_c0 (pressure: rho_c0 c^2).
//
// A species that only exchanges energy with itself obeys the continuity equation, drho/dx = -3 (rho + p), which is the
// default of drhodx; coupled species (interacting dark energy, ...) override it. d2rhodx2, the second derivative (used
// by the perturbations of some theories), is a 5-point central difference of drhodx (step 1e-3) by default; the
// species here give it analytically.
//
// The standard species are here:
//   PowerLawSpecies         rho = Omega0 a^{-n}: baryons, CDM (n = 3), photons, massless neutrinos (n = 4), Lambda
//   CPLDarkEnergy           w = w0 + wa (1 - a)
//   MassiveNeutrinoSpecies  one Fermi-Dirac species of mass m (relativistic at early times, matter today)
//   ScaledSpecies           another species with rho and p times a constant (other density normalisations)
//   ScalarFieldSpecies      the base of the scalar-field dark energies (BackgroundModel_Quintessence.h, _EDE.h), which
//                           solve the field in their constructor (see below)
//
// In the Einstein-Boltzmann solver a species also has perturbations; when the solver moves onto this module its
// Species will derive from this one and add perturbations().
//
// Scalar fields (ScalarFieldSpecies):
// A minimally coupled scalar field phi (units of the reduced Planck mass M_pl) with the potential
//   V(phi) = 3 H0^2 M_pl^2 v(phi)      (v is a density in units of rho_c0)
// in general relativity, together with the other species rho_o (all of the background except the field), the
// curvature and an optional bare cosmological constant Omega_L. With x = ln a and phi_x = dphi/dx:
//   rho_phi = E^2 phi_x^2 / 6 + v,   p_phi = E^2 phi_x^2 / 6 - v              [units of rho_c0]
//   E^2 = (rho_o + Omega_k e^{-2x} + Omega_L + v) / (1 - phi_x^2 / 6)
//   phi_xx = -(3 + dlnE/dx) phi_x - 3 v_phi / E^2,   dlnE/dx = (rho_o' - 2 Omega_k e^{-2x} - E^2 phi_x^2) / (2 E^2)
//   drho_phi/dx = -3 (rho_phi + p_phi) = -E^2 phi_x^2
// The field starts at x_start = min(ln 1e-10, x_min) (deep in the radiation era) at phi_ini with
// phi_x = dphidlna_ini (0: frozen by Hubble friction) and is integrated (GSL rk8pd, relative error 1e-12) to x_end and
// stored on a uniform grid in x (spacing 0.002, nodes at x_start + i dx) with phi, phi_x and phi_xx; between the nodes
// phi and phi_x come from quintic Hermite interpolation, and rho and p are computed from them (the Einstein-Boltzmann
// solver splines phi and phi_x).
//
// Potentials with a minimum: once the field oscillates about it fast (at least 3 crossings, the last half period
// shorter than 0.025 in x, i.e. m / H > ~125) it is replaced by a fluid with the cycle-averaged equation of state
// w_oscillating(), rho = rho_switch e^{-3 (1 + w)(x - x_switch)} (as SimulationCode does; the Einstein-Boltzmann solver
// integrates the oscillations to the end, which undersamples them on the stored grid when m >> H). The switch is at a
// crossing of the minimum, where the instantaneous density is on its cycle-averaged trend.
//
// Needs (besides Spline) the ODESolver and Math (find_root) modules of FML.
// =====================================================================================================================

#include <FML/Cosmology/Background/Background.h>
#include <FML/Cosmology/Background/BackgroundParameters.h>
#include <FML/Math/NumericalMethods.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    class Species {
      public:
        virtual ~Species() = default;
        virtual std::string name() const = 0;
        virtual Component component() const = 0;
        /// Energy density / rho_c0
        virtual double rho(double x) const = 0;
        /// Pressure / (rho_c0 c^2)
        virtual double p(double x) const = 0;
        /// drho/dx (default: the continuity equation of an uncoupled species)
        virtual double drhodx(double x) const { return -3.0 * (rho(x) + p(x)); }
        /// d^2rho/dx^2 (default: 5-point central difference of drhodx, step 1e-3, error ~ h^4 rho^(5) / 30)
        virtual double d2rhodx2(double x) const {
            const double h = 1e-3;
            return (drhodx(x - 2.0 * h) - 8.0 * drhodx(x - h) + 8.0 * drhodx(x + h) - drhodx(x + 2.0 * h)) / (12.0 * h);
        }
        double w(double x) const { return p(x) / rho(x); }
        virtual std::string info() const { return name(); }
    };

    /// rho = Omega0 a^{-n}, p = (n/3 - 1) rho
    class PowerLawSpecies : public Species {
      public:
        PowerLawSpecies(std::string name, Component c, double Omega0, double n)
            : label(std::move(name)), comp(c), Omega0(Omega0), n(n) {}
        std::string name() const override { return label; }
        Component component() const override { return comp; }
        double rho(double x) const override { return Omega0 * std::exp(-n * x); }
        double p(double x) const override { return (n / 3.0 - 1.0) * rho(x); }
        double drhodx(double x) const override { return -n * rho(x); }
        double d2rhodx2(double x) const override { return n * n * rho(x); }
        std::string info() const override {
            std::ostringstream s;
            s << label << ": Omega0 = " << Omega0 << ", rho ~ a^-" << n;
            return s.str();
        }

      private:
        std::string label;
        Component comp;
        double Omega0, n;
    };

    /// Dark energy with w = w0 + wa (1 - a): rho = Omega0 a^{-3(1 + w0 + wa)} e^{-3 wa (1 - a)}
    class CPLDarkEnergy : public Species {
      public:
        CPLDarkEnergy(double Omega0, double w0, double wa) : Omega0(Omega0), w0(w0), wa(wa) {}
        std::string name() const override { return "w0wa"; }
        Component component() const override { return Component::DarkEnergy; }
        double w_of_x(double x) const { return w0 + wa * (1.0 - std::exp(x)); }
        double rho(double x) const override {
            return Omega0 * std::exp(-3.0 * (1.0 + w0 + wa) * x - 3.0 * wa * (1.0 - std::exp(x)));
        }
        double p(double x) const override { return w_of_x(x) * rho(x); }
        /// rho'' = 3 wa a rho + 9 (1 + w)^2 rho  (rho' = -3 (1 + w) rho, w' = -wa a)
        double d2rhodx2(double x) const override {
            const double opw = 1.0 + w_of_x(x);
            return (3.0 * wa * std::exp(x) + 9.0 * opw * opw) * rho(x);
        }
        std::string info() const override {
            std::ostringstream s;
            s << "w0wa dark energy: Omega0 = " << Omega0 << ", w0 = " << w0 << ", wa = " << wa;
            return s.str();
        }
        double get_w0() const { return w0; }
        double get_wa() const { return wa; }

      private:
        double Omega0, w0, wa;
    };

    // -----------------------------------------------------------------------------------------------------------------
    // Massive neutrinos
    // -----------------------------------------------------------------------------------------------------------------

    /// The Fermi-Dirac momentum integrals of one massive species (neutrino + antineutrino), f0(q) = 1/(e^q + 1),
    /// with q the momentum, M = m c^2 a / (k_B T_nu0) the mass in units of the temperature and eps = sqrt(q^2 + M^2):
    ///   I_rho(M) = int q^2 eps f0 dq,   I_p(M) = int q^4 / (3 eps) f0 dq,   I_pp(M) = int q^6 / (3 eps^3) f0 dq
    /// so that rho / rho_gamma = (15/pi^4) (T_nu/T_gamma)^4 I_rho (I_rho(0) = 7 pi^4/120: (7/8) (T_nu/T_gamma)^4).
    /// I_pp is the pseudo-pressure of Lesgourgues & Tram 2011 (dI_p/dlnM = I_pp - I_p, so dp/dx = -(5p - pseudo_p)).
    /// Tabulated once (Simpson in q on [0, 60], spline in log M on [1e-4, 1e7]) with the series beyond the table:
    /// small M: I_rho_0 + pi^2 M^2/24, I_rho_0/3 - pi^2 M^2/72, I_rho_0/3 - pi^2 M^2/24; large M: (3/2) zeta(3) M +
    /// (45/4) zeta(5)/M, (15/2) zeta(5)/M, (945/4) zeta(7)/M^3.
    class FermiDiracIntegrals {
      public:
        static const FermiDiracIntegrals & get() {
            static const FermiDiracIntegrals tables;
            return tables;
        }
        double I_rho(double M) const {
            if (M <= M_lo)
                return I_rho_0 + M * M * M_PI * M_PI / 24.0;
            if (M >= M_hi)
                return 1.5 * zeta3 * M + 11.25 * zeta5 / M;
            return std::exp(log_I_rho(std::log(M)));
        }
        double I_p(double M) const {
            if (M <= M_lo)
                return I_rho_0 / 3.0 - M * M * M_PI * M_PI / 72.0;
            if (M >= M_hi)
                return 7.5 * zeta5 / M;
            return std::exp(log_I_p(std::log(M)));
        }
        double I_pp(double M) const {
            if (M <= M_lo)
                return I_rho_0 / 3.0 - M * M * M_PI * M_PI / 24.0;
            if (M >= M_hi)
                return 236.25 * zeta7 / (M * M * M);
            return std::exp(log_I_pp(std::log(M)));
        }
        static constexpr double I_rho_0 = 7.0 * M_PI * M_PI * M_PI * M_PI / 120.0;

      private:
        static constexpr double zeta3 = 1.2020569031595942, zeta5 = 1.0369277551433699, zeta7 = 1.0083492773819228;
        static constexpr double M_lo = 1e-4, M_hi = 1e7;
        FML::INTERPOLATION::SPLINE::GSLSpline log_I_rho, log_I_p, log_I_pp;

        FermiDiracIntegrals() {
            const int nq = 6000, nM = 1500;
            const double qmax = 60.0, dq = qmax / nq;
            std::vector<double> q(nq + 1), wf(nq + 1);
            for (int i = 0; i <= nq; i++) {
                q[i] = i * dq;
                const double simpson = (i == 0 or i == nq) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
                wf[i] = simpson * dq / 3.0 / (std::exp(q[i]) + 1.0);
            }
            std::vector<double> logM(nM), lr(nM), lp(nM), lpp(nM);
            for (int j = 0; j < nM; j++) {
                logM[j] = std::log(M_lo) + (std::log(M_hi) - std::log(M_lo)) * j / double(nM - 1);
                const double M = std::exp(logM[j]);
                double r = 0.0, pp = 0.0, ps = 0.0;
                for (int i = 1; i <= nq; i++) {
                    const double q2 = q[i] * q[i], eps = std::sqrt(q2 + M * M);
                    r += wf[i] * q2 * eps;
                    pp += wf[i] * q2 * q2 / (3.0 * eps);
                    ps += wf[i] * q2 * q2 * q2 / (3.0 * eps * eps * eps);
                }
                lr[j] = std::log(r);
                lp[j] = std::log(pp);
                lpp[j] = std::log(ps);
            }
            log_I_rho.create(logM, lr, "Fermi-Dirac log I_rho(log M)");
            log_I_p.create(logM, lp, "Fermi-Dirac log I_p(log M)");
            log_I_pp.create(logM, lpp, "Fermi-Dirac log I_pp(log M)");
        }
    };

    /// One massive neutrino species of mass m [eV] with temperature T_ncdm_over_TCMB * T_CMB today:
    ///   rho = C a^-4 I_rho(M(x)), p = C a^-4 I_p(M), pseudo_p = C a^-4 I_pp(M),  M(x) = M_today e^x,
    ///   C = Omega_gamma (15/pi^4) (T_ncdm/T_CMB)^4,   dp/dx = -(5p - pseudo_p)
    class MassiveNeutrinoSpecies : public Species {
      public:
        MassiveNeutrinoSpecies(double m_eV, const BackgroundParameters & p) : m_eV(m_eV) {
            using namespace FML::UTILS::SI;
            const double T_ncdm = p.T_ncdm_over_TCMB * p.T_CMB;
            if (not(m_eV > 0.0) or not(T_ncdm > 0.0))
                throw std::invalid_argument("[MassiveNeutrinoSpecies] Need m > 0 and T_ncdm > 0");
            M0 = m_eV * eV / (k_b * T_ncdm);
            FermiDiracIntegrals::get(); // tabulated here, not first inside a parallel region
            C = p.Omega_gamma() * std::pow(p.T_ncdm_over_TCMB, 4) * 15.0 / std::pow(M_PI, 4);
        }
        std::string name() const override { return "massive_nu"; }
        Component component() const override { return Component::MassiveNeutrinos; }
        double rho(double x) const override {
            return C * std::exp(-4.0 * x) * FermiDiracIntegrals::get().I_rho(M_of_x(x));
        }
        double p(double x) const override { return C * std::exp(-4.0 * x) * FermiDiracIntegrals::get().I_p(M_of_x(x)); }
        /// The pseudo-pressure (Lesgourgues & Tram 2011) [rho_c0 c^2]
        double pseudo_p(double x) const { return C * std::exp(-4.0 * x) * FermiDiracIntegrals::get().I_pp(M_of_x(x)); }
        /// dp/dx = -(5p - pseudo_p)
        double dpdx(double x) const { return -(5.0 * p(x) - pseudo_p(x)); }
        /// rho'' = -3 (rho' + p')
        double d2rhodx2(double x) const override { return -3.0 * (drhodx(x) + dpdx(x)); }
        /// rho a^4 / I_rho(M) [rho_c0]
        double prefactor() const { return C; }
        /// m c^2 a / (k_B T_nu0): the mass in units of the temperature at x, and today
        double M_of_x(double x) const { return M0 * std::exp(x); }
        double M_today() const { return M0; }
        double mass_eV() const { return m_eV; }
        std::string info() const override {
            std::ostringstream s;
            s << "massive neutrino: m = " << m_eV << " eV, Omega0 = " << rho(0.0);
            return s.str();
        }

      private:
        double m_eV, M0, C;
    };

    /// A species with its density and pressure multiplied by a constant (another normalisation of the density
    /// parameters, e.g. the Brans-Dicke density definitions); species() and factor() give what it wraps
    class ScaledSpecies : public Species {
      public:
        ScaledSpecies(std::shared_ptr<const Species> s, double factor) : s(std::move(s)), f(factor) {
            if (not this->s)
                throw std::invalid_argument("[ScaledSpecies] Null species");
        }
        std::string name() const override { return s->name(); }
        Component component() const override { return s->component(); }
        double rho(double x) const override { return f * s->rho(x); }
        double p(double x) const override { return f * s->p(x); }
        double drhodx(double x) const override { return f * s->drhodx(x); }
        double d2rhodx2(double x) const override { return f * s->d2rhodx2(x); }
        std::string info() const override {
            std::ostringstream o;
            o << s->info() << " (x " << f << ")";
            return o.str();
        }
        /// The wrapped species and the factor
        const Species & species() const { return *s; }
        const std::shared_ptr<const Species> & species_ptr() const { return s; }
        double factor() const { return f; }

      private:
        std::shared_ptr<const Species> s;
        double f;
    };

    /// The MassiveNeutrinoSpecies of s, unwrapped from (nested) ScaledSpecies (nullptr if s is not one); factor is
    /// multiplied by their factors
    inline const MassiveNeutrinoSpecies * as_massive_neutrinos(const Species & s, double & factor) {
        const Species * inner = &s;
        while (const auto * scaled = dynamic_cast<const ScaledSpecies *>(inner)) {
            factor *= scaled->factor();
            inner = &scaled->species();
        }
        return dynamic_cast<const MassiveNeutrinoSpecies *>(inner);
    }
    /// Degenerate massive neutrino species: the same mass and temperature (identical perturbations)
    inline bool degenerate(const MassiveNeutrinoSpecies & a, const MassiveNeutrinoSpecies & b) {
        return a.M_today() == b.M_today() and a.prefactor() == b.prefactor();
    }

    // -----------------------------------------------------------------------------------------------------------------
    // Scalar fields
    // -----------------------------------------------------------------------------------------------------------------

    // -----------------------------------------------------------------------------------------------------------------
    // The base: a scalar field in GR together with the other species
    // -----------------------------------------------------------------------------------------------------------------

    class ScalarFieldSpecies : public Species {
      public:
        using SpeciesList = std::vector<std::shared_ptr<const Species>>;

        /// others: all other species of the background (not a bare cosmological constant: that is Omega_L, which the
        /// preset adds to the list as its own species)
        ScalarFieldSpecies(SpeciesList others, double Omega_k, double Omega_L = 0.0)
            : others(std::move(others)), Omega_k(Omega_k), OmegaL(Omega_L) {
            for (const auto & s : this->others)
                if (not s)
                    throw std::invalid_argument("[ScalarFieldSpecies] Null species");
        }
        Component component() const override { return Component::DarkEnergy; }

        double rho(double x) const override {
            if (in_fluid(x))
                return rho_switch * std::exp(-3.0 * (1.0 + w_oscillating()) * (x - x_switch));
            const double f = phi(x), df = dphidx(x);
            return E2(x, f, df) * df * df / 6.0 + v(f);
        }
        double p(double x) const override {
            if (in_fluid(x))
                return w_oscillating() * rho(x);
            const double f = phi(x), df = dphidx(x);
            return E2(x, f, df) * df * df / 6.0 - v(f);
        }
        double drhodx(double x) const override {
            if (in_fluid(x))
                return -3.0 * (1.0 + w_oscillating()) * rho(x);
            const double f = phi(x), df = dphidx(x);
            return -E2(x, f, df) * df * df;
        }

        /// rho'' = -(E^2' phi_x^2 + 2 E^2 phi_x phi_xx), E^2' = rho_o' + rho_phi' (the fluid: 9 (1 + w)^2 rho)
        double d2rhodx2(double x) const override {
            if (in_fluid(x)) {
                const double opw = 1.0 + w_oscillating();
                return 9.0 * opw * opw * rho(x);
            }
            const double f = phi(x), df = dphidx(x), e2 = E2(x, f, df);
            const double de2 = drho_others(x) - e2 * df * df;
            return -(de2 * df * df + 2.0 * e2 * df * ddphi(x, f, df));
        }

        /// The potential v = V / (3 H0^2 M_pl^2) and its derivatives dv/dphi, d^2v/dphi^2 (phi in M_pl; the default
        /// of v_phiphi is a central difference of v_phi, the potentials here give it analytically)
        virtual double v(double phi) const = 0;
        virtual double v_phi(double phi) const = 0;
        virtual double v_phiphi(double phi) const {
            const double h = 1e-5 * std::max(1.0, std::abs(phi));
            return (v_phi(phi + h) - v_phi(phi - h)) / (2.0 * h);
        }
        /// A potential with a minimum: where it is and the cycle-averaged w of fast oscillations about it
        virtual bool has_minimum() const { return false; }
        virtual double phi_min() const { return 0.0; }
        virtual double w_oscillating() const { return 0.0; }

        /// The field [M_pl] and dphi/dx (phi_min and 0 once it is a fluid)
        double phi(double x) const {
            check(x);
            return in_fluid(x) ? phi_min() : interpolate(x, false);
        }
        double dphidx(double x) const {
            check(x);
            return in_fluid(x) ? 0.0 : interpolate(x, true);
        }
        /// d^2phi/dx^2 from the Klein-Gordon equation (0 once it is a fluid)
        double ddphidx(double x) const {
            check(x);
            return in_fluid(x) ? 0.0 : ddphi(x, phi(x), dphidx(x));
        }
        /// (H/H0)^2 of the model (as summed by GR) and the fraction rho_phi / rho_tot
        double E2(double x) const { return rho_others(x) + rho(x); }
        double fraction(double x) const { return rho(x) / E2(x); }
        /// The bare cosmological constant the field is solved with
        double Omega_Lambda_bare() const { return OmegaL; }
        /// Where the field was replaced by a fluid (+infinity if never)
        double x_fluid() const { return x_switch; }
        /// The range of the solution
        double x_start() const { return xs; }
        double x_end() const { return xe; }

        /// The range [x_start, x_end] a field solves on for a background on r: x_start = min(ln 1e-10, x_min), where
        /// the initial conditions are imposed (as in SimulationCode; matters only for a kinetic start), x_end = x_max
        /// + 0.1
        static std::pair<double, double> solution_range(const BackgroundRange & r) {
            return {std::min(std::log(1e-10), r.x_min), r.x_max + 0.1};
        }

      protected:
        SpeciesList others;
        double Omega_k, OmegaL;

        /// Everything but the field: the other species, curvature and the bare Lambda, and the x-derivative
        double rho_others(double x) const {
            double r = Omega_k * std::exp(-2.0 * x) + OmegaL;
            for (const auto & s : others)
                r += s->rho(x);
            return r;
        }
        double drho_others(double x) const {
            double r = -2.0 * Omega_k * std::exp(-2.0 * x);
            for (const auto & s : others)
                r += s->drhodx(x);
            return r;
        }
        double E2(double x, double phi, double dphi) const {
            return (rho_others(x) + v(phi)) / (1.0 - dphi * dphi / 6.0);
        }
        /// phi_xx from the Klein-Gordon equation
        double ddphi(double x, double phi, double dphi) const {
            const double e2 = E2(x, phi, dphi);
            const double dlnE = (drho_others(x) - e2 * dphi * dphi) / (2.0 * e2);
            return -(3.0 + dlnE) * dphi - 3.0 * v_phi(phi) / e2;
        }

        /// Integrate from (phi_ini, phi_x = dphi_ini) at x_from to x_to (the grid x_from + i dx, extended to cover
        /// x_to) and store the solution. Throws if the field makes E^2 <= 0 or phi_x^2 >= 6.
        void solve(double phi_ini, double dphi_ini, double x_from, double x_to) {
            const std::string msg = "[" + name() + "] ";
            if (not(dphi_ini * dphi_ini < 6.0))
                throw std::invalid_argument(msg + "Need |dphi/dlna| < sqrt(6) initially");
            if (not(x_to > x_from))
                throw std::invalid_argument(msg + "Need x_end > x_start");
            const int n = int(std::ceil((x_to - x_from) / dx - 1e-9)) + 1;
            xs = x_from;
            xe = x_from + (n - 1) * dx;
            x_switch = std::numeric_limits<double>::infinity();
            rho_switch = 0.0;

            FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, const double * y, double * dydx) {
                dydx[0] = y[1];
                dydx[1] = ddphi(x, y[0], y[1]);
                return GSL_SUCCESS;
            };

            f.assign(1, phi_ini);
            df.assign(1, dphi_ini);
            ddf.assign(1, ddphi(xs, phi_ini, dphi_ini));
            f.reserve(n);
            df.reserve(n);
            ddf.reserve(n);
            const bool osc = has_minimum();
            const double fmin = phi_min();
            int ncross = 0;
            double last_crossing = -std::numeric_limits<double>::infinity();
            bool fluid = false;
            std::vector<double> y{phi_ini, dphi_ini};
            for (int i0 = 0; i0 < n - 1 and not fluid; i0 += chunk) {
                const int i1 = std::min(i0 + chunk, n - 1);
                std::vector<double> xx(i1 - i0 + 1);
                for (int i = i0; i <= i1; i++)
                    xx[i - i0] = xs + i * dx;
                FML::SOLVERS::ODESOLVER::ODESolver ode(1e-4, 1e-14, 1e-12);
                ode.solve(deriv, xx, y, gsl_odeiv2_step_rk8pd);
                const auto sol = ode.get_data();
                for (size_t j = 1; j < xx.size(); j++) {
                    const double e2 = E2(xx[j], sol[j][0], sol[j][1]);
                    if (not(e2 > 0.0) or not std::isfinite(e2) or not(sol[j][1] * sol[j][1] < 6.0))
                        throw std::runtime_error(
                            msg + "The field gives E^2 <= 0 or phi_x^2 >= 6 at x = " + std::to_string(xx[j]));
                    f.push_back(sol[j][0]);
                    df.push_back(sol[j][1]);
                    ddf.push_back(ddphi(xx[j], sol[j][0], sol[j][1]));
                    if (not osc)
                        continue;
                    const double s0 = sol[j - 1][0] - fmin, s1 = sol[j][0] - fmin;
                    if (s0 * s1 < 0.0) {
                        const double xc = xx[j - 1] + (xx[j] - xx[j - 1]) * s0 / (s0 - s1);
                        if (++ncross >= 3 and xc - last_crossing < fluid_half_period) {
                            fluid = true;
                            x_switch = xx[j];
                            rho_switch = e2 * sol[j][1] * sol[j][1] / 6.0 + v(sol[j][0]);
                            break;
                        }
                        last_crossing = xc;
                    }
                }
                y = sol.back();
            }
        }

        /// The maximum of rho_phi / rho_tot on the stored grid and the x where it is (parabola through the three nodes
        /// around the largest)
        void peak(double & f_max, double & x_peak) const {
            const int N = int(std::lround((xe - xs) / dx));
            std::vector<double> F(N + 1);
            int imax = 0;
            for (int i = 0; i <= N; i++) {
                F[i] = fraction(std::min(xs + i * dx, xe));
                if (F[i] > F[imax])
                    imax = i;
            }
            f_max = F[imax];
            x_peak = xs + imax * dx;
            if (imax == 0 or imax == N)
                return;
            const double fm = F[imax - 1], f0 = F[imax], fp = F[imax + 1];
            const double t = 0.5 * (fm - fp) / (fm - 2.0 * f0 + fp);
            x_peak = xs + (imax + t) * dx;
            f_max = f0 - 0.25 * (fm - fp) * t;
        }

      private:
        static constexpr double dx = 0.002;
        static constexpr int chunk = 125;
        static constexpr double fluid_half_period = 0.025;
        double xs{0.0}, xe{0.0};
        double x_switch{std::numeric_limits<double>::infinity()}, rho_switch{0.0};
        /// phi, phi_x, phi_xx on the nodes x_start + i dx (up to the switch to the fluid)
        std::vector<double> f, df, ddf;

        /// phi (or phi_x) between the nodes: quintic Hermite interpolation of phi from phi, phi_x, phi_xx at the two
        /// nodes around x (error ~ (omega dx)^6 for a field oscillating with frequency omega in x; a cubic spline of
        /// phi and phi_x gives 1e-5 errors in E when the field oscillates with m / H ~ 100)
        double interpolate(double x, bool derivative) const {
            const int last = int(f.size()) - 2;
            const int i = std::clamp(int(std::floor((x - xs) / dx)), 0, last);
            const double t = (x - xs) / dx - i, t2 = t * t, t3 = t2 * t;
            const double h = dx, h2 = dx * dx;
            if (not derivative) {
                const double t4 = t3 * t, t5 = t4 * t;
                return f[i] * (1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5) +
                       h * df[i] * (t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5) +
                       h2 * ddf[i] * 0.5 * (t2 - 3.0 * t3 + 3.0 * t4 - t5) +
                       f[i + 1] * (10.0 * t3 - 15.0 * t4 + 6.0 * t5) +
                       h * df[i + 1] * (-4.0 * t3 + 7.0 * t4 - 3.0 * t5) + h2 * ddf[i + 1] * 0.5 * (t3 - 2.0 * t4 + t5);
            }
            const double t4 = t3 * t;
            return (f[i] * (-30.0 * t2 + 60.0 * t3 - 30.0 * t4) +
                    h * df[i] * (1.0 - 18.0 * t2 + 32.0 * t3 - 15.0 * t4) +
                    h2 * ddf[i] * 0.5 * (2.0 * t - 9.0 * t2 + 12.0 * t3 - 5.0 * t4) +
                    f[i + 1] * (30.0 * t2 - 60.0 * t3 + 30.0 * t4) +
                    h * df[i + 1] * (-12.0 * t2 + 28.0 * t3 - 15.0 * t4) +
                    h2 * ddf[i + 1] * 0.5 * (3.0 * t2 - 8.0 * t3 + 5.0 * t4)) /
                   h;
        }

        bool in_fluid(double x) const { return x > x_switch; }
        void check(double x) const {
            if (not(x >= xs - 1e-12 and x <= xe + 1e-12))
                throw std::out_of_range("[" + name() + "] x = " + std::to_string(x) + " outside the field solution [" +
                                        std::to_string(xs) + ", " + std::to_string(xe) + "]");
        }
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
