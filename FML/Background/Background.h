#ifndef FML_BACKGROUND_BACKGROUND_H
#define FML_BACKGROUND_BACKGROUND_H

// =====================================================================================================================
// Background: the expansion history of a homogeneous universe, the one interface every part of FML that needs a
// background uses (the Einstein-Boltzmann solver, SimulationCode, LPT, N-body time stepping, spherical collapse, the
// halo model, survey and lightcone distances, ...).
//
// Time variable x = ln a. Everything is dimensionless or in h-units:
//   E(x) = H/H0, dlogE/dx = dlnH/dlna, Omega(component, x) = rho_i(x) / rho_crit(x)
//   conformal time eta and distances in Mpc/h, cosmic time in Gyr (or in units of 1/H0)
//
// A derived class gives the virtual core: E, dlogEdx, Omega, the parameters and the valid range, and calls
// finalize() at the end of its constructor. finalize() integrates the conformal time, the cosmic time and the
// comoving distance once (5-point Gauss-Legendre on a fine grid in x, with the exact power-law solution at x_min) and
// everything else (distances with curvature, sin_K / cot_K, times, the inverse x(chi), a- and z-versions) is
// non-virtual here.
//
// Implementations:
//   ComponentBackground  species (Species.h) + field equations (FieldEquations.h): the physical models
//   TabulatedBackground  tables of E(x) and Omega_i(x) (e.g. a CLASS _background.dat)
//   FunctionBackground   user functions E(x), ... (old callback interfaces, tests)
// and the models (BackgroundModel_<name>.h: make_lcdm, make_w0wa, ...) with their registry (Presets.h:
// make_background(name, ...)).
//
// A Background is immutable after construction; share it as std::shared_ptr<const Background>. All methods are
// const and thread-safe.
// =====================================================================================================================

#include <FML/Background/BackgroundParameters.h>
#include <FML/Spline/Spline.h>
#include <FML/Units/Units.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::BACKGROUND {

    /// The components of a background. Curvature is Omega_k e^{-2x} / E^2. Other: anything that is not one of the
    /// named kinds (a scalar field that is neither matter nor dark energy, ...)
    enum class Component { Baryons, Photons, CDM, MasslessNeutrinos, MassiveNeutrinos, DarkEnergy, Curvature, Other };
    inline constexpr std::array<Component, 8> all_components{Component::Baryons,
                                                             Component::Photons,
                                                             Component::CDM,
                                                             Component::MasslessNeutrinos,
                                                             Component::MassiveNeutrinos,
                                                             Component::DarkEnergy,
                                                             Component::Curvature,
                                                             Component::Other};
    inline std::string component_name(Component c) {
        switch (c) {
            case Component::Baryons:
                return "baryons";
            case Component::Photons:
                return "photons";
            case Component::CDM:
                return "cdm";
            case Component::MasslessNeutrinos:
                return "massless_nu";
            case Component::MassiveNeutrinos:
                return "massive_nu";
            case Component::DarkEnergy:
                return "dark_energy";
            case Component::Curvature:
                return "curvature";
            default:
                return "other";
        }
    }

    class Background {
      public:
        virtual ~Background() = default;

        // ---- The virtual core ---------------------------------------------------------------------------------------
        virtual std::string name() const = 0;
        virtual const BackgroundParameters & parameters() const = 0;
        /// H(x) / H0
        virtual double E(double x) const = 0;
        /// dln H / dx
        virtual double dlogEdx(double x) const = 0;
        /// rho_i(x) / rho_crit(x) (Curvature: Omega_k e^{-2x} / E^2). 0 for a component the background does not have
        virtual double Omega(Component c, double x) const = 0;
        /// Description for info()
        virtual std::string description() const { return name(); }

        // ---- Range ------------------------------------------------------------------------------------------------
        double x_min() const { return range.x_min; }
        double x_max() const { return range.x_max; }
        bool in_range(double x) const { return x >= range.x_min - 1e-12 and x <= range.x_max + 1e-12; }

        // ---- Parameters and today's values
        // ---------------------------------------------------------------------------
        double h() const { return parameters().h; }
        /// H0 in h/Mpc (with c = 1: 1 / 2997.92458)
        double H0_hmpc() const { return 1.0 / FML::UTILS::SI::c_over_H0_mpch; }
        /// H0 in 1/s
        double H0_SI() const { return parameters().H0_SI(); }
        /// Omega_i today
        double Omega0(Component c) const { return Omega(c, 0.0); }
        /// Matter today: baryons + CDM + massive neutrinos (the latter including their small relativistic part)
        double Omega_m0() const { return Omega_m(0.0); }

        // ---- Expansion ----------------------------------------------------------------------------------------------
        /// H in 1/s
        double H_SI(double x) const { return H0_SI() * E(x); }
        /// H in km/s/Mpc
        double H_kms_mpc(double x) const { return 100.0 * h() * E(x); }
        /// Hp = a H / H0 (conformal Hubble rate in units of H0)
        double Hp(double x) const { return std::exp(x) * E(x); }
        /// Matter: baryons + CDM + massive neutrinos
        double Omega_m(double x) const {
            return Omega(Component::Baryons, x) + Omega(Component::CDM, x) + Omega(Component::MassiveNeutrinos, x);
        }
        /// Radiation: photons + massless neutrinos
        double Omega_r(double x) const { return Omega(Component::Photons, x) + Omega(Component::MasslessNeutrinos, x); }
        /// The effective equation of state of the total: -1 - (2/3) dlnH/dx (curvature counted as a fluid w = -1/3)
        double w_eff(double x) const { return -1.0 - 2.0 / 3.0 * dlogEdx(x); }
        /// The deceleration parameter q = -1 - dlnH/dx
        double deceleration(double x) const { return -1.0 - dlogEdx(x); }

        // ---- Times and distances (tabulated by finalize)
        // -------------------------------------------------------------
        /// Conformal time in Mpc/h
        double eta(double x) const {
            check(x, "eta");
            return FML::UTILS::SI::c_over_H0_mpch * std::exp(log_eta_spline(x));
        }
        /// Conformal time today in Mpc/h
        double eta0() const { return eta(0.0); }
        /// Cosmic time in units of 1/H0
        double t_H0(double x) const {
            check(x, "t_H0");
            return std::exp(log_t_spline(x));
        }
        /// Cosmic time in Gyr
        double t_Gyr(double x) const { return t_H0(x) * hubble_time_Gyr(); }
        /// Age of the universe in Gyr
        double age_Gyr() const { return t_Gyr(0.0); }
        /// 1/H0 in Gyr
        double hubble_time_Gyr() const { return 1.0 / H0_SI() / FML::UTILS::SI::Gyr; }

        /// Comoving (line-of-sight) distance to x in Mpc/h (negative for x > 0)
        double chi(double x) const {
            check(x, "chi");
            return FML::UTILS::SI::c_over_H0_mpch * chi_spline(x);
        }
        /// x at comoving distance chi [Mpc/h] (the inverse of chi(x))
        double x_of_chi(double chi_mpch) const {
            const double d = chi_mpch / FML::UTILS::SI::c_over_H0_mpch;
            if (not(d >= chi_range.first - 1e-14 and d <= chi_range.second + 1e-14))
                throw std::out_of_range("[" + name() + "] x_of_chi: chi = " + std::to_string(chi_mpch) +
                                        " Mpc/h outside the range of the background");
            return x_of_chi_spline(d);
        }
        /// Transverse comoving distance D_M in Mpc/h (with curvature)
        double D_M(double x) const { return sinK(chi(x)); }
        /// Angular diameter distance in Mpc/h
        double D_A(double x) const { return std::exp(x) * D_M(x); }
        /// Luminosity distance in Mpc/h
        double D_L(double x) const { return D_M(x) / std::exp(x); }
        /// Hubble distance c/H(x) in Mpc/h
        double D_H(double x) const { return FML::UTILS::SI::c_over_H0_mpch / E(x); }

        // ---- Spatial geometry
        // -----------------------------------------------------------------------------------------
        /// The spatial curvature K = -Omega_k (H0/c)^2 in (h/Mpc)^2 (K > 0 closed, K < 0 open)
        double K_hmpc2() const {
            const double H0c = 1.0 / FML::UTILS::SI::c_over_H0_mpch;
            return -parameters().Omega_k * H0c * H0c;
        }
        /// sin_K(chi) [Mpc/h]: the transverse comoving distance for the comoving radial distance chi [Mpc/h],
        /// sin(sqrt(K) chi)/sqrt(K) (closed), chi (flat), sinh(sqrt(-K) chi)/sqrt(-K) (open). Series for
        /// |K chi^2| < 1e-4, so smooth through Omega_k = 0 (as the Einstein-Boltzmann solver)
        double sinK(double chi_mpch) const {
            const double u = K_hmpc2() * chi_mpch * chi_mpch;
            if (std::abs(u) < 1e-4)
                return chi_mpch * (1.0 - u / 6.0 * (1.0 - u / 20.0 * (1.0 - u / 42.0)));
            const double s = std::sqrt(std::abs(u));
            return chi_mpch * (u > 0.0 ? std::sin(s) : std::sinh(s)) / s;
        }
        /// cot_K(chi) = d ln sin_K / d chi [h/Mpc] (1/chi when flat; series for |K chi^2| < 1e-4)
        double cotK(double chi_mpch) const {
            const double u = K_hmpc2() * chi_mpch * chi_mpch;
            if (std::abs(u) < 1e-4)
                return (1.0 - u / 3.0 - u * u / 45.0 - 2.0 * u * u * u / 945.0) / chi_mpch;
            const double s = std::sqrt(std::abs(u));
            return (u > 0.0 ? s / std::tan(s) : s / std::tanh(s)) / chi_mpch;
        }

        // ---- Convenience versions in a and z ------------------------------------------------------------------------
        static double x_of_a(double a) { return std::log(a); }
        static double x_of_z(double z) { return -std::log1p(z); }
        static double z_of_x(double x) { return std::expm1(-x); }
        double HoverH0_of_a(double a) const { return E(std::log(a)); }
        double dlogHdloga_of_a(double a) const { return dlogEdx(std::log(a)); }
        double E_of_z(double z) const { return E(x_of_z(z)); }
        double chi_of_z(double z) const { return chi(x_of_z(z)); }
        double z_of_chi(double chi_mpch) const { return z_of_x(x_of_chi(chi_mpch)); }
        double D_M_of_z(double z) const { return D_M(x_of_z(z)); }
        double D_A_of_z(double z) const { return D_A(x_of_z(z)); }
        double D_L_of_z(double z) const { return D_L(x_of_z(z)); }

        // ---- Output -------------------------------------------------------------------------------------------------
        void info(std::ostream & os = std::cout) const {
            const auto & p = parameters();
            // NaN for a component the background cannot give (FunctionBackground without Omega)
            const auto O = [&](Component c) {
                try {
                    return Omega0(c);
                } catch (const std::logic_error &) {
                    return std::nan("");
                }
            };
            os << "Background [" << name() << "]: " << description() << "\n"
               << "  h = " << p.h << "  Omega_b = " << O(Component::Baryons) << "  Omega_cdm = " << O(Component::CDM)
               << "  Omega_mnu = " << O(Component::MassiveNeutrinos) << "\n"
               << "  Omega_gamma = " << O(Component::Photons)
               << "  Omega_nu (massless) = " << O(Component::MasslessNeutrinos)
               << "  Omega_de = " << O(Component::DarkEnergy) << "  Omega_k = " << O(Component::Curvature) << "\n"
               << "  T_CMB = " << p.T_CMB << " K  N_eff = " << p.N_eff << "  sum m_nu = " << p.sum_m_nu_eV() << " eV\n"
               << "  Age = " << age_Gyr() << " Gyr  eta0 = " << eta0() << " Mpc/h  valid for a in ["
               << std::exp(x_min()) << ", " << std::exp(x_max()) << "]\n";
        }

        /// Table of x, a, z, E, dlogE/dx, Omega_i, eta, chi, t on [x_lo, x_hi] (default: the valid range)
        void output(const std::string & filename, int n = 1000, double x_lo = 1.0, double x_hi = -1.0) const {
            if (x_lo > x_hi) {
                x_lo = x_min();
                x_hi = x_max();
            }
            std::ofstream fp(filename);
            if (not fp)
                throw std::runtime_error("[" + name() + "] Cannot write [" + filename + "]");
            fp << "# x  a  z  E  dlogEdx";
            for (auto c : all_components)
                fp << "  Omega_" << component_name(c);
            fp << "  eta[Mpc/h]  chi[Mpc/h]  t[Gyr]\n";
            fp << std::setprecision(10);
            for (int i = 0; i < n; i++) {
                const double x = x_lo + (x_hi - x_lo) * i / double(n - 1);
                fp << x << " " << std::exp(x) << " " << z_of_x(x) << " " << E(x) << " " << dlogEdx(x);
                for (auto c : all_components)
                    fp << " " << Omega(c, x);
                fp << " " << eta(x) << " " << chi(x) << " " << t_Gyr(x) << "\n";
            }
        }

      protected:
        BackgroundRange range{};

        /// Tabulate the conformal time, the cosmic time and the comoving distance on range. Called at the end of the
        /// constructor of a derived class (the virtual core must work). x_min must be in an era where E ~ a^{-n} with
        /// n > 1 (radiation or matter domination): eta(x_min) = 1 / (a E (n - 1)), t(x_min) = 1 / (E n).
        void finalize() {
            const std::string msg = "[" + name() + "] ";
            if (not(range.x_min < 0.0 and range.x_max >= 0.0))
                throw std::invalid_argument(msg + "The range in x = ln a must contain today: x_min < 0 <= x_max");
            if (range.points_per_unit_x < 10)
                throw std::invalid_argument(msg + "Need at least 10 points per unit x");
            // A grid with a node at x = 0
            const int n_lo = int(std::ceil(-range.x_min * range.points_per_unit_x));
            const int n_hi = int(std::ceil(range.x_max * range.points_per_unit_x));
            std::vector<double> x(n_lo + n_hi + 1);
            for (int i = 0; i <= n_lo; i++)
                x[i] = range.x_min * (1.0 - double(i) / n_lo);
            for (int i = 1; i <= n_hi; i++)
                x[n_lo + i] = range.x_max * double(i) / n_hi;

            const double n_start = -dlogEdx(range.x_min);
            if (not(n_start > 1.01))
                throw std::invalid_argument(msg + "At x_min = " + std::to_string(range.x_min) +
                                            " the background must be matter or radiation dominated (dlnH/dx = " +
                                            std::to_string(-n_start) + ")");
            const double E_start = E(range.x_min);
            std::vector<double> eta(x.size()), t(x.size()), deta(x.size(), 0.0);
            eta[0] = 1.0 / (std::exp(range.x_min) * E_start * (n_start - 1.0));
            t[0] = 1.0 / (E_start * n_start);

            // 5-point Gauss-Legendre on each interval
            static constexpr double gx[5] = {0.0,
                                             -0.5384693101056831,
                                             0.5384693101056831,
                                             -0.9061798459386640,
                                             0.9061798459386640};
            static constexpr double gw[5] = {0.5688888888888889,
                                             0.4786286704993665,
                                             0.4786286704993665,
                                             0.2369268850561891,
                                             0.2369268850561891};
            for (size_t i = 1; i < x.size(); i++) {
                const double mid = 0.5 * (x[i] + x[i - 1]), half = 0.5 * (x[i] - x[i - 1]);
                double de = 0.0, dt = 0.0;
                for (int j = 0; j < 5; j++) {
                    const double xx = mid + half * gx[j], e = E(xx);
                    de += gw[j] / (std::exp(xx) * e);
                    dt += gw[j] / e;
                }
                deta[i] = de * half;
                eta[i] = eta[i - 1] + deta[i];
                t[i] = t[i - 1] + dt * half;
            }
            // chi = eta0 - eta, summed from x = 0 (no cancellation at low redshift)
            std::vector<double> chi(x.size(), 0.0);
            for (int i = n_lo - 1; i >= 0; i--)
                chi[i] = chi[i + 1] + deta[i + 1];
            for (size_t i = n_lo + 1; i < x.size(); i++)
                chi[i] = chi[i - 1] - deta[i];

            std::vector<double> log_eta(x.size()), log_t(x.size());
            for (size_t i = 0; i < x.size(); i++) {
                log_eta[i] = std::log(eta[i]);
                log_t[i] = std::log(t[i]);
            }
            log_eta_spline.create(x, log_eta, name() + " log eta(x)");
            log_t_spline.create(x, log_t, name() + " log t(x)");
            chi_spline.create(x, chi, name() + " chi(x)");
            // chi decreases with x: the inverse on the reversed arrays. In the far future of a model whose E grows
            // very fast (e.g. CPL with wa > 0, rho ~ exp(3 wa a)) chi stops changing in double precision; the inverse
            // then only covers the part from today back where chi is strictly monotonic
            std::vector<double> chi_rev(chi.rbegin(), chi.rend()), x_rev(x.rbegin(), x.rend());
            size_t first = chi_rev.size() - 1;
            while (first > 0 and chi_rev[first] > chi_rev[first - 1])
                first--;
            if (x_rev[first] < 0.0)
                throw std::runtime_error(msg + "The comoving distance is not monotonic in x");
            chi_rev.erase(chi_rev.begin(), chi_rev.begin() + first);
            x_rev.erase(x_rev.begin(), x_rev.begin() + first);
            x_of_chi_spline.create(chi_rev, x_rev, name() + " x(chi)");
            chi_range = {chi_rev.front(), chi_rev.back()};
        }

        void check(double x, const char * what) const {
            if (not in_range(x))
                throw std::out_of_range("[" + name() + "] " + what + ": x = " + std::to_string(x) +
                                        " (a = " + std::to_string(std::exp(x)) + ") outside the valid range [" +
                                        std::to_string(range.x_min) + ", " + std::to_string(range.x_max) + "]");
        }

      private:
        using Spline = FML::INTERPOLATION::SPLINE::GSLSpline;
        Spline log_eta_spline, log_t_spline, chi_spline, x_of_chi_spline;
        std::pair<double, double> chi_range{0.0, 0.0};
    };

} // namespace FML::COSMOLOGY::BACKGROUND

#endif
