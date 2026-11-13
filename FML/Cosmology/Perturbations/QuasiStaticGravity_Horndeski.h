#ifndef FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_HORNDESKI_H
#define FML_COSMOLOGY_PERTURBATIONS_QUASISTATICGRAVITY_HORNDESKI_H

// =====================================================================================================================
// A reduced Horndeski theory (K(phi, X), G3(phi, X), G4(phi): luminal gravitational waves) in the quasi-static limit
// (QuasiStaticGravity.h), from a table of functions of a made by SimulationCode/scripts/horndeski_frontend.py from the
// Lagrangian (as Hi-COLA, Wright et al. 2023, arXiv:2209.01666, with the equations of Kimura, Kobayashi & Yamamoto
// 2012). The force on matter is
//   Phi = GG4 Phi_N + phi5,   GG4 = 1 / (2 G4) (the Newtonian force rescaled by the running Planck mass)
// with the fifth-force potential phi5 of the DGP brane-bending equation (QuasiStaticGravity_DGP.h; phi = 2 phi5)
//   nabla^2 phi + (r_c^2 / (3 beta a^2)) [(nabla^2 phi)^2 - (nabla_i nabla_j phi)^2] = 8 pi G a^2 delta rho / (3 beta)
// with b5 = 1/(3 beta)(a) and c2 = (r_c H0/c)^2 b5^2 (a) from the table (c2 = 0: no cubic term, a linear fifth
// force; c2 < 0: the nonlinear term has the other sign, anti-screening in overdensities). Hence
//   mu = GG4 + b5,   Sigma from the table
//   second_order_source = 1 - 2 r_c^2 rho_m / (9 beta^3 mu) = 1 - 6 c2 b5 Omega_m(a) E^2 / mu   (DGP's)
//   screening (spherical top-hat): 2 (sqrt(1 + e) - 1) / e,  e = 8 r_c^2 rho_m delta / (9 beta^2)
//                                                                = 8 c2 Omega_m(a) E^2 delta,  e >= -1
// (delta = rho/rho_mean - 1; no real solution for e < -1, where e = -1 is used, as for the cubic Galileon). For the
// cubic Galileon this is QuasiStaticGravity_CubicGalileon.h.
//
// The table: rows "a E dlnE/dlna GG4 mu Sigma b5 c2 ..." (comment lines start with #), increasing a. Outside its range
// the values at its ends are used. E of the table should be that of the background (the frontend also writes it
// in the format of the Tabulated background); max_relative_E_difference() compares them (a <= 1).
//
// References:
//   Horndeski 1974, Int. J. Theor. Phys. 10, 363; Kobayashi 2019, Rept. Prog. Phys. 82, 086901, arXiv:1901.07183
//       (review)
//   Kimura, Kobayashi & Yamamoto 2012, PRD 85, 024023, arXiv:1111.6749   (the quasi-static equations with the
//       Vainshtein nonlinearity on a cosmological background: A0, A1, A2, B0)
//   Wright, Sen Gupta, Baker, Valogiannis & Fiorini 2023 (Hi-COLA), JCAP 03, 040, arXiv:2209.01666   (the same
//       approach in COLA: a Lagrangian front-end, beta(a) and the spherical screening factor; code:
//       https://github.com/Hi-COLACode/Hi-COLA)
//   Sen Gupta, Fiorini & Baker 2024, JCAP 11, 052, arXiv:2407.00855   (Hi-COLA with K-mouflage: the Newtonian force
//       rescaled by G_G4/G_N)
//   Koyama & Silva 2007, PRD 75, 084040, arXiv:hep-th/0702169   (the DGP equation on a cosmological background)
//   Winther & Ferreira 2015, PRD 91, 123507, arXiv:1403.6492; Brando, Koyama & Winther 2023, JCAP 06, 045,
//       arXiv:2303.09549   (the screening approximation in fast simulations)
//   Barreira et al. 2013, arXiv:1306.3219   (no real solution in voids: the discriminant set to zero)
// =====================================================================================================================

#include <FML/Cosmology/Perturbations/QuasiStaticGravity.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::COSMOLOGY::PERTURBATIONS {

    class QuasiStaticGravity_Horndeski : public QuasiStaticGravity {
        using Spline = FML::INTERPOLATION::SPLINE::Spline;

      public:
        /// The columns of the table (a increasing)
        struct Table {
            std::vector<double> a, E, dlnEdx, GG4, mu, Sigma, b5, c2;
        };

        /// Read the table written by horndeski_frontend.py (columns a E dlnE/dlna GG4 mu Sigma b5 c2 ...)
        static Table read_table(const std::string & filename) {
            std::ifstream fp(filename);
            if (not fp.is_open())
                throw std::runtime_error("[QuasiStaticGravity_Horndeski] Cannot open [" + filename + "]");
            Table t;
            std::string line;
            while (std::getline(fp, line)) {
                if (line.empty() or line[0] == '#')
                    continue;
                std::istringstream is(line);
                double v[8];
                for (double & x : v)
                    if (not(is >> x))
                        throw std::runtime_error("[QuasiStaticGravity_Horndeski] Bad line in [" + filename +
                                                 "] (need a E dlnE/dlna GG4 mu Sigma b5 c2): " + line);
                t.a.push_back(v[0]);
                t.E.push_back(v[1]);
                t.dlnEdx.push_back(v[2]);
                t.GG4.push_back(v[3]);
                t.mu.push_back(v[4]);
                t.Sigma.push_back(v[5]);
                t.b5.push_back(v[6]);
                t.c2.push_back(v[7]);
            }
            return t;
        }

        QuasiStaticGravity_Horndeski(const BACKGROUND::Background & bg, const Table & t, std::string label = "")
            : QuasiStaticGravity(bg), label(std::move(label)) {
            const size_t n = t.a.size();
            if (n < 4)
                throw std::invalid_argument("[QuasiStaticGravity_Horndeski] Need a table with 4 or more rows");
            std::vector<double> x(n);
            for (size_t i = 0; i < n; i++) {
                if (not(t.a[i] > 0.0) or (i > 0 and not(t.a[i] > t.a[i - 1])))
                    throw std::invalid_argument("[QuasiStaticGravity_Horndeski] The table needs increasing a > 0");
                x[i] = std::log(t.a[i]);
                if (not std::isfinite(t.GG4[i] + t.b5[i] + t.c2[i] + t.Sigma[i]))
                    throw std::invalid_argument("[QuasiStaticGravity_Horndeski] Non-finite value in the table at a = " +
                                                std::to_string(t.a[i]));
            }
            x_lo = x.front();
            x_hi = x.back();
            GG4_spline = Spline(x, t.GG4, "GG4(x)");
            b5_spline = Spline(x, t.b5, "b5(x)");
            c2_spline = Spline(x, t.c2, "c2(x)");
            Sigma_spline = Spline(x, t.Sigma, "Sigma(x)");
            // (a <= 1: the frontend's _background.txt for a Tabulated background stops at a = 1)
            for (size_t i = 0; i < n; i++)
                if (x[i] > bg.x_min() and x[i] < bg.x_max() and x[i] <= 0.0)
                    max_dE = std::max(max_dE, std::fabs(t.E[i] / bg.E(x[i]) - 1.0));
        }
        std::string name() const override { return "Horndeski"; }
        std::string info() const override {
            std::ostringstream os;
            os << "reduced Horndeski from a table" << (label.empty() ? "" : " (" + label + ")")
               << ": mu(a = 1) = " << mu(0.0, 0.0) << ", G_G4/G = " << GG4(0.0) << ", 1/(3 beta) = " << b5(0.0)
               << ", (r_c H0/c)^2 b5^2 = " << c2(0.0) << "; quasi-static";
            return os.str();
        }

        /// The Newtonian force relative to GR, 1 / (2 G4)
        double GG4(double x) const { return GG4_spline(clamp(x)); }
        /// The fifth force in the linear regime relative to GR's Newtonian force, 1/(3 beta)
        double b5(double x) const { return b5_spline(clamp(x)); }
        /// (r_c H0/c)^2 b5^2
        double c2(double x) const { return c2_spline(clamp(x)); }
        /// beta and r_c H0/c of the DGP form (beta infinite without a fifth force; r_c = sqrt|c2| / |b5|)
        double beta(double x) const {
            const double b = b5(x);
            return b != 0.0 ? 1.0 / (3.0 * b) : HUGE_VAL;
        }
        double rcH0(double x) const {
            const double b = b5(x);
            return b != 0.0 ? std::sqrt(std::fabs(c2(x))) / std::fabs(b) : 0.0;
        }

        double mu(double x, double /*koverH0*/) const override { return GG4(x) + b5(x); }
        double Sigma(double x, double /*koverH0*/) const override { return Sigma_spline(clamp(x)); }
        double second_order_source(double x, double koverH0) const override {
            const double E = bg.E(x);
            return 1.0 - 6.0 * c2(x) * b5(x) * bg.Omega_m(x) * E * E / mu(x, koverH0);
        }
        Screening screening() const override {
            Screening sc;
            sc.type = ScreeningType::Density;
            sc.factor = [this](double x, double rho, double efficiency) {
                const double E = bg.E(x);
                const double e = std::max(8.0 * c2(x) * bg.Omega_m(x) * E * E * efficiency * (rho - 1.0), -1.0);
                return std::fabs(e) < 1e-5 ? 1.0 : 2.0 * (std::sqrt(1.0 + e) - 1.0) / e;
            };
            sc.description = "Vainshtein (spherical top-hat): 2 (sqrt(1 + e) - 1) / e, e = 8 (r_c H0/c)^2 b5^2 "
                             "Omega_m(a) E^2 delta >= -1";
            return sc;
        }

        /// The largest relative difference of E between the table and the background (inside both ranges)
        double max_relative_E_difference() const { return max_dE; }
        double a_min() const { return std::exp(x_lo); }
        double a_max() const { return std::exp(x_hi); }

      private:
        std::string label;
        double x_lo{0.0}, x_hi{0.0}, max_dE{0.0};
        Spline GG4_spline, b5_spline, c2_spline, Sigma_spline;
        double clamp(double x) const { return std::min(std::max(x, x_lo), x_hi); }
    };

} // namespace FML::COSMOLOGY::PERTURBATIONS

#endif
