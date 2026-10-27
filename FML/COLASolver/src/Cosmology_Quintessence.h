#ifndef COSMOLOGY_QUINTESSENCE_HEADER
#define COSMOLOGY_QUINTESSENCE_HEADER

#include "ParameterDefaults.h"

#include "Cosmology_ScalarField.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

//=============================================================================
/// Quintessence: a minimally coupled scalar field as the dark energy (no cosmological
/// constant). V(phi) = V0 u(phi) with phi in units of the reduced Planck mass and
///
///   exponential        : u = exp(-lambda phi)
///   inverse_power_law  : u = phi^(-alpha)                (Ratra-Peebles, has a tracker)
///   pngb               : u = 1 + cos(phi / f)            (pseudo Nambu-Goldstone boson / axion-like)
///   hilltop            : u = (1 - (|phi| / mu)^p)^2      (flat top at phi = 0, minimum at phi = mu)
///
/// The field starts frozen deep in the radiation era at phi_ini with dphi/dlna = dphidlna_ini
/// (thawing models: dphidlna_ini = 0 and the field starts to roll when H ~ m; tracking models:
/// the inverse power law approaches its tracker from a wide range of phi_ini). V0 is found by
/// shooting so that Omega_phi today is 1 - OmegaM - OmegaR - OmegaNu - OmegaK.
//=============================================================================
class CosmologyQuintessence final : public CosmologyScalarField {
  public:
    CosmologyQuintessence() { name = "Quintessence"; }

    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        potential = param.get<std::string>("cosmology_quintessence_potential");
        phi_ini = COLASOLVER::get_parameter<double>(param, "cosmology_quintessence_phi_ini");
        dphidlna_ini = COLASOLVER::get_parameter<double>(param, "cosmology_quintessence_dphidlna_ini");
        if (potential == "exponential") {
            lambda = param.get<double>("cosmology_quintessence_lambda");
        } else if (potential == "inverse_power_law") {
            alpha = param.get<double>("cosmology_quintessence_alpha");
            if (not(phi_ini > 0.0))
                throw std::runtime_error("Quintessence: inverse_power_law needs cosmology_quintessence_phi_ini > 0");
        } else if (potential == "pngb") {
            f = param.get<double>("cosmology_quintessence_f");
            if (not(f > 0.0))
                throw std::runtime_error("Quintessence: pngb needs cosmology_quintessence_f > 0");
        } else if (potential == "hilltop") {
            mu = param.get<double>("cosmology_quintessence_mu");
            p = param.get<double>("cosmology_quintessence_p");
            if (not(mu > 0.0) or not(p > 0.0))
                throw std::runtime_error("Quintessence: hilltop needs cosmology_quintessence_mu > 0 and _p > 0");
        } else {
            throw std::runtime_error("Quintessence: unknown cosmology_quintessence_potential = " + potential +
                                     " (exponential, inverse_power_law, pngb, hilltop)");
        }
        if (dphidlna_ini * dphidlna_ini >= 6.0)
            throw std::runtime_error("Quintessence: need |cosmology_quintessence_dphidlna_ini| < sqrt(6)");
    }

    void init() override {
        Cosmology::init();
        // The dark energy density today we want (OmegaLambda is set to this by Cosmology::init)
        const double OmegaDE = OmegaLambda;
        OmegaLambda_bare = 0.0;
        if (u(phi_ini) <= 0.0)
            throw std::runtime_error("Quintessence: the potential is zero at phi_ini");

        // Shoot for V0 so that rho_phi(a=1) = OmegaDE (i.e. H(a=1) = H0)
        auto rho_today_minus_target = [&](double logV0) {
            V0 = std::exp(logV0);
            solve_field(phi_ini, dphidlna_ini);
            return std::log(rho_phi(1.0) / OmegaDE);
        };
        const double guess = std::log(OmegaDE / u(phi_ini));
        const double logV0 = find_root_increasing(rho_today_minus_target, guess - 1.0, guess + 1.0, 1e-10, name);
        rho_today_minus_target(logV0);
        if (std::fabs(HoverH0_of_a(1.0) - 1.0) > 1e-6)
            throw std::runtime_error(
                "Quintessence: no potential amplitude gives the dark energy density today for these parameters "
                "(E(a=1) = " + std::to_string(HoverH0_of_a(1.0)) +
                "). Typically the field rolls (or oscillates) too early: use a flatter potential (e.g. a larger f, mu "
                "or smaller lambda) or start closer to the top of the potential");

        if (FML::ThisTask == 0) {
            std::cout << "# Quintessence: V0 / (3 H0^2 Mpl^2) = " << V0 << " gives E(a=1) = " << HoverH0_of_a(1.0)
                      << ", w(a=1) = " << w_phi(1.0) << "\n";
            if (last_solution.switched_to_fluid)
                std::cout << "# Quintessence: the field oscillates; treated as a fluid with w = 0 for a > "
                          << last_solution.a_fluid << "\n";
        }
    }

    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# Potential               : " << potential << "\n";
            if (potential == "exponential")
                std::cout << "# lambda                  : " << lambda << "\n";
            if (potential == "inverse_power_law")
                std::cout << "# alpha                   : " << alpha << "\n";
            if (potential == "pngb")
                std::cout << "# f / Mpl                 : " << f << "\n";
            if (potential == "hilltop")
                std::cout << "# mu / Mpl, p             : " << mu << " , " << p << "\n";
            std::cout << "# phi_ini / Mpl           : " << phi_ini << "\n";
            std::cout << "# dphi/dlna ini           : " << dphidlna_ini << "\n";
            std::cout << "# V0 / (3 H0^2 Mpl^2)     : " << V0 << "\n";
            std::cout << "# w(z = 0, 1, 2)          : " << w_phi(1.0) << " " << w_phi(0.5) << " " << w_phi(1.0 / 3.0)
                      << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

  protected:
    double u(double phi) const {
        if (potential == "exponential")
            return std::exp(-lambda * phi);
        if (potential == "inverse_power_law")
            return phi > 0.0 ? std::pow(phi, -alpha) : 1e300;
        if (potential == "pngb")
            return 1.0 + std::cos(phi / f);
        // hilltop
        const double s = 1.0 - std::pow(std::fabs(phi) / mu, p);
        return s * s;
    }
    double dudphi(double phi) const {
        if (potential == "exponential")
            return -lambda * std::exp(-lambda * phi);
        if (potential == "inverse_power_law")
            return phi > 0.0 ? -alpha * std::pow(phi, -alpha - 1.0) : 0.0;
        if (potential == "pngb")
            return -std::sin(phi / f) / f;
        // hilltop
        const double x = std::fabs(phi) / mu;
        const double sign = phi >= 0.0 ? 1.0 : -1.0;
        return phi == 0.0 ? 0.0 : -2.0 * (1.0 - std::pow(x, p)) * p * std::pow(x, p - 1.0) / mu * sign;
    }

    double U(double phi) const override { return V0 * u(phi); }
    double dUdphi(double phi) const override { return V0 * dudphi(phi); }

    bool has_minimum() const override { return potential == "pngb" or potential == "hilltop"; }
    double phi_min() const override {
        if (potential == "pngb")
            return M_PI * f;
        return mu;
    }
    // Both minima are quadratic (but the hilltop minimum is at V = 0, the PNGB one too): w = 0
    double w_oscillating() const override { return 0.0; }

    std::string potential;
    double phi_ini{0.0};
    double dphidlna_ini{0.0};
    double V0{1.0};
    double lambda{1.0};
    double alpha{1.0};
    double f{1.0};
    double mu{1.0};
    double p{2.0};
};

#endif
