#ifndef COSMOLOGY_EDE_HEADER
#define COSMOLOGY_EDE_HEADER

#include "Cosmology_ScalarField.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

//=============================================================================
/// Early dark energy (axion-like, e.g. Poulin et al. 2018, Smith et al. 2020):
///
///   V(phi) = m^2 f^2 (1 - cos(phi / f))^n     plus a cosmological constant for the late-time dark energy
///
/// parametrized as in those papers by
///   fEDE      : the maximum of rho_EDE / rho_total
///   zc        : the redshift of that maximum
///   theta_ini : the initial angle phi_ini / f (the field starts frozen)
///   n         : the power (n = 3 is the usual choice; the field then decays like radiation or faster,
///               cycle-averaged w = (n-1)/(n+1))
/// m and f are found by shooting. The cosmological constant is set so that H(a=1) = H0.
///
/// NB: only the background is modified. The EDE perturbations are neglected, which is fine after
/// the EDE has decayed (start the simulation well after zc with initial conditions from a Boltzmann
/// code that includes EDE, which is where most of its effect on the matter power spectrum is).
//=============================================================================
class CosmologyEDE final : public CosmologyScalarField {
  public:
    CosmologyEDE() { name = "EDE"; }

    void read_parameters(ParameterMap & param) override {
        Cosmology::read_parameters(param);
        fEDE = param.get<double>("cosmology_ede_fEDE");
        zc = param.get<double>("cosmology_ede_zc");
        theta_ini = param.get<double>("cosmology_ede_theta_ini");
        n = param.get<double>("cosmology_ede_n");
        if (not(fEDE > 0.0 and fEDE < 0.5))
            throw std::runtime_error("EDE: need 0 < cosmology_ede_fEDE < 0.5");
        if (not(zc > 10.0 and zc < 1e7))
            throw std::runtime_error("EDE: need 10 < cosmology_ede_zc < 1e7");
        if (not(theta_ini > 0.0 and theta_ini < M_PI))
            throw std::runtime_error("EDE: need 0 < cosmology_ede_theta_ini < pi");
        if (not(n >= 1.0))
            throw std::runtime_error("EDE: need cosmology_ede_n >= 1");
    }

    void init() override {
        Cosmology::init();
        const double OmegaDE = OmegaLambda;
        OmegaLambda_bare = OmegaDE;

        // Initial guesses: the field thaws when H ~ m, and rho_EDE ~ U(theta_ini) then
        const double ac = 1.0 / (1.0 + zc);
        const double E2c = rho_other(ac);
        m = std::sqrt(E2c);
        f = std::sqrt(3.0 * fEDE * E2c / (m * m * std::pow(1.0 - std::cos(theta_ini), n)));

        const double a_stop = std::min(a_end, 50.0 * ac);
        for (int iter = 0; iter < 30; iter++) {
            // m from zc (with f fixed). The peak is earlier (higher z) for a heavier field
            auto log_zpeak_minus_target = [&](double logm) {
                m = std::exp(logm);
                solve_field(f * theta_ini, 0.0, a_stop);
                return std::log(1.0 / last_solution.a_peak) - std::log(1.0 / ac);
            };
            m = std::exp(find_root_increasing(log_zpeak_minus_target,
                                              std::log(m) - 1.0,
                                              std::log(m) + 1.0,
                                              1e-6,
                                              name + " (zc)"));
            solve_field(f * theta_ini, 0.0, a_stop);
            const double ratio = last_solution.peak_fraction / fEDE;
            const double zerr = std::fabs(last_solution.a_peak / ac - 1.0);
            if (FML::ThisTask == 0)
                std::cout << "# EDE: iteration " << iter << " m/H0 = " << m << " f/Mpl = " << f
                          << " fEDE = " << last_solution.peak_fraction << " zc = " << 1.0 / last_solution.a_peak - 1.0
                          << "\n";
            if (std::fabs(ratio - 1.0) < 1e-5 and zerr < 1e-5)
                break;
            if (iter == 29)
                throw std::runtime_error("EDE: the shooting for (m, f) did not converge");
            // The peak fraction is ~ proportional to f^2 (exactly if the EDE did not change H)
            f /= std::sqrt(ratio);
        }

        // The full solution with the cosmological constant that gives H(a=1) = H0
        for (int iter = 0; iter < 5; iter++) {
            solve_field(f * theta_ini, 0.0);
            OmegaLambda_bare = OmegaDE - rho_phi(1.0);
        }
        solve_field(f * theta_ini, 0.0);
        if (std::fabs(HoverH0_of_a(1.0) - 1.0) > 1e-6)
            throw std::runtime_error("EDE: could not get H(a=1) = H0");

        if (FML::ThisTask == 0) {
            std::cout << "# EDE: m/H0 = " << m << " f/Mpl = " << f << " fEDE = " << last_solution.peak_fraction
                      << " at z = " << 1.0 / last_solution.a_peak - 1.0 << ", E(a=1) = " << HoverH0_of_a(1.0) << "\n";
            if (last_solution.switched_to_fluid)
                std::cout << "# EDE: treated as a fluid with w = " << w_oscillating() << " for a > "
                          << last_solution.a_fluid << "\n";
        }
    }

    void info() const override {
        Cosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "# fEDE                    : " << fEDE << "\n";
            std::cout << "# zc                      : " << zc << "\n";
            std::cout << "# theta_ini               : " << theta_ini << "\n";
            std::cout << "# n                       : " << n << "\n";
            std::cout << "# m / H0                  : " << m << "\n";
            std::cout << "# f / Mpl                 : " << f << "\n";
            std::cout << "# OmegaLambda (bare)      : " << OmegaLambda_bare << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

  protected:
    // V / (3 H0^2 Mpl^2) with m in units of H0 and phi, f in units of Mpl
    double U(double phi) const override { return m * m * f * f / 3.0 * std::pow(1.0 - std::cos(phi / f), n); }
    double dUdphi(double phi) const override {
        const double c = 1.0 - std::cos(phi / f);
        if (c <= 0.0)
            return 0.0;
        return m * m * f / 3.0 * n * std::pow(c, n - 1.0) * std::sin(phi / f);
    }
    bool has_minimum() const override { return true; }
    double phi_min() const override { return 0.0; }
    double w_oscillating() const override { return (n - 1.0) / (n + 1.0); }

    double fEDE{0.1};
    double zc{3500.0};
    double theta_ini{2.8};
    double n{3.0};
    double m{1.0};
    double f{1.0};
};

#endif
