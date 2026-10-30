//==========================================================================
// Tests of normalizations in the gravity models and the initial conditions
//
//  1. m(a),beta(a) model: phi(a)/Mpl = 9 OmegaM Int_0^a beta / (a^3 (m/H0)^2) dlog(a). For beta = beta0 a^n and
//     m = m0 H0 a^m this is 9 OmegaM beta0 / m0^2 a^p / p with p = n - 3 - 2m. The integral used to start at
//     a = 0.001 (missing amin^p of it: 3% for p = 0.5, 25% for p = 0.2)
//  2. Non-gaussian IC: the large-scale potential is Phi = (3/5) g zeta with g(z = 0) = D(1)/D_EdS(1) (reference
//     values from scipy). The code used 1.5 OmegaM instead of (3/5) g, right only for OmegaM ~ 0.32
//  3. Neutrino background: the Fermi-Dirac integrals for rho_nu and p_nu (reference values from scipy) also for
//     y = m/T > 1000, where the splines used to be held constant, and the sound speed with the mass of one neutrino
//  4. Tabulated H(z) (LCDM): dlogH/dloga and w_DE at the end of the table (a = 1)
//  5. JBD beyond the end of its splines (a > 10): H(a) consistent with dlogH/dloga
//  6. The growth factors reach a = 10 (the end of the spline range; they used to stop at a = 9.886). Test 1 also
//     checks phi(a = 5) of the m(a),beta(a) model (the phi spline used to end at a = 1)
//
// Build and run: make test_models && ./test_models (in this folder)
//==========================================================================
#include "../src/Cosmology_JBD.h"
#include "../src/Cosmology_LCDM.h"
#include "../src/Cosmology_Tabulated.h"
#include "../src/GravityModel_GR.h"
#include "../src/GravityModel_mbeta.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
    using ParameterMap = FML::UTILS::ParameterMap;
    constexpr int NDIM = 3;

    int fails = 0;
    void check(bool ok, const std::string & what) {
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
        if (not ok)
            fails++;
    }

    ParameterMap cosmology_parameters(double OmegaCDM, double OmegaK = 0.0) {
        ParameterMap p;
        p["cosmology_OmegaMNu"] = 0.0;
        p["cosmology_Omegab"] = 0.05;
        p["cosmology_OmegaCDM"] = OmegaCDM;
        p["cosmology_OmegaK"] = OmegaK;
        p["cosmology_h"] = 0.7;
        p["cosmology_As"] = 2.1e-9;
        p["cosmology_ns"] = 0.96;
        p["cosmology_kpivot_mpc"] = 0.05;
        p["cosmology_Neffective"] = 3.046;
        p["cosmology_TCMB_kelvin"] = 2.7255;
        p["ic_initial_redshift"] = 19.0;
        p["ic_type_of_input"] = std::string("powerspectrum");
        return p;
    }

    template <class F>
    void quietly(F && f) {
        std::ostringstream sink;
        auto * old = std::cout.rdbuf(sink.rdbuf());
        f();
        std::cout.rdbuf(old);
    }
} // namespace

int main() {
    // 1. phi(a) of the m(a),beta(a) power-law models
    for (auto pars :
         std::vector<std::vector<double>>{{0.5, 3.4, 1.0, 0.0}, {1.0, 3.5, 2.0, 0.0}, {0.3, 4.0, 1.0, 0.25}}) {
        auto param = cosmology_parameters(0.25);
        param["force_use_finite_difference_force"] = false;
        param["force_finite_difference_stencil_order"] = 4;
        param["gravity_model_screening"] = true;
        param["gravity_model_screening_enforce_largescale_linear"] = false;
        param["gravity_model_screening_linear_scale_hmpc"] = 0.1;
        param["gravity_model_screening_efficiency"] = 1.0;
        param["gravity_model_mbeta_params"] = pars;
        auto cosmo = std::make_shared<CosmologyLCDM>();
        auto grav = std::make_shared<GravityModelmbeta<NDIM>>(cosmo);
        quietly([&]() {
            cosmo->read_parameters(param);
            cosmo->init();
            grav->read_parameters(param);
            grav->init();
        });
        const double p = pars[1] - 3.0 - 2.0 * pars[3];
        const double OmegaM = cosmo->get_OmegaM();
        for (double a : {0.1, 1.0, 5.0}) {
            const double expected = 9.0 * OmegaM * pars[0] / (pars[2] * pars[2]) * std::pow(a, p) / p;
            const double err = grav->phi_over_mpl_of_a(a) / expected - 1.0;
            check(std::fabs(err) < 2e-3,
                  "mbeta n = " + std::to_string(pars[1]) + ", m = " + std::to_string(pars[3]) +
                      ": phi(a = " + std::to_string(a) + ") / analytic - 1 = " + std::to_string(err));
        }
    }

    // 2. Growth suppression of the potential today (reference: scipy quad of the Heath integral)
    const std::vector<std::array<double, 3>> refs{{0.25, 0.0, 0.7462473310137687},
                                                  {0.35, 0.0, 0.8069723805655855},
                                                  {1.0, 0.0, 0.9999999999999998},
                                                  {0.30, 0.05, 0.7417801785360228}};
    for (auto & r : refs) {
        // OmegaM = OmegaCDM + Omegab (0.05)
        auto param = cosmology_parameters(r[0] - 0.05, r[1]);
        auto cosmo = std::make_shared<CosmologyLCDM>();
        quietly([&]() {
            cosmo->read_parameters(param);
            cosmo->init();
        });
        const double g = cosmo->get_potential_growth_suppression_today();
        const double OmegaM = cosmo->get_OmegaM();
        check(std::fabs(g / r[2] - 1.0) < 1e-6,
              "g(z = 0) for OmegaM = " + std::to_string(OmegaM) + ", OmegaK = " + std::to_string(r[1]) + ": " +
                  std::to_string(g) + " (scipy " + std::to_string(r[2]) +
                  "); the old factor 1.5 OmegaM gave fNL_eff / fNL = " + std::to_string(1.5 * OmegaM / (0.6 * g)));
    }

    // 3. Neutrino Boltzmann integrals F(y) = int x^2 sqrt(x^2 + y^2) / (e^x + 1) dx and
    //    G(y) = int x^4 / (3 sqrt(x^2 + y^2)) / (e^x + 1) dx over F(0) (reference: scipy quad). The splines used to
    //    stop at y = 1000 and hold the end value of F / (F(0) + C y) (0.28% low at y = 1e4) and of G (G falls as 1/y:
    //    10x too large at y = 1e4)
    {
        auto param = cosmology_parameters(0.25);
        param["cosmology_OmegaMNu"] = 0.6 / 93.14 / (0.7 * 0.7); // Mnu = 0.6 eV
        auto cosmo = std::make_shared<CosmologyLCDM>();
        quietly([&]() {
            cosmo->read_parameters(param);
            cosmo->init();
        });
        const std::vector<std::array<double, 3>> nu_refs{{1.0, 1.0639626203517059, 0.3153347125924614},
                                                         {10.0, 3.3658091566873827, 0.12135914816042563},
                                                         {100.0, 31.752700846855834, 0.01366565960247472},
                                                         {999.0, 317.0066003881602, 0.0013700023905826686},
                                                         {1001.0, 317.64124001676225, 0.001367265206827081},
                                                         {1e4, 3173.218877658416, 0.00013686531818528702},
                                                         {1e5, 31732.186744134033, 1.3686533893784033e-05},
                                                         {1e6, 317321.8672380953, 1.3686533914536596e-06}};
        const double F0 = 7.0 / 120.0 * std::pow(M_PI, 4);
        for (auto & r : nu_refs) {
            const double errF = cosmo->get_neutrino_boltzmann_integral_energydensity(r[0]) / F0 / r[1] - 1.0;
            const double errG = cosmo->get_neutrino_boltzmann_integral_pressure(r[0]) / F0 / r[2] - 1.0;
            check(std::fabs(errF) < 1e-5 and std::fabs(errG) < 1e-4,
                  "neutrino F(y), G(y) at y = " + std::to_string(r[0]) + ": F / scipy - 1 = " + std::to_string(errF) +
                      ", G / scipy - 1 = " + std::to_string(errG));
        }
        // Non-relativistic today and later: rho_nu ~ a^-3 (dlog(rho_nu)/dloga = -3 + O(1/y^2)), w_nu ~ 1/y
        for (double a : {1.0, 10.0}) {
            const double y = cosmo->get_Mnu_eV() / cosmo->get_N_nu() / cosmo->get_neutrino_temperature_eV(a);
            const double dlogrho = cosmo->get_drhoNudloga_exact(a) / cosmo->get_rhoNu_exact(a);
            const double w = cosmo->get_pNu_exact(a) / cosmo->get_rhoNu_exact(a);
            // <p/rho> -> <x^2>/(3 y^2) = I4 / (3 I2 y^2) with I4 / I2 = 15 zeta(5) / zeta(3)
            const double w_expected = 15.0 * 1.03692775514336 / 1.20205690315959 / (3.0 * y * y);
            check(std::fabs(dlogrho + 3.0) < 1e-5 and std::fabs(w / w_expected - 1.0) < 1e-3,
                  "Mnu = 0.6 eV at a = " + std::to_string(a) + " (y = " + std::to_string(y) +
                      "): dlog(rho_nu)/dloga = " + std::to_string(dlogrho) +
                      ", w_nu / expected - 1 = " + std::to_string(w / w_expected - 1.0));
        }
        // The sound speed uses the mass of one neutrino (Mnu / N_nu), not the sum
        const double cs = cosmo->get_neutrino_sound_speed_cs_over_c(1.0);
        const double cs_expected = std::sqrt(25.0 * 1.03692775514336 / 1.20205690315959 / 3.0) *
                                   cosmo->get_neutrino_temperature_eV(1.0) / (cosmo->get_Mnu_eV() / 3.0);
        check(std::fabs(cs / cs_expected - 1.0) < 1e-10,
              "neutrino sound speed today: " + std::to_string(cs) + " (expected " + std::to_string(cs_expected) + ")");
    }

    // 4. Tabulated H(z) of an LCDM background: dlogH/dloga at the end of the table (a = 1). The natural end condition
    //    of the spline gave dlogH/dloga 6e-3 off (w_DE(z = 0) = -0.994) for this table with dlog(a) = 0.023
    {
        auto param = cosmology_parameters(0.25);
        auto lcdm = std::make_shared<CosmologyLCDM>();
        quietly([&]() {
            lcdm->read_parameters(param);
            lcdm->init();
        });
        const std::string filename = "test_models_tabulated_lcdm.txt";
        {
            std::ofstream fp(filename);
            fp << std::setprecision(17);
            const int n = 400;
            for (int i = 0; i < n; i++) {
                const double z = std::exp(std::log(1e4 + 1.0) * i / double(n - 1)) - 1.0;
                fp << z << " " << lcdm->HoverH0_of_a(1.0 / (1.0 + z)) << "\n";
            }
        }
        param["cosmology_tabulated_filename"] = filename;
        param["cosmology_tabulated_format"] = std::string("H");
        auto tab = std::make_shared<CosmologyTabulated>();
        quietly([&]() {
            tab->read_parameters(param);
            tab->init();
        });
        std::remove(filename.c_str());
        for (double a : {1.0, 0.98, 0.5}) {
            const double err = tab->dlogHdloga_of_a(a) - lcdm->dlogHdloga_of_a(a);
            check(std::fabs(err) < 1e-5 and std::fabs(tab->w_DE(a) + 1.0) < 1e-5,
                  "Tabulated LCDM at a = " + std::to_string(a) + ": dlogH/dloga - LCDM = " + std::to_string(err) +
                      ", w_DE + 1 = " + std::to_string(tab->w_DE(a) + 1.0));
        }
    }

    // 5. JBD beyond the end of the splines (a > 10): H(a) must be consistent with dlogH/dloga (H was held constant
    //    while dlogH/dloga was the slope at the end)
    {
        auto param = cosmology_parameters(0.25);
        param["cosmology_JBD_wBD"] = 100.0;
        param["cosmology_JBD_GeffG_today"] = 1.0;
        param["cosmology_JBD_density_parameter_definition"] = std::string("hi-class");
        auto jbd = std::make_shared<CosmologyJBD>();
        quietly([&]() {
            jbd->read_parameters(param);
            jbd->init();
        });
        for (double a : {9.9, 20.0}) {
            const double fd = std::log(jbd->HoverH0_of_a(a * 1.01) / jbd->HoverH0_of_a(a / 1.01)) / std::log(1.0201);
            const double err = jbd->dlogHdloga_of_a(a) - fd;
            check(std::fabs(err) < 1e-6,
                  "JBD at a = " + std::to_string(a) + ": dlogH/dloga = " + std::to_string(jbd->dlogHdloga_of_a(a)) +
                      ", finite difference of H = " + std::to_string(fd));
        }
    }

    // 6. The growth factors are splined up to a = 10 (the grid used to stop at a = 9.886, so D(a) was constant
    //    beyond): D(10) / D(9.9) = exp(int f dloga)
    {
        auto param = cosmology_parameters(0.25);
        param["force_use_finite_difference_force"] = false;
        param["force_finite_difference_stencil_order"] = 4;
        auto cosmo = std::make_shared<CosmologyLCDM>();
        auto grav = std::make_shared<GravityModelGR<NDIM>>(cosmo);
        quietly([&]() {
            cosmo->read_parameters(param);
            cosmo->init();
            grav->read_parameters(param);
            grav->init();
        });
        const double dlogD = std::log(grav->get_D_1LPT(10.0) / grav->get_D_1LPT(9.9));
        const double expected = grav->get_f_1LPT(9.95) * std::log(10.0 / 9.9);
        check(std::fabs(dlogD / expected - 1.0) < 1e-3,
              "GR growth: log(D(10)/D(9.9)) = " + std::to_string(dlogD) + " (f dloga = " + std::to_string(expected) +
                  ")");
    }

    std::cout << (fails == 0 ? "All model tests passed\n" : std::to_string(fails) + " model test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
