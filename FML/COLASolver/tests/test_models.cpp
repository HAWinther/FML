//==========================================================================
// Tests of normalizations in the gravity models and the initial conditions
//
//  1. m(a),beta(a) model: phi(a)/Mpl = 9 OmegaM Int_0^a beta / (a^3 (m/H0)^2) dlog(a). For beta = beta0 a^n and
//     m = m0 H0 a^m this is 9 OmegaM beta0 / m0^2 a^p / p with p = n - 3 - 2m. The integral used to start at
//     a = 0.001 (missing amin^p of it: 3% for p = 0.5, 25% for p = 0.2)
//  2. Non-gaussian IC: the large-scale potential is Phi = (3/5) g zeta with g(z = 0) = D(1)/D_EdS(1) (reference
//     values from scipy). The code used 1.5 OmegaM instead of (3/5) g, right only for OmegaM ~ 0.32
//
// Build and run: make test_models && ./test_models (in this folder)
//==========================================================================
#include "../src/Cosmology_LCDM.h"
#include "../src/GravityModel_mbeta.h"

#include <cmath>
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
        for (double a : {0.1, 1.0}) {
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

    std::cout << (fails == 0 ? "All model tests passed\n" : std::to_string(fails) + " model test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
