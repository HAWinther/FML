//==========================================================================
// Tests of the COLA frame and the time-stepping against the growth factors of the gravity model
//
//  1. Time-stepping: a linear (Zel'dovich) mode x = D(a), v = a^3 E dD/da integrated with the Quinn and Tassev
//     steps (no COLA) from a = 0.05 to 1 in 20 steps. The force is computed at the time of the positions apos, so
//     the kick has to be normalized there (and the Tassev drift weighted with the time of the velocities). With
//     logarithmic or power-law spacing the midpoints were used: the growth was 0.76% too small with 20 logarithmic
//     Quinn steps instead of 0.38% (linear spacing is unchanged)
//  2. The COLA frame kick (scale-independent COLA): the kicks of the LPT force over many small steps must add up to
//     minus the change of the LPT velocity of the growth factors, i.e. the frame kick must be the force of the
//     growth equations (with the source factors: the nonlinear DGP 2LPT source, 1 - f_nu for >= 2LPT). GR, DGP with
//     and without screening
//
// Build and run: make test_cola_frame && ./test_cola_frame (in this folder)
//==========================================================================
// (the cosmology before the gravity models and COLA)
#include "../src/Cosmology_LCDM.h"
#include "../src/GravityModel_DGP.h"
#include "../src/GravityModel_GR.h"
#include "../src/COLA.h"
#include "../src/Particles.h"
#include "../src/TimeStepping.h"

#include <cmath>
#include <iostream>
#include <sstream>

namespace {
    using ParameterMap = FML::UTILS::ParameterMap;
    constexpr int NDIM = 3;

    ParameterMap base_parameters() {
        ParameterMap p;
        p["cosmology_OmegaMNu"] = 0.0;
        p["cosmology_Omegab"] = 0.049;
        p["cosmology_OmegaCDM"] = 0.2685;
        p["cosmology_OmegaK"] = 0.0;
        p["cosmology_h"] = 0.671;
        p["cosmology_As"] = 2.215e-9;
        p["cosmology_ns"] = 0.966;
        p["cosmology_kpivot_mpc"] = 0.05;
        p["cosmology_Neffective"] = 3.046;
        p["cosmology_TCMB_kelvin"] = 2.7255;
        p["ic_initial_redshift"] = 19.0;
        p["ic_type_of_input"] = std::string("powerspectrum");
        p["force_use_finite_difference_force"] = false;
        p["force_finite_difference_stencil_order"] = 4;
        p["gravity_model_dgp_rcH0overc"] = 1.0;
        p["gravity_model_dgp_exact_solution"] = false;
        p["gravity_model_screening_enforce_largescale_linear"] = false;
        p["gravity_model_screening_linear_scale_hmpc"] = 0.1;
        p["gravity_model_screening_efficiency"] = 1.0;
        p["gravity_model_dgp_smoothing_filter"] = std::string("tophat");
        p["gravity_model_dgp_smoothing_scale_over_boxsize"] = 0.0;
        return p;
    }

    std::shared_ptr<GravityModel<NDIM>> make_model(const std::string & name, bool screening) {
        auto param = base_parameters();
        param["gravity_model_screening"] = screening;
        auto cosmo = std::make_shared<CosmologyLCDM>();
        cosmo->read_parameters(param);
        cosmo->init();
        std::shared_ptr<GravityModel<NDIM>> grav;
        if (name == "DGP")
            grav = std::make_shared<GravityModelDGP<NDIM>>(cosmo);
        else
            grav = std::make_shared<GravityModelGR<NDIM>>(cosmo);
        grav->read_parameters(param);
        grav->init();
        return grav;
    }

    // Silence the output of the code while f runs
    template <class F>
    void quietly(F && f) {
        std::ostringstream sink;
        auto * old = std::cout.rdbuf(sink.rdbuf());
        f();
        std::cout.rdbuf(old);
    }
} // namespace

int main() {
    int failed = 0;
    auto check = [&](bool ok, const std::string & message) {
        failed += !ok;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << message << '\n';
    };

    std::shared_ptr<GravityModel<NDIM>> gr, dgp, dgp_noscreening;
    quietly([&]() {
        gr = make_model("GR", false);
        dgp = make_model("DGP", true);
        dgp_noscreening = make_model("DGP", false);
    });

    //======================================================================
    // 1. Time-stepping of a linear mode
    //======================================================================
    {
        auto cosmo = gr->get_cosmo();
        const double amin = 0.05, amax = 1.0;
        const int nsteps = 20;
        auto D = [&](double a) { return gr->get_D_1LPT(a); };
        auto v_linear = [&](double a) { return a * a * cosmo->HoverH0_of_a(a) * gr->get_f_1LPT(a) * D(a); };
        auto integrate = [&](COLASOLVER::TimestepMethod method, COLASOLVER::ScaleFactorSpacing spacing) {
            COLASOLVER::TimestepSettings settings;
            settings.method = method;
            settings.spacing = spacing;
            settings.spacing_power = 0.5;
            settings.cola_nLPT = -2.5;
            COLASOLVER::TimeStepping<NDIM> ts(settings, cosmo, gr);
            double x = D(amin), v = v_linear(amin);
            for (auto & s : ts.steps(amin, amax, nsteps)) {
                v *= s.velocity_scale;
                v += 1.5 * cosmo->get_OmegaM() * s.apos * gr->GeffOverG(s.apos) * x * s.delta_time_kick;
                x += v * s.delta_time_drift;
            }
            return std::make_pair(x / D(amax) - 1.0, v / v_linear(amax) - 1.0);
        };
        using M = COLASOLVER::TimestepMethod;
        using S = COLASOLVER::ScaleFactorSpacing;
        // Maximum |error| in the growth and the velocity with 20 steps. With the kick normalized at the midpoint of the
        // velocity times the errors were (growth / velocity): Quinn logarithmic -0.76% / -0.84%, Quinn a^0.5 -0.84% /
        // -0.88%, Tassev logarithmic +3.7% / +4.2% (now -0.38% / -0.31%, -0.51% / -0.51%, +1.6% / +2.2%; linear spacing
        // is unchanged: -1.16% / -1.20%)
        struct Case {
            M method;
            S spacing;
            std::string name;
            double max_err_D, max_err_v;
        };
        for (const auto & c : {Case{M::Quinn, S::Linear, "Quinn linear", 0.015, 0.015},
                               Case{M::Quinn, S::Logarithmic, "Quinn logarithmic", 0.0055, 0.0055},
                               Case{M::Quinn, S::Powerlaw, "Quinn a^0.5", 0.0065, 0.0065},
                               Case{M::Tassev, S::Logarithmic, "Tassev logarithmic", 0.025, 0.03}}) {
            auto [errD, errv] = integrate(c.method, c.spacing);
            std::ostringstream msg;
            msg << "time-stepping, " << c.name << " (20 steps, a = 0.05 -> 1): linear growth error " << errD * 100
                << "%, velocity " << errv * 100 << "%";
            check(std::abs(errD) < c.max_err_D and std::abs(errv) < c.max_err_v, msg.str());
        }
    }

    //======================================================================
    // 2. The COLA frame kick adds up to minus the change of the LPT velocity
    //======================================================================
    for (auto & [name, grav] : {std::make_pair(std::string("GR"), gr),
                                std::make_pair(std::string("DGP rcH0 = 1, screening"), dgp),
                                std::make_pair(std::string("DGP rcH0 = 1, no screening"), dgp_noscreening)}) {
        auto cosmo = grav->get_cosmo();
        const double aini = 1.0 / 20.0, aend = 1.0;
        // One particle with Psi_1LPT = (1, 0, 0) and Psi_2LPT = (0, 1, 0)
        using Particle = ParticleCOLA<NDIM>;
        std::vector<Particle> p(1);
        for (int idim = 0; idim < NDIM; idim++) {
            p[0].pos[idim] = 0.5;
            p[0].vel[idim] = 0.0;
            p[0].Psi_1LPT[idim] = idim == 0 ? 1.0 : 0.0;
            p[0].Psi_2LPT[idim] = idim == 1 ? 1.0 : 0.0;
        }
        FML::PARTICLE::MPIParticles<Particle> part;
        part.create(p.data(), 1, 1, 0.0, 1.0, false);

        COLASOLVER::TimestepSettings settings;
        settings.method = COLASOLVER::TimestepMethod::Quinn;
        settings.spacing = COLASOLVER::ScaleFactorSpacing::Linear;
        COLASOLVER::TimeStepping<NDIM> ts(settings, cosmo, grav);
        std::shared_ptr<GravityModel<NDIM>> g = grav;
        quietly([&]() {
            for (auto & s : ts.steps(aini, aend, 2000))
                cola_kick_drift<NDIM,
                                Particle>(part, g, 2, aini, s.apos, s.apos_new, s.delta_time_kick, 0.0, true, false);
        });
        // The LPT velocity (per unit Psi) of order n: a^2 E f_n D_n / D_n(aini)
        auto vLPT = [&](double a, int order) {
            const double fac = a * a * cosmo->HoverH0_of_a(a);
            return order == 1 ? fac * grav->get_f_1LPT(a) * grav->get_D_1LPT(a) / grav->get_D_1LPT(aini) :
                                fac * grav->get_f_2LPT(a) * grav->get_D_2LPT(a) / grav->get_D_2LPT(aini);
        };
        for (int order : {1, 2}) {
            const double expected = -(vLPT(aend, order) - vLPT(aini, order));
            const double err = part[0].vel[order - 1] / expected - 1.0;
            std::ostringstream msg;
            msg << name << ": " << order << "LPT COLA frame kick = -(change of the LPT velocity) (rel. diff " << err
                << ")";
            check(std::abs(err) < 1e-4, msg.str());
        }
    }

    if (failed)
        std::cout << failed << " test(s) failed\n";
    else
        std::cout << "All tests passed\n";
    return failed ? 1 : 0;
}
