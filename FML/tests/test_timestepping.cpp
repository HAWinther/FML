#include <FML/SimulationCode/src/TimeStepping.h>
#include <FML/LPT/GrowthHistory.h>
#include <FML/NBody/TimeStepping.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::near;

using FML::TESTS::check_local;
using namespace FML::NBODY;
using FML::COSMOLOGY::GrowthHistory;
GrowthHistory eds_growth(double normalization = 1.0) {
    return {[=](double a) { return normalization * a; },
            [](double) { return 1.0; },
            [=](double a) { return -3.0 / 7.0 * normalization * normalization * a * a; },
            [](double) { return 2.0; }};
}
double eds_H(double a) { return std::pow(a, -1.5); }
double eds_force(double a) { return 1.5 * a; }
void test_analytic_coefficients() {
    for (auto method : {TimestepMethod::Quinn, TimestepMethod::Tassev}) {
        TimestepOptions options;
        options.method = method;
        options.cola_nLPT = -2.5;
        TimeStepping stepping(options, eds_H, eds_force);
        bool correct = true;
        for (auto step : stepping.steps(0.05, 1.0, 7)) {
            const double lo = step.apos, hi = step.apos_new;
            const double vl = step.avel, vh = step.avel_new;
            double drift, kick;
            if (method == TimestepMethod::Quinn) {
                drift = 2 * (1 / std::sqrt(lo) - 1 / std::sqrt(hi));
                const double amid = vl == 0.05 ? vl : vh == 1.0 ? vh : (vl + vh) / 2;
                kick = 2 * (std::sqrt(vh) - std::sqrt(vl)) / amid;
            } else {
                const double n = options.cola_nLPT;
                drift = (std::pow(hi, n - 0.5) - std::pow(lo, n - 0.5)) / ((n - 0.5) * std::pow((lo + hi) / 2, n));
                const double amid = vl == 0.05 ? vl : vh == 1.0 ? vh : (vl + vh) / 2;
                kick = (std::pow(vh, n) - std::pow(vl, n)) / (n * std::pow(amid, n + 0.5));
            }
            correct = correct && near(step.delta_time_drift, drift, 2e-6) && near(step.delta_time_kick, kick, 2e-6);
        }
        check_local(correct,
                    method == TimestepMethod::Quinn ? "Quinn coefficients match independent EdS integrals" :
                                                      "Tassev coefficients match independent EdS integrals");
    }
}
void test_growing_modes() {
    for (auto method : {TimestepMethod::FastPM, TimestepMethod::BullFrog}) {
        bool linear = true, second_order = true, normalization_independent = true;
        for (auto spacing : {ScaleFactorSpacing::Linear,
                             ScaleFactorSpacing::Logarithmic,
                             ScaleFactorSpacing::Powerlaw,
                             ScaleFactorSpacing::GrowthFactor}) {
            for (int count : {1, 4, 17}) {
                for (double normalization : {1.0, 23.0}) {
                    TimestepOptions options;
                    options.method = method;
                    options.spacing = spacing;
                    options.spacing_power = 0.4;
                    const auto growth = eds_growth(normalization);
                    TimeStepping stepping(options, eds_H, eds_force, growth);
                    double x1 = growth.linear(0.05), x2 = growth.second_order(0.05);
                    double v1 = growth.linear_momentum(0.05, eds_H(0.05));
                    double v2 = v1 * growth.second_order_slope(0.05);
                    const auto steps = stepping.steps(0.05, 1.0, count);
                    for (auto s : steps) {
                        // Evaluate the perturbative force at the staggered position. The second-order
                        // coefficient follows g2 = x2 / D - D; it is not the true endpoint LPT force.
                        const double D = growth.linear(s.apos), Gf = eds_force(s.apos) * D;
                        v1 = s.velocity_scale * v1 + s.delta_time_kick * Gf;
                        v2 = s.velocity_scale * v2 + s.delta_time_kick * Gf * (x2 / D - D);
                        x1 += s.delta_time_drift * v1;
                        x2 += s.delta_time_drift * v2;
                    }
                    linear = linear && near(x1, growth.linear(1.0), 2e-9) &&
                             near(v1, growth.linear_momentum(1.0, 1.0), 2e-9);
                    if (method == TimestepMethod::BullFrog)
                        second_order =
                            second_order && near(x2, growth.second_order(1.0), 2e-9) &&
                            near(v2, growth.linear_momentum(1.0, 1.0) * growth.second_order_slope(1.0), 2e-9);
                    normalization_independent = normalization_independent && near(x1 / normalization, 1.0, 2e-9);
                }
            }
        }
        check_local(linear,
                    method == TimestepMethod::FastPM ? "FastPM follows the linear growing mode at coarse steps" :
                                                       "BullFrog follows the linear growing mode at coarse steps");
        if (method == TimestepMethod::BullFrog)
            check_local(second_order, "BullFrog follows the EdS second-order displacement and velocity");
        check_local(normalization_independent, "growth normalization is consistent across all timestep spacings");
    }
}
/// EdS with a constant velocity friction Gamma (dv/dlna = -Gamma v + force): D1 = a^p, D2 = c2 a^2p (see
/// Cosmology/Growth/tests/test_growth.cpp) and the friction integral I(a) = Gamma ln a. FastPM must follow the linear
/// mode and BullFrog the linear and second-order modes exactly at any step size; Quinn and Tassev converge to them at
/// second order (without the friction integral they do not). A free particle (no force) slows down as
/// v = v0 e^-(I(a) - I(amin)) exactly with Quinn, Tassev and FastPM, and with Quinn also its position is exact
void test_friction() {
    const double Gamma = 0.7, b = 0.5 + Gamma;
    const double p = 0.5 * (-b + std::sqrt(b * b + 6.0));
    const double c2 = -1.5 / (4 * p * p + 2 * b * p - 1.5);
    GrowthHistory growth([=](double a) { return std::pow(a, p); },
                         [=](double) { return p; },
                         [=](double a) { return c2 * std::pow(a, 2 * p); },
                         [=](double) { return 2 * p; });
    auto friction_integral = [=](double a) { return Gamma * std::log(a); };
    const double amin = 0.05, amax = 1.0;

    // Evolve the linear (x1) and second-order (x2) modes as in test_growing_modes; returns the largest relative error
    // of x1, v1 (and of x2, v2 with second) at amax
    auto evolve = [&](TimestepMethod method, ScaleFactorSpacing spacing, int nsteps, bool with_friction, bool second) {
        TimestepOptions options;
        options.method = method;
        options.spacing = spacing;
        options.spacing_power = 0.4;
        options.cola_nLPT = -2.5;
        std::function<double(double)> I;
        if (with_friction)
            I = friction_integral;
        TimeStepping stepping(options, eds_H, eds_force, growth, I);
        double x1 = growth.linear(amin), x2 = growth.second_order(amin);
        double v1 = growth.linear_momentum(amin, eds_H(amin));
        double v2 = v1 * growth.second_order_slope(amin);
        for (auto s : stepping.steps(amin, amax, nsteps)) {
            const double D = growth.linear(s.apos), Gf = eds_force(s.apos) * D;
            v1 = s.velocity_scale * v1 + s.delta_time_kick * Gf;
            v2 = s.velocity_scale * v2 + s.delta_time_kick * Gf * (x2 / D - D);
            x1 += s.delta_time_drift * v1;
            x2 += s.delta_time_drift * v2;
        }
        const double v1true = growth.linear_momentum(amax, eds_H(amax));
        double err = std::max(std::fabs(x1 / growth.linear(amax) - 1), std::fabs(v1 / v1true - 1));
        if (second)
            err = std::max({err,
                            std::fabs(x2 / growth.second_order(amax) - 1),
                            std::fabs(v2 / (v1true * growth.second_order_slope(amax)) - 1)});
        return err;
    };

    bool fastpm = true, bullfrog = true;
    for (auto spacing : {ScaleFactorSpacing::Linear,
                         ScaleFactorSpacing::Logarithmic,
                         ScaleFactorSpacing::Powerlaw,
                         ScaleFactorSpacing::GrowthFactor})
        for (int count : {1, 4, 17}) {
            fastpm = fastpm && evolve(TimestepMethod::FastPM, spacing, count, true, false) < 2e-9;
            bullfrog = bullfrog && evolve(TimestepMethod::BullFrog, spacing, count, true, true) < 2e-9;
        }
    check_local(fastpm, "friction: FastPM follows the linear growing mode at coarse steps");
    check_local(bullfrog, "friction: BullFrog follows the linear and second-order modes at coarse steps");

    // A free particle: v = e^-(I(a) - I(amin)) = (amin/a)^Gamma, x = int_amin^amax (amin/a)^Gamma / (a^3 E) da with
    // a^3 E = a^(3/2) (no force: FastPM's linear-mode kick is also zero, the force is)
    const double x_free = (std::pow(amax, -0.5 - Gamma) - std::pow(amin, -0.5 - Gamma)) / (-0.5 - Gamma) *
                          std::pow(amin, Gamma);
    for (auto method : {TimestepMethod::Quinn, TimestepMethod::Tassev, TimestepMethod::FastPM}) {
        for (bool with_friction : {true, false}) {
            TimestepOptions options;
            options.method = method;
            options.cola_nLPT = -2.5;
            std::function<double(double)> I;
            if (with_friction)
                I = friction_integral;
            TimeStepping stepping(options, eds_H, eds_force, growth, I);
            double x = 0.0, v = 1.0;
            for (auto s : stepping.steps(amin, amax, 7)) {
                v *= s.velocity_scale;
                x += s.delta_time_drift * v;
            }
            const std::string name = method == TimestepMethod::Quinn ? "Quinn" :
                                     method == TimestepMethod::Tassev ? "Tassev" :
                                                                        "FastPM";
            const bool v_exact = near(v, std::pow(amin / amax, Gamma), 1e-12);
            if (with_friction) {
                check_local(v_exact, "friction: " + name + " slows a free particle down exactly");
                if (method == TimestepMethod::Quinn)
                    check_local(near(x, x_free, 1e-7), "friction: Quinn moves a free particle exactly");
            } else
                check_local(not v_exact, "friction: " + name + " without the friction integral does not");
        }
    }

    for (auto method : {TimestepMethod::Quinn, TimestepMethod::Tassev}) {
        const double e1 = evolve(method, ScaleFactorSpacing::Linear, 20, true, false);
        const double e2 = evolve(method, ScaleFactorSpacing::Linear, 40, true, false);
        const double e4 = evolve(method, ScaleFactorSpacing::Linear, 80, true, false);
        const double e_nofriction = evolve(method, ScaleFactorSpacing::Linear, 80, false, false);
        const std::string name = method == TimestepMethod::Quinn ? "Quinn" : "Tassev";
        std::cout << "       friction: " << name << " error with 20, 40, 80 steps: " << e1 << " " << e2 << " " << e4
                  << " (80 steps without the friction integral: " << e_nofriction << ")\n";
        // (As without friction, where the errors are 0.7 times these; Tassev is meant for the COLA frame)
        check_local(e2 < e1 / 3.0 and e4 < e2 / 3.0 and e4 < (method == TimestepMethod::Quinn ? 1e-3 : 1e-2),
                    "friction: " + name + " converges to the growing mode at second order");
        check_local(e_nofriction > 10 * e4, "friction: " + name + " without the friction integral is wrong");
    }
}
/// Flat LCDM with OmegaM = 0.3 (the cosmology is a model of FML's background module, made from parameters)
std::shared_ptr<Cosmology> make_test_cosmology() {
    FML::UTILS::ParameterMap p;
    p["cosmology_Omegab"] = 0.05;
    p["cosmology_OmegaCDM"] = 0.25;
    p["cosmology_OmegaMNu"] = 0.0;
    p["cosmology_neutrino_masses_eV"] = std::vector<double>{};
    p["cosmology_OmegaK"] = 0.0;
    p["cosmology_h"] = 0.7;
    p["cosmology_As"] = 2.1e-9;
    p["cosmology_ns"] = 0.965;
    p["cosmology_kpivot_mpc"] = 0.05;
    p["cosmology_Neffective"] = 3.046;
    p["cosmology_TCMB_kelvin"] = 2.7255;
    auto cosmo = std::make_shared<Cosmology>("LCDM");
    cosmo->read_parameters(p);
    cosmo->init();
    return cosmo;
}
class TestGravity : public GravityModel<3> {
  public:
    /// The growth factors with G_eff/G = 1 + 0.1 a (a D(a) that is not a power law)
    explicit TestGravity(std::shared_ptr<Cosmology> cosmo) : GravityModel<3>(cosmo, "time-step test") {
        scaledependent_growth = false;
        aini = 0.05;
        compute_growth_factors();
    }
    double GeffOverG(double a, double = 0) const override { return 1 + 0.1 * a; }
    void use_finite_difference(bool enabled) { force_use_finite_difference_force = enabled; }
    void compute_force(double a,
                       double,
                       FFTWGrid<3> & density,
                       std::string method,
                       std::array<FFTWGrid<3>, 3> & force,
                       const FML::NBODY::ForceOptions & options) const override {
        compute_force_from_total_density(density, 1.5 * cosmo->get_OmegaM() * a, method, force, options);
    }
};

/// Several species with different frictions that feel the same force (dark scattering with CDM and baryon particles:
/// the CDM has the friction, the baryons none). EdS with a constant Gamma on species c (weight f_c) and none on b: the
/// growing mode is D_s = A_s a^p with A_s (p^2 + (1/2 + Gamma_s) p) = 3/2 (f_c A_c + f_b A_b). One TimeStepping per
/// species (its friction integral, growth history and the growth of the force) and restep of reference steps: FastPM
/// follows both species exactly at any step size, Quinn converges at second order. restep of a TimeStepping's own
/// steps gives them back (to round-off); BullFrog has no restep
void test_species_friction() {
    const double Gamma = 0.8, f_c = 0.84, f_b = 1.0 - f_c;
    // A_b = 1, A_c = (p^2 + p/2) / (p^2 + (1/2 + Gamma) p), and p^2 + p/2 = 3/2 (f_c A_c + f_b): bisection in p
    auto A_c_of = [=](double p) { return (p * p + 0.5 * p) / (p * p + (0.5 + Gamma) * p); };
    auto residual = [=](double p) { return p * p + 0.5 * p - 1.5 * (f_c * A_c_of(p) + f_b); };
    double plo = 0.1, phi = 1.0;
    for (int i = 0; i < 200; i++) {
        const double pm = 0.5 * (plo + phi);
        (residual(pm) < 0 ? plo : phi) = pm;
    }
    const double p = 0.5 * (plo + phi), A_c = A_c_of(p), A_cb = f_c * A_c + f_b;
    auto history = [=](double A) {
        return GrowthHistory([=](double a) { return A * std::pow(a, p); },
                             [=](double) { return p; },
                             [=](double a) { return -3.0 / 7.0 * A * A * std::pow(a, 2 * p); },
                             [=](double) { return 2 * p; });
    };
    auto force_growth = [=](double a) { return A_cb * std::pow(a, p); };
    auto I_c = [=](double a) { return Gamma * std::log(a); };
    const double amin = 0.05, amax = 1.0;

    auto evolve = [&](TimestepMethod method, ScaleFactorSpacing spacing, int nsteps) {
        TimestepOptions options;
        options.method = method;
        options.spacing = spacing;
        options.spacing_power = 0.4;
        // The reference steps (their scale factors) from the CDM+baryon fluid; the species' coefficients by restep
        TimeStepping reference(options, eds_H, eds_force, history(A_cb), {}, {});
        TimeStepping cdm(options, eds_H, eds_force, history(A_c), I_c, force_growth);
        TimeStepping baryons(options, eds_H, eds_force, history(1.0), {}, force_growth);
        const auto steps = reference.steps(amin, amax, nsteps);
        const auto sc = cdm.restep(steps), sb = baryons.restep(steps);
        double xc = A_c * std::pow(amin, p), xb = std::pow(amin, p);
        double vc = history(A_c).linear_momentum(amin, eds_H(amin)), vb = history(1.0).linear_momentum(amin, eds_H(amin));
        for (size_t i = 0; i < steps.size(); i++) {
            const double F = eds_force(steps[i].apos) * (f_c * xc + f_b * xb);
            vc = sc[i].velocity_scale * vc + sc[i].delta_time_kick * F;
            vb = sb[i].velocity_scale * vb + sb[i].delta_time_kick * F;
            xc += sc[i].delta_time_drift * vc;
            xb += sb[i].delta_time_drift * vb;
        }
        return std::max({std::fabs(xc / (A_c * std::pow(amax, p)) - 1),
                         std::fabs(xb / std::pow(amax, p) - 1),
                         std::fabs(vc / history(A_c).linear_momentum(amax, eds_H(amax)) - 1),
                         std::fabs(vb / history(1.0).linear_momentum(amax, eds_H(amax)) - 1)});
    };

    bool fastpm = true;
    for (auto spacing : {ScaleFactorSpacing::Linear,
                         ScaleFactorSpacing::Logarithmic,
                         ScaleFactorSpacing::Powerlaw,
                         ScaleFactorSpacing::GrowthFactor})
        for (int count : {1, 4, 17})
            fastpm = fastpm && evolve(TimestepMethod::FastPM, spacing, count) < 2e-9;
    check_local(fastpm, "species friction: FastPM follows both species exactly at coarse steps (p = " +
                            std::to_string(p) + ", A_c = " + std::to_string(A_c) + ")");
    const double e1 = evolve(TimestepMethod::Quinn, ScaleFactorSpacing::Linear, 20);
    const double e2 = evolve(TimestepMethod::Quinn, ScaleFactorSpacing::Linear, 40);
    const double e4 = evolve(TimestepMethod::Quinn, ScaleFactorSpacing::Linear, 80);
    std::cout << "       species friction: Quinn error with 20, 40, 80 steps: " << e1 << " " << e2 << " " << e4 << "\n";
    check_local(e1 / e2 > 3.5 and e2 / e4 > 3.5 and e4 < 2e-3,
                "species friction: Quinn converges to the two-species growing mode at second order");

    // restep of a TimeStepping's own steps gives them back (all KDK methods and spacings, with and without friction;
    // to round-off: steps() of Quinn/Tassev recompute a_min of the first step as exp(log(a_min)) with log spacing)
    bool same = true;
    for (auto method : {TimestepMethod::Quinn, TimestepMethod::Tassev, TimestepMethod::FastPM})
        for (auto spacing : {ScaleFactorSpacing::Linear,
                             ScaleFactorSpacing::Logarithmic,
                             ScaleFactorSpacing::Powerlaw,
                             ScaleFactorSpacing::GrowthFactor})
            for (bool friction : {false, true}) {
                TimestepOptions options;
                options.method = method;
                options.spacing = spacing;
                options.spacing_power = 0.4;
                options.cola_nLPT = -2.5;
                TimeStepping s(options, eds_H, eds_force, history(A_c), friction ? std::function<double(double)>(I_c) : nullptr);
                const auto a = s.steps(amin, amax, 7), b = s.restep(a);
                for (size_t i = 0; i < a.size(); i++)
                    same = same and a[i].apos == b[i].apos and a[i].apos_new == b[i].apos_new and
                           a[i].avel == b[i].avel and a[i].avel_new == b[i].avel_new and
                           near(a[i].delta_time_drift, b[i].delta_time_drift, 1e-12) and
                           near(a[i].delta_time_kick, b[i].delta_time_kick, 1e-12) and
                           a[i].velocity_scale == b[i].velocity_scale;
            }
    check_local(same, "restep of a TimeStepping's own steps gives them back (to round-off)");
    TimestepOptions bf;
    bf.method = TimestepMethod::BullFrog;
    TimeStepping bullfrog(bf, eds_H, eds_force, history(1.0));
    bool threw = false;
    try {
        bullfrog.restep(bullfrog.steps(amin, amax, 4));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check_local(threw, "restep is refused for BullFrog");
}

void test_schedules() {
    // Nonlinear D(a) ensures growth spacing tests an actual inverse, rather than D=a.
    GrowthHistory growth([](double a) { return std::pow(a, 0.8); }, [](double) { return 0.8; });
    bool correct = true, synchronized = true;
    for (auto spacing : {ScaleFactorSpacing::Linear,
                         ScaleFactorSpacing::Logarithmic,
                         ScaleFactorSpacing::Powerlaw,
                         ScaleFactorSpacing::GrowthFactor}) {
        TimestepOptions options;
        options.method = TimestepMethod::FastPM;
        options.spacing = spacing;
        options.spacing_power = -0.4;
        TimeStepping stepping(options, eds_H, [](double a) { return 0.8 * 1.3 * a; }, growth);
        for (int i = 0; i <= 10; i++) {
            const double t = i / 10.0, lo = 0.05, hi = 1;
            const double expected =
                spacing == ScaleFactorSpacing::Linear ?
                    lo + (hi - lo) * t :
                spacing == ScaleFactorSpacing::Logarithmic ?
                    lo * std::pow(hi / lo, t) :
                spacing == ScaleFactorSpacing::Powerlaw ?
                    std::pow(std::pow(lo, -0.4) + (std::pow(hi, -0.4) - std::pow(lo, -0.4)) * t, -2.5) :
                    std::pow(std::pow(lo, 0.8) + (std::pow(hi, 0.8) - std::pow(lo, 0.8)) * t, 1.25);
            correct = correct && near(stepping.scale_factor_of_step(lo, hi, 10, i), expected, 1e-13);
        }
        const auto steps = stepping.steps(0.05, 1, 10);
        synchronized = synchronized && steps.size() == 11 && steps.front().apos == 0.05 && steps.front().avel == 0.05 &&
                       steps.back().apos_new == 1 && steps.back().avel_new == 1;
        for (size_t i = 1; i < steps.size(); i++)
            synchronized =
                synchronized && steps[i].apos == steps[i - 1].apos_new && steps[i].avel == steps[i - 1].avel_new;
    }
    check_local(correct, "all spacing policies match independent schedules with nonlinear growth and negative power");
    check_local(synchronized, "staggered schedules are continuous and synchronize at the output epoch");
}
void test_cola_force_options() {
    auto cosmo = make_test_cosmology();
    auto grav = std::make_shared<TestGravity>(cosmo);
    FML::GRID::FFTWGrid<3> density(16);
    density.fill_real_grid([](const std::array<double, 3> & x) { return std::cos(6 * M_PI * x[0]); });
    density.fftw_r2c();
    const FML::NBODY::ForceOptions options{FML::NBODY::DISCRETE_GREENS_FUNCTION_FROM_FOUR_POINT,
                                           FML::NBODY::DISCRETE_FOUR_POINT_GRADIENT,
                                           true};
    bool matches = true;
    for (bool finite_difference : {false, true}) {
        grav->use_finite_difference(finite_difference);
        std::array<FML::GRID::FFTWGrid<3>, 3> actual, expected;
        grav->compute_force(0.5, 1, density, "CIC", actual, options);
        FML::NBODY::compute_force_from_density_fourier<3>(density, expected, "CIC", 1.5 * 0.3 * 0.5, options);
        for (int d = 0; d < 3; d++)
            for (auto i : actual[d].get_real_range())
                matches = matches && near(actual[d].get_real_from_index(i), expected[d].get_real_from_index(i), 1e-12);
    }
    check_local(matches,
                "SimulationCode forwards explicit options through Fourier and finite-difference gravity paths");
}
void test_cola_adapter() {
    auto cosmo = make_test_cosmology();
    auto grav = std::make_shared<TestGravity>(cosmo);
    bool same = true, ownership = true;
    for (auto method :
         {TimestepMethod::Quinn, TimestepMethod::Tassev, TimestepMethod::FastPM, TimestepMethod::BullFrog}) {
        SIMCODE::TimestepSettings settings{};
        settings.method = method;
        settings.spacing = ScaleFactorSpacing::GrowthFactor;
        settings.cola_nLPT = -2.5;
        SIMCODE::TimeStepping<3> adapter(settings, cosmo, grav);
        TimeStepping shared(
            {method, settings.spacing, settings.cola_nLPT, settings.spacing_power},
            [cosmo](double a) { return cosmo->HoverH0_of_a(a); },
            [cosmo, grav](double a) { return 1.5 * cosmo->get_OmegaM() * grav->GeffOverG(a) * a; },
            SIMCODE::large_scale_growth_history<3>(grav));
        const auto a = adapter.steps(0.05, 1.0, 5), b = shared.steps(0.05, 1.0, 5);
        same = same && a.size() == b.size();
        for (size_t i = 0; i < a.size(); i++)
            same = same && a[i].apos == b[i].apos && a[i].apos_new == b[i].apos_new && a[i].avel == b[i].avel &&
                   a[i].avel_new == b[i].avel_new && a[i].delta_time_drift == b[i].delta_time_drift &&
                   a[i].delta_time_kick == b[i].delta_time_kick && a[i].velocity_scale == b[i].velocity_scale;
        ownership = ownership && !adapter.steps(0.05, 1.0, 1).empty();
    }
    check_local(same, "SimulationCode adapts every policy to the shared coefficient generator exactly");
    auto owned = SIMCODE::large_scale_growth_history<3>(grav);
    const double expected = grav->get_D_1LPT(0.5);
    grav.reset();
    cosmo.reset();
    ownership = ownership && owned.linear(0.5) == expected;
    check_local(ownership, "SimulationCode growth adapter retains its model ownership");
}
template <class Function>
void expect_error(Function fn, const std::string & label) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception &) {
        caught = true;
    }
    check_local(caught, label);
}
void test_contracts() {
    TimeStepping standard({}, eds_H, eds_force);
    check_local(standard.steps(0.05, 1.0, 0).empty() && standard.deltatime_KDK(0.05, 1.0, 0).first.empty(),
                "zero requested steps produce no updates");
    expect_error([&] { standard.steps(0, 1, 5); }, "nonpositive scale factors are rejected");
    expect_error([&] { standard.steps(1, 0.05, 5); }, "reversed integration intervals are rejected");
    expect_error([&] { standard.steps(0.05, 1, -1); }, "negative step counts are rejected");
    expect_error([&] { standard.steps(0.05, 0.05, 1); }, "zero-width advancing intervals are rejected");
    expect_error([&] { TimeStepping({}, {}, eds_force); }, "missing background callbacks are rejected");
    TimestepOptions options;
    options.method = TimestepMethod::FastPM;
    expect_error([&] { TimeStepping(options, eds_H, eds_force).steps(0.05, 1, 1); },
                 "growth-aware policies require a growth history");
    options.method = TimestepMethod::Tassev;
    options.cola_nLPT = 0;
    expect_error([&] { TimeStepping(options, eds_H, eds_force); }, "zero Tassev exponent is rejected");
    options = {};
    options.spacing = ScaleFactorSpacing::Powerlaw;
    options.spacing_power = 0;
    expect_error([&] { TimeStepping(options, eds_H, eds_force); }, "zero powerlaw exponent is rejected");
    options.spacing = ScaleFactorSpacing::GrowthFactor;
    GrowthHistory decreasing([](double a) { return 1 / a; }, [](double) { return -1; });
    expect_error([&] { TimeStepping(options, eds_H, eds_force, decreasing).steps(0.05, 1, 3); },
                 "growth-time spacing rejects decreasing endpoint growth");
    options = {};
    expect_error([&] { TimeStepping(options, [](double) { return 0.0; }, eds_force).steps(0.05, 1, 2); },
                 "nonpositive expansion rates are rejected");
    options.method = TimestepMethod::FastPM;
    expect_error([&] { TimeStepping(options, eds_H, [](double) { return 0.0; }, eds_growth()).steps(0.05, 1, 2); },
                 "zero force normalization is rejected by growth-aware coefficients");
    GrowthHistory zero_momentum([](double a) { return a; }, [](double) { return 0; });
    expect_error([&] { TimeStepping(options, eds_H, eds_force, zero_momentum).steps(0.05, 1, 2); },
                 "zero momentum growth is rejected by growth-aware coefficients");
    GrowthHistory invalid([](double) { return std::numeric_limits<double>::quiet_NaN(); });
    expect_error([&] { invalid.linear(0.5); }, "nonfinite growth is rejected");
    GrowthHistory stationary([](double) { return 1; },
                             [](double) { return 0; },
                             [](double) { return -1; },
                             [](double) { return 0; });
    expect_error([&] { stationary.second_order_slope(0.5); }, "undefined second-order slopes are rejected");
}
int main() {
    test_analytic_coefficients();
    test_growing_modes();
    test_friction();
    test_species_friction();
    test_schedules();
    test_cola_adapter();
    test_cola_force_options();
    test_contracts();
    return FML::TESTS::summary("time stepping");
}
