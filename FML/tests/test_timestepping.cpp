#include <FML/COLASolver/src/TimeStepping.h>
#include <FML/LPT/GrowthHistory.h>
#include <FML/NBody/TimeStepping.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
using namespace FML::NBODY;
using FML::COSMOLOGY::GrowthHistory;
int failures = 0;
void check(bool ok, const std::string & label) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << '\n';
    if (!ok)
        failures++;
}
bool near(double x, double y, double tolerance = 2e-9) {
    return std::isfinite(x) && std::isfinite(y) &&
           std::abs(x - y) <= tolerance * std::max({1.0, std::abs(x), std::abs(y)});
}
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
        check(correct,
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
                    linear = linear && near(x1, growth.linear(1.0)) && near(v1, growth.linear_momentum(1.0, 1.0));
                    if (method == TimestepMethod::BullFrog)
                        second_order = second_order && near(x2, growth.second_order(1.0)) &&
                                       near(v2, growth.linear_momentum(1.0, 1.0) * growth.second_order_slope(1.0));
                    normalization_independent = normalization_independent && near(x1 / normalization, 1.0);
                }
            }
        }
        check(linear,
              method == TimestepMethod::FastPM ? "FastPM follows the linear growing mode at coarse steps" :
                                                 "BullFrog follows the linear growing mode at coarse steps");
        if (method == TimestepMethod::BullFrog)
            check(second_order, "BullFrog follows the EdS second-order displacement and velocity");
        check(normalization_independent, "growth normalization is consistent across all timestep spacings");
    }
}
class TestCosmology : public Cosmology {
  public:
    TestCosmology() { OmegaM = 0.3; }
    double HoverH0_of_a(double a) const override { return std::sqrt(0.3 / (a * a * a) + 0.7); }
    double dlogHdloga_of_a(double a) const override { return -1.5 * 0.3 / (0.3 + 0.7 * a * a * a); }
};
class TestGravity : public GravityModel<3> {
  public:
    explicit TestGravity(std::shared_ptr<Cosmology> cosmo) : GravityModel<3>(cosmo, "time-step test") {
        std::vector<double> x(600), d1(600), d2(600);
        scaledependent_growth = false;
        for (size_t i = 0; i < x.size(); i++) {
            x[i] = std::log(0.01) + (std::log(1.1) - std::log(0.01)) * i / (x.size() - 1);
            const double a = std::exp(x[i]);
            d1[i] = a / (1 + 0.2 * a);
            d2[i] = -3.0 / 7.0 * d1[i] * d1[i] * (1 + 0.01 * a);
        }
        D_1LPT_of_loga.create(x, d1);
        D_2LPT_of_loga.create(x, d2);
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
    check(correct, "all spacing policies match independent schedules with nonlinear growth and negative power");
    check(synchronized, "staggered schedules are continuous and synchronize at the output epoch");
}
void test_cola_force_options() {
    auto cosmo = std::make_shared<TestCosmology>();
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
    check(matches, "COLASolver forwards explicit options through Fourier and finite-difference gravity paths");
}
void test_cola_adapter() {
    auto cosmo = std::make_shared<TestCosmology>();
    auto grav = std::make_shared<TestGravity>(cosmo);
    bool same = true, ownership = true;
    for (auto method :
         {TimestepMethod::Quinn, TimestepMethod::Tassev, TimestepMethod::FastPM, TimestepMethod::BullFrog}) {
        COLASOLVER::TimestepSettings settings{};
        settings.method = method;
        settings.spacing = ScaleFactorSpacing::GrowthFactor;
        settings.cola_nLPT = -2.5;
        COLASOLVER::TimeStepping<3> adapter(settings, cosmo, grav);
        TimeStepping shared(
            {method, settings.spacing, settings.cola_nLPT, settings.spacing_power},
            [cosmo](double a) { return cosmo->HoverH0_of_a(a); },
            [cosmo, grav](double a) { return 1.5 * cosmo->get_OmegaM() * grav->GeffOverG(a) * a; },
            COLASOLVER::large_scale_growth_history<3>(grav));
        const auto a = adapter.steps(0.05, 1.0, 5), b = shared.steps(0.05, 1.0, 5);
        same = same && a.size() == b.size();
        for (size_t i = 0; i < a.size(); i++)
            same = same && a[i].apos == b[i].apos && a[i].apos_new == b[i].apos_new && a[i].avel == b[i].avel &&
                   a[i].avel_new == b[i].avel_new && a[i].delta_time_drift == b[i].delta_time_drift &&
                   a[i].delta_time_kick == b[i].delta_time_kick && a[i].velocity_scale == b[i].velocity_scale;
        ownership = ownership && !adapter.steps(0.05, 1.0, 1).empty();
    }
    check(same, "COLASolver adapts every policy to the shared coefficient generator exactly");
    auto owned = COLASOLVER::large_scale_growth_history<3>(grav);
    const double expected = grav->get_D_1LPT(0.5);
    grav.reset();
    cosmo.reset();
    ownership = ownership && owned.linear(0.5) == expected;
    check(ownership, "COLASolver growth adapter retains its model ownership");
}
template <class Function>
void expect_error(Function fn, const std::string & label) {
    bool caught = false;
    try {
        fn();
    } catch (const std::exception &) {
        caught = true;
    }
    check(caught, label);
}
void test_contracts() {
    TimeStepping standard({}, eds_H, eds_force);
    check(standard.steps(0.05, 1.0, 0).empty() && standard.deltatime_KDK(0.05, 1.0, 0).first.empty(),
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
    test_schedules();
    test_cola_adapter();
    test_cola_force_options();
    test_contracts();
    FML::SumOverTasks(&failures);
    return failures ? 1 : 0;
}
