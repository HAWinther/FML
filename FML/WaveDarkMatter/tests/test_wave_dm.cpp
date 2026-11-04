#include <FML/WaveDarkMatter/SchrodingerPoisson.h>
#include <FML/WaveDarkMatter/Checkpoint.h>
#include <FML/tests/TestHelpers.h>
#include "EvolutionReference.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
using namespace FML::WAVEDM;
using FML::TESTS::check;
static constexpr double pi = 3.14159265358979323846;
static double tolerance() { return sizeof(Real) == 4 ? 3e-5 : 3e-12; }
static Complex initial(std::array<double, 3> x) {
    const double amp = 1 + .1 * std::cos(x[0]) + .05 * std::cos(2 * x[1]), phase = .08 * std::sin(x[2]);
    return Complex(Real(amp * std::cos(phase)), Real(amp * std::sin(phase)));
}
static std::vector<Complex> values(const Solver & s) { return {s.wave().data(), s.wave().data() + s.wave().size()}; }
static double difference(const Solver & s, const std::vector<Complex> & ref) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        error += double(std::norm(s.wave().data()[i] - ref[i]));
        norm += double(std::norm(ref[i]));
    }
    FML::SumOverTasks(&error);
    FML::SumOverTasks(&norm);
    return std::sqrt(error / std::max(norm, 1e-300));
}
static void remove_checkpoint(const std::string & p) {
    std::remove(io_detail::filename(p, FML::ThisTask).c_str());
    if (FML::ThisTask == 0)
        std::remove((p + ".manifest").c_str());
}
static void fft_test(int mesh) {
    ComplexGrid g(mesh);
    std::vector<Complex> original(g.size());
    auto sample = [](int x, int y, int z) { return Complex(Real(.13 + x + .07 * y * z), Real(.2 + y - .11 * x * z)); };
    for (size_t i = 0; i < g.size(); ++i) {
        auto x = g.coordinate(i);
        original[i] = g.data()[i] = sample(x[0], x[1], x[2]);
    }
    g.forward();
    double error = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        const auto k = g.mode(i);
        std::complex<double> expected{};
        for (int x = 0; x < mesh; ++x)
            for (int y = 0; y < mesh; ++y)
                for (int z = 0; z < mesh; ++z) {
                    const double phase = -2 * pi * (k[0] * x + k[1] * y + k[2] * z) / mesh;
                    const auto v = sample(x, y, z);
                    expected += std::complex<double>(v.real(), v.imag()) * std::polar(1., phase);
                }
        expected /= double(mesh * mesh * mesh);
        const auto actual = g.data()[i];
        error = std::max(error, std::abs(std::complex<double>(actual.real(), actual.imag()) - expected));
    }
    FML::MaxOverTasks(&error);
    check(error < tolerance() * 100, "full complex FFT agrees with independent DFT, mesh " + std::to_string(mesh));
    g.inverse();
    error = 0;
    for (size_t i = 0; i < g.size(); ++i)
        error = std::max(error, double(std::abs(g.data()[i] - original[i])));
    FML::MaxOverTasks(&error);
    check(error < tolerance() * 100, "complex FFT round trip including empty slabs, mesh " + std::to_string(mesh));
}
int main(int argc, char ** argv) {
    std::cout << std::setprecision(17);
    if (argc == 3 && std::string(argv[1]) == "--write") {
        Solver s(10, 2 * pi, {.7, .4}, 0, .7);
        s.initialize(initial);
        std::vector<Real> ext(s.wave().size());
        for (size_t i = 0; i < ext.size(); ++i)
            ext[i] = Real(.1 * (1 + std::cos(s.position(i)[0])));
        s.set_external_density(ext);
        s.step(.01);
        save_checkpoint(s, argv[2]);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--read") {
        auto s = load_checkpoint(argv[2]);
        Solver expected(10, 2 * pi, {.7, .4}, 0, .7);
        expected.initialize(initial);
        std::vector<Real> ext(expected.wave().size());
        for (size_t i = 0; i < ext.size(); ++i)
            ext[i] = Real(.1 * (1 + std::cos(expected.position(i)[0])));
        expected.set_external_density(ext);
        expected.step(.01);
        check(difference(*s, values(expected)) < tolerance() * 10 && s->has_external_density() && s->steps() == 1 &&
                  s->time() == .01 && s->scale_factor() == .7,
              "changed-rank checkpoint retains wave/source/epoch");
        s->step(.01);
        expected.step(.01);
        check(difference(*s, values(expected)) < tolerance() * 20, "changed-rank restart continues same trajectory");
        return FML::TESTS::summary("WaveDM cross-rank checkpoint");
    }
    fft_test(2);
    fft_test(3);
    fft_test(4);
    const auto p = Physics::from_si(1e-22, FML::UTILS::SI::kpc, FML::UTILS::SI::Gyr, 1e7 * FML::UTILS::SI::Msun);
    const auto p2 = Physics::from_si(2e-22, FML::UTILS::SI::kpc, FML::UTILS::SI::Gyr, 1e7 * FML::UTILS::SI::Msun);
    check(p.eta > 0 && p.G > 0 && p.eta == 2 * p2.eta && p.G == p2.G, "axion mass and SI/code-unit scaling");
    check(FML::TESTS::throws([] { Physics::from_si(1e-22, 1e120, 1, 1); }),
          "unit conversion rejects gravitational underflow instead of silently disabling gravity");
    Solver plane(12, 2 * pi, {.7, 0});
    auto plane_initial = [](auto x) {
        const double phase = x[0] - 2 * x[1] + 3 * x[2];
        return Complex(Real(std::cos(phase)), Real(std::sin(phase)));
    };
    plane.initialize(plane_initial);
    const auto original = values(plane);
    plane.step(.2);
    double error = 0;
    const auto factor = std::polar(1., -.7 * 14 * .2 / 2);
    for (size_t i = 0; i < original.size(); ++i) {
        auto z = original[i];
        error = std::max(error,
                         std::abs(std::complex<double>(plane.wave().data()[i].real(), plane.wave().data()[i].imag()) -
                                  std::complex<double>(z.real(), z.imag()) * factor));
    }
    FML::MaxOverTasks(&error);
    check(error < tolerance() * 10, "free-wave dispersion and independent positive/negative modes");
    const auto d = plane.diagnostics();
    check(std::abs(d.mass / std::pow(2 * pi, 3) - 1) < tolerance() * 10 &&
              std::abs(d.kinetic / (.5 * .7 * .7 * 14 * d.mass) - 1) < tolerance() * 10,
          "mass and Parseval kinetic energy normalization");
    check(d.spectral_tail_fraction < tolerance(), "resolved wave has negligible near-Nyquist mass");
    plane.step(-.2);
    check(difference(plane, original) < tolerance() * 10, "free time reversal");
    Solver interference(12, 2 * pi, {1, 0});
    interference.initialize([](auto x) {
        return Complex(Real(std::cos(x[0]) + .4 * std::cos(2 * x[0])), Real(std::sin(x[0]) - .4 * std::sin(2 * x[0])));
    });
    interference.step(.17);
    error = 0;
    for (size_t i = 0; i < interference.wave().size(); ++i) {
        const auto x = interference.position(i);
        const double exact = 1.16 + .8 * std::cos(3 * x[0] + 1.5 * .17);
        error = std::max(error, std::abs(double(std::norm(interference.wave().data()[i])) - exact));
    }
    FML::MaxOverTasks(&error);
    check(error < tolerance() * 10, "two-wave interference has analytic beat phase/density");
    Solver poisson(12, 2 * pi, {1, .2});
    poisson.initialize_density_phase([](auto x) { return 1 + .1 * std::cos(x[0]); }, [](auto) { return 0.; });
    const auto & U = poisson.potential();
    error = 0;
    double mean = 0;
    for (size_t i = 0; i < U.size(); ++i) {
        error = std::max(error, std::abs(U[i] + 4 * pi * .2 * .1 * std::cos(poisson.position(i)[0])));
        mean += U[i];
    }
    FML::MaxOverTasks(&error);
    FML::SumOverTasks(&mean);
    check(error < tolerance() * 10 && std::abs(mean) < tolerance() * 100,
          "periodic Poisson sign, source normalization and zero potential mean");
    std::vector<Real> external(poisson.wave().size());
    for (size_t i = 0; i < external.size(); ++i)
        external[i] = Real(.5 + .2 * std::cos(2 * poisson.position(i)[0]));
    poisson.set_external_density(external);
    auto acceleration = poisson.gravity_acceleration();
    error = 0;
    for (size_t i = 0; i < external.size(); ++i) {
        const double x = poisson.position(i)[0];
        error =
            std::max(error,
                     std::abs(double(acceleration[0][i]) + 4 * pi * .2 * (.1 * std::sin(x) + .1 * std::sin(2 * x))));
        error = std::max(error, std::abs(double(acceleration[1][i])) + std::abs(double(acceleration[2][i])));
    }
    FML::MaxOverTasks(&error);
    check(error < tolerance() * 30, "external density and comoving particle-force hook");
    const auto pd = poisson.diagnostics();
    check(std::abs(pd.potential / (-pi * .2 * std::pow(2 * pi, 3) * (.01 + .04 / 4)) - 1) < tolerance() * 30,
          "total grid gravitational energy includes external/self/cross terms");
    poisson.initialize(initial);
    check(!poisson.wave().spectral(), "reinitialization after force diagnostics preserves position-space state");
    Solver exact(4, 2 * pi, {.7, .4});
    exact.initialize(initial);
    for (int j = 0; j < 7; ++j)
        exact.step(.01);
    error = 0;
    for (size_t i = 0; i < exact.wave().size(); ++i) {
        const auto ix = exact.wave().coordinate(i);
        const auto ref = wave_reference::evolved[(ix[0] * 4 + ix[1]) * 4 + ix[2]];
        error = std::max(error,
                         std::abs(std::complex<double>(exact.wave().data()[i].real(), exact.wave().data()[i].imag()) -
                                  std::complex<double>(ref[0], ref[1])));
    }
    FML::MaxOverTasks(&error);
    check(error < 3e-5, "nonlinear evolution agrees with independent adaptive DOP853 reference");
    Solver reference(12, 2 * pi, {.7, .4});
    reference.initialize(initial);
    for (int j = 0; j < 160; ++j)
        reference.step(.0025);
    const auto truth = values(reference);
    double errors[3]{}, energy_errors[3]{};
    for (int resolution = 0; resolution < 3; ++resolution) {
        Solver s(12, 2 * pi, {.7, .4});
        s.initialize(initial);
        const auto start = s.diagnostics();
        const int steps = 4 * (1 << resolution);
        const double dt = .4 / steps;
        for (int j = 0; j < steps; ++j)
            s.step(dt);
        errors[resolution] = difference(s, truth);
        const auto end = s.diagnostics();
        energy_errors[resolution] = std::abs(end.energy() - start.energy());
        check(std::abs(end.mass / start.mass - 1) < tolerance() * 10,
              "nonlinear split-step norm conservation at timestep " + std::to_string(dt));
    }
    if (FML::ThisTask == 0)
        std::cout << "TIME_ERRORS " << errors[0] << ' ' << errors[1] << ' ' << errors[2] << " ENERGY_ERRORS "
                  << energy_errors[0] << ' ' << energy_errors[1] << ' ' << energy_errors[2] << '\n';
    check(errors[1] < errors[0] / 2.5 && errors[2] < errors[1] / 2.5, "second-order nonlinear timestep convergence");
    check(energy_errors[1] < energy_errors[0] / 2 && energy_errors[2] < energy_errors[1] / 2,
          "static energy drift decreases under timestep refinement");
    Solver reversible(12, 2 * pi, {.7, .4});
    reversible.initialize(initial);
    const auto reversible_start = values(reversible);
    for (int i = 0; i < 10; ++i)
        reversible.step(.01);
    for (int i = 0; i < 10; ++i)
        reversible.step(-.01);
    check(difference(reversible, reversible_start) < tolerance() * 100,
          "self-gravitating symmetric-step time reversal");
    const auto snapshot = values(reversible);
    const double time = reversible.time();
    const auto steps = reversible.steps();
    check(FML::TESTS::throws([&] { reversible.step(std::numeric_limits<double>::max()); }) &&
              difference(reversible, snapshot) == 0 && reversible.time() == time && reversible.steps() == steps,
          "phase overflow rejects transaction without changing live state");
    check(FML::TESTS::throws([&] { reversible.initialize([](auto) { return Complex(NAN, 0); }); }) &&
              difference(reversible, snapshot) == 0,
          "invalid initializer leaves wave unchanged");
    const auto EdS = [](double a) { return 2 * std::pow(a, -1.5); };
    const double a0 = .5, a1 = .6;
    const auto factors = expansion_factors(a0, a1, EdS);
    check(std::abs(factors.dt - (std::pow(a1, 1.5) - std::pow(a0, 1.5)) / 3) < 1e-14 &&
              std::abs(factors.kinetic - (1 / std::sqrt(a0) - 1 / std::sqrt(a1))) < 1e-14 &&
              std::abs(factors.potential - (std::sqrt(a1) - std::sqrt(a0))) < 1e-14,
          "EdS cosmic-time integrals match analytic results");
    Solver expanding(12, 2 * pi, {.7, 0}, 0, a0);
    expanding.initialize(plane_initial);
    const auto begin = values(expanding);
    expanding.step_scale_factor(a1, EdS);
    error = 0;
    const auto analytic_phase = std::polar(1., -.7 * 14 * factors.kinetic / 2);
    for (size_t i = 0; i < begin.size(); ++i)
        error = std::max(
            error,
            std::abs(std::complex<double>(expanding.wave().data()[i].real(), expanding.wave().data()[i].imag()) -
                     std::complex<double>(begin[i].real(), begin[i].imag()) * analytic_phase));
    FML::MaxOverTasks(&error);
    check(error < tolerance() * 20 && expanding.scale_factor() == a1, "expanding free wave uses integral dt/a^2");
    expanding.step_scale_factor(a0, EdS);
    check(difference(expanding, begin) < tolerance() * 30, "expanding time-factor reversal");
    double balances[2]{};
    for (int refinement = 0; refinement < 2; ++refinement) {
        Solver s(8, 2 * pi, {.7, .4}, 0, .5);
        s.initialize(initial);
        EnergyBalance balance;
        auto first = s.diagnostics();
        balance.observe(.5, first.kinetic, first.potential);
        const int count = (sizeof(Real) == 4 ? 2 : 20) * (1 << refinement);
        for (int j = 1; j <= count; ++j) {
            s.step_scale_factor(.5 + .1 * j / count, EdS);
            auto diag = s.diagnostics();
            balances[refinement] =
                std::abs(balance.observe(s.scale_factor(), diag.kinetic, diag.potential)) / std::abs(first.energy());
        }
    }
    if (FML::ThisTask == 0)
        std::cout << "COSMIC_BALANCE " << balances[0] << ' ' << balances[1] << '\n';
    check(balances[1] < balances[0] / 2 && balances[1] < 2e-4, "cosmological Layzer-Irvine residual converges");
    check(FML::TESTS::throws([&] { expansion_factors(.5, .6, [](double) { return -1.; }); }),
          "nonpositive H rejected collectively");
    const std::string prefix = "wave_test_" + std::to_string(FML::NTasks);
    remove_checkpoint(prefix);
    save_checkpoint(poisson, prefix);
    auto restored = load_checkpoint(prefix);
    check(difference(*restored, values(poisson)) == 0 && restored->external_density() == poisson.external_density() &&
              restored->has_external_density(),
          "checksummed checkpoint roundtrip retains complex field and external source");
    poisson.step(.01);
    restored->step(.01);
    check(difference(*restored, values(poisson)) == 0, "checkpoint restart is exact at the same precision/layout");
    check(FML::TESTS::throws([&] { save_checkpoint(poisson, prefix); }), "checkpoint overwrite rejected");
    if (FML::ThisTask == 0) {
        std::fstream bad(io_detail::filename(prefix, 0), std::ios::binary | std::ios::in | std::ios::out);
        std::string line;
        std::getline(bad, line);
        const auto at = bad.tellg();
        bad.seekp(at);
        bad.put('X');
    }
    check(FML::TESTS::throws([&] { load_checkpoint(prefix); }), "checkpoint payload corruption rejected collectively");
    remove_checkpoint(prefix);
    const auto memory = poisson.memory();
    check(memory.complex_buffers >= 3 * poisson.wave().size() * sizeof(Complex) && memory.total() > 0 &&
              poisson.fft_count() > 0,
          "memory and FFT-work counters reported");
    check(poisson.suggested_timestep() > 0, "conservative phase-based timestep suggestion");
    if (FML::NTasks > 1)
        check(FML::TESTS::throws([&] { reversible.step(FML::ThisTask ? .01 : .02); }),
              "rank-inconsistent evolution parameters rejected");
    return FML::TESTS::summary("WaveDM");
}
