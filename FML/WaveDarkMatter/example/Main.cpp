#include <FML/WaveDarkMatter/SchrodingerPoisson.h>
#include <FML/WaveDarkMatter/Checkpoint.h>
#include <chrono>
#include <iomanip>
#include <iostream>
using namespace FML::WAVEDM;
int main(int argc, char ** argv) {
    const double pi = std::acos(-1.);
    auto field = [](auto x) {
        const double amp = 1 + .1 * std::cos(x[0]) + .05 * std::cos(2 * x[1]), phase = .08 * std::sin(x[2]);
        return Complex(Real(amp * std::cos(phase)), Real(amp * std::sin(phase)));
    };
    Solver box(24, 2 * pi, {.7, .4});
    box.initialize(field);
    const auto initial = box.diagnostics();
    const auto before = box.fft_count();
    auto clock = std::chrono::steady_clock::now();
    for (int i = 0; i < 40; ++i)
        box.step(.005);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
    FML::MaxOverTasks(&seconds);
    const auto step_ffts = box.fft_count() - before;
    const auto bytes = box.memory().total();
    const auto final = box.diagnostics();
    Solver cosmology(24, 2 * pi, {.7, .4}, 0, .5);
    cosmology.initialize(field);
    EnergyBalance energy;
    const auto start = cosmology.diagnostics();
    energy.observe(.5, start.kinetic, start.potential);
    double residual = 0;
    for (int i = 1; i <= 20; ++i) {
        cosmology.step_scale_factor(.5 + .05 * i / 20, [](double a) { return 2 * std::pow(a, -1.5); });
        auto d = cosmology.diagnostics();
        residual = energy.observe(cosmology.scale_factor(), d.kinetic, d.potential);
    }
    if (argc > 1) {
        save_checkpoint(cosmology, argv[1]);
        auto restored = load_checkpoint(argv[1]);
        (void)restored;
    }
    const auto physical = Physics::from_si(1e-22, FML::UTILS::SI::kpc, FML::UTILS::SI::Gyr, 1e7 * FML::UTILS::SI::Msun);
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(12) << "Analytic toy in code units; not a calibrated cosmological run\n"
                  << "static relative mass drift " << final.mass / initial.mass - 1 << " energy drift "
                  << final.energy() / initial.energy() - 1 << '\n'
                  << "spectral near-Nyquist fraction " << final.spectral_tail_fraction << '\n'
                  << "40 steps: max-rank seconds " << seconds << " local complex FFTs " << step_ffts
                  << " local accounted bytes " << bytes << '\n'
                  << "expanding a=0.5->0.55 energy-balance residual / |E0| " << residual / std::abs(start.energy())
                  << '\n'
                  << "example physical conversion m=1e-22 eV, kpc/Gyr/1e7 Msun: eta=" << physical.eta
                  << " G=" << physical.G << '\n';
}
