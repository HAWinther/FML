#include <FML/WaveDarkMatter/SchrodingerPoisson.h>
#ifdef WAVE_PENCIL
#include <FML/PencilDomain/CPencilGrid.h>
using TestGrid = FML::GRID::CPencilGrid<3>;
#else
using TestGrid = FML::GRID::CFFTWGrid<3>;
#endif
#include <FML/tests/TestHelpers.h>
#include "SolitonReference.h"
#include <iomanip>
#include <iostream>
using namespace FML::WAVEDM;
using TestSolver = SolverT<TestGrid>;
using FML::TESTS::check;
struct Measured {
    double density_error{}, residual{}, tail{}, energy_error{}, mass{};
};
static Measured soliton(int n, double L = 12, int evolved_steps = 100) {
    TestSolver s(n, L);
    s.initialize([L](auto x) {
        double r2 = 0;
        for (auto y : x)
            r2 += (y - L * .5) * (y - L * .5);
        return Complex(Real(wave_reference::profile(std::sqrt(r2))), 0);
    });
    const auto initial = s.diagnostics();
    std::vector<Complex> original(s.wave().get_real_grid(), s.wave().get_real_grid() + s.local_size());
    const auto U = s.potential();
    auto lap = detail::make_grid<TestGrid>(n);
    detail::copy_values(lap, s.wave());
    lap.fftw_forward();
    for (size_t i = 0; i < size_t(lap.get_ntot_fourier()); ++i) {
        const auto mode = lap.get_fourier_integer_wavevector_from_index(FML::IndexIntType(i));
        double k2 = 0;
        for (auto k : mode)
            k2 += std::pow(2 * std::acos(-1.) * k / L, 2);
        lap.get_fourier_grid()[i] *= Real(-k2);
    }
    lap.fftw_backward();
    double norm = 0, mu = 0;
    for (size_t i = 0; i < s.local_size(); ++i) {
        const auto H = Real(-.5) * lap.get_real_grid()[i] + Real(U[i]) * original[i];
        norm += double(std::norm(original[i]));
        mu += double((std::conj(original[i]) * H).real());
    }
    FML::SumOverTasks(&norm);
    FML::SumOverTasks(&mu);
    mu /= norm;
    double residual = 0;
    for (size_t i = 0; i < s.local_size(); ++i) {
        const auto H = Real(-.5) * lap.get_real_grid()[i] + Real(U[i]) * original[i];
        residual += double(std::norm(H - Real(mu) * original[i]));
    }
    FML::SumOverTasks(&residual);
    for (int step = 0; step < evolved_steps; ++step)
        s.step(.005);
    const auto final = s.diagnostics();
    double error = 0, densitynorm = 0;
    for (size_t i = 0; i < original.size(); ++i) {
        const double rho = double(std::norm(original[i])), now = double(std::norm(s.wave().get_real_grid()[i]));
        error += (rho - now) * (rho - now);
        densitynorm += rho * rho;
    }
    FML::SumOverTasks(&error);
    FML::SumOverTasks(&densitynorm);
    const auto result = Measured{std::sqrt(error / densitynorm),
                                 std::sqrt(residual / norm) / std::abs(wave_reference::soliton_mu),
                                 initial.spectral_tail_fraction,
                                 std::abs(final.energy() / initial.energy() - 1),
                                 initial.mass};
    if (FML::ThisTask == 0)
        std::cout << "SOLITON " << n << " mass " << result.mass << " density_drift " << result.density_error
                  << " eigen_residual " << result.residual << " tail " << result.tail << " energy_drift "
                  << result.energy_error << '\n';
    check(std::abs(final.mass / initial.mass - 1) < (sizeof(Real) == 4 ? 3e-5 : 3e-12),
          "soliton mass conserved at mesh " + std::to_string(n));
    return result;
}
int main() {
    std::cout << std::setprecision(17);
    const auto coarse = soliton(32), fine = soliton(64);
    check(std::abs(fine.mass / wave_reference::soliton_mass - 1) < 3e-4,
          "independent radial soliton mass reproduced on Cartesian grid");
    check(fine.residual < .015 && std::abs(fine.residual - coarse.residual) < .001,
          "isolated radial soliton has a bounded, resolution-stable periodic eigen-residual");
    check(fine.density_error < .006 && std::abs(fine.density_error - coarse.density_error) < .001,
          "resolved soliton retains stationary density over the tested interval");
    check(fine.tail < coarse.tail && fine.tail < 1e-5, "soliton spectral-tail/de-Broglie resolution improves");
    const auto wider = soliton(80, 16, 0);
    check(wider.residual < fine.residual * .6, "isolated-profile mismatch decreases in a larger periodic volume");
    check(fine.energy_error < (sizeof(Real) == 4 ? 2e-4 : 1e-5), "static soliton energy drift remains bounded");
    return FML::TESTS::summary("WaveDM soliton");
}
