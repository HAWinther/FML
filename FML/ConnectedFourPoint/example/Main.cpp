#include <FML/ConnectedFourPoint/Fourier.h>
#include <iostream>
int main() {
    using namespace FML::FOURPOINT;
    FourierGrid density(16);
    density.fill_real_grid(
        [](std::array<double, 3> & x) { return std::cos(2 * M_PI * x[0]) + .5 * std::cos(4 * M_PI * x[1]); });
    density.fftw_r2c();
    ShellOptions shells{100, {0, 1.1 * 2 * M_PI / 100, 2.1 * 2 * M_PI / 100}};
    const auto connected = connected_shell_four_point(density, shells, {{0, 0, 0, 0}, {0, 0, 1, 1}});
    if (!FML::ThisTask)
        for (const auto & r : connected)
            std::cout << "bins " << r.bins[0] << r.bins[1] << r.bins[2] << r.bins[3]
                      << " quartets=" << r.closing_quartets << " raw=" << r.raw << " Wick=" << r.disconnected
                      << " connected=" << r.connected << '\n';
    // Toy independent-box power vectors. In a simulation workflow each row is one
    // independent box, added on one ensemble owner after the collective FFT measurement.
    FML::ENSEMBLE::RunningCovariance powers(2);
    for (int i = FML::ThisTask; i < 8; i += FML::NTasks)
        powers.add({100. + i, 200. + 2 * i});
    // Illustrative EXTERNAL Gaussian model: sum P_model(k)^2 per shell and full signed-k counts.
    const auto gaussian = gaussian_covariance_disjoint_shells({400., 900.}, {20, 30});
    const auto collapsed = connected_power_covariance(powers.global().summary(), gaussian, 100);
    if (!FML::ThisTask)
        std::cout << "toy collapsed T[0,1]=" << collapsed.collapsed_trispectrum[1] << '\n';
}
