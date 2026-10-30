#include <FML/EnsembleStatistics/MatchedSpectra.h>
#include <FML/Morphology/VoxelMorphology.h>
#include <iostream>
int main() {
    using namespace FML::ENSEMBLE;
    FML::GRID::FFTWGrid<3> control(8), target(8);
    control.fill_real_grid(
        [](std::array<double, 3> & x) { return std::cos(2 * M_PI * x[0]) + 0.5 * std::sin(4 * M_PI * x[1]); });
    target.fill_real_grid(
        [](std::array<double, 3> & x) { return 2 * std::cos(2 * M_PI * x[0]) + std::sin(4 * M_PI * x[1]); });
    auto morphology = FML::MORPHOLOGY::excursions(control, {0}, 80);
    control.fftw_r2c();
    target.fftw_r2c();
    auto s = matched_spectra(target, control, 80, 2 * M_PI / 80, 6 * M_PI / 80, 3);
    auto scaled = matched_spectra(target, control, 160, 2 * M_PI / 160, 6 * M_PI / 160, 3);
    bool ok = morphology.size() == 1;
    int tested = 0;
    for (size_t i = 0; i < s.k.size(); ++i)
        if (s.control[i] > 1e-10) {
            ++tested;
            ok &= std::abs(s.target[i] / s.control[i] - 4) < 1e-5 && std::abs(s.cross[i] / s.control[i] - 2) < 1e-5 &&
                  std::abs(s.correlation[i] - 1) < 1e-5 && std::abs(scaled.control[i] / s.control[i] - 8) < 1e-5 &&
                  std::abs(scaled.k[i] / s.k[i] - .5) < 1e-5;
        }
    ok &= tested >= 2;
    bool rejected = false;
    try {
        matched_spectra(target, control, 80, 0, 1, 1);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    ok &= rejected;
    int failed = !ok;
    FML::MaxOverTasks(&failed);
    if (!FML::ThisTask)
        std::cout << (failed ? "[FAIL] " : "[PASS] ")
                  << "matched auto/cross spectra, coherence, physical box scaling and FFTW morphology adapter\n";
    return failed;
}
