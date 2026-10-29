#include <FML/Survey/SurveyReconstruction.h>
#include <FML/Survey/SurveyWindow.h>
#include <fstream>
#include <iomanip>
#include <random>
using namespace FML::SURVEY;

void write_power(const std::string & path, const PowerSpectrum & power) {
    if (FML::ThisTask)
        return;
    std::ofstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open " + path);
    file << std::setprecision(15) << "# k nmodes P0 P2 P4; alpha=" << power.alpha << " I2=" << power.i2 << '\n';
    for (size_t b = 0; b < power.k.size(); ++b) {
        file << power.k[b] << ' ' << power.nmodes[b];
        for (const auto & pole : power.power)
            file << ' ' << pole[b];
        file << '\n';
    }
}
int main() {
    Geometry geometry;
    geometry.nmesh = 32;
    geometry.boxsize = 800; // e.g. Mpc/h; all positions and reconstruction radii use the same units.
    geometry.minimum = {-200, -400, 0};
    geometry.observer = {0, 0, 0};
    std::mt19937_64 rng(14352);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<CatalogPoint> data, randoms;
    const int nd = 512, nr = 4096;
    const double nbar = nd / (120.0 * 100 * 100);
    for (int kind = 0; kind < 2; ++kind)
        for (int i = 0; i < (kind ? nr : nd); ++i) {
            CatalogPoint point;
            point.position = {60 + 120 * uniform(rng), -50 + 100 * uniform(rng), 200 + 100 * uniform(rng)};
            point.nbar = nbar;
            point.weight = fkp_weight(nbar, 5000);
            if (i % FML::NTasks == FML::ThisTask)
                (kind ? randoms : data).push_back(point);
        }
    PowerOptions power_options;
    power_options.edges = {0, .015, .03, .045, .06, .075, .09};
    const auto before = estimate_power(data, randoms, geometry, power_options);
    write_power("survey_before.txt", before);
    std::vector<double> separations;
    for (int i = 0; i <= 12; ++i)
        separations.push_back(i * 25.0);
    const auto window = estimate_window(randoms, geometry, separations, before.alpha, before.i2);
    // Theory input bins finer than the measured ones and well beyond the last one (the window mixes higher-k power
    // into the measured bins)
    std::vector<double> input_edges;
    for (int i = 0; i <= 200; ++i)
        input_edges.push_back(0.0025 * i);
    const auto matrix = make_binned_window_matrix(window, input_edges, power_options.edges);
    ReconstructionOptions reconstruction;
    reconstruction.smoothing_radius = 30;
    reconstruction.bias = 2;
    reconstruction.growth_rate = .8;
    reconstruction.convention = ReconstructionConvention::RecSym;
    const auto shifted = reconstruct(data, randoms, geometry, reconstruction);
    const auto after = estimate_power(shifted.data, randoms, geometry, power_options, &shifted.randoms);
    write_power("survey_after.txt", after);
    if (FML::ThisTask == 0) {
        std::ofstream window_file("survey_window.txt"), matrix_file("survey_window_matrix.txt");
        window_file << std::setprecision(15) << "# s Q0 Q2 Q4 Q6 Q8 (CIC self pairs removed)\n";
        for (size_t b = 0; b < window.s.size(); ++b) {
            window_file << window.s[b];
            for (const auto & q : window.q)
                window_file << ' ' << q[b];
            window_file << '\n';
        }
        matrix_file << std::setprecision(15)
                    << "# ell-major rows and columns; input bandpowers, k-squared weighted output bands; global IC\n";
        for (const auto & row : matrix.values) {
            for (double x : row)
                matrix_file << x << ' ';
            matrix_file << '\n';
        }
        std::cout << "Reconstruction converged in " << shifted.solver.iterations << " iterations, residual "
                  << shifted.solver.final_residual << '\n';
    }
}
