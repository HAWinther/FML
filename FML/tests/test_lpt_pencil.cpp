// LPT, the Hessian / tidal tensor and the Fourier-space smoothing on pencils. With a PencilGrid the
// results are those of an FFTWGrid (the same field in a different decomposition) on every pencil layout:
//   - LPT: the 1LPT, 2LPT (also with padding), 3LPT (with the curl term, also with padding), spherical collapse and
//     ALPT potentials, the displacement fields (also scale dependent) and compute_3LPT_displacement_field
//   - ComputeHessianWithFT and SymmetricTensorEigensystem
//   - the smoothing filters, the convolutions, the shifts, whitening and the PDF
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/TidalTensor/Hessian.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
namespace LPT = FML::COSMOLOGY::LPT;

namespace {

    template <int N>
    double field(const std::array<double, N> & x) {
        double f = 0.0;
        for (int d = 0; d < N; d++)
            f += std::sin(2 * M_PI * (d + 1) * x[d] + 0.2 * d) + 0.3 * std::cos(6 * M_PI * x[d]);
        double p = 1.0;
        for (int d = 0; d < N; d++)
            p *= std::cos(2 * M_PI * x[d] + 0.1);
        return 0.2 * (f + 2.0 * p + 0.5 * std::sin(10 * M_PI * x[0]) * std::cos(8 * M_PI * x[N - 1]));
    }

    // The results of a grid type: name -> the whole field (real or Fourier, as doubles)
    using Results = std::map<std::string, std::vector<double>>;

    template <class V>
    std::vector<double> flat(const V & v) {
        std::vector<double> r;
        for (auto & x : v) {
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>, ComplexType>) {
                r.push_back(std::real(x));
                r.push_back(std::imag(x));
            } else {
                r.push_back(x);
            }
        }
        return r;
    }

    template <int N, template <int> class Grid>
    Results compute(Grid<N> delta) {
        Results r;
        auto F = [&](const std::string & name, const Grid<N> & g) { r[name] = flat(global_fourier(g)); };
        auto R = [&](const std::string & name, const Grid<N> & g) { r[name] = flat(global_real(g)); };

        // The density in Fourier space
        delta.fftw_r2c();
        F("delta", delta);

        // LPT potentials
        Grid<N> phi1, phi2, phi2pad, phisc, phialpt;
        LPT::compute_1LPT_potential_fourier<N>(delta, phi1);
        F("phi1", phi1);
        LPT::compute_2LPT_potential_fourier<N>(delta, phi2);
        F("phi2", phi2);
        LPT::compute_2LPT_potential_fourier<N>(delta, phi2pad, FML::COSMOLOGY::LPT::Dealiasing::Padding);
        F("phi2 padding", phi2pad);
        LPT::compute_spherical_collapse_potential<N>(delta, phisc);
        F("spherical collapse", phisc);
        LPT::compute_ALPT_potential_fourier<N>(delta, phi2, phialpt, 0.05, "gaussian");
        F("ALPT", phialpt);
        for (auto dealias : {FML::COSMOLOGY::LPT::Dealiasing::None, FML::COSMOLOGY::LPT::Dealiasing::Padding}) {
            const std::string pad = dealias == FML::COSMOLOGY::LPT::Dealiasing::Padding ? " padding" : "";
            Grid<N> p1, p2, pa, pb;
            std::array<Grid<N>, N> avec;
            LPT::compute_3LPT_potential_fourier<N>(delta, p1, p2, pa, pb, avec, false, dealias);
            F("3LPT a" + pad, pa);
            F("3LPT b" + pad, pb);
            F("3LPT Avec" + pad, avec[0]);
            std::array<Grid<N>, N> psi, dpsi;
            LPT::compute_3LPT_displacement_field<N>(delta, psi, dpsi, 1.3, false, dealias);
            R("3LPT Psi_x" + pad, psi[0]);
            R("3LPT dPsidt_y" + pad, dpsi[1]);
        }

        // Displacement fields
        std::array<Grid<N>, N> psi;
        LPT::from_LPT_potential_to_displacement_vector<N>(phi2, psi, 0.7);
        for (int d = 0; d < N; d++)
            R("psi2 " + std::to_string(d), psi[d]);
        std::array<Grid<N>, N> psik;
        LPT::from_LPT_potential_to_displacement_vector_scaledependent<N>(
            phi1, psik, [](double k) { return 1.0 / (1.0 + 0.01 * k); });
        R("psi1 scale dependent", psik[N - 1]);

        // Hessian and the eigenvalues
        Grid<N> real = delta;
        real.fftw_c2r();
        std::vector<Grid<N>> hessian, eigenvalues, eigenvectors;
        FML::HESSIAN::ComputeHessianWithFT<N>(real, hessian, 1.5, true);
        for (size_t i = 0; i < hessian.size(); i++)
            R("hessian " + std::to_string(i), hessian[i]);
        FML::HESSIAN::SymmetricTensorEigensystem<N>(hessian, eigenvalues, eigenvectors, true);
        R("eigenvalue 0", eigenvalues[0]);
        R("eigenvector 1", eigenvectors[1]);

        // Smoothing
        for (std::string method : {"gaussian", "tophat", "sharpk"}) {
            Grid<N> s = delta;
            FML::GRID::smoothing_filter_fourier_space<N>(s, 0.07, method);
            F("smoothing " + method, s);
        }
        Grid<N> conv;
        FML::GRID::convolution_fourier_space<N>(delta, phi1, conv);
        F("convolution", conv);
        Grid<N> shifted = delta;
        FML::GRID::shift_grid_fourier<N>(shifted, {0.13, -0.07, 0.31});
        F("shift", shifted);
        Grid<N> white = delta;
        FML::GRID::whitening_fourier_space<N>(white);
        F("whitening", white);
        std::vector<double> x, pdf;
        FML::GRID::compute_grid_PDF<N>(real, 20, x, pdf);
        r["PDF"] = pdf;
        return r;
    }

} // namespace

int main() {
    constexpr int N = 3;
    const int nmesh = FML::NTasks % 3 == 0 ? 24 : 32;
    FML::GRID::FFTWGrid<N> d(nmesh, 1, 1);
    fill<N>(d, field<N>);
    const auto ref = compute<N>(d);

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0);
        FML::GRID::PencilGrid<N> dp(nmesh, 1, 1, layout);
        fill<N>(dp, field<N>);
        const auto res = compute<N>(dp);
        double worst = 0.0;
        std::string worst_name, missing;
        for (auto & [key, v] : ref) {
            if (not res.count(key)) {
                missing = key;
                continue;
            }
            const double diff = max_rel_diff(res.at(key), v);
            if (diff > worst) {
                worst = diff;
                worst_name = key;
            }
        }
        check(worst < 1e-12 and missing.empty(),
              name + ": " + std::to_string(ref.size()) + " fields the same as with FFTWGrid (largest rel. diff " +
                  num(worst, 2) + (worst_name.empty() ? "" : " in " + worst_name) + ")");
    }
    return FML::TESTS::summary("LPT, Hessian and smoothing on pencils");
}
