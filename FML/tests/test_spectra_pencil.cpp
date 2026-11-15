// The polyspectra (mono-, bi- and trispectrum, with and without the padding that removes aliasing, the bispectrum
// also with shell overlap weights) on
// pencils give the same as on the FFTWGrid slabs (the same density field) on every pencil layout
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/ConnectedFourPoint/Fourier.h>
#include <FML/EnsembleStatistics/MatchedSpectra.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::TESTS::GRIDS;
using namespace FML::CORRELATIONFUNCTIONS;

namespace {
    double field(const std::array<double, 3> & x) {
        double f = 0.0;
        for (int d = 0; d < 3; d++)
            f += std::sin(2 * M_PI * (d + 1) * x[d] + 0.3 * d) + 0.4 * std::cos(4 * M_PI * x[d] + 0.1);
        f += 0.7 * std::sin(2 * M_PI * (x[0] + x[1])) * std::cos(6 * M_PI * x[2]);
        return f + 0.3 * f * f;
    }

    template <template <int> class Grid>
    std::vector<double> spectra(Grid<3> d) {
        std::vector<double> r;
        d.fftw_r2c();
        const int nmesh = d.get_nmesh();
        for (int variant = 0; variant < 3; variant++) {
            BispectrumBinning<3> b(0.0, 2.0 * M_PI * nmesh / 2.5, 5, BispectrumBinning<3>::LINEAR_SPACING);
            b.dealias = variant > 0;
            b.shell_overlap_weights = variant == 2;
            compute_bispectrum(d, b);
            r.insert(r.end(), b.P123.begin(), b.P123.end());
            r.insert(r.end(), b.N123.begin(), b.N123.end());
            r.insert(r.end(), b.pofk.begin(), b.pofk.end());
        }
        TrispectrumBinning<3> t(0.0, 2.0 * M_PI * nmesh / 3, 4, TrispectrumBinning<3>::LINEAR_SPACING);
        compute_trispectrum(d, t);
        r.insert(r.end(), t.P123.begin(), t.P123.end());
        MonospectrumBinning<3> m(0.0, 2.0 * M_PI * nmesh / 2, 6, MonospectrumBinning<3>::LINEAR_SPACING);
        compute_monospectrum(d, m);
        r.insert(r.end(), m.P123.begin(), m.P123.end());

        // The connected four-point function of shells and the matched spectra of two fields
        FML::FOURPOINT::ShellOptions options;
        options.boxsize = 1.0;
        options.k_edges = {0.0, 2 * M_PI * 2.5, 2 * M_PI * 4.5, 2 * M_PI * 6.5};
        const auto four = FML::FOURPOINT::connected_shell_four_point(d, options, {{0, 0, 1, 1}, {0, 1, 1, 2}});
        for (auto & f : four)
            r.insert(r.end(), {f.raw, f.disconnected, f.connected, f.closing_quartets});
        auto d2 = d;
        FML::GRID::smoothing_filter_fourier_space<3>(d2, 0.1, "gaussian");
        const auto ms = FML::ENSEMBLE::matched_spectra(d, d2, 1.0, 2 * M_PI, 2 * M_PI * nmesh / 3, 5);
        r.insert(r.end(), ms.cross.begin(), ms.cross.end());
        r.insert(r.end(), ms.correlation.begin(), ms.correlation.end());
        return r;
    }
} // namespace

int main() {
    const int nmesh = FML::NTasks % 3 == 0 ? 24 : 16;
    FML::GRID::FFTWGrid<3> d(nmesh);
    fill<3>(d, field);
    const auto ref = spectra(d);
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> dp(nmesh, 0, 0, layout);
        fill<3>(dp, field);
        const double diff = max_rel_diff(spectra(dp), ref);
        check(diff < 1e-12,
              "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0) +
                  ": the bi- (also padded and with shell overlap weights), tri- and monospectrum, the connected "
                  "4-point function and the matched spectra of the slabs (rel. diff " +
                  num(diff, 2) + ")");
    }
    return FML::TESTS::summary("polyspectra on pencils");
}
