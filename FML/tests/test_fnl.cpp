//==========================================================
// Tests for local primordial non-gaussianity (RandomFields/NonLocalGaussianRandomField.h) measured with the
// bispectrum (ComputePowerSpectra): for Phi = phi + fNL (phi^2 - <phi^2>) the reduced bispectrum is
// Q = B / (P1 P2 + P2 P3 + P3 P1) = 2 fNL for every triangle (to leading order in fNL Phi).
//
// Measured in bins the expectation is 2 fNL sum_triangles (P1 P2 + cyc) / N123 / (Pbar1 Pbar2 + cyc), which for
// P ~ k^-3 is a few percent above 2 fNL (P varies over a bin; 3% for 8 bins on a 64^3 grid). The test computes this
// binned prediction exactly (with the same FFT sums as the estimator) and checks:
//   - the estimate of fNL (Q / 2 averaged over all triangles) for fNL = 0, +fNL, -fNL
//   - the same-seed difference (Q(fNL) - Q(0)) / 2, which removes most of the noise of the gaussian field
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

constexpr int NDIM = 3;
using Grid = FML::GRID::FFTWGrid<NDIM>;
using Binning = FML::CORRELATIONFUNCTIONS::BispectrumBinning<NDIM>;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << std::setprecision(4) << x;
    return os.str();
}

const int Nmesh = 64;
const int nbin = 8;
const double kmax = 2.0 * M_PI * Nmesh / 4.0; // kNyquist / 2: no triangle wraps around the grid
const double Delta2 = 1e-7;                   // Delta^2_Phi: fNL Phi ~ 0.03 for fNL = 100 (corrections < 1%)
const double fNL = 100.0;
const int nseeds = 4;

// P(kB) / V for a scale invariant spectrum k^3 P(k) / (2 pi^2) = Delta2 (kB = k * boxsize)
double Pofk(double kB) { return kB == 0.0 ? 0.0 : 2.0 * M_PI * M_PI * Delta2 / (kB * kB * kB); }

// The modes of bin i (the same selection as compute_polyspectrum: [klow, khigh) without k = 0)
bool in_bin(const Binning & b, int i, double kmag2) {
    return kmag2 > 0.0 and kmag2 >= b.klow[i] * b.klow[i] and kmag2 < b.khigh[i] * b.khigh[i];
}

// The weights of the estimator and the binned prediction of Q / (2 fNL) for each configuration
struct Prediction {
    std::vector<double> weight, ratio;
    double total_ratio{0.0};
};
Prediction binned_prediction(const Binning & b) {
    // For each bin the real space fields of the shell (1 and P(k)) and the mean P over the shell
    std::vector<Grid> S(nbin), SP(nbin);
    std::vector<double> Pbar(nbin, 0.0);
    for (int i = 0; i < nbin; i++) {
        S[i] = Grid(Nmesh);
        SP[i] = Grid(Nmesh);
        double sumP = 0.0, nmodes = 0.0;
        std::array<double, NDIM> kvec;
        double kmag2;
        for (auto && index : S[i].get_fourier_range()) {
            S[i].get_fourier_wavevector_and_norm2_by_index(index, kvec, kmag2);
            const bool in = in_bin(b, i, kmag2);
            const double P = in ? Pofk(std::sqrt(kmag2)) : 0.0;
            S[i].set_fourier_from_index(index, in ? 1.0 : 0.0);
            SP[i].set_fourier_from_index(index, P);
            if (in) {
                const auto last = size_t(index) % size_t(Nmesh / 2 + 1);
                const double w = (last > 0 and last < size_t(Nmesh / 2)) ? 2.0 : 1.0;
                sumP += w * P;
                nmodes += w;
            }
        }
        FML::SumOverTasks(&sumP);
        FML::SumOverTasks(&nmodes);
        Pbar[i] = nmodes > 0.0 ? sumP / nmodes : 0.0;
        S[i].fftw_c2r();
        SP[i].fftw_c2r();
    }
    auto sum3 = [&](const Grid & a, const Grid & c, const Grid & d) {
        double s = 0.0;
        for (auto && index : a.get_real_range())
            s += double(a.get_real_from_index(index)) * c.get_real_from_index(index) * d.get_real_from_index(index);
        FML::SumOverTasks(&s);
        return s;
    };
    Prediction pred;
    double sw = 0.0, swr = 0.0;
    for (int i = 0; i < nbin; i++)
        for (int j = i; j < nbin; j++)
            for (int k = j; k < nbin; k++) {
                const double ntri = sum3(S[i], S[j], S[k]);
                double w = 0.0, r = 0.0;
                if (ntri > 0.5 and Pbar[i] > 0.0 and Pbar[j] > 0.0 and Pbar[k] > 0.0) {
                    const double T = (sum3(SP[i], SP[j], S[k]) + sum3(SP[j], SP[k], S[i]) + sum3(SP[k], SP[i], S[j])) / ntri;
                    const double Sm = Pbar[i] * Pbar[j] + Pbar[j] * Pbar[k] + Pbar[k] * Pbar[i];
                    // Inverse variance of Q for a gaussian field (up to a constant)
                    w = ntri * Sm * Sm / (Pbar[i] * Pbar[j] * Pbar[k]);
                    r = T / Sm;
                }
                pred.weight.push_back(w);
                pred.ratio.push_back(r);
                sw += w;
                swr += w * r;
            }
    pred.total_ratio = swr / sw;
    return pred;
}

Binning measure(double fnl, unsigned int seed) {
    FML::RANDOM::RandomGenerator rng;
    rng.set_seed(seed);
    Grid phi(Nmesh);
    FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier(phi, &rng, Pofk, false, fnl, "local");
    Binning b(0.0, kmax, nbin);
    FML::CORRELATIONFUNCTIONS::compute_bispectrum(phi, b);
    return b;
}

// fNL = sum_c w_c Q_c / 2 / sum_c w_c (minus the same for b0 if given)
double estimate(Binning & b, Binning * b0, const Prediction & pred) {
    double sw = 0.0, swq = 0.0;
    size_t c = 0;
    for (int i = 0; i < nbin; i++)
        for (int j = i; j < nbin; j++)
            for (int k = j; k < nbin; k++, c++) {
                if (pred.weight[c] == 0.0 or b.get_bincount(i, j, k) <= 0.0)
                    continue;
                double Q = b.get_reduced_spectrum(i, j, k);
                if (b0)
                    Q -= b0->get_reduced_spectrum(i, j, k);
                sw += pred.weight[c];
                swq += pred.weight[c] * Q / 2.0;
            }
    return swq / sw;
}

int main() {
    const std::string label = std::to_string(FML::NTasks) + " task(s): ";

    // Binned prediction (needs the bin edges only)
    Binning edges(0.0, kmax, nbin);
    const Prediction pred = binned_prediction(edges);
    const double expected = fNL * pred.total_ratio;
    if (FML::ThisTask == 0)
        std::cout << "Binned prediction of the estimator: " << num(pred.total_ratio) << " fNL\n";
    check(pred.total_ratio > 1.0 and pred.total_ratio < 1.06,
          label + "binned prediction within a few percent of Q = 2 fNL (" + num(pred.total_ratio) + ")");

    double mean0 = 0.0, meanp = 0.0, meanm = 0.0, meand = 0.0;
    for (int s = 1; s <= nseeds; s++) {
        auto b0 = measure(0.0, s);
        auto bp = measure(fNL, s);
        auto bm = measure(-fNL, s);
        const double e0 = estimate(b0, nullptr, pred), ep = estimate(bp, nullptr, pred), em = estimate(bm, nullptr, pred);
        const double ed = estimate(bp, &b0, pred);
        if (FML::ThisTask == 0)
            std::cout << "  seed " << s << ": fNL = 0: " << num(e0) << "  fNL = " << fNL << ": " << num(ep)
                      << "  fNL = " << -fNL << ": " << num(em) << "  same seed difference: " << num(ed) << "\n";
        mean0 += e0 / nseeds;
        meanp += ep / nseeds;
        meanm += em / nseeds;
        meand += ed / nseeds;
    }

    // The scatter between seeds (64^3, 8 bins, Delta2 = 1e-7, 10 seeds) is 8 for the estimate and 1.5% of fNL for the
    // same-seed difference; the limits are about 4 sigma of the mean of 4 seeds
    check(std::fabs(mean0) < 16.0, label + "gaussian field: fNL = " + num(mean0) + " consistent with 0");
    check(std::fabs(meanp - expected) < 16.0,
          label + "fNL = " + num(fNL) + ": estimate " + num(meanp) + " (expected " + num(expected) + ")");
    check(std::fabs(meanm + expected) < 16.0,
          label + "fNL = " + num(-fNL) + ": estimate " + num(meanm) + " (expected " + num(-expected) + ")");
    check(std::fabs(meand / expected - 1.0) < 0.03,
          label + "same-seed difference (Q(fNL) - Q(0)) / 2 = " + num(meand) + " (expected " + num(expected) + ")");

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All fNL tests passed\n" : std::to_string(fails) + " fNL test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
