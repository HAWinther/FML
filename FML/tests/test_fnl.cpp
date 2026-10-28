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
                    const double T =
                        (sum3(SP[i], SP[j], S[k]) + sum3(SP[j], SP[k], S[i]) + sum3(SP[k], SP[i], S[j])) / ntri;
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

// Full four-term reference: always build every inverse-power field and transform
// every product, independently of which coefficients are zero.
template <int N>
FML::GRID::FFTWGrid<N> full_kernel_reference(
    FML::RANDOM::RandomGenerator & rng, const FML::RANDOM::GAUSSIAN::RandomFieldOptions & options,
    bool fixed, double fnl, std::array<double, 4> coefficients) {
    using G = FML::GRID::FFTWGrid<N>;
    using F = FML::GRID::FloatType;
    G gaussian(16, 1, 2);
    auto power = [](double k) { return k == 0 ? 0.0 : 1e-5 / (1 + k * k * k); };
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<N>(
        gaussian, &rng, [](double) { return 1.0; }, fixed, options);
    for (auto index : gaussian.get_fourier_range()) {
        std::array<double, N> wave;
        double k;
        gaussian.get_fourier_wavevector_and_norm_by_index(index, wave, k);
        if (k > 0)
            gaussian.set_fourier_from_index(index, gaussian.get_fourier_from_index(index) * F(std::sqrt(power(k))));
    }
    if (FML::ThisTask == 0)
        gaussian.set_fourier_from_index(0, 0);
    if (fnl == 0)
        return gaussian;
    for (auto & c : coefficients)
        c *= fnl;
    G real = gaussian, source(16, 1, 2);
    real.fftw_c2r();
    double mean = 0, mean2 = 0;
    long long cells = 0;
    for (auto index : real.get_real_range()) {
        const F value = real.get_real_from_index(index), square = value * value;
        mean += value;
        mean2 += square;
        ++cells;
        source.set_real_from_index(index, square);
    }
    FML::SumOverTasks(&mean);
    FML::SumOverTasks(&mean2);
    FML::SumOverTasks(&cells);
    mean /= cells;
    mean2 /= cells;
    for (auto index : real.get_real_range())
        source.set_real_from_index(index, (real.get_real_from_index(index) - mean) +
                                  F(coefficients[0]) * (source.get_real_from_index(index) - mean2));
    source.fftw_r2c();
    if (FML::ThisTask == 0)
        source.set_fourier_from_index(0, 0);
    std::array<G, 3> filtered;
    for (auto & grid : filtered)
        grid = G(16);
    for (auto index : gaussian.get_fourier_range()) {
        std::array<double, N> wave;
        double k;
        gaussian.get_fourier_wavevector_and_norm_by_index(index, wave, k);
        const F p13 = k ? F(std::pow(power(k), -1.0 / 3.0)) : F(0);
        const F p23 = p13 * p13, p33 = p23 * p13;
        const auto phi = gaussian.get_fourier_from_index(index);
        filtered[0].set_fourier_from_index(index, phi * p13);
        filtered[1].set_fourier_from_index(index, phi * p23);
        filtered[2].set_fourier_from_index(index, phi * p33);
    }
    for (auto & grid : filtered)
        grid.fftw_c2r();
    for (auto index : real.get_real_range()) {
        const F phi = real.get_real_from_index(index);
        // real has halos, but physical real indices have the same padding/stride.
        const F a = filtered[0].get_real_from_index(index), b = filtered[1].get_real_from_index(index),
                c = filtered[2].get_real_from_index(index);
        filtered[0].set_real_from_index(index, phi * a);
        filtered[1].set_real_from_index(index, phi * b - a * a);
        filtered[2].set_real_from_index(index, phi * c - a * b);
    }
    for (auto & grid : filtered) {
        grid.fftw_r2c();
        if (FML::ThisTask == 0)
            grid.set_fourier_from_index(0, 0);
    }
    for (auto index : source.get_fourier_range()) {
        std::array<double, N> wave;
        double k;
        source.get_fourier_wavevector_and_norm_by_index(index, wave, k);
        const F p13 = F(std::pow(power(k), 1.0 / 3.0)), p23 = p13 * p13, p33 = p23 * p13;
        const auto a = filtered[0].get_fourier_from_index(index) * p13;
        const auto b = filtered[1].get_fourier_from_index(index) * p23;
        const auto c = filtered[2].get_fourier_from_index(index) * p33;
        source.set_fourier_from_index(index, source.get_fourier_from_index(index) +
                                     (a * F(coefficients[1]) + b * F(coefficients[2]) + c * F(coefficients[3])));
    }
    if (FML::ThisTask == 0)
        source.set_fourier_from_index(0, 0);
    return source;
}

template <int N>
void test_active_kernels() {
    using namespace FML::RANDOM;
    struct Case {
        std::string type;
        double u, fnl;
        std::array<double, 4> coefficients;
    };
    std::vector<Case> cases{{"gaussian", 0, 0, {}}, {"local", 0, 10, {1, 0, 0, 0}},
                             {"generic", 0, 10, {}}, {"generic", 0, 0, {1, 2, 3, 4}}};
    for (double u : {0.0, 0.25, 1.0, 4.0 / 3.0}) {
        cases.push_back({"equilateral", u, 10, {-3 * (1 - u), 4 - 3 * u, 2, -3 * u}});
        cases.push_back({"orthogonal", u, -10, {-9 * (1 - u), 10 - 9 * u, 8, -9 * u}});
    }
    for (int mask = 1; mask < 8; ++mask)
        cases.push_back({"generic", 0, -10, {0.7, mask & 1 ? 0.5 : 0, mask & 2 ? -0.25 : 0, mask & 4 ? 0.125 : 0}});
    for (auto backend : {GAUSSIAN::RandomFieldBackend::NGenIC, GAUSSIAN::RandomFieldBackend::Philox})
        for (bool fixed : {false, true}) {
            bool valid = true, layout = true, rng_state = true;
            double worst = 0;
            GAUSSIAN::RandomFieldOptions options;
            options.backend = backend;
            options.seed = 845;
            options.stream = 7;
            for (const auto & test : cases) {
                RandomGenerator reference_rng(845), actual_rng(845);
                const auto reference = full_kernel_reference<N>(reference_rng, options, fixed, test.fnl, test.coefficients);
                FML::GRID::FFTWGrid<N> actual(16, 1, 2);
                NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier<N>(
                    actual, &actual_rng, [](double k) { return k == 0 ? 0.0 : 1e-5 / (1 + k * k * k); },
                    fixed, test.fnl, test.type, test.u,
                    std::vector<double>(test.coefficients.begin(), test.coefficients.end()), options);
                double error = 0, scale = 0;
                for (auto index : actual.get_fourier_range()) {
                    auto a = actual.get_fourier_from_index(index), b = reference.get_fourier_from_index(index);
                    valid = valid && std::isfinite(a.real()) && std::isfinite(a.imag());
                    error = std::max(error, double(std::abs(a - b)));
                    scale = std::max(scale, double(std::abs(b)));
                }
                FML::MaxOverTasks(&error);
                FML::MaxOverTasks(&scale);
                const double relative = error / std::max(scale, 1e-30);
                worst = std::max(worst, relative);
                layout = layout && actual.get_nmesh() == 16 && actual.get_n_extra_slices_left() == 1 &&
                         actual.get_n_extra_slices_right() == 2 && !actual.get_grid_status_real();
                if (FML::ThisTask == 0)
                    layout = layout && actual.get_fourier_from_index(0) == FML::GRID::ComplexType(0);
                rng_state = rng_state && reference_rng.generate_uniform() == actual_rng.generate_uniform();
            }
            int bad[] = {int(!valid), int(!layout), int(!rng_state)};
            for (auto & value : bad)
                FML::MaxOverTasks(&value);
            const std::string label = std::to_string(N) + "D backend=" + std::to_string(int(backend)) + " fixed=" + std::to_string(fixed);
            check(!bad[0] && worst < (sizeof(FML::GRID::FloatType) == sizeof(float) ? 3e-5 : 3e-12),
                  label + ": all named/generic kernels match the full reference (rel err " + num(worst) + ")");
            check(!bad[1] && !bad[2], label + ": DC, halos, Fourier state and RNG continuation are preserved");
        }
}

int main(int argc, char ** argv) {
    test_active_kernels<2>();
    test_active_kernels<3>();
    if (argc > 1 && std::string(argv[1]) == "kernels")
        return fails ? 1 : 0;
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
        const double e0 = estimate(b0, nullptr, pred), ep = estimate(bp, nullptr, pred),
                     em = estimate(bm, nullptr, pred);
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
