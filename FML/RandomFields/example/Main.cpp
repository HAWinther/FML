#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>

template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;

const double box = 1.0; // Size of the box
const int Ndim = 3;     // Number of dimensions we are working in

//=========================================================================
// Naive way of estimating fNL (just for testing)
// We basically compute fNL = <B/B_theory> (with B_theory for fNL=1)
//=========================================================================
template <int N>
std::pair<double, double> estimate_fnl(FML::CORRELATIONFUNCTIONS::BispectrumBinning<N> & bofk, std::string fnl_type);

//=========================================================================
// Generate a random field and compute its power-spectrum
//=========================================================================
template <int N>
void compute_power_spectrum(int Nmesh,
                            FML::RANDOM::RandomGenerator * rng,
                            std::function<double(double)> Powspec,
                            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<N> & pofk) {
    const bool fix_amplitude = false;
    FFTWGrid<N> grid(Nmesh);
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(grid, rng, Powspec, fix_amplitude);
    FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(grid, pofk);
    pofk.scale(box);
}

//=========================================================================
// Compute mean power-spectrum over many realisations
//=========================================================================
template <int N>
void generate_mean_over_realisations(int Nreal, int Nmesh, std::function<double(double)> Powspec) {
    std::shared_ptr<FML::RANDOM::RandomGenerator> rng = std::make_shared<FML::RANDOM::RandomGenerator>();

    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<N> pofk_total(Nmesh / 2);
    for (int i = 0; i < Nreal; i++) {

        // Progress bar
        if (FML::ThisTask == 0)
            if ((i * 10) / Nreal != ((i + 1) * 10) / Nreal)
                std::cout << "Integrating up " << 100 * (i + 1) / Nreal << " %\n";
        ;

        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<N> pofk(Nmesh / 2);
        compute_power_spectrum<N>(Nmesh, rng.get(), Powspec, pofk);

        // Init/add up
        for (int j = 0; j < pofk_total.n; j++) {
            if (i == 0) {
                pofk_total.kbin[j] = pofk.kbin[j];
                pofk_total.pofk[j] = 0.0;
            }
            pofk_total.pofk[j] += pofk.pofk[j] / double(Nreal);
        }
    }

    if (FML::ThisTask == 0)
        for (int j = 0; j < pofk_total.n; j++)
            std::cout << pofk_total.kbin[j] << " "
                      << pofk_total.pofk[j] / (Powspec(pofk_total.kbin[j] * box) * std::pow(box, N)) << "\n";
}

int main() {

    //=========================================================================
    // Set up stuff
    //=========================================================================
    FML::RANDOM::RandomGenerator * rng = new FML::RANDOM::RandomGenerator;

    const int Nmesh = 128;
    FFTWGrid<Ndim> grid(Nmesh);

    // Non-gaussianity
    const double fNL = 200.0;
    const std::string type_of_fnl = "local";
    const bool fix_amplitude = true;

    // Binning in k (kmin should be 0 for linear spacing - we will anyway use klow=0 for the smallest bin)
    const int nbin = 32;
    const double kmin = 0.0;
    const double kmax = 2.0 * M_PI * Nmesh / 2;

    // For binning up the power-spectrum
    FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk_all(
        kmin, kmax, nbin, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim>::LINEAR_SPACING);

    // For binning up the monospectrum P(k1,k2) and the bispectrum B(k1,k2,k3)
    // (these are using the same algorithm)
    FML::CORRELATIONFUNCTIONS::BispectrumBinning<Ndim> bofk_all(
        kmin, kmax, nbin, FML::CORRELATIONFUNCTIONS::BispectrumBinning<Ndim>::LINEAR_SPACING);
    FML::CORRELATIONFUNCTIONS::MonospectrumBinning<Ndim> monoofk_all(
        kmin, kmax, nbin, FML::CORRELATIONFUNCTIONS::MonospectrumBinning<Ndim>::LINEAR_SPACING);
    std::vector<double> N123; // To store bincount to help speed up bispectrum computation

    //=========================================================================
    // Analytical P(k)
    // This is P(k) / Volume with P(k) = 1/k^NDIM  and Delta = k^NDIM P(k) ~ 1e-6
    // so scale-invariant spectrum in this example
    //=========================================================================
    auto Powspec = [&](double kBox) -> double {
        if (kBox == 0.0)
            return 0.0;
        double k = kBox / box;                        // k in physical units
        double volume = std::pow(box, Ndim);          // Volume of the box in physical units
        double pofk = 1e-6 * std::pow(1.0 / k, Ndim); // pofk in physical units
        return pofk / volume;                         // Dimensionless P/V
    };

    // Generate 100 gaussian spectra and take the mean
    // generate_mean_over_realisations<Ndim>(100, Nmesh, Powspec);
    // exit(1);

    //=========================================================================
    // Generate nreal realisation
    //=========================================================================
    const int nreal = 10;
    double running_mean = 0.0;
    double running_std = 0.0;
    for (int s = 0; s < nreal; s++) {
        if (FML::ThisTask == 0)
            std::cout << s << "\n";

        //=========================================================================
        // Make a random field in fourier space
        // Test generating non-gaussian potential Phi
        //=========================================================================
        FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier(
            grid, rng, Powspec, fix_amplitude, fNL, type_of_fnl);

        //=========================================================================
        // Compute power-spectrum
        //=========================================================================
        FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> pofk(nbin);
        FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(grid, pofk);
        pofk.scale(box);

        //=========================================================================
        // Compare to input
        //=========================================================================
        if (FML::ThisTask == 0 and s == 0) {
            std::cout << "k    P(k) / Pinput(k): \n";
            for (int i = 0; i < pofk.n; i++) {
                double integer_k = pofk.k[i] * box / (2.0 * M_PI);
                double pofk_over_pofk_input = pofk.pofk[i] / (Powspec(pofk.kbin[i] * box) * std::pow(box, Ndim));
                std::vector<int> coord{i, i};
                std::cout << integer_k << " " << pofk_over_pofk_input << " "
                          << "\n";
            }
        }

        //=========================================================================
        // Compute bispectrum
        //=========================================================================
        FML::CORRELATIONFUNCTIONS::BispectrumBinning<Ndim> bofk(kmin, kmax, nbin);
        if (N123.size() > 0) // Just for speeding it up with many real: only need to compute N123 once
            bofk.set_bincount(N123);
        FML::CORRELATIONFUNCTIONS::compute_bispectrum(grid, bofk);
        if (N123.size() == 0) // Just for speeding it up with many real: only need to compute N123 once
            N123 = bofk.N123;
        bofk.scale(box);

        //=========================================================================
        // Compute monospectrum (matrix <d(k1)d(k2)> so diagonal is just P(k)
        // - this uses the same algorithm as the bispectrum so useful as a test)
        //=========================================================================
        FML::CORRELATIONFUNCTIONS::MonospectrumBinning<Ndim> monoofk(kmin, kmax, nbin);
        FML::CORRELATIONFUNCTIONS::compute_monospectrum(grid, monoofk);
        monoofk.scale(box);

        //=========================================================================
        // Estimate fNL as a test that is works
        //=========================================================================
        auto res = estimate_fnl(bofk, type_of_fnl);
        running_mean += res.first;
        running_std += res.second;
        if (FML::ThisTask == 0)
            std::cout << "Estimate of fNL current: " << res.first << "  Std: " << res.second / std::sqrt(s + 1)
                      << " Estimate of fNL all: " << running_mean / (s + 1.0)
                      << " Error on the mean: " << running_std / (s + 1.0) / std::sqrt(s + 1) << "\n";

        //=========================================================================
        // In case we do many realisations add up
        // We bin P(k) and B(k,k,k) over all the realisations
        //=========================================================================
        bofk_all.combine(bofk);
        pofk_all.combine(pofk);
        monoofk_all.combine(monoofk);
    }

    //=========================================================================
    // Output
    //=========================================================================

    // This is B123 / (2p1p2 + 2p2p3 + 2p3p1)
    auto analytic = [&](int i, int j, int k, std::string _fnl_type) -> double {
        double pk1 = bofk_all.pofk[i];
        double pk2 = bofk_all.pofk[j];
        double pk3 = bofk_all.pofk[k];

        pk1 = Powspec(bofk_all.kbin[i]);
        pk2 = Powspec(bofk_all.kbin[j]);
        pk3 = Powspec(bofk_all.kbin[k]);

        double res = 0.0;
        if (_fnl_type == "local") {
            res = 2.0 * (pk1 * pk2 + pk2 * pk3 + pk3 * pk1);
        } else if (_fnl_type == "equilateral") {
            res = (-6.0 * (pk1 * pk2 + pk2 * pk3 + pk3 * pk1) +
                   6.0 * std::pow(pk1, 1 / 3.) * std::pow(pk3, 2 / 3.) * std::pow(pk2, 3 / 3.) +
                   6.0 * std::pow(pk1, 1 / 3.) * std::pow(pk2, 2 / 3.) * std::pow(pk3, 3 / 3.) +
                   6.0 * std::pow(pk2, 1 / 3.) * std::pow(pk1, 2 / 3.) * std::pow(pk3, 3 / 3.) +
                   6.0 * std::pow(pk2, 1 / 3.) * std::pow(pk3, 2 / 3.) * std::pow(pk1, 3 / 3.) +
                   6.0 * std::pow(pk3, 1 / 3.) * std::pow(pk2, 2 / 3.) * std::pow(pk1, 3 / 3.) +
                   6.0 * std::pow(pk3, 1 / 3.) * std::pow(pk1, 2 / 3.) * std::pow(pk2, 3 / 3.) -
                   12.0 * std::pow(pk1 * pk2 * pk3, 2 / 3.));
        } else if (_fnl_type == "orthogonal") {
            res = (-18.0 * (pk1 * pk2 + pk2 * pk3 + pk3 * pk1) +
                   18.0 * std::pow(pk1, 1 / 3.) * std::pow(pk3, 2 / 3.) * std::pow(pk2, 3 / 3.) +
                   18.0 * std::pow(pk1, 1 / 3.) * std::pow(pk2, 2 / 3.) * std::pow(pk3, 3 / 3.) +
                   18.0 * std::pow(pk2, 1 / 3.) * std::pow(pk1, 2 / 3.) * std::pow(pk3, 3 / 3.) +
                   18.0 * std::pow(pk2, 1 / 3.) * std::pow(pk3, 2 / 3.) * std::pow(pk1, 3 / 3.) +
                   18.0 * std::pow(pk3, 1 / 3.) * std::pow(pk2, 2 / 3.) * std::pow(pk1, 3 / 3.) +
                   18.0 * std::pow(pk3, 1 / 3.) * std::pow(pk1, 2 / 3.) * std::pow(pk2, 3 / 3.) -
                   48.0 * std::pow(pk1 * pk2 * pk3, 2 / 3.));
        } else {
            assert(false);
        }

        // Transform to reduced bispectrum
        res /= (pk1 * pk2 + pk2 * pk3 + pk3 * pk1);

        return res;
    };

    if (FML::ThisTask == 0) {
        std::ofstream fp("p12.txt");
        for (int i = 0; i < monoofk_all.n; i++) {
            for (int j = 0; j < monoofk_all.n; j++) {
                fp << i << "  " << j << " " << monoofk_all.get_spectrum(i, j) << " "
                   << monoofk_all.get_reduced_spectrum(i, j) << " " << monoofk_all.get_bincount(i, j) << "\n";
            }
        }
    }

    if (FML::ThisTask == 0) {
        std::ofstream fp("b123.txt");
        for (int i = 0; i < bofk_all.n; i++) {
            for (int j = 0; j < bofk_all.n; j++) {
                for (int k = 0; k < bofk_all.n; k++) {

                    double pij = (Powspec(bofk_all.kbin[i]) * Powspec(bofk_all.kbin[j]) +
                                  Powspec(bofk_all.kbin[j]) * Powspec(bofk_all.kbin[k]) +
                                  Powspec(bofk_all.kbin[k]) * Powspec(bofk_all.kbin[i]));

                    double B = bofk_all.get_spectrum(i, j, k);
                    double N = bofk_all.get_bincount(i, j, k);

                    // The binned bispectrum divided by (p1p2+...) with pi binned up in the same way as B
                    double q = bofk_all.get_reduced_spectrum(i, j, k);

                    // The same as above just using the analytical result for P(k)
                    // instead of the P(k) we estimated when computing the bispectrum
                    double q2 = bofk_all.get_spectrum(i, j, k) / (2.0 * pij);

                    // Analytical estimate for the given fNL type (q / anal ~ fNL)
                    double anal = analytic(i, j, k, type_of_fnl);

                    if (pij == 0.0 or anal == 0.0)
                        continue;

                    fp << bofk_all.kbin[i] << " " << bofk_all.kbin[j] << " " << bofk_all.kbin[k] << " " << q / anal
                       << " " << q2 / anal << " " << B << " " << q << " " << N << "\n";
                }
            }
        }
    }

    //=========================================================================
    // Compute density PDF for the last realization
    //=========================================================================

    // To real space
    grid.fftw_c2r();

    //=========================================================================
    // Compute delta_min,max and mean
    //=========================================================================
    double delta_min = 1e100;
    double delta_max = -1e100;
    double delta_mean = 0.0;
    for (auto & real_index : grid.get_real_range()) {
        auto delta = grid.get_real_from_index(real_index);
        if (delta < delta_min)
            delta_min = delta;
        if (delta > delta_max)
            delta_max = delta;
        delta_mean += delta;
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &delta_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &delta_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &delta_mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    delta_mean /= std::pow(grid.get_nmesh(), Ndim);

    if (FML::ThisTask == 0)
        std::cout << "Min/Max/Mean density contrast: " << delta_min << " " << delta_max << " " << delta_mean << "\n";

    //=========================================================================
    // Bin up
    //=========================================================================
    int nbins = 100;
    double weight = 1.0 / std::pow(Nmesh, Ndim); // So that it sums to 1
    std::vector<double> count(nbins, 0.0);
    for (auto & real_index : grid.get_real_range()) {
        auto delta = grid.get_real_from_index(real_index);
        int index = int((delta - delta_min) / (delta_max - delta_min) * nbins + 0.5);
        if (index < nbins and index >= 0) {
            count[index] += weight;
        }
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, count.data(), nbins, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    //=========================================================================
    // Output PDF of the density field
    //=========================================================================
    if (FML::ThisTask == 0) {
        std::ofstream fp("pdf.txt");
        for (int i = 0; i < nbins; i++)
            fp << delta_min + (delta_max - delta_min) * i / double(nbins) << " " << count[i] << "\n";
    }
}

//=========================================================================
// Naive method of estimating fNL. Use the theoretical expressions for
// the bispectrum and just sum the ratio over all the bins
// (does not work well for orthogonal as norm can be zero)
//=========================================================================
template <int N>
std::pair<double, double> estimate_fnl(FML::CORRELATIONFUNCTIONS::BispectrumBinning<N> & bofk, std::string fnl_type) {
    int nbins = bofk.n;
    std::vector<double> & k_bin = bofk.kbin;

    // Q123 = B123/(P1P2+cyc) = fNL * analytic(i,j,k)
    auto analytic = [&](int i, int j, int k, std::string _fnl_type) -> double {
        double fac = (bofk.pofk[i] * bofk.pofk[j] + bofk.pofk[j] * bofk.pofk[k] + bofk.pofk[k] * bofk.pofk[i]);

        if (_fnl_type == "local") {
            return 2.0 * fac;
        } else if (_fnl_type == "equilateral") {
            return (
                -6.0 * (bofk.pofk[i] * bofk.pofk[j] + bofk.pofk[j] * bofk.pofk[k] + bofk.pofk[k] * bofk.pofk[i]) +
                6.0 * std::pow(bofk.pofk[i], 1 / 3.) * std::pow(bofk.pofk[j], 2 / 3.) * std::pow(bofk.pofk[k], 3 / 3.) +
                6.0 * std::pow(bofk.pofk[j], 1 / 3.) * std::pow(bofk.pofk[i], 2 / 3.) * std::pow(bofk.pofk[k], 3 / 3.) +
                6.0 * std::pow(bofk.pofk[j], 1 / 3.) * std::pow(bofk.pofk[k], 2 / 3.) * std::pow(bofk.pofk[i], 3 / 3.) +
                6.0 * std::pow(bofk.pofk[k], 1 / 3.) * std::pow(bofk.pofk[j], 2 / 3.) * std::pow(bofk.pofk[i], 3 / 3.) +
                6.0 * std::pow(bofk.pofk[k], 1 / 3.) * std::pow(bofk.pofk[i], 2 / 3.) * std::pow(bofk.pofk[j], 3 / 3.) +
                6.0 * std::pow(bofk.pofk[i], 1 / 3.) * std::pow(bofk.pofk[k], 2 / 3.) * std::pow(bofk.pofk[j], 3 / 3.) -
                12.0 * std::pow(bofk.pofk[i] * bofk.pofk[j] * bofk.pofk[k], 2 / 3.));
        } else if (_fnl_type == "orthogonal") {
            return (-18.0 * (bofk.pofk[i] * bofk.pofk[j] + bofk.pofk[j] * bofk.pofk[k] + bofk.pofk[k] * bofk.pofk[i]) +
                    18.0 * std::pow(bofk.pofk[i], 1 / 3.) * std::pow(bofk.pofk[j], 2 / 3.) *
                        std::pow(bofk.pofk[k], 3 / 3.) +
                    18.0 * std::pow(bofk.pofk[j], 1 / 3.) * std::pow(bofk.pofk[i], 2 / 3.) *
                        std::pow(bofk.pofk[k], 3 / 3.) +
                    18.0 * std::pow(bofk.pofk[j], 1 / 3.) * std::pow(bofk.pofk[k], 2 / 3.) *
                        std::pow(bofk.pofk[i], 3 / 3.) +
                    18.0 * std::pow(bofk.pofk[k], 1 / 3.) * std::pow(bofk.pofk[j], 2 / 3.) *
                        std::pow(bofk.pofk[i], 3 / 3.) +
                    18.0 * std::pow(bofk.pofk[k], 1 / 3.) * std::pow(bofk.pofk[i], 2 / 3.) *
                        std::pow(bofk.pofk[j], 3 / 3.) +
                    18.0 * std::pow(bofk.pofk[i], 1 / 3.) * std::pow(bofk.pofk[k], 2 / 3.) *
                        std::pow(bofk.pofk[j], 3 / 3.) -
                    48.0 * std::pow(bofk.pofk[i] * bofk.pofk[j] * bofk.pofk[k], 2 / 3.));
        } else {
            assert(false);
            return 0.0;
        }
    };

    double mean = 0.0;
    double std = 0.0;
    double count = 0;
    for (int i = 0; i < nbins; i++) {
        for (int j = 0; j < nbins; j++) {
            for (int k = 0; k < nbins; k++) {
                std::vector<double> inds{k_bin[i], k_bin[j], k_bin[k]};
                std::sort(inds.begin(), inds.end(), std::less<double>());
                if (inds[0] + inds[1] >= inds[2]) {
                    double norm = analytic(i, j, k, fnl_type);

                    if (norm > 0.0) {
                        double value = bofk.get_spectrum(i, j, k) / norm;
                        double weight = 1.0;
                        // weight = bofk.get_bincount(i, j, k);
                        value *= weight;
                        mean += value;
                        std += value * value;
                        count += weight;
                    }
                }
            }
        }
    }

    // All tasks have the same data so no need for communication here
    mean /= double(count);
    std /= double(count);
    std = std::sqrt(std - mean * mean) / std::sqrt(count);

    return {mean, std};
}
