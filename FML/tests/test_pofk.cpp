//==========================================================
// Tests for FML::CORRELATIONFUNCTIONS (ComputePowerSpectra) against brute force:
//   - the shot noise subtracted from P(k) and B(k1,k2,k3) for particles with masses and delta-f weights:
//     the exact value sum m^2/(sum m)^2 (sum w^2/N^2) and P(k) of Poisson particles minus it ~ 0
//   - NGP with interlacing conserves the mass (also with one task)
//   - the number of triangles N123 of the bispectrum bins = a brute-force count over the grid (also small bins)
//   - high-frequency triangles close as integer vectors, without Fourier-grid wrapping
//   - computed geometry counts are marked reusable automatically
//   - even/odd meshes count conjugate modes correctly in power spectra and monopoles
//   - the P(k) bin index below the first bin is -1
//   - the monopole from compute_power_spectrum_multipoles_fourier = P(k) from bin_up_power_spectrum (counts too)
//   - PolyspectrumBinning: the binning type is kept, reset() clears the bin count, empty bins give 0 (not NaN)
// The header is also included in test_pofk_odr.cpp (the header must be usable from several translation units)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::CORRELATIONFUNCTIONS;
template <int N>
using Grid = FML::GRID::FFTWGrid<N>;

int pofk_odr_check(); // defined in test_pofk_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << x;
    return os.str();
}

struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
struct ParticleMass {
    double pos[3];
    double mass;
    double * get_pos() { return pos; }
    double get_mass() const { return mass; }
    constexpr int get_ndim() const { return 3; }
};
struct ParticleDeltaf {
    double pos[3];
    double w;
    double * get_pos() { return pos; }
    double get_deltaf_weight() const { return w; }
    constexpr int get_ndim() const { return 3; }
};

// Poisson particles in the local slab of the grid; weight() sets the mass/weight
template <class T, class F>
std::vector<T> make_particles(size_t npart_total, uint64_t seed, F weight) {
    std::mt19937_64 rng(seed + 7919 * FML::ThisTask);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const size_t n = npart_total / FML::NTasks;
    std::vector<T> part(n);
    for (auto & p : part) {
        p.pos[0] = FML::xmin_domain + (FML::xmax_domain - FML::xmin_domain) * uniform(rng);
        p.pos[1] = uniform(rng);
        p.pos[2] = uniform(rng);
        weight(p, uniform(rng));
    }
    return part;
}

// Mean of P(k) (weighted by the number of modes) over the bins
template <int N>
double mean_power(const PowerSpectrumBinning<N> & pofk) {
    double sum = 0.0, count = 0.0;
    for (int i = 0; i < pofk.n; i++) {
        sum += pofk.pofk[i] * pofk.count[i];
        count += pofk.count[i];
    }
    return sum / count;
}

// P(k) of Poisson particles with and without shot noise subtraction. Returns {P_sub - P_nosub (all bins equal),
// mean P_sub / shot noise} and checks that the difference is the same in all bins
template <class T>
std::pair<double, double> shotnoise_test(std::vector<T> & part, size_t npart_total, double S2_expected) {
    const int Nmesh = 32;
    PowerSpectrumBinning<3> sub(0.0, 2.0 * M_PI * Nmesh / 2.0 * 0.75, 12, PowerSpectrumBinning<3>::LINEAR_SPACING);
    PowerSpectrumBinning<3> nosub = sub;
    nosub.subtract_shotnoise = false;
    compute_power_spectrum<3>(Nmesh, part.data(), part.size(), npart_total, sub, "PCS", true);
    compute_power_spectrum<3>(Nmesh, part.data(), part.size(), npart_total, nosub, "PCS", true);
    double maxdiff = 0.0;
    for (int i = 0; i < sub.n; i++)
        if (sub.count[i] > 0)
            maxdiff = std::max(maxdiff, std::fabs((nosub.pofk[i] - sub.pofk[i]) / S2_expected - 1.0));
    return {maxdiff, mean_power(sub) / S2_expected};
}

// A high-frequency shell deliberately includes triangles that close only modulo
// Nmesh. The reference enumerates integer wavevectors without wrapping them.
void test_high_frequency_triangle_closure() {
    constexpr int Nmesh = 16, nbins = 6, shell = 4;
    BispectrumBinning<3> b(0.0, 2.0 * M_PI * 7.9, nbins, BispectrumBinning<3>::LINEAR_SPACING);
    auto in_shell = [&](const std::array<int, 3> & k) {
        const double k2 = 4.0 * M_PI * M_PI * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
        return k2 >= b.klow[shell] * b.klow[shell] && k2 < b.khigh[shell] * b.khigh[shell];
    };
    std::vector<std::array<int, 3>> modes;
    for (int x = -7; x <= 7; x++)
        for (int y = -7; y <= 7; y++)
            for (int z = -7; z <= 7; z++)
                if (in_shell({x, y, z}))
                    modes.push_back({x, y, z});
    size_t closed = 0, wrapped = 0;
    for (const auto & a : modes)
        for (const auto & c : modes) {
            std::array<int, 3> third{-a[0] - c[0], -a[1] - c[1], -a[2] - c[2]};
            closed += in_shell(third);
            for (auto & k : third) {
                k = (k % Nmesh + Nmesh) % Nmesh;
                if (k > Nmesh / 2)
                    k -= Nmesh;
            }
            wrapped += in_shell(third);
        }
    check(closed > 0 && wrapped > closed, "high-frequency fixture distinguishes true from wrapped triangle closure");
    compute_polyspectrum_bincount<3, 3>(Nmesh, b);
    const double measured = b.get_bincount(shell, shell, shell) * std::pow(2.0 * M_PI, 3);
    check(std::abs(measured - double(closed)) < 1e-6,
          "high-frequency N123 counts exactly closed triangles: measured " + num(measured) + ", expected " +
              std::to_string(closed) + ", modulo-grid count " + std::to_string(wrapped));

    // None of the signed triples of these three wavevectors sums to zero.
    // Their all-positive sum is (16,0,0), so the original mesh spuriously
    // produces a nonzero bispectrum unless the product is de-aliased.
    Grid<3> field(Nmesh);
    field.fill_real_grid([](const std::array<double, 3> & x) {
        return std::cos(2 * M_PI * 6 * x[0]) + std::cos(2 * M_PI * (5 * x[0] + 3 * x[1])) +
               std::cos(2 * M_PI * (5 * x[0] - 3 * x[1]));
    });
    field.fftw_r2c();
    compute_bispectrum(field, b);
    const double bispectrum = b.get_spectrum(shell, shell, shell);
    check(std::abs(bispectrum) < 1e-12,
          "non-closing high-frequency plane waves have zero bispectrum (measured " + num(bispectrum) + ")");
}

void test_automatic_geometry_cache() {
    constexpr int Nmesh = 16;
    BispectrumBinning<3> geometry(0.0, 2.0 * M_PI * Nmesh / 4, 6, BispectrumBinning<3>::LINEAR_SPACING);
    compute_polyspectrum_bincount<3, 3>(Nmesh, geometry);
    check(geometry.bincount_is_set, "automatically computed geometry counts are marked reusable");

    Grid<3> field(Nmesh);
    field.fill_fourier_grid(0.0);
    field.set_grid_status_real(false);
    BispectrumBinning<3> result(0.0, 2.0 * M_PI * Nmesh / 4, 6, BispectrumBinning<3>::LINEAR_SPACING);
    compute_bispectrum(field, result);
    check(result.bincount_is_set, "bispectrum estimation retains reusable geometry counts");
}

void test_fourier_multiplicities() {
    for (int Nmesh : {16, 15}) {
        // FFTWGrid requires the mesh width to be divisible by the MPI rank count.
        if (Nmesh % FML::NTasks != 0) {
            if (FML::ThisTask == 0)
                std::cout << "[SKIP] " << Nmesh << "^3 mode counts: mesh width is not divisible by MPI tasks\n";
            continue;
        }
        Grid<3> field(Nmesh);
        for (auto index : field.get_fourier_range()) {
            const int kz = int(index % (Nmesh / 2 + 1));
            field.set_fourier_from_index(index, kz == Nmesh / 2 ? 2.0 : 1.0);
        }
        field.set_grid_status_real(false);
        // Enclose the full Fourier cube. All modes except DC contribute.
        PowerSpectrumBinning<3> power(0.0, 2.0 * M_PI * Nmesh, 10, PowerSpectrumBinning<3>::LINEAR_SPACING);
        bin_up_power_spectrum(field, power);
        std::vector<PowerSpectrumBinning<3>> multipoles(1, power);
        compute_power_spectrum_multipoles_fourier<3>(field, multipoles, {0.0, 0.0, 1.0});
        const double expected_count = Nmesh * Nmesh * Nmesh - 1;
        // The enhanced final stored plane has one copy for even meshes, two
        // conjugate copies for odd meshes. Its power is four instead of one.
        const double expected_power_sum = expected_count + 3.0 * Nmesh * Nmesh * (Nmesh % 2 == 0 ? 1 : 2);
        auto validate = [&](const PowerSpectrumBinning<3> & b, const std::string & estimator) {
            const double count = std::accumulate(b.count.begin(), b.count.end(), 0.0);
            double power_sum = 0.0;
            for (int i = 0; i < b.n; i++)
                power_sum += b.pofk[i] * b.count[i];
            const auto label = estimator + " on " + std::to_string(Nmesh) + "^3 mesh";
            check(count == expected_count,
                  label + " counts all nonzero modes: " + num(count) + " expected " + num(expected_count));
            check(std::abs(power_sum - expected_power_sum) < 1e-9,
                  label + " weights the final Fourier plane correctly: " + num(power_sum) + " expected " +
                      num(expected_power_sum));
        };
        validate(power, "P(k)");
        validate(multipoles[0], "Monopole");
    }
}

int main() {
    check(pofk_odr_check() == 1, "ComputePowerSpectrum.h can be included in several translation units");

    // Shot noise: equal masses, masses and delta-f weights
    {
        const size_t Np = 400000 / FML::NTasks * FML::NTasks;
        auto part = make_particles<Particle>(Np, 1, [](Particle &, double) {});
        auto [d0, m0] = shotnoise_test(part, Np, 1.0 / double(Np));
        check(d0 < 1e-12 and std::fabs(m0) < 0.08,
              "equal masses: shot noise 1/N subtracted, mean P(k) - 1/N = " + num(m0) + " x 1/N");

        auto pm = make_particles<ParticleMass>(Np, 2, [](ParticleMass & p, double u) { p.mass = 0.2 + 2.8 * u; });
        double s1 = 0.0, s2 = 0.0, s3 = 0.0;
        for (auto & p : pm) {
            s1 += p.mass;
            s2 += p.mass * p.mass;
            s3 += p.mass * p.mass * p.mass;
        }
        FML::SumOverTasks(&s1);
        FML::SumOverTasks(&s2);
        FML::SumOverTasks(&s3);
        const double S2 = s2 / (s1 * s1), S3 = s3 / (s1 * s1 * s1);
        auto [d1, m1] = shotnoise_test(pm, Np, S2);
        check(d1 < 1e-10 and std::fabs(m1) < 0.08,
              "masses: shot noise sum m^2/(sum m)^2 subtracted (" + num(d1) + "), mean P(k) - shot noise = " + num(m1) +
                  " x shot noise (" + num(1.0 - 1.0 / (Np * S2)) + " with 1/N)");

        auto pw = make_particles<ParticleDeltaf>(Np, 3, [](ParticleDeltaf & p, double u) { p.w = 2.0 * u - 1.0; });
        double w2 = 0.0;
        for (auto & p : pw)
            w2 += p.w * p.w;
        FML::SumOverTasks(&w2);
        auto [d2, m2] = shotnoise_test(pw, Np, w2 / double(Np) / double(Np));
        check(d2 < 1e-10 and std::fabs(m2) < 0.08,
              "delta-f weights: shot noise sum w^2/N^2 subtracted, mean P(k) - shot noise = " + num(m2) +
                  " x shot noise");

        // Bispectrum with masses: B_nosub - B_sub = S2 (P1 + P2 + P3) + S3 with P the subtracted P(k)
        const int Nmesh = 16, nbins = 4;
        BispectrumBinning<3> bsub(0.0, 2.0 * M_PI * Nmesh / 2.0, nbins, BispectrumBinning<3>::LINEAR_SPACING);
        BispectrumBinning<3> bnosub = bsub;
        bnosub.subtract_shotnoise = false;
        compute_polyspectrum<3, ParticleMass, 3>(Nmesh, pm.data(), pm.size(), Np, bsub, "CIC", false);
        compute_polyspectrum<3, ParticleMass, 3>(Nmesh, pm.data(), pm.size(), Np, bnosub, "CIC", false);
        double maxerr = 0.0;
        int nconf = 0;
        for (size_t i = 0; i < bsub.P123.size(); i++) {
            if (bsub.N123[i] <= 0.0)
                continue;
            auto ik = bsub.get_coord_from_index(i);
            const double expected = S2 * (bsub.pofk[ik[0]] + bsub.pofk[ik[1]] + bsub.pofk[ik[2]]) + S3;
            maxerr = std::max(maxerr, std::fabs((bnosub.P123[i] - bsub.P123[i]) / expected - 1.0));
            nconf++;
        }
        check(nconf > 0 and maxerr < 1e-8,
              "bispectrum with masses: S2 (P1 + P2 + P3) + S3 subtracted (" + std::to_string(nconf) +
                  " configurations)");
    }

    // NGP (and CIC) with interlacing conserves the mass: the k = 0 mode (mean delta) is 0
    for (std::string method : {"NGP", "CIC"}) {
        const size_t Np = 20000 / FML::NTasks * FML::NTasks;
        auto part = make_particles<Particle>(Np, 4, [](Particle &, double) {});
        // Some particles in the last half cell (these are shifted outside the box for the interlaced grid)
        for (size_t i = 0; i < part.size(); i += 10)
            part[i].pos[0] = std::min(FML::xmax_domain, 1.0) - 0.25 / 24.0;
        auto lr = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(method);
        Grid<3> grid(24, lr.first, lr.second + 1);
        FML::INTERPOLATION::particles_to_fourier_grid_interlacing(part.data(), part.size(), Np, grid, method);
        double dc = FML::ThisTask == 0 ? std::abs(grid.get_fourier_from_index(0)) : 0.0;
        FML::SumOverTasks(&dc);
        check(dc < 1e-12, method + " with interlacing: mean density contrast " + num(dc) + " (should be 0)");
    }

    // The number of triangles in the bispectrum bins against a brute-force count (small bins included)
    {
        const int Nmesh = 16, nbins = 6;
        const double kmax = 2.0 * M_PI * Nmesh / 4.0; // |k1 + k2| < Nmesh/2: no wrap-around
        BispectrumBinning<3> b(0.0, kmax, nbins, BispectrumBinning<3>::LINEAR_SPACING);
        compute_polyspectrum_bincount<3, 3>(Nmesh, b);
        auto bin_of = [&](const std::array<int, 3> & k) {
            const double k2 = std::pow(2.0 * M_PI, 2) * (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]);
            if (k2 == 0.0)
                return -1;
            for (int i = 0; i < nbins; i++)
                if (k2 >= b.klow[i] * b.klow[i] and k2 < b.khigh[i] * b.khigh[i])
                    return i;
            return -1;
        };
        std::vector<std::array<int, 3>> modes;
        std::vector<int> mode_bin;
        const int h = Nmesh / 4;
        for (int x = -h; x <= h; x++)
            for (int y = -h; y <= h; y++)
                for (int z = -h; z <= h; z++)
                    if (bin_of({x, y, z}) >= 0) {
                        modes.push_back({x, y, z});
                        mode_bin.push_back(bin_of({x, y, z}));
                    }
        std::vector<double> count(b.N123.size(), 0.0);
        for (size_t i = 0; i < modes.size(); i++)
            for (size_t j = 0; j < modes.size(); j++) {
                const std::array<int, 3> k3{-modes[i][0] - modes[j][0],
                                            -modes[i][1] - modes[j][1],
                                            -modes[i][2] - modes[j][2]};
                const int b3 = bin_of(k3);
                if (b3 >= 0)
                    count[b.get_index_from_coord(std::array<int, 3>{mode_bin[i], mode_bin[j], b3})] += 1.0;
            }
        double maxerr = 0.0;
        int nchecked = 0, nsmall = 0;
        for (size_t i = 0; i < count.size(); i++) {
            auto ik = b.get_coord_from_index(i);
            if (not(ik[0] <= ik[1] and ik[1] <= ik[2]))
                continue;
            const double n123 = b.N123[i] * std::pow(2.0 * M_PI, 3);
            maxerr = std::max(maxerr, std::fabs(n123 - count[i]));
            nchecked++;
            nsmall += count[i] > 0 and count[i] < 248;
        }
        check(maxerr < 1e-6,
              "N123 x (2pi)^3 = brute-force number of triangles (" + std::to_string(nchecked) + " bins, " +
                  std::to_string(nsmall) + " with < (2pi)^3 triangles), max error " + num(maxerr));
    }

    // The bin index below the first bin
    {
        PowerSpectrumBinning<3> lin(1.0, 11.0, 11, PowerSpectrumBinning<3>::LINEAR_SPACING);
        PowerSpectrumBinning<3> log(1.0, 100.0, 3, PowerSpectrumBinning<3>::LOG_SPACING);
        check(lin.get_bin_index(0.3, 1.0, 11.0, 11, lin.bin_type) == -1 and
                  lin.get_bin_index(0.6, 1.0, 11.0, 11, lin.bin_type) == 0 and
                  log.get_bin_index(0.05, 1.0, 100.0, 3, log.bin_type) == -1 and
                  log.get_bin_index(0.5, 1.0, 100.0, 3, log.bin_type) == 0,
              "P(k) bin index below the first bin is -1 (linear and log)");
    }

    // Monopole from the multipole estimator = P(k), with the same mode counts (over the MPI tasks)
    {
        const size_t Np = 50000 / FML::NTasks * FML::NTasks;
        auto part = make_particles<Particle>(Np, 5, [](Particle &, double) {});
        Grid<3> grid(32, 1, 2);
        FML::INTERPOLATION::particles_to_grid<3, Particle>(part.data(), part.size(), Np, grid, "CIC");
        grid.fftw_r2c();
        PowerSpectrumBinning<3> pofk(0.0, 2.0 * M_PI * 16, 16, PowerSpectrumBinning<3>::LINEAR_SPACING);
        pofk.subtract_shotnoise = false;
        bin_up_power_spectrum<3>(grid, pofk);
        std::vector<PowerSpectrumBinning<3>> Pell(3, pofk);
        for (auto & p : Pell)
            p.reset();
        compute_power_spectrum_multipoles_fourier<3>(grid, Pell, {0.0, 0.0, 1.0});
        double maxerr = 0.0, countdiff = 0.0;
        for (int i = 0; i < pofk.n; i++) {
            if (pofk.count[i] > 0)
                maxerr = std::max(maxerr, std::fabs(Pell[0].pofk[i] / pofk.pofk[i] - 1.0));
            countdiff = std::max(countdiff, std::fabs(Pell[0].count[i] - pofk.count[i]));
        }
        check(maxerr < 1e-10 and countdiff == 0.0, "multipoles: P_0 = P(k) and the same mode counts");
    }

    // PolyspectrumBinning
    {
        BispectrumBinning<3> b(0.1, 10.0, 4, BispectrumBinning<3>::LOG_SPACING);
        check(b.bin_type == BispectrumBinning<3>::LOG_SPACING, "PolyspectrumBinning keeps the binning type");
        std::vector<double> N123(b.N123.size(), 1.0);
        b.set_bincount(N123);
        b.reset();
        check(not b.bincount_is_set, "PolyspectrumBinning::reset() clears the bin count");
        check(b.get_reduced_spectrum({0, 0, 3}) == 0.0, "reduced bispectrum of an empty bin is 0");
    }

    test_high_frequency_triangle_closure();
    test_automatic_geometry_cache();
    test_fourier_multiplicities();

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All P(k) tests passed\n" : std::to_string(fails) + " P(k) test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
