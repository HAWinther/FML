//==========================================================
// Tests of the redshift-space hybrid bias spectra:
//   - compute_cross_power_spectrum_multipoles_fourier (ComputePowerSpectrum.h): the cross multipoles of a grid with
//     itself equal the auto multipoles, and for delta_2 = (1 + beta mu^2) delta_1 the cross monopole and quadrupole
//     are (1 + beta/3) P and (2 beta/3) P
//   - compute_zeldovich_hybrid_bias_multipoles (HybridBiasRedshiftSpace.h) with f = 0: the monopole equals the
//     real-space spectra of the same Zel'dovich field (compute_zeldovich_hybrid_bias_spectra)
//   - compute_hybrid_bias_multipoles with particles at x = q + Psi and velocities Psi (velocity_to_displacement = f)
//     equals compute_zeldovich_hybrid_bias_multipoles (the same redshift-space positions)
//   - the multipoles of the Zel'dovich field (f = 0.8) against the exact redshift-space Zel'dovich prediction
//     zeldovich_hybrid_bias_multipoles (HybridBiasZeldovich.h), all 15 spectra, l = 0, 2, 4
//   - the one-loop prediction (HybridBiasOneLoopRSD.h): f = 0 equals the real-space one-loop prediction
//     (HybridBiasOneLoop.h), the operator machinery with F = 1 reproduces the delta_L components of
//     LPTRedshiftSpace, the large-scale (Kaiser) limits, and the agreement with the Zel'dovich prediction at low k
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/ComputePowerSpectra/HybridBiasOneLoopRSD.h>
#include <FML/ComputePowerSpectra/HybridBiasRedshiftSpace.h>
#include <FML/ComputePowerSpectra/HybridBiasZeldovich.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/RandomFields/GaussianRandomField.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check;
using FML::TESTS::check_local;
using FML::TESTS::num;
using namespace FML::CORRELATIONFUNCTIONS;
using namespace FML::CORRELATIONFUNCTIONS::HYBRIDBIAS;
using Grid = FML::GRID::FFTWGrid<3>;
using DVector = std::vector<double>;
constexpr int NDIM = 3;

struct Particle {
    double pos[NDIM];
    double vel[NDIM];
    long long int id;
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    long long int get_id() const { return id; }
    void set_id(long long int i) { id = i; }
};

// The largest |a - b| over the bins up to nmax, relative to scale
double maxdiff(const PowerSpectrumBinning<NDIM> & a, const PowerSpectrumBinning<NDIM> & b, int nmax, double scale) {
    double d = 0.0;
    for (int i = 0; i < std::min(nmax, a.n); i++)
        d = std::max(d, std::abs(a.pofk[i] - b.pofk[i]) / scale);
    return d;
}

int main() {
    const int Npart_1D = 64, Nmesh = 128;
    const double boxsize = 500.0;
    const double kcut = M_PI * Npart_1D; // units of 1/box
    std::function<double(double)> Pofk_over_volume = [](double kBox) {
        return kBox > 0.0 ? 1e-4 * std::pow(kBox / (2.0 * M_PI), -1.5) : 0.0;
    };
    Grid delta_linear(Npart_1D);
    FML::RANDOM::RandomGenerator rng;
    rng.set_seed(1234);
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(delta_linear, &rng, Pofk_over_volume, true);
    FML::GRID::multiply_fourier_by_function_of_k<NDIM>(delta_linear, [&](double k) { return k < kcut ? 1.0 : 0.0; });

    HybridBiasOptions options;
    options.kcut = kcut;
    options.density_assignment_method = "CIC";
    options.interlacing = true;

    //==========================================================
    // 1. The cross multipole estimator
    //==========================================================
    {
        Grid g1(Nmesh);
        FML::RANDOM::RandomGenerator rng2;
        rng2.set_seed(42);
        // fixed amplitude: |delta|^2 = P for every mode, so the ratios below only depend on the mu of the modes
        FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(g1, &rng2, Pofk_over_volume, true);
        const std::vector<double> los{0.0, 0.0, 1.0};
        const int nb = Nmesh / 2;
        std::vector<PowerSpectrumBinning<NDIM>> Pa(5, PowerSpectrumBinning<NDIM>(2 * M_PI, M_PI * Nmesh, nb, PowerSpectrumBinning<NDIM>::LINEAR_SPACING));
        auto Pc = Pa;
        compute_power_spectrum_multipoles_fourier<NDIM>(g1, Pa, los);
        compute_cross_power_spectrum_multipoles_fourier<NDIM>(g1, g1, Pc, los);
        double d = 0.0;
        for (int ell = 0; ell <= 4; ell += 2)
            d = std::max(d, maxdiff(Pa[ell], Pc[ell], nb, Pa[0].pofk[nb / 2]));
        check(d < 1e-12, "cross multipoles of a grid with itself = auto multipoles (" + num(d, 3) + ")");

        // delta_2 = (1 + beta mu^2) delta_1
        const double beta = 0.7;
        Grid g2 = g1;
        FML::GRID::for_each_nonzero_fourier_mode(
            g2, [&](FML::IndexIntType index, const std::array<double, NDIM> & kvec, double kmag, double) {
                const double mu = kvec[2] / kmag;
                g2.set_fourier_from_index(index, g1.get_fourier_from_index(index) * FML::GRID::FloatType(1 + beta * mu * mu));
            });
        compute_cross_power_spectrum_multipoles_fourier<NDIM>(g1, g2, Pc, los);
        double d0 = 0.0, d2 = 0.0;
        for (int i = 8; i < nb; i++) {
            d0 = std::max(d0, std::abs(Pc[0].pofk[i] / Pa[0].pofk[i] - (1 + beta / 3)));
            d2 = std::max(d2, std::abs(Pc[2].pofk[i] / Pa[0].pofk[i] - 2 * beta / 3));
        }
        check(d0 < 0.02 and d2 < 0.02,
              "cross multipoles of delta and (1 + beta mu^2) delta: P0 / P = 1 + beta/3 (" + num(d0, 3) +
                  "), P2 / P = 2 beta / 3 (" + num(d2, 3) + ")");
    }

    //==========================================================
    // 2. f = 0: the monopole of the Zel'dovich multipoles = the real-space spectra
    //==========================================================
    {
        HybridBiasMultipoles<NDIM> m;
        compute_zeldovich_hybrid_bias_multipoles<NDIM>(delta_linear, Npart_1D, Nmesh, 0.0, options, m, 4, {}, nullptr, kcut);
        HybridBiasSpectra<NDIM> s;
        compute_zeldovich_hybrid_bias_spectra<NDIM>(delta_linear, Npart_1D, Nmesh, options, s, nullptr, kcut);
        double d = 0.0;
        for (size_t p = 0; p < s.pairs.size(); p++) {
            const double scale = std::sqrt(std::abs(s.get(s.pairs[p][0], s.pairs[p][0]).pofk[10] *
                                                    s.get(s.pairs[p][1], s.pairs[p][1]).pofk[10]));
            d = std::max(d, maxdiff(m.pofk[p][0], s.pofk[p], Nmesh / 2, scale));
        }
        check(d < 1e-10, "f = 0: monopole of the redshift-space multipoles = real-space spectra (" + num(d, 3) + ")");
    }

    //==========================================================
    // 3. Particles with velocities = the Zel'dovich field in redshift space
    //==========================================================
    const double f = 0.8;
    HybridBiasMultipoles<NDIM> zel;
    compute_zeldovich_hybrid_bias_multipoles<NDIM>(delta_linear, Npart_1D, Nmesh, f, options, zel, 4, {}, nullptr, kcut);
    {
        FML::PARTICLE::MPIParticles<Particle> part;
        part.create_particle_grid(Npart_1D, 1.5);
        long long int npart_local = part.get_npart();
        auto npart_per_task = FML::GatherFromTasks(&npart_local);
        long long int id = 0;
        for (int t = 0; t < FML::ThisTask; t++)
            id += npart_per_task[t];
        for (auto & p : part)
            p.set_id(id++);
        // The displacement on the lattice: Psi(k) = i k / k^2 delta_L(k)
        std::array<Grid, NDIM> Psi;
        {
            Grid phi;
            FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier<NDIM>(delta_linear, phi);
            FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<NDIM>(phi, Psi);
        }
        for (auto & p : part) {
            const long long int i = p.get_id();
            std::array<int, NDIM> coord{int(i / Npart_1D / Npart_1D), int((i / Npart_1D) % Npart_1D), int(i % Npart_1D)};
            coord[0] -= int(Psi[0].get_local_x_start());
            const auto index = Psi[0].get_index_real(coord);
            for (int idim = 0; idim < NDIM; idim++) {
                p.vel[idim] = Psi[idim].get_real_from_index(index);
                p.pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(p.pos[idim] + p.vel[idim]);
            }
        }
        part.communicate_particles();
        HybridBiasMultipoles<NDIM> m;
        compute_hybrid_bias_multipoles<NDIM>(part.get_particles_ptr(),
                                             part.get_npart(),
                                             part.get_npart_total(),
                                             delta_linear,
                                             Npart_1D,
                                             Nmesh,
                                             f,
                                             options,
                                             m);
        double d = 0.0;
        for (size_t p = 0; p < m.pairs.size(); p++) {
            const double scale = std::sqrt(std::abs(zel.pofk[m.index(m.pairs[p][0], m.pairs[p][0])][0].pofk[10] *
                                                    zel.pofk[m.index(m.pairs[p][1], m.pairs[p][1])][0].pofk[10]));
            for (int ell = 0; ell <= 4; ell += 2)
                d = std::max(d, maxdiff(m.pofk[p][ell], zel.pofk[p][ell], Nmesh / 2, scale));
        }
        check(d < 1e-5, "particles with velocities = Zel'dovich field in redshift space, 15 spectra, l = 0, 2, 4 (" +
                            num(d, 3) + ")");
    }

    //==========================================================
    // 4. The Zel'dovich field against the redshift-space Zel'dovich prediction. The prediction is averaged over the
    //    grid modes of each bin as the estimator does: (2l+1) <P(k_m, nu_m) L_l(nu_m)>_modes with P(k, nu) from the
    //    predicted multipoles up to l = 8 (the modes of a bin are not uniform in nu, which mixes the multipoles at
    //    low k). Fixed-amplitude initial conditions, so the measured spectra are close to the mean
    //==========================================================
    zel.scale(boxsize);
    {
        auto pofk = [&](double k) { return Pofk_over_volume(k * boxsize) * std::pow(boxsize, 3); };
        const double kfund = 2.0 * M_PI / boxsize;
        auto W = [&](double k) { return (k * boxsize < kcut and k >= kfund) ? 1.0 : 0.0; };
        ZeldovichOptions zo;
        zo.kmin_integration = kfund;
        zo.kmax_integration = kcut / boxsize;
        const double kmax_test = 0.15;
        // The prediction on a fine k-grid, l <= 8
        DVector kg;
        for (int i = 0; i < 48; i++)
            kg.push_back(0.9 * kfund * std::pow(1.05 * kmax_test / (0.9 * kfund), i / 47.0));
        std::vector<std::array<int, 2>> pairs;
        const int lmax = 8;
        const auto th = zeldovich_hybrid_bias_multipoles(pofk, W, W, kg, f, lmax, zo, 12, nullptr, &pairs);
        auto th_at = [&](int p, int ell, double k) {
            size_t i = size_t(std::upper_bound(kg.begin(), kg.end(), k) - kg.begin());
            i = std::max<size_t>(1, std::min(i, kg.size() - 1));
            const double t = (k - kg[i - 1]) / (kg[i] - kg[i - 1]);
            return th[p][ell][i - 1] * (1 - t) + th[p][ell][i] * t;
        };
        // The modes in the bins below kmax_test (the same binning as the measurement, in box units)
        auto binning = zel.pofk[0][0];
        const int nbins = binning.n;
        const double kmin_b = binning.kmin, kmax_b = binning.kmax; // box units (scale() does not change them)
        const int M = int(std::ceil(kmax_test * boxsize / (2 * M_PI))) + 1;
        std::vector<std::vector<std::array<double, 2>>> modes(nbins); // (k in h/Mpc, nu)
        for (int mx = -M; mx <= M; mx++)
            for (int my = -M; my <= M; my++)
                for (int mz = -M; mz <= M; mz++) {
                    const double m2 = double(mx) * mx + double(my) * my + double(mz) * mz;
                    if (m2 == 0)
                        continue;
                    const double kbox = 2 * M_PI * std::sqrt(m2);
                    if (kbox / boxsize > kmax_test)
                        continue;
                    const int ib = binning.get_bin_index(kbox, kmin_b, kmax_b, nbins, binning.bin_type);
                    if (ib >= 0 and ib < nbins)
                        modes[ib].push_back({kbox / boxsize, mz / std::sqrt(m2)});
                }
        std::vector<int> bins;
        for (int ib = 1; ib < nbins; ib++) {
            // only bins whose modes are all below kmax_test
            if (modes[ib].empty() or binning.kbin[ib] > kmax_test - 2 * kfund)
                continue;
            bins.push_back(ib);
        }
        auto predicted = [&](int p, int ell, int ib) {
            double sum = 0.0;
            for (auto & m : modes[ib]) {
                double pk = 0.0;
                for (int l = 0; l <= lmax; l += 2)
                    pk += th_at(p, l, m[0]) * FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::detail::legendre(l, m[1]);
                sum += pk * FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::detail::legendre(ell, m[1]);
            }
            return (2 * ell + 1) * sum / double(modes[ib].size());
        };
        // The deviation in units of the Gaussian error of one line of sight, sigma^2 = (2l+1) (P_aa P_bb + P_ab^2) /
        // N_modes (an upper bound here: the amplitudes are fixed, so only the non-linear part of the field scatters)
        double worst[3] = {0, 0, 0}, chi2 = 0.0;
        int nchi2 = 0;
        for (size_t p = 0; p < pairs.size(); p++) {
            const int a = pairs[p][0], b = pairs[p][1];
            for (int ib : bins) {
                const double Paa = predicted(zel.index(a, a), 0, ib), Pbb = predicted(zel.index(b, b), 0, ib);
                const double Pab = predicted(int(p), 0, ib);
                for (int ell = 0; ell <= 4; ell += 2) {
                    const double sigma =
                        std::sqrt((2 * ell + 1) * (std::abs(Paa * Pbb) + Pab * Pab) / double(modes[ib].size() / 2));
                    const double dev = std::abs(zel.pofk[p][ell].pofk[ib] - predicted(int(p), ell, ib)) / sigma;
                    worst[ell / 2] = std::max(worst[ell / 2], dev);
                    chi2 += dev * dev;
                    nchi2++;
                }
            }
        }
        chi2 /= nchi2;
        if (FML::ThisTask == 0) {
            std::cout << "Zel'dovich field vs prediction (f = 0.8), k < " << kmax_test
                      << ": largest |deviation| / sigma_Gaussian l=0 " << worst[0] << " l=2 " << worst[1] << " l=4 "
                      << worst[2] << ", mean (deviation / sigma)^2 = " << chi2 << "\n";
            std::cout << "  k      P_11^0 (theory)    P_11^2 (theory)    P_11^4 (theory)    P_s2s2^0 (theory)    P_1n^2 (theory)\n";
            for (int ib : bins)
                std::cout << "  " << binning.kbin[ib] << "  " << zel.get(0, 0, 0).pofk[ib] << " (" << predicted(0, 0, ib)
                          << ")  " << zel.get(0, 0, 2).pofk[ib] << " (" << predicted(0, 2, ib) << ")  "
                          << zel.get(0, 0, 4).pofk[ib] << " (" << predicted(0, 4, ib) << ")  "
                          << zel.get(3, 3, 0).pofk[ib] << " (" << predicted(zel.index(3, 3), 0, ib) << ")  "
                          << zel.get(0, 4, 2).pofk[ib] << " (" << predicted(zel.index(0, 4), 2, ib) << ")\n";
        }
        check(chi2 < 0.5 and std::max(worst[0], std::max(worst[1], worst[2])) < 3.0,
              "Zel'dovich field in redshift space = Zel'dovich prediction (15 spectra, l = 0, 2, 4, k < 0.15 h/Mpc; "
              "mean (deviation / sigma)^2 = " + num(chi2, 3) + ")");
    }

    //==========================================================
    // 5. The one-loop prediction
    //==========================================================
    if (FML::ThisTask == 0) {
        // The linear P(k) of the LPT tests (velocileptors' reference)
        DVector k, p;
        {
            std::ifstream in("../PerturbationTheory/tests/reference/lptrsd_linear_pk.txt");
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() or line[0] == '#')
                    continue;
                std::istringstream s(line);
                double a, b;
                s >> a >> b;
                k.push_back(a);
                p.push_back(b);
            }
        }
        check_local(k.size() > 10, "linear P(k) of the LPT tests found");
        FML::PERTURBATION::LPTRedshiftSpace::Options o;
        o.cutoff = 1.0;
        HybridBiasOneLoopRSD hb(k, p, o);
        auto & lpt = hb.lpt_model();
        const FML::PERTURBATION::detail::LogInterp P(lpt.k_internal(), lpt.p_internal());

        // 5a. The operator machinery with F = 1 gives the delta_L components
        {
            const auto & kint = lpt.k_internal();
            const auto & pint = lpt.p_internal();
            DVector logk(kint.size());
            for (size_t i = 0; i < kint.size(); i++)
                logk[i] = std::log(kint[i]);
            const double kmin = kint.front(), kmax = kint.back();
            std::function<double(double)> Pf = [&](double kk) {
                return (kk < kmin or kk > kmax) ? 0.0 : FML::PERTURBATION::detail::interp_clamped(std::log(kk), logk, pint);
            };
            std::function<double(double)> F1 = [](double) { return 1.0; };
            DVector q5(kint.size()), r1(kint.size()), r2(kint.size());
            for (size_t i = 0; i < kint.size(); i++) {
                const auto L = FML::CORRELATIONFUNCTIONS::HYBRIDBIAS::detail::loop_spectra(kint[i] * (i + 1 == kint.size() ? 1 - 1e-12 : 1.0), Pf, kmin, kmax, F1);
                q5[i] = L.Q5F;
                r1[i] = L.R1F;
                r2[i] = L.R2F;
            }
            auto oq = lpt.operator_qfunctions(F1, q5, r1, r2);
            const DVector kv{0.02, 0.05, 0.1, 0.2};
            double w = 0.0;
            for (double ff : {0.0, 0.8})
                for (double nu : {0.2, 0.9}) {
                    const auto c = lpt.pk_components(ff, nu, kv);
                    const auto oc = lpt.operator_components(oq, ff, nu, kv);
                    for (size_t i = 0; i < kv.size(); i++) {
                        const double ref[5] = {0.5 * c[i][1], c[i][2], c[i][4], 0.5 * c[i][7], c[i][2]};
                        // (O, O) for F = 1 has the k_ref constant removed (P_dd does not go to zero): compare (O, O)
                        // with P_dd minus its value at k_ref
                        for (int j = 0; j < 4; j++)
                            w = std::max(w, std::abs(oc[i][j] - ref[j]) / std::abs(c[i][2]));
                    }
                }
            check_local(w < 3e-3,
                        "operator spectra with F = 1 = the delta_L components of LPTRedshiftSpace (" + num(w, 3) + ")");
        }

        // 5b. f = 0 against the real-space one-loop prediction
        {
            const DVector kv{0.05, 0.1, 0.2};
            ZeldovichOptions zo;
            zo.kmin_integration = 1e-4;
            zo.kmax_integration = 4.0;
            std::vector<std::array<int, 2>> pairs;
            const auto rs = oneloop_hybrid_bias_spectra([&](double kk) { return P(kk); },
                                                        [](double) { return 1.0; },
                                                        kv,
                                                        zo,
                                                        nullptr,
                                                        &pairs);
            const auto mine = hb.pkmu(0.0, 0.5, kv);
            double w = 0.0, wn = 0.0;
            for (size_t ip = 0; ip < pairs.size(); ip++) {
                const int a = pairs[ip][0], b = pairs[ip][1];
                // (delta^2, nabla^2 delta) and (s^2, nabla^2 delta): the real-space prediction keeps the two-loop
                // Zel'dovich terms (D^2 D_L, S:S D_L), not compared
                if (b == NABLA2DELTA and (a == DELTA2 or a == S2))
                    continue;
                for (size_t i = 0; i < kv.size(); i++) {
                    const double scale = std::sqrt(std::abs(rs[ip][i] * rs[ip][i]));
                    const double dev = std::abs(mine[ip][i] - rs[ip][i]) / scale;
                    (b == NABLA2DELTA ? wn : w) = std::max(b == NABLA2DELTA ? wn : w, dev);
                }
            }
            check_local(w < 3e-3, "f = 0: one-loop = real-space one-loop, 10 spectra without nabla^2 delta (" + num(w, 3) + ")");
            check_local(wn < 5e-3,
                        "f = 0: one-loop = real-space one-loop, (1,n), (d,n), (n,n) (" + num(wn, 3) + ")");
        }

        // 5c. Without the loop terms the components that are exact at Zel'dovich order equal the redshift-space
        // Zel'dovich prediction (independent calculations: hypergeometric series + FFTLog against the q-integral
        // with J_0): (1,1), (1,d), (d,d), (1,d2), (1,s2), (1,n), (d,n), (n,n)
        {
            FML::PERTURBATION::LPTRedshiftSpace::Options oz = o;
            oz.one_loop = false;
            HybridBiasOneLoopRSD hz(k, p, oz);
            const double ff = 0.8;
            const DVector kv{0.05, 0.1, 0.2};
            ZeldovichOptions zo;
            zo.kmin_integration = 1e-4;
            zo.kmax_integration = 4.0;
            double w = 0.0;
            for (double nu : {0.3, 0.9}) {
                const auto lz = hz.pkmu(ff, nu, kv);
                const auto za = zeldovich_hybrid_bias_pkmu([&](double kk) { return P(kk); },
                                                           [](double) { return 1.0; },
                                                           [](double) { return 1.0; },
                                                           kv,
                                                           ff,
                                                           nu,
                                                           zo);
                const auto pr = HybridBiasOneLoopRSD::pairs();
                for (size_t ip = 0; ip < pr.size(); ip++) {
                    const int a = pr[ip][0], b = pr[ip][1];
                    const bool exact = (a == ONE and b != NABLA2DELTA) or (a == DELTA and b == DELTA) or b == NABLA2DELTA;
                    if (not exact or (b == NABLA2DELTA and (a == DELTA2 or a == S2)))
                        continue;
                    for (size_t i = 0; i < kv.size(); i++) {
                        const double scale = std::abs(za[ip][i]);
                        w = std::max(w, std::abs(lz[ip][i] - za[ip][i]) / scale);
                    }
                }
            }
            check_local(w < 3e-3, "one-loop code without loops = redshift-space Zel'dovich (f = 0.8; 8 spectra, " +
                                      num(w, 3) + ")");
        }
    }
    return FML::TESTS::summary("hybrid bias in redshift space");
}
