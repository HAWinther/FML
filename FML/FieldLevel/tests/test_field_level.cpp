// Tests of the field-level BAO prototype: adjoint gradients against finite differences (linear and 1LPT), the mock
// normalisation, HMC against the exact marginal posterior of the linear model, and recovery of beta with 1LPT.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <cstdlib>
#include <iostream>

using namespace FML::FIELDLEVEL;
static int failures = 0;
static void check(bool ok, const std::string & what) {
    std::cout << (ok ? "  ok   " : "  FAIL ") << what << "\n";
    failures += !ok;
}

static Settings small_settings(Model model) {
    Settings st;
    st.nmesh = 24;
    st.boxsize = 1000;
    st.kmax = .07;
    st.bias = 1.5;
    st.noise_power = 1500;
    st.model = model;
    return st;
}

static void test_gradient(Settings st, const std::string & name) {
    st.beta_ref = .98;
    FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());
    std::mt19937_64 rng(11);
    const auto truth = fl.white_noise(rng);
    fl.set_data(fl.make_mock(truth, 1.0, rng));
    const auto s = fl.white_noise(rng);
    const double beta = 1.03;
    Field grad;
    double grad_beta;
    fl.potential_and_gradient(s, beta, grad, grad_beta);
    const auto v = fl.white_noise(rng);
    double analytic = 0;
    for (std::size_t c = 0; c < v.size(); ++c)
        analytic += grad[c] * v[c];
    const double h = 1e-5;
    auto shifted = [&](double sign) {
        Field x = s;
        for (std::size_t c = 0; c < x.size(); ++c)
            x[c] += sign * h * v[c];
        return fl.potential(x, beta);
    };
    const double numeric = (shifted(1) - shifted(-1)) / (2 * h);
    const double hb = 1e-6;
    const double numeric_beta = (fl.potential(s, beta + hb) - fl.potential(s, beta - hb)) / (2 * hb);
    std::cout << name << ": dU.v analytic " << analytic << " numeric " << numeric << "; dU/dbeta analytic "
              << grad_beta << " numeric " << numeric_beta << " (relative differences " << std::abs(analytic / numeric - 1) << ", "
              << std::abs(grad_beta / numeric_beta - 1) << ")\n";
    check(std::abs(analytic - numeric) < 1e-5 * std::abs(numeric), name + " gradient in s");
    check(std::abs(grad_beta - numeric_beta) < 1e-4 * std::abs(numeric_beta) + 1e-6, name + " gradient in beta");
}

static void test_gradient(Model model, double centering, const std::string & name) {
    auto st = small_settings(model);
    st.centering = centering;
    test_gradient(st, name);
}

static Settings lpt_settings(int order, double b2, double bG2, double f) {
    auto st = small_settings(Model::LPT);
    st.lpt_order = order;
    st.b2 = b2;
    st.bG2 = bG2;
    st.growth_rate = f;
    return st;
}

// Model::LPT at first order with b = 1 (no bias weights) and f = 0 equals Model::LPT1 with b = 1
static void test_lpt_reduces_to_lpt1() {
    auto a = lpt_settings(1, 0, 0, 0), b = small_settings(Model::LPT1);
    a.bias = b.bias = 1;
    FieldLevelBAO fa(a, BAOSpectrum::eisenstein_hu()), fb(b, BAOSpectrum::eisenstein_hu());
    std::mt19937_64 rng(3);
    const auto x = fa.white_noise(rng);
    const auto pa = fa.predict(x, 1.01), pb = fb.predict(x, 1.01);
    double diff = 0, norm = 0;
    for (std::size_t i = 0; i < pa.size(); ++i)
        diff += std::norm(pa[i] - pb[i]), norm += std::norm(pb[i]);
    std::cout << "LPT order 1, b = 1 vs LPT1: relative difference " << std::sqrt(diff / norm) << "\n";
    check(std::sqrt(diff / norm) < 1e-12, "LPT at first order without bias equals LPT1");
}

// Two orthogonal plane waves delta = A cos(k1 x) + B cos(k2 y): Psi1 = -A sin(k1 x) / k1 xhat - ..., and 2LPT
// Psi2 = -(3/7) grad phi2 with nabla^2 phi2 = S2 = A B cos(k1 x) cos(k2 y), i.e.
// Psi2 = -(3/7) A B / (k1^2 + k2^2) (k1 sin(k1 x) cos(k2 y), k2 cos(k1 x) sin(k2 y), 0) (Crocce et al. 2006:
// x = q - grad phi1 + D2 grad phi2, D2 = -3/7, nabla^2 phi2 = sum_{i>j} phi1_ii phi1_jj - phi1_ij^2); in redshift space along z
// nothing changes (no z dependence), so the test uses waves along x and z to check the 1 + f and 1 + 2 f factors.
static void test_two_plane_waves() {
    for (double f : {0., .6}) {
        auto st = lpt_settings(2, 0, 0, f);
        FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());
        const int N = st.nmesh, n1 = 2, n2 = 3;
        const double A = .3, B = .2, k1 = 2 * M_PI * n1, k2 = 2 * M_PI * n2, beta = 1;
        Field delta(fl.ncells());
        for (int ix = 0; ix < N; ++ix)
            for (int iy = 0; iy < N; ++iy)
                for (int iz = 0; iz < N; ++iz)
                    delta[(std::size_t(ix) * N + iy) * N + iz] =
                        A * std::cos(k1 * ix / N) + B * std::cos(k2 * iz / N);
        std::vector<double> inverse(fl.modes().size());
        for (std::size_t i = 0; i < inverse.size(); ++i)
            inverse[i] = 1 / fl.effective_transfer(fl.modes()[i], beta);
        const auto psi = fl.displacement(fl.filter(delta, inverse), beta);
        double err = 0, scale = 0;
        for (int ix = 0; ix < N; ++ix)
            for (int iz = 0; iz < N; ++iz) {
                const auto p = (std::size_t(ix) * N + 1) * N + iz;
                const double sx = std::sin(k1 * ix / N), cx = std::cos(k1 * ix / N);
                const double sz = std::sin(k2 * iz / N), cz = std::cos(k2 * iz / N);
                const double c2 = 3. / 7 * A * B / (k1 * k1 + k2 * k2);
                const double ex = -A * sx / k1 - c2 * k1 * sx * cz;
                const double ez = (1 + f) * (-B * sz / k2) - (1 + 2 * f) * c2 * k2 * cx * sz;
                err = std::max({err, std::abs(psi[0][p] - ex), std::abs(psi[1][p]), std::abs(psi[2][p] - ez)});
                scale = std::max(scale, std::abs(c2 * k1));
            }
        std::cout << "two plane waves, f = " << f << ": max |Psi - Psi_exact| = " << err << " (2LPT amplitude " << scale
                  << ")\n";
        check(err < 1e-10, "1LPT + 2LPT displacement of two plane waves" + std::string(f ? " in redshift space" : ""));
    }
}

// Smooth assignment: the predicted modes equal the direct Fourier sum (1/N_p) sum_p exp(-i k.x_p) of the particles
static void test_smooth_assignment_accuracy() {
    for (int width : {4, 6, 8}) {
        auto st = small_settings(Model::LPT1);
        st.bias = 1;
        st.kernel_width = width;
        FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu(20000)); // large displacements: rms several cells
        std::mt19937_64 rng(4);
        const auto x = fl.white_noise(rng);
        const auto pred = fl.predict(x, 1.);
        const auto psi = fl.displacement(x, 1.);
        const int N = st.nmesh;
        double err = 0, norm = 0, psi2 = 0;
        for (std::size_t i = 0; i < pred.size(); i += 151) {
            const auto & m = fl.modes()[i];
            Complex direct = 0;
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz < N; ++iz) {
                        const auto p = (std::size_t(ix) * N + iy) * N + iz;
                        const double phase = -2 * M_PI *
                                             (m.n[0] * (double(ix) / N + psi[0][p]) + m.n[1] * (double(iy) / N + psi[1][p]) +
                                              m.n[2] * (double(iz) / N + psi[2][p]));
                        direct += Complex(std::cos(phase), std::sin(phase));
                    }
            direct /= double(fl.ncells());
            err += std::norm(pred[i] - direct);
            norm += std::norm(direct);
        }
        for (const auto & c : psi)
            for (double v : c)
                psi2 += v * v;
        std::cout << "smooth assignment, width " << width << ": |pred - direct sum| / |direct| = "
                  << std::sqrt(err / norm) << " (rms displacement " << std::sqrt(psi2 / fl.ncells()) * 2 * N
                  << " density cells)\n";
        check(std::sqrt(err / norm) < (width == 4 ? 1e-2 : width == 6 ? 1e-4 : 1e-5), "smooth assignment equals the direct Fourier sum");
    }
}

// A small plane wave must produce its second harmonic only at second order in the amplitude. With particles on the
// density-grid nodes CIC responds through |Psi| and the harmonic was ~40% of the fundamental at linear order.
static void test_no_linear_harmonics() {
    auto st = small_settings(Model::LPT1);
    st.assignment = Assignment::CIC;
    st.bias = 1;
    st.centering = 0;
    FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());
    const int N = st.nmesh, n1 = 3;
    for (double A : {1e-4, 1e-3}) {
        Field delta(fl.ncells());
        for (std::size_t p = 0; p < delta.size(); ++p)
            delta[p] = A * std::cos(2 * M_PI * n1 * double(p / (std::size_t(N) * N)) / N);
        std::vector<double> inverse(fl.modes().size());
        for (std::size_t i = 0; i < inverse.size(); ++i)
            inverse[i] = 1 / fl.effective_transfer(fl.modes()[i], 1.);
        const auto pred = fl.predict(fl.filter(delta, inverse), 1.);
        double fundamental = 0, harmonic = 0;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            const auto & m = fl.modes()[i];
            if (m.n[1] == 0 && m.n[2] == 0 && m.n[0] == n1)
                fundamental = std::abs(pred[i]);
            if (m.n[1] == 0 && m.n[2] == 0 && m.n[0] == 2 * n1)
                harmonic = std::abs(pred[i]);
        }
        std::cout << "plane wave A = " << A << ": |second harmonic| / |fundamental| = " << harmonic / fundamental
                  << "\n";
        check(harmonic / fundamental < 10 * A, "1LPT CIC: second harmonic is second order in the amplitude");
    }
}

// In the linear regime the prediction is (b + f mu^2) delta_lin
static void test_large_scale_amplitude() {
    auto st = lpt_settings(2, .5, -.3, .8);
    st.centering = 0; // x is the white noise, so delta_k = T x_k independently of the bias
    auto lin_settings = st;
    lin_settings.model = Model::Linear;
    lin_settings.bias = 1;
    const auto spectrum = BAOSpectrum::eisenstein_hu(.5); // tiny amplitude: rms delta ~ 1e-2
    FieldLevelBAO fl(st, spectrum), lin(lin_settings, spectrum);
    std::mt19937_64 rng(8);
    const auto x = fl.white_noise(rng);
    const auto pred = fl.predict(x, 1.), dlin = lin.predict(x, 1.);
    const int Nd = st.nmesh_density ? st.nmesh_density : 2 * st.nmesh;
    double num = 0, den = 0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        const auto & m = fl.modes()[i];
        const double window = 1; // smooth assignment is deconvolved
        (void)Nd;
        const Complex expected = fl.linear_amplitude(m.n) * window * dlin[i];
        num += m.weight * std::norm(pred[i] - expected);
        den += m.weight * std::norm(expected);
    }
    std::cout << "linear regime: |pred - (b + f mu^2) delta_lin| / |...| = " << std::sqrt(num / den) << "\n";
    check(std::sqrt(num / den) < .03, "2LPT + bias + RSD prediction approaches (b + f mu^2) delta_lin on large scales");
}

static void test_mock_normalisation() {
    FieldLevelBAO fl(small_settings(Model::Linear), BAOSpectrum::eisenstein_hu());
    std::mt19937_64 rng(5);
    double chi2 = 0, dof = 0;
    for (int r = 0; r < 20; ++r) {
        const auto d = fl.make_mock(fl.white_noise(rng), 1.0, rng);
        for (std::size_t i = 0; i < d.size(); ++i) {
            const auto & m = fl.modes()[i];
            const double C = 1.5 * 1.5 * fl.spectrum()(m.k, 1.0) / 1e9 + fl.noise_variance();
            chi2 += m.weight * std::norm(d[i]) / C;
            dof += m.weight;
        }
    }
    std::cout << "mock: sum |d|^2/C / N_modes = " << chi2 / dof << " (" << dof << " modes)\n";
    check(std::abs(chi2 / dof - 1) < 5 / std::sqrt(dof), "mock power equals (b^2 P + P_eps) / V");
}

static void moments(const std::vector<double> & x, double & mean, double & sd) {
    mean = 0;
    for (double v : x)
        mean += v;
    mean /= double(x.size());
    double var = 0;
    for (double v : x)
        var += (v - mean) * (v - mean);
    sd = std::sqrt(var / double(x.size() - 1));
}

static void test_linear_posterior() {
    auto st = small_settings(Model::Linear);
    st.beta_min = .85;
    st.beta_max = 1.15;
    FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());
    std::mt19937_64 rng(2);
    const double beta_true = 1.02;
    fl.set_data(fl.make_mock(fl.white_noise(rng), beta_true, rng));
    // Exact marginal posterior on a fine grid
    const int nb = 3001;
    std::vector<double> lnl(nb);
    double lmax = -HUGE_VAL;
    for (int i = 0; i < nb; ++i) {
        lnl[i] = fl.linear_marginal_log_likelihood(st.beta_min + (st.beta_max - st.beta_min) * i / (nb - 1));
        lmax = std::max(lmax, lnl[i]);
    }
    double w = 0, m1 = 0, m2 = 0;
    for (int i = 0; i < nb; ++i) {
        const double b = st.beta_min + (st.beta_max - st.beta_min) * i / (nb - 1), p = std::exp(lnl[i] - lmax);
        w += p;
        m1 += p * b;
        m2 += p * b * b;
    }
    const double mean_exact = m1 / w, sd_exact = std::sqrt(m2 / w - mean_exact * mean_exact);
    FML::MCMC::HMCOptions opt;
    opt.warmup = 150;
    opt.samples = 600;
    opt.n_leapfrog = 12;
    opt.target_acceptance = .65;
    opt.seed = 3;
    std::vector<FML::MCMC::HMCResult> runs;
    runs.push_back(sample(fl, fl.white_noise(rng), 1.0, opt));
    ++opt.seed;
    runs.push_back(sample(fl, fl.white_noise(rng), 1.04, opt));
    const auto diagnostic = FML::MCMC::diagnose_hmc_chains(runs);
    check(diagnostic.valid(), "linear HMC diagnostics valid");
    check(diagnostic.rhat < 1.05, "linear HMC rank-normalized split Rhat");
    check(diagnostic.ess_bulk > 50 && diagnostic.ess_tail > 30, "linear HMC bulk/tail ESS");
    std::cout << "linear diagnostics: Rhat " << diagnostic.rhat << " bulk ESS " << diagnostic.ess_bulk
              << " tail ESS " << diagnostic.ess_tail << "\n";
    const auto & chain = runs.front();
    const auto beta = FML::MCMC::recorded_column(chain, 0);
    double mean, sd;
    moments(beta, mean, sd);
    const double tau = FML::MCMC::integrated_autocorrelation_time(beta);
    const double ess = double(beta.size()) / tau;
    check(std::isfinite(tau) && ess > 0, "linear HMC nondegenerate legacy ESS");
    std::cout << "linear: exact beta = " << mean_exact << " +- " << sd_exact << "; HMC " << mean << " +- " << sd
              << " (acceptance " << chain.acceptance << ", tau " << tau << ", ESS " << ess << ")\n";
    check(std::abs(mean - mean_exact) < 4 * sd_exact / std::sqrt(ess), "linear HMC mean of beta matches exact marginal");
    check(std::abs(sd / sd_exact - 1) < 4 / std::sqrt(2 * ess) + .05, "linear HMC std of beta matches exact marginal");
    check(chain.acceptance > .4, "linear HMC acceptance");
}

static void test_lpt_recovery() {
    auto st = small_settings(Model::LPT1);
    FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());
    std::mt19937_64 rng(7);
    const double beta_true = .97;
    fl.set_data(fl.make_mock(fl.white_noise(rng), beta_true, rng));
    FML::MCMC::HMCOptions opt;
    opt.warmup = 150;
    opt.samples = 300;
    opt.n_leapfrog = 12;
    opt.target_acceptance = .65;
    opt.seed = 9;
    const auto chain = sample(fl, fl.white_noise(rng), 1.0, opt);
    const auto beta = FML::MCMC::recorded_column(chain, 0);
    double mean, sd;
    moments(beta, mean, sd);
    std::cout << "1LPT: beta_true " << beta_true << "; HMC " << mean << " +- " << sd << " (acceptance "
              << chain.acceptance << ", tau " << FML::MCMC::integrated_autocorrelation_time(beta) << ")\n";
    check(std::abs(mean - beta_true) < 3.5 * sd, "1LPT HMC recovers beta");
    check(chain.acceptance > .4, "1LPT HMC acceptance");
}

int main(int argc, char ** argv) {
    FML::init_mpi(&argc, &argv);
    test_gradient(Model::Linear, 0, "linear, white");
    test_gradient(Model::Linear, 1, "linear, centred");
    test_gradient(Model::Linear, -1, "linear, S/(S+N) centred");
    test_gradient(Model::LPT1, 0, "1LPT, white");
    test_gradient(Model::LPT1, -1, "1LPT, S/(S+N) centred");
    test_gradient(lpt_settings(1, 0, 0, 0), "LPT order 1, b1L only");
    test_gradient(lpt_settings(1, .4, -.3, .7), "LPT order 1, b2, bG2, RSD");
    test_gradient(lpt_settings(2, 0, 0, 0), "2LPT, b1L only");
    test_gradient(lpt_settings(2, .4, -.3, .7), "2LPT, b2, bG2, RSD");
    test_lpt_reduces_to_lpt1();
    test_two_plane_waves();
    test_no_linear_harmonics();
    test_smooth_assignment_accuracy();
    test_large_scale_amplitude();
    test_mock_normalisation();
    test_linear_posterior();
    test_lpt_recovery();
    std::cout << (failures ? "FAILED" : "All field-level tests passed") << "\n";
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
