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

static void test_gradient(Model model, double centering, const std::string & name) {
    auto st = small_settings(model);
    st.centering = centering;
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
    test_mock_normalisation();
    test_linear_posterior();
    test_lpt_recovery();
    std::cout << (failures ? "FAILED" : "All field-level tests passed") << "\n";
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
