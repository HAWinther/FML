// Simulation-based calibration (Talts et al. 2018) of the field-level BAO sampler. For each mock, beta_true is drawn
// from the prior, a white-noise initial field from its prior and the data from the forward model plus noise; the
// posterior is then sampled. If the model, gradients and sampler are correct, the posterior quantile of beta_true is
// uniform on [0, 1] and z = (posterior mean - beta_true) / posterior std has mean ~0 and std ~1.
// For the linear model the exact quantile is also computed from the analytic marginal posterior.
// Usage: ./calibration [linear|1lpt|2lpt] nmesh kmax n_mocks first_seed n_samples output
// 2lpt: Model::LPT, second order, Lagrangian bias b = 1.5, b2 = 0.3, bG2 = -0.2, redshift space with f = 0.7.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <fstream>
#include <iostream>

using namespace FML::FIELDLEVEL;

int main(int argc, char ** argv) {
    FML::init_mpi(&argc, &argv);
    Settings st;
    const std::string model = argc > 1 ? argv[1] : "1lpt";
    st.model = model == "linear" ? Model::Linear : model == "2lpt" ? Model::LPT : Model::LPT1;
    if (st.model == Model::LPT) {
        st.lpt_order = 2;
        st.b2 = .3;
        st.bG2 = -.2;
        st.growth_rate = .7;
    }
    st.nmesh = argc > 2 ? std::atoi(argv[2]) : 32;
    st.kmax = argc > 3 ? std::atof(argv[3]) : .095;
    const int n_mocks = argc > 4 ? std::atoi(argv[4]) : 10;
    const int first_seed = argc > 5 ? std::atoi(argv[5]) : 1;
    FML::MCMC::HMCOptions opt;
    opt.warmup = 150;
    opt.samples = argc > 6 ? std::atoi(argv[6]) : 400;
    opt.n_leapfrog = 12;
    opt.target_acceptance = .65;
    const std::string output = argc > 7 ? argv[7] : "calibration.txt";
    st.boxsize = 1000;
    st.bias = 1.5;
    st.noise_power = 1000;
    st.beta_min = .85;
    st.beta_max = 1.15;
    FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());

    std::ofstream out(output);
    out << "# seed beta_true mean std quantile tau acceptance divergences exact_mean exact_std exact_quantile\n";
    for (int m = 0; m < n_mocks; ++m) {
        const int seed = first_seed + m;
        std::mt19937_64 rng(1000003ULL * seed);
        const double beta_true = st.beta_min + (st.beta_max - st.beta_min) * std::uniform_real_distribution<double>()(rng);
        fl.set_data(fl.make_mock(fl.white_noise(rng), beta_true, rng));
        opt.seed = seed;
        const auto chain = sample(fl, fl.white_noise(rng), 1.0, opt);
        const auto beta = FML::MCMC::recorded_column(chain, 0);
        double mean = 0, var = 0, below = 0;
        for (double b : beta) {
            mean += b;
            below += b < beta_true;
        }
        mean /= double(beta.size());
        for (double b : beta)
            var += (b - mean) * (b - mean);
        const double sd = std::sqrt(var / double(beta.size() - 1));
        double em = NAN, es = NAN, eq = NAN;
        if (st.model == Model::Linear) {
            const int nb = 4001;
            std::vector<double> lnl(nb), grid(nb);
            double lmax = -HUGE_VAL;
            for (int i = 0; i < nb; ++i) {
                grid[i] = st.beta_min + (st.beta_max - st.beta_min) * i / (nb - 1);
                lmax = std::max(lmax, lnl[i] = fl.linear_marginal_log_likelihood(grid[i]));
            }
            double w = 0, m1 = 0, m2 = 0, wb = 0;
            for (int i = 0; i < nb; ++i) {
                const double p = std::exp(lnl[i] - lmax);
                w += p, m1 += p * grid[i], m2 += p * grid[i] * grid[i];
                if (grid[i] < beta_true)
                    wb += p;
            }
            em = m1 / w, es = std::sqrt(m2 / w - em * em), eq = wb / w;
        }
        out << seed << " " << beta_true << " " << mean << " " << sd << " " << below / double(beta.size()) << " "
            << FML::MCMC::integrated_autocorrelation_time(beta) << " " << chain.acceptance << " " << chain.divergences
            << " " << em << " " << es << " " << eq << std::endl;
        std::cout << "mock " << seed << ": beta_true " << beta_true << ", posterior " << mean << " +- " << sd
                  << ", z " << (mean - beta_true) / sd << std::endl;
    }
}
