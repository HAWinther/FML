// Field-level BAO prototype: make a mock from a known initial field and beta, then sample (s, beta) with HMC.
// Usage: ./fieldlevel [linear|1lpt] [nmesh] [kmax h/Mpc] [n_samples] [beta_true] [output file]
// With the linear model the exact marginal posterior of beta is printed and written to <output file>.exact.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <fstream>
#include <iostream>

using namespace FML::FIELDLEVEL;

int main(int argc, char ** argv) {
    FML::init_mpi(&argc, &argv);
    Settings st;
    st.model = (argc > 1 && std::string(argv[1]) == "linear") ? Model::Linear : Model::LPT1;
    st.nmesh = argc > 2 ? std::atoi(argv[2]) : 48;
    st.kmax = argc > 3 ? std::atof(argv[3]) : .14;
    st.boxsize = 1000;
    st.bias = 1.5;
    st.noise_power = 1000;
    FML::MCMC::HMCOptions opt;
    opt.warmup = 100;
    opt.samples = argc > 4 ? std::atoi(argv[4]) : 300;
    opt.n_leapfrog = 12;
    opt.target_acceptance = .65;
    opt.verbose_every = 20;
    const double beta_true = argc > 5 ? std::atof(argv[5]) : 1.02;
    const std::string output = argc > 6 ? argv[6] : "chain.txt";

    FieldLevelBAO fl(st, BAOSpectrum::eisenstein_hu());
    std::mt19937_64 rng(42);
    const auto truth = fl.white_noise(rng);
    fl.set_data(fl.make_mock(truth, beta_true, rng));
    std::cout << "Model " << (st.model == Model::Linear ? "linear" : "1LPT") << ", nmesh " << st.nmesh << ", kmax "
              << st.kmax << ", " << fl.modes().size() << " half-complex modes, beta_true " << beta_true << "\n";

    if (st.model == Model::Linear) {
        double w = 0, m1 = 0, m2 = 0, lmax = -HUGE_VAL;
        const int nb = 4001;
        std::vector<double> lnl(nb);
        for (int i = 0; i < nb; ++i)
            lmax = std::max(
                lmax,
                lnl[i] = fl.linear_marginal_log_likelihood(st.beta_min + (st.beta_max - st.beta_min) * i / (nb - 1)));
        for (int i = 0; i < nb; ++i) {
            const double b = st.beta_min + (st.beta_max - st.beta_min) * i / (nb - 1), p = std::exp(lnl[i] - lmax);
            w += p, m1 += p * b, m2 += p * b * b;
        }
        std::ofstream curve(output + ".exact");
        curve << "# beta ln_posterior (unnormalised)\n";
        for (int i = 0; i < nb; ++i)
            curve << st.beta_min + (st.beta_max - st.beta_min) * i / (nb - 1) << " " << lnl[i] - lmax << "\n";
        std::cout << "Exact marginal posterior: beta = " << m1 / w << " +- " << std::sqrt(m2 / w - m1 * m1 / w / w)
                  << "\n";
    }

    const auto chain = sample(fl, fl.white_noise(rng), 1.0, opt);
    const auto beta = FML::MCMC::recorded_column(chain, 0);
    double mean = 0, var = 0;
    for (double b : beta)
        mean += b;
    mean /= double(beta.size());
    for (double b : beta)
        var += (b - mean) * (b - mean);
    std::cout << "HMC: beta = " << mean << " +- " << std::sqrt(var / double(beta.size() - 1)) << " (acceptance "
              << chain.acceptance << ", step " << chain.step_size << ", autocorrelation time "
              << FML::MCMC::integrated_autocorrelation_time(beta) << ")\n";
    std::ofstream out(output);
    out << "# sample beta potential\n";
    for (std::size_t i = 0; i < beta.size(); ++i)
        out << i << " " << beta[i] << " " << chain.potential[i] << "\n";
}
