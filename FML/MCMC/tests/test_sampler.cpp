// =============================================================================
// Tests of the MCMC sampler (Sampler.h). Run with `make run` in this folder.
//
//   * Cholesky factorization
//   * A correlated 3D Gaussian (means, covariance, R-1, acceptance), a flat bounded prior (uniform moments), a
//     Gaussian prior, rejection of points where the likelihood fails
//   * GetDist output, fast/slow sampling, checkpoint and resume
// =============================================================================

#include "TestHelpers.h"
#include <FML/MCMC/Sampler.h>

#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <sstream>

using namespace FML::MCMC;

using Test::check;
using Test::check_close;
using Test::check_rel;

namespace {
    SamplerOptions quiet(int chains = 4, int burn = 2000, int samples = 20000) {
        SamplerOptions o;
        o.chains = chains;
        o.burn_in = burn;
        o.min_samples = samples;
        o.max_steps = burn + samples;
        o.check_every = samples;
        o.verbose = false;
        o.Rminus1_stop = 1.0; // run exactly min_samples
        return o;
    }
    SampledParameter P(const std::string & n, double start, double lo, double hi, double prop) {
        SampledParameter p;
        p.name = n;
        p.start = start;
        p.min = lo;
        p.max = hi;
        p.proposal = prop;
        return p;
    }
} // namespace

int main() {
    std::cout << std::unitbuf;

    std::cout << "Cholesky\n";
    {
        std::vector<std::vector<double>> A = {{4, 2, 0.4}, {2, 5, 1}, {0.4, 1, 3}}, L;
        check("positive definite matrix factorized", FML::MCMC::cholesky(A, L));
        double e = 0.0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                double s = 0.0;
                for (int k = 0; k < 3; k++)
                    s += L[i][k] * L[j][k];
                e = std::max(e, std::fabs(s - A[i][j]));
            }
        check_close("L L^T = A", e, 0.0, 1e-14);
        check("indefinite matrix rejected", !FML::MCMC::cholesky({{1, 2}, {2, 1}}, L));
    }

    std::cout << "Correlated 3D Gaussian\n";
    {
        const std::vector<double> mu = {1.0, -2.0, 0.5}, sig = {0.5, 2.0, 0.1};
        const double rho[3][3] = {{1, 0.8, -0.3}, {0.8, 1, 0.1}, {-0.3, 0.1, 1}};
        std::vector<std::vector<double>> C(3, std::vector<double>(3)), Ci;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                C[i][j] = rho[i][j] * sig[i] * sig[j];
        // inverse via Cholesky
        std::vector<std::vector<double>> L;
        FML::MCMC::cholesky(C, L);
        Ci.assign(3, std::vector<double>(3, 0.0));
        for (int col = 0; col < 3; col++) {
            std::vector<double> e(3, 0.0), y(3), x(3);
            e[col] = 1.0;
            for (int i = 0; i < 3; i++) {
                double s = e[i];
                for (int k = 0; k < i; k++)
                    s -= L[i][k] * y[k];
                y[i] = s / L[i][i];
            }
            for (int i = 2; i >= 0; i--) {
                double s = y[i];
                for (int k = i + 1; k < 3; k++)
                    s -= L[k][i] * x[k];
                x[i] = s / L[i][i];
            }
            for (int i = 0; i < 3; i++)
                Ci[i][col] = x[i];
        }
        LikelihoodFunction like = [&](const std::vector<double> & x) {
            LikelihoodResult r;
            double c2 = 0.0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    c2 += (x[i] - mu[i]) * Ci[i][j] * (x[j] - mu[j]);
            r.loglike = -0.5 * c2;
            r.derived = {x[0] + x[1]};
            return r;
        };
        // deliberately poor initial proposal and start: the adaptation has to find the covariance
        Sampler mc({P("a", 0.0, -10, 10, 1.0), P("b", 0.0, -20, 20, 1.0), P("c", 0.0, -5, 5, 1.0)},
                   {"a_plus_b"},
                   {"a+b"},
                   like,
                   quiet(4, 4000, 30000));
        mc.run();
        const auto diagnostic = mc.diagnostics(3); // includes derived a+b and rejected-step repeats
        check("modern sampler diagnostics valid", diagnostic.valid());
        check("modern sampler rank Rhat", diagnostic.rhat < 1.02);
        check("modern sampler bulk/tail ESS", diagnostic.ess_bulk > 400 && diagnostic.ess_tail > 400);
        std::vector<std::vector<double>> traces(4);
        for (int i = 0; i < 4; ++i) for (const auto & row : mc.samples(i)) traces[i].push_back(row[3]);
        check_close("sampler adapter preserves full traces", diagnostic.ess_bulk, diagnose_chains(traces).ess_bulk, 1e-12);
        bool invalid_column = false;
        try { mc.diagnostics(4); } catch (const std::out_of_range &) { invalid_column = true; }
        check("sampler diagnostic invalid column", invalid_column);
        const auto m = mc.mean(), s = mc.stddev();
        const auto cv = mc.covariance();
        double em = 0.0, ec = 0.0;
        for (int i = 0; i < 3; i++) {
            em = std::max(em, std::fabs(m[i] - mu[i]) / sig[i]);
            for (int j = 0; j < 3; j++)
                ec = std::max(ec, std::fabs(cv[i][j] - C[i][j]) / (sig[i] * sig[j]));
        }
        std::cout << "  acceptance " << mc.acceptance() << ", R-1 = " << mc.Rminus1() << "\n";
        check_close("means (in units of sigma)", em, 0.0, 0.05);
        check_close("covariance (in units of sigma_i sigma_j)", ec, 0.0, 0.05);
        const double sab = std::sqrt(C[0][0] + C[1][1] + 2 * C[0][1]);
        check_rel("derived a + b: standard deviation", s[3], sab, 0.05);
        check("R-1 < 0.01", mc.Rminus1() < 0.01);
        check("acceptance near 0.25-0.35 (adapted)", mc.acceptance() > 0.15 && mc.acceptance() < 0.45);
        check_close("best fit near the peak", std::fabs(mc.best_fit()[0] - mu[0]) / sig[0], 0.0, 0.3);
    }

    std::cout << "Priors\n";
    {
        LikelihoodFunction flat = [](const std::vector<double> &) {
            LikelihoodResult r;
            r.loglike = 0.0;
            return r;
        };
        Sampler u({P("u", 0.5, 0.0, 1.0, 0.2)}, {}, {}, flat, quiet(4, 1000, 40000));
        u.run();
        check_close("flat prior on [0, 1]: mean 1/2", u.mean()[0], 0.5, 0.01);
        check_rel("flat prior: variance 1/12", u.stddev()[0] * u.stddev()[0], 1.0 / 12.0, 0.03);
        auto g = P("g", 0.0, -100, 100, 1.0);
        g.gaussian_prior = true;
        g.prior_mean = 3.0;
        g.prior_sigma = 0.7;
        Sampler gm({g}, {}, {}, flat, quiet(4, 1000, 40000));
        gm.run();
        check_close("Gaussian prior: mean", gm.mean()[0], 3.0, 0.03);
        check_rel("Gaussian prior: sigma", gm.stddev()[0], 0.7, 0.03);
        // likelihood that fails for x < 0: no sample there, and the run does not stop
        LikelihoodFunction half = [](const std::vector<double> & x) {
            if (x[0] < 0.0)
                throw std::runtime_error("unphysical");
            LikelihoodResult r;
            r.loglike = -0.5 * x[0] * x[0];
            return r;
        };
        Sampler hm({P("x", 0.5, -5, 5, 0.5)}, {}, {}, half, quiet(2, 1000, 20000));
        hm.run();
        double xmin = 1e9;
        for (int c = 0; c < 2; c++)
            for (const auto & r : hm.samples(c))
                xmin = std::min(xmin, r[0]);
        check("failed evaluations are rejected (no sample with x < 0)", xmin >= 0.0);
        check_close("half-normal mean sqrt(2/pi)", hm.mean()[0], std::sqrt(2.0 / M_PI), 0.03);
    }

    std::cout << "GetDist output\n";
    {
        const std::string dir = "data/test_mcmc";
        std::filesystem::create_directories(dir);
        LikelihoodFunction g = [](const std::vector<double> & x) {
            LikelihoodResult r;
            r.loglike = -0.5 * x[0] * x[0];
            r.derived = {2 * x[0]};
            return r;
        };
        auto o = quiet(2, 500, 3000);
        o.output_root = dir + "/toy";
        Sampler mc({P("x", 0.0, -10, 10, 1.0)}, {"twox"}, {"2x"}, g, o);
        mc.run();
        double w = 0.0;
        int cols = 0;
        std::ifstream f(dir + "/toy_1.txt");
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream is(line);
            double v;
            std::vector<double> row;
            while (is >> v)
                row.push_back(v);
            cols = int(row.size());
            w += row[0];
            if (std::fabs(row[3] - 2.0 * row[2]) > 1e-8)
                cols = -1;
        }
        check("chain file: weight, -ln posterior, x, derived", cols == 4);
        check_close("weights sum to the number of steps", w, 3000.0, 1e-12);
        check("paramnames, ranges, covmat, summary written",
              std::filesystem::exists(dir + "/toy.paramnames") && std::filesystem::exists(dir + "/toy.ranges") &&
                  std::filesystem::exists(dir + "/toy.covmat") && std::filesystem::exists(dir + "/toy.summary.txt"));
    }

    std::cout << "Fast/slow sampling\n";
    {
        // 2D Gaussian, correlation 0.6; b is fast. The likelihood counts the calls whose slow value is not one of the
        // last two (= theory recomputations with a two-entry cache, as in ./mcmc)
        const double rho = 0.6;
        long slow_calls = 0, calls = 0;
        double last_a[2] = {1e300, 1e300};
        LikelihoodFunction like = [&](const std::vector<double> & x) {
            calls++;
            if (x[0] != last_a[0] && x[0] != last_a[1]) {
                slow_calls++;
                last_a[1] = last_a[0];
                last_a[0] = x[0];
            } else if (x[0] == last_a[1])
                std::swap(last_a[0], last_a[1]);
            LikelihoodResult r;
            const double a = x[0], b = (x[1] - 1.0) / 0.5;
            r.loglike = -0.5 * (a * a - 2.0 * rho * a * b + b * b) / (1.0 - rho * rho);
            return r;
        };
        auto pa = P("a", 0.0, -10, 10, 1.0), pb = P("b", 1.0, -10, 10, 0.5);
        pb.fast = true;
        SamplerOptions o = quiet(4, 2000, 20000);
        o.threads = 1;
        o.oversample_fast = 4;
        Sampler mc({pa, pb}, {}, {}, like, o);
        mc.run();
        const auto m = mc.mean(), sd = mc.stddev();
        const auto cv = mc.covariance();
        std::cout << "  acceptance " << mc.acceptance() << ", " << calls << " calls, " << slow_calls
                  << " with a new slow value\n";
        check_close("mean of the slow parameter", m[0], 0.0, 0.05);
        check_close("mean of the fast parameter", m[1], 1.0, 0.025);
        check_rel("sigma of the fast parameter", sd[1], 0.5, 0.03);
        check_close("correlation", cv[0][1] / (sd[0] * sd[1]), rho, 0.03);
        check("4 fast calls per slow step: ~5 calls per step", calls > 4.5 * 4 * 22000 && calls < 5.5 * 4 * 22000);
        check("fast updates reuse the slow part: one new slow value per step", slow_calls < 1.01 * 4 * 22000 + 100);
        bool threw = false;
        try {
            Sampler({pb, pa}, {}, {}, like, o);
        } catch (const std::exception &) {
            threw = true;
        }
        check("fast parameters must come last", threw);
    }

    std::cout << "Checkpoint and resume\n";
    {
        const std::string dir = "test_output_mcmc_ckpt";
        std::filesystem::create_directories(dir);
        LikelihoodFunction like = [](const std::vector<double> & x) {
            LikelihoodResult r;
            r.loglike = -0.5 * (x[0] * x[0] + std::pow((x[1] - 2.0) / 3.0, 2));
            r.derived = {x[0] * x[1]};
            return r;
        };
        auto run = [&](const std::string & root, long max_steps, bool resume) {
            SamplerOptions o = quiet(3, 400, 1200);
            o.check_every = 200;
            o.max_steps = max_steps;
            o.output_root = root;
            o.resume = resume;
            o.seed = 7;
            Sampler mc({P("u", 0.0, -10, 10, 0.5), P("v", 2.0, -20, 20, 1.0)}, {"uv"}, {"uv"}, like, o);
            mc.run();
            return mc.samples(0);
        };
        const auto full = run(dir + "/full", 1600, false);
        run(dir + "/part", 1000, false); // "interrupted" after 600 samples per chain
        check("checkpoint written", std::filesystem::exists(dir + "/part_2.checkpoint"));
        const auto resumed = run(dir + "/part", 1600, true);
        bool same = full.size() == resumed.size();
        for (size_t i = 0; same && i < full.size(); i++)
            for (size_t j = 0; j < full[i].size(); j++)
                same = same && std::fabs(full[i][j] - resumed[i][j]) <= 1e-10 * (1.0 + std::fabs(full[i][j]));
        check("resumed run = uninterrupted run (" + std::to_string(full.size()) + " samples of chain 1)", same);
        check("resume without checkpoints starts afresh", run(dir + "/new", 1600, true).size() == full.size());
        std::filesystem::remove_all(dir);
    }

    {
        LikelihoodFunction flat = [](const std::vector<double> &) { LikelihoodResult r; r.loglike = 0; return r; };
        auto options = quiet(2, 0, 16);
        options.max_steps = 32;
        Sampler frozen({P("frozen", .1, 0., 2., 1e-100)}, {}, {}, flat, options);
        frozen.run();
        check("legacy stopping rejects zero within-chain variance", std::isinf(frozen.Rminus1()));
        check("frozen sampler reaches step cap", frozen.n_samples() == 64);
        check("modern frozen sampler status", frozen.diagnostics(0).status == DiagnosticStatus::ConstantChain);
    }
    return Test::summary();
}
