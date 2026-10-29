#include "TestHelpers.h"
#include "DiagnosticsReference.h"
#include <FML/MCMC/HMC.h>
#include <random>
using namespace FML::MCMC;
using Chains = std::vector<std::vector<double>>;
int main() {
    for (const auto & fixture : diagnostic_fixtures()) {
        const auto d = diagnose_chains(fixture.chains);
        Test::check(std::string("reference status ") + fixture.name, d.valid());
        const std::array<double, 7> values =
            {d.rhat, d.rhat_bulk, d.rhat_folded, d.ess_bulk, d.ess_tail, d.ess_lower, d.ess_upper};
        for (std::size_t j = 0; j < values.size(); ++j)
            Test::check_close(std::string("ArviZ ") + fixture.name + " metric " + std::to_string(j),
                              values[j],
                              fixture.expected[j],
                              2e-8);
    }
    std::mt19937_64 rng(17);
    std::normal_distribution<double> normal;
    Chains x(4, std::vector<double>(2048));
    for (auto & c : x)
        for (auto & v : c)
            v = normal(rng);
    const auto iid = diagnose_chains(x);
    Test::check("IID chains finite", iid.valid());
    Test::check("IID Rhat near one", std::abs(iid.rhat - 1) < .02);
    Test::check("IID bulk/tail ESS", iid.ess_bulk > 3000 && iid.ess_tail > 3000);
    auto shifted = x, scaled = x, correlated = x;
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (auto & v : shifted[i])
            v += 3 * i;
        for (auto & v : scaled[i])
            v *= std::pow(4., i);
        for (std::size_t j = 1; j < x[i].size(); ++j)
            correlated[i][j] += .9 * correlated[i][j - 1];
    }
    Test::check("location mismatch detected", diagnose_chains(shifted).rhat > 1.2);
    const auto scales = diagnose_chains(scaled);
    Test::check("folding detects scale mismatch", scales.rhat_folded > 1.2 && scales.rhat_bulk < 1.02);
    Test::check("correlation reduces ESS", diagnose_chains(correlated).ess_bulk < iid.ess_bulk / 5);
    auto affine = x;
    for (auto & c : affine)
        for (auto & v : c)
            v = 3 * v + 2;
    Test::check_close("bulk rank affine invariance", diagnose_chains(affine).rhat_bulk, iid.rhat_bulk, 1e-12);
    Test::check_close("ESS affine invariance", diagnose_chains(affine).ess_bulk, iid.ess_bulk, 1e-10);
    // Check the FFT autocovariance against independent direct lag products.
    const std::vector<double> seq = {1, 2, -3, 7, -2, 4, 5};
    const auto ac = diagnostic_detail::autocovariance(seq);
    const double mean = std::accumulate(seq.begin(), seq.end(), 0.) / seq.size();
    for (std::size_t lag = 0; lag < seq.size(); ++lag) {
        double direct = 0;
        for (std::size_t j = 0; j + lag < seq.size(); ++j)
            direct += (seq[j] - mean) * (seq[j + lag] - mean) / seq.size();
        Test::check_close("FFT autocovariance lag " + std::to_string(lag), ac[lag], direct, 1e-12);
    }
    Test::check("no chains explicit", diagnose_chains({}).status == DiagnosticStatus::InsufficientChains);
    Test::check("one chain explicit", diagnose_chains({x[0]}).status == DiagnosticStatus::InsufficientChains);
    Test::check("short chains explicit",
                diagnose_chains(Chains(2, std::vector<double>(7))).status == DiagnosticStatus::InsufficientDraws);
    bool threw = false;
    try {
        diagnose_chains({{1}, {1, 2}});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    Test::check("ragged chains rejected", threw);
    auto frozen = x;
    frozen[0].assign(x[0].size(), 1);
    Test::check("one frozen chain rejected", diagnose_chains(frozen).status == DiagnosticStatus::ConstantChain);
    const auto constant = diagnose_chains(Chains(4, std::vector<double>(100, .1)));
    Test::check("identical frozen chains not converged",
                !constant.valid() && std::isnan(constant.rhat) && std::isnan(constant.ess_bulk));
    for (auto bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto invalid = x;
        invalid[0][0] = bad;
        Test::check("nonfinite draws rejected", diagnose_chains(invalid).status == DiagnosticStatus::NonFinite);
    }
    const auto discrete = diagnose_chains({{0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}});
    Test::check("constant tail indicator undefined",
                !discrete.valid() && std::isnan(discrete.ess_upper) && std::isnan(discrete.ess_tail));
    const auto folded_constant = diagnose_chains({{-1, 1, -1, 1, -1, 1, -1, 1}, {1, -1, 1, -1, 1, -1, 1, -1}});
    Test::check_close("constant folding has no scale information", folded_constant.rhat_folded, 1., 0.);
    const auto only_middle = diagnose_chains({{0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 0, 0, 0, 0}});
    Test::check("discarded middle cannot fake information", !only_middle.valid() && std::isnan(only_middle.ess_bulk));
    auto odd = x;
    for (auto & c : odd)
        c.push_back(c.front());
    Test::check("odd lengths count retained draws", diagnose_chains(odd).draws_used == x.size() * x[0].size());
    Test::check("no original chain silently truncated", diagnose_chains(odd).draws_per_chain == 2049);
    Test::check("diagnostic status labels",
                std::string(diagnostic_status_name(DiagnosticStatus::ConstantChain)) == "constant_chain");
    Test::check("legacy nonfinite IAT gives ESS zero",
                std::isinf(integrated_autocorrelation_time({1, 2, 3, std::numeric_limits<double>::quiet_NaN()})));
    Test::check("legacy frozen IAT gives ESS zero",
                std::isinf(integrated_autocorrelation_time(std::vector<double>(100, .1))));
    Test::check("legacy short IAT gives ESS zero", std::isinf(integrated_autocorrelation_time({1, 2, 3})));
    Test::check("legacy constant Rhat fails closed", std::isinf(gelman_rubin(Chains(2, std::vector<double>(100, .1)))));
    std::vector<HMCResult> runs(4);
    for (std::size_t i = 0; i < x.size(); ++i)
        for (double v : x[i])
            runs[i].recorded.push_back({v});
    Test::check_close("HMC adapter", diagnose_hmc_chains(runs).rhat, iid.rhat, 1e-12);
    return Test::summary();
}
