#include <FML/tests/TestHelpers.h>
#include "DiagnosticsReference.h"
#include <FML/MCMC/HMC.h>
#include <random>
using namespace FML::MCMC;
using Chains = std::vector<std::vector<double>>;
int main() {
    for (const auto & fixture : diagnostic_fixtures()) {
        const auto d = diagnose_chains(fixture.chains);
        FML::TESTS::check(d.valid(), std::string("reference status ") + fixture.name);
        const std::array<double, 7> values =
            {d.rhat, d.rhat_bulk, d.rhat_folded, d.ess_bulk, d.ess_tail, d.ess_lower, d.ess_upper};
        for (std::size_t j = 0; j < values.size(); ++j)
            FML::TESTS::check_close(values[j],
                                    fixture.expected[j],
                                    2e-8,
                                    std::string("ArviZ ") + fixture.name + " metric " + std::to_string(j));
    }
    std::mt19937_64 rng(17);
    std::normal_distribution<double> normal;
    Chains x(4, std::vector<double>(2048));
    for (auto & c : x)
        for (auto & v : c)
            v = normal(rng);
    const auto iid = diagnose_chains(x);
    FML::TESTS::check(iid.valid(), "IID chains finite");
    FML::TESTS::check(std::abs(iid.rhat - 1) < .02, "IID Rhat near one");
    FML::TESTS::check(iid.ess_bulk > 3000 && iid.ess_tail > 3000, "IID bulk/tail ESS");
    auto shifted = x, scaled = x, correlated = x;
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (auto & v : shifted[i])
            v += 3 * i;
        for (auto & v : scaled[i])
            v *= std::pow(4., i);
        for (std::size_t j = 1; j < x[i].size(); ++j)
            correlated[i][j] += .9 * correlated[i][j - 1];
    }
    FML::TESTS::check(diagnose_chains(shifted).rhat > 1.2, "location mismatch detected");
    const auto scales = diagnose_chains(scaled);
    FML::TESTS::check(scales.rhat_folded > 1.2 && scales.rhat_bulk < 1.02, "folding detects scale mismatch");
    FML::TESTS::check(diagnose_chains(correlated).ess_bulk < iid.ess_bulk / 5, "correlation reduces ESS");
    auto affine = x;
    for (auto & c : affine)
        for (auto & v : c)
            v = 3 * v + 2;
    FML::TESTS::check_close(diagnose_chains(affine).rhat_bulk, iid.rhat_bulk, 1e-12, "bulk rank affine invariance");
    FML::TESTS::check_close(diagnose_chains(affine).ess_bulk, iid.ess_bulk, 1e-10, "ESS affine invariance");
    // Check the FFT autocovariance against independent direct lag products.
    const std::vector<double> seq = {1, 2, -3, 7, -2, 4, 5};
    const auto ac = diagnostic_detail::autocovariance(seq);
    const double mean = std::accumulate(seq.begin(), seq.end(), 0.) / seq.size();
    for (std::size_t lag = 0; lag < seq.size(); ++lag) {
        double direct = 0;
        for (std::size_t j = 0; j + lag < seq.size(); ++j)
            direct += (seq[j] - mean) * (seq[j + lag] - mean) / seq.size();
        FML::TESTS::check_close(ac[lag], direct, 1e-12, "FFT autocovariance lag " + std::to_string(lag));
    }
    FML::TESTS::check(diagnose_chains({}).status == DiagnosticStatus::InsufficientChains, "no chains explicit");
    FML::TESTS::check(diagnose_chains({x[0]}).status == DiagnosticStatus::InsufficientChains, "one chain explicit");
    FML::TESTS::check(diagnose_chains(Chains(2, std::vector<double>(7))).status == DiagnosticStatus::InsufficientDraws,
                      "short chains explicit");
    bool threw = false;
    try {
        diagnose_chains({{1}, {1, 2}});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    FML::TESTS::check(threw, "ragged chains rejected");
    auto frozen = x;
    frozen[0].assign(x[0].size(), 1);
    FML::TESTS::check(diagnose_chains(frozen).status == DiagnosticStatus::ConstantChain, "one frozen chain rejected");
    const auto constant = diagnose_chains(Chains(4, std::vector<double>(100, .1)));
    FML::TESTS::check(!constant.valid() && std::isnan(constant.rhat) && std::isnan(constant.ess_bulk),
                      "identical frozen chains not converged");
    for (auto bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto invalid = x;
        invalid[0][0] = bad;
        FML::TESTS::check(diagnose_chains(invalid).status == DiagnosticStatus::NonFinite, "nonfinite draws rejected");
    }
    const auto discrete = diagnose_chains({{0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}});
    FML::TESTS::check(!discrete.valid() && std::isnan(discrete.ess_upper) && std::isnan(discrete.ess_tail),
                      "constant tail indicator undefined");
    const auto folded_constant = diagnose_chains({{-1, 1, -1, 1, -1, 1, -1, 1}, {1, -1, 1, -1, 1, -1, 1, -1}});
    FML::TESTS::check_close(folded_constant.rhat_folded, 1., 0., "constant folding has no scale information");
    const auto only_middle = diagnose_chains({{0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 2, 0, 0, 0, 0}});
    FML::TESTS::check(!only_middle.valid() && std::isnan(only_middle.ess_bulk),
                      "discarded middle cannot fake information");
    auto odd = x;
    for (auto & c : odd)
        c.push_back(c.front());
    FML::TESTS::check(diagnose_chains(odd).draws_used == x.size() * x[0].size(), "odd lengths count retained draws");
    FML::TESTS::check(diagnose_chains(odd).draws_per_chain == 2049, "no original chain silently truncated");
    FML::TESTS::check(std::string(diagnostic_status_name(DiagnosticStatus::ConstantChain)) == "constant_chain",
                      "diagnostic status labels");
    FML::TESTS::check(std::isinf(integrated_autocorrelation_time({1, 2, 3, std::numeric_limits<double>::quiet_NaN()})),
                      "legacy nonfinite IAT gives ESS zero");
    FML::TESTS::check(std::isinf(integrated_autocorrelation_time(std::vector<double>(100, .1))),
                      "legacy frozen IAT gives ESS zero");
    FML::TESTS::check(std::isinf(integrated_autocorrelation_time({1, 2, 3})), "legacy short IAT gives ESS zero");
    FML::TESTS::check(std::isinf(gelman_rubin(Chains(2, std::vector<double>(100, .1)))),
                      "legacy constant Rhat fails closed");
    std::vector<HMCResult> runs(4);
    for (std::size_t i = 0; i < x.size(); ++i)
        for (double v : x[i])
            runs[i].recorded.push_back({v});
    FML::TESTS::check_close(diagnose_hmc_chains(runs).rhat, iid.rhat, 1e-12, "HMC adapter");
    return FML::TESTS::summary("MCMC diagnostics");
}
