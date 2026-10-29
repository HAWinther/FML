// Optional independent-reference harness; no Python dependencies in native tests.
#include <FML/MCMC/Diagnostics.h>
#include <iomanip>
#include <iostream>
int main() {
    std::size_t m, n;
    while (std::cin >> m >> n) {
        std::vector<std::vector<double>> chains(m, std::vector<double>(n));
        for (auto & c : chains)
            for (auto & x : c)
                if (!(std::cin >> x))
                    return 2;
        const auto d = FML::MCMC::diagnose_chains(chains);
        std::cout << std::setprecision(17) << d.rhat << ' ' << d.rhat_bulk << ' ' << d.rhat_folded << ' ' << d.ess_bulk
                  << ' ' << d.ess_tail << ' ' << d.ess_lower << ' ' << d.ess_upper << '\n';
    }
}
