#include <FML/MCMC/Diagnostics.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc, char ** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("metric_diagnostics PREFIX required");
        const std::string prefix = argv[1];
        std::array<std::vector<std::vector<double>>, 6> projections;
        for (int chain = 0; chain < 4; ++chain) {
            std::ifstream input(prefix + ".chain" + std::to_string(chain) + ".csv");
            if (!input)
                throw std::runtime_error("Missing metric chain");
            std::string line;
            std::getline(input, line);
            std::array<std::vector<double>, 6> values;
            size_t row = 0;
            while (std::getline(input, line)) {
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream in(line);
                size_t draw;
                if (!(in >> draw) || draw != row++)
                    throw std::runtime_error("Invalid draw index");
                for (auto & v : values) {
                    double x;
                    if (!(in >> x) || !std::isfinite(x))
                        throw std::runtime_error("Invalid metric projection");
                    v.push_back(x);
                }
                std::string extra;
                if (in >> extra)
                    throw std::runtime_error("Extra chain fields");
            }
            if (row != 128)
                throw std::runtime_error("Metric probe draw count changed");
            for (size_t j = 0; j < 6; ++j)
                projections[j].push_back(std::move(values[j]));
        }
        std::ofstream out(prefix + ".diagnostics.csv");
        out << std::setprecision(17) << "projection,rhat,bulk_ess,tail_ess,status\n";
        for (size_t j = 0; j < 6; ++j) {
            const auto d = FML::MCMC::diagnose_chains(projections[j]);
            out << j << ',' << d.rhat << ',' << d.ess_bulk << ',' << d.ess_tail << ','
                << FML::MCMC::diagnostic_status_name(d.status) << '\n';
        }
        out.close();
        if (!out)
            throw std::runtime_error("Cannot write diagnostics");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
