#include <FML/MCMC/Diagnostics.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
int main(int argc, char ** argv) {
    try {
        if (argc != 2 && argc != 3)
            throw std::invalid_argument("calibration_diagnostics PREFIX [DRAWS] required");
        const std::string prefix = argv[1];
        size_t expected = 2048;
        if (argc == 3) {
            size_t used;
            const auto n = std::stoll(argv[2], &used);
            if (used != std::string(argv[2]).size() || n < 4 || n > 1000000)
                throw std::invalid_argument("Invalid declared draw count");
            expected = size_t(n);
        }
        std::array<std::vector<std::vector<double>>, 12> projections;
        for (int chain = 0; chain < 4; ++chain) {
            std::ifstream input(prefix + ".chain" + std::to_string(chain) + ".csv");
            if (!input)
                throw std::runtime_error("Missing metric chain");
            std::string line;
            std::getline(input, line);
            std::array<std::vector<double>, 12> values;
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
            if (row != expected)
                throw std::runtime_error("Declared posterior draw count differs");
            for (size_t j = 0; j < 12; ++j)
                projections[j].push_back(std::move(values[j]));
        }
        std::ofstream out(prefix + ".diagnostics.csv");
        out << std::setprecision(17) << "projection,rhat,bulk_ess,tail_ess,status\n";
        for (size_t j = 0; j < 12; ++j) {
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
