// ./example input.txt output.txt basic|dome h Omega_b Omega_cdm Omega_axion mass_eV z k_max [allow_extrapolation]
// Input columns: k[h/Mpc] P_cold[(Mpc/h)^3] P_axion[(Mpc/h)^3], at the requested redshift.
#include <FML/AxionHMcode/AxionHMcode.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char ** argv) {
    using A = FML::COSMOLOGY::AxionHMcode;
    try {
        if (argc != 11 && argc != 12)
            throw std::invalid_argument("Usage: example input.txt output.txt basic|dome h Omega_b Omega_cdm Omega_axion mass_eV z k_max [allow_extrapolation]");
        A::Options options;
        const std::string version = argv[3];
        if (version != "basic" && version != "dome") throw std::invalid_argument("version must be basic or dome");
        options.version = version == "basic" ? A::Version::Basic : A::Version::Dome2024;
        if (argc == 12 && std::string(argv[11]) != "true" && std::string(argv[11]) != "false")
            throw std::invalid_argument("allow_extrapolation must be true or false");
        options.allow_extrapolation = argc == 12 && std::string(argv[11]) == "true";
        A::Cosmology c;
        c.h = std::stod(argv[4]); c.Omega_b = std::stod(argv[5]); c.Omega_cdm = std::stod(argv[6]);
        c.Omega_axion = std::stod(argv[7]); c.mass_eV = std::stod(argv[8]); c.z = std::stod(argv[9]);
        const double kmax = std::stod(argv[10]);
        std::ifstream input(argv[1]);
        if (!input) throw std::runtime_error("Cannot open input linear spectra");
        A::Input linear;
        std::string line;
        std::vector<double> output_k;
        while (std::getline(input, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream s(line);
            double k, pc, pa;
            if (!(s >> k >> pc >> pa)) throw std::runtime_error("Expected three input columns");
            linear.k.push_back(k); linear.cold.push_back(pc); linear.axion.push_back(pa);
            if (k <= kmax) output_k.push_back(k);
        }
        const auto result = A::spectrum(c, linear, output_k, options);
        std::ofstream output(argv[2]);
        if (!output) throw std::runtime_error("Cannot open output file");
        output << "# Native AxionHMcode " << version << "; z=" << c.z << "; extrapolated=" << result.extrapolated
               << "; clustered_fraction=" << result.clustered_fraction << "\n"
               << "# k[h/Mpc] P_linear_total P_total P_cold P_cold_axion P_axion [(Mpc/h)^3]\n";
        output << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < result.k.size(); ++i)
            output << result.k[i] << " " << result.linear_total[i] << " " << result.total[i] << " "
                   << result.cold[i] << " " << result.cold_axion[i] << " " << result.axion[i] << "\n";
        std::cout << "Wrote " << result.k.size() << " samples; clustered fraction=" << result.clustered_fraction
                  << ", cutoff mass=" << result.cutoff_mass << " Msun/h; rejected axion hosts="
                  << result.rejected_axion_hosts.size() << "\n";
    } catch (const std::exception & e) { std::cerr << e.what() << "\n"; return 1; }
}
