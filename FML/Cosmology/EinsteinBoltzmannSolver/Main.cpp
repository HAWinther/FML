// =============================================================================
// Einstein-Boltzmann solver: main program
//
// Runs the EinsteinBoltzmannSolver (EinsteinBoltzmannSolver.h: background, recombination, perturbations, spectra)
// with the parameters of a parameter file and writes its outputs.
//
// Usage:
//   ./cmb [parameter file] [name=value ...]
//   ./cmb --write-defaults params/default.ini      write a complete parameter file with all defaults
// Every cosmological, precision and output parameter can be set in the file or on the command line
// (see ParameterFile.h and params/default.ini). Examples:
//   ./cmb                                  fiducial model, outputs in data/
//   ./cmb params/planck2018.ini lensing=true
//   ./cmb m_nu=0.06 OmegaK=0.01 output_dir=data/test
// The parameters actually used are written to <output_dir>/params_used.ini.
// =============================================================================

#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace FML::COSMOLOGY;

int run(int argc, char ** argv);

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

int run(int argc, char ** argv) {
    std::string file;
    std::vector<std::string> overrides;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--write-defaults" && i + 1 < argc) {
            ParameterFile::write(RunSettings{}, argv[++i]);
            std::cout << "Wrote " << argv[i] << "\n";
            return 0;
        } else if (a == "-h" || a == "--help") {
            std::cout << "Usage: " << argv[0] << " [parameter file] [name=value ...]\n"
                      << "       " << argv[0] << " --write-defaults <file>\n";
            return 0;
        } else if (a.find('=') != std::string::npos)
            overrides.push_back(a);
        else if (file.empty())
            file = a;
        else {
            std::cerr << "Unexpected argument '" << a << "' (see --help)\n";
            return 1;
        }
    }

    RunSettings s;
    try {
        s = ParameterFile::read(file, overrides);
    } catch (const std::exception & e) {
        std::cerr << "Error in the parameters: " << e.what() << "\n";
        return 1;
    }
    Utils::Timer timer("total");
    EinsteinBoltzmannSolver ebs(std::move(s));
    ebs.solve();
    ebs.output();
    timer.print();
    return 0;
}
