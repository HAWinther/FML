// =============================================================================
// Einstein-Boltzmann solver: main program
//
// Pipeline:
//   1. BackgroundCosmology   - H(x), eta(x), t(x), distances
//   2. RecombinationHistory  - X_e(x), tau(x), visibility function g(x)
//   3. Perturbations         - Phi, Psi, delta, v, Theta_ell (x, k), source functions
//   4. PowerSpectrum         - C_ell (lensed), P(k), two-point functions
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

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Isocurvature.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Perturbations/Tensors.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace FML::COSMOLOGY;

namespace {
    std::string tag(double v) {
        char s[32];
        std::snprintf(s, sizeof(s), "%g", v);
        return s;
    }
} // namespace

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
    using Constants::Mpc;
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
    const std::string out = s.output_dir + "/";
    std::filesystem::create_directories(s.output_dir);
    ParameterFile::write(s, out + "params_used.ini");
    s.cosmo.print();
    s.cosmo.write(out + "params.txt");

    // Step 1: background
    BackgroundCosmology cosmo(s.cosmo);
    cosmo.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
    cosmo.info();
    if (s.write_background)
        cosmo.output(out + "background.txt");

    // Step 2: recombination and reionization
    RecombinationHistory rec(cosmo, s.rec);
    rec.solve();
    rec.info();
    if (s.write_recombination)
        rec.output(out + "recombination.txt");

    // Step 3: perturbations (with isocurvature modes: one solution per independent component, Isocurvature.h;
    // the perturbation outputs are those of the first, which contains the adiabatic mode)
    const auto comp_opts = Isocurvature::components(s.cosmo, s.pert);
    Perturbations pert(cosmo, rec, comp_opts[0]);
    pert.solve();
    pert.info();
    std::vector<std::unique_ptr<Perturbations>> extra;
    for (size_t i = 1; i < comp_opts.size(); i++) {
        extra.push_back(std::make_unique<Perturbations>(cosmo, rec, comp_opts[i]));
        extra.back()->solve();
    }
    if (!extra.empty())
        std::cout << "Isocurvature: " << comp_opts.size() << " independent components\n";
    const auto & ks = pert.get_k_array();
    for (double k_Mpc : s.perturbation_k_output) {
        double k = k_Mpc / Mpc;
        const double K = cosmo.get_K();
        if (K > 0.0) { // closed universe: nearest physical mode k = sqrt(K (nu^2 - 1)), integer nu >= 3
            const double nu = std::max(3.0, std::round(std::sqrt(k * k / K + 1.0)));
            k = std::sqrt(K * (nu * nu - 1.0));
        }
        if (k < ks.front() * (1 - 1e-12) || k > ks.back() * (1 + 1e-12)) {
            std::cerr << "Skipping perturbation output at k = " << k_Mpc << "/Mpc (outside the k-range)\n";
            continue;
        }
        pert.output(k, out + "perturbations_k" + tag(k_Mpc) + ".txt");
    }

    // Step 4: power spectra and two-point functions (tensor modes when r > 0)
    PowerSpectrum power(cosmo, rec, pert, s.ps);
    for (const auto & e : extra)
        power.add_component(*e);
    std::unique_ptr<Tensors> tensors;
    if (s.compute_cells && s.cosmo.r > 0.0) {
        tensors = std::make_unique<Tensors>(cosmo, rec, s.tensor);
        tensors->solve();
        tensors->info();
        tensors->output_cells(out + "cells_tensor.txt");
        power.set_tensors(tensors.get());
    }
    if (s.compute_cells) {
        power.solve();
        power.info();
        power.output_cells(out + "cells.txt");
        if (s.ps.lensing)
            power.output_cells_lensed(out + "cells_lensed.txt");
        if (!s.transfer_ells.empty()) {
            // only the ells where C_ell was computed (the default list goes to ell = 1000, above a small ell_max)
            std::vector<int> tl;
            const auto & computed = power.get_ells();
            for (int l : s.transfer_ells) {
                if (std::find(computed.begin(), computed.end(), l) != computed.end())
                    tl.push_back(l);
                else
                    std::cerr << "Warning: transfer_ells: ell = " << l
                              << " is not in the computed ell list, skipped\n";
            }
            if (!tl.empty())
                power.output_transfer(out + "transfer.txt", tl);
        }
        if (s.cmb_correlation) { // unlensed (and lensed), and with the beam if one is given
            std::vector<bool> lensed = {false};
            if (s.ps.lensing)
                lensed.push_back(true);
            for (bool l : lensed) {
                const std::string base = out + "cmb_correlation_" + (l ? "lensed" : "unlensed");
                power.output_cmb_correlation(base + ".txt", l);
                if (s.cmb_correlation_beam_fwhm > 0.0)
                    power.output_cmb_correlation(base + "_beam" + tag(s.cmb_correlation_beam_fwhm) + "deg.txt",
                                                 l,
                                                 s.cmb_correlation_beam_fwhm * M_PI / 180.0);
            }
        }
    }
    if (s.write_matter_power)
        for (double z : s.matter_power_z)
            power.output_matter_power(
                out + (z == 0.0 ? std::string("matter_power.txt") : "matter_power_z" + tag(z) + ".txt"),
                z == 0.0 ? 0.0 : -std::log1p(z));
    if (s.matter_correlation) {
        power.output_matter_correlation(out + "matter_correlation.txt",
                                        0.0,
                                        s.matter_correlation_smoothing * Mpc / s.cosmo.h);
        std::cout << "sigma_8 = " << power.get_sigma8() << " (linear)\n";
    }
    for (double z0 : s.angular_z)
        power.output_angular_matter(out + "angular_cl_z" + tag(z0) + ".txt",
                                    out + "angular_w_z" + tag(z0) + ".txt",
                                    TwoPoint::RedshiftWindow::gaussian(z0, s.angular_sigma_z, s.angular_bias),
                                    s.angular_ell_max);

    std::cout << "Outputs written to " << s.output_dir << "/\n";
    timer.print();
    return 0;
}
