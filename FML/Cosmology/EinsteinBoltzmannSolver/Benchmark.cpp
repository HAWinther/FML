// =============================================================================
// Run-time benchmark of the main options.
//
//   ./benchmark [config ...]      (default: all configurations; OMP_NUM_THREADS sets the threads)
//
// For each configuration the wall-clock time of each stage is measured: background + recombination,
// perturbations, line-of-sight integration + C_ell (+ lensing), and the two-point functions for the
// fiducial model. Prints a markdown table and writes it to data/benchmark.txt.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace FML::COSMOLOGY;

struct Config {
    std::string name, description;
    std::function<void(CosmologyParameters &, PerturbationOptions &, PowerSpectrumOptions &)> set;
    bool twopoint = false;
};

int main(int argc, char ** argv) {
    using Constants::Mpc;
    std::vector<Config> configs = {
        {"T-only",
         "temperature only, no neutrinos (Neff = 0)",
         [](CosmologyParameters & p, PerturbationOptions & o, PowerSpectrumOptions &) {
             p.Neff = 0.0;
             o.polarization = false;
             o.neutrinos = false;
         }},
        {"T+pol",
         "+ polarization (no neutrinos)",
         [](CosmologyParameters & p, PerturbationOptions & o, PowerSpectrumOptions &) {
             p.Neff = 0.0;
             o.neutrinos = false;
         }},
        {"fiducial", "+ massless neutrinos (default; ell_max = 2000)", [](auto &, auto &, auto &) {}},
        {"ell1000", "fiducial, ell_max = 1000", [](auto &, auto &, PowerSpectrumOptions & s) { s.ell_max = 1000; }},
        {"ell2500", "fiducial, ell_max = 2500", [](auto &, auto &, PowerSpectrumOptions & s) { s.ell_max = 2500; }},
        {"lensing", "fiducial + lensing", [](auto &, auto &, PowerSpectrumOptions & s) { s.lensing = true; }},
        {"mnu",
         "fiducial + massive neutrino (0.06 eV)",
         [](CosmologyParameters & p, auto &, auto &) { p.m_nu = 0.06; }},
        {"mnu-nofa",
         "massive neutrino, no fluid approximation",
         [](CosmologyParameters & p, PerturbationOptions & o, auto &) {
             p.m_nu = 0.06;
             o.ncdm_fluid_trigger = 1e30;
         }},
        {"closed", "Omega_K = -0.05 (closed)", [](CosmologyParameters & p, auto &, auto &) { p.OmegaK = -0.05; }},
        {"open", "Omega_K = +0.05 (open)", [](CosmologyParameters & p, auto &, auto &) { p.OmegaK = 0.05; }},
        {"open-noflat",
         "Omega_K = +0.05 (open), no flat approximation",
         [](CosmologyParameters & p, auto &, PowerSpectrumOptions & s) {
             p.OmegaK = 0.05;
             s.flat_approximation_nu = 1e30;
         }},
        {"kmax1",
         "perturbations to k = 1/Mpc (as params/twopoint.ini), CMB LOS to 0.3/Mpc",
         [](auto &, PerturbationOptions & o, PowerSpectrumOptions & s) {
             o.k_max = 1.0 / Mpc;
             o.n_k = 460;
             s.k_max = 0.3 / Mpc;
         }},
        {"twopoint",
         "fiducial + lensing + all two-point functions",
         [](auto &, auto &, PowerSpectrumOptions & s) { s.lensing = true; },
         true},
    };
    std::vector<std::string> want(argv + 1, argv + argc);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    std::ostringstream tab;
    tab << "| configuration | background + recomb. [s] | perturbations [s] | LOS + C_ell [s] | two-point [s] | total "
           "[s] |\n"
        << "|---|---|---|---|---|---|\n";
    for (const auto & cfg : configs) {
        if (!want.empty() && std::find(want.begin(), want.end(), cfg.name) == want.end())
            continue;
        std::cout << "\n=== " << cfg.name << ": " << cfg.description << " ===\n" << std::flush;
        CosmologyParameters p = CosmologyParameters::fiducial();
        PerturbationOptions po;
        PowerSpectrumOptions so;
        cfg.set(p, po, so);

        Utils::Timer total, t;
        BackgroundCosmology cosmo(p);
        cosmo.solve();
        RecombinationHistory rec(cosmo);
        rec.solve();
        const double t_bg = t.elapsed();
        t.start();
        Perturbations pert(cosmo, rec, po);
        pert.solve();
        const double t_pert = t.elapsed();
        t.start();
        PowerSpectrum ps(cosmo, rec, pert, so);
        ps.solve();
        const double t_los = t.elapsed();
        double t_2p = 0.0;
        if (cfg.twopoint) {
            t.start();
            std::vector<double> th;
            for (int i = 0; i <= 1800; i++)
                th.push_back(i * 0.1 * M_PI / 180.0);
            ps.get_cmb_correlation(th, true);
            std::vector<double> r;
            for (int i = 1; i <= 400; i++)
                r.push_back(0.5 * i * Mpc / p.h);
            ps.get_xi_matter(0.0, r);
            ps.get_sigma8();
            std::vector<double> thw = Utils::logspace(0.01 * M_PI / 180.0, 30.0 * M_PI / 180.0, 300);
            ps.get_angular_correlation_matter(TwoPoint::RedshiftWindow::gaussian(1.0, 0.1), thw); // C_ell + w(theta)
            t_2p = t.elapsed();
        }
        char row[512];
        std::snprintf(row,
                      sizeof(row),
                      "| %s | %.1f | %.1f | %.1f | %s | %.1f |\n",
                      cfg.description.c_str(),
                      t_bg,
                      t_pert,
                      t_los,
                      cfg.twopoint ? std::to_string(t_2p).substr(0, 4).c_str() : "-",
                      total.elapsed());
        tab << row;
        std::cout << row << std::flush;
    }
    std::ostringstream head;
    head << "Wall-clock times, " << threads << " OpenMP thread(s).\n\n";
    std::cout << "\n" << head.str() << tab.str();
    std::filesystem::create_directories("data");
    std::ofstream("data/benchmark.txt") << head.str() << tab.str();
    return 0;
}
