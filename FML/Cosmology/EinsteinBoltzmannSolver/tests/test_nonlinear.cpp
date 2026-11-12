// =============================================================================
// Tests of the nonlinear matter power spectrum (Halofit.h). Run with `make test`.
//
//   * Power law Delta^2_L = (k/k0)^{n+3}: n_eff = n, C = 0 and sigma(1/k_sigma) = 1 analytically
//   * Linear regime: P_NL = P_L on large scales
//   * solve_matter(): the matter outputs (with Halofit) without the C_ell
//   * P_NL/P_L against CLASS v3.4 (non_linear = halofit) for Planck 2018 at z = 0 and 1 (here with precision = fast;
//     with the default precision and pert_k_max = 1/Mpc P_NL agrees to < 1e-3, python/compare_class_halofit.py)
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Spectra/Halofit.h>
#include <FML/HaloModel/HMcode2016.h>
#include <FML/HaloModel/HMcode2020.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/ParameterFile.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/tests/TestHelpers.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using namespace FML::COSMOLOGY;

using namespace Constants;
using FML::TESTS::check;
using FML::TESTS::check_close;
using FML::TESTS::check_rel;

int main() {
    Utils::verbose = false;

    std::cout << "Power-law spectrum\n";
    {
        std::vector<double> lnk, D2;
        for (double l = std::log(1e-4); l <= std::log(1e3); l += 0.01)
            lnk.push_back(l);
        const double n = -1.5, k0 = 0.2; // Delta^2 = (k/k0)^{1.5}
        for (double l : lnk)
            D2.push_back(std::pow(std::exp(l) / k0, n + 3.0));
        std::vector<double> NL;
        double ks, ne, C;
        const bool ok = Halofit::nonlinear(lnk, D2, 0.3, 0.7, -1.0, NL, ks, ne, C);
        // sigma^2(R) = Gamma((n+3)/2)/2 (k0 R)^{-(n+3)}  ->  k_sigma = k0 [Gamma((n+3)/2)/2]^{-1/(n+3)}
        const double ks_exact = k0 * std::pow(std::tgamma((n + 3.0) / 2.0) / 2.0, -1.0 / (n + 3.0));
        check(ok, "nonlinear scale found");
        check_rel(ks, ks_exact, 1e-4, "k_sigma");
        check_close(ne, n, 1e-4, "n_eff = n");
        check_close(C, 0.0, 1e-4, "C = 0");
    }

    std::cout << "Planck 2018: P_NL/P_L against CLASS (non_linear = halofit)\n";
    {
        const RunSettings s = ParameterFile::read("", {"preset=planck2018", "precision=fast", "pert_k_max=1"});
        BackgroundCosmology bg(s.cosmo);
        bg.solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
        RecombinationHistory rec(bg, s.rec);
        rec.solve();
        Perturbations pert(bg, rec, s.pert);
        pert.solve();
        PowerSpectrum ps(bg, rec, pert, s.ps);
        Halofit hf(bg, [&](double x, double k) { return ps.get_matter_power_spectrum_extrapolated(x, k); });
        const double h = s.cosmo.h;
        // Delta^2_Q ~ Delta^2_L exp(-y/4) for y = k/k_sigma -> 0: linear only as k -> 0 (0.3% below at 0.005 h/Mpc, as
        // CLASS)
        check_close(hf.ratio(0.0, 2e-4 * h / Mpc), 1.0, 5e-4, "linear on large scales (k = 2e-4 h/Mpc, z = 0)");
        check_close(hf.ratio(-std::log(20.0), 1.0 / Mpc), 1.0, 0.0, "ratio = 1 above z_max");
        // CLASS: pk/pk_lin at k = 0.1, 0.3, 1 h/Mpc
        struct Ref {
            double z, k, r;
        };
        for (const Ref & r : {Ref{0.0, 0.1, 1.01289},
                              Ref{0.0, 0.3, 1.54388},
                              Ref{0.0, 1.0, 6.00259},
                              Ref{1.0, 0.1, 1.00999},
                              Ref{1.0, 0.3, 1.24156},
                              Ref{1.0, 1.0, 3.28870}}) {
            const double v = hf.ratio(-std::log1p(r.z), r.k * h / Mpc);
            check_rel(v,
                      r.r,
                      5e-3,
                      "z = " + std::to_string(int(r.z)) + ", k = " + std::to_string(r.k).substr(0, 3) + " h/Mpc");
        }

        // P(k) outputs without the C_ell (compute_cells = false): solve_matter() sets up Halofit only
        PowerSpectrumOptions o = s.ps;
        o.nonlinear = PowerSpectrumOptions::Nonlinear::Halofit;
        PowerSpectrum pm(bg, rec, pert, o);
        const std::string file = "data/test_nonlinear_matter_power.txt";
        bool threw = false;
        try {
            pm.output_matter_power(file, 0.0);
        } catch (const std::exception &) {
            threw = true;
        }
        check(threw, "output_matter_power before solve_matter() is an error");
        pm.solve_matter();
        pm.output_matter_power(file, 0.0);
        std::ifstream in(file);
        std::string line;
        std::getline(in, line);
        std::getline(in, line);
        std::istringstream ls(line);
        double v, n = 0;
        while (ls >> v)
            n++;
        check(n == 4, "solve_matter(): matter_power with the Halofit column");
        const double k = 0.3 * h / Mpc;
        check_rel(pm.get_matter_power_spectrum_nonlinear(0.0, k) / pm.get_matter_power_spectrum_extrapolated(0.0, k),
                  hf.ratio(0.0, k),
                  1e-12,
                  "solve_matter(): P_NL/P_L = Halofit");
        o.nonlinear = PowerSpectrumOptions::Nonlinear::HalofitBird;
        PowerSpectrum bird(bg, rec, pert, o);
        bird.solve_matter();
        check_rel(bird.get_matter_power_spectrum_nonlinear(0.0, k),
                  pm.get_matter_power_spectrum_nonlinear(0.0, k), 1e-12,
                  "massless neutrinos: Bird and existing Halofit agree through the solver");
        o.nonlinear = PowerSpectrumOptions::Nonlinear::HMcode2020;
        PowerSpectrum hm(bg, rec, pert, o);
        hm.solve_matter();
        hm.solve_matter(); // idempotent
        check_rel(hm.get_matter_power_spectrum_cb_extrapolated(0.0, k),
                  hm.get_matter_power_spectrum_cb(0.0, k), 1e-12,
                  "HMcode cold-spectrum extrapolation preserves the computed k range");
        auto total = [&](double x, double kk) { return hm.get_matter_power_spectrum_extrapolated(x, kk); };
        auto cold = [&](double x, double kk) { return hm.get_matter_power_spectrum_cb_extrapolated(x, kk); };
        const auto direct = HALOMODEL::HMcode2020::spectrum(bg, total, cold, -std::log(2.0), {0.3});
        const double V = std::pow(Mpc / h, 3);
        check_rel(hm.get_matter_power_spectrum_nonlinear(-std::log(2.0), k) / V,
                  direct.nonlinear[0], 2e-4, "HMcode time/k table matches direct evaluation off the table nodes");
        check_rel(hm.get_matter_power_spectrum_nonlinear(-std::log(20.0), k),
                  total(-std::log(20.0), k), 1e-12, "HMcode: linear above z_max");
        hm.output_matter_power(file, 0.0);
        std::ifstream hin(file);
        std::getline(hin, line);
        check(line.find("HMcode-2020") != std::string::npos, "matter output identifies HMcode-2020");
        std::getline(hin, line);
        std::istringstream hls(line);
        n = 0;
        while (hls >> v)
            n++;
        check(n == 4, "HMcode nonlinear power is written as the fourth column");
        {
            // HMcode-2016 through the solver (the general HaloModel)
            auto o16 = o;
            o16.nonlinear = PowerSpectrumOptions::Nonlinear::HMcode2016;
            PowerSpectrum hm16(bg, rec, pert, o16);
            hm16.solve_matter();
            auto total16 = [&](double x, double kk) { return hm16.get_matter_power_spectrum_extrapolated(x, kk); };
            auto cold16 = [&](double x, double kk) { return hm16.get_matter_power_spectrum_cb_extrapolated(x, kk); };
            const auto direct16 = HALOMODEL::HMcode2016::spectrum(bg, total16, cold16, -std::log(2.0), {0.3});
            check_rel(hm16.get_matter_power_spectrum_nonlinear(-std::log(2.0), k) / V, direct16.nonlinear[0], 2e-4,
                      "HMcode-2016 time/k table matches direct evaluation off the table nodes");
            check_rel(hm16.get_matter_power_spectrum_nonlinear(-std::log(20.0), k), total16(-std::log(20.0), k), 1e-12,
                      "HMcode-2016: linear above z_max");
            const double r16 = hm16.get_matter_power_spectrum_nonlinear(0.0, k) / hm.get_matter_power_spectrum_nonlinear(0.0, k);
            check(std::abs(r16 - 1) < 0.05, "HMcode-2016 and HMcode-2020 within 5% at k = 0.3 h/Mpc, z = 0 (ratio " +
                                                std::to_string(r16) + ")");
            hm16.output_matter_power(file, 0.0);
            std::ifstream in16(file);
            std::getline(in16, line);
            check(line.find("HMcode-2016") != std::string::npos, "matter output identifies HMcode-2016");
        }
        {
            o.nonlinear = PowerSpectrumOptions::Nonlinear::ReACT;
            o.react_n_z = 4;
            o.react_n_k = 64;
            o.react_mass_loop = 10;
            PowerSpectrum reaction(bg, rec, pert, o);
            reaction.solve_matter();
            for (double z : {0.0, 1.0, 5.0})
                check_rel(reaction.get_matter_power_spectrum_nonlinear(-std::log1p(z), k),
                          hm.get_matter_power_spectrum_nonlinear(-std::log1p(z), k), 1e-12,
                          "massless GR: ReACT x matched pseudo = HMcode exactly, z=" + std::to_string(z));
            reaction.output_matter_power(file, 0);
            std::ifstream rin(file);
            std::getline(rin, line);
            check(line.find("ReACT") != std::string::npos, "matter output identifies ReACT");
        }
        std::remove(file.c_str());
    }

    return FML::TESTS::summary("EBS nonlinear");
}
