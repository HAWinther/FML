#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>
#include <FML/Cosmology/Perturbations/Isocurvature.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    namespace {
        std::string tag(double v) {
            char s[32];
            std::snprintf(s, sizeof(s), "%g", v);
            return s;
        }
        const char * stage_name(EinsteinBoltzmannSolver::Stage stage) {
            static const char * names[] = {"background", "recombination", "perturbations", "spectra"};
            return names[int(stage)];
        }
    } // namespace

    EinsteinBoltzmannSolver::EinsteinBoltzmannSolver(RunSettings settings) : s(std::move(settings)) {}
    EinsteinBoltzmannSolver::~EinsteinBoltzmannSolver() = default;

    EinsteinBoltzmannSolver EinsteinBoltzmannSolver::from_file(const std::string & filename,
                                                               const std::vector<std::string> & overrides) {
        return EinsteinBoltzmannSolver(ParameterFile::read(filename, overrides));
    }

    void EinsteinBoltzmannSolver::require(Stage stage, const char * what) const {
        if (not is_solved(stage))
            throw std::logic_error(std::string("[EinsteinBoltzmannSolver] ") + what + ": the " + stage_name(stage) +
                                   " stage is not solved (call solve())");
    }

    void EinsteinBoltzmannSolver::solve(Stage last) {
        if (n_solved == 0 and int(last) >= int(Stage::Background)) {
            if (verbose)
                s.cosmo.print();
            cosmo = std::make_unique<BackgroundCosmology>(s.cosmo);
            cosmo->solve(s.bg_x_start, s.bg_x_end, s.bg_n_points);
            if (verbose)
                cosmo->info();
            n_solved = 1;
        }
        if (n_solved == 1 and int(last) >= int(Stage::Recombination)) {
            rec = std::make_unique<RecombinationHistory>(*cosmo, s.rec);
            rec->solve();
            if (verbose)
                rec->info();
            n_solved = 2;
        }
        if (n_solved == 2 and int(last) >= int(Stage::Perturbations)) {
            // With isocurvature modes: one solution per independent component (Isocurvature.h)
            const auto comp_opts = Isocurvature::components(s.cosmo, s.pert);
            for (size_t i = 0; i < comp_opts.size(); i++) {
                pert.push_back(std::make_unique<Perturbations>(*cosmo, *rec, comp_opts[i]));
                pert.back()->solve();
                if (i == 0 and verbose)
                    pert.back()->info();
            }
            if (pert.size() > 1 and verbose)
                std::cout << "Isocurvature: " << pert.size() << " independent components\n";
            n_solved = 3;
        }
        if (n_solved == 3 and int(last) >= int(Stage::Spectra)) {
            power = std::make_unique<PowerSpectrum>(*cosmo, *rec, *pert[0], s.ps);
            for (size_t i = 1; i < pert.size(); i++)
                power->add_component(*pert[i]);
            if (s.compute_cells) {
                if (s.cosmo.r > 0.0) {
                    tens = std::make_unique<Tensors>(*cosmo, *rec, s.tensor);
                    tens->solve();
                    if (verbose)
                        tens->info();
                    power->set_tensors(tens.get());
                }
                power->solve();
                if (verbose)
                    power->info();
            } else {
                power->solve_matter(); // P(k) (and Halofit) without the C_ell
            }
            n_solved = 4;
        }
    }

    void EinsteinBoltzmannSolver::reset(Stage first) {
        const int f = int(first);
        if (f <= int(Stage::Spectra)) {
            power.reset();
            tens.reset();
        }
        if (f <= int(Stage::Perturbations))
            pert.clear();
        if (f <= int(Stage::Recombination))
            rec.reset();
        if (f <= int(Stage::Background))
            cosmo.reset();
        n_solved = std::min(n_solved, f);
    }

    const BackgroundCosmology & EinsteinBoltzmannSolver::background() const {
        require(Stage::Background, "background()");
        return *cosmo;
    }
    const RecombinationHistory & EinsteinBoltzmannSolver::recombination() const {
        require(Stage::Recombination, "recombination()");
        return *rec;
    }
    const Perturbations & EinsteinBoltzmannSolver::perturbations() const {
        require(Stage::Perturbations, "perturbations()");
        return *pert[0];
    }
    std::vector<const Perturbations *> EinsteinBoltzmannSolver::perturbation_components() const {
        require(Stage::Perturbations, "perturbation_components()");
        std::vector<const Perturbations *> v;
        for (const auto & p : pert)
            v.push_back(p.get());
        return v;
    }
    const PowerSpectrum & EinsteinBoltzmannSolver::power_spectrum() const {
        require(Stage::Spectra, "power_spectrum()");
        return *power;
    }

    void EinsteinBoltzmannSolver::output() const {
        using Constants::Mpc;
        const std::string out = s.output_dir + "/";
        std::filesystem::create_directories(s.output_dir);
        ParameterFile::write(s, out + "params_used.ini");
        s.cosmo.write(out + "params.txt");

        if (is_solved(Stage::Background) and s.write_background)
            cosmo->output(out + "background.txt");
        if (is_solved(Stage::Recombination) and s.write_recombination)
            rec->output(out + "recombination.txt");

        if (is_solved(Stage::Perturbations)) {
            const auto & p = *pert[0];
            const auto & ks = p.get_k_array();
            for (double k_Mpc : s.perturbation_k_output) {
                double k = k_Mpc / Mpc;
                const double K = cosmo->get_K();
                if (K > 0.0) { // closed universe: nearest physical mode k = sqrt(K (nu^2 - 1)), integer nu >= 3
                    const double nu = std::max(3.0, std::round(std::sqrt(k * k / K + 1.0)));
                    k = std::sqrt(K * (nu * nu - 1.0));
                }
                if (k < ks.front() * (1 - 1e-12) or k > ks.back() * (1 + 1e-12)) {
                    std::cerr << "Skipping perturbation output at k = " << k_Mpc << "/Mpc (outside the k-range)\n";
                    continue;
                }
                p.output(k, out + "perturbations_k" + tag(k_Mpc) + ".txt");
            }
        }

        if (not is_solved(Stage::Spectra))
            return;
        if (tens)
            tens->output_cells(out + "cells_tensor.txt");
        if (s.compute_cells) {
            power->output_cells(out + "cells.txt");
            if (s.ps.lensing)
                power->output_cells_lensed(out + "cells_lensed.txt");
            if (not s.transfer_ells.empty()) {
                // only the ells where C_ell was computed (the default list goes to ell = 1000, above a small ell_max)
                std::vector<int> tl;
                const auto & computed = power->get_ells();
                for (int l : s.transfer_ells) {
                    if (std::find(computed.begin(), computed.end(), l) != computed.end())
                        tl.push_back(l);
                    else
                        std::cerr << "Warning: transfer_ells: ell = " << l
                                  << " is not in the computed ell list, skipped\n";
                }
                if (not tl.empty())
                    power->output_transfer(out + "transfer.txt", tl);
            }
            if (s.cmb_correlation) { // unlensed (and lensed), and with the beam if one is given
                std::vector<bool> lensed = {false};
                if (s.ps.lensing)
                    lensed.push_back(true);
                for (bool l : lensed) {
                    const std::string base = out + "cmb_correlation_" + (l ? "lensed" : "unlensed");
                    power->output_cmb_correlation(base + ".txt", l);
                    if (s.cmb_correlation_beam_fwhm > 0.0)
                        power->output_cmb_correlation(base + "_beam" + tag(s.cmb_correlation_beam_fwhm) + "deg.txt",
                                                      l,
                                                      s.cmb_correlation_beam_fwhm * M_PI / 180.0);
                }
            }
        }
        if (s.write_matter_power)
            for (double z : s.matter_power_z)
                power->output_matter_power(
                    out + (z == 0.0 ? std::string("matter_power.txt") : "matter_power_z" + tag(z) + ".txt"),
                    z == 0.0 ? 0.0 : -std::log1p(z));
        if (s.matter_correlation) {
            power->output_matter_correlation(out + "matter_correlation.txt",
                                             0.0,
                                             s.matter_correlation_smoothing * Mpc / s.cosmo.h);
            std::cout << "sigma_8 = " << power->get_sigma8() << " (linear)\n";
        }
        for (double z0 : s.angular_z)
            power->output_angular_matter(out + "angular_cl_z" + tag(z0) + ".txt",
                                         out + "angular_w_z" + tag(z0) + ".txt",
                                         TwoPoint::RedshiftWindow::gaussian(z0, s.angular_sigma_z, s.angular_bias),
                                         s.angular_ell_max);
        std::cout << "Outputs written to " << s.output_dir << "/\n";
    }

} // namespace FML::COSMOLOGY
