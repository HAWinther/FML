// =============================================================================
// Computes all data needed for the report (Milestones I-IV) and writes it to report/data/.
//
//   make apps && ./milestones
//   python3 report/make_figures.py
//
// Fiducial cosmology: h = 0.67, TCMB = 2.7255 K, Neff = 3.046, Omega_b = 0.05,
// Omega_cdm = 0.267, Omega_k = 0, n_s = 0.965, A_s = 2.1e-9, Yp = 0.245,
// z_reion = 8, dz_reion = 0.5, z_He_reion = 3.5, dz_He_reion = 0.5.
// Physics: helium + reionization, photon polarization, massless neutrinos. Milestones II-IV use the Peebles
// recombination model of the course (RecombinationOptions::Model::Peebles); RECFAST, the code's default, and the
// models beyond LCDM are in the extension chapter (models_chapter below).
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>
#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

using namespace FML::COSMOLOGY;

using namespace Constants;

static const std::string DIR = "report/data/";

// The course's recombination model (Saha + Peebles)
static RecombinationOptions peebles() {
    RecombinationOptions o;
    o.model = RecombinationOptions::Model::Peebles;
    return o;
}

// Simple "key value" writer
struct Numbers {
    std::map<std::string, double> v;
    std::vector<std::string> order;
    void set(const std::string & k, double x) {
        if (!v.count(k))
            order.push_back(k);
        v[k] = x;
    }
    void write(const std::string & file) const {
        std::ofstream fp(DIR + file);
        fp << std::setprecision(10);
        for (const auto & k : order)
            fp << k << " " << v.at(k) << "\n";
    }
};

// =============================================================================
// Milestone I
// =============================================================================
static void milestone1(const BackgroundCosmology & cosmo) {
    Numbers n;
    const auto & p = cosmo.get_params();
    auto epoch = [&](const std::string & name, double x) {
        n.set(name + "_x", x);
        n.set(name + "_z", std::exp(-x) - 1.0);
        n.set(name + "_t_yr", cosmo.t_of_x(x) / yr);
    };
    epoch("radiation_matter_equality", cosmo.x_matter_radiation_equality());
    epoch("matter_lambda_equality", cosmo.x_matter_DE_equality());
    epoch("acceleration_onset", cosmo.x_acceleration_onset());
    n.set("t0_Gyr", cosmo.t0() / Gyr);
    n.set("eta0_over_c_Gyr", cosmo.eta0() / c / Gyr);
    n.set("eta0_Mpc", cosmo.eta0() / Mpc);
    n.set("OmegaB0", cosmo.OmegaB_of_x(0.0));
    n.set("OmegaCDM0", cosmo.OmegaCDM_of_x(0.0));
    n.set("OmegaGamma0", cosmo.OmegaGamma_of_x(0.0));
    n.set("OmegaNu0", cosmo.OmegaNu_of_x(0.0));
    n.set("OmegaK0", cosmo.OmegaK_of_x(0.0));
    n.set("OmegaLambda0", cosmo.OmegaDE_of_x(0.0));
    n.set("Omega_sum0",
          cosmo.OmegaB_of_x(0.0) + cosmo.OmegaCDM_of_x(0.0) + cosmo.OmegaGamma_of_x(0.0) + cosmo.OmegaNu_of_x(0.0) +
              cosmo.OmegaK_of_x(0.0) + cosmo.OmegaDE_of_x(0.0));
    n.set("H0_kms_Mpc", cosmo.get_H0() * Mpc / km);
    n.set("h", p.h);
    const double x_eq = cosmo.x_matter_radiation_equality();
    n.set("k_eq_hMpc", cosmo.Hp_of_x(x_eq) / c * Mpc / p.h);
    n.write("m1_numbers.txt");
    cosmo.output(DIR + "background.txt", -20.0, 5.0, 5000);
    std::cout << "Milestone I: done\n";
}

// =============================================================================
// Milestone II
// =============================================================================
static void recombination_numbers(Numbers & n,
                                  const std::string & tag,
                                  const BackgroundCosmology & cosmo,
                                  const RecombinationHistory & rec) {
    auto epoch = [&](const std::string & name, double x) {
        n.set(tag + "_" + name + "_x", x);
        n.set(tag + "_" + name + "_z", std::exp(-x) - 1.0);
        n.set(tag + "_" + name + "_t_yr", cosmo.t_of_x(x) / yr);
    };
    const double x_ls = rec.x_last_scattering();
    epoch("decoupling_tau1", x_ls);
    epoch("decoupling_gpeak", rec.x_visibility_peak());
    epoch("recombination_Xe0.1", Utils::find_root([&](double x) { return rec.Xe_of_x(x) - 0.1; }, -9.0, -5.0));
    epoch("Xe0.5", Utils::find_root([&](double x) { return rec.Xe_of_x(x) - 0.5; }, -9.0, -5.0));
    n.set(tag + "_Xe_today", rec.Xe_of_x(0.0));
    n.set(tag + "_rs_decoupling_Mpc", rec.sound_horizon_of_x(x_ls) / Mpc);

    // Drag epoch: the baryons feel the photon drag at the rate tau' R (see the v_b equation, with
    // R = 4 rho_gamma / (3 rho_b)), so the baryon optical depth is tau_b(x) = int_x^0 (-tau' R) dx'.
    auto tau_b = [&](double x) {
        return Utils::integrate([&](double xp) { return -rec.dtaudx_of_x(xp) * rec.R_of_x(xp); }, x, 0.0, 0.0, 1e-8);
    };
    const double x_drag = Utils::find_root([&](double x) { return tau_b(x) - 1.0; }, -8.0, -6.0, 1e-10, 1e-10);
    epoch("drag", x_drag);
    n.set(tag + "_rs_drag_Mpc", rec.sound_horizon_of_x(x_drag) / Mpc);
}

static void milestone2(const BackgroundCosmology & cosmo) {
    Numbers n;

    // Fiducial: helium + reionization
    RecombinationHistory rec_fid(cosmo, peebles());
    rec_fid.solve();
    rec_fid.output(DIR + "recombination_fiducial.txt", -12.0, 0.0, 12000);
    n.set("fiducial_tau_reion", rec_fid.tau_reionization());
    n.set("fiducial_Xe_today", rec_fid.Xe_of_x(0.0));
    n.set("f_He", rec_fid.f_He());

    // Numbers are quoted with reionization turned off
    RecombinationOptions o_noreion = peebles();
    o_noreion.reionization = false;
    RecombinationHistory rec_nr(cosmo, o_noreion);
    rec_nr.solve();
    rec_nr.output(DIR + "recombination_noreion.txt", -12.0, 0.0, 12000);
    recombination_numbers(n, "peebles", cosmo, rec_nr);

    // Saha only (never switch to Peebles), same physics otherwise
    RecombinationOptions o_saha = o_noreion;
    o_saha.Xe_saha_limit = 0.0;
    RecombinationHistory rec_saha(cosmo, o_saha);
    rec_saha.solve();
    rec_saha.output(DIR + "recombination_saha.txt", -12.0, 0.0, 12000);
    recombination_numbers(n, "saha", cosmo, rec_saha);

    // Hydrogen only (helium neutral), no reionization
    RecombinationOptions o_h = o_noreion;
    o_h.helium = false;
    RecombinationHistory rec_h(cosmo, o_h);
    rec_h.solve();
    rec_h.output(DIR + "recombination_Honly.txt", -12.0, 0.0, 12000);
    recombination_numbers(n, "Honly", cosmo, rec_h);

    // RECFAST-like fudge factor
    RecombinationOptions o_f = o_noreion;
    o_f.peebles_fudge = 1.14;
    RecombinationHistory rec_f(cosmo, o_f);
    rec_f.solve();
    recombination_numbers(n, "fudge", cosmo, rec_f);

    // Consistency: with Yp = 0 the helium solver must reproduce the hydrogen-only result
    {
        CosmologyParameters p0 = cosmo.get_params();
        p0.Yp = 0.0;
        BackgroundCosmology c0(p0);
        c0.solve();
        RecombinationOptions a = o_noreion, b = o_noreion;
        b.helium = false;
        RecombinationHistory ra(c0, a), rb(c0, b);
        ra.solve();
        rb.solve();
        double max_rel = 0.0;
        for (double x = -12.0; x <= 0.0; x += 0.001)
            max_rel = std::max(max_rel, std::fabs(ra.Xe_of_x(x) / rb.Xe_of_x(x) - 1.0));
        n.set("Yp0_helium_vs_Honly_max_rel_diff_Xe", max_rel);
    }

    // Normalisation of g
    {
        auto xs = Utils::linspace(-20.0, 0.0, 400001);
        std::vector<double> gs(xs.size());
        for (size_t i = 0; i < xs.size(); i++)
            gs[i] = rec_fid.g_tilde_of_x(xs[i]);
        n.set("fiducial_int_g_dx", Utils::trapz(xs, gs));
    }
    n.write("m2_numbers.txt");
    std::cout << "Milestone II: done\n";
}

// =============================================================================
// Milestones III and IV
// =============================================================================
static void milestone34(const BackgroundCosmology & cosmo) {
    RecombinationHistory rec(cosmo, peebles());
    rec.solve();

    PerturbationOptions po;
    po.store_source_components = true;
    Perturbations pert(cosmo, rec, po);
    pert.solve();
    pert.info();

    Numbers n3;
    for (double k_Mpc : {0.001, 0.01, 0.1, 0.3}) {
        const double k = k_Mpc / Mpc;
        char name[64];
        std::snprintf(name, sizeof(name), "perturbations_k%g.txt", k_Mpc);
        pert.output(k, DIR + name);
        // Horizon crossing: ck = Hp
        const double x_hc = Utils::find_root([&](double x) { return c * k / cosmo.Hp_of_x(x) - 1.0; }, -18.0, 0.0);
        char key[64];
        std::snprintf(key, sizeof(key), "k%g_horizon_entry_x", k_Mpc);
        n3.set(key, x_hc);
        std::snprintf(key, sizeof(key), "k%g_tc_end_x", k_Mpc);
        n3.set(key, pert.x_tight_coupling_end(k));
    }
    n3.write("m3_numbers.txt");
    std::cout << "Milestone III: done\n";

    // Milestone IV (ell up to 2500 to cover the Planck data; k_max = 0.3/Mpc gives k eta0 ~ 4300)
    PowerSpectrumOptions so_fid;
    so_fid.ell_max = 2500;
    so_fid.lensing = true; // also C_ell^phiphi and the lensed spectra (unlensed computed to ell = 3333)
    PowerSpectrum ps(cosmo, rec, pert, so_fid);
    ps.solve();
    ps.info();
    ps.output_cells(DIR + "cells_fiducial.txt");
    ps.output_cells_lensed(DIR + "cells_lensed_fiducial.txt");
    ps.output_matter_power(DIR + "matter_power_fiducial.txt");
    ps.output_transfer(DIR + "transfer_fiducial.txt", {6, 100, 200, 500, 1000});

    const std::vector<std::pair<Perturbations::SourceTerm, std::string>> terms = {
        {Perturbations::SourceTerm::SW, "SW"},
        {Perturbations::SourceTerm::ISW, "ISW"},
        {Perturbations::SourceTerm::Doppler, "Doppler"},
        {Perturbations::SourceTerm::Quadrupole, "Quadrupole"}};
    for (const auto & [term, name] : terms) {
        PowerSpectrumOptions so = so_fid;
        so.source_term = term;
        so.lensing = false;
        PowerSpectrum pst(cosmo, rec, pert, so);
        pst.solve();
        pst.output_cells(DIR + "cells_" + name + ".txt");
    }

    Numbers n4;
    double l_peak = 0, D_peak = 0;
    for (double l = 100; l <= 400; l += 1.0)
        if (ps.get_Dell_TT(l) > D_peak) {
            D_peak = ps.get_Dell_TT(l);
            l_peak = l;
        }
    n4.set("TT_first_peak_ell", l_peak);
    n4.set("TT_first_peak_Dell", D_peak);
    n4.set("TT_Dell_10", ps.get_Dell_TT(10));
    {
        // P(k) maximum
        double kmax = 0, Pmax = 0;
        for (double k : Utils::logspace(1e-3 / Mpc, 0.2 / Mpc, 2000))
            if (ps.get_matter_power_spectrum(0.0, k) > Pmax) {
                Pmax = ps.get_matter_power_spectrum(0.0, k);
                kmax = k;
            }
        const double h = cosmo.get_params().h;
        n4.set("Pk_peak_k_hMpc", kmax * Mpc / h);
        n4.set("Pk_peak_P_Mpch3", Pmax / std::pow(Mpc / h, 3));
    }
    n4.write("m4_numbers.txt");
    std::cout << "Milestone IV: done\n";
}

// Toy cosmology of the course reference figures: Omega_b = 0.05, Omega_cdm = 0.45,
// Omega_Lambda = 0.5, h = 0.7, no neutrinos, Yp = 0, no reionization, temperature only
static void toy_cosmology() {
    CosmologyParameters p;
    p.h = 0.7;
    p.OmegaB = 0.05;
    p.OmegaCDM = 0.45;
    p.Neff = 0.0;
    p.Yp = 0.0;
    p.reionization = false;
    BackgroundCosmology cosmo(p);
    cosmo.solve();
    RecombinationHistory rec(cosmo, peebles());
    rec.solve();
    PerturbationOptions po;
    po.polarization = false;
    Perturbations pert(cosmo, rec, po);
    pert.solve();
    PowerSpectrum ps(cosmo, rec, pert);
    ps.solve();
    ps.output_cells(DIR + "cells_toy.txt");
    ps.output_matter_power(DIR + "matter_power_toy.txt");
    std::cout << "Toy cosmology: done\n";
}

int main() {
    Utils::Timer timer("milestones");
    std::filesystem::create_directories(DIR);

    CosmologyParameters params = CosmologyParameters::fiducial();
    params.print();
    params.write(DIR + "params.txt");

    BackgroundCosmology cosmo(params);
    cosmo.solve();
    cosmo.info();

    milestone1(cosmo);
    milestone2(cosmo);
    milestone34(cosmo);
    toy_cosmology();

    timer.print();
    return 0;
}
