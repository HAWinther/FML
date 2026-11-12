#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/HaloModel/HMcode2020.h>
#include <FML/HaloModel/HaloModelReaction.h>
#include <FML/PerturbationTheory/MGOneLoop.h>
#include <FML/SphericalCollapse/ScreenedSphericalCollapse.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>

//=====================================================================================================================
// nonlinear = react: P_NL = R(k, z) P_NL,pseudo(k, z), the halo-model reaction of ReACT (Cataneo et al. 2019; Bose et
// al. 2020, 2021) computed natively (HaloModel/HaloModelReaction.h with the screened spherical collapse of
// SphericalCollapse/ScreenedSphericalCollapse.h and the modified-gravity one-loop P of PerturbationTheory/MGOneLoop.h
// for k*), times HMcode-2020 of the matched pseudo cosmology (LCDM background, GR, massless clustering matter with the
// target's total OmegaM, the target's total linear P at each z).
//
// f(R): the third-order screening source gamma3 is the correct one (Koyama, Taruya & Hiramatsu 2009 eq. 3.6; Bose &
// Koyama 2016 eqs. 2.14, 2.34). ACTio-ReACTio (a36dda02) uses 1/3 of it; react_compat_gamma3 = true reproduces that
// (R changes by up to ~3e-3 for |fR0| = 1e-5). Validation against ReACT: tests/test_halomodel_reaction.cpp,
// PerturbationTheory/tests/test_mg_oneloop.cpp; solver: EinsteinBoltzmannSolver/tests/test_react_physics.cpp.
//=====================================================================================================================

namespace FML::COSMOLOGY {
    using namespace Constants;

    void PowerSpectrum::solve_react() {
        namespace PT = FML::PERTURBATION;
        namespace SC = SPHERICALCOLLAPSE;
        using HALOMODEL::HaloModelReaction;
        const auto & p = cosmo.get_params();
        if (p.OmegaK != 0 || p.alpha_s != 0 || comps.size() != 1 ||
            std::any_of(p.iso_f.begin(), p.iso_f.end(), [](double f) { return f != 0; }))
            throw std::invalid_argument("ReACT solver integration requires a flat, adiabatic cosmology without running");
        if (options.react_n_z < 4 || options.react_n_k < 32 || options.react_mass_loop < 10 ||
            options.react_mass_loop > 1000 || !(options.react_z_max > 0 && options.react_z_max <= 10))
            throw std::invalid_argument("ReACT: require n_z >= 4, n_k >= 32, 10 <= mass_loop <= 1000, 0 < z_max <= 10");
        if (cosmo.perturbation_model()->name() != cosmo.model_name())
            throw std::invalid_argument("ReACT requires the standard perturbation model matching its cosmological parameters");

        const double Ob = cosmo.rhoB(0), Onu = cosmo.rhoNcdm(0);
        const double Ocb = cosmo.rhoCDM(0) + Ob, Om = Ocb + Onu;
        HaloModelReaction::Cosmology hc;
        hc.OmegaM = Om;
        hc.OmegaNu = Onu;
        HaloModelReaction::FifthForceOfRadius fifth_force;
        PT::MGKernelModel kernels; // the real one-loop kernels (modified gravity only)
        bool mg = false;
        const auto & model = cosmo.model_name();
        if (model == "fr") {
            if (p.model_param("fR_n") != 1)
                throw std::invalid_argument("ReACT supports Hu-Sawicki f(R) with fR_n=1 only");
            const double fR0 = std::abs(p.model_param("fR0"));
            if (fR0 > 0) {
                mg = true;
                fifth_force = [fR0, Om](double R) { return SC::fifth_force_hu_sawicki(fR0, 1.0, Om, R); };
                kernels = PT::mg_hu_sawicki(fR0, Om, Ocb, options.react_compat_gamma3);
            }
        } else if (model == "dgp") {
            if (p.model_param("dgp_branch") != -1 || p.model_param("dgp_lcdm_background") != 1)
                throw std::invalid_argument("ReACT nDGP requires dgp_branch=-1 and dgp_lcdm_background=1");
            const double rc = p.model_param("Omega_rc");
            if (rc > 0) {
                mg = true;
                SC::SphericalCollapseModel lcdm;
                lcdm.OmegaM = Om;
                lcdm.set_model_lcdm();
                const auto F = SC::fifth_force_ndgp(rc, lcdm);
                fifth_force = [F](double) { return F; };
                kernels = PT::mg_ndgp(rc, Om, Ocb);
            }
        } else if (model == "w0wa") {
            // The target background of the real halos (flat, the same OmegaM as the LCDM background of the pseudo)
            const double w0 = p.model_param("w0"), wa = p.model_param("wa");
            auto de = [=](double x) { return (1 - Om) * std::exp(-3 * (1 + w0 + wa) * x - 3 * wa * (1 - std::exp(x))); };
            hc.Eofx = [=](double x) { return std::sqrt(Om * std::exp(-3 * x) + de(x)); };
            hc.logEprimeofx = [=](double x) {
                const double m = Om * std::exp(-3 * x);
                return 0.5 * (-3 * m + de(x) * (-3 * (1 + w0 + wa) + 3 * wa * std::exp(x))) / (m + de(x));
            };
        } else if (model != "lcdm") {
            throw std::invalid_argument("ReACT solver integration supports lcdm, fr, LCDM-background nDGP and w0wa");
        }

        // The pseudo cosmology has GR/LCDM expansion and massless clustering matter of density Omega_m,total. At each
        // z it matches the TARGET total linear P(k). HMcode-2020 gives its nonlinear spectrum.
        auto pseudo_params = p;
        pseudo_params.model = "lcdm";
        pseudo_params.model_params.clear();
        pseudo_params.OmegaCDM += Onu;
        pseudo_params.m_nu = 0;
        pseudo_params.m_nu_list.clear();
        BackgroundCosmology pseudo_bg(pseudo_params);
        auto target = [this](double x, double k) { return get_matter_power_spectrum_extrapolated(x, k); };
        hmcode = std::make_unique<HALOMODEL::HMcode2020>(pseudo_bg, target, target);

        // A massless LCDM target is its own pseudo cosmology: R = 1 exactly
        if (model == "lcdm" && Onu == 0) {
            react.reset();
            return;
        }

        // Linear spectra in h/Mpc and (Mpc/h)^3; the LCDM cold spectrum with the target's primordial parameters and
        // neutrinos (for sigma(M) of the real halos) from a separate linear LCDM run
        const double length = Mpc / p.h, volume = length * length * length;
        std::unique_ptr<BackgroundCosmology> ref_bg;
        std::unique_ptr<RecombinationHistory> ref_rec;
        std::unique_ptr<Perturbations> ref_pert;
        std::unique_ptr<PowerSpectrum> ref_ps;
        const PowerSpectrum * lcdm_ps = this;
        if (model != "lcdm") {
            auto ref_params = p;
            ref_params.model = "lcdm";
            ref_params.model_params.clear();
            ref_bg = std::make_unique<BackgroundCosmology>(ref_params);
            ref_bg->solve(cosmo.get_x_start(), cosmo.get_x_end());
            ref_rec = std::make_unique<RecombinationHistory>(*ref_bg, rec.get_options());
            ref_rec->solve();
            ref_pert = std::make_unique<Perturbations>(*ref_bg, *ref_rec, pert.get_options());
            ref_pert->solve();
            ref_ps = std::make_unique<PowerSpectrum>(*ref_bg, *ref_rec, *ref_pert); // linear only
            lcdm_ps = ref_ps.get();
        }
        auto spectra_at = [&](double x) {
            HaloModelReaction::LinearSpectra S;
            S.P = [this, x, length, volume](double kh) {
                return get_matter_power_spectrum_extrapolated(x, kh / length) / volume;
            };
            S.Pcb = [this, x, length, volume](double kh) {
                return get_matter_power_spectrum_cb_extrapolated(x, kh / length) / volume;
            };
            S.Pcb_lcdm = [lcdm_ps, x, length, volume](double kh) {
                return lcdm_ps->get_matter_power_spectrum_cb_extrapolated(x, kh / length) / volume;
            };
            return S;
        };

        // Redshifts uniform in ln a (increasing z for the table), k in h/Mpc
        std::vector<double> z;
        for (double x : Utils::linspace(0, -std::log1p(options.react_z_max), options.react_n_z))
            z.push_back(std::expm1(-x));
        z.back() = options.react_z_max;
        const auto k = Utils::logspace(1e-5, 100.0, options.react_n_k);

        // The one-loop P(k0) of the real (P_cb, the model's kernels) and pseudo (P, GR kernels of the LCDM
        // background) cosmologies at all redshifts for k* (modified gravity only)
        HALOMODEL::HaloModelReactionOptions ho;
        ho.n_mass = options.react_mass_loop;
        std::vector<double> a(z.size()), loop_real, loop_pseudo;
        for (size_t i = 0; i < z.size(); i++)
            a[i] = 1.0 / (1.0 + z[z.size() - 1 - i]); // increasing a
        if (mg) {
            std::vector<HaloModelReaction::LinearSpectra> S;
            for (double aa : a)
                S.push_back(spectra_at(std::log(aa)));
            loop_real = PT::mg_one_loop(kernels, ho.k0, a, [&](double q, size_t i) { return S[i].Pcb(q); });
            loop_pseudo =
                PT::mg_one_loop(PT::mg_lcdm(Om, Ocb), ho.k0, a, [&](double q, size_t i) { return S[i].P(q); });
        }

        std::vector<std::vector<double>> R(z.size(), std::vector<double>(k.size()));
        for (size_t i = 0; i < a.size(); i++) {
            HaloModelReaction hm(hc, a[i], spectra_at(std::log(a[i])), fifth_force, ho);
            if (mg)
                hm.set_one_loop(loop_real[i], loop_pseudo[i]);
            auto & row = R[z.size() - 1 - i];
            for (size_t j = 0; j < k.size(); j++)
                row[j] = hm.reaction(k[j]);
        }
        react = std::make_unique<HALOMODEL::ReactionTable>(z, k, R);
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::cerr << "ReACT: for z > " << options.react_z_max
                      << ", using the matched HMcode-2020 pseudo baseline with reaction R=1 (react_z_max)\n";
    }
} // namespace FML::COSMOLOGY
