#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Spectra/HMcode.h>
#include <FML/ReACT/ReACT.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace FML::COSMOLOGY {
    using namespace Constants;

    void PowerSpectrum::solve_react() {
        if (!ReACT::available())
            throw std::runtime_error("nonlinear=react requires USE_REACT=true and the official backend (FML/ReACT/README.md)");
        const auto & p = cosmo.get_params();
        if (p.OmegaK != 0 || p.alpha_s != 0 || comps.size() != 1 ||
            std::any_of(p.iso_f.begin(), p.iso_f.end(), [](double f) { return f != 0; }))
            throw std::invalid_argument("ReACT solver integration requires a flat, adiabatic cosmology without running");
        if (options.react_n_z < 4 || options.react_n_k < 32 || options.react_mass_loop < 10 ||
            options.react_mass_loop > 1000 || !(options.react_z_max > 0 && options.react_z_max <= 2.5))
            throw std::invalid_argument("ReACT: require n_z >= 4, n_k >= 32, 10 <= mass_loop <= 1000, 0 < z_max <= 2.5");
        ReACT::Options ro;
        ro.mass_loop = options.react_mass_loop;
        const auto & model = cosmo.model_name();
        if (model == "fr") {
            if (p.model_param("fR_n") != 1)
                throw std::invalid_argument("ReACT supports Hu-Sawicki f(R) with fR_n=1 only");
            const double fR0 = p.model_param("fR0");
            if (fR0 > 0) {
                ro.model = ReACT::Model::HuSawicki;
                ro.parameters[0] = fR0;
            }
        } else if (model == "dgp") {
            if (p.model_param("dgp_branch") != -1 || p.model_param("dgp_lcdm_background") != 1)
                throw std::invalid_argument("ReACT nDGP requires dgp_branch=-1 and dgp_lcdm_background=1");
            const double rc = p.model_param("Omega_rc");
            if (rc > 0) {
                ro.model = ReACT::Model::NormalDGP;
                ro.parameters[0] = rc;
            }
        } else if (model == "w0wa") {
            ro.model = ReACT::Model::CPL;
            ro.parameters[0] = p.model_param("w0");
            ro.parameters[1] = p.model_param("wa");
        } else if (model != "lcdm") {
            throw std::invalid_argument("ReACT solver integration supports lcdm, fr, LCDM-background nDGP and w0wa");
        }
        if (cosmo.perturbation_model()->name() != model)
            throw std::invalid_argument("ReACT requires the standard perturbation model matching its cosmological parameters");

        ReACT::Input in;
        in.cosmology.h = p.h;
        in.cosmology.n_s = p.n_s;
        in.cosmology.A_s = p.A_s;
        in.cosmology.k_pivot = p.k_pivot;
        in.cosmology.Omega_b = cosmo.rhoB(0);
        in.cosmology.Omega_nu = cosmo.rhoNcdm(0);
        in.cosmology.Omega_m = cosmo.rhoCDM(0) + in.cosmology.Omega_b + in.cosmology.Omega_nu;
        // Uniform ln(a) table; public ReACT interface uses increasing redshift.
        for (double x : Utils::linspace(0, -std::log1p(options.react_z_max), options.react_n_z))
            in.z.push_back(std::expm1(-x));
        in.z.back() = options.react_z_max;
        in.k = Utils::logspace(1e-5, 100.0, options.react_n_k);
        const double length = Mpc / p.h, volume = length * length * length;
        auto fill = [&](const PowerSpectrum & ps, bool cold) {
            ReACT::Grid values;
            for (double z : in.z) {
                std::vector<double> row;
                for (double kh : in.k)
                    row.push_back((cold ? ps.get_matter_power_spectrum_cb_extrapolated(-std::log1p(z), kh / length) :
                                          ps.get_matter_power_spectrum_extrapolated(-std::log1p(z), kh / length)) / volume);
                values.push_back(std::move(row));
            }
            return values;
        };
        in.total = fill(*this, false);
        in.cold = fill(*this, true);
        if (model == "lcdm") {
            in.lcdm_cold = in.cold;
        } else {
            auto ref_params = p;
            ref_params.model = "lcdm";
            ref_params.model_params.clear();
            BackgroundCosmology ref_bg(ref_params);
            ref_bg.solve(cosmo.get_x_start(), cosmo.get_x_end());
            RecombinationHistory ref_rec(ref_bg, rec.get_options());
            ref_rec.solve();
            Perturbations ref_pert(ref_bg, ref_rec, pert.get_options());
            ref_pert.solve();
            PowerSpectrum ref_ps(ref_bg, ref_rec, ref_pert); // linear only; no recursive ReACT calculation
            in.lcdm_cold = fill(ref_ps, true);
        }
        auto result = ReACT::compute(in, ro);
        // The massless GR target is exactly its own pseudo cosmology. Avoid importing the small numerical
        // real/pseudo halo mismatch of the upstream engine into this known identity. Massive-neutrino GR
        // still needs its reaction, since its matched pseudo cosmology has no massive neutrinos.
        if (ro.model == ReACT::Model::GR && in.cosmology.Omega_nu == 0)
            for (auto & row : result.reaction)
                std::fill(row.begin(), row.end(), 1.0);
        auto reaction = std::make_unique<ReACT>(result);

        // The pseudo cosmology has GR/LCDM expansion and massless clustering matter of density Omega_m,total.
        // At each z it matches the TARGET total linear P(k). It is distinct from the LCDM auxiliary spectrum
        // above, which shares primordial parameters and generally has a different linear spectrum.
        auto pseudo_params = p;
        pseudo_params.model = "lcdm";
        pseudo_params.model_params.clear();
        pseudo_params.OmegaCDM += in.cosmology.Omega_nu;
        pseudo_params.m_nu = 0;
        pseudo_params.m_nu_list.clear();
        BackgroundCosmology pseudo_bg(pseudo_params);
        auto target = [this](double x, double k) { return get_matter_power_spectrum_extrapolated(x, k); };
        auto pseudo = std::make_unique<HMcode>(pseudo_bg, target, target);
        hmcode = std::move(pseudo);
        react = std::move(reaction);
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::cerr << "ReACT: for z > " << options.react_z_max
                      << ", using the matched HMcode-2020 pseudo baseline with reaction R=1 (outside ReACT's redshift range)\n";
    }
} // namespace FML::COSMOLOGY
