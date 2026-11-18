#include <FML/CosmologyFitting/Theory.h>
#include <FML/Cosmology/EinsteinBoltzmannSolver/EinsteinBoltzmannSolver.h>
#include <FML/Cosmology/Common.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_BransDicke.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_CubicGalileon.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
#include <FML/Cosmology/Perturbations/QuasiStaticGravity_MuSigma.h>

#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY::FITTING {

    using namespace Constants;

    Theory::Theory(const RunSettings & settings, Options options) : s(settings), opt(options) {
        RunSettings run = s; // the CMB is computed, lensed, to cmb_ell_max (only P(k) if no likelihood needs it)
        run.compute_cells = opt.cmb;
        run.ps.lensing = opt.cmb;
        run.ps.ell_max = opt.cmb_ell_max;
        ebs = std::make_unique<EinsteinBoltzmannSolver>(run);
        ebs->set_verbose(false);
        ebs->solve(EinsteinBoltzmannSolver::Stage::Background);
        bg = &ebs->background();
    }

    Theory::~Theory() = default;

    const RecombinationHistory & Theory::recombination() const {
        if (!rec) {
            RecombinationOptions o = s.rec;
            o.x_start = opt.rec_x_start;
            o.reionization = false; // z_* and z_d are defined by recombination alone (as in CAMB and CLASS)
            o.n_points = opt.rec_points;
            o.recfast_relerr = opt.rec_relerr;
            rec = std::make_unique<RecombinationHistory>(*bg, o);
            rec->solve();
        }
        return *rec;
    }

    double Theory::D_M(double z) const { return bg->transverse_distance_of_x(-std::log1p(z)) / Mpc; }
    double Theory::D_H(double z) const { return c / bg->H_of_x(-std::log1p(z)) / Mpc; }
    double Theory::D_V(double z) const {
        const double dm = D_M(z);
        return std::cbrt(z * dm * dm * D_H(z));
    }

    double Theory::omega_b() const { return s.cosmo.OmegaB * s.cosmo.h * s.cosmo.h; }
    double Theory::omega_m() const {
        return (s.cosmo.OmegaB + s.cosmo.OmegaCDM + s.cosmo.OmegaNcdm()) * s.cosmo.h * s.cosmo.h;
    }

    // r_s(z) = int c_s deta, c_s = c / sqrt(3 (1 + R)), R = 3 rho_b / (4 rho_gamma) (cosmomc: R = 3e4 a omega_b, the
    // approximation of CAMB's CosmomcTheta); below x = bg_x_start c_s = c/sqrt(3)
    double Theory::sound_horizon_bg(double z, bool cosmomc) const {
        const double x1 = -std::log1p(z), x0 = s.bg_x_start, wb = omega_b();
        auto f = [&](double x) {
            const double Rb = cosmomc ? 3.0e4 * std::exp(x) * wb : 3.0 * bg->rhoB(x) / (4.0 * bg->rhoGamma(x));
            return c / std::sqrt(3.0 * (1.0 + Rb)) / bg->Hp_of_x(x);
        };
        return (bg->eta_of_x(x0) / std::sqrt(3.0) + Utils::integrate(f, x0, x1, 0.0, 1e-10)) / Mpc;
    }

    void Theory::compute_star() const {
        if (zs >= 0.0)
            return;
        if (opt.sound_horizon == Options::SoundHorizon::Fit) {
            // Hu & Sugiyama (1996)
            const double wb = omega_b(), wm = omega_m();
            const double g1 = 0.0783 * std::pow(wb, -0.238) / (1.0 + 39.5 * std::pow(wb, 0.763));
            const double g2 = 0.560 / (1.0 + 21.1 * std::pow(wb, 1.81));
            zs = 1048.0 * (1.0 + 0.00124 * std::pow(wb, -0.738)) * (1.0 + g1 * std::pow(wm, g2));
            rs = sound_horizon_bg(zs);
        } else {
            const auto & r = recombination();
            const double x = r.x_last_scattering();
            zs = std::expm1(-x);
            rs = r.sound_horizon_of_x(x) / Mpc;
        }
    }

    void Theory::compute_drag() const {
        if (zd >= 0.0)
            return;
        if (opt.sound_horizon == Options::SoundHorizon::Fit) {
            // Aubourg et al. (2015), eq. (16); z_d is not defined by the fit
            const double h2 = s.cosmo.h * s.cosmo.h, wnu = s.cosmo.OmegaNcdm() * h2;
            const double wcb = (s.cosmo.OmegaB + s.cosmo.OmegaCDM) * h2, wb = omega_b();
            rd =
                55.154 * std::exp(-72.3 * std::pow(wnu + 0.0006, 2)) / (std::pow(wcb, 0.25351) * std::pow(wb, 0.12807));
            zd = 0.0;
            return;
        }
        // baryon optical depth tau_b(x) = int_x^0 (-tau' 4 rho_gamma/(3 rho_b)) dx', tabulated and splined
        const auto & r = recombination();
        const int n = 6001;
        const double xa = -8.5;
        const auto xs = Utils::linspace(xa, 0.0, n);
        std::vector<double> f(n), tb(n, 0.0);
        for (int i = 0; i < n; i++)
            f[i] = -r.dtaudx_of_x(xs[i]) * r.R_of_x(xs[i]);
        for (int i = n - 2; i >= 0; i--) { // Simpson on each interval with the midpoint
            const double xm = 0.5 * (xs[i] + xs[i + 1]);
            const double fm = -r.dtaudx_of_x(xm) * r.R_of_x(xm);
            tb[i] = tb[i + 1] + (xs[i + 1] - xs[i]) / 6.0 * (f[i] + 4.0 * fm + f[i + 1]);
        }
        if (tb[0] < 1.0)
            throw std::runtime_error("Theory: baryon drag optical depth < 1 at x = -8.5");
        Spline1D sp(xs, tb, "tau_b");
        const double x = Utils::find_root([&](double xx) { return sp(xx) - 1.0; }, xa, -5.0, 1e-12, 1e-12);
        zd = std::expm1(-x);
        rd = r.sound_horizon_of_x(x) / Mpc;
    }

    double Theory::z_star() const {
        compute_star();
        return zs;
    }
    double Theory::r_star() const {
        compute_star();
        return rs;
    }
    double Theory::z_drag() const {
        compute_drag();
        return zd;
    }
    double Theory::r_drag() const {
        compute_drag();
        return rd;
    }
    // --- Chen, Huang & Wang (2019) conventions for the distance priors ---------------------------------
    double Theory::E_chen(double z) const {
        const double x = -std::log1p(z), a = std::exp(x), h2 = s.cosmo.h * s.cosmo.h, T = s.cosmo.TCMB;
        const double wm = omega_m(), Om = wm / h2;
        const double zeq = 2.5e4 * wm * std::pow(T / 2.7, -4), Or = Om / (1.0 + zeq);
        const double E2 = bg->E2_of_x(x) - bg->rhoGamma(x) - bg->rhoNu(x) - bg->rhoNcdm(x) + Or / (a * a * a * a) +
                          s.cosmo.OmegaNcdm() / (a * a * a);
        return std::sqrt(E2);
    }
    double Theory::D_M_chen(double z) const {
        const double dH0 = c / bg->get_H0() / Mpc; // c/H0 [Mpc]
        const double chi = Utils::integrate([&](double zz) { return 1.0 / E_chen(zz); }, 0.0, z, 0.0, 1e-11);
        const double ok = s.cosmo.OmegaK, q = std::sqrt(std::fabs(ok));
        if (ok > 0.0)
            return dH0 * std::sinh(q * chi) / q;
        if (ok < 0.0)
            return dH0 * std::sin(q * chi) / q;
        return dH0 * chi;
    }
    double Theory::r_s_chen(double z) const {
        const double T = s.cosmo.TCMB, wb = omega_b(), dH0 = c / bg->get_H0() / Mpc;
        auto f = [&](double a) {
            return 1.0 /
                   (a * a * E_chen(1.0 / a - 1.0) * std::sqrt(3.0 * (1.0 + 31500.0 * wb * std::pow(T / 2.7, -4) * a)));
        };
        return dH0 * Utils::integrate(f, 1e-9, 1.0 / (1.0 + z), 0.0, 1e-11);
    }
    double Theory::R_shift() const {
        if (R_c < 0.0)
            R_c = std::sqrt(s.cosmo.OmegaM() + s.cosmo.OmegaNcdm()) * D_M_chen(z_star()) / (c / bg->get_H0() / Mpc);
        return R_c;
    }
    double Theory::l_A() const {
        if (lA_c < 0.0)
            lA_c = M_PI * D_M_chen(z_star()) / r_s_chen(z_star());
        return lA_c;
    }

    // --- acoustic angles -----------------------------------------------------------------------------------
    double Theory::theta_MC() const {
        const double wb = omega_b(), wm = omega_m();
        const double g1 = 0.0783 * std::pow(wb, -0.238) / (1.0 + 39.5 * std::pow(wb, 0.763));
        const double g2 = 0.560 / (1.0 + 21.1 * std::pow(wb, 1.81));
        const double z = 1048.0 * (1.0 + 0.00124 * std::pow(wb, -0.738)) * (1.0 + g1 * std::pow(wm, g2));
        return 100.0 * sound_horizon_bg(z, true) / D_M(z);
    }
    double Theory::theta_star() const { return 100.0 * r_star() / D_M(z_star()); }

    // --- CMB -------------------------------------------------------------------------------------------------
    const RecombinationHistory & Theory::recombination_reionized() const {
        ebs->solve(EinsteinBoltzmannSolver::Stage::Recombination);
        return ebs->recombination();
    }

    double Theory::tau_reio() const {
        const auto & r = recombination_reionized();
        return r.has_reionization() ? r.tau_reionization() : 0.0;
    }

    const CMBSpectra & Theory::cmb() const {
        if (spectra)
            return *spectra;
        if (!opt.cmb)
            throw std::logic_error("Theory::cmb: the theory was set up without the CMB (TheoryOptions::cmb = false)");
        ebs->solve();
        const auto & ps = ebs->power_spectrum();
        auto c = std::make_unique<CMBSpectra>();
        const int L = opt.cmb_ell_max;
        c->ell_max = L;
        for (auto * v : {&c->tt, &c->ee, &c->te, &c->bb, &c->pp})
            v->assign(L + 1, 0.0);
        for (int l = 2; l <= L; l++) {
            c->tt[l] = ps.get_Dell_TT_lensed(l);
            c->ee[l] = ps.get_Dell_EE_lensed(l);
            c->te[l] = ps.get_Dell_TE_lensed(l);
            c->bb[l] = ps.get_Dell_BB_lensed(l);
            const double ll = double(l) * (l + 1.0);
            c->pp[l] = ll * ll * ps.get_Cell_phiphi(l) / (2.0 * M_PI);
        }
        sigma8_ = ps.get_sigma8(0.0);
        compute_growth();
        spectra = std::move(c);
        // the spectra are all that is needed: free the perturbations (the theory may be cached by the sampler)
        ebs->reset(EinsteinBoltzmannSolver::Stage::Perturbations);
        return *spectra;
    }

    // sigma_8 from the matter power spectrum of the CMB run (linear). With pert_k_max = 0.3/Mpc, P(k) above k_max is
    // extrapolated (PowerSpectrum::get_sigma_R); at the Planck best fit with precision = fast sigma_8 = 0.8114
    // (Planck 0.8111 +- 0.0060, tests/test_fitting.cpp).
    double Theory::sigma8() const {
        if (opt.cmb)
            cmb();
        else
            solve_matter();
        return sigma8_;
    }

    // --- growth: sigma_8(z) and f sigma_8(z) ---------------------------------------------------------------------
    void Theory::compute_growth() const {
        if (growth_done)
            return;
        const auto & ps = ebs->power_spectrum();
        sigma8cb_z.clear();
        fsigma8_z.clear();
        sigma8th_z.clear();
        for (double z : opt.growth_z) {
            const double x = -std::log1p(z);
            sigma8cb_z.push_back(ps.get_sigma8_cb(x));
            fsigma8_z.push_back(ps.get_fsigma8(x));
            sigma8th_z.push_back(ps.get_sigma8_theta(x));
        }
        pk_tables.clear();
        const double h = s.cosmo.h, unit = Mpc / h; // k [h/Mpc] -> [1/m]: k h/Mpc; P [m^3] -> [(Mpc/h)^3]
        for (double z : opt.pk_z) {
            const double x = -std::log1p(z);
            std::vector<double> P;
            for (double k : pk_k_grid())
                P.push_back(ps.get_matter_power_spectrum_cb_extrapolated(x, k / unit) / (unit * unit * unit));
            pk_tables.push_back(P);
        }
        pk_nl_tables.clear();
        growth_D_z.clear();
        const double k_D = 1e-4 / unit, P0 = ps.get_matter_power_spectrum_extrapolated(0.0, k_D);
        for (double z : opt.pk_nl_z) {
            const double x = -std::log1p(z);
            std::vector<double> P;
            for (double k : pk_nl_k_grid())
                P.push_back(ps.get_matter_power_spectrum_nonlinear(x, k / unit) / (unit * unit * unit));
            pk_nl_tables.push_back(P);
            growth_D_z.push_back(std::sqrt(ps.get_matter_power_spectrum_extrapolated(x, k_D) / P0));
        }
        growth_done = true;
    }

    // Without the CMB: the perturbations and P(k) (compute_cells = false), the growth, then free the perturbations
    void Theory::solve_matter() const {
        if (growth_done)
            return;
        ebs->solve();
        sigma8_ = ebs->power_spectrum().get_sigma8(0.0);
        compute_growth();
        ebs->reset(EinsteinBoltzmannSolver::Stage::Perturbations);
    }

    size_t Theory::growth_index(double z) const {
        for (size_t i = 0; i < opt.growth_z.size(); i++)
            if (std::abs(opt.growth_z[i] - z) < 1e-9)
                return i;
        throw std::logic_error("Theory: sigma8(z) / fsigma8(z) at z = " + std::to_string(z) +
                               " which is not in TheoryOptions::growth_z");
    }

    void Theory::ensure_growth() const {
        if (opt.cmb)
            cmb();
        else
            solve_matter();
    }

    double Theory::sigma8_cb(double z) const {
        const size_t i = growth_index(z);
        ensure_growth();
        return sigma8cb_z[i];
    }

    double Theory::fsigma8(double z) const {
        const size_t i = growth_index(z);
        ensure_growth();
        return fsigma8_z[i];
    }

    double Theory::sigma8_theta(double z) const {
        const size_t i = growth_index(z);
        ensure_growth();
        return sigma8th_z[i];
    }

    const std::vector<double> & Theory::pk_k_grid() {
        static const std::vector<double> k = [] {
            std::vector<double> v(1024);
            for (int i = 0; i < 1024; i++)
                v[i] = std::pow(10.0, -7.0 + 9.0 * i / 1023.0);
            return v;
        }();
        return k;
    }

    double Theory::lensing_Sigma(double z) const {
        namespace P = PERTURBATIONS;
        const double x = -std::log1p(z);
        const std::string & m = s.cosmo.model;
        const auto & b = bg->background();
        if (m == "jbd")
            return P::QuasiStaticGravity_BransDicke(b).Sigma(x, 0.0);
        if (m == "musigma")
            return P::QuasiStaticGravity_MuSigma(b, s.cosmo.model_param("mu0"), s.cosmo.model_param("Sigma0")).Sigma(x, 0.0);
        if (m == "cubic_galileon")
            return P::QuasiStaticGravity_CubicGalileon(b).Sigma(x, 0.0);
        if (m == "fr")
            return P::QuasiStaticGravity_HuSawicky(b, s.cosmo.model_param("fR0"), s.cosmo.model_param("fR_n")).Sigma(x, 0.0);
        if (m == "horndeski" || m == "horndeski_table")
            throw std::invalid_argument("Theory::lensing_Sigma: not implemented for " + m);
        return 1.0;
    }

    const std::vector<double> & Theory::pk_nl_k_grid() {
        static const std::vector<double> k = [] {
            std::vector<double> v(400);
            for (int i = 0; i < 400; i++)
                v[i] = std::pow(10.0, -5.0 + 7.0 * i / 399.0);
            return v;
        }();
        return k;
    }

    size_t Theory::pk_nl_index(double z) const {
        for (size_t i = 0; i < opt.pk_nl_z.size(); i++)
            if (std::abs(opt.pk_nl_z[i] - z) < 1e-9)
                return i;
        throw std::logic_error("Theory: pk_nl / growth_D at z = " + std::to_string(z) +
                               " which is not in TheoryOptions::pk_nl_z");
    }

    const std::vector<double> & Theory::pk_nl(double z) const {
        const size_t i = pk_nl_index(z);
        ensure_growth();
        return pk_nl_tables[i];
    }

    double Theory::growth_D(double z) const {
        const size_t i = pk_nl_index(z);
        ensure_growth();
        return growth_D_z[i];
    }

    const std::vector<double> & Theory::pk_cb(double z) const {
        for (size_t i = 0; i < opt.pk_z.size(); i++)
            if (std::abs(opt.pk_z[i] - z) < 1e-9) {
                ensure_growth();
                return pk_tables[i];
            }
        throw std::logic_error("Theory::pk_cb at z = " + std::to_string(z) + " which is not in TheoryOptions::pk_z");
    }

} // namespace FML::COSMOLOGY::FITTING
