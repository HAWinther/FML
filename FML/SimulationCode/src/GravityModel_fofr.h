#ifndef GRAVITYMODEL_FOFR_HEADER
#define GRAVITYMODEL_FOFR_HEADER

#include "GravityModel_Screened.h"
#include "SimulationGrid.h"
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
#include <FML/Emulator/EMantis/EMantis.h> // GPL-3.0-or-later (port of e-MANTIS): see Emulator/EMantis/README.md
#include <FML/MultigridSolver/FofrSolver.h>

/// Hu-Sawicky f(R): mu(k, a), f_R(a) and the scalaron mass from the shared quasi-static theory
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h, R of the cosmology's background), and its screening
/// (applied to the grid in GravityModelScreened); the multigrid solver here
template <int NDIM>
class GravityModelFofR final : public GravityModelScreened<NDIM> {
  protected:
    double fofr0{0.0};
    double nfofr{1.0};

    std::string multigrid_method{"sqrt"}; // The field the multigrid solver solves for (sqrt or log)

    // The nonlinear boost P_f(R)/P_GR: the e-MANTIS emulator (n = 1 only; null if not available) and its parameters
    std::shared_ptr<const FML::EMULATOR::EMANTIS::NonLinearMGBoostEmulator> emantis;
    FML::EMULATOR::EMANTIS::Parameters emantis_params;
    std::string emantis_status; // why there is no boost, or what was clamped (for info_model)

  public:
    template <int N>
    using FFTWGrid = SimulationGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    explicit GravityModelFofR(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "f(R)", &SIMCODE::GravityModelSettings::fofr_exact_solution) {}

    //========================================================================
    // The shared theory (built in init)
    //========================================================================
    using QuasiStaticGravity_HuSawicky = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_HuSawicky;
    const QuasiStaticGravity_HuSawicky & hu_sawicky() const {
        return static_cast<const QuasiStaticGravity_HuSawicky &>(this->qs());
    }
    double GeffOverG_highk(double a) const override {
        const double fR = hu_sawicky().fR(std::log(a));
        return fR == 0 ? 1. : (4. / 3.) / (1 + fR);
    }
    double short_range_comoving_range(double a) const override {
        return hu_sawicky().compton_wavelength(std::log(a)) / (a * this->H0_hmpc);
    }
    void init() override {
        this->quasi_static =
            std::make_shared<const QuasiStaticGravity_HuSawicky>(this->cosmo->background(), fofr0, nfofr);
        GravityModelScreened<NDIM>::init();
        init_emantis();
    }

    /// The e-MANTIS f(R) boost (Saez-Casares et al., arXiv:2303.08899): Hu-Sawicky n = 1, 1e-7 <= |fR0| <= 1e-4.
    /// The cosmology is Omega_b, Omega_m, h, n_s and sigma8_lcdm, the sigma8 of LCDM with the same A_s (from A_s with
    /// e-MANTIS' own sigma8 emulator, A_s converted to the pivot 0.05/Mpc). Parameters outside the emulation range are
    /// clamped to it (extrapolate_cosmo)
    void init_emantis() {
        namespace EM = FML::EMULATOR::EMANTIS;
        emantis.reset();
        emantis_status.clear();
        if (nfofr != 1.0) {
            emantis_status = "none (the e-MANTIS f(R) emulator is for n = 1)";
            return;
        }
        double logfR0 = -std::log10(std::abs(fofr0));
        if (logfR0 < 4.0 - 1e-10 or logfR0 > 7.0 + 1e-10) {
            emantis_status = "none (|fR0| outside the e-MANTIS range 1e-7 - 1e-4)";
            return;
        }
        logfR0 = std::clamp(logfR0, 4.0, 7.0); // (rounding at the ends)
        try {
            auto emu = std::make_shared<EM::NonLinearMGBoostEmulator>("fR");
            EM::Parameters p;
            p.Omega_b = this->cosmo->get_Omegab();
            p.Omega_m = this->cosmo->get_OmegaM();
            p.h = this->cosmo->get_h();
            p.n_s = this->cosmo->get_ns();
            p.logfR0 = logfR0;
            p.A_s = this->cosmo->get_As() * std::pow(0.05 / this->cosmo->get_kpivot_mpc(), p.n_s - 1.0);
            EM::Sigma8Emulator s8("LCDM");
            p.sigma8 = s8.sigma8(p);
            p.A_s = NAN;
            // The parameters that will be clamped to the emulation range
            std::string clamped;
            for (auto [name, value] : {std::pair<std::string, double>{"Omega_b", p.Omega_b},
                                       {"Omega_m", p.Omega_m},
                                       {"h", p.h},
                                       {"n_s", p.n_s},
                                       {"sigma8_lcdm", p.sigma8}}) {
                auto r = emu->range(name);
                if (value < r.first or value > r.second)
                    clamped += (clamped.empty() ? "" : ", ") + name + " = " + std::to_string(value);
            }
            if (not clamped.empty())
                emantis_status = "[outside the e-MANTIS range, clamped: " + clamped + "]";
            emantis_params = p;
            emantis = emu;
        } catch (std::exception & e) {
            emantis_status = std::string("none (e-MANTIS: ") + e.what() + ")";
        }
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        // The solution is a^2 f_R / (2 H0Box^2)
        const bool verbose = false;
        FofrSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), nfofr, fofr0, H0Box, verbose);
        // The background curvature of the theory (the cosmology's background) instead of the solver's flat LCDM
        const auto & hs = hu_sawicky();
        mgsolver.set_background_ricci([&hs](double aa) { return hs.R(std::log(aa)); });
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.set_relative_criterion(this->multigrid_relative_criterion);
        mgsolver.set_method(multigrid_method);
        // Start from the previous solution (rescaled to the new background)
        if (this->multigrid_previous_a > 0.0)
            mgsolver.set_previous_solution(this->multigrid_previous_solution, this->multigrid_previous_a);
        mgsolver.solve(a, density_real, potential_real);
        if (FML::ThisTask == 0)
            std::cout << "f(R) multigrid solve at a = " << a << ": " << mgsolver.get_number_of_vcycles()
                      << " V-cycles\n";
    }

    std::string fifth_force_label() const override { return "f(R)"; }

    /// The nonlinear P_f(R)/P_GR (same initial conditions) of the e-MANTIS emulator (FML/Emulator/EMantis):
    /// Hu-Sawicky n = 1, 1e-7 <= |fR0| <= 1e-4, z <= 3 (none at higher redshifts), k <= 9.7 h/Mpc (the value there
    /// above it), extrapolated below its first bin (0.029 h/Mpc) as in e-MANTIS
    std::function<double(double)> nonlinear_boost_over_GR(double redshift) const override {
        if (not emantis)
            return {};
        const double a = 1.0 / (1.0 + redshift);
        if (a < emantis->range("aexp").first)
            return {};
        const double kmax = emantis->range("k").second;
        return [emu = emantis, p = emantis_params, a, kmax](double k) {
            return emu->boost(p, a, {std::min(k, kmax)}, true, true)[0];
        };
    }
    void info_model() const override {
        GravityModelScreened<NDIM>::info_model();
        if (FML::ThisTask != 0)
            return;
        if (not emantis)
            std::cout << "# Nonlinear P(k) prediction: " << emantis_status << "\n";
        else
            std::cout << "# Nonlinear P(k) prediction: e-MANTIS f(R) boost (Saez-Casares et al.; z <= 3), "
                      << "sigma8_lcdm = " << emantis_params.sigma8
                      << (emantis_status.empty() ? "" : " " + emantis_status) << "\n";
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        fofr0 = this->settings.fofr_fofr0;
        nfofr = this->settings.fofr_nfofr;
        if (this->solve_exact_equation)
            multigrid_method = this->settings.fofr_multigrid_method;
    }
};

#endif
