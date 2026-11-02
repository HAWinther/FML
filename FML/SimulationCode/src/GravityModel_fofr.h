#ifndef GRAVITYMODEL_FOFR_HEADER
#define GRAVITYMODEL_FOFR_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_HuSawicky.h>
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

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    GravityModelFofR() : GravityModelScreened<NDIM>("f(R)") {}
    GravityModelFofR(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "f(R)")

    {
        assert(this->get_name() == "f(R)");
    }

    /// The model parameters (for the LPT predictions of the hybrid bias spectra)
    double get_fofr0() const { return fofr0; }
    double get_nfofr() const { return nfofr; }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModelScreened<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# fofr0            : " << fofr0 << "\n";
            std::cout << "# nfofr            : " << nfofr << "\n";
        }
        this->info_screening();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //========================================================================
    // The shared theory (built in init)
    //========================================================================
    using QuasiStaticGravity_HuSawicky = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_HuSawicky;
    const QuasiStaticGravity_HuSawicky & hu_sawicky() const {
        return static_cast<const QuasiStaticGravity_HuSawicky &>(this->qs());
    }
    void init() override {
        this->quasi_static = std::make_shared<const QuasiStaticGravity_HuSawicky>(this->cosmo->background(), fofr0, nfofr);
        GravityModelScreened<NDIM>::init();
    }

    /// m^2(a) / H0^2 (the scalaron mass, c = 1)
    double mass_over_H0_squared(double a) const { return hu_sawicky().mass2(std::log(a)); }

    //========================================================================
    // Effective newtonian constant: [1 + (1/3) k^2 / (k^2 + a^2 m^2)] / (1 + f_R)
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        return this->qs().mu(std::log(a), koverH0);
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

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        fofr0 = param.get<double>("gravity_model_fofr_fofr0");
        nfofr = param.get<double>("gravity_model_fofr_nfofr");
        // |f_R0| (a negative value gives m^2 < 0 and a singular Geff)
        FML::assert_mpi(fofr0 > 0.0, "gravity_model_fofr_fofr0 must be > 0 (it is |f_R0|, e.g. 1e-5)");
        this->read_screening_parameters(param);
        this->read_multigrid_parameters(param, "gravity_model_fofr_exact_solution");
        if (this->solve_exact_equation)
            multigrid_method = param.get<std::string>("gravity_model_fofr_multigrid_method");
        this->scaledependent_growth = true;
    }
};

#endif
