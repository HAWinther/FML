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

    explicit GravityModelFofR(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "f(R)", &SIMCODE::GravityModelSettings::fofr_exact_solution) {}

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
