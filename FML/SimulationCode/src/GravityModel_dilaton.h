#ifndef GRAVITYMODEL_DILATON_HEADER
#define GRAVITYMODEL_DILATON_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_Dilaton.h>
#include <FML/MultigridSolver/DilatonSolver.h>

/// The (generalised) environment-dependent dilaton of Brax, Davis, Li, Winther & Zhao 2012 (arXiv:1206.3568):
/// mu(k, a), beta(a), the mass and the screening (Phi_crit = 1 / (2 A2)) from the shared quasi-static theory
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_Dilaton.h); the multigrid solver here
/// (FML/MultigridSolver/DilatonSolver.h)
template <int NDIM>
class GravityModelDilaton final : public GravityModelScreened<NDIM> {
  protected:
    double A2{5e5};
    double beta0{0.5};
    double r{1.0};
    double xi{1e-3};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;

    explicit GravityModelDilaton(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "Dilaton", &SIMCODE::GravityModelSettings::dilaton_exact_solution) {}

    //========================================================================
    // The shared theory (built in init): mu(k, a), beta(a), the mass and the screening
    //========================================================================
    using QuasiStaticGravity_Dilaton = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_Dilaton;
    const QuasiStaticGravity_Dilaton & dilaton() const {
        return static_cast<const QuasiStaticGravity_Dilaton &>(this->qs());
    }
    double GeffOverG_highk(double a) const override {
        const double b = dilaton().beta(std::log(a));
        return 1 + 2 * b * b;
    }
    double short_range_comoving_range(double a) const override {
        return 1 / (a * std::sqrt(dilaton().mass2(std::log(a))) * this->H0_hmpc);
    }
    void init() override {
        this->quasi_static =
            std::make_shared<const QuasiStaticGravity_Dilaton>(this->cosmo->background(), A2, beta0, r, xi);
        GravityModelScreened<NDIM>::init();
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        // The solution is the potential a^2 ln A(phi) / H0Box^2
        const bool verbose = false;
        DilatonSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), A2, beta0, r, xi, H0Box, verbose);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.set_relative_criterion(this->multigrid_relative_criterion);
        // Start from the previous solution (the same phi / phi_bar)
        if (this->multigrid_previous_a > 0.0)
            mgsolver.set_previous_solution(this->multigrid_previous_solution, this->multigrid_previous_a);
        mgsolver.solve(a, density_real, potential_real);
        if (FML::ThisTask == 0)
            std::cout << "Dilaton multigrid solve at a = " << a << ": " << mgsolver.get_number_of_vcycles()
                      << " V-cycles\n";
    }

    std::string fifth_force_label() const override { return "dilaton"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        A2 = this->settings.dilaton_A2;
        beta0 = this->settings.dilaton_beta0;
        r = this->settings.dilaton_r;
        xi = this->settings.dilaton_xi;
    }
};

#endif
