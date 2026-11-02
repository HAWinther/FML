#ifndef GRAVITYMODEL_DGP_HEADER
#define GRAVITYMODEL_DGP_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/NBody/NBody.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_DGP.h>
#include <FML/MultigridSolver/DGPSolver.h>

/// The normal branch of DGP (nDGP): beta(a), mu(a) and the second-order source from the shared quasi-static theory
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_DGP.h, H(a) of the cosmology's background), and its screening
/// (applied to the grid in GravityModelScreened); the multigrid solver here
template <int NDIM>
class GravityModelDGP final : public GravityModelScreened<NDIM> {
  protected:
    double rcH0_DGP;

    // For screening method
    std::string smoothing_filter{"tophat"};
    double smoothing_scale_over_boxsize{0.0};


  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    /// The model parameters (for the LPT predictions of the hybrid bias spectra)
    double get_rcH0() const { return rcH0_DGP; }

    GravityModelDGP() : GravityModelScreened<NDIM>("DGP") {}
    GravityModelDGP(std::shared_ptr<Cosmology> cosmo) : GravityModelScreened<NDIM>(cosmo, "DGP") {}

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModelScreened<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# rcH0             : " << rcH0_DGP << "\n";
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
    using QuasiStaticGravity_DGP = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_DGP;
    const QuasiStaticGravity_DGP & dgp() const { return static_cast<const QuasiStaticGravity_DGP &>(this->qs()); }
    void init() override {
        this->quasi_static = std::make_shared<const QuasiStaticGravity_DGP>(this->cosmo->background(), rcH0_DGP, -1);
        GravityModelScreened<NDIM>::init();
    }

    /// beta(a) = 1 + 2 rcH0 E (1 + dlogH/dloga / 3)
    double get_beta_dgp(double a) const { return dgp().beta(std::log(a)); }

    //========================================================================
    // Effective newtonian constant 1 + 1/(3 beta)
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        return this->qs().mu(std::log(a), koverH0);
    }

    //========================================================================
    // For the 2LPT equation: the nonlinear (brane-bending) term of the field equation adds a source -eps D1^2 with
    //   eps = 2 rc^2 rho_m(a) / (9 beta^3 GeffOverG)
    // relative to the GeffOverG (D2 - D1^2) of the linear force, i.e. D2 - (1 - eps) D1^2. It multiplied D2 as well
    // (D2/D1^2 at z = 0 was 0.6% too small for rcH0 = 1). It is the Vainshtein screening, so it is only there when
    // the force has it (gravity_model_screening or the exact solution), as in the hybrid bias LPT kernels
    //========================================================================
    double nonlinear_source_factor_2LPT(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        if (not this->get_use_screening())
            return 1.0;
        return this->qs().second_order_source(std::log(a), koverH0);
    }

    //========================================================================
    // The screening approximation is the theory's (QuasiStaticGravity_DGP::screening(): the spherical Vainshtein
    // factor with rho/rho_mean in place of delta, see there), applied in GravityModelScreened to the density field
    // smoothed on gravity_model_dgp_smoothing_scale_over_boxsize (0: no smoothing; 1, the old default, makes the
    // screening very weak)
    //========================================================================
    double density_smoothing_scale_over_boxsize() const override { return smoothing_scale_over_boxsize; }
    std::string density_smoothing_filter() const override { return smoothing_filter; }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        const bool verbose = false;
        DGPSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), rcH0_DGP, get_beta_dgp(a), H0Box, verbose);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        // Start from the previous solution
        if (this->multigrid_previous_a > 0.0)
            mgsolver.set_previous_solution(this->multigrid_previous_solution);
        mgsolver.solve(a, density_real, potential_real);
        if (FML::ThisTask == 0)
            std::cout << "DGP multigrid solve at a = " << a << ": " << mgsolver.get_number_of_vcycles()
                      << " V-cycles\n";
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        rcH0_DGP = param.get<double>("gravity_model_dgp_rcH0overc");
        this->read_screening_parameters(param);
        if (this->use_screening_method) {
            smoothing_scale_over_boxsize = param.get<double>("gravity_model_dgp_smoothing_scale_over_boxsize");
            smoothing_filter = param.get<std::string>("gravity_model_dgp_smoothing_filter");
        }
        this->read_multigrid_parameters(param, "gravity_model_dgp_exact_solution");
        this->scaledependent_growth = this->cosmo->get_OmegaMNu() > 0.0;
    }
};

#endif
