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

#include <FML/MultigridSolver/DGPSolver.h>

/// DGP model
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
    // Internal method: the beta function
    //========================================================================
    double get_beta_dgp(double a) const {
        double E = this->cosmo->HoverH0_of_a(a);
        double dlogHdloga = this->cosmo->dlogHdloga_of_a(a);
        return 1.0 + 2.0 * rcH0_DGP * E * (1.0 + dlogHdloga / 3.0);
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        return 1.0 + 1.0 / (3.0 * get_beta_dgp(a));
    }

    //========================================================================
    // For the 2LPT equation: the nonlinear (brane-bending) term of the field equation adds a source -eps D1^2 with
    //   eps = 2 rc^2 OmegaM / (9 a^3 beta^3 GeffOverG)
    // relative to the GeffOverG (D2 - D1^2) of the linear force, i.e. D2 - (1 - eps) D1^2. It multiplied D2 as well
    // (D2/D1^2 at z = 0 was 0.6% too small for rcH0 = 1). It is the Vainshtein screening, so it is only there when
    // the force has it (gravity_model_screening or the exact solution), as in the hybrid bias LPT kernels
    //========================================================================
    double nonlinear_source_factor_2LPT(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        if (not this->get_use_screening())
            return 1.0;
        const double betadgp = get_beta_dgp(a);
        return 1.0 - (2.0 * rcH0_DGP * rcH0_DGP * this->cosmo->get_OmegaM()) /
                         (9.0 * a * a * a * (betadgp * betadgp * betadgp) * (1 + 1.0 / (3.0 * betadgp)));
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta (GR)
    // and D^2 phi_DGP = C delta where Phi = Phi_GR + phi_DGP
    // With screening we compute C as a function of delta
    //========================================================================
    void compute_screened_fifth_force(double a,
                                      [[maybe_unused]] double H0Box,
                                      [[maybe_unused]] double norm_poisson_equation,
                                      const typename GravityModelScreened<NDIM>::CouplingFunction & coupling,
                                      FFTWGrid<NDIM> & density_fourier,
                                      FFTWGrid<NDIM> & density_fifth_force) const override {
        // Approximate screening method
        //
        // NB: this is not the spherical Vainshtein solution of the field equation. That solution (the one the
        // multigrid solver DGPSolver.h solves, reduced to spherical symmetry) is
        //   fac = 8 OmegaM (rc H0)^2 / (9 beta^2 a^3) * delta,
        // with the density contrast delta (and 1 + fac >= 0), while the code below uses 1 + delta (it is passed the
        // density rho/rhobar, see FML::NBODY::compute_delta_fifth_force_density_screening) and no 1/a^3: it does not
        // tend to 1 (no screening) for delta -> 0 (0.990 at a = 1 for rcH0 = 1) and screens much less at z > 0
        // (0.77 instead of 0.26 for delta = 100 at a = 1/3). It is kept as it is because it gives results closer to
        // the exact (multigrid) solution in simulations; the default smoothing of the density field over the whole
        // box (gravity_model_dgp_smoothing_scale_over_boxsize = 1) also makes the screening very weak.
        const double OmegaM = this->cosmo->get_OmegaM();
        auto screening_function_dgp = [=, eff = this->screening_efficiency](double density_contrast) {
            double fac = 8.0 * OmegaM * std::pow(rcH0_DGP * (GeffOverG(a) - 1.0), 2) * (density_contrast);
            fac *= eff;
            return fac < 1e-5 ? 1.0 : 2.0 * (std::sqrt(1.0 + fac) - 1) / fac;
        };

        FML::NBODY::compute_delta_fifth_force_density_screening(density_fourier,
                                                                density_fifth_force,
                                                                coupling,
                                                                screening_function_dgp,
                                                                smoothing_scale_over_boxsize,
                                                                smoothing_filter);
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        const bool verbose = true;
        DGPSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), rcH0_DGP, get_beta_dgp(a), H0Box, verbose);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.solve(a, density_real, potential_real);
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
