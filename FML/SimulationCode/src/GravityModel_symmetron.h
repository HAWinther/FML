#ifndef GRAVITYMODEL_SYMMETRON_HEADER
#define GRAVITYMODEL_SYMMETRON_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_Symmetron.h>
#include <FML/MultigridSolver/SymmetronSolver.h>

/// Symmetron: mu(k, a), the background field, the mass and the screening from the shared quasi-static theory
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_Symmetron.h); the multigrid solver here
template <int NDIM>
class GravityModelSymmetron final : public GravityModelScreened<NDIM> {
  protected:
    double assb{0.5};
    double beta{1.0};
    double L_mpch{1.0};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    explicit GravityModelSymmetron(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "Symmetron", &SIMCODE::GravityModelSettings::symmetron_exact_solution) {}

    //========================================================================
    // Print some info
    //========================================================================
    void info_model() const override {
        if (FML::ThisTask == 0) {
            std::cout << "# assb             : " << assb << "\n";
            std::cout << "# beta             : " << beta << "\n";
            std::cout << "# L_mpch           : " << L_mpch << "\n";
        }
        this->info_screening();
    }

    //========================================================================
    // The shared theory (built in init): mu(k, a), phi(a), the mass and the screening
    //========================================================================
    using QuasiStaticGravity_Symmetron = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_Symmetron;
    const QuasiStaticGravity_Symmetron & symmetron() const {
        return static_cast<const QuasiStaticGravity_Symmetron &>(this->qs());
    }
    void init() override {
        this->quasi_static =
            std::make_shared<const QuasiStaticGravity_Symmetron>(this->cosmo->background(), assb, beta, L_mpch);
        GravityModelScreened<NDIM>::init();
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        // The solution is the potential C phi^2
        const bool verbose = false;
        SymmetronSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), assb, beta, L_mpch, H0Box, verbose);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.set_relative_criterion(this->multigrid_relative_criterion);
        // (No warm start from the previous solution as for f(R) and DGP: it gave the same number of V-cycles in a test
        // with 64^3 cells, Nov 2026, and the sign of phi is lost in the returned potential)
        mgsolver.solve(a, density_real, potential_real);
        if (FML::ThisTask == 0)
            std::cout << "Symmetron multigrid solve at a = " << a << ": " << mgsolver.get_number_of_vcycles()
                      << " V-cycles\n";
    }

    std::string fifth_force_label() const override { return "symmetron"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        assb = this->settings.symmetron_assb;
        beta = this->settings.symmetron_beta;
        L_mpch = this->settings.symmetron_L_mpch;
    }
};

#endif
