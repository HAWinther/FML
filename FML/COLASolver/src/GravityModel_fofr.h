#ifndef GRAVITYMODEL_FOFR_HEADER
#define GRAVITYMODEL_FOFR_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/MultigridSolver/FofrSolver.h>

/// f(R) model
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
    // Internal method: m^2(a) / H0^2
    //========================================================================
    double mass_over_H0_squared(double a) const {
        double OmegaM = this->cosmo->get_OmegaM();
        double OmegaLambda = 1.0 - OmegaM;
        double a3 = a * a * a;
        double fac = OmegaM / a3 + 4.0 * OmegaLambda;
        double fac0 = OmegaM + 4.0 * OmegaLambda;
        double mass2 = fac * std::pow(fac / fac0, nfofr + 1.0) / ((nfofr + 1.0) * fofr0);
        return mass2;
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        double mass2a2 = a * a * mass_over_H0_squared(a);
        double koverH02 = koverH0 * koverH0;
        return 1.0 + (1.0 / 3.0) * koverH02 / (koverH02 + mass2a2);
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta
    // and D^2 phi + m^2 phi = C delta where Phi = Phi_GR + phi
    // With screening C is a function of the newtonian potential
    //========================================================================
    void compute_screened_fifth_force(double a,
                                      double H0Box,
                                      double norm_poisson_equation,
                                      const typename GravityModelScreened<NDIM>::CouplingFunction & coupling,
                                      FFTWGrid<NDIM> & density_fourier,
                                      FFTWGrid<NDIM> & density_fifth_force) const override {
        // Critical threshold for screening in Hu-Sawicky (basically 3/2 f_R(a))
        const double OmegaM = this->cosmo->get_OmegaM();
        double PhiCrit = 1.5 * fofr0 *
                         std::pow((OmegaM + 4.0 * (1.0 - OmegaM)) / (1.0 / (a * a * a) * OmegaM + 4.0 * (1.0 - OmegaM)),
                                  nfofr + 1.0);

        auto screening_function_fofr = [=, eff = this->screening_efficiency](double PhiNewton) {
            double screenfac = std::abs(PhiCrit / PhiNewton) * eff;
            return screenfac > 1.0 ? 1.0 : screenfac;
        };

        if (FML::ThisTask == 0)
            std::cout << "Adding fifth-force f(R) (screening)\n";
        FML::NBODY::compute_delta_fifth_force_potential_screening(density_fourier,
                                                                  density_fifth_force,
                                                                  coupling,
                                                                  screening_function_fofr,
                                                                  norm_poisson_equation * std::pow(H0Box / a, 2));
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        // The solution is a^2 f_R / (2 H0Box^2)
        const bool verbose = true;
        FofrSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), nfofr, fofr0, H0Box, verbose);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.set_method(multigrid_method);
        mgsolver.solve(a, density_real, potential_real);
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
