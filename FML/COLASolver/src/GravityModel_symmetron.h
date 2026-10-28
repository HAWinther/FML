#ifndef GRAVITYMODEL_SYMMETRON_HEADER
#define GRAVITYMODEL_SYMMETRON_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/MultigridSolver/SymmetronSolver.h>

/// Symmetron model
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

    GravityModelSymmetron() : GravityModelScreened<NDIM>("Symmetron") {}
    GravityModelSymmetron(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "Symmetron")

    {
        assert(this->get_name() == "Symmetron");
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModelScreened<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# assb             : " << assb << "\n";
            std::cout << "# beta             : " << beta << "\n";
            std::cout << "# L_mpch           : " << L_mpch << "\n";
        }
        this->info_screening();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    // This is phi/phi0
    double phi_background(double a) const {
        return a < assb ? 0.0 : std::sqrt(1.0 - (assb * assb * assb) / (a * a * a));
    }

    //========================================================================
    // Internal method: m^2(a) / H0^2
    //========================================================================
    double mass_over_H0_squared(double a) const {
        double mu2H02 = 1.0 / (2.0 * this->H0_hmpc * this->H0_hmpc * L_mpch * L_mpch);
        double rhorhossb = (assb * assb * assb) / (a * a * a);
        return a < assb ? mu2H02 * (rhorhossb - 1.0) : 2.0 * mu2H02 * (1.0 - rhorhossb);
    }

    //========================================================================
    // Effective newtonian constant
    //========================================================================
    double GeffOverG(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        double mass2a2 = a * a * mass_over_H0_squared(a);
        double koverH02 = koverH0 * koverH0;
        double phi = phi_background(a);
        // No fifth force before symmetry breaking. At a = assb the mass is also zero so k = 0 would give 0 / 0
        if (phi == 0.0)
            return 1.0;
        return 1.0 + 2.0 * beta * beta * phi * phi * koverH02 / (koverH02 + mass2a2);
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
        const double OmegaM = this->cosmo->get_OmegaM();
        const double PhiCrit =
            a < assb ? 0.0 : 3.0 * OmegaM * (this->H0_hmpc * L_mpch) * (this->H0_hmpc * L_mpch) / (assb * assb * assb);
        auto screening_function_symmetron = [=, eff = this->screening_efficiency](double PhiNewton) {
            double screenfac = std::abs(PhiCrit / PhiNewton) * eff;
            return screenfac > 1.0 ? 1.0 : screenfac;
        };

        if (FML::ThisTask == 0)
            std::cout << "Adding fifth-force symmetron (screening)\n";
        FML::NBODY::compute_delta_fifth_force_potential_screening(density_fourier,
                                                                  density_fifth_force,
                                                                  coupling,
                                                                  screening_function_symmetron,
                                                                  norm_poisson_equation * std::pow(H0Box / a, 2));
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        // The solution is the potential C phi^2
        const bool verbose = true;
        SymmetronSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), assb, beta, L_mpch, H0Box, verbose);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.solve(a, density_real, potential_real);
    }

    std::string fifth_force_label() const override { return "symmetron"; }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        assb = param.get<double>("gravity_model_symmetron_assb");
        beta = param.get<double>("gravity_model_symmetron_beta");
        L_mpch = param.get<double>("gravity_model_symmetron_L_mpch");
        this->read_screening_parameters(param);
        this->read_multigrid_parameters(param, "gravity_model_symmetron_exact_solution");
        this->scaledependent_growth = true;
    }
};

#endif
