#ifndef GRAVITYMODEL_CUBICGALILEON_HEADER
#define GRAVITYMODEL_CUBICGALILEON_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_CubicGalileon.h>
#include <FML/MultigridSolver/DGPSolver.h>

/// The cubic Galileon on its tracker (cosmology_model = CubicGalileon, which this model needs: the background of the
/// Galileon). Its quasi-static field equation is that of the DGP brane-bending mode with a time-dependent cross-over
/// scale r_c(a) and beta(a) (FML/Cosmology/Perturbations/QuasiStaticGravity_CubicGalileon.h): G_eff/G = 1 + 1/(3 beta)
/// = 1.77 today, the Vainshtein screening of the theory (gravity_model_screening, a spherical top-hat factor of the
/// local density) or the exact solution with DGP's multigrid solver (gravity_model_cubic_galileon_exact_solution).
/// N-body simulations of the model: Barreira et al. 2013, arXiv:1306.3219 (full N-body); Wright et al. 2023 (Hi-COLA),
/// arXiv:2209.01666, and Brando, Koyama & Winther 2023, arXiv:2303.09549 (COLA with approximate screening)
template <int NDIM>
class GravityModelCubicGalileon final : public GravityModelScreened<NDIM> {
    std::string smoothing_filter{"tophat"};
    double smoothing_scale_over_boxsize{0.0};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using QuasiStaticGravity_CubicGalileon = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_CubicGalileon;

    explicit GravityModelCubicGalileon(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo,
                                     "CubicGalileon",
                                     &SIMCODE::GravityModelSettings::cubic_galileon_exact_solution) {}

    const QuasiStaticGravity_CubicGalileon & galileon() const {
        return static_cast<const QuasiStaticGravity_CubicGalileon &>(this->qs());
    }
    double GeffOverG_highk(double a) const override { return this->GeffOverG(a, 0); }
    double short_range_comoving_range(double /*a*/) const override { return 0.; }
    void init() override {
        this->quasi_static = std::make_shared<const QuasiStaticGravity_CubicGalileon>(this->cosmo->background());
        GravityModelScreened<NDIM>::init();
    }

    /// The screening approximation: the density smoothed as for DGP (gravity_model_cubic_galileon_smoothing_*)
    double density_smoothing_scale_over_boxsize() const override { return smoothing_scale_over_boxsize; }
    std::string density_smoothing_filter() const override { return smoothing_filter; }
    void read_parameters(FML::UTILS::ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        if (this->use_screening_method) {
            smoothing_scale_over_boxsize = this->settings.cubic_galileon_smoothing_scale_over_boxsize;
            smoothing_filter = this->settings.cubic_galileon_smoothing_filter;
        }
    }

    /// The 2LPT source of the nonlinear term of the field equation (only with screening, as for DGP)
    double nonlinear_source_factor_2LPT(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        if (not this->get_use_screening())
            return 1.0;
        return this->qs().second_order_source(std::log(a), koverH0);
    }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        const double x = std::log(a);
        const double beta = galileon().beta(x), rcH0 = galileon().rcH0(x);
        DGPSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), rcH0, beta, H0Box, false);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.set_relative_criterion(this->multigrid_relative_criterion);
        if (this->multigrid_previous_a > 0.0)
            mgsolver.set_previous_solution(this->multigrid_previous_solution);
        // In deep voids the equation has no real solution (1 + e < 0 for delta < -(1 - 2 eps - Omega_phi)^2 /
        // (4 Omega_m(a)), -0.6 today): the square root of its discriminant is set to zero there
        mgsolver.set_allow_negative_discriminant(true);
        mgsolver.solve(a, density_real, potential_real);
        if (FML::ThisTask == 0) {
            std::cout << "Cubic Galileon multigrid solve at a = " << a << " (r_c H0/c = " << rcH0 << ", beta = " << beta
                      << "): " << mgsolver.get_number_of_vcycles() << " V-cycles";
            if (mgsolver.get_number_of_negative_discriminant_cells() > 0)
                std::cout << " (no real solution in " << mgsolver.get_number_of_negative_discriminant_cells()
                          << " cells: discriminant set to zero)";
            std::cout << "\n";
        }
    }

    void info_model() const override {
        if (FML::ThisTask == 0)
            std::cout << "# " << galileon().info() << "\n";
    }
};

#endif
