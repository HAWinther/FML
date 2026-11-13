#ifndef GRAVITYMODEL_HORNDESKI_HEADER
#define GRAVITYMODEL_HORNDESKI_HEADER

#include "GravityModel_Screened.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/ParameterMap/ParameterMap.h>

#include <FML/Cosmology/Perturbations/QuasiStaticGravity_Horndeski.h>
#include <FML/MultigridSolver/DGPSolver.h>

/// A generic reduced Horndeski theory (K(phi, X), G3(phi, X), G4(phi)) from the table of its quasi-static force made by
/// scripts/horndeski_frontend.py (that of cosmology_model = Horndeski, which also gives the background and the IC from
/// the internal Einstein-Boltzmann solver; or gravity_model_horndeski_filename with cosmology_model = Tabulated and the
/// H table the frontend also writes): the Newtonian force rescaled by G_G4/G = 1 / (2 G4) (newtonian_factor)
/// plus the fifth force of the DGP form with the table's beta(a) and r_c(a)
/// (FML/Cosmology/Perturbations/QuasiStaticGravity_Horndeski.h): linear, the screening approximation (the density
/// smoothed with gravity_model_horndeski_smoothing_*) or the exact equation with DGP's multigrid solver
/// (gravity_model_horndeski_exact_solution; needs r_c^2 > 0). This follows Hi-COLA (Wright et al. 2023, JCAP 03, 040,
/// arXiv:2209.01666; Sen Gupta, Fiorini & Baker 2024, arXiv:2407.00855 for the rescaled Newtonian force) with the
/// quasi-static equations of Kimura, Kobayashi & Yamamoto 2012 (arXiv:1111.6749); references in
/// QuasiStaticGravity_Horndeski.h
template <int NDIM>
class GravityModelHorndeski final : public GravityModelScreened<NDIM> {
    std::string filename;
    std::string smoothing_filter{"gaussian"};
    double smoothing_scale_over_boxsize{0.0};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using QuasiStaticGravity_Horndeski = FML::COSMOLOGY::PERTURBATIONS::QuasiStaticGravity_Horndeski;

    explicit GravityModelHorndeski(std::shared_ptr<Cosmology> cosmo)
        : GravityModelScreened<NDIM>(cosmo, "Horndeski", &SIMCODE::GravityModelSettings::horndeski_exact_solution) {}

    const QuasiStaticGravity_Horndeski & horndeski() const {
        return static_cast<const QuasiStaticGravity_Horndeski &>(this->qs());
    }
    double GeffOverG_highk(double a) const override { return this->GeffOverG(a, 0); }
    double short_range_comoving_range(double /*a*/) const override { return 0.; }
    double newtonian_factor(double a) const override { return horndeski().GG4(std::log(a)); }

    void init() override {
        if (filename.empty())
            filename = this->cosmo->background().parameters().model_file;
        if (filename.empty())
            throw std::runtime_error("Horndeski: gravity_model_horndeski_filename is empty and the cosmology has no "
                                     "table (use cosmology_model = Horndeski, or give the frontend's _gravity.txt)");
        const auto table = QuasiStaticGravity_Horndeski::read_table(filename);
        auto qs = std::make_shared<const QuasiStaticGravity_Horndeski>(this->cosmo->background(), table, filename);
        if (qs->max_relative_E_difference() > 1e-3)
            throw std::runtime_error("Horndeski: H(a) of the table [" + filename + "] differs from the background by " +
                                     std::to_string(qs->max_relative_E_difference()) +
                                     " (use cosmology_model = Horndeski with the same table, or Tabulated with the "
                                     "frontend's _background.txt, and the same OmegaM)");
        this->quasi_static = qs;
        GravityModelScreened<NDIM>::init();
    }

    /// The 2LPT source of the nonlinear term of the field equation (only with screening, as for DGP)
    double nonlinear_source_factor_2LPT(double a, [[maybe_unused]] double koverH0 = 0.0) const override {
        if (not this->get_use_screening())
            return 1.0;
        return this->qs().second_order_source(std::log(a), koverH0);
    }

    double density_smoothing_scale_over_boxsize() const override { return smoothing_scale_over_boxsize; }
    std::string density_smoothing_filter() const override { return smoothing_filter; }

    void solve_exact_fifth_force(double a,
                                 double H0Box,
                                 FFTWGrid<NDIM> & density_real,
                                 FFTWGrid<NDIM> & potential_real) const override {
        const double x = std::log(a);
        if (not(horndeski().c2(x) > 0.0 and horndeski().b5(x) > 0.0))
            throw std::runtime_error("Horndeski: the exact solution needs r_c^2 > 0 and a fifth force 1/(3 beta) > 0 "
                                     "(the DGP solver) at a = " +
                                     std::to_string(a));
        const double beta = horndeski().beta(x), rcH0 = horndeski().rcH0(x);
        DGPSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), rcH0, beta, H0Box, false);
        mgsolver.set_ngs_steps(this->multigrid_nsweeps, this->multigrid_nsweeps, this->multigrid_nsweeps_first_step);
        mgsolver.set_epsilon(this->multigrid_solver_residual_convergence);
        mgsolver.set_relative_criterion(this->multigrid_relative_criterion);
        if (this->multigrid_previous_a > 0.0)
            mgsolver.set_previous_solution(this->multigrid_previous_solution);
        mgsolver.set_allow_negative_discriminant(true);
        mgsolver.solve(a, density_real, potential_real);
        if (FML::ThisTask == 0) {
            std::cout << "Horndeski multigrid solve at a = " << a << " (r_c H0/c = " << rcH0 << ", beta = " << beta
                      << "): " << mgsolver.get_number_of_vcycles() << " V-cycles";
            if (mgsolver.get_number_of_negative_discriminant_cells() > 0)
                std::cout << " (no real solution in " << mgsolver.get_number_of_negative_discriminant_cells()
                          << " cells: discriminant set to zero)";
            std::cout << "\n";
        }
    }

    void read_parameters(FML::UTILS::ParameterMap & param) override {
        GravityModelScreened<NDIM>::read_parameters(param);
        filename = this->settings.horndeski_filename;
        if (this->use_screening_method) {
            smoothing_scale_over_boxsize = this->settings.horndeski_smoothing_scale_over_boxsize;
            smoothing_filter = this->settings.horndeski_smoothing_filter;
        }
    }

    void info_model() const override {
        if (FML::ThisTask == 0) {
            std::cout << "# " << horndeski().info() << "\n";
            std::cout << "# Table [" << filename << "] for a = " << horndeski().a_min() << " - " << horndeski().a_max()
                      << " (the end values outside), H(a) as the background's to "
                      << horndeski().max_relative_E_difference() << "\n";
        }
    }
};

#endif
