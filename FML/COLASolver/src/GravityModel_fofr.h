#ifndef GRAVITYMODEL_FOFR_HEADER
#define GRAVITYMODEL_FOFR_HEADER

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <FML/MultigridSolver/FofrSolver.h>

/// f(R) model
template <int NDIM>
class GravityModelFofR final : public GravityModel<NDIM> {
  protected:
    double fofr0{0.0};
    double nfofr{1.0};

    // For using the screening approximation
    bool use_screening_method{true};
    bool screening_enforce_largescale_linear{false};
    double screening_linear_scale_hmpc{0.0};
    double screening_efficiency{1.0};

    // For solving the exact equation
    bool solve_exact_equation{false};
    int multigrid_nsweeps_first_step{10};
    int multigrid_nsweeps{10};
    double multigrid_solver_residual_convergence{1e-7};

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;

    GravityModelFofR() : GravityModel<NDIM>("f(R)") {}
    GravityModelFofR(std::shared_ptr<Cosmology> cosmo)
        : GravityModel<NDIM>(cosmo, "f(R)")

    {
        assert(this->get_name() == "f(R)");
    }

    //========================================================================
    // Print some info
    //========================================================================
    void info() const override {
        GravityModel<NDIM>::info();
        if (FML::ThisTask == 0) {
            std::cout << "# fofr0            : " << fofr0 << "\n";
            std::cout << "# nfofr            : " << nfofr << "\n";
            std::cout << "# Screening method : " << use_screening_method << "\n";
            if (use_screening_method) {
                std::cout << "# Enforce correct linear evolution : " << screening_enforce_largescale_linear << "\n";
                std::cout << "# Scale for which we enforce this  : " << screening_linear_scale_hmpc << " h/Mpc\n";
                std::cout << "# Screening efficiency             : " << screening_efficiency << "\n";
            }
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
    void compute_force(double a,
                       double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Compute fifth-force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        auto coupling = [&](double kBox) { return GeffOverG(a, kBox / H0Box) - 1.0; };
        FFTWGrid<NDIM> density_fifth_force;

        if (solve_exact_equation) {

            // Solve the exactl f(R) equation using the multigridsolver (this is slow)
            // Intenden mainly to be used for testing things so not super optimized
            FFTWGrid<NDIM> density_real = density_fourier;
            // Get back the real-space density field
            density_real.fftw_c2r();
            // Set up the solver, set the settings, and solve the solution
            const bool verbose = true;
            FofrSolverCosmology<NDIM, double> mgsolver(this->cosmo->get_OmegaM(), nfofr, fofr0, H0Box, verbose);
            mgsolver.set_ngs_steps(multigrid_nsweeps, multigrid_nsweeps, multigrid_nsweeps_first_step);
            mgsolver.set_epsilon(multigrid_solver_residual_convergence);
            mgsolver.solve(a, density_real, density_fifth_force);
            // It returns it in real-space so go back to fourier space
            density_fifth_force.fftw_r2c();

            // Multiply by -k^2 / (1.5 OmegaM a) to go from potential a^2f_R/(2 H0Box^2) to "force density"
            // (defined such that the total force is 1.5 OmegaM a * D( 1/D^2 delta_FF + 1/D^2 delta) )
            const auto Local_nx = density_fifth_force.get_local_nx();
            const auto Local_x_start = density_fifth_force.get_local_x_start();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] std::array<double, NDIM> kvec;
                double kmag2;
                for (auto && fourier_index : density_fifth_force.get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode (k=0)
                    density_fifth_force.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                    auto value = density_fifth_force.get_fourier_from_index(fourier_index);
                    value *= -kmag2 / norm_poisson_equation;
                    density_fifth_force.set_fourier_from_index(fourier_index, value);
                }
            }
            // DC mode
            if (Local_x_start == 0)
                density_fifth_force.set_fourier_from_index(0, 0.0);

        } else if (use_screening_method) {
               
            // Critial treshold for screening in Hu-Sawicky (basically 3/2 f_R(a))
            const double OmegaM = this->cosmo->get_OmegaM();
            double PhiCrit = 1.5 * fofr0 *
              std::pow((OmegaM + 4.0 * (1.0 - OmegaM)) / (1.0 / (a * a * a) * OmegaM + 4.0 * (1.0 - OmegaM)), nfofr + 1.0);

            // Approximate screening method
            auto screening_function_fofr = [=](double PhiNewton) {
                double screenfac = std::abs(PhiCrit / PhiNewton) * screening_efficiency;
                return screenfac > 1.0 ? 1.0 : screenfac;
            };

            if (FML::ThisTask == 0)
                std::cout << "Adding fifth-force f(R) (screening)\n";
            FML::NBODY::compute_delta_fifth_force_potential_screening(density_fourier,
                                                                      density_fifth_force,
                                                                      coupling,
                                                                      screening_function_fofr,
                                                                      norm_poisson_equation * std::pow(H0Box / a, 2));

            // Ensure that the large scales are behaving correctly
            // We set delta_fifth_force => A * (1-f) + B * f
            // i.e. we use the linear prediction on large sales and the
            // screened prediction on small scales
            if (screening_enforce_largescale_linear and screening_linear_scale_hmpc > 0.0) {

                if (FML::ThisTask == 0) {
                    std::cout << "Combining screened solution with linear solution\n";
                    std::cout << "We use linear solution for k < " << screening_linear_scale_hmpc << " h/Mpc\n";
                }

                // Make a spline of the low-pass filter we use
                const int npts = 1000;
                const double kBoxmin = M_PI;
                const double kBoxmax = M_PI * std::sqrt(NDIM) * density_fourier.get_nmesh();
                const double kcutBox = screening_linear_scale_hmpc / this->H0_hmpc * H0Box;
                std::vector<double> kBox(npts);
                std::vector<double> f(npts);
                for (int i = 0; i < npts; i++) {
                    kBox[i] = kBoxmin + (kBoxmax - kBoxmin) * i / double(npts - 1);
                    f[i] = std::exp(-0.5 * kBox[i] * kBox[i] / (kcutBox * kcutBox));
                }
                Spline f_spline(kBox, f, "Low-pass filter");

                // Combine the screened and the linear solution together
                auto Local_nx = density_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    [[maybe_unused]] std::array<double, NDIM> kvec;
                    double kmag;
                    for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
                        density_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                        auto delta = density_fourier.get_fourier_from_index(fourier_index);
                        auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                        auto delta_fifth_force_linear = delta * FML::GRID::FloatType(GeffOverG(a, kmag / H0Box) - 1.0);
                        FML::GRID::FloatType filter = f_spline(kmag);
                        auto value = delta_fifth_force_linear * filter + delta_fifth_force * FML::GRID::FloatType(1.0 - filter);
                        density_fifth_force.set_fourier_from_index(fourier_index, value);
                    }
                }
            }

        } else {

            // No screening - linear evolution
            if (FML::ThisTask == 0)
                std::cout << "Adding fifth-force f(R) (linear)\n";
            FML::NBODY::compute_delta_fifth_force<NDIM>(density_fourier, density_fifth_force, coupling);
        }

        // Add on density corresponding to the gravitational force to get total force
        auto Local_nx = density_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && fourier_index : density_fourier.get_fourier_range(islice, islice + 1)) {
                auto delta = density_fourier.get_fourier_from_index(fourier_index);
                auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                density_fifth_force.set_fourier_from_index(fourier_index, delta + delta_fifth_force);
            }
        }

        // Compute total force
        if (this->force_use_finite_difference_force) {
          // Use a by default a 4 point formula (using phi(i+/-2), phi(i+/-1) to compute DPhi)
          // This requires 2 boundary cells (stencil_order=2,4,6 implemented so far)
          const int stencil_order = this->force_finite_difference_stencil_order;
          const int nboundary_cells = stencil_order/2;

          FFTWGrid<NDIM> potential_real;
          FML::NBODY::compute_potential_real_from_density_fourier<NDIM>(density_fifth_force,
              potential_real,
              norm_poisson_equation,
              nboundary_cells);

          FML::NBODY::compute_force_from_potential_real<NDIM>(potential_real,
              force_real,
              density_assignment_method_used,
              stencil_order);

        } else {
          // Computes gravitational force using fourier-methods
          FML::NBODY::compute_force_from_density_fourier<NDIM>(
              density_fifth_force, force_real, density_assignment_method_used, norm_poisson_equation);
        }
    }

    //========================================================================
    // Read the parameters we need
    //========================================================================
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        fofr0 = param.get<double>("gravity_model_fofr_fofr0");
        nfofr = param.get<double>("gravity_model_fofr_nfofr");
        use_screening_method = param.get<bool>("gravity_model_screening");
        if (use_screening_method) {
            screening_enforce_largescale_linear = param.get<bool>("gravity_model_screening_enforce_largescale_linear");
            screening_linear_scale_hmpc = param.get<double>("gravity_model_screening_linear_scale_hmpc");
            screening_efficiency = param.get<double>("gravity_model_screening_efficiency", 1.0);
        }
        solve_exact_equation = param.get<bool>("gravity_model_fofr_exact_solution");
        if (solve_exact_equation) {
            multigrid_nsweeps_first_step = param.get<int>("multigrid_nsweeps_first_step");
            multigrid_nsweeps = param.get<int>("multigrid_nsweeps");
            multigrid_solver_residual_convergence = param.get<double>("multigrid_solver_residual_convergence");
        }
        this->scaledependent_growth = true;
    }
};

#endif
