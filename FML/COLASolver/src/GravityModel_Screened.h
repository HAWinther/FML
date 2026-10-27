#ifndef GRAVITYMODEL_SCREENED_HEADER
#define GRAVITYMODEL_SCREENED_HEADER

#include "ParameterDefaults.h"

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/NBody/NBody.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

//==========================================================================================
/// Base class for gravity models with a fifth force: D^2 Phi = 1.5 OmegaM a (delta + delta_FF), where the
/// fifth-force "density" delta_FF is computed in one of three ways:
///  - exact: solving the (non-linear) field equation with the multigrid solver (if the model has one)
///  - screening: the screening approximation (with optionally the linear solution on large scales)
///  - linear: delta_FF(k) = (GeffOverG(a,k) - 1) delta(k)
/// A model provides GeffOverG, the screened fifth force and (optionally) the exact solver. The rest (the parameters
/// for screening and the multigrid solver, the large-scale blending and the force) is shared here.
//==========================================================================================
template <int NDIM>
class GravityModelScreened : public GravityModel<NDIM> {
  protected:
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
    using CouplingFunction = std::function<double(double)>;

    using GravityModel<NDIM>::GravityModel;

    bool get_use_screening() const { return use_screening_method or solve_exact_equation; }

    //========================================================================
    // Model specific parts
    //========================================================================

    /// Compute the screened fifth-force density from the density (Fourier space). coupling(kBox) = GeffOverG - 1
    virtual void compute_screened_fifth_force(double a,
                                              double H0Box,
                                              double norm_poisson_equation,
                                              const CouplingFunction & coupling,
                                              FFTWGrid<NDIM> & density_fourier,
                                              FFTWGrid<NDIM> & density_fifth_force) const = 0;

    /// Solve the exact field equation (density in real space) for the fifth-force potential (real space) with the
    /// multigrid solver, normalized such that -k^2 / (1.5 OmegaM a) times it is the fifth-force density
    virtual void solve_exact_fifth_force([[maybe_unused]] double a,
                                         [[maybe_unused]] double H0Box,
                                         [[maybe_unused]] FFTWGrid<NDIM> & density_real,
                                         [[maybe_unused]] FFTWGrid<NDIM> & potential_real) const {
        throw std::runtime_error("The gravity model " + this->get_name() + " has no exact (multigrid) solver");
    }

    /// Label used when printing which fifth force is added ("" for no printing)
    virtual std::string fifth_force_label() const { return ""; }

    //========================================================================
    // Shared parts
    //========================================================================

    /// Print the screening parameters (part of info())
    void info_screening() const {
        if (FML::ThisTask == 0) {
            std::cout << "# Screening method : " << use_screening_method << "\n";
            if (use_screening_method) {
                std::cout << "# Enforce correct linear evolution : " << screening_enforce_largescale_linear << "\n";
                std::cout << "# Scale for which we enforce this  : " << screening_linear_scale_hmpc << " h/Mpc\n";
                std::cout << "# Screening efficiency             : " << screening_efficiency << "\n";
            }
        }
    }

    /// Read the screening parameters (if gravity_model_screening is true)
    void read_screening_parameters(ParameterMap & param) {
        use_screening_method = param.get<bool>("gravity_model_screening");
        if (use_screening_method) {
            screening_enforce_largescale_linear = param.get<bool>("gravity_model_screening_enforce_largescale_linear");
            screening_linear_scale_hmpc = param.get<double>("gravity_model_screening_linear_scale_hmpc");
            screening_efficiency = COLASOLVER::get_parameter<double>(param, "gravity_model_screening_efficiency");
        }
    }

    /// Read the multigrid parameters if the exact equation is to be solved (exact_solution_parameter is the name of
    /// the model's switch, e.g. gravity_model_fofr_exact_solution)
    void read_multigrid_parameters(ParameterMap & param, const std::string & exact_solution_parameter) {
        solve_exact_equation = param.get<bool>(exact_solution_parameter);
        if (solve_exact_equation) {
            multigrid_nsweeps_first_step = param.get<int>("multigrid_nsweeps_first_step");
            multigrid_nsweeps = param.get<int>("multigrid_nsweeps");
            multigrid_solver_residual_convergence = param.get<double>("multigrid_solver_residual_convergence");
        }
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta and the fifth force delta_FF
    //========================================================================
    void compute_force(double a,
                       double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real) const override {

        // Compute fifth-force
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        auto coupling = [&](double kBox) { return this->GeffOverG(a, kBox / H0Box) - 1.0; };
        FFTWGrid<NDIM> density_fifth_force;

        if (solve_exact_equation) {

            // Solve the exact equation using the multigridsolver (this is slow)
            // Intended mainly to be used for testing things so not super optimized
            FFTWGrid<NDIM> density_real = density_fourier;
            // Get back the real-space density field
            density_real.fftw_c2r();
            solve_exact_fifth_force(a, H0Box, density_real, density_fifth_force);
            // It returns it in real-space so go back to fourier space
            density_fifth_force.fftw_r2c();

            // Multiply by -k^2 / (1.5 OmegaM a) to go from the potential to "force density"
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

            // Approximate screening method
            compute_screened_fifth_force(
                a, H0Box, norm_poisson_equation, coupling, density_fourier, density_fifth_force);

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
                        auto delta_fifth_force_linear =
                            delta * FML::GRID::FloatType(this->GeffOverG(a, kmag / H0Box) - 1.0);
                        FML::GRID::FloatType filter = f_spline(kmag);
                        auto value =
                            delta_fifth_force_linear * filter + delta_fifth_force * FML::GRID::FloatType(1.0 - filter);
                        density_fifth_force.set_fourier_from_index(fourier_index, value);
                    }
                }
            }

        } else {

            // No screening - linear evolution
            if (FML::ThisTask == 0 and not fifth_force_label().empty())
                std::cout << "Adding fifth-force " << fifth_force_label() << " (linear)\n";
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
        this->compute_force_from_total_density(
            density_fifth_force, norm_poisson_equation, density_assignment_method_used, force_real);
    }
};

#endif
