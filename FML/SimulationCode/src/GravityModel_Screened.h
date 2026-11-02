#ifndef GRAVITYMODEL_SCREENED_HEADER
#define GRAVITYMODEL_SCREENED_HEADER

#include "ParameterDefaults.h"

#include "GravityModel.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/NBody/NBody.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

//==========================================================================================
/// Base class for gravity models with a fifth force (and a shared quasi-static theory with its screening,
/// FML/Cosmology/Perturbations/QuasiStaticGravity.h): D^2 Phi = 1.5 OmegaM a (delta + delta_FF), where the
/// fifth-force "density" delta_FF is computed in one of three ways:
///  - exact: solving the (non-linear) field equation with the multigrid solver (if the model has one)
///  - screening: the screening approximation (with optionally the linear solution on large scales)
///  - linear: delta_FF(k) = (GeffOverG(a,k) - 1) delta(k)
/// A model provides GeffOverG and its theory (whose screening() is applied here) and (optionally) the exact solver.
/// The rest (the parameters for screening and the multigrid solver, the large-scale blending and the force) is shared
/// here.
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
    int multigrid_nsweeps{3};
    double multigrid_solver_residual_convergence{1e-7};
    // Solve the field equation on a grid coarser by this factor (1: the force grid): the density and the solution go
    // between the grids in Fourier space (the modes below the coarse Nyquist frequency; the solution is zero above
    // it on the force grid). Cf. Barreira, Bose & Li 2015 (arXiv:1511.08200) for nDGP
    int multigrid_coarse_factor{1};
    // The last exact (multigrid) solution in real space and its scale factor: the starting point of the next solve
    mutable FML::GRID::FFTWGrid<NDIM> multigrid_previous_solution;
    mutable double multigrid_previous_a{-1.0};
    std::string exact_solution_parameter;

  public:
    template <int N>
    using FFTWGrid = FML::GRID::FFTWGrid<N>;
    using ParameterMap = FML::UTILS::ParameterMap;
    using Spline = FML::INTERPOLATION::SPLINE::Spline;
    using CouplingFunction = std::function<double(double)>;

    /// exact_solution_parameter: the name of the model's switch for the exact (multigrid) solution, e.g.
    /// gravity_model_fofr_exact_solution ("" if the model has no multigrid solver)
    GravityModelScreened(std::shared_ptr<Cosmology> cosmo, std::string name, std::string exact_solution_parameter = "")
        : GravityModel<NDIM>(cosmo, name), exact_solution_parameter(exact_solution_parameter) {}

    /// The base parameters, the screening parameters and (if the model has a multigrid solver) the multigrid ones
    void read_parameters(ParameterMap & param) override {
        GravityModel<NDIM>::read_parameters(param);
        read_screening_parameters(param);
        if (not exact_solution_parameter.empty())
            read_multigrid_parameters(param, exact_solution_parameter);
    }

    bool get_use_screening() const { return use_screening_method or solve_exact_equation; }

    //========================================================================
    // Model specific parts
    //========================================================================

    /// Compute the screened fifth-force density from the density (Fourier space). coupling(kBox) = GeffOverG - 1.
    /// The default applies the screening of the model's theory (QuasiStaticGravity::screening(): the type decides
    /// whether the screening factor is a function of the Newtonian potential or of the density) times
    /// gravity_model_screening_efficiency
    virtual void compute_screened_fifth_force(double a,
                                              double H0Box,
                                              double norm_poisson_equation,
                                              const CouplingFunction & coupling,
                                              FFTWGrid<NDIM> & density_fourier,
                                              FFTWGrid<NDIM> & density_fifth_force) const {
        using FML::COSMOLOGY::PERTURBATIONS::ScreeningType;
        const auto screening = this->qs().screening();
        const double x = std::log(a);
        const double eff = screening_efficiency;
        auto factor = [&screening, x, eff](double value) { return screening.factor(x, value, eff); };
        if (FML::ThisTask == 0 and not fifth_force_label().empty())
            std::cout << "Adding fifth-force " << fifth_force_label() << " (screening)\n";
        switch (screening.type) {
            case ScreeningType::Potential:
                FML::NBODY::compute_delta_fifth_force_potential_screening(
                    density_fourier, density_fifth_force, coupling, factor, norm_poisson_equation * std::pow(H0Box / a, 2));
                return;
            case ScreeningType::Density:
                FML::NBODY::compute_delta_fifth_force_density_screening(density_fourier,
                                                                        density_fifth_force,
                                                                        coupling,
                                                                        factor,
                                                                        density_smoothing_scale_over_boxsize(),
                                                                        density_smoothing_filter());
                return;
            default:
                throw std::runtime_error("The gravity model " + this->get_name() + " has no screening of type " +
                                         FML::COSMOLOGY::PERTURBATIONS::screening_type_name(screening.type) +
                                         " implemented here");
        }
    }
    /// Density screening: the smoothing of the density field before the screening factor is computed
    virtual double density_smoothing_scale_over_boxsize() const { return 0.0; }
    virtual std::string density_smoothing_filter() const { return "tophat"; }

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
            screening_efficiency = SIMCODE::get_parameter<double>(param, "gravity_model_screening_efficiency");
        }
    }

    /// The density on the grid coarser by factor (Nmesh / factor) in real space: the Fourier modes below its Nyquist
    /// frequency (a sharp-k filter). Not the mean of the factor^NDIM cells: with the particles on a lattice aligned
    /// with the coarse cells (half a force cell offset, particle_Npart_1D = Nmesh / 2) the mean of a block is the
    /// mass of the one particle in it until it leaves the block, so the linear density is lost. Cells with delta < -1
    /// (from the truncation) are set to -1
    static FFTWGrid<NDIM> restrict_to_coarse_grid(const FFTWGrid<NDIM> & density_fourier, int factor) {
        const int nfine = density_fourier.get_nmesh();
        const int ncoarse = nfine / factor;
        FML::assert_mpi(ncoarse * factor == nfine and ncoarse % FML::NTasks == 0,
                        "multigrid_coarse_factor: the force grid / factor must be an integer divisible by the number "
                        "of MPI tasks");
        FFTWGrid<NDIM> coarse(
            ncoarse, density_fourier.get_n_extra_slices_left(), density_fourier.get_n_extra_slices_right());
        // (A Gaussian filter with R = 0.5 or 1 coarse cells instead of the sharp cut avoids the delta < -1 but was
        // less accurate in a test with nDGP and f(R))
        FML::NUFFT::copy_fourier_modes<NDIM>(
            density_fourier, coarse, [](const std::array<int, NDIM> &) { return 1.0; });
        coarse.fftw_c2r();
        // The truncation rings: delta < -1 in a few cells (no solution of the DGP equation there). Set to -1
        long long nclamped = 0;
        for (auto && index : coarse.get_real_range()) {
            if (coarse.get_real_from_index(index) < -1.0) {
                coarse.set_real_from_index(index, -1.0);
                nclamped++;
            }
        }
        FML::SumOverTasks(&nclamped);
        if (nclamped > 0 and FML::ThisTask == 0)
            std::cout << "multigrid_coarse_factor: delta < -1 set to -1 in " << nclamped << " coarse cells\n";
        return coarse;
    }

    /// Read the multigrid parameters if the exact equation is to be solved (exact_solution_parameter is the name of
    /// the model's switch, e.g. gravity_model_fofr_exact_solution)
    void read_multigrid_parameters(ParameterMap & param, const std::string & exact_solution_parameter) {
        solve_exact_equation = param.get<bool>(exact_solution_parameter);
        if (solve_exact_equation) {
            multigrid_nsweeps_first_step = param.get<int>("multigrid_nsweeps_first_step");
            multigrid_nsweeps = param.get<int>("multigrid_nsweeps");
            multigrid_solver_residual_convergence = param.get<double>("multigrid_solver_residual_convergence");
            multigrid_coarse_factor = SIMCODE::get_parameter<int>(param, "multigrid_coarse_factor");
            FML::assert_mpi(multigrid_coarse_factor >= 1, "multigrid_coarse_factor must be >= 1");
        }
    }

    //========================================================================
    // The three ways of computing the fifth-force density delta_FF (Fourier space)
    //========================================================================

    /// Exact: solve the field equation with the multigrid solver (slow; intended mainly for testing) on the force grid
    /// or a grid coarser by multigrid_coarse_factor (its modes below the coarse Nyquist frequency; the others zero)
    void compute_exact_fifth_force(double a,
                                   double H0Box,
                                   double norm_poisson_equation,
                                   const FFTWGrid<NDIM> & density_fourier,
                                   FFTWGrid<NDIM> & density_fifth_force) const {
        // The real-space density field
        FFTWGrid<NDIM> density_real;
        if (multigrid_coarse_factor > 1) {
            density_real = restrict_to_coarse_grid(density_fourier, multigrid_coarse_factor);
        } else {
            density_real = density_fourier;
            density_real.fftw_c2r();
        }
        const int nmesh_solver = density_real.get_nmesh();
        solve_exact_fifth_force(a, H0Box, density_real, density_fifth_force);
        // The starting point of the next solve
        multigrid_previous_solution = density_fifth_force;
        multigrid_previous_a = a;
        // It returns it in real-space so go back to fourier space
        density_fifth_force.fftw_r2c();
        // From the coarse grid: the modes below its Nyquist frequency on the force grid (the others zero)
        if (multigrid_coarse_factor > 1) {
            FFTWGrid<NDIM> fine(density_fourier.get_nmesh(),
                                density_fourier.get_n_extra_slices_left(),
                                density_fourier.get_n_extra_slices_right());
            FML::NUFFT::copy_fourier_modes<NDIM>(
                density_fifth_force, fine, [](const std::array<int, NDIM> &) { return 1.0; });
            density_fifth_force = std::move(fine);
        }

        // Multiply by -lambda(k) / (1.5 OmegaM a) to go from the potential to "force density" (defined such that the
        // total force is 1.5 OmegaM a * D( 1/D^2 delta_FF + 1/D^2 delta) ). lambda(k) = sum_i 4 N^2 sin^2(k_i / 2N) is
        // the eigenvalue of the (2 NDIM + 1)-point Laplacian the solver inverted on its N^NDIM grid, so in the linear
        // regime delta_FF is exactly (GeffOverG - 1) delta for every mode. The continuum k^2 over-estimated it by
        // k^2 / lambda (+3% at a quarter and +13% at half the Nyquist frequency of the solver grid)
        const double nmesh2x4 = 4.0 * double(nmesh_solver) * double(nmesh_solver);
        const double half_h = 0.5 / double(nmesh_solver);
        const auto Local_x_start = density_fifth_force.get_local_x_start();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (FML::IndexIntType irow = 0; irow < density_fifth_force.get_local_nrows(); irow++) {
            std::array<double, NDIM> kvec;
            [[maybe_unused]] double kmag2;
            for (auto && fourier_index : density_fifth_force.get_fourier_range_of_rows(irow, irow + 1)) {
                if (Local_x_start == 0 and fourier_index == 0)
                    continue; // DC mode (k=0)
                density_fifth_force.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
                double lambda = 0.0;
                for (int idim = 0; idim < NDIM; idim++) {
                    const double sn = std::sin(kvec[idim] * half_h);
                    lambda += nmesh2x4 * sn * sn;
                }
                auto value = density_fifth_force.get_fourier_from_index(fourier_index);
                value *= -lambda / norm_poisson_equation;
                density_fifth_force.set_fourier_from_index(fourier_index, value);
            }
        }
        // DC mode
        if (Local_x_start == 0)
            density_fifth_force.set_fourier_from_index(0, 0.0);
    }

    /// The screening approximation, with (gravity_model_screening_enforce_largescale_linear) the linear solution on
    /// large scales: delta_FF -> linear * f + screened * (1 - f) with the low-pass filter f = exp(-k^2 / 2 k_cut^2)
    void compute_screened_fifth_force_blended(double a,
                                              double H0Box,
                                              double norm_poisson_equation,
                                              const CouplingFunction & coupling,
                                              FFTWGrid<NDIM> & density_fourier,
                                              FFTWGrid<NDIM> & density_fifth_force) const {
        compute_screened_fifth_force(a, H0Box, norm_poisson_equation, coupling, density_fourier, density_fifth_force);
        if (not(screening_enforce_largescale_linear and screening_linear_scale_hmpc > 0.0))
            return;

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
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (FML::IndexIntType irow = 0; irow < density_fourier.get_local_nrows(); irow++) {
            [[maybe_unused]] std::array<double, NDIM> kvec;
            double kmag;
            for (auto && fourier_index : density_fourier.get_fourier_range_of_rows(irow, irow + 1)) {
                density_fourier.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
                auto delta = density_fourier.get_fourier_from_index(fourier_index);
                auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                auto delta_fifth_force_linear = delta * FML::GRID::FloatType(this->GeffOverG(a, kmag / H0Box) - 1.0);
                FML::GRID::FloatType filter = f_spline(kmag);
                auto value = delta_fifth_force_linear * filter + delta_fifth_force * FML::GRID::FloatType(1.0 - filter);
                density_fifth_force.set_fourier_from_index(fourier_index, value);
            }
        }
    }

    /// No screening (linear evolution): delta_FF(k) = (GeffOverG(a, k) - 1) delta(k)
    void compute_linear_fifth_force(const CouplingFunction & coupling,
                                    const FFTWGrid<NDIM> & density_fourier,
                                    FFTWGrid<NDIM> & density_fifth_force) const {
        if (FML::ThisTask == 0 and not fifth_force_label().empty())
            std::cout << "Adding fifth-force " << fifth_force_label() << " (linear)\n";
        FML::NBODY::compute_delta_fifth_force<NDIM>(density_fourier, density_fifth_force, coupling);
    }

    //========================================================================
    // Compute the force DPhi from the density field delta in fourier space
    // We compute this from D^2 Phi_GR = norm_poisson_equation * delta and the fifth force delta_FF
    //========================================================================
    void compute_force(double a,
                       double H0Box,
                       FFTWGrid<NDIM> & density_fourier,
                       std::string density_assignment_method_used,
                       std::array<FFTWGrid<NDIM>, NDIM> & force_real,
                       const FML::NBODY::ForceOptions & options = {}) const override {
        const double norm_poisson_equation = 1.5 * this->cosmo->get_OmegaM() * a;
        auto coupling = [&](double kBox) { return this->GeffOverG(a, kBox / H0Box) - 1.0; };

        FFTWGrid<NDIM> density_fifth_force;
        if (solve_exact_equation)
            compute_exact_fifth_force(a, H0Box, norm_poisson_equation, density_fourier, density_fifth_force);
        else if (use_screening_method)
            compute_screened_fifth_force_blended(
                a, H0Box, norm_poisson_equation, coupling, density_fourier, density_fifth_force);
        else
            compute_linear_fifth_force(coupling, density_fourier, density_fifth_force);

        // Add on density corresponding to the gravitational force to get total force
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (FML::IndexIntType irow = 0; irow < density_fourier.get_local_nrows(); irow++) {
            for (auto && fourier_index : density_fourier.get_fourier_range_of_rows(irow, irow + 1)) {
                auto delta = density_fourier.get_fourier_from_index(fourier_index);
                auto delta_fifth_force = density_fifth_force.get_fourier_from_index(fourier_index);
                density_fifth_force.set_fourier_from_index(fourier_index, delta + delta_fifth_force);
            }
        }

        // Compute total force
        this->compute_force_from_total_density(
            density_fifth_force, norm_poisson_equation, density_assignment_method_used, force_real, options);
    }
};

#endif
