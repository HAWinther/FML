#ifndef DGPSOLVER_HEADER
#define DGPSOLVER_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIGrid/ConvertMPIGridFFTWGrid.h>
#include <FML/MultigridSolver/MultiGridSolver.h>

#include <cmath>
#include <iostream>
#include <limits>

// Type aliases used below
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using namespace FML::SOLVERS::MULTIGRIDSOLVER;

//======================================================================
/// Solver the DGP gravity equation in a cosmological setting
/// The solver returns PhiF so that in a
/// cosmological setting where PhiGR = a^2 Phi / (H0 Box)^2 we simply
/// have the total force = D(PhiGR + PhiF)
/// This class (kind of) assumes a LCDM cosmology
//======================================================================
template <int NDIM, typename SolverType = double>
class DGPSolverCosmology {
  private:
    int Nmesh{};
    int Nlevels{-1};
    int nleft{1};
    int nright{1};

    int ngs_fine{10};
    int ngs_coarse{10};
    int ngs_first{40};

    int maxsteps{1000};

    /// Convergence criterion: residual < epsilon
    double epsilon{1e-6};

    /// Matter density parameter today
    double OmegaM{0.3};
    /// The cross-over scale rcH0/c
    double rcH0_DGP{1.0};
    /// The beta-function (coupling strength = 1/3beta)
    double beta_DGP{1e-6};
    /// The dimensionless number H0 * Box / c
    double H0Box{};

    /// Print info as we go along
    bool verbose{true};

    /// The box is periodic?
    bool periodic{true};

  public:
    DGPSolverCosmology(double OmegaM, double rcH0_DGP, double beta_DGP, double H0Box, bool verbose)
        : OmegaM(OmegaM), rcH0_DGP(rcH0_DGP), beta_DGP(beta_DGP), H0Box(H0Box), verbose(verbose) {}

    /// Set how many sweeps over the grid to use when solving a given level
    void set_ngs_steps(double _ngs_fine, double _ngs_coarse, double _ngs_first) {
        ngs_fine = _ngs_fine;
        ngs_coarse = _ngs_coarse;
        ngs_first = _ngs_first;
    }

    /// Set convergenc criterion
    void set_epsilon(double _epsilon) { epsilon = _epsilon; }

    /// Set maximum number of steps to take (in case residual saturates)
    void set_maxsteps(int _maxsteps) { maxsteps = _maxsteps; }

    /// Do the solving and return the fifth-force potential.
    /// We solve for f where a phi / phi0 f_R / (2 (H0Box)^2) = e^f and return e^f
    void solve(double a, FFTWGrid<NDIM> & overdensity_real, FFTWGrid<NDIM> & fifth_force_potential_real) {

        Nmesh = overdensity_real.get_nmesh();
        auto Local_nx = overdensity_real.get_local_nx();

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Running multigridsolver for DGP\n";
            std::cout << "# Nmesh       : " << Nmesh << "\n";
            std::cout << "# a           : " << a << "\n";
            std::cout << "# OmegaM      : " << OmegaM << "\n";
            std::cout << "# rcH0/c      : " << rcH0_DGP << "\n";
            std::cout << "# beta_DGP    : " << beta_DGP << "\n";
            std::cout << "# H0Box       : " << H0Box << "\n";
            std::cout << "# Convergence : residual < " << epsilon << "\n";
            std::cout << "# Ngs_sweeps  : " << ngs_fine << " (fine) , " << ngs_coarse << " (coarse)\n";
            std::cout << "# Ngs_sweeps  : " << ngs_first << " (first step)\n";
            std::cout << "#=====================================================\n";
        }

        // Set up multigrid for density
        MPIMultiGrid<NDIM, SolverType> density_multigrid(Nmesh, Nlevels, nleft, nright);
        auto & grid = density_multigrid.get_grid();

#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && real_index : overdensity_real.get_real_range(islice, islice + 1)) {
                // Fetch the value in the cell
                auto density_in_cell = overdensity_real.get_real_from_index(real_index);
                // Fetch the coordinates of the cell (ix,iy,iz) and index
                auto coord = overdensity_real.get_coord_from_index(real_index);
                auto density_grid_index = grid.index_from_coord(coord);
                // Set the value
                grid.set_y(density_grid_index, density_in_cell);
            }
        }
        density_multigrid.restrict_down_all();

        // Factors to define the equation
        const double alpha_DGP = 3.0 * beta_DGP * (a * a * a * a) / (rcH0_DGP * rcH0_DGP);

        // The background value of the field we solve for (we solve for perturbations about the background)
        const double f0 = 0.0;

        // Set up solver
        MultiGridSolver<NDIM, SolverType> g(Nmesh, Nlevels, verbose, periodic, nleft, nright);

        // Set some options
        g.set_epsilon(epsilon);
        g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
        g.set_epsilon(epsilon);
        g.set_maxsteps(maxsteps);

        // Set the initial guess
        g.set_initial_guess(f0);

        //======================================================================
        // Set the convergence criterion
        // (ConvergenceCriterionResidual is rms-residual < epsilon)
        //======================================================================
        MultiGridConvCrit ConvergenceCriterion = [&](double rms_residual, double rms_residual_ini, int step_number) {
            return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
        };

        //======================================================================
        // Implement the equation to be solved
        //======================================================================
        MultiGridFunction<NDIM, double> Equation =
            [&](MultiGridSolver<NDIM, SolverType> * sol, int level, IndexInt index) {
                const auto h = sol->get_Gridspacing(level);

                //======================================================================
                // Set the (over) density field from formula
                //======================================================================
                auto delta = density_multigrid.get_y(level, index);

                //======================================================================
                // Get a list of the index of the closest 27 cells
                //======================================================================
                auto cube_index_list = sol->get_cube_gridindex(level, index);
                double phi = sol->get_Field(level, index);

                //======================================================================
                // Second derivatives (as written this only works for NDIM=3)
                //======================================================================
                const int MID = FML::power(3, NDIM) / 2;
                const int NDIM2 = NDIM * NDIM;
                
                double phi_xy = (sol->get_Field(level, cube_index_list[MID + NDIM + 1]) +
                                 sol->get_Field(level, cube_index_list[MID - NDIM - 1]) -
                                 sol->get_Field(level, cube_index_list[MID + NDIM - 1]) -
                                 sol->get_Field(level, cube_index_list[MID - NDIM + 1])) *
                                0.25 / (h * h);

                double phi_yz = 0.0, phi_zx = 0.0, phi_zz = 0.0;
                if constexpr (NDIM == 3) {
                    phi_yz = (sol->get_Field(level, cube_index_list[MID + NDIM2 + NDIM]) +
                              sol->get_Field(level, cube_index_list[MID - NDIM2 - NDIM]) -
                              sol->get_Field(level, cube_index_list[MID + NDIM2 - NDIM]) -
                              sol->get_Field(level, cube_index_list[MID - NDIM2 + NDIM])) *
                             0.25 / (h * h);

                    phi_zx = (sol->get_Field(level, cube_index_list[MID + NDIM2 + 1]) +
                              sol->get_Field(level, cube_index_list[MID - NDIM2 - 1]) -
                              sol->get_Field(level, cube_index_list[MID + NDIM2 - 1]) -
                              sol->get_Field(level, cube_index_list[MID - NDIM2 + 1])) *
                             0.25 / (h * h);
                    phi_zz = (sol->get_Field(level, cube_index_list[MID + NDIM2]) +
                              sol->get_Field(level, cube_index_list[MID - NDIM2]) - 2 * phi) /
                             (h * h);
                }

                double phi_xx = (sol->get_Field(level, cube_index_list[MID + 1]) +
                                 sol->get_Field(level, cube_index_list[MID - 1]) - 2 * phi) /
                                (h * h);

                double phi_yy = (sol->get_Field(level, cube_index_list[MID + NDIM]) +
                                 sol->get_Field(level, cube_index_list[MID - NDIM]) - 2 * phi) /
                                (h * h);

                //======================================================================
                // The equation D^2 phi + fac*(Sum phi_iiphi_jj - phi_ij^2) - source = 0
                // and the derivative of it. Here we use the operator splitting method to make it stable
                //======================================================================
                const double w = 1.0 / 3.0;
                const double laplacian = phi_xx + phi_yy + phi_zz;
                const double cross = phi_xx * phi_xx + phi_yy * phi_yy + phi_zz * phi_zz +
                                     2.0 * (phi_xy * phi_xy + phi_yz * phi_yz + phi_zx * phi_zx);
                const double sigma = (cross + OmegaM * a * alpha_DGP / beta_DGP * delta - w * laplacian * laplacian);
                const double disc = 1.0 + 4.0 * (1 - w) * sigma / (alpha_DGP * alpha_DGP);
                const double eq = laplacian - 0.5 * alpha_DGP * (std::sqrt(disc) - 1.0) / (1.0 - w);
                const double deq = -2.0 * NDIM / (h * h);

                // Check for fatal error
                if (disc < 0.0) {
                    throw std::runtime_error(
                        "The discriminant is negative for DGP (this can happen!). Solution fails to remain real!");
                }

                //======================================================================
                // Returns the equation and the derivative to the solver
                //======================================================================
                return std::pair<SolverType, SolverType>{eq, deq};
            };

        // Solve the equation and fetch the solution
        g.solve(Equation, ConvergenceCriterion);
        auto sol = g.get_grid(0);
        ConvertToFFTWGrid(sol, fifth_force_potential_real);

        // Convert to fifth-force potential
        const double forcenorm = 0.5;

        FML::GRID::FloatType fmin = std::numeric_limits<FML::GRID::FloatType>::max();
        FML::GRID::FloatType fmax = -std::numeric_limits<FML::GRID::FloatType>::max();
        FML::GRID::FloatType fmean = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : fmax) reduction(min : fmin) reduction(+ : fmean)
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && real_index : fifth_force_potential_real.get_real_range(islice, islice + 1)) {
                auto value = fifth_force_potential_real.get_real_from_index(real_index);
                value *= forcenorm;
                fifth_force_potential_real.set_real_from_index(real_index, value);

                fmean += value;
                fmax = std::max(fmax, value);
                fmin = std::min(fmin, value);
            }
        }
        FML::SumOverTasks(&fmean);
        fmean /= std::pow(Nmesh, NDIM);

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "#=====================================================\n";
            std::cout << "# The minimum value of phi_DGP : " << fmin - fmean << "\n";
            std::cout << "# The mean value of phi_DGP: " << fmean << "\n";
            std::cout << "# The maximum value phi_DGP: " << fmax - fmean << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
