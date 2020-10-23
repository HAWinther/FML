#ifndef SYMMETRONSOLVER_HEADER
#define SYMMETRONSOLVER_HEADER

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
/// Solver the symmetron gravity equation in a cosmological setting
/// The solver returns PhiF so that in a
/// cosmological setting where PhiGR = a^2 Phi / (H0 Box)^2 we simply
/// have the total force = D(PhiGR + PhiF)
/// This class (kind of) assumes a LCDM cosmology
//======================================================================
template <int NDIM, typename SolverType = double>
class SymmetronSolverCosmology {
  private:
    int Nmesh{};
    int Nlevels{-1};
    int nleft{1};
    int nright{1};

    int ngs_fine{10};
    int ngs_coarse{10};
    int ngs_first{40};

    /// Convergence criterion: residual < epsilon
    double epsilon{1e-6};

    /// Matter density parameter today
    double OmegaM{0.3};
    /// The symmetry breaking scalefactor
    double assb{1e-6};
    /// The coupling strength
    double beta{1.0};
    /// The range today in Mpc/h
    double L_mpch{1e-6};
    /// The dimensionless number H0 * Box / c
    double H0Box{};

    /// Print info as we go along
    bool verbose{true};

    /// The box is periodic?
    bool periodic{true};

  public:
    SymmetronSolverCosmology(double OmegaM, double assb, double beta, double L_mpch, double H0Box, bool verbose)
        : OmegaM(OmegaM), assb(assb), beta(beta), L_mpch(L_mpch), H0Box(H0Box), verbose(verbose) {}

    /// Set how many sweeps over the grid to use when solving a given level
    void set_ngs_steps(double _ngs_fine, double _ngs_coarse, double _ngs_first) {
        ngs_fine = _ngs_fine;
        ngs_coarse = _ngs_coarse;
        ngs_first = _ngs_first;
    }

    /// Set convergenc criterion
    void set_epsilon(double _epsilon) { epsilon = _epsilon; }

    /// The cosmological background value f_R(a)
    double get_phi_background(double a) { return a < assb ? 0.0 : std::sqrt(1.0 - (assb * assb * assb) / (a * a * a)); }

    /// Do the solving and return the fifth-force potential.
    /// We solve for f where a phi / phi0 f_R / (2 (H0Box)^2) = e^f and return e^f
    void solve(double a, FFTWGrid<NDIM> & overdensity_real, FFTWGrid<NDIM> & fifth_force_potential_real) {

        Nmesh = overdensity_real.get_nmesh();
        auto Local_nx = overdensity_real.get_local_nx();

        if (FML::ThisTask == 0 and verbose) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Running multigridsolver for f(R)\n";
            std::cout << "# Nmesh       : " << Nmesh << "\n";
            std::cout << "# a           : " << a << "\n";
            std::cout << "# OmegaM      : " << OmegaM << "\n";
            std::cout << "# assb        : " << assb << "\n";
            std::cout << "# beta        : " << beta << "\n";
            std::cout << "# L_mpch      : " << L_mpch << "\n";
            std::cout << "# H0Box       : " << H0Box << "\n";
            std::cout << "# phi/phi0(a) : " << get_phi_background(a) << "\n";
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
        const double H0L = L_mpch / 2997.92458;
        const double norm = 0.5 * a * a * (H0Box * H0Box) / (H0L * H0L);
        const double fac = (assb * assb * assb) / (a * a * a);

        // The background value of the field we solve for
        const double f0 = get_phi_background(a);

        // Set up solver
        MultiGridSolver<NDIM, SolverType> g(Nmesh, Nlevels, verbose, periodic, nleft, nright);

        // Set some options
        g.set_epsilon(epsilon);
        g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
        g.set_epsilon(epsilon);

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
                //======================================================================
                // Get a list of the index of the closest cells needed to compute
                // gradients and the laplacian
                //======================================================================
                auto index_list = sol->get_neighbor_gridindex(level, index);

                //======================================================================
                // The solution in the current cell
                //======================================================================
                const double f = sol->get_Field(level, index);

                //======================================================================
                // Set the (over) density field from formula
                //======================================================================
                auto delta = density_multigrid.get_y(level, index);

                //======================================================================
                // Calculate the kinetic termi
                //======================================================================
                auto kinetic = sol->get_Laplacian(level, index_list);
                auto dkinetic = sol->get_derivLaplacian(level, index_list);

                //======================================================================
                // The right hand side source and the derivative of it
                //======================================================================
                double source = norm * (fac * delta + (f - f0) * (f + f0)) * f;
                double dsource = norm * (fac * (1.0 + delta) - 1.0 + 3.0 * f * f);

                //======================================================================
                // Returns the equation and the derivative to the solver
                //======================================================================
                return std::pair<SolverType, SolverType>{kinetic - source, dkinetic - dsource};
            };

        // Solve the equation and fetch the solution
        g.solve(Equation, ConvergenceCriterion);
        auto sol = g.get_grid(0);
        ConvertToFFTWGrid(sol, fifth_force_potential_real);

        const double forcenorm =
            3.0 * OmegaM * a * a * beta * beta * (H0L / H0Box) * (H0L / H0Box) / (assb * assb * assb);

        // Convert to fifth-force potential a^2 f_R / (2 (H0Box)^2)
        double fmin = std::numeric_limits<double>::max();
        double fmax = -std::numeric_limits<double>::max();
        double fmean = 0.0;
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
            std::cout << "# The minimum value of phi/phi0 : " << fmin / forcenorm << "\n";
            std::cout << "# The mean value of phi/phi0: " << fmean / forcenorm << "\n";
            std::cout << "# The maximum value phi/phi0: " << fmax / forcenorm << "\n";
            std::cout << "# Background field value phi(a) : " << f0 << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
