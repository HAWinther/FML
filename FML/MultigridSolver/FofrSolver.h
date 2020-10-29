#ifndef FOFRSOLVER_HEADER
#define FOFRSOLVER_HEADER

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
/// Solver the f(R) gravity equation in a cosmological setting
/// The solver returns PhiF = a^2 f_R / (2 (H0 Box)^2) so that in a
/// cosmological setting where PhiGR = a^2 Phi / (H0 Box)^2 we simply
/// have the total force = D(PhiGR + PhiF)
/// This class (kind of) assumes a LCDM cosmology 
//======================================================================
template <int NDIM, typename SolverType = double>
class FofrSolverCosmology {
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
    /// The f(R) n-index
    double nfofr{1.0};
    /// The f(R) f_R0 value
    double fofr0{1e-6};
    /// The dimensionless number H0 * Box / c
    double H0Box{};
    
    /// Print info as we go along
    bool verbose{true};

    /// The box is periodic?
    bool periodic{true};

  public:
    FofrSolverCosmology(double OmegaM, double nfofr, double fofr0, double H0Box, bool verbose)
        : OmegaM(OmegaM), nfofr(nfofr), fofr0(fofr0), H0Box(H0Box), verbose(verbose) {}

    /// Set how many sweeps over the grid to use when solving a given level
    void set_ngs_steps(double _ngs_fine, double _ngs_coarse, double _ngs_first) {
        ngs_fine = _ngs_fine;
        ngs_coarse = _ngs_coarse;
        ngs_first = _ngs_first;
    }

    /// Set convergenc criterion
    void set_epsilon(double _epsilon) { epsilon = _epsilon; }

    /// The cosmological background value f_R(a)
    double get_fofr_background(double a) {
        const double fac1 = 1.0 + 4.0 * (1.0 - OmegaM) / OmegaM;
        const double fac2 = 1.0 / (a * a * a) + 4.0 * (1.0 - OmegaM) / OmegaM;
        return fofr0 * std::pow(fac1 / fac2, nfofr + 1);
    }

    /// Do the solving and return the fifth-force potential. 
    /// We solve for f where a^2 f_R / (2 (H0Box)^2) = e^f and return e^f
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
            std::cout << "# nfofr       : " << nfofr << "\n";
            std::cout << "# H0Box       : " << H0Box << "\n";
            std::cout << "# f_R0        : " << fofr0 << "\n";
            std::cout << "# f_R(a)      : " << get_fofr_background(a) << "\n";
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

        // Factors to define the equation (the first term is GR times 1/3)
        const double poisson_norm = (1.0 / 3.0) * (1.5 * OmegaM * a);
        const double prefac = 0.5 * OmegaM * a * a * a * a * (1.0 / (a * a * a) + 4.0 * (1.0 - OmegaM) / OmegaM);

        // The background value of the field we solve for
        const double f0 = std::log(a * a * get_fofr_background(a) / 2.0 / std::pow(H0Box, 2));

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
                // Calculate the kinetic term: D[ b[f] Df ] with b = exp(x)
                // (Faster to not use the built in function and code this directly
                // as we do a lot of double work)
                //======================================================================
                std::function<SolverType(int, IndexInt)> b = [&](int lev, IndexInt ind) -> double {
                    return std::exp(sol->get_Field(lev, ind));
                };
                std::function<SolverType(int, IndexInt)> db = [&](int lev, IndexInt ind) -> double {
                    return std::exp(sol->get_Field(lev, ind));
                };
                auto kinetic = sol->get_BLaplacian(level, index_list, b);
                auto dkinetic = sol->get_derivBLaplacian(level, index_list, b, db);

                //======================================================================
                // The right hand side source and the derivative of it
                //======================================================================
                double source = poisson_norm * delta - prefac * (std::exp((f0 - f) / (1.0 + nfofr)) - 1.0);
                double dsource = prefac * (std::exp((f0 - f) / (1.0 + nfofr))) / (1.0 + nfofr);

                //======================================================================
                // Returns the equation and the derivative to the solver
                //======================================================================
                return std::pair<SolverType, SolverType>{kinetic - source, dkinetic - dsource};
            };

        // Solve the equation and fetch the solution
        g.solve(Equation, ConvergenceCriterion);
        auto sol = g.get_grid(0);
        ConvertToFFTWGrid(sol, fifth_force_potential_real);

        // Convert to fifth-force potential a^2 f_R / (2 (H0Box)^2)
        FML::GRID::FloatType fmin = std::numeric_limits<FML::GRID::FloatType>::max();
        FML::GRID::FloatType fmax = -std::numeric_limits<FML::GRID::FloatType>::max();
        FML::GRID::FloatType fmean = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : fmax) reduction(min : fmin) reduction(+ : fmean)
#endif
        for (int islice = 0; islice < Local_nx; islice++) {
            for (auto && real_index : fifth_force_potential_real.get_real_range(islice, islice + 1)) {
                auto value = fifth_force_potential_real.get_real_from_index(real_index);
                value = std::exp(value);
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
            std::cout << "# The minimum value of f_R / f_R(a): " << fmin / std::exp(f0) << "\n";
            std::cout << "# The mean value of f_R / f_R(a): " << fmean / std::exp(f0) << "\n";
            std::cout << "# The maximum value of f_R / f_R(a): " << fmax / std::exp(f0) << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }
};

#endif
