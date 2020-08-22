#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/Reconstruction.h>
#include <FML/MPIGrid/ConvertMPIGridFFTWGrid.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/MultigridSolver/MultiGridSolver.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <cmath>
#include <iostream>

//======================================================================
// Example on how to do RSD reconstruction on simulation data
// We solve DPsi + f D((Psi*r)r) = -delta/b with r = (0,0,1)
//======================================================================

using namespace FML::SOLVERS::MULTIGRIDSOLVER;

//======================================================================
// Particle class
//======================================================================
const int NDIM = 3;
struct Particle {
    double x[NDIM];
    Particle() {}
    Particle(double * _x) { std::memcpy(x, _x, NDIM * sizeof(double)); }
    double * get_pos() { return x; }
    double * get_vel() { return nullptr; }
    int get_particle_byte_size() { return NDIM * sizeof(double); }
    void append_to_buffer(char * data) { std::memcpy(data, x, NDIM * sizeof(double)); }
    void assign_from_buffer(char * data) { std::memcpy(x, data, NDIM * sizeof(double)); }
};

void PoissonSolver() {

    //======================================================================
    // Solver parameters
    //======================================================================
    using SolverType = double;
    const int N = 128;
    const int Nlevels = -1;
    const int nleft = 1;
    const int nright = 1;
    const bool verbose = true;
    const bool periodic = true;
    const double epsilon = 1e-7;
    const int ngs_fine = 10;
    const int ngs_coarse = 10;
    const int ngs_first = 10;

    //=================================================================================
    // Read ascii file that has the format [x,y,z] and fetch the position
    // (This file has positions in real-space so no recon is needed, but fuck it its just to test it)
    //=================================================================================
    const std::string filename = "../../../TestData/particles_B1024.txt"; // Filename of particles
    const double box = 1024.0;                                            // The boxsize of the simulation data
    const int ncols = 3;                                                  // Number of columns in file
    const int nskip_header = 0;                                           // Number of header lines to skip
    const std::vector<int> cols_to_keep{0, 1, 2};                         // The columns we want to store
    auto data = FML::FILEUTILS::read_regular_ascii(filename, ncols, cols_to_keep, nskip_header);

    //======================================================================
    // Recon option
    //======================================================================
    const double b = 1.0;       // Galaxy bias
    const double f = 1.0;       // Growthrate
    const double beta = f / b;  // This is all the RSD recon depends on
    const double radius = 10.0; // Smoothing radius in Mpc/h
    const double smoothing_scale = radius / box;
    const std::string smoothing_filter = "gaussian";

    //=================================================================================
    // Create particles and scale them to lie in [0,1)
    //=================================================================================
    std::vector<Particle> part;
    for (auto & pos : data) {
        for (auto & x : pos)
            x /= box;
        part.push_back(Particle(pos.data()));
    }

    //=================================================================================
    // For MPI use: create MPI particles by letting each task keep only the particles
    // that falls in its domain. If you run with one task you don't need this
    //=================================================================================
    FML::PARTICLE::MPIParticles<Particle> p;
    const bool all_tasks_have_the_same_particles = true;
    const int nalloc_per_task = part.size() / FML::NTasks * 2;
    p.create(part.data(),
             part.size(),
             nalloc_per_task,
             FML::xmin_domain,
             FML::xmax_domain,
             all_tasks_have_the_same_particles);

    //=================================================================================
    // This can be used to displace particles to redshift-space if we have velocities
    // particles_to_redshiftspace(p, std::vector<double>{0,0,1}, 1.0);
    //=================================================================================

    //=================================================================================
    // Particles to grid
    //=================================================================================
    // The density assignement method (Cloud in Cell)
    const std::string density_assignment_method = "CIC";
    // Set up density grid
    FML::GRID::FFTWGrid<NDIM> density(N, 0, 1);
    // Bin particles to the grid
    FML::INTERPOLATION::particles_to_grid<NDIM, Particle>(
        p.get_particles().data(), p.get_npart(), p.get_npart_total(), density, density_assignment_method);

    //=================================================================================
    // Smoothing in Fourier space
    //=================================================================================
    density.fftw_r2c();
    FML::GRID::smoothing_filter_fourier_space(density, smoothing_scale, smoothing_filter);
    density.fftw_c2r();

    //======================================================================
    // We don't just need delta on the main grid, but at all grid-levels
    // so we need to compute this. We do this by setting up a multigrid
    // this is a stack of grids with Nmesh, Nmesh/2, ... , 2, 1 cells
    // per dimension and then we interpolate down the grid (restrict_down_all)
    // to get delta at all levels
    //======================================================================
    MPIMultiGrid<NDIM, double> density_multigrid(N);
    auto & grid = density_multigrid.get_grid();
    for (auto real_index : density.get_real_range()) {
        // Fetch the value in the cell
        auto density_in_cell = density.get_real_from_index(real_index);
        // Fetch the coordinates of the cell (ix,iy,iz)
        auto coord = density.get_coord_from_index(real_index);
        auto density_grid_index = grid.index_from_coord(coord);
        // Set the value
        grid.set_y(density_grid_index, density_in_cell);
    }
    density_multigrid.restrict_down_all();

    //======================================================================
    // Set up the solver
    //======================================================================
    MultiGridSolver<NDIM, SolverType> g(N, Nlevels, verbose, periodic, nleft, nright);

    // Set some options
    g.set_epsilon(epsilon);
    g.set_ngs_sweeps(ngs_fine, ngs_coarse, ngs_first);
    g.set_epsilon(epsilon);

    // Set the initial guess
    g.set_initial_guess(SolverType(0.0));

    //======================================================================
    // Set the convergence criterion
    // (ConvergenceCriterionResidual is rms-residual < epsilon)
    //======================================================================
    MultiGridConvCrit ConvergenceCriterion = [&](double rms_residual, double rms_residual_ini, int step_number) {
        return g.ConvergenceCriterionResidual(rms_residual, rms_residual_ini, step_number);
    };

    //======================================================================
    // Define the equation D Psi + f/b D( (Psi*r)r) = -delta
    // where Psi is really b*Psi
    //======================================================================
    MultiGridFunction<NDIM, SolverType> Equation = [&](MultiGridSolver<NDIM, SolverType> * sol,
                                                       int level,
                                                       IndexInt index) {
        auto h = sol->get_Gridspacing(level);
        auto index_list = sol->get_neighbor_gridindex(level, index);
        auto d2phidz2 = (sol->get_Field(level, index_list[2 * NDIM]) + sol->get_Field(level, index_list[2 * NDIM - 1]) -
                         2 * sol->get_Field(level, index_list[0])) /
                        (h * h);
        auto delta = density_multigrid.get_y(level, index);
        auto L = sol->get_Laplacian(level, index_list) + beta * d2phidz2 + delta;
        auto dL = -2 * (NDIM + beta) / (h * h);
        return std::pair<SolverType, SolverType>{L, dL};
    };

    //======================================================================
    // Solve the equation and fetch the solution
    //======================================================================
    g.solve(Equation, ConvergenceCriterion);
    auto sol = g.get_grid(0);

    //======================================================================
    // Compute the RSD shift
    // Psi_rsd = -beta[(Psi*r)r] = -beta phi_z in the z direction
    //======================================================================
    MPIGrid<NDIM, double> rsd_shift(
        N, true, 0, 1); // The extra slices we set here gets propagated to the FFTWGrid below
    for (IndexInt index = 0; index < sol.get_NtotLocal(); index++) {
        auto gradient = sol.get_gradient(index);
        rsd_shift[index] = -beta * gradient[NDIM - 1];
    }

    //======================================================================
    // Convert to FFTWGrid to be use the interpolation routines
    //======================================================================
    FML::GRID::FFTWGrid<NDIM> rsd_shift_fftw;
    ConvertToFFTWGrid(rsd_shift, rsd_shift_fftw);

    //======================================================================
    // Interpolate RSD shift to particle positions
    //======================================================================
    std::vector<double> interpolated_values;
    FML::INTERPOLATION::interpolate_grid_to_particle_positions(
        rsd_shift_fftw, p.get_particles_ptr(), p.get_npart(), interpolated_values, "CIC");

    //======================================================================
    // Subtract RSD
    //======================================================================
    double max_shift = 0.0;
    double avg_shift = 0.0;
    for (size_t i = 0; i < p.get_npart(); i++) {
        auto * pos = p[i].get_pos();
        const double shift = interpolated_values[i];
        pos[NDIM - 1] -= shift;
        if (std::abs(shift) > max_shift)
            max_shift = std::abs(shift);
        avg_shift += shift;
    }
    std::cout << "maximum shift " << max_shift << " avg shift " << avg_shift / double(p.get_npart()) << "\n";

    // We could alternatively have done this with an iterative fourier method (its not exact, but makes up for it by
    // doing it iteratively):
    // FML::COSMOLOGY::LPT::RSDReconstructionFourierMethod<NDIM, Particle>(
    //    p, "CIC", std::vector<double>{0.0, 0.0, 1.0}, N, 3, beta, {"gaussian", smoothing_scale}, false);
}

int main() { PoissonSolver(); }
