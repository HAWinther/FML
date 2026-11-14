#ifndef SIMULATIONGRID_HEADER
#define SIMULATIONGRID_HEADER

// The grid type of the simulation: FFTWGrid (slabs, the fiducial) or with USE_PENCIL_GRID (make USE_PENCIL_GRID=true)
// PencilGrid (pencils: the domain FML::domain is then a PencilDomain, set at the start of the run). All grids are made
// with FML::GRID::make_grid (decomposed as FML::domain, where the particles are)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <string>
#ifdef USE_PENCIL_GRID
#include <FML/PencilDomain/PencilGrid.h>
#endif

#ifdef USE_PENCIL_GRID
template <int N>
using SimulationGrid = FML::GRID::PencilGrid<N>;
#else
template <int N>
using SimulationGrid = FML::GRID::FFTWGrid<N>;
#endif

/// Is the simulation grid the FFTWGrid? (some analyses are only available with it: the hybrid bias spectra, the sheet
/// density)
template <int N>
inline constexpr bool simulation_grid_is_fftwgrid = FML::GRID::is_fftwgrid_v<SimulationGrid<N>>;

/// Abort for a method that is not available with the pencils (USE_PENCIL_GRID)
inline void not_available_with_pencils(const std::string & what) {
    FML::assert_mpi(false, (what + " is not available with the pencils (USE_PENCIL_GRID) yet").c_str());
}

/// A simulation grid of Nmesh cells per dim decomposed as the particles (FML::domain)
template <int N>
SimulationGrid<N> make_simulation_grid(int Nmesh, int n_extra_left = 0, int n_extra_right = 0) {
    return FML::GRID::make_grid<SimulationGrid<N>>(Nmesh, n_extra_left, n_extra_right);
}

#endif
