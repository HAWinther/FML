#ifndef SIMULATIONGRID_HEADER
#define SIMULATIONGRID_HEADER

// The grid type of the simulation: FFTWGrid (slabs, the default) or, with USE_PENCIL_GRID (make USE_PENCIL_GRID=true),
// PencilGrid (pencils: the domain FML::domain is then a PencilDomain, set at the start of the run from
// FML_PENCILS="px py" or MPI_Dims_create). Every part of the simulation (all gravity models, COLA, lightcones, neutrinos,
// ICs, every output and analysis) works with both. All grids are made with make_simulation_grid / FML::GRID::make_grid
// (decomposed as FML::domain, where the particles are)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
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

/// A simulation grid of Nmesh cells per dim decomposed as the particles (FML::domain)
template <int N>
SimulationGrid<N> make_simulation_grid(int Nmesh, int n_extra_left = 0, int n_extra_right = 0) {
    return FML::GRID::make_grid<SimulationGrid<N>>(Nmesh, n_extra_left, n_extra_right);
}

#endif
