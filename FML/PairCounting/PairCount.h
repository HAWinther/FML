#ifndef PAIRCOUNT_HEADER
#define PAIRCOUNT_HEADER

#include <algorithm>
#include <array>
#include <cassert>
#include <string>
#include <type_traits>
#include <cmath>
#include <functional>
#include <iomanip>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_GSL
#include <FML/Spline/Spline.h>
#endif

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/ParticlesInBoxes/ParticlesInBoxes.h>

namespace FML {

    namespace CORRELATIONFUNCTIONS {

        //===========================================================================
        ///
        /// This namespace deals with paircounting and contains
        /// general auto and cross paircount methods in 1D, 2D or 3D for simulation boxes and survey data
        /// (with randoms). Implemented radial and angular (r,mu). For the latter one can specify the
        /// observer position. Fiducial choice is to use the midpoint of the pair to define the angle wrt
        /// the line of sight.
        ///
        /// Allows you to also bin up whatever other quantities you want on the pairs by
        /// providing a function that does this (e.g. v12 = v1-v2).
        ///
        /// Parallelized with both OpenMP and MPI. For MPI its assumed that all tasks have *all* the particles
        /// and each task is then responsible for a certain part of the pairs we want to count.
        ///
        /// Particles are assumed to reside in [0,1)^n
        /// rmax is the maximum pair separation (in [0,1)) that we are interested in
        /// Periodic means the particle positions wraps around (0.0 == 1.0)
        ///
        /// We allow for different particle types for all particle/randoms to be correlated
        /// as long as they have the basic method below.
        ///
        /// Particles must have the methods:
        /// auto *get_pos()
        /// int get_ndim()
        /// double get_weight() (just return 1.0 if no weight)
        ///
        /// For the available estimators for survey correlation functions see the method
        /// CorrelationFunctionEstimator below (LZ, DP, HAW, HP, PH, ...).
        ///
        //===========================================================================
        namespace PAIRCOUNTS {

            /// The function that is evaluated at each pair. Does all the binning
            template <typename T, typename U = T>
            using BinningFunction = std::function<void(int, double *, T &, U &)>;

            /// If we want to bin up other quantities this is the function we provide
            template <typename T1, typename T2 = T1>
            using ExtraQuantitiesToBinFunction = std::function<
                void(const double * dr, double r, double mu, const T1 & part1, const T2 & part2, double * storage)>;

            // Typedefs used below
            using DVector = std::vector<double>;
            using DVector2D = std::vector<DVector>;
            using DVector3D = std::vector<DVector2D>;
#ifdef USE_GSL
            using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
            using Spline = FML::INTERPOLATION::SPLINE::Spline;
#endif

            // For fiducial parameters
            inline DVector3D empty_3D_array{};
            inline DVector2D empty_2D_array{};

            /// A large value that we can use that ensures that if the observer is at
            /// say (0,0,-effective_infinity) then we have a fixed line of sight
            /// direction (z-axis is this example).
            inline double effective_infinity = 1e10;

            /// For binning particles to cells. The minimum number of cells to reach the maximum distance
            inline int ncells_to_rmax = 8;
            /// For binning particles to cells. The minimum grid-size.
            inline int ngrid_min_size = 10;

            /// This is the general algorithm for computing pair counts.
            /// It finds all pairs within a separation rmax and calls the binning_function
            /// for all pairs from which you can take care of everything.
            /// NB: the binning_function is called
            template <class T, class U>
            void GeneralPairCounter(FML::PARTICLE::ParticlesInBoxes<T> & grid,
                                    FML::PARTICLE::ParticlesInBoxes<U> & grid2,
                                    const std::function<void(int, double *, T &, U &)> & binning_function,
                                    bool auto_pair_binning,
                                    double rmax,
                                    bool periodic_box,
                                    bool verbose = false);

            /// Estimates the correlation function from the paircounts. Most common estimators are included.
            /// NB: assumes normalized paircounts here, e.g. pairs / total_number_of_pairs.
            inline double CorrelationFunctionEstimator(double D1D2,
                                                double D1R2,
                                                double R1D2,
                                                double R1R2,
                                                const std::string & estimator) {

                // The estimators below that were proposed for auto pair-counts have been
                // symmetrized to also give a result for cross pair-counts
                // See DOI: 10.1051/0004-6361/201220790 for a summary
                // and https://arxiv.org/pdf/astro-ph/9912088.pdf for a comparison
                if (estimator == "LZ" or estimator == "LS") {
                    // Landy & Szalay 1993
                    if (R1R2 == 0.0)
                        return 0.0;
                    return (D1D2 + R1R2 - (D1R2 + R1D2)) / R1R2;
                } else if (estimator == "HAM") {
                    // Hamilton 1993
                    if (D1R2 == 0.0 or R1D2 == 0.0)
                        return 0.0;
                    return D1D2 * R1R2 / (D1R2 * R1D2) - 1.0;
                } else if (estimator == "HAM_no_R1") {
                    if (D1R2 == 0.0)
                        return 0.0;
                    return D1D2 * R1R2 / (D1R2 * D1R2) - 1.0;
                } else if (estimator == "HAM_no_R2") {
                    if (R1D2 == 0.0)
                        return 0.0;
                    return D1D2 * R1R2 / (R1D2 * R1D2) - 1.0;
                } else if (estimator == "HEW") {
                    // Hewett 1982
                    if (R1R2 == 0.0)
                        return 0.0;
                    return (D1D2 - 0.5 * (D1R2 + R1D2)) / R1R2;
                } else if (estimator == "HEW_no_R1") {
                    if (R1R2 == 0.0)
                        return 0.0;
                    return (D1D2 - D1R2) / R1R2;
                } else if (estimator == "HEW_no_R2") {
                    if (R1R2 == 0.0)
                        return 0.0;
                    return (D1D2 - R1D2) / R1R2;
                } else if (estimator == "DP") {
                    // Davis & Peebles 1983
                    if (D1R2 + R1D2 == 0.0)
                        return 0.0;
                    return D1D2 / (0.5 * (D1R2 + R1D2)) - 1.0;
                } else if (estimator == "DP_no_R2") {
                    if (R1D2 == 0.0)
                        return 0.0;
                    return D1D2 / R1D2 - 1.0;
                } else if (estimator == "DP_no_R1") {
                    if (D1R2 == 0.0)
                        return 0.0;
                    return D1D2 / D1R2 - 1.0;
                } else if (estimator == "PH") {
                    // Peebles & Hauser 1974
                    if (R1R2 == 0.0)
                        return 0.0;
                    return D1D2 / R1R2 - 1.0;
                } else {
                    throw std::runtime_error(
                        "[CorrelationFunctionEstimator] Unknown estimator " + estimator +
                        ". Options: LZ (= LS), HAM, HAM_no_R1, HAM_no_R2, HEW, HEW_no_R1, HEW_no_R2, PH, DP, "
                        "DP_no_R1, DP_no_R2\n");
                }
            }

            //====================================================
            /// Cross pair counts using grid to speed it up
            /// Cross seems to be faster if we loop over the coarsest
            /// grid first
            //====================================================
            template <class T, class U>
            void GeneralPairCounter(FML::PARTICLE::ParticlesInBoxes<T> & grid1,
                                    FML::PARTICLE::ParticlesInBoxes<U> & grid2,
                                    const std::function<void(int, double *, T &, U &)> & binning_function,
                                    bool auto_pair_binning,
                                    double rmax,
                                    bool periodic_box,
                                    bool verbose) {

                // Initialize OpenMP
                const int nthreads = FML::NThreads;
                int id = 0;

                // Initialize MPI
                [[maybe_unused]] int mpi_rank = 0;
                [[maybe_unused]] int mpi_size = 1;
#if defined(USE_MPI)
                MPI_Comm_rank(FML::communicator, &mpi_rank);
                MPI_Comm_size(FML::communicator, &mpi_size);
                verbose = verbose and mpi_rank == 0;
#endif

                // Fetch ndim
                // (from objects: GetNDIM(T()) needs a particle that can be built at compile time)
                T tmp_part1;
                U tmp_part2;
                constexpr int ndim = FML::PARTICLE::GetNDIM(tmp_part1);
                static_assert(ndim == FML::PARTICLE::GetNDIM(tmp_part2), "[GeneralPairCounter] Particles must have the same NDIM");

                // Fetch data from the grid
                auto & cells = grid1.get_cells();
                const int ngrid1 = grid1.get_ngrid();
                int max_ix = ngrid1 - 1;
                int max_iy = ngrid1 - 1;
                int max_iz = ngrid1 - 1;

                // Fetch data from the grid2
                auto & cells2 = grid2.get_cells();
                const int ngrid2 = grid2.get_ngrid();
                int max_ix2 = ngrid2 - 1;
                int max_iy2 = ngrid2 - 1;
                int max_iz2 = ngrid2 - 1;

                // Find maximum ix,iy,iz for which there are particles
                // in case for surveys where after boxing the particles
                // only occupy a part of the box
                if (not periodic_box) {
                    max_ix = max_iy = max_iz = 0;
                    for (int ix = ngrid1 - 1; ix >= 0; ix--) {
                        for (int iy = ngrid1 - 1; iy >= 0; iy--) {
                            for (int iz = ngrid1 - 1; iz >= 0; iz--) {
                                size_t index;
                                if (ndim == 1) {
                                    index = ix;
                                    iy = iz = 0;
                                }
                                if (ndim == 2) {
                                    index = ix * ngrid1 + iy;
                                    iz = 0;
                                }
                                if (ndim == 3) {
                                    index = (ix * ngrid1 + iy) * ngrid1 + iz;
                                }
                                bool nonempty = cells[index].get_np() > 0;
                                if (nonempty) {
                                    max_ix = std::max(ix, max_ix);
                                    max_iy = std::max(iy, max_iy);
                                    max_iz = std::max(iz, max_iz);
                                }
                            }
                        }
                    }
                    max_ix2 = max_iy2 = max_iz2 = 0;
                    for (int ix = ngrid2 - 1; ix >= 0; ix--) {
                        for (int iy = ngrid2 - 1; iy >= 0; iy--) {
                            for (int iz = ngrid2 - 1; iz >= 0; iz--) {
                                size_t index;
                                if (ndim == 1) {
                                    index = ix;
                                    iy = iz = 0;
                                }
                                if (ndim == 2) {
                                    index = ix * ngrid2 + iy;
                                    iz = 0;
                                }
                                if (ndim == 3) {
                                    index = (ix * ngrid2 + iy) * ngrid2 + iz;
                                }
                                bool nonempty = cells2[index].get_np() > 0;
                                if (nonempty) {
                                    max_ix2 = std::max(ix, max_ix2);
                                    max_iy2 = std::max(iy, max_iy2);
                                    max_iz2 = std::max(iz, max_iz2);
                                }
                            }
                        }
                    }
                }
                if (ndim == 1)
                    max_iz = max_iz2 = max_iy = max_iy2 = 0;
                if (ndim == 2)
                    max_iz = max_iz2 = 0;

                // How many cells in each direction we must search in the second grid around the cell of grid2 that
                // contains the particle (a neighbour within rmax is at most ceil(rmax * ngrid2) cells away; +1 for
                // round-off). With a periodic box we never need more than the whole grid (each cell once)
                int delta_ncells2 = (int)(ceil(rmax * ngrid2)) + 1;
                if (periodic_box and delta_ncells2 > ngrid2 / 2)
                    delta_ncells2 = ngrid2 / 2;
                int delta_ncells2_high = delta_ncells2, delta_ncells2_low = delta_ncells2;
                if (delta_ncells2_low + delta_ncells2_high == ngrid2)
                    delta_ncells2_low--;

                // Print some data
                if (verbose) {
                    std::cout << "\n#=====================================================\n";
                    std::cout << "#                                                       \n";
                    std::cout << "#       __________        .__                           \n";
                    std::cout << "#       \\______   \\_____  |__|______  ______          \n";
                    std::cout << "#        |     ___/\\__  \\ |  \\_  __ \\/  ___/        \n";
                    std::cout << "#        |    |     / __ \\|  ||  | \\/\\___ \\         \n";
                    std::cout << "#        |____|    (____  /__||__|  /____  >            \n";
                    std::cout << "#                       \\/               \\/           \n";
                    std::cout << "#                                                       \n";
                    std::cout << "#    Pair counting with particles binned to a grid      \n";
                    std::cout << "#                                                       \n";
                    if (auto_pair_binning) {
                        std::cout << "# We are binning [autopairs]\n";
                    } else {
                        std::cout << "# We are binning [crosspairs]\n";
                        std::cout << "# Particle grids Ngrid1: " << ngrid1 << " Ngrid2: " << ngrid2 << "\n";
                    }
                    std::cout << "# Using " << nthreads << " threads\n";
                    std::cout << "# Using " << mpi_size << " MPI tasks\n";
                    std::cout << "#\n";
                    if (auto_pair_binning) {
                        std::cout << "# Ngrid particle grid " << ngrid1 << " (*)\n";
                    } else {
                        std::cout << "# Particle grids Ngrid1: " << ngrid1 << " Ngrid2: " << ngrid2 << " (*)\n";
                    }
                    std::cout << "# From each cell in (*) we go " << delta_ncells2_low << " cells left\n";
                    std::cout << "# From each cell in (*) we go " << delta_ncells2_high << " cells right\n";
                    std::cout << "# Fraction of total pairs we have to do ~ "
                              << std::pow((delta_ncells2_low + delta_ncells2_high + 1) / (double)(ngrid2), ndim) * 100.0
                              << " %\n";
                    std::cout << "#\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "\n";
                    std::cout << "Progress: ";
                    std::cout << std::flush;
                }

                //==========================================================
                // Loop over all the cells in grid1
                // The loops only go up to max_ix etc. since we wan to skip
                // looping over cells that we know are empty
                //==========================================================

                int ix0 = 0;
                [[maybe_unused]] int num_processed = 0;
                int istart = 0, iend = max_ix + 1;
#if defined(USE_OMP) && !defined(USE_MPI)
#pragma omp parallel for private(id) schedule(dynamic, 1)
#elif defined(USE_MPI)
                int i_per_task = (max_ix + 1) / mpi_size;
                istart = i_per_task * mpi_rank;
                iend = i_per_task * (mpi_rank + 1);
                if (mpi_rank == mpi_size - 1)
                    iend = max_ix + 1;
#endif
                for (ix0 = istart; ix0 < iend; ix0++) {
#if defined(USE_OMP)
                    id = omp_get_thread_num();
#else
                    id = 0;
#endif

                    // If both OpenMP and MPI then OpenMP loop along y axis
#if defined(USE_OMP) && defined(USE_MPI)
#pragma omp parallel for private(id) schedule(dynamic, 1)
#endif
                    for (int iy0 = 0; iy0 <= max_iy; iy0++) {
#if defined(USE_OMP) && defined(USE_MPI)
                        id = omp_get_thread_num();
#endif
                        for (int iz0 = 0; iz0 <= max_iz; iz0++) {

                            // Index of current cell
                            int index{};
                            if (ndim == 1)
                                index = ix0;
                            if (ndim == 2)
                                index = ix0 * ngrid1 + iy0;
                            if (ndim == 3)
                                index = (ix0 * ngrid1 + iy0) * ngrid1 + iz0;

                            // Pointer to current cell
                            FML::PARTICLE::Cell<T> & curcell = cells[index];

                            // Number of galaxies in current cell
                            int np_cell = curcell.get_np();

                            // Loop over all galaxies in current cell
                            for (int ipart_cell = 0; ipart_cell < np_cell; ipart_cell++) {

                                // Current particle
                                T & curpart_cell = curcell.get_part(ipart_cell);

                                //========================================
                                // Now we find the index of the second grid
                                // this grid corresponds to
                                //========================================
                                // The cell of grid2 that contains the particle. NB: not the cell of grid2 at the
                                // corner of the cell of grid1: with a finer grid2 (e.g. many more randoms than data)
                                // the particle can be several cells of grid2 away from it and the search missed pairs
                                int ix_grid2 = ix0, iy_grid2 = iy0, iz_grid2 = iz0;
                                if (not auto_pair_binning) {
                                    const auto * pos_cell = FML::PARTICLE::GetPos(curpart_cell);
                                    int igrid2[3] = {0, 0, 0};
                                    for (int idim = 0; idim < ndim; idim++)
                                        igrid2[idim] = std::min(std::max(int(pos_cell[idim] * ngrid2), 0), ngrid2 - 1);
                                    ix_grid2 = igrid2[0];
                                    iy_grid2 = igrid2[1];
                                    iz_grid2 = igrid2[2];
                                }

                                // We now want to loop over nearby cells by looking at cube of cells around current cell
                                int ix2_left, iy2_left, iz2_left, ix2_right, iy2_right, iz2_right;
                                if (periodic_box) {
                                    ix2_left = -delta_ncells2_low, ix2_right = delta_ncells2_high;
                                    iy2_left = -delta_ncells2_low, iy2_right = delta_ncells2_high;
                                    iz2_left = -delta_ncells2_low, iz2_right = delta_ncells2_high;
                                } else {
                                    ix2_right = ix_grid2 + delta_ncells2_high <= max_ix2 ?
                                                    ix_grid2 + delta_ncells2_high :
                                                    max_ix2;
                                    iy2_right = iy_grid2 + delta_ncells2_high <= max_iy2 ?
                                                    iy_grid2 + delta_ncells2_high :
                                                    max_iy2;
                                    iz2_right = iz_grid2 + delta_ncells2_high <= max_iz2 ?
                                                    iz_grid2 + delta_ncells2_high :
                                                    max_iz2;
                                    ix2_left = ix_grid2 - delta_ncells2_low >= 0 ? ix_grid2 - delta_ncells2_low : 0;
                                    iy2_left = iy_grid2 - delta_ncells2_low >= 0 ? iy_grid2 - delta_ncells2_low : 0;
                                    iz2_left = iz_grid2 - delta_ncells2_low >= 0 ? iz_grid2 - delta_ncells2_low : 0;
                                }
                                if (ndim == 1)
                                    iy2_left = iy2_right = iz2_left = iz2_right = 0;
                                if (ndim == 2)
                                    iz2_left = iz2_right = 0;

                                // Loop over neighbor cells
                                for (int delta_ix2 = ix2_left; delta_ix2 <= ix2_right; delta_ix2++) {
                                    int ix2 = delta_ix2;
                                    if (periodic_box) {
                                        ix2 = ix_grid2 + delta_ix2;
                                        while (ix2 >= ngrid2)
                                            ix2 -= ngrid2;
                                        while (ix2 < 0)
                                            ix2 += ngrid2;
                                    }
                                    // Avoid double counting so we skip cells that have been correlated with this one
                                    // before
                                    if (auto_pair_binning and ix2 < ix0)
                                        continue;

                                    for (int delta_iy2 = iy2_left; delta_iy2 <= iy2_right; delta_iy2++) {
                                        int iy2 = delta_iy2;
                                        if (periodic_box) {
                                            iy2 = iy_grid2 + delta_iy2;
                                            while (iy2 >= ngrid2)
                                                iy2 -= ngrid2;
                                            while (iy2 < 0)
                                                iy2 += ngrid2;
                                        }
                                        // Avoid double counting so we skip cells that have been correlated with this
                                        // one before
                                        if (auto_pair_binning and ix2 == ix0 and iy2 < iy0)
                                            continue;

                                        for (int delta_iz2 = iz2_left; delta_iz2 <= iz2_right; delta_iz2++) {
                                            int iz2 = delta_iz2;
                                            if (periodic_box) {
                                                iz2 = iz_grid2 + delta_iz2;
                                                while (iz2 >= ngrid2)
                                                    iz2 -= ngrid2;
                                                while (iz2 < 0)
                                                    iz2 += ngrid2;
                                            }
                                            // Avoid double counting so we skip cells that have been correlated with
                                            // this one before
                                            if (auto_pair_binning and ix2 == ix0 and iy2 == iy0 and iz2 < iz0)
                                                continue;

                                            // Index of neighboring cell
                                            int index_neighbor_cell{};
                                            if (ndim == 1)
                                                index_neighbor_cell = ix2;
                                            if (ndim == 2)
                                                index_neighbor_cell = (ix2 * ngrid2 + iy2);
                                            if (ndim == 3)
                                                index_neighbor_cell = (ix2 * ngrid2 + iy2) * ngrid2 + iz2;

                                            // Pointer to neighboring cell
                                            FML::PARTICLE::Cell<U> & neighborcell = cells2[index_neighbor_cell];

                                            // Number of galaxies in neighboring cell
                                            int npart_neighbor_cell = neighborcell.get_np();

                                            // Careful: if the nbor cell is the same as the current cell then
                                            // we will overcount if we do all particles so only correlate with
                                            // partices we haven't touched yet
                                            int istart_nbor_cell = 0;
                                            if (auto_pair_binning and index == index_neighbor_cell)
                                                istart_nbor_cell = ipart_cell + 1;

                                            // Loop over galaxies in neighbor cells
                                            for (int ipart_neighbor_cell = istart_nbor_cell;
                                                 ipart_neighbor_cell < npart_neighbor_cell;
                                                 ipart_neighbor_cell++) {

                                                // Galaxy in neighboring cell
                                                U & curpart_neighbor_cell = neighborcell.get_part(ipart_neighbor_cell);

                                                // ==================================================================
                                                // We now count up the pair [curpart_cell] x [curpart_neighbor_cell]
                                                // ==================================================================
                                                auto pos = FML::PARTICLE::GetPos(curpart_cell);
                                                auto pos_nbor = FML::PARTICLE::GetPos(curpart_neighbor_cell);
                                                double dist[ndim];
                                                if (periodic_box) {
                                                    for (int idim = 0; idim < ndim; idim++) {
                                                        dist[idim] = (pos[idim] - pos_nbor[idim]);
                                                        if (dist[idim] > 0.5)
                                                            dist[idim] -= 1.0;
                                                        if (dist[idim] < -0.5)
                                                            dist[idim] += 1.0;
                                                    }
                                                } else {
                                                    for (int idim = 0; idim < ndim; idim++)
                                                        dist[idim] = (pos[idim] - pos_nbor[idim]);
                                                }

                                                // Add to bin
                                                binning_function(id, dist, curpart_cell, curpart_neighbor_cell);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Show progress...
#if defined(USE_OMP)
                    int num_max = max_ix;
#if defined(USE_MPI)
                    num_max = max_iy;
#endif
#pragma omp critical
                    {
                        if (verbose) {
                            // Progress bar
                            int ntot = num_max / mpi_size;
                            if (ntot > 0) {
                                if ((10 * num_processed) / ntot != (10 * num_processed + 10) / ntot or
                                    num_processed == 0) {
                                    std::cout << (100 * num_processed + 100) / ntot << "% " << std::flush;
                                    if (num_processed == ntot - 1) {
                                        std::cout << "\nFinished on first task, waiting for the rest to finish\n"
                                                  << std::endl;
                                    }
                                }
                            }
                        }
                        num_processed++;
                    }
#endif
                }
            }

            namespace detail {

                //====================================================
                /// The particles of a catalogue sorted into the cells of a grid (ngrid^NDIM cells covering [0,1)^NDIM):
                /// positions, weights and the original index in one array each, cell c holding [start[c], start[c+1])
                //====================================================
                template <int NDIM>
                struct CellList {
                    int ngrid{0};
                    std::vector<size_t> start;
                    std::vector<double> pos;
                    std::vector<double> weight;
                    std::vector<size_t> index;
                    double sum_weights{0.0};
                    double sum_weights_squared{0.0};
                    size_t size() const { return index.size(); }
                };

                /// The grid size for pair counting up to rmax: cells of size ~ rmax / 2 (the neighbours of a cell are then
                /// within 2 cells), but on average at least ~2 particles per cell of the largest catalogue (with fewer
                /// the loop over the cells costs more than the pairs) and at most 2^27 cells
                inline int choose_ngrid(int ndim, double rmax, size_t npart_max) {
                    const double ncells_max = std::min(std::max(0.5 * double(npart_max), 1.0), double(1 << 27));
                    double ngrid = std::min(2.0 / rmax, std::pow(ncells_max, 1.0 / ndim));
                    return std::max(1, int(ngrid));
                }

                template <int NDIM, class T>
                CellList<NDIM> make_cell_list(const T * part, size_t npart, int ngrid) {
                    CellList<NDIM> cl;
                    cl.ngrid = ngrid;
                    size_t ncells = 1;
                    for (int d = 0; d < NDIM; d++)
                        ncells *= size_t(ngrid);
                    std::vector<size_t> cell(npart);
                    std::vector<size_t> count(ncells + 1, 0);
                    for (size_t i = 0; i < npart; i++) {
                        const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(part[i]));
                        size_t c = 0;
                        for (int d = 0; d < NDIM; d++)
                            c = c * ngrid + size_t(std::min(std::max(int(pos[d] * ngrid), 0), ngrid - 1));
                        cell[i] = c;
                        count[c + 1]++;
                    }
                    for (size_t c = 0; c < ncells; c++)
                        count[c + 1] += count[c];
                    cl.start = count;
                    cl.pos.resize(npart * NDIM);
                    cl.weight.resize(npart);
                    cl.index.resize(npart);
                    for (size_t i = 0; i < npart; i++) {
                        const size_t k = count[cell[i]]++;
                        const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(part[i]));
                        for (int d = 0; d < NDIM; d++)
                            cl.pos[k * NDIM + d] = double(pos[d]);
                        double w = 1.0;
                        if constexpr (FML::PARTICLE::has_get_weight<T>())
                            w = FML::PARTICLE::GetWeight(part[i]);
                        cl.weight[k] = w;
                        cl.index[k] = i;
                        cl.sum_weights += w;
                        cl.sum_weights_squared += w * w;
                    }
                    return cl;
                }

                //====================================================
                /// Calls pair_function(thread, i, j, dist, r2) for all the pairs (i in A, j in B, positions of the cell
                /// lists, dist = x_i - x_j, minimum image if periodic) with r2 = |dist|^2 < rmax^2. Auto pairs (A = B):
                /// every pair once. The grid cells are distributed over the MPI tasks (by estimated cost) and the threads
                //====================================================
                template <int NDIM, class F>
                void count_pairs(const CellList<NDIM> & A,
                                 const CellList<NDIM> & B,
                                 bool autopairs,
                                 double rmax,
                                 bool periodic,
                                 F && pair_function) {
                    const int ngrid = A.ngrid;
                    assert_mpi(B.ngrid == ngrid, "[count_pairs] The cell lists must have the same grid");
                    const double rmax2 = rmax * rmax;
                    size_t ncells = 1;
                    for (int d = 0; d < NDIM; d++)
                        ncells *= size_t(ngrid);

                    // The offsets to the neighbour cells (pruned by the smallest distance between the cells). With a
                    // periodic box that needs more than the whole grid each cell is visited once
                    const int m = int(std::ceil(rmax * ngrid));
                    const bool cover_all = periodic and 2 * m + 1 >= ngrid;
                    const int lo = cover_all ? -((ngrid - 1) / 2) : -m;
                    const int hi = cover_all ? lo + ngrid - 1 : m;
                    std::vector<std::array<int, NDIM>> offsets;
                    {
                        std::array<int, NDIM> o;
                        o.fill(lo);
                        while (true) {
                            double d2 = 0.0;
                            for (int d = 0; d < NDIM; d++) {
                                const double dd = std::max(std::abs(o[d]) - 1, 0) / double(ngrid);
                                d2 += dd * dd;
                            }
                            if (d2 < rmax2 or cover_all)
                                offsets.push_back(o);
                            int d = NDIM - 1;
                            while (d >= 0 and o[d] == hi) {
                                o[d] = lo;
                                d--;
                            }
                            if (d < 0)
                                break;
                            o[d]++;
                        }
                    }

                    // The cells of A (with particles) of this task: largest estimated cost first, each to the task
                    // with the least work so far (the same on all tasks)
                    std::vector<std::pair<double, size_t>> cost;
                    for (size_t c = 0; c < ncells and FML::NTasks > 1; c++) {
                        const double na = double(A.start[c + 1] - A.start[c]);
                        if (na > 0.0)
                            cost.push_back({na * (double(B.start[c + 1] - B.start[c]) + 1.0), c});
                    }
                    std::sort(cost.begin(), cost.end(), [](auto & a, auto & b) {
                        return a.first > b.first or (a.first == b.first and a.second < b.second);
                    });
                    std::vector<size_t> mycells;
                    if (FML::NTasks == 1) {
                        // In the order of the cells (better memory locality)
                        for (size_t c = 0; c < ncells; c++)
                            if (A.start[c + 1] > A.start[c])
                                mycells.push_back(c);
                    } else {
                        std::vector<double> load(FML::NTasks, 0.0);
                        for (auto & c : cost) {
                            const int task = int(std::min_element(load.begin(), load.end()) - load.begin());
                            load[task] += c.first;
                            if (task == FML::ThisTask)
                                mycells.push_back(c.second);
                        }
                    }

#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
                    for (size_t k = 0; k < mycells.size(); k++) {
#ifdef USE_OMP
                        const int thread = omp_get_thread_num();
#else
                        const int thread = 0;
#endif
                        const size_t c1 = mycells[k];
                        std::array<int, NDIM> i1;
                        {
                            size_t c = c1;
                            for (int d = NDIM - 1; d >= 0; d--) {
                                i1[d] = int(c % size_t(ngrid));
                                c /= size_t(ngrid);
                            }
                        }
                        for (auto & o : offsets) {
                            size_t c2 = 0;
                            bool inside = true;
                            // The periodic image of cell c2 next to c1: x_j + shift (without covering the whole grid
                            // this is the minimum image for all the pairs within rmax)
                            double shift[NDIM];
                            for (int d = 0; d < NDIM; d++) {
                                int i2 = i1[d] + o[d];
                                shift[d] = 0.0;
                                if (periodic) {
                                    if (i2 < 0) {
                                        i2 += ngrid;
                                        shift[d] = -1.0;
                                    }
                                    if (i2 >= ngrid) {
                                        i2 -= ngrid;
                                        shift[d] = 1.0;
                                    }
                                } else if (i2 < 0 or i2 >= ngrid) {
                                    inside = false;
                                    break;
                                }
                                c2 = c2 * ngrid + size_t(i2);
                            }
                            if (not inside or (autopairs and c2 < c1))
                                continue;
                            const size_t jbegin = B.start[c2], jend = B.start[c2 + 1];
                            if (jbegin == jend)
                                continue;
                            for (size_t i = A.start[c1]; i < A.start[c1 + 1]; i++) {
                                const double * xi = &A.pos[i * NDIM];
                                for (size_t j = (autopairs and c2 == c1) ? i + 1 : jbegin; j < jend; j++) {
                                    const double * xj = &B.pos[j * NDIM];
                                    double dist[NDIM];
                                    double r2 = 0.0;
                                    for (int d = 0; d < NDIM; d++) {
                                        double dx = xi[d] - (xj[d] + shift[d]);
                                        if (cover_all) {
                                            if (dx > 0.5)
                                                dx -= 1.0;
                                            else if (dx < -0.5)
                                                dx += 1.0;
                                        }
                                        dist[d] = dx;
                                        r2 += dx * dx;
                                    }
                                    if (r2 < rmax2)
                                        pair_function(thread, i, j, dist, r2);
                                }
                            }
                        }
                    }
                }

                //====================================================
                /// The (weighted) pair counts of the catalogues A (particles part1) and B (part2) in (r, mu) bins, and
                /// the extra quantities: nquantities = 1 + nextratobin numbers per bin, bin (ir, imu) at
                /// nquantities * (ir + imu * nrbins). Summed over threads and MPI tasks
                //====================================================
                template <int NDIM, typename T1, typename T2>
                DVector angular_pair_counts(const CellList<NDIM> & A,
                                            const T1 * part1,
                                            const CellList<NDIM> & B,
                                            const T2 * part2,
                                            bool autopairs,
                                            double rmin,
                                            double rmax,
                                            int nrbins,
                                            double mumin,
                                            double mumax,
                                            int nmubins,
                                            const DVector & observer_position,
                                            bool periodic_box,
                                            int nextratobin,
                                            const ExtraQuantitiesToBinFunction<T1, T2> & extra_quantities_to_bin) {
                    const int nthreads = FML::NThreads;
                    const double rmin2 = rmin * rmin;
                    const int nquantities = 1 + nextratobin;
                    const size_t nbinstotal = size_t(nquantities) * nrbins * nmubins;
                    std::vector<DVector> count_threads(nthreads, DVector(nbinstotal, 0.0));

                    const bool compute_mu = nmubins > 1 or mumax < 1.0 or mumin > -1.0;
                    // Auto pairs: the order of the two particles (the sign of mu) is arbitrary, so every pair is
                    // counted half at mu and half at -mu (and the extra quantities for (p1,p2) at mu and (p2,p1) at -mu)
                    const bool symmetrize_mu = compute_mu and autopairs;
                    const double r_scale = nrbins / (rmax - rmin);
                    const double mu_scale = nmubins / (mumax - mumin);
                    std::array<double, NDIM> observer{};
                    if (compute_mu)
                        for (int d = 0; d < NDIM; d++)
                            observer[d] = observer_position[d];

                    auto in_mu_range = [mumin, mumax](double mu) {
                        return mu >= mumin and (mu < mumax or (mu == mumax and mumax >= 1.0));
                    };

                    // half: half of the extra quantities (symmetrized auto pairs); swapped: for (p2, p1)
                    auto add_pair = [&](int thread, int ir, int imu, double weight, const double * dist, double r,
                                        double mu, size_t i, size_t j, bool half, bool swapped) {
                        double * bin = &count_threads[thread][size_t(nquantities) * (ir + size_t(imu) * nrbins)];
                        bin[0] += weight;
                        if (nquantities == 1)
                            return;
                        const T1 & p1 = part1[A.index[i]];
                        const T2 & p2 = part2[B.index[j]];
                        if (not half) {
                            extra_quantities_to_bin(dist, r, mu, p1, p2, bin + 1);
                            return;
                        }
                        std::vector<double> tmp(nquantities - 1, 0.0);
                        if constexpr (std::is_same_v<T1, T2>) {
                            if (swapped)
                                extra_quantities_to_bin(dist, r, mu, p2, p1, tmp.data());
                            else
                                extra_quantities_to_bin(dist, r, mu, p1, p2, tmp.data());
                        } else {
                            extra_quantities_to_bin(dist, r, mu, p1, p2, tmp.data());
                        }
                        for (int q = 0; q < nquantities - 1; q++)
                            bin[1 + q] += 0.5 * tmp[q];
                    };

                    count_pairs<NDIM>(A, B, autopairs, rmax, periodic_box, [&](int thread, size_t i, size_t j, const double * dist, double r2) {
                        if (r2 < rmin2 or r2 == 0.0)
                            return;
                        const double r = std::sqrt(r2);
                        const int ir = std::min(int((r - rmin) * r_scale), nrbins - 1);
                        const double w = A.weight[i] * B.weight[j];
                        if (not compute_mu) {
                            add_pair(thread, ir, 0, w, dist, r, 0.0, i, j, false, false);
                            return;
                        }
                        // The line of sight: from the observer to the midpoint of the pair
                        const double * x1 = &A.pos[i * NDIM];
                        const double * x2 = &B.pos[j * NDIM];
                        double los[NDIM], norm2 = 0.0;
                        for (int d = 0; d < NDIM; d++) {
                            double dx = 0.5 * (x1[d] + x2[d]) - observer[d];
                            if (periodic_box) {
                                // The observer is "at infinity" in some direction or inside the box: a single wrap
                                if (dx < -0.5)
                                    dx += 1.0;
                                if (dx > 0.5)
                                    dx -= 1.0;
                            }
                            los[d] = dx;
                            norm2 += dx * dx;
                        }
                        double dot = 0.0;
                        for (int d = 0; d < NDIM; d++)
                            dot += dist[d] * los[d];
                        const double mu = dot / (r * std::sqrt(norm2));
                        if (symmetrize_mu) {
                            for (int s = 0; s < 2; s++) {
                                const double smu = s == 0 ? mu : -mu;
                                if (not in_mu_range(smu))
                                    continue;
                                const int imu = std::min(int((smu - mumin) * mu_scale), nmubins - 1);
                                if (s == 0 or nquantities == 1) {
                                    add_pair(thread, ir, imu, 0.5 * w, dist, r, smu, i, j, true, false);
                                } else {
                                    double mdist[NDIM];
                                    for (int d = 0; d < NDIM; d++)
                                        mdist[d] = -dist[d];
                                    add_pair(thread, ir, imu, 0.5 * w, mdist, r, smu, i, j, true, true);
                                }
                            }
                            return;
                        }
                        if (not in_mu_range(mu))
                            return;
                        const int imu = std::min(int((mu - mumin) * mu_scale), nmubins - 1);
                        add_pair(thread, ir, imu, w, dist, r, mu, i, j, false, false);
                    });

                    DVector count(nbinstotal, 0.0);
                    for (auto & c : count_threads)
                        for (size_t k = 0; k < nbinstotal; k++)
                            count[k] += c[k];
                    FML::SumArrayOverTasks(count.data(), count.size());
                    return count;
                }

                /// The fraction of a shell (in NDIM dimensions) with mu in [mu0, mu1): uniform in mu in 3D, in the
                /// angle in 2D and the two directions mu = -1, +1 in 1D (mu = 1 in the bin that ends at 1)
                template <int NDIM>
                double mu_fraction_of_shell(double mu0, double mu1) {
                    mu0 = std::max(mu0, -1.0);
                    mu1 = std::min(mu1, 1.0);
                    if constexpr (NDIM == 3)
                        return (mu1 - mu0) / 2.0;
                    if constexpr (NDIM == 2)
                        return (std::acos(mu0) - std::acos(mu1)) / M_PI;
                    return 0.5 * ((mu0 <= -1.0 and -1.0 < mu1) ? 1.0 : 0.0) + 0.5 * ((mu0 <= 1.0 and mu1 >= 1.0) ? 1.0 : 0.0);
                }

                /// The r and mu bins (centers) and edges
                inline void make_bins(double rmin, double rmax, int nrbins, double mumin, double mumax, int nmubins,
                                      DVector & r_array, DVector & r_edge, DVector & mu_array, DVector & mu_edge) {
                    r_array = DVector(nrbins);
                    r_edge = DVector(nrbins + 1, rmax);
                    for (int i = 0; i < nrbins; i++) {
                        r_array[i] = rmin + (rmax - rmin) * (i + 0.5) / double(nrbins);
                        r_edge[i] = rmin + (rmax - rmin) * i / double(nrbins);
                    }
                    mu_array = DVector(nmubins);
                    mu_edge = DVector(nmubins + 1, mumax);
                    for (int i = 0; i < nmubins; i++) {
                        mu_array[i] = mumin + (mumax - mumin) * (i + 0.5) / double(nmubins);
                        mu_edge[i] = mumin + (mumax - mumin) * i / double(nmubins);
                    }
                }

                /// Pair counts (count from angular_pair_counts) in arrays [imu][ir] (and the extra quantities)
                inline void extract_counts(const DVector & count, int nrbins, int nmubins, int nextratobin, double norm,
                                           DVector2D & paircounts, DVector3D & extra) {
                    const int nquantities = 1 + nextratobin;
                    paircounts = DVector2D(nmubins, DVector(nrbins, 0.0));
                    if (nextratobin > 0)
                        extra = DVector3D(nmubins, DVector2D(nrbins, DVector(nextratobin, 0.0)));
                    for (int imu = 0; imu < nmubins; imu++)
                        for (int ir = 0; ir < nrbins; ir++) {
                            const size_t k = size_t(nquantities) * (ir + size_t(imu) * nrbins);
                            paircounts[imu][ir] = count[k] * norm;
                            for (int q = 0; q < nextratobin; q++)
                                extra[imu][ir][q] = count[k + 1 + q];
                        }
                }
            } // namespace detail

            /// The (weighted) auto (part2 = part1 or nullptr) or cross pair counts in (r, mu) bins of particles in
            /// [0,1)^NDIM (periodic or not) and the correlation function paircounts / (expected pairs) - 1. mu is the
            /// cosine of the angle between the pair separation (x1 - x2) and the line of sight from the observer to
            /// the midpoint of the pair (observer_position, e.g. at -effective_infinity along an axis for a fixed
            /// line of sight); auto pairs are counted half at mu and half at -mu. With nextratobin > 0 the function
            /// extra_quantities_to_bin adds nextratobin numbers for every pair (in its bin)
            template <typename T1, typename T2>
            void AngularCorrelationFunctionBox(const T1 * part1,
                                               size_t npart1,
                                               const T2 * part2,
                                               size_t npart2,
                                               double rmin,
                                               double rmax,
                                               int nrbins,
                                               double mumin,
                                               double mumax,
                                               int nmubins,
                                               DVector & r_array,
                                               DVector & mu_array,
                                               DVector2D & paircounts_array,
                                               bool normalize_paircounts,
                                               DVector2D & corr_func_array,
                                               DVector observer_position,
                                               bool periodic_box,
                                               bool verbose,
                                               int nextratobin = 0,
                                               DVector3D & extra_quantities_array = empty_3D_array,
                                               ExtraQuantitiesToBinFunction<T1, T2> extra_quantities_to_bin = {}) {

                // Sanity checks
                if (npart1 == 0)
                    return;
                assert_mpi(part1 != nullptr, "AngularCorrelationFunctionBox :: Got nullptr for part1\n");
                assert_mpi(rmax > rmin, "AngularCorrelationFunctionBox :: Error rmax < rmin\n");
                assert_mpi(mumax > mumin, "AngularCorrelationFunctionBox :: Error mumax < mumin\n");
                assert_mpi(nrbins > 0, "AngularCorrelationFunctionBox :: Error nrbins <= 0\n");
                assert_mpi(nmubins > 0, "AngularCorrelationFunctionBox :: Error nmubins <= 0\n");
                assert_mpi(nextratobin == 0 or extra_quantities_to_bin,
                           "AngularCorrelationFunctionBox :: nextratobin > 0 needs extra_quantities_to_bin\n");

                // The dimension at compile time (from objects: GetNDIM(T()) needs a particle that can be built at
                // compile time)
                T1 tmp_part1;
                T2 tmp_part2;
                constexpr int NDIM = FML::PARTICLE::GetNDIM(tmp_part1);
                static_assert(NDIM > 0 and NDIM <= 3, "AngularCorrelationFunctionBox :: NDIM must be 1, 2 or 3");
                static_assert(NDIM == FML::PARTICLE::GetNDIM(tmp_part2),
                              "AngularCorrelationFunctionBox :: particles must have the same NDIM");

                const bool compute_mu = nmubins > 1 or mumax < 1.0 or mumin > -1.0;
                if (compute_mu)
                    assert_mpi(observer_position.size() >= NDIM,
                               "AngularCorrelationFunctionBox :: Error observer position has not enough coordinates\n");

                // Only one catalogue: auto pairs
                if (part2 == nullptr) {
                    part2 = reinterpret_cast<const T2 *>(part1);
                    npart2 = npart1;
                }
                const bool autopairs = reinterpret_cast<const void *>(part1) == reinterpret_cast<const void *>(part2);
                if (npart2 == 0)
                    return;

                // The particles in cells (the same grid for both catalogues)
                const int ngrid = detail::choose_ngrid(NDIM, rmax, std::max(npart1, npart2));
                const auto A = detail::make_cell_list<NDIM>(part1, npart1, ngrid);
                detail::CellList<NDIM> Bcells;
                if (not autopairs)
                    Bcells = detail::make_cell_list<NDIM>(part2, npart2, ngrid);
                const auto & B = autopairs ? A : Bcells;
                if (verbose and FML::ThisTask == 0)
                    std::cout << "# Pair counting (" << (autopairs ? "auto" : "cross") << " pairs, " << npart1 << " x "
                              << npart2 << " particles) on a grid of " << ngrid << "^" << NDIM << " cells with "
                              << FML::NTasks << " task(s) and " << FML::NThreads << " thread(s)\n";

                const auto count = detail::angular_pair_counts<NDIM, T1, T2>(A, part1, B, part2, autopairs, rmin, rmax,
                                                                             nrbins, mumin, mumax, nmubins,
                                                                             observer_position, periodic_box,
                                                                             nextratobin, extra_quantities_to_bin);

                // The total (weighted) number of pairs
                const double numpairs = autopairs ? (A.sum_weights * A.sum_weights - A.sum_weights_squared) / 2.0 :
                                                    A.sum_weights * B.sum_weights;

                DVector r_edge, mu_edge;
                detail::make_bins(rmin, rmax, nrbins, mumin, mumax, nmubins, r_array, r_edge, mu_array, mu_edge);
                DVector2D raw;
                DVector3D extra;
                detail::extract_counts(count, nrbins, nmubins, nextratobin, 1.0, raw, extra);

                // The correlation function in a box of volume 1: pairs / (numpairs * volume of the (r, mu) bin) - 1
                paircounts_array = DVector2D(nmubins, DVector(nrbins, 0.0));
                corr_func_array = DVector2D(nmubins, DVector(nrbins, 0.0));
                for (int ir = 0; ir < nrbins; ir++) {
                    for (int imu = 0; imu < nmubins; imu++) {
                        double volumebin = std::pow(M_PI, NDIM / 2.0) / std::tgamma(NDIM / 2.0 + 1.0) *
                                           (std::pow(r_edge[ir + 1], NDIM) - std::pow(r_edge[ir], NDIM));
                        if (compute_mu)
                            volumebin *= detail::mu_fraction_of_shell<NDIM>(mu_edge[imu], mu_edge[imu + 1]);
                        paircounts_array[imu][ir] = raw[imu][ir] * (normalize_paircounts ? 1.0 / numpairs : 1.0);
                        corr_func_array[imu][ir] = raw[imu][ir] / (numpairs * volumebin) - 1.0;
                    }
                }
                if (nextratobin > 0)
                    extra_quantities_array = extra;
            }

            template <typename T1>
            void AngularCorrelationFunctionBox(const T1 * part1,
                                               size_t npart1,
                                               double rmin,
                                               double rmax,
                                               int nrbins,
                                               double mumin,
                                               double mumax,
                                               int nmubins,
                                               DVector & r_array,
                                               DVector & mu_array,
                                               DVector2D & paircounts_array,
                                               bool normalize_paircounts,
                                               DVector2D & corr_func_array,
                                               DVector observer_position,
                                               bool periodic_box,
                                               bool verbose,
                                               int nextratobin = 0,
                                               DVector3D & extra_quantities_array = empty_3D_array,
                                               ExtraQuantitiesToBinFunction<T1, T1> extra_quantities_to_bin = {}) {

                AngularCorrelationFunctionBox<T1, T1>(part1,
                                                      npart1,
                                                      part1,
                                                      npart1,
                                                      rmin,
                                                      rmax,
                                                      nrbins,
                                                      mumin,
                                                      mumax,
                                                      nmubins,
                                                      r_array,
                                                      mu_array,
                                                      paircounts_array,
                                                      normalize_paircounts,
                                                      corr_func_array,
                                                      observer_position,
                                                      periodic_box,
                                                      verbose,
                                                      nextratobin,
                                                      extra_quantities_array,
                                                      extra_quantities_to_bin);
            }

            /// General function for finding auto or cross pairs in a simulation box
            /// which can be periodic or not. If particles has weights this is accounted for.
            /// The method is easily modified to bin up any other quantity over the pair (e.g. relative velocity v12).
            /// All you need to do is to set nquantity = 1 + number of scalar quantities you want
            /// to bin up. And then add a line for each quantity to the binning_function which adds up the quantity for
            /// each pair we visit.
            template <typename T1, typename T2>
            void RadialCorrelationFunctionBox(const T1 * part1,
                                              size_t npart1,
                                              const T2 * part2,
                                              size_t npart2,
                                              double rmin,
                                              double rmax,
                                              int nrbins,
                                              DVector & r_array,
                                              DVector & paircounts_array,
                                              bool normalize_paircounts,
                                              DVector & corr_func_array,
                                              bool periodic_box,
                                              bool verbose,
                                              int nextratobin = 0,
                                              DVector2D & extra_quantities_array = empty_2D_array,
                                              ExtraQuantitiesToBinFunction<T1, T2> extra_quantities_to_bin = {}) {

                // These are irrelevant
                DVector observer_position;
                DVector mu_array;

                // We need one bin covering the entire domain
                double mumin = -1.0;
                double mumax = 1.0;
                int nmubins = 1;

                // Temp storage
                DVector2D _paircounts_array;
                DVector2D _corr_func_array;
                DVector3D _extra_quantities_array;

                AngularCorrelationFunctionBox<T1, T2>(part1,
                                                      npart1,
                                                      part2,
                                                      npart2,
                                                      rmin,
                                                      rmax,
                                                      nrbins,
                                                      mumin,
                                                      mumax,
                                                      nmubins,
                                                      r_array,
                                                      mu_array,
                                                      _paircounts_array,
                                                      normalize_paircounts,
                                                      _corr_func_array,
                                                      observer_position,
                                                      periodic_box,
                                                      verbose,
                                                      nextratobin,
                                                      _extra_quantities_array,
                                                      extra_quantities_to_bin);

                // Extract the data
                paircounts_array = _paircounts_array[0];
                corr_func_array = _corr_func_array[0];
                if (nextratobin > 0)
                    extra_quantities_array = _extra_quantities_array[0];
            }

            template <typename T1>
            void RadialCorrelationFunctionBox(const T1 * part1,
                                              size_t npart1,
                                              double rmin,
                                              double rmax,
                                              int nrbins,
                                              DVector & r_array,
                                              DVector & paircounts_array,
                                              bool normalize_paircounts,
                                              DVector & corr_func_array,
                                              bool periodic_box,
                                              bool verbose,
                                              int nextratobin = 0,
                                              DVector2D & extra_quantities_array = empty_2D_array,
                                              ExtraQuantitiesToBinFunction<T1, T1> extra_quantities_to_bin = {}) {

                RadialCorrelationFunctionBox<T1, T1>(part1,
                                                     npart1,
                                                     part1,
                                                     npart1,
                                                     rmin,
                                                     rmax,
                                                     nrbins,
                                                     r_array,
                                                     paircounts_array,
                                                     normalize_paircounts,
                                                     corr_func_array,
                                                     periodic_box,
                                                     verbose,
                                                     nextratobin,
                                                     extra_quantities_array,
                                                     extra_quantities_to_bin);
            }

            /// Auto (part2 = part1) or cross correlation function of survey data with randoms (rand2 = rand1 for the
            /// same randoms) in (r, mu) bins: the normalized pair counts D1D2, D1R2, R1D2 and R1R2 (each divided by the
            /// total weighted number of pairs) and the estimator (see CorrelationFunctionEstimator; LZ = LS =
            /// Landy & Szalay). Positions in [0,1)^NDIM (see e.g. Survey/GalaxiesToBox.h). Each catalogue is put on
            /// the grid once. With nextratobin > 0 the extra quantities are binned for all four pair types (the
            /// particle types must then be the same)
            template <typename T1, typename T2 = T1, typename R1 = T1, typename R2 = T2>
            void AngularCorrelationFunctionSurvey(const T1 * part1,
                                                  size_t npart1,
                                                  const R1 * rand1,
                                                  size_t nrand1,
                                                  const T2 * part2,
                                                  size_t npart2,
                                                  const R2 * rand2,
                                                  size_t nrand2,
                                                  double rmin,
                                                  double rmax,
                                                  int nrbins,
                                                  double mumin,
                                                  double mumax,
                                                  int nmubins,
                                                  DVector & r_array,
                                                  DVector & mu_array,
                                                  DVector2D & paircounts_D1D2_array,
                                                  DVector2D & paircounts_D1R2_array,
                                                  DVector2D & paircounts_R1D2_array,
                                                  DVector2D & paircounts_R1R2_array,
                                                  DVector2D & corr_func_array,
                                                  DVector observer_position,
                                                  bool periodic_box,
                                                  bool verbose,
                                                  int nextratobin = 0,
                                                  DVector3D & extra_quantities_D1D2_array = empty_3D_array,
                                                  DVector3D & extra_quantities_D1R2_array = empty_3D_array,
                                                  DVector3D & extra_quantities_R1D2_array = empty_3D_array,
                                                  DVector3D & extra_quantities_R1R2_array = empty_3D_array,
                                                  ExtraQuantitiesToBinFunction<T1, T2> extra_quantities_to_bin = {},
                                                  std::string estimator = "LZ") {

                T1 tmp_part1; // (GetNDIM(T()) needs a particle that can be built at compile time)
                constexpr int NDIM = FML::PARTICLE::GetNDIM(tmp_part1);
                static_assert(NDIM > 0 and NDIM <= 3, "AngularCorrelationFunctionSurvey :: NDIM must be 1, 2 or 3");
                assert_mpi(rmax > rmin and mumax > mumin and nrbins > 0 and nmubins > 0,
                           "AngularCorrelationFunctionSurvey :: invalid bins\n");
                assert_mpi(part1 != nullptr and rand1 != nullptr and npart1 > 0 and nrand1 > 0,
                           "AngularCorrelationFunctionSurvey :: no data or randoms\n");
                const bool compute_mu = nmubins > 1 or mumax < 1.0 or mumin > -1.0;
                if (compute_mu)
                    assert_mpi(observer_position.size() >= NDIM,
                               "AngularCorrelationFunctionSurvey :: observer position has not enough coordinates\n");
                if (part2 == nullptr) {
                    part2 = reinterpret_cast<const T2 *>(part1);
                    npart2 = npart1;
                }
                if (rand2 == nullptr) {
                    rand2 = reinterpret_cast<const R2 *>(rand1);
                    nrand2 = nrand1;
                }
                const bool cross_correlation = reinterpret_cast<const void *>(part1) != reinterpret_cast<const void *>(part2);
                const bool same_randoms = reinterpret_cast<const void *>(rand1) == reinterpret_cast<const void *>(rand2);

                // The extra quantities need the same types for all the pair types
                constexpr bool same_types = std::is_same_v<T1, T2> and std::is_same_v<T1, R1> and std::is_same_v<T1, R2>;
                assert_mpi(nextratobin == 0 or (same_types and extra_quantities_to_bin),
                           "AngularCorrelationFunctionSurvey :: extra quantities need the same particle type for data "
                           "and randoms and extra_quantities_to_bin\n");
                auto extra_for = [&](auto * a, auto * b) {
                    using A = std::remove_const_t<std::remove_pointer_t<decltype(a)>>;
                    using B = std::remove_const_t<std::remove_pointer_t<decltype(b)>>;
                    ExtraQuantitiesToBinFunction<A, B> f{};
                    if constexpr (std::is_same_v<A, T1> and std::is_same_v<B, T2>)
                        f = extra_quantities_to_bin;
                    return f;
                };

                // All the catalogues on the same grid (once)
                const size_t nmax = std::max(std::max(npart1, npart2), std::max(nrand1, nrand2));
                const int ngrid = detail::choose_ngrid(NDIM, rmax, nmax);
                const auto D1 = detail::make_cell_list<NDIM>(part1, npart1, ngrid);
                const auto Rl1 = detail::make_cell_list<NDIM>(rand1, nrand1, ngrid);
                detail::CellList<NDIM> D2cells, R2cells;
                if (cross_correlation)
                    D2cells = detail::make_cell_list<NDIM>(part2, npart2, ngrid);
                if (not same_randoms)
                    R2cells = detail::make_cell_list<NDIM>(rand2, nrand2, ngrid);
                const auto & D2 = cross_correlation ? D2cells : D1;
                const auto & Rl2 = same_randoms ? Rl1 : R2cells;
                if (verbose and FML::ThisTask == 0)
                    std::cout << "# Survey pair counts (" << (cross_correlation ? "cross" : "auto") << ") on a grid of "
                              << ngrid << "^" << NDIM << " cells\n";

                DVector r_edge, mu_edge;
                detail::make_bins(rmin, rmax, nrbins, mumin, mumax, nmubins, r_array, r_edge, mu_array, mu_edge);
                auto norm = [](const auto & a, const auto & b, bool autopairs) {
                    const double n = autopairs ? (a.sum_weights * a.sum_weights - a.sum_weights_squared) / 2.0 :
                                                 a.sum_weights * b.sum_weights;
                    return n > 0.0 ? 1.0 / n : 0.0;
                };
                auto counts = [&](const auto & A, const auto * pA, const auto & B, const auto * pB, bool autopairs,
                                  DVector2D & paircounts, DVector3D & extra) {
                    using TA = std::remove_const_t<std::remove_pointer_t<decltype(pA)>>;
                    using TB = std::remove_const_t<std::remove_pointer_t<decltype(pB)>>;
                    const auto count = detail::angular_pair_counts<NDIM, TA, TB>(
                        A, pA, B, pB, autopairs, rmin, rmax, nrbins, mumin, mumax, nmubins, observer_position,
                        periodic_box, nextratobin, extra_for(pA, pB));
                    detail::extract_counts(count, nrbins, nmubins, nextratobin, norm(A, B, autopairs), paircounts, extra);
                };

                counts(D1, part1, D2, part2, not cross_correlation, paircounts_D1D2_array, extra_quantities_D1D2_array);
                counts(D1, part1, Rl2, rand2, false, paircounts_D1R2_array, extra_quantities_D1R2_array);
                if (cross_correlation) {
                    counts(Rl1, rand1, D2, part2, false, paircounts_R1D2_array, extra_quantities_R1D2_array);
                } else {
                    paircounts_R1D2_array = paircounts_D1R2_array;
                    extra_quantities_R1D2_array = extra_quantities_D1R2_array;
                }
                counts(Rl1, rand1, Rl2, rand2, same_randoms, paircounts_R1R2_array, extra_quantities_R1R2_array);

                corr_func_array = paircounts_D1D2_array;
                for (size_t i = 0; i < paircounts_D1D2_array.size(); i++)
                    for (size_t j = 0; j < paircounts_D1D2_array[i].size(); j++)
                        corr_func_array[i][j] = CorrelationFunctionEstimator(paircounts_D1D2_array[i][j],
                                                                             paircounts_D1R2_array[i][j],
                                                                             paircounts_R1D2_array[i][j],
                                                                             paircounts_R1R2_array[i][j],
                                                                             estimator);
            }

            template <typename T1, typename T2 = T1, typename R1 = T1, typename R2 = T2>
            void RadialCorrelationFunctionSurvey(const T1 * part1,
                                                 size_t npart1,
                                                 const R1 * rand1,
                                                 size_t nrand1,
                                                 const T2 * part2,
                                                 size_t npart2,
                                                 const R2 * rand2,
                                                 size_t nrand2,
                                                 double rmin,
                                                 double rmax,
                                                 int nrbins,
                                                 DVector & r_array,
                                                 DVector & paircounts_D1D2_array,
                                                 DVector & paircounts_D1R2_array,
                                                 DVector & paircounts_R1D2_array,
                                                 DVector & paircounts_R1R2_array,
                                                 DVector & corr_func_array,
                                                 DVector observer_position,
                                                 bool periodic_box,
                                                 bool verbose,
                                                 int nextratobin = 0,
                                                 DVector2D & extra_quantities_D1D2_array = empty_2D_array,
                                                 DVector2D & extra_quantities_D1R2_array = empty_2D_array,
                                                 DVector2D & extra_quantities_R1D2_array = empty_2D_array,
                                                 DVector2D & extra_quantities_R1R2_array = empty_2D_array,
                                                 ExtraQuantitiesToBinFunction<T1, T2> extra_quantities_to_bin = {},
                                                 std::string estimator = "LZ") {

                DVector mu_array;
                const double mumin = -1.0;
                const double mumax = 1.0;
                const int nmubins = 1;

                DVector2D _paircounts_D1D2_array;
                DVector2D _paircounts_D1R2_array;
                DVector2D _paircounts_R1D2_array;
                DVector2D _paircounts_R1R2_array;
                DVector2D _corr_func_array;
                DVector3D _extra_quantities_D1D2_array;
                DVector3D _extra_quantities_D1R2_array;
                DVector3D _extra_quantities_R1D2_array;
                DVector3D _extra_quantities_R1R2_array;

                AngularCorrelationFunctionSurvey<T1, T2, R1, R2>(part1,
                                                                 npart1,
                                                                 rand1,
                                                                 nrand1,
                                                                 part2,
                                                                 npart2,
                                                                 rand2,
                                                                 nrand2,
                                                                 rmin,
                                                                 rmax,
                                                                 nrbins,
                                                                 mumin,
                                                                 mumax,
                                                                 nmubins,
                                                                 r_array,
                                                                 mu_array,
                                                                 _paircounts_D1D2_array,
                                                                 _paircounts_D1R2_array,
                                                                 _paircounts_R1D2_array,
                                                                 _paircounts_R1R2_array,
                                                                 _corr_func_array,
                                                                 observer_position,
                                                                 periodic_box,
                                                                 verbose,
                                                                 nextratobin,
                                                                 _extra_quantities_D1D2_array,
                                                                 _extra_quantities_D1R2_array,
                                                                 _extra_quantities_R1D2_array,
                                                                 _extra_quantities_R1R2_array,
                                                                 extra_quantities_to_bin,
                                                                 estimator);

                // Set output
                corr_func_array = _corr_func_array[0];
                paircounts_D1D2_array = _paircounts_D1D2_array[0];
                paircounts_D1R2_array = _paircounts_D1R2_array[0];
                paircounts_R1D2_array = _paircounts_R1D2_array[0];
                paircounts_R1R2_array = _paircounts_R1R2_array[0];
                if (nextratobin > 0) {
                    extra_quantities_D1D2_array = _extra_quantities_D1D2_array[0];
                    extra_quantities_D1R2_array = _extra_quantities_D1R2_array[0];
                    extra_quantities_R1D2_array = _extra_quantities_R1D2_array[0];
                    extra_quantities_R1R2_array = _extra_quantities_R1R2_array[0];
                }
            }

#ifdef USE_GSL
            /// Compute multipoles xi_ell(r) from a spline xi(mu,r)
            /// xiell(r) = (2ell+1) Int Pell(mu)xi(r,mu)dmu / Int dmu
            inline void FromAngularCorrelationToMultipoles(const DVector & r_array,
                                                    const Spline2D & xi_mu_r_spline,
                                                    std::vector<DVector> & multipoles,
                                                    double mumin = -1.0,
                                                    double mumax = 1.0) {

                // Integration range and number of points
                // We compute (2ell+1) Int Pell(mu) xi(r,mu) dmu / Int dmu
                const int nint = 1000;

                // Legendre polynomials
                // P_ell(mu) for ell = 0, ..., nell - 1
                auto P_ell_of_mu = [](double mu, int nell) -> DVector {
                    DVector P_ell_array(nell);
                    if (nell >= 1)
                        P_ell_array[0] = 1.0;
                    if (nell >= 2)
                        P_ell_array[1] = mu;
                    for (int ell = 2; ell < nell; ell++)
                        P_ell_array[ell] =
                            ((2 * ell - 1.) * mu * P_ell_array[ell - 1] - (ell - 1.) * P_ell_array[ell - 2]) / ell;
                    return P_ell_array;
                };

                // Initialize results
                multipoles = DVector2D(multipoles.size(), DVector(r_array.size(), 0.0));

                // Integrate up using the trapezoidal rule
                const double dmu = (mumax - mumin) / nint;
                for (size_t ir = 0; ir < r_array.size(); ir++) {
                    const double r = r_array[ir];
                    for (int i = 0; i < nint; i++) {
                        const double mu = mumin + (i + 0.5) * dmu;
                        const double xirmu = xi_mu_r_spline(mu, r);
                        const auto P_ell_of_mu_array = P_ell_of_mu(mu, multipoles.size());
                        for (size_t ell = 0; ell < multipoles.size(); ell++) {
                            multipoles[ell][ir] +=
                                (2 * ell + 1) * dmu / (mumax - mumin) * xirmu * P_ell_of_mu_array[ell];
                        }
                    }
                }
            }
#endif

        } // namespace PAIRCOUNTS
    }     // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
