#ifndef PAIRCOUNT_HEADER
#define PAIRCOUNT_HEADER

#include <cassert>
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
        /// and each task is then reponsible for a certain part of the pairs we want to count.
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
        /// For the availiable estimators for survey correlation functions see the method
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
            DVector3D empty_3D_array{};
            DVector2D empty_2D_array{};

            /// A large value that we can use that ensures that if the observer is at
            /// say (0,0,-effective_infinity) then we have a fixed line of sight
            /// direction (z-axis is this example).
            double effective_infinity = 1e10;

            /// For binning particles to cells. The minimum number of cells to reach the maximum distance
            int ncells_to_rmax = 8;
            /// For binning particles to cells. The minimum grid-size.
            int ngrid_min_size = 10;

            /// This is the general algorithm for computing pair counts.
            /// It finds all pairs within a separation rmax and calls the binning_function
            /// for all pairs from which you can take care of everything.
            /// NB: the binning_function is called
            template <class T, class U>
            void GeneralPairCounter(FML::PARTICLE::ParticlesInBoxes<T> & grid,
                                    FML::PARTICLE::ParticlesInBoxes<U> & grid2,
                                    std::function<void(int, double *, T &, U &)> & binning_function,
                                    bool auto_pair_binning,
                                    double rmax,
                                    bool periodic_box,
                                    bool verbose = false);

            /// Estimates the correlation function from the paircounts. Most common estimators are included.
            /// NB: assumes normalized paircounts here, e.g. pairs / total_number_of_pairs.
            double CorrelationFunctionEstimator(double D1D2,
                                                double D1R2,
                                                double R1D2,
                                                double R1R2,
                                                const std::string & estimator) {

                // The estimators below that were proposed for auto pair-counts have been
                // symmetrized to also give a result for cross pair-counts
                // See DOI: 10.1051/0004-6361/201220790 for a summary
                // and https://arxiv.org/pdf/astro-ph/9912088.pdf for a comparison
                if (estimator == "LZ") {
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
                        "[CorrelationFunctionEstimator] Unknown estimator" + estimator +
                        ". Options: LZ, HAM, HAM_no_R1, HAM_no_R2, HEW, HEW_no_R1, HEW_no_R2, PH, DP, "
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
                                    std::function<void(int, double *, T &, U &)> & binning_function,
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
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
                verbose = verbose and mpi_rank == 0;
#endif

                // Fetch ndim
                constexpr int ndim = FML::PARTICLE::GetNDIM(T());
                assert(ndim == FML::PARTICLE::GetNDIM(U()));

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

                // How many cells in each direction we must search in the second grid
                int delta_ncells2 = (int)(ceil(rmax * ngrid2)) + 2;
                if (delta_ncells2 > ngrid2 / 2)
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
                                int ix_grid2 = (int)(ix0 * (double)ngrid2 / (double)ngrid1);
                                int iy_grid2 = (int)(iy0 * (double)ngrid2 / (double)ngrid1);
                                int iz_grid2 = (int)(iz0 * (double)ngrid2 / (double)ngrid1);

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

                                // Loop over neightbor cells
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

                // Get number of threads
                const int nthreads = FML::NThreads;

                // Squared rmin and rmax
                const double rmin2 = rmin * rmin;
                const double rmax2 = rmax * rmax;

                // Sanity checks
                if (npart1 == 0 or npart2 == 0)
                    return;
                assert_mpi(part1 != nullptr, "AngularCorrelationFunctionBox :: Got nullptr for part1\n");
                assert_mpi(rmax > rmin, "AngularCorrelationFunctionBox :: Error rmax < rmin\n");
                assert_mpi(mumax > mumin, "AngularCorrelationFunctionBox :: Error mumax < mumin\n");
                assert_mpi(nrbins > 0, "AngularCorrelationFunctionBox :: Error nrbins <= 0\n");
                assert_mpi(nmubins > 0, "AngularCorrelationFunctionBox :: Error nmubins <= 0\n");

                // Fetch the dimension at compiletime
                constexpr int NDIM = FML::PARTICLE::GetNDIM(T1());
                assert_mpi(NDIM > 0 and NDIM <= 3, "AngularCorrelationFunctionBox :: Error NDIM must be 1, 2 or 3\n");
                assert_mpi(NDIM == FML::PARTICLE::GetNDIM(T2()),
                           "AngularCorrelationFunctionBox :: Error particles must have same NDIM as method\n");

                // Check observer position
                if (nmubins > 1 or (mumin > -1.0 or mumax < 1.0)) {
                    assert_mpi(observer_position.size() >= NDIM,
                               "AngularCorrelationFunctionBox :: Error observer position has not enough coordinates\n");
                }

                // If only one catalogue is provided then its auto-pair counting
                // We set the second catalogue equal to the first as this will
                // set cross_pair_counting = false below
                if (part2 == nullptr) {
                    if (npart2 == 0)
                        return;
                    part2 = part1;
                    npart2 = npart1;
                }

                // If the particle pointers are the same then we are doing an auto binning
                const bool cross_pair_counting = (part1 != part2);

                // Total number of quantities to bin up
                // As fiducial we only bin up the paircounts themselves
                const int nquantities = 1 + nextratobin;
                const int nbinstotal = nquantities * nrbins * nmubins;

                // Storage for binning
                DVector2D count_threads(nthreads, DVector(nbinstotal, 0.0));

                //===============================================================
                // How to do the binning. The fidual choice is to just
                //===============================================================
                // Defines the r-edges of all bins
                const auto r_binedge_of_index = [rmin, rmax, nrbins](int i) {
                    return rmin + (rmax - rmin) * i / double(nrbins);
                };
                // Defines the r-bin centers of all bins
                const auto r_bincenter_of_index = [rmin, rmax, nrbins](int i) {
                    return rmin + (rmax - rmin) * (i + 0.5) / double(nrbins);
                };
                // How to compute the r-array index ir from a given r
                const auto ir_index_of_r = [rmin, rmax, nrbins](double r) {
                    return int((r - rmin) / (rmax - rmin) * nrbins);
                };
                // Defines the mu-bin centers of all bins
                auto mu_binedge_of_index = [mumin, mumax, nmubins](int i) {
                    return mumin + (mumax - mumin) * i / double(nmubins);
                };
                // Defines the mu-bin centers of all bins
                auto mu_bincenter_of_index = [mumin, mumax, nmubins](int i) {
                    return mumin + (mumax - mumin) * (i + 0.5) / double(nmubins);
                };
                // How to compute the mu-array index imu from a given mu
                auto imu_of_mu = [mumin, mumax, nmubins](double mu) {
                    return int((mu - mumin) / (mumax - mumin) * nmubins);
                };
                //===============================================================

                // Make the r array
                r_array = DVector(nrbins, 0.0);
                DVector r_edge_array(nrbins + 1, rmax);
                for (int j = 0; j < nrbins; j++) {
                    r_array[j] = r_bincenter_of_index(j);
                    r_edge_array[j] = r_binedge_of_index(j);
                }

                // Make the mu array
                mu_array = DVector(nmubins, 0.0);
                DVector mu_edge_array(nmubins + 1, mumax);
                for (int j = 0; j < nmubins; j++) {
                    mu_array[j] = mu_bincenter_of_index(j);
                    mu_edge_array[j] = mu_binedge_of_index(j);
                }

                // Binning function
                std::function<void(int, double *, T1 &, T2 &)> binning_function =
                    [&](int thread_id, double * dist, T1 & p1, T2 & p2) {
                        // Weight
                        double weight1 = 1.0, weight2 = 1.0;
                        if constexpr (FML::PARTICLE::has_get_weight<T1>())
                            weight1 = FML::PARTICLE::GetWeight(p1);
                        if constexpr (FML::PARTICLE::has_get_weight<T2>())
                            weight2 = FML::PARTICLE::GetWeight(p2);

                        // Compute squared distance between pairs
                        double r2 = dist[0] * dist[0];
                        if constexpr (NDIM >= 2)
                            r2 += dist[1] * dist[1];
                        if constexpr (NDIM >= 3)
                            r2 += dist[2] * dist[2];

                        // If out of range return
                        if (r2 < rmin2)
                            return;
                        if (r2 >= rmax2)
                            return;
                        if (r2 == 0.0)
                            return;

                        // Compute bin index
                        const double r = std::sqrt(r2);
                        const int ir = ir_index_of_r(r);

                        // Skip evaluating mu if we don't need to
                        double mu{};
                        int imu = 0;
                        if (nmubins > 1 or mumax < 1.0 or mumin > -1.0) {
                            std::array<double, NDIM> los_direction_local{};

                            // Compute the los direction as it depends on the position of the particle
                            auto pos1 = FML::PARTICLE::GetPos(p1);
                            auto pos2 = FML::PARTICLE::GetPos(p2);
                            double norm2 = 0.0;
                            for (int i = 0; i < NDIM; i++) {
                                double dx = (pos1[i] + pos2[i]) / 2.0 - observer_position[i];

                                if (periodic_box) {
                                    // In this case ther the observer is "at infinity"
                                    // in some direction or the observer is assumed to be inside the box
                                    // We therefore do a single wrap. Does not change anything for the former case
                                    if (dx < -0.5)
                                        dx += 1.0;
                                    if (dx > 0.5)
                                        dx -= 1.0;
                                }

                                los_direction_local[i] = dx;
                                norm2 += dx * dx;
                            }
                            norm2 = std::sqrt(norm2);
                            for (int i = 0; i < NDIM; i++)
                                los_direction_local[i] /= norm2;

                            // Compute mu
                            double dotproduct = dist[0] * los_direction_local[0];
                            if constexpr (NDIM >= 2)
                                dotproduct += dist[1] * los_direction_local[1];
                            if constexpr (NDIM >= 3)
                                dotproduct += dist[2] * los_direction_local[2];
                            mu = dotproduct / r;
                            if (mu < mumin or mu >= mumax)
                                return;
                            imu = imu_of_mu(mu);
                        }

                        // Add to bin
                        count_threads[thread_id][nquantities * (ir + imu * nrbins) + 0] += weight1 * weight2;

                        if (nquantities > 1) {
                            // Add up other stuff:
                            // count_threads[thread_id][nquantities * ir + 1              ] += ...;
                            // count_threads[thread_id][nquantities * ir + 2              ] += ...;
                            // ...
                            // count_threads[thread_id][nquantities * ir + (nquantities-1)] += ...;
                            // This is done in the provided function
                            double * storage = &count_threads[thread_id][nquantities * (ir + imu * nrbins) + 1];
                            extra_quantities_to_bin(dist, r, mu, p1, p2, storage);
                        }
                    };

                // Add particles to a grid
                const int ngrid1 =
                    std::max(ngrid_min_size,
                             std::min(int(ncells_to_rmax / rmax), int(std::pow(npart1 / 2.0, 1. / double(NDIM)))));
                FML::PARTICLE::ParticlesInBoxes<T1> grid1;
                FML::PARTICLE::ParticlesInBoxes<T2> grid2;
                grid1.create(part1, npart1, ngrid1);
                if (verbose)
                    grid1.info();

                // Compute sum of weights
                double sum1_weights = npart1;
                double sum1_weights_squared = npart1;
                if constexpr (FML::PARTICLE::has_get_weight<T1>()) {
                    sum1_weights = 0.0;
                    sum1_weights_squared = 0.0;
                    for (size_t i = 0; i < npart1; i++) {
                        double w = FML::PARTICLE::GetWeight(part1[i]);
                        sum1_weights += w;
                        sum1_weights_squared += w * w;
                    }
                }

                double sum2_weights{};
                if (cross_pair_counting) {
                    int ngrid2 =
                        std::max(ngrid_min_size,
                                 std::min(int(ncells_to_rmax / rmax), int(std::pow(npart2 / 2.0, 1. / double(NDIM)))));

                    // Add particles to a grid
                    grid2.create(part2, npart2, ngrid2);
                    if (verbose)
                        grid2.info();

                    // Compute sum of weights
                    sum2_weights = npart2;
                    if constexpr (FML::PARTICLE::has_get_weight<T2>()) {
                        sum2_weights = 0.0;
                        for (size_t i = 0; i < npart2; i++)
                            sum2_weights += FML::PARTICLE::GetWeight(part2[i]);
                    }
                }

                // Do the pair counts
                double numpairs{};
                if (cross_pair_counting) {
                    GeneralPairCounter<T1, T2>(
                        grid1, grid2, binning_function, not cross_pair_counting, rmax, periodic_box, verbose);
                    numpairs = sum1_weights * sum2_weights;
                } else {
                    GeneralPairCounter<T1, T2>(
                        grid1, grid1, binning_function, not cross_pair_counting, rmax, periodic_box, verbose);
                    numpairs = (sum1_weights * sum1_weights - sum1_weights_squared) / 2.0;
                }

                // Sum up over threads
                DVector count(nbinstotal, 0.0);
                for (int j = 0; j < nbinstotal; j++)
                    for (int i = 0; i < nthreads; i++)
                        count[j] += count_threads[i][j];

                // Sum over MPI tasks
                FML::SumArrayOverTasks(count.data(), count.size());

                // Normalize to get the correlation function
                paircounts_array = DVector2D(nmubins, DVector(nrbins, 0.0));
                corr_func_array = DVector2D(nmubins, DVector(nrbins, 0.0));
                for (int ir = 0; ir < nrbins; ir++) {
                    for (int imu = 0; imu < nmubins; imu++) {
                        const int index_in_count = (nquantities * (ir + imu * nrbins) + 0);
                        auto paircounts = count[index_in_count];
                        double volumebin =
                            4.0 * M_PI / 3.0 * (std::pow(r_edge_array[ir + 1], 3) - std::pow(r_edge_array[ir], 3));
                        volumebin *= (mu_edge_array[imu + 1] - mu_edge_array[imu]) / 2.0;
                        paircounts_array[imu][ir] = paircounts * (normalize_paircounts ? 1.0 / numpairs : 1.0);
                        corr_func_array[imu][ir] = paircounts / (numpairs * volumebin) - 1.0;
                    }
                }

                // If binning up other quantities we extract them in an array which can be returned if needed
                if (nquantities > 1) {
                    extra_quantities_array = DVector3D(nmubins, DVector2D(nrbins, DVector(nquantities, 0.0)));
                    if (verbose) {
                        std::cout << "\n#=====================================================\n";
                        std::cout << "# Extra quantities we have binned up:\n";
                        std::cout << "#=====================================================\n";
                    }
                    for (int ir = 0; ir < nrbins; ir++) {
                        for (int imu = 0; imu < nmubins; imu++) {
                            const int ir_imu_index = ir + nrbins * imu;
                            if (verbose) {
                                std::cout << "# r: " << std::setw(15) << r_array[ir] << " ";
                                if (nmubins > 1)
                                    std::cout << " mu: " << std::setw(15) << mu_array[imu] << " ";
                                std::cout << "Paircount: " << std::setw(15) << paircounts_array[imu][ir] << " ";
                                std::cout << "SumOverPairs => ";
                            }
                            for (int i = 1; i < nquantities; i++) {
                                const int index_in_count = (nquantities * ir_imu_index + i);
                                extra_quantities_array[imu][ir][i - 1] = count[index_in_count];
                                if (verbose)
                                    std::cout << "Quantity[" << i << "]: " << std::setw(15)
                                              << extra_quantities_array[imu][ir][i - 1] << " ";
                            }
                            if (verbose)
                                std::cout << "\n";
                        }
                    }
                    if (verbose)
                        std::cout << "#=====================================================\n";
                }
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

                // We need one bin convering the entire domain
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
                                                  ExtraQuantitiesToBinFunction<T1, T2> extra_quantities_to_bin = {}) {

                const bool normalize_paircounts = true;
                const bool cross_correlation = (part1 != part2);

                // D1D2 paircounts
                DVector2D corr_func_D1D2_array;
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
                                                      paircounts_D1D2_array,
                                                      normalize_paircounts,
                                                      corr_func_D1D2_array,
                                                      observer_position,
                                                      periodic_box,
                                                      verbose,
                                                      nextratobin,
                                                      extra_quantities_D1D2_array,
                                                      extra_quantities_to_bin);

                // D1R2 paircounts
                DVector2D corr_func_D1R2_array;
                AngularCorrelationFunctionBox<T1, R1>(part1,
                                                      npart1,
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
                                                      paircounts_D1R2_array,
                                                      normalize_paircounts,
                                                      corr_func_D1R2_array,
                                                      observer_position,
                                                      periodic_box,
                                                      verbose,
                                                      nextratobin,
                                                      extra_quantities_D1R2_array,
                                                      extra_quantities_to_bin);

                // R1D2 paircounts
                DVector2D corr_func_R1D2_array;
                if (not cross_correlation) {
                    corr_func_R1D2_array = corr_func_D1R2_array;
                    paircounts_R1D2_array = paircounts_D1R2_array;
                    extra_quantities_R1D2_array = extra_quantities_D1R2_array;
                } else {
                    AngularCorrelationFunctionBox<T2, R1>(part2,
                                                          npart2,
                                                          rand1,
                                                          nrand1,
                                                          rmin,
                                                          rmax,
                                                          nrbins,
                                                          mumin,
                                                          mumax,
                                                          nmubins,
                                                          r_array,
                                                          mu_array,
                                                          paircounts_R1D2_array,
                                                          normalize_paircounts,
                                                          corr_func_R1D2_array,
                                                          observer_position,
                                                          periodic_box,
                                                          verbose,
                                                          nextratobin,
                                                          extra_quantities_R1D2_array,
                                                          extra_quantities_to_bin);
                }

                // R1R2 paircounts
                DVector2D corr_func_R1R2_array;
                AngularCorrelationFunctionBox<R1, R2>(rand1,
                                                      nrand1,
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
                                                      paircounts_R1R2_array,
                                                      normalize_paircounts,
                                                      corr_func_R1R2_array,
                                                      observer_position,
                                                      periodic_box,
                                                      verbose,
                                                      nextratobin,
                                                      extra_quantities_R1R2_array,
                                                      extra_quantities_to_bin);

                // We now have all the paircounts
                // Process this into a correlation function
                std::string estimator = "LZ";
                corr_func_array = paircounts_D1D2_array;
                for (size_t i = 0; i < paircounts_D1D2_array.size(); i++) {
                    for (size_t j = 0; j < paircounts_D1D2_array[i].size(); j++) {
                        double D1D2 = paircounts_D1D2_array[i][j];
                        double D1R2 = paircounts_D1R2_array[i][j];
                        double R1D2 = paircounts_R1D2_array[i][j];
                        double R1R2 = paircounts_R1R2_array[i][j];

                        corr_func_array[i][j] = CorrelationFunctionEstimator(D1D2, D1R2, R1D2, R1R2, estimator);
                    }
                }
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
                                                 ExtraQuantitiesToBinFunction<T1, T2> extra_quantities_to_bin = {}) {

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
                                                                 extra_quantities_to_bin);

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
            void FromAngularCorrelationToMultipoles(const DVector & r_array,
                                                    const Spline2D & xi_mu_r_spline,
                                                    std::vector<DVector> & multipoles,
                                                    double mumin = -1.0,
                                                    double mumax = 1.0) {

                // Integration range and number of points
                // We compute (2ell+1) Int Pell(mu) xi(r,mu) dmu / Int dmu
                const int nint = 1000;

                // Legendre polynomials
                auto P_ell_of_mu = [](double mu, int ell_max) -> DVector {
                    DVector P_ell_array(ell_max);
                    if (ell_max >= 0)
                        P_ell_array[0] = 1.0;
                    if (ell_max >= 1)
                        P_ell_array[1] = mu;
                    for (int ell = 2; ell < ell_max; ell++)
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
