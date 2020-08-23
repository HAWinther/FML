#ifndef PAIRCOUNT_HEADER
#define PAIRCOUNT_HEADER

#include <cassert>
#include <cmath>
#include <functional>
#include <iomanip>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

//==============================================================
// This is needed to speed up the calculation
// The particles are binned to a grid and we do the correlation
// by going over cells x cells
//==============================================================
#include <FML/ParticlesInBoxes/ParticlesInBoxes.h>

//===========================================================================
//
// General paircount methods in 1D, 2D or 3D. What exact properties you would like to bin
// is defined in the binning function provided
//
// Works with OpenMP, MPI or both. For MPI its assumed that all tasks has all the particles
// and each task is then reponsible for a certain part of the pairs we want to count
//
// Particles are assumed to reside in [0,1)
// rmax is the maximum pair separation (in [0,1)) that we are interested in
// Periodic means the particle positions wraps around
//
// User is completely responsible for taking care of what to bin up, see
// Auto/Cross CorrelationFunction for example of how to do this
//
// Made for both auto and cross pair counts. For cross pair counts we allow
// for different particle types
//
// Particles must have the methods:
// auto *get_pos()
// int get_ndim()
// double get_weight() (just return 1.0 if no weight)
//
// Cells must have the methods:
// auto& get_cells()
// int get_ngrid()
// size_t get_npart()
// int get_np()
// vecto<Particle> &get_part()
//
// Fiducial results is the number of pairs in each bin
//
//===========================================================================

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        /// @brief The result struct for auto pair counting
        struct AutoPairCountData {
            std::vector<double> r;
            std::vector<double> r_edge;
            std::vector<double> paircount;
            double sum_weights{0.0};
            double sum_weights_squared{0.0};
            double norm{0.0};
        };

        /// @brief The result struct for cross pair counting
        struct CrossPairCountData {
            std::vector<double> r;
            std::vector<double> r_edge;
            std::vector<double> paircount;
            double sum_weights{0.0};
            double sum_weights_squared{0.0};
            double sum2_weights{0.0};
            double sum2_weights_squared{0.0};
            double norm{0.0};
        };

        /// Do paircount of a set of particles. The source of this shows how to use the other methods.
        ///
        /// @tparam T The particle class
        ///
        /// @param[in] particles List of particles
        /// @param[in] nbins The number of bins
        /// @param[in] rmax Maximum radius we want the paircount up to
        /// @param[in] periodic Periodic box?
        /// @param[in] verbose Show info while running
        ///
        /// \return AutoPairCountData containing the result of the binning.
        ///
        template <class T>
        AutoPairCountData
        AutoPairCount(std::vector<T> & particles, int nbins, double rmax, bool periodic, bool verbose);

        /// Do cross paircount of a set of particles. The source of this shows how to use the other methods.
        ///
        /// @tparam T The particle class for the first set of particles
        /// @tparam U The particle class for the second set of particles
        ///
        /// @param[in] particles1 List of particles
        /// @param[in] particles2 List of particles
        /// @param[in] nbins The number of bins
        /// @param[in] rmax Maximum radius we want the paircount up to
        /// @param[in] periodic Periodic box?
        /// @param[in] verbose Show info while running
        ///
        /// \return CrossPairCountData containing the result of the binning.
        ///
        template <class T, class U>
        CrossPairCountData CrossPairCount(std::vector<T> & particles1,
                                          std::vector<U> & particles2,
                                          int nbins,
                                          double rmax,
                                          bool periodic,
                                          bool verbose);

        /// The general algorithm. Called by the other methods.
        /// This is the method that does the hard work.
        /// bin is the binning function telling us what to do
        /// with every pair
        template <class T>
        void AutoPairCountGridMethod(FML::PARTICLE::ParticlesInBoxes<T> & grid,
                                     std::function<void(int, double *, T &, T &)> & bin,
                                     double rmax,
                                     bool periodic,
                                     bool verbose);

        /// The general algorithm. Called by the other methods.
        /// This is the method that does the hard work.
        /// bin is the binning function telling us what to do
        /// with every pair
        template <class T, class U>
        void CrossPairCountGridMethod(FML::PARTICLE::ParticlesInBoxes<T> & grid,
                                      FML::PARTICLE::ParticlesInBoxes<U> & grid2,
                                      std::function<void(int, double *, T &, U &)> & bin,
                                      double rmax,
                                      bool periodic,
                                      bool verbose);

        /// Some estimators for correlation functions in surveys
        /// The paircounts is the number of pairs divided by the the total number of pairs
        /// If one of the paircounts are not needed for an estimator and you dont have it just
        /// pass nullptr in place of it
        std::vector<double>
        AutoCorrelationEstimator(double * DD, double * DR, double * RR, int nbins, std::string estimator);
        std::vector<double> CrossCorrelationEstimator(double * D1D2,
                                                      double * D1R2,
                                                      double * R1D2,
                                                      double * R1R2,
                                                      int nbins,
                                                      std::string estimator);

        //===========================================================================

        std::vector<double>
        AutoCorrelationEstimator(double * DD, double * DR, double * RR, int nbins, std::string estimator) {
            std::vector<double> corr(nbins);

            // See DOI: 10.1051/0004-6361/201220790 for a summary
            // and https://arxiv.org/pdf/astro-ph/9912088.pdf for a comparison
            if (estimator == "LZ") {
                // Landy & Szalay 1993
                assert(DD != nullptr and DR != nullptr and RR != nullptr);
                for (int i = 0; i < nbins; i++) {
                    corr[i] = (DD[i] + RR[i] - 2 * DR[i]) / RR[i];
                }
            } else if (estimator == "HAM") {
                // Hamilton 1993
                assert(DD != nullptr and DR != nullptr and RR != nullptr);
                for (int i = 0; i < nbins; i++) {
                    corr[i] = DD[i] * RR[i] / (DR[i] * DR[i]) - 1.0;
                }
            } else if (estimator == "HEW") {
                // Hewett 1982
                assert(DD != nullptr and DR != nullptr and RR != nullptr);
                for (int i = 0; i < nbins; i++) {
                    corr[i] = (DD[i] - DR[i]) / RR[i];
                }
            } else if (estimator == "DP") {
                // Davis & Peebles 1983
                assert(DD != nullptr and DR != nullptr);
                for (int i = 0; i < nbins; i++) {
                    corr[i] = DD[i] / DR[i] - 1.0;
                }
            } else if (estimator == "PH") {
                // Peebles & Hauser 1974
                assert(DD != nullptr and RR != nullptr);
                for (int i = 0; i < nbins; i++) {
                    corr[i] = DD[i] / RR[i] - 1.0;
                }
            } else {
                throw std::runtime_error("[AutoCorrelationEstimator] Unknown estimator" + estimator +
                                         ". Options: LZ, DP, HAM, HEW, PH\n");
            }
            return corr;
        }

        /// Some estimators for cross correlation functions in surveys (see AutoCorrelationEstimator for more info).
        std::vector<double> CrossCorrelationEstimator(double * D1D2,
                                                      double * D1R2,
                                                      double * R1D2,
                                                      double * R1R2,
                                                      int nbins,
                                                      std::string estimator) {
            std::vector<double> corr(nbins);

            // See Blake et al 2006 https://doi.org/10.1111/j.1365-2966.2006.10158.x
            if (estimator == "LZ") {
                // LZ modified to cross
                assert(D1D2 != nullptr and D1R2 != nullptr and R1D2 != nullptr and R1R2 != nullptr);
                for (int j = 0; j < nbins; j++) {
                    corr[j] = (D1D2[j] + R1R2[j] - D1R2[j] - R1D2[j]) / R1R2[j];
                }
            } else if (estimator == "HAM") {
                // Hamilton modified to cross
                assert(D1D2 != nullptr and D1R2 != nullptr and R1D2 != nullptr and R1R2 != nullptr);
                for (int j = 0; j < nbins; j++) {
                    corr[j] = D1D2[j] * R1R2[j] / (D1R2[j] * R1D2[j]) - 1.0;
                }
            } else if (estimator == "DP_NO_R1") {
                // David Peebles version 1 (no random for cat 1)
                assert(D1D2 != nullptr and D1R2 != nullptr);
                for (int j = 0; j < nbins; j++) {
                    corr[j] = D1D2[j] / D1R2[j] - 1.0;
                }
            } else if (estimator == "DP_NO_R2") {
                // David Peebles version 1 (no random for cat 2)
                assert(D1D2 != nullptr and R1D2 != nullptr);
                for (int j = 0; j < nbins; j++) {
                    corr[j] = D1D2[j] / R1D2[j] - 1.0;
                }
            } else {
                throw std::runtime_error(
                    "[CrossCorrelationEstimator] Unknown estimator. Options: LZ, HAM, DP_NO_R1, DP_NO_R2\n");
            }
            return corr;
        }

        /// Compute the correlation function for particles in a periodic box
        template <class T>
        void CorrelationFunctionSimulationBox(std::vector<T> & particles, int nbins, double rmax, bool verbose) {

            // Fetch the number of dimensions we are working in
            T ptemp;
            const int ndim = ptemp.get_ndim();

            // Compute pair counts
            auto paircountdata = AutoPairCount(particles, nbins, rmax, true, verbose);
            auto & r = paircountdata.r;
            auto & r_edge = paircountdata.r_edge;
            auto & paircount = paircountdata.paircount;
            auto norm = paircountdata.norm * 2;

            // Convert to correlation function
            auto xi = r;
            for (int j = 0; j < nbins; j++) {
                double r0 = r_edge[j];
                double r1 = r_edge[j + 1];
                double vol;
                if (ndim == 1)
                    vol = 2 * (r1 - r0);
                if (ndim == 2)
                    vol = M_PI * (r1 * r1 - r0 * r0);
                if (ndim == 3)
                    vol = 4.0 * M_PI / 3.0 * (r1 * r1 * r1 - r0 * r0 * r0);
                xi[j] = paircount[j] / (norm * vol) - 1.0;
                std::cout << r[j] << " " << xi[j] << "\n";
            }
        }

        /// Compute the correlation function for galaxies from a survey.
        /// Using the LZ estimator as standard
        template <class T>
        void CorrelationFunctionSurvey(std::vector<T> & particles_xyz,
                                       std::vector<T> & randoms_xyz,
                                       int nbins,
                                       double rmax,
                                       bool verbose) {

            const bool periodic = false;
            const std::string estimator = "LZ";

            // Compute pair counts
            auto paircountdata = AutoPairCount(particles_xyz, nbins, rmax, periodic, verbose);
            auto & r = paircountdata.r;
            auto & DD = paircountdata.paircount;
            auto & norm_DD = paircountdata.norm;

            auto paircountdata2 = AutoPairCount(randoms_xyz, nbins, rmax, periodic, verbose);
            auto & RR = paircountdata2.paircount;
            auto & norm_RR = paircountdata2.norm;

            auto paircountdata3 = CrossPairCount(particles_xyz, randoms_xyz, nbins, rmax, periodic, verbose);
            auto & DR = paircountdata3.paircount;
            auto & norm_DR = paircountdata3.norm;

            // Normalize pair count
            for (auto & x : DD)
                x /= norm_DD;
            for (auto & x : RR)
                x /= norm_RR;
            for (auto & x : DR)
                x /= norm_DR;

            // Correlation function
            auto xi = AutoCorrelationEstimator(DD.data(), DR.data(), RR.data(), nbins, estimator);
            ;
            for (int j = 0; j < nbins; j++) {
                std::cout << std::setw(10) << r[j] << " " << std::setw(10) << xi[j] << " " << std::setw(10)
                          << DD[j] * norm_DD << " " << std::setw(10) << DR[j] * norm_DR << " " << std::setw(10)
                          << RR[j] * norm_RR << "\n";
            }

            // Comparison of other estimators
            auto LZ = AutoCorrelationEstimator(DD.data(), DR.data(), RR.data(), nbins, "LZ");
            auto DP = AutoCorrelationEstimator(DD.data(), DR.data(), RR.data(), nbins, "DP");
            auto HEW = AutoCorrelationEstimator(DD.data(), DR.data(), RR.data(), nbins, "HEW");
            auto PH = AutoCorrelationEstimator(DD.data(), DR.data(), RR.data(), nbins, "PH");
            auto HAM = AutoCorrelationEstimator(DD.data(), DR.data(), RR.data(), nbins, "HAM");
            for (int j = 0; j < nbins; j++) {
                std::cout << r[j] << " " << LZ[j] << " " << DP[j] << " " << HEW[j] << " " << PH[j] << " " << HAM[j]
                          << "\n";
            }
        }

        /// Compute the cross correlation function for galaxies from a survey.
        /// Assuming all catalogues are different
        template <class T>
        void CrossCorrelationFunctionSurvey(std::vector<T> & particles1_xyz,
                                            std::vector<T> & particles2_xyz,
                                            std::vector<T> & randoms1_xyz,
                                            std::vector<T> & randoms2_xyz,
                                            int nbins,
                                            double rmax,
                                            bool verbose) {

            const double periodic = false;
            const std::string estimator = "LZ";

            // Compute pair counts
            auto paircountdata = CrossPairCount(particles1_xyz, particles2_xyz, nbins, rmax, periodic, verbose);
            auto & r = paircountdata.r;
            auto & D1D2 = paircountdata.paircount;
            auto & norm_D1D2 = paircountdata.norm;

            auto paircountdata2 = CrossPairCount(particles1_xyz, randoms2_xyz, nbins, rmax, periodic, verbose);
            auto & D1R2 = paircountdata2.paircount;
            auto & norm_D1R2 = paircountdata2.norm;

            auto paircountdata3 = CrossPairCount(particles2_xyz, randoms1_xyz, nbins, rmax, periodic, verbose);
            auto & R1D2 = paircountdata3.paircount;
            auto & norm_R1D2 = paircountdata3.norm;

            auto paircountdata4 = CrossPairCount(randoms1_xyz, randoms2_xyz, nbins, rmax, periodic, verbose);
            auto & R1R2 = paircountdata4.paircount;
            auto & norm_R1R2 = paircountdata4.norm;

            // Normalize pair count
            for (auto & x : D1D2)
                x /= norm_D1D2;
            for (auto & x : R1R2)
                x /= norm_R1R2;
            for (auto & x : D1R2)
                x /= norm_D1R2;
            for (auto & x : R1D2)
                x /= norm_R1D2;

            // Correlation function
            auto xi = CrossCorrelationEstimator(D1D2, D1R2, R1D2, R1R2, nbins, estimator);
            for (int j = 0; j < nbins; j++) {
                std::cout << r[j] << " " << xi[j] << "\n";
            }

            // Comparison of other estimators
            auto LZ = CrossCorrelationEstimator(D1D2, D1R2, R1D2, R1R2, nbins, "LZ");
            auto HAM = CrossCorrelationEstimator(D1D2, D1R2, R1D2, R1R2, nbins, "HAM");
            auto DP1 = CrossCorrelationEstimator(D1D2, D1R2, R1D2, R1R2, nbins, "DP_NO_R1");
            auto DP2 = CrossCorrelationEstimator(D1D2, D1R2, R1D2, R1R2, nbins, "DP_NO_R2");
            for (int j = 0; j < nbins; j++) {
                std::cout << r[j] << " " << LZ[j] << " " << DP1[j] << " " << DP2[j] << " " << HAM[j] << "\n";
            }
        }

        template <class T>
        void AutoPairCountGridMethod(FML::PARTICLE::ParticlesInBoxes<T> & grid,
                                     std::function<void(int, double *, T &, T &)> & bin,
                                     double rmax,
                                     bool periodic,
                                     bool verbose) {

            // Initialize OpenMP
            int nthreads = 1, id = 0;
#if defined(USE_OMP)
#pragma omp parallel
            {
                if (omp_get_thread_num() == 0)
                    nthreads = omp_get_num_threads();
            }
#endif

            // Initialize MPI
            [[maybe_unused]] int mpi_rank = 0;
            [[maybe_unused]] int mpi_size = 1;
#if defined(USE_MPI)
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            verbose = verbose and mpi_rank == 0;
#endif

            // Only works for ndim <= 3
            T ptemp;
            const int ndim = ptemp.get_ndim();
            assert(ptemp.get_ndim() <= 3);

            // Fetch data from grid
            auto & cells = grid.get_cells();
            const int ngrid = grid.get_ngrid();
            int max_ix = ngrid - 1;
            int max_iy = ngrid - 1;
            int max_iz = ngrid - 1;

            // Find maximum ix,iy,iz for which there are particles
            // in case for surveys where after boxing the particles
            // only occupy a part of the box
            if (!periodic) {
                max_ix = max_iy = max_iz = 0;
                for (int ix = ngrid - 1; ix >= 0; ix--) {
                    for (int iy = ngrid - 1; iy >= 0; iy--) {
                        for (int iz = ngrid - 1; iz >= 0; iz--) {
                            size_t index;
                            if (ndim == 1) {
                                index = ix;
                                iy = iz = 0;
                            }
                            if (ndim == 2) {
                                index = ix * ngrid + iy;
                                iz = 0;
                            }
                            if (ndim == 3) {
                                index = (ix * ngrid + iy) * ngrid + iz;
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
            }
            if (ndim <= 2)
                max_iz = 0;
            if (ndim <= 1)
                max_iy = 0;

            // How many cells in each direction we must search
            int delta_ncells = (int)(ceil(rmax * ngrid)) + 1;

            // Pirnt some info
            if (verbose) {
                std::cout << "\n====================================\n";
                std::cout << "Auto pair counting using grid:\n";
                std::cout << "====================================\n";
                std::cout << "Using " << nthreads << " threads and " << mpi_size << " MPI tasks\n";
                std::cout << "We will go left and right: " << delta_ncells << " cells\n";
            }

            //==========================================================
            // Loop over all the cells
            // The loops only go up to max_ix etc. since we wan to skip
            // looping over cells that we know are empty
            //==========================================================

            int ix0 = 0;
            [[maybe_unused]] int num_processed = 0;
            int istart = 0, iend = max_ix + 1;
#if defined(USE_OMP) && !defined(USE_MPI)
#pragma omp parallel for private(id) schedule(dynamic)
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
#pragma omp parallel for private(id) schedule(dynamic)
#endif
                for (int iy0 = 0; iy0 <= max_iy; iy0++) {
#if defined(USE_OMP) && defined(USE_MPI)
                    id = omp_get_thread_num();
#endif

                    for (int iz0 = 0; iz0 <= max_iz; iz0++) {

                        // Index of current cell
                        int index = 0;
                        if (ndim == 1)
                            index = ix0;
                        if (ndim == 2)
                            index = (ix0 * ngrid + iy0);
                        if (ndim == 3)
                            index = (ix0 * ngrid + iy0) * ngrid + iz0;

                        // Current cell
                        FML::PARTICLE::Cell<T> & curcell = cells[index];

                        // Number of particles in current cell
                        int np_cell = curcell.get_np();

                        // Loop over all particles in current cell
                        for (int ipart_cell = 0; ipart_cell < np_cell; ipart_cell++) {

                            // Current particle
                            T & curpart_cell = curcell.get_part(ipart_cell);

                            // We now want to loop over nearby cells by looking at cube of cells around current cell
                            int ix_left, iy_left, iz_left;
                            int ix_right, iy_right, iz_right;
                            if (periodic) {
                                ix_left = -delta_ncells, ix_right = delta_ncells;
                                iy_left = -delta_ncells, iy_right = delta_ncells;
                                iz_left = -delta_ncells, iz_right = delta_ncells;
                            } else {
                                ix_right = ix0 + delta_ncells <= max_ix ? ix0 + delta_ncells : max_ix;
                                iy_right = iy0 + delta_ncells <= max_iy ? iy0 + delta_ncells : max_iy;
                                iz_right = iz0 + delta_ncells <= max_iz ? iz0 + delta_ncells : max_iz;
                                ix_left = ix0 - delta_ncells >= 0 ? ix0 - delta_ncells : 0;
                                iy_left = iy0 - delta_ncells >= 0 ? iy0 - delta_ncells : 0;
                                iz_left = iz0 - delta_ncells >= 0 ? iz0 - delta_ncells : 0;
                            }

                            if (ndim == 1)
                                iz_left = iz_right = iy_left = iy_right = 0;
                            if (ndim == 2)
                                iz_left = iz_right = 0;

                            // Loop over neightbor cells
                            for (int delta_ix = ix_left; delta_ix <= ix_right; delta_ix++) {
                                int ix = delta_ix;
                                if (periodic) {
                                    ix = ix0 + delta_ix;
                                    while (ix >= ngrid)
                                        ix -= ngrid;
                                    while (ix < 0)
                                        ix += ngrid;
                                } else {
                                    // Avoid double counting so we skip cells that have been correlated with this one
                                    // before
                                    if (ix < ix0)
                                        continue;
                                }

                                for (int delta_iy = iy_left; delta_iy <= iy_right; delta_iy++) {
                                    int iy = delta_iy;
                                    if (periodic) {
                                        iy = iy0 + delta_iy;
                                        while (iy >= ngrid)
                                            iy -= ngrid;
                                        while (iy < 0)
                                            iy += ngrid;
                                    } else {
                                        // Avoid double counting so we skip cells that have been correlated with this
                                        // one before
                                        if (ix == ix0 && iy < iy0)
                                            continue;
                                    }

                                    for (int delta_iz = iz_left; delta_iz <= iz_right; delta_iz++) {
                                        int iz = delta_iz;
                                        if (periodic) {
                                            iz = iz0 + delta_iz;
                                            while (iz >= ngrid)
                                                iz -= ngrid;
                                            while (iz < 0)
                                                iz += ngrid;
                                        } else {
                                            // Avoid double counting so we skip cells that have been correlated with
                                            // this one before
                                            if (ix == ix0 && iy == iy0 && iz < iz0)
                                                continue;
                                        }

                                        // Index of neighboring cell
                                        int index_neighbor_cell;
                                        if (ndim == 1)
                                            index_neighbor_cell = ix;
                                        if (ndim == 2)
                                            index_neighbor_cell = (ix * ngrid + iy);
                                        if (ndim == 3)
                                            index_neighbor_cell = (ix * ngrid + iy) * ngrid + iz;

                                        // Pointer to neighboring cell
                                        FML::PARTICLE::Cell<T> & neighborcell = cells[index_neighbor_cell];

                                        // Number of galaxies in neighboring cell
                                        const int npart_neighbor_cell = neighborcell.get_np();

                                        // Careful: if the nbor cell is the same as the current cell then
                                        // we will overcount if we do all particles so only correlate with
                                        // partices we haven't touched yet
                                        int istart_nbor_cell = 0;
                                        if (!periodic)
                                            if (index == index_neighbor_cell)
                                                istart_nbor_cell = ipart_cell + 1;

                                        // Loop over galaxies in neighbor cells
                                        for (int ipart_neighbor_cell = istart_nbor_cell;
                                             ipart_neighbor_cell < npart_neighbor_cell;
                                             ipart_neighbor_cell++) {

                                            // Galaxy in neighboring cell
                                            T & curpart_neighbor_cell = neighborcell.get_part(ipart_neighbor_cell);

                                            // ==================================================================
                                            // We now count up the pair [curpart_cell] x [curpart_neighbor_cell]
                                            // ==================================================================

                                            // The distance between the two galaxies
                                            auto pos = curpart_cell.get_pos();
                                            auto pos_nbor = curpart_neighbor_cell.get_pos();
                                            double dist[ndim];
                                            if (periodic) {
                                                for (int idim = 0; idim < ndim; idim++) {
                                                    dist[idim] = (pos[idim] - pos_nbor[idim]);
                                                    if (dist[idim] > 1.0 / 2.0)
                                                        dist[idim] -= 1.0;
                                                    if (dist[idim] < -1.0 / 2.0)
                                                        dist[idim] += 1.0;
                                                }
                                            } else {
                                                for (int idim = 0; idim < ndim; idim++)
                                                    dist[idim] = (pos[idim] - pos_nbor[idim]);
                                            }

                                            // Add to bin
                                            bin(id, dist, curpart_cell, curpart_neighbor_cell);
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
                    if (verbose)
                        std::cout << "Processed (" << num_processed << " / " << num_max << ")\n";
                    num_processed++;
                }
#endif
            }
        }

        //====================================================
        // Cross pair counts using grid to speed it up
        // Cross seems to be faster if we loop over the coarsest
        // grid first
        //====================================================
        template <class T, class U>
        void CrossPairCountGridMethod(FML::PARTICLE::ParticlesInBoxes<T> & grid,
                                      FML::PARTICLE::ParticlesInBoxes<U> & grid2,
                                      std::function<void(int, double *, T &, U &)> & bin,
                                      double rmax,
                                      bool periodic,
                                      bool verbose) {

            // Initialize OpenMP
            int nthreads = 1, id = 0;
#if defined(USE_OMP)
#pragma omp parallel
            {
                if (omp_get_thread_num() == 0)
                    nthreads = omp_get_num_threads();
            }
#endif

            // Initialize MPI
            [[maybe_unused]] int mpi_rank = 0;
            [[maybe_unused]] int mpi_size = 1;
#if defined(USE_MPI)
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            verbose = verbose and mpi_rank == 0;
#endif

            // Fetch ndim
            T ptemp;
            U utemp;
            const int ndim = ptemp.get_ndim();
            assert(ndim == utemp.get_ndim());

            // Fetch data from the grid
            auto & cells = grid.get_cells();
            const int ngrid = grid.get_ngrid();
            int max_ix = ngrid - 1;
            int max_iy = ngrid - 1;
            int max_iz = ngrid - 1;

            // Fetch data from the grid2
            auto & cells2 = grid2.get_cells();
            const int ngrid2 = grid2.get_ngrid();
            int max_ix2 = ngrid2 - 1;
            int max_iy2 = ngrid2 - 1;
            int max_iz2 = ngrid2 - 1;

            // Find maximum ix,iy,iz for which there are particles
            // in case for surveys where after boxing the particles
            // only occupy a part of the box
            if (!periodic) {
                max_ix = max_iy = max_iz = 0;
                for (int ix = ngrid - 1; ix >= 0; ix--) {
                    for (int iy = ngrid - 1; iy >= 0; iy--) {
                        for (int iz = ngrid - 1; iz >= 0; iz--) {
                            size_t index;
                            if (ndim == 1) {
                                index = ix;
                                iy = iz = 0;
                            }
                            if (ndim == 2) {
                                index = ix * ngrid + iy;
                                iz = 0;
                            }
                            if (ndim == 3) {
                                index = (ix * ngrid + iy) * ngrid + iz;
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

            // Print some data
            if (verbose) {
                std::cout << "\n====================================\n";
                std::cout << "Cross pair counting using grid:\n";
                std::cout << "====================================\n";
                std::cout << "Using " << nthreads << " threads and " << mpi_size << " MPI tasks\n";
                std::cout << "We will go left and right: " << delta_ncells2 << " cells\n";
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
#pragma omp parallel for private(id) schedule(dynamic)
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
#pragma omp parallel for private(id) schedule(dynamic)
#endif
                for (int iy0 = 0; iy0 <= max_iy; iy0++) {
#if defined(USE_OMP) && defined(USE_MPI)
                    id = omp_get_thread_num();
#endif
                    for (int iz0 = 0; iz0 <= max_iz; iz0++) {

                        // Index of current cell
                        int index;
                        if (ndim == 1)
                            index = ix0;
                        if (ndim == 2)
                            index = ix0 * ngrid + iy0;
                        if (ndim == 3)
                            index = (ix0 * ngrid + iy0) * ngrid + iz0;

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
                            int ix_grid2 = (int)(ix0 * (double)ngrid2 / (double)ngrid);
                            int iy_grid2 = (int)(iy0 * (double)ngrid2 / (double)ngrid);
                            int iz_grid2 = (int)(iz0 * (double)ngrid2 / (double)ngrid);

                            // We now want to loop over nearby cells by looking at cube of cells around current cell
                            int ix2_left, iy2_left, iz2_left, ix2_right, iy2_right, iz2_right;
                            if (periodic) {
                                ix2_left = -delta_ncells2, ix2_right = delta_ncells2;
                                iy2_left = -delta_ncells2, iy2_right = delta_ncells2;
                                iz2_left = -delta_ncells2, iz2_right = delta_ncells2;
                            } else {
                                ix2_right = ix_grid2 + delta_ncells2 <= max_ix2 ? ix_grid2 + delta_ncells2 : max_ix2;
                                iy2_right = iy_grid2 + delta_ncells2 <= max_iy2 ? iy_grid2 + delta_ncells2 : max_iy2;
                                iz2_right = iz_grid2 + delta_ncells2 <= max_iz2 ? iz_grid2 + delta_ncells2 : max_iz2;
                                ix2_left = ix_grid2 - delta_ncells2 >= 0 ? ix_grid2 - delta_ncells2 : 0;
                                iy2_left = iy_grid2 - delta_ncells2 >= 0 ? iy_grid2 - delta_ncells2 : 0;
                                iz2_left = iz_grid2 - delta_ncells2 >= 0 ? iz_grid2 - delta_ncells2 : 0;
                            }
                            if (ndim == 1)
                                iy2_left = iy2_right = iz2_left = iz2_right = 0;
                            if (ndim == 2)
                                iz2_left = iz2_right = 0;

                            // Loop over neightbor cells
                            for (int delta_ix2 = ix2_left; delta_ix2 <= ix2_right; delta_ix2++) {
                                int ix2 = delta_ix2;
                                if (periodic) {
                                    ix2 = ix_grid2 + delta_ix2;
                                    while (ix2 >= ngrid2)
                                        ix2 -= ngrid2;
                                    while (ix2 < 0)
                                        ix2 += ngrid2;
                                }

                                for (int delta_iy2 = iy2_left; delta_iy2 <= iy2_right; delta_iy2++) {
                                    int iy2 = delta_iy2;
                                    if (periodic) {
                                        iy2 = iy_grid2 + delta_iy2;
                                        while (iy2 >= ngrid2)
                                            iy2 -= ngrid2;
                                        while (iy2 < 0)
                                            iy2 += ngrid2;
                                    }

                                    for (int delta_iz2 = iz2_left; delta_iz2 <= iz2_right; delta_iz2++) {
                                        int iz2 = delta_iz2;
                                        if (periodic) {
                                            iz2 = iz_grid2 + delta_iz2;
                                            while (iz2 >= ngrid2)
                                                iz2 -= ngrid2;
                                            while (iz2 < 0)
                                                iz2 += ngrid2;
                                        }

                                        // Index of neighboring cell
                                        int index_neighbor_cell;
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

                                        // Loop over galaxies in neighbor cells
                                        for (int ipart_neighbor_cell = 0; ipart_neighbor_cell < npart_neighbor_cell;
                                             ipart_neighbor_cell++) {

                                            // Galaxy in neighboring cell
                                            U & curpart_neighbor_cell = neighborcell.get_part(ipart_neighbor_cell);

                                            // ==================================================================
                                            // We now count up the pair [curpart_cell] x [curpart_neighbor_cell]
                                            // ==================================================================
                                            auto pos = curpart_cell.get_pos();
                                            auto pos_nbor = curpart_neighbor_cell.get_pos();
                                            double dist[ndim];
                                            if (periodic) {
                                                for (int idim = 0; idim < ndim; idim++) {
                                                    dist[idim] = (pos[idim] - pos_nbor[idim]);
                                                    if (dist[idim] > 1.0 / 2.0)
                                                        dist[idim] -= 1.0;
                                                    if (dist[idim] < -1.0 / 2.0)
                                                        dist[idim] += 1.0;
                                                }
                                            } else {
                                                for (int idim = 0; idim < ndim; idim++)
                                                    dist[idim] = (pos[idim] - pos_nbor[idim]);
                                            }

                                            // Add to bin
                                            bin(id, dist, curpart_cell, curpart_neighbor_cell);
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
                    if (verbose)
                        std::cout << "Processed (" << num_processed << " / " << num_max << ")\n";
                    num_processed++;
                }
#endif
            }
        }

        //======================================================================
        // Computes the correlation function
        // Assuming particles in a box of size 1 so all positions in [0,1)
        // rmax is maximum radius in units of the boxsize, i.e. in [0,1)
        //======================================================================
        template <class T>
        AutoPairCountData
        AutoPairCount(std::vector<T> & particles, int nbins, double rmax, bool periodic, bool verbose) {
            const double rmax2 = rmax * rmax;

            // Fetch how many dimensions we are working in
            T ptemp;
            const int ndim = ptemp.get_ndim();

            // Get number of threads
            int nthreads = 1;
#ifdef USE_OMP
#pragma omp parallel
            {
                int id = omp_get_thread_num();
                if (id == 0)
                    nthreads = omp_get_num_threads();
            }
#endif

            // How many pairs in each bin
            std::vector<std::vector<double>> count_threads(nthreads, std::vector<double>(nbins, 0.0));

            //========================================
            // Define the binning function
            //========================================
            std::function<void(int, double *, T &, T &)> binning = [&](int thread_id, double * dist, T & p1, T & p2) {
                const double weight1 = p1.get_weight();
                const double weight2 = p2.get_weight();

                // Compute squared distance between pairs
                double dist2 = dist[0] * dist[0];
                if (ndim >= 2)
                    dist2 += dist[1] * dist[1];
                if (ndim >= 3)
                    dist2 += dist[2] * dist[2];
                if (dist2 >= rmax2)
                    return;
                if (dist2 == 0.0)
                    return;

                // Compute bin index and add to bin
                const int ibin = int(sqrt(dist2 / rmax2) * nbins);
                count_threads[thread_id][ibin] += weight1 * weight2;

                // ...add other things to bin here...
            };

            // Select a good ngrid size
            // 8 cells to get to rmax
            // 2 particles per cells on average
            // Minimum 10 cells per dim
            int ngrid = std::min(int(8.0 / rmax), int(std::pow(particles.size() / 2.0, 1. / double(ndim))));
            if (ngrid < 10)
                ngrid = 10;

            // Add particles to a grid
            FML::PARTICLE::ParticlesInBoxes<T> grid;
            grid.create(particles.data(), particles.size(), ngrid);
            grid.info();

            // Do the pair counts
            AutoPairCountGridMethod<T>(grid, binning, rmax, periodic, verbose);

            // Sum up over threads
            std::vector<double> count(nbins, 0.0);
            std::vector<double> r(nbins, 0.0);
            std::vector<double> r_edge(nbins + 1, 0.0);
            for (int j = 0; j < nbins; j++) {
                for (int i = 0; i < nthreads; i++) {
                    count[j] += count_threads[i][j];
                }
                r[j] = rmax * (j + 0.5) / double(nbins);
                r_edge[j] = rmax * j / double(nbins);
            }
            r_edge[nbins] = rmax;

            // Compute sum of weights
            double sum_weights = 0.0;
            double sum_weights_squared = 0.0;
            auto & cells = grid.get_cells();
            for (auto & cell : cells) {
                for (auto & p : cell.get_part()) {
                    double w = p.get_weight();
                    sum_weights += w;
                    sum_weights_squared += w * w;
                }
            }

#ifdef USE_MPI
            // Gather data from all CPUs
            int mpi_rank = 0, mpi_size = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

            // General method to reduce a single grid over all processes
            auto reduce_grid_MPI = [&](std::vector<double> & grid) {
                size_t nsize = grid.size();
                std::vector<double> recv(mpi_size * nsize);
                MPI_Allgather(grid.data(), nsize, MPI_DOUBLE, recv.data(), nsize, MPI_DOUBLE, MPI_COMM_WORLD);
                std::vector<double> reduced_grid(nsize, 0.0);
                for (int task = 0; task < mpi_size; task++)
                    for (size_t i = 0; i < nsize; i++)
                        reduced_grid[i] += recv[task * nsize + i];
                return reduced_grid;
            };

            // Reduce all grids
            count = reduce_grid_MPI(count);
#endif

            AutoPairCountData result;
            result.r = r;
            result.r_edge = r_edge;
            result.paircount = count;
            result.sum_weights = sum_weights;
            result.sum_weights_squared = sum_weights_squared;
            result.norm = (sum_weights * sum_weights - sum_weights_squared) / 2.0;

            return result;
        }

        //======================================================================
        // Computes the cross correlation function
        // Assuming particles in a box of size 1 so all positions in [0,1)
        // rmax is maximum radius in units of the boxsize, i.e. in [0,1)
        //======================================================================
        template <class T, class U>
        CrossPairCountData CrossPairCount(std::vector<T> & particles1,
                                          std::vector<U> & particles2,
                                          int nbins,
                                          double rmax,
                                          bool periodic,
                                          bool verbose) {
            const double rmax2 = rmax * rmax;

            // Fetch how many dimensions we are working in
            T ptemp;
            const int ndim = ptemp.get_ndim();

            // Get number of threads
            int nthreads = 1;
#ifdef USE_OMP
#pragma omp parallel
            {
                int id = omp_get_thread_num();
                if (id == 0)
                    nthreads = omp_get_num_threads();
            }
#endif

            // How many pairs in each bin
            std::vector<std::vector<double>> count_threads(nthreads, std::vector<double>(nbins, 0.0));

            //========================================
            // Define the binning function
            //========================================
            std::function<void(int, double *, T &, U &)> binning = [&](int thread_id, double * dist, T & p1, U & p2) {
                const double weight1 = p1.get_weight();
                const double weight2 = p2.get_weight();

                // Compute squared distance between pairs
                double dist2 = dist[0] * dist[0];
                if (ndim >= 2)
                    dist2 += dist[1] * dist[1];
                if (ndim >= 3)
                    dist2 += dist[2] * dist[2];
                if (dist2 >= rmax2)
                    return;

                // Compute bin index and add to bin
                const int ibin = int(sqrt(dist2 / rmax2) * nbins);
                count_threads[thread_id][ibin] += weight1 * weight2;

                // ...add other things to bin here...
            };

            // Select a good ngrid size
            // 8 cells to get to rmax
            // 2 particles per cells on average
            // Minimum 10 cells per dim
            int ngrid1 = std::min(int(8.0 / rmax), int(std::pow(particles1.size() / 2.0, 1. / double(ndim))));
            if (ngrid1 < 10)
                ngrid1 = 10;
            int ngrid2 = std::min(int(8.0 / rmax), int(std::pow(particles2.size() / 2.0, 1. / double(ndim))));
            if (ngrid2 < 10)
                ngrid2 = 10;

            // Assign particles to a grid
            FML::PARTICLE::ParticlesInBoxes<T> grid1;
            FML::PARTICLE::ParticlesInBoxes<U> grid2;
            grid1.create(particles1.data(), particles1.size(), ngrid1);
            grid2.create(particles2.data(), particles2.size(), ngrid2);

            // Do the pair counts
            CrossPairCountGridMethod<T, U>(grid1, grid2, binning, rmax, periodic, verbose);

            // Sum up over threads...
            std::vector<double> count(nbins, 0.0);
            std::vector<double> r(nbins, 0.0);
            std::vector<double> r_edge(nbins + 1, 0.0);
            for (int j = 0; j < nbins; j++) {
                for (int i = 0; i < nthreads; i++) {
                    count[j] += count_threads[i][j];
                }
                r[j] = rmax * (j + 0.5) / double(nbins);
                r_edge[j] = rmax * j / double(nbins);
            }
            r_edge[nbins] = rmax;

            // Compute sum of weights NB: No MPI comm needed here
            // as we assume all tasks have all the particles
            double sum_weights = 0.0;
            double sum_weights_squared = 0.0;
            auto & cells1 = grid1.get_cells();
            for (auto & cell : cells1) {
                for (auto & p : cell.get_part()) {
                    double w = p.get_weight();
                    sum_weights += w;
                    sum_weights_squared += w * w;
                }
            }
            double sum2_weights = 0.0;
            double sum2_weights_squared = 0.0;
            auto & cells2 = grid2.get_cells();
            for (auto & cell : cells2) {
                for (auto & p : cell.get_part()) {
                    double w = p.get_weight();
                    sum2_weights += w;
                    sum2_weights_squared += w * w;
                }
            }

#ifdef USE_MPI
            // Gather data from all CPUs
            int mpi_rank = 0, mpi_size = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

            // General method to reduce a single grid over all processes
            auto reduce_grid_MPI = [&](std::vector<double> & grid) {
                size_t nsize = grid.size();
                std::vector<double> recv(mpi_size * nsize);
                MPI_Allgather(grid.data(), nsize, MPI_DOUBLE, recv.data(), nsize, MPI_DOUBLE, MPI_COMM_WORLD);
                std::vector<double> reduced_grid(nsize, 0.0);
                for (int task = 0; task < mpi_size; task++)
                    for (size_t i = 0; i < nsize; i++)
                        reduced_grid[i] += recv[task * nsize + i];
                return reduced_grid;
            };

            // Reduce all grids...
            count = reduce_grid_MPI(count);
#endif

            CrossPairCountData result;
            result.r = r;
            result.r_edge = r_edge;
            result.paircount = count;
            result.sum_weights = sum_weights;
            result.sum_weights_squared = sum_weights_squared;
            result.sum2_weights = sum2_weights;
            result.sum2_weights_squared = sum2_weights_squared;
            result.norm = sum_weights * sum2_weights;

            return result;
        }
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
