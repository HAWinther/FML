#ifndef GALAXIES_TO_BOX_HEADER
#define GALAXIES_TO_BOX_HEADER

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

// We need these for the r(z) spline
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

//==============================================================================
//
// Takes a set of galaxies with (RA, DEC, z) and converts them to
// particles in a 3D cartesian box. This method only works in 3D
//
// Galaxies must have the methods:
// get_RA()
// get_DEC()
// get_z()
// and positions must have the methods:
// auto * get_pos();
// int get_ndim() ( which should return 3 )
//
// We require the Hubble function H(z)/c to be provided by the user, e.g.
// auto hubble_over_c_of_z = [](double z){
//  return (1.0 / 2997.9) * sqrt(OmegaM * pow(1+z,3) + 1.0 - OmegaM);
// };
// The positions/boxsize will be in whatever units c/H is provided in.
// The above example corresponds to Mpc/h
//
// If shiftPositions we shift the positions to that they are in
// [0,Box] where Box is the smallest distance
//
// If scalePositions we scale the positions so that they are in [0,1). This require
// that we also shift the positions
//
// The boxsize contains the maximum distance over all the coordinates in Mpc/h
//
// OpenMP parallelized and safe to run with MPI
//
//==============================================================================

namespace FML {

    /// This namespace contains methods for dealing with surveys. So far its not much here just some conversion from
    /// equitorial to cartesian coordinates.
    namespace SURVEY {

        //==============================================================================
        /// Take a set of galaxies galaxies_ra_dec_z with (RA,DEC,z) and convert them to
        /// cartesian coordinates (x,y,z) stored in particles_xyz
        ///
        /// The positions will be in the same units as the 1.0/hubble_over_c_of_z
        /// Gives back the min/max of the positions (useful for boxing the catalog)
        ///
        /// @tparam T Particle class for the galaxies
        /// @tparam U Particle class for the particles we make from the galaxies
        ///
        /// @param[in] galaxies_ra_dec_z Particles with RA, DEC and Z.
        /// @param[in] ngalaxies Number of galaxies
        /// @param[out] particles_xyz Vector with galaxies as particles with cartesian coordinates.
        /// @param[in] hubble_over_c_of_z This is the function \f$ H(z)/c \f$ used to compute the redshift-comobing
        /// distance relationship. Postions units is the same as those of \f$ c/H(z) \f$ (so e.g. if you want Mpc/h then
        /// we need H0 = 100, if you want kpc/s then use H0 = 10^5 and so on.
        /// @param[out] min_max_x The min/max values of x-postions
        /// @param[out] min_max_y The min/max values of x-postions
        /// @param[out] min_max_z The min/max values of x-postions
        ///
        //==============================================================================
        template <class T, class U>
        void EquitorialToCartesianCoordinates(T * galaxies_ra_dec_z,
                                              size_t ngalaxies,
                                              std::vector<U> & particles_xyz,
                                              std::function<double(double)> & hubble_over_c_of_z,
                                              std::pair<double, double> & min_max_x,
                                              std::pair<double, double> & min_max_y,
                                              std::pair<double, double> & min_max_z) {

            // Find maximum redshift
            double z_max = 0.0;
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < ngalaxies; i++) {
                const double z = galaxies_ra_dec_z[i].get_z();
                z_max = std::max(z, z_max);
            }

            // Make a z-array
            const int nz = 10000;
            std::vector<double> z_arr(nz);
            std::vector<double> r_arr(nz);
            for (int i = 0; i < nz; i++) {
                z_arr[i] = 1.1 * z_max * i / double(nz - 1);
            }

            // Solve the ODE for the co-moving distance
            using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
            using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
            using DVector = FML::SOLVERS::ODESOLVER::DVector;
            ODEFunction deriv = [&](double z, [[maybe_unused]] const double * y, double * dydx) {
                dydx[0] = 1.0 / hubble_over_c_of_z(z);
                return GSL_SUCCESS;
            };

            DVector r_ini{0.0};
            ODESolver r_ode(1e-3, 1e-10, 1e-10);
            r_ode.solve(deriv, z_arr, r_ini);
            r_arr = r_ode.get_data_by_component(0);

            // Make a spline of r(z)
            FML::INTERPOLATION::SPLINE::Spline r_of_z_spline(z_arr, r_arr, "r_of_z_spline");

            // Fetch ndim from particles and check that we have the right dimensions
            U utemp;
            assert(utemp.get_ndim() == 3);

            // Make particles and convert to cartesian coordinates
            particles_xyz = std::vector<U>(ngalaxies);
            double max_x = -1e100;
            double max_y = -1e100;
            double max_z = -1e100;
            double min_x = +1e100;
            double min_y = +1e100;
            double min_z = +1e100;

            const double degrees_to_radial = 2.0 * M_PI / 360.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_x, max_y, max_z) reduction(min : min_x, min_y, min_z)
#endif
            for (size_t i = 0; i < ngalaxies; i++) {
                auto * Pos = particles_xyz[i].get_pos();
                const double RA = galaxies_ra_dec_z[i].get_RA();
                const double DEC = galaxies_ra_dec_z[i].get_DEC();
                const double redshift = galaxies_ra_dec_z[i].get_z();
                const double r = r_of_z_spline(redshift);

                const double cosTheta = std::cos((90.0 - RA) * degrees_to_radial);
                const double sinTheta = std::sqrt(1.0 - cosTheta * cosTheta);
                const double phi = DEC * degrees_to_radial;

                const double x = r * sinTheta * std::cos(phi);
                const double y = r * sinTheta * std::sin(phi);
                const double z = r * cosTheta;

                // Compute max/min
                max_x = std::max(x, max_x);
                max_y = std::max(y, max_y);
                max_z = std::max(z, max_z);
                min_x = std::min(x, min_x);
                min_y = std::min(y, min_y);
                min_z = std::min(z, min_z);

                // Assign positions
                Pos[0] = x;
                Pos[1] = y;
                Pos[2] = z;
            }

            min_max_x = {min_x, max_x};
            min_max_y = {min_y, max_y};
            min_max_z = {min_z, max_z};
        }

        //==============================================================================
        /// @brief Transform galaxies with positions defined by RA,DEC,Z and transform these to
        /// cartesian positions in [0,1).
        ///
        /// @tparam T Particle class for the galaxies
        /// @tparam U Particle class for the particles we make from the galaxies
        ///
        /// @param[in] galaxies_ra_dec_z Particles with RA, DEC and Z.
        /// @param[in] ngalaxies Number of galaxies
        /// @param[out] particles_xyz Vector with galaxies as particles with cartesian coordinates.
        /// @param[in] hubble_over_c_of_z This is the function \f$ H(z)/c \f$ used to compute the redshift-comobing
        /// distance relationship. Postions units is the same as those of \f$ c/H(z) \f$ (so e.g. if you want Mpc/h then
        /// we need H0 = 100, if you want kpc/s then use H0 = 10^5 and so on.
        /// @param[out] boxsize The boxsize we need to place the particles in a cubic box.
        /// @param[in] shiftPositions Shift the positions such that all are >=0
        /// @param[in] scalePositions Scale positions so that all are in [0,1). This requires also shifting positions.
        /// @param[in] verbose Print info while doing this.
        ///
        //==============================================================================
        template <class T, class U>
        void GalaxiesToBox(const T * galaxies_ra_dec_z,
                           size_t ngalaxies,
                           std::vector<U> & particles_xyz,
                           std::function<double(double)> & hubble_over_c_of_z,
                           double & boxsize,
                           bool shiftPositions,
                           bool scalePositions,
                           bool verbose) {

            // If we are to scale the positions then they must be shifted so they lie in [0,1)
            if (scalePositions)
                assert(shiftPositions);

            // Fetch ndim from particles and check that we have the right dimensions
            U utemp;
            assert(utemp.get_ndim() == 3);

#ifdef USE_MPI
            // If we run with MPI only print once
            if (verbose) {
                int mpi_rank = 0, mpi_size = 1;
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
                verbose = (mpi_rank == 0);
            }
#endif

            // To cartesian coordinates
            std::pair<double, double> min_max_x, min_max_y, min_max_z;
            EquitorialToCartesianCoordinates(
                galaxies_ra_dec_z, ngalaxies, particles_xyz, hubble_over_c_of_z, min_max_x, min_max_y, min_max_z);

            double min_x = min_max_x.first;
            double min_y = min_max_y.first;
            double min_z = min_max_z.first;
            double max_x = min_max_x.second;
            double max_y = min_max_y.second;
            double max_z = min_max_z.second;

            max_x -= min_x;
            max_y -= min_y;
            max_z -= min_z;

            // Compute the minimum boxsize we can fit these particle in
            boxsize = (1. + 1e-10) * std::max(std::max(max_x, max_y), max_z);

            if (verbose) {
                std::cout << "Maximum Pos: " << max_x << " " << max_y << " " << max_z << "\n";
                std::cout << "Boxsize: " << boxsize << "\n";
            }

            // Shift positions and possily scale them so they are in [0,1)
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < ngalaxies; i++) {
                auto * Pos = particles_xyz[i].get_pos();

                if (shiftPositions) {
                    Pos[0] -= min_x;
                    Pos[1] -= min_y;
                    Pos[2] -= min_z;
                }

                if (scalePositions) {
                    Pos[0] /= boxsize;
                    Pos[1] /= boxsize;
                    Pos[2] /= boxsize;
                }
            }
        }

        //==============================================================================
        /// @brief Transform galaxies and randos with positions defined by RA,DEC,Z and transform these to
        /// cartesian positions in [0,1. Useful if we have more catalogues like galaxy + randoms
        /// we should use the same shift and same boxsize so do it together.
        ///
        /// @tparam T Particle class for the galaxies/randoms
        /// @tparam U Particle class for the particles we make from the galaxies/randoms
        ///
        /// @param[in] galaxies_ra_dec_z Galaxies with RA, DEC and Z.
        /// @param[in] ngalaxies Number of galaxies
        /// @param[in] randoms_ra_dec_z Random galaxies with RA, DEC and Z.
        /// @param[in] nrandoms Number of random galaxies
        /// @param[out] particles_xyz Vector with galaxies as particles with cartesian coordinates.
        /// @param[out] randoms_xyz Vector with random galaxies as particles with cartesian coordinates.
        /// @param[in] hubble_over_c_of_z This is the function \f$ H(z)/c \f$ used to compute the redshift-comobing
        /// distance relationship. Postions units is the same as those of \f$ c/H(z) \f$ (so e.g. if you want Mpc/h then
        /// we need H0 = 100, if you want kpc/s then use H0 = 10^5 and so on.
        /// @param[out] boxsize The boxsize we need to place the galaxies and randoms in a cubic box.
        /// @param[in] shiftPositions Shift the positions such that all are >=0
        /// @param[in] scalePositions Scale positions so that all are in [0,1). This requires also shifting positions.
        /// @param[in] verbose Print info while doing this.
        ///
        //==============================================================================
        template <class T, class U>
        void GalaxiesRandomsToBox(const T * galaxies_ra_dec_z,
                                  size_t ngalaxies,
                                  const T * randoms_ra_dec_z,
                                  size_t nrandoms,
                                  std::vector<U> & galaxies_xyz,
                                  std::vector<U> & randoms_xyz,
                                  std::function<double(double)> & hubble_over_c_of_z,
                                  double & boxsize,
                                  bool shiftPositions,
                                  bool scalePositions,
                                  bool verbose) {

            double boxsize1, boxsize2;

            GalaxiesToBox(
                galaxies_ra_dec_z, ngalaxies, galaxies_xyz, hubble_over_c_of_z, boxsize1, false, false, verbose);

            GalaxiesToBox(randoms_ra_dec_z, nrandoms, randoms_xyz, hubble_over_c_of_z, boxsize2, false, false, verbose);

            double max_x = -1e100, max_y = -1e100, max_z = -1e100;
            double min_x = +1e100, min_y = +1e100, min_z = +1e100;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_x, max_y, max_z) reduction(min : min_x, min_y, min_z)
#endif
            for (size_t i = 0; i < galaxies_xyz.size(); i++) {
                auto * Pos = galaxies_xyz[i].get_pos();
                max_x = std::max(Pos[0], max_x);
                max_y = std::max(Pos[1], max_y);
                max_z = std::max(Pos[2], max_z);
                min_x = std::min(Pos[0], min_x);
                min_y = std::min(Pos[1], min_y);
                min_z = std::min(Pos[2], min_z);
            }
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_x, max_y, max_z) reduction(min : min_x, min_y, min_z)
#endif
            for (size_t i = 0; i < randoms_xyz.size(); i++) {
                auto * Pos = randoms_xyz[i].get_pos();
                max_x = std::max(Pos[0], max_x);
                max_y = std::max(Pos[1], max_y);
                max_z = std::max(Pos[2], max_z);
                min_x = std::min(Pos[0], min_x);
                min_y = std::min(Pos[1], min_y);
                min_z = std::min(Pos[2], min_z);
            }

            max_x -= min_x;
            max_y -= min_y;
            max_z -= min_z;

            // The boxsize we use is the maximum of the two
            boxsize = 1.1 * std::max(std::max(max_x, max_y), max_z);

            if (shiftPositions) {
                for (auto & p : galaxies_xyz) {
                    auto * Pos = p.get_pos();
                    Pos[0] += min_x;
                    Pos[1] += min_y;
                    Pos[2] += min_z;
                }
                for (auto & p : randoms_xyz) {
                    auto * Pos = p.get_pos();
                    Pos[0] += min_x;
                    Pos[1] += min_y;
                    Pos[2] += min_z;
                }
            }

            if (scalePositions) {
                for (auto & p : galaxies_xyz) {
                    auto * Pos = p.get_pos();
                    Pos[0] /= boxsize;
                    Pos[1] /= boxsize;
                    Pos[2] /= boxsize;
                }
                for (auto & p : randoms_xyz) {
                    auto * Pos = p.get_pos();
                    Pos[0] /= boxsize;
                    Pos[1] /= boxsize;
                    Pos[2] /= boxsize;
                }
            }
        }
    } // namespace SURVEY
} // namespace FML

#endif
