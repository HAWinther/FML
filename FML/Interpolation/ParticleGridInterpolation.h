#ifndef PARTICLEGRIDINTERPOLATION_HEADER
#define PARTICLEGRIDINTERPOLATION_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <functional>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>

namespace FML {

    //============================================================================
    /// This namespace deals with interpolation and density assignment.
    ///
    /// We give methods to assign particles to a grid to compute the density contrast
    /// and to interpolate a grid to any given position using the same
    /// B spline kernels (this is basically a convolution of the kernels with the grid).
    /// This kind of interpolation is useful for computing forces from a density
    /// field of particles. Using the interpolation method corresponding to the
    /// density assignment help prevent unphysical self-forces.
    ///
    /// If the particle class has a get_mass method then we will use this mass (divided by the mean mass of all
    /// particles so the absolute size of the masses does not matter) when assigning the particles to the grid.
    /// Otherwise we will assume all particles have the same mass. The resulting density-field delta will always have
    /// mean 0.
    ///
    /// The assignment function is a B spline kernel of any order,
    /// i.e. H*H*...*H with H being a tophat and * convolution.
    /// The fourier space window functions of these are just sinc(pi/2 * k / kny)^ORDER
    /// Order 1=NGP, 2=CIC, 3=TSC, 4=PCS, 5=PQS and higher orders are easily added if you
    /// for some strange reason needs this (just add the kernel function).
    ///
    /// Also contains a method for doing a convolution of a grid with a general kernel.
    ///
    /// Compile time defines:
    ///
    /// DEBUG_INTERPOL           : Check that the interpolation weights
    ///                            sum to unity for density assignment
    ///
    /// CELLCENTERSHIFTED        : Shift the position of the cell (located at center of cell
    ///                            vs at the corners). Use with care. Not using this option
    ///                            saves a slice for even order interpoation and using it
    ///                            saves a slice for odd ordered interpolation (TSC+).
    ///                            Only relevant if memory is really tight and you need to use
    ///                            TSC or PQS
    ///
    //============================================================================

    namespace INTERPOLATION {

        // The float type that we use for FFTE
        using FloatType = FML::GRID::FloatType;

        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;

        /// @brief Interpolate a grid to a set of positions given by the positions of particles.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class. Must have a get_pos() method.
        /// @tparam ORDER The order of the B-spline interpolation (1=NGP, 2=CIC, 3=TSC, 4=PCS, 5=PQS, ...)
        ///
        /// @param[in] grid A grid.
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[out] interpolated_values A vector with the interpolated values, one per particle. Allocated in the
        /// method.
        ///
        template <int N, int ORDER, class T>
        void interpolate_grid_to_particle_positions(const FFTWGrid<N> & grid,
                                                    const T * part,
                                                    size_t NumPart,
                                                    std::vector<FloatType> & interpolated_values);

        /// @brief Interpolate a vector of grids (e.g. force vector) to a set of positions given by the positions of
        /// particles.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class. Must have a get_pos() method.
        /// @tparam ORDER The order of the B-spline interpolation (1=NGP, 2=CIC, 3=TSC, 4=PCS, 5=PQS, ...)
        ///
        /// @param[in] grid_vec A N-dimensional array of grids
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[out] interpolated_values_vec The interpolated values, one per grid per particle.
        /// Allocated in the method.
        ///
        template <int N, int ORDER, class T>
        void
        interpolate_grid_vector_to_particle_positions(const std::array<FFTWGrid<N>, N> & grid_vec,
                                                      const T * part,
                                                      size_t NumPart,
                                                      std::array<std::vector<FloatType>, N> & interpolated_values_vec);

        /// @brief Interpolate a grid to a set of positions given by the positions of particles.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] grid A grid.
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[out] interpolated_values A vector with the interpolated values, one per particle. Allocated in the
        /// method.
        /// @param[in] interpolation_method The interpolation method: NGP, CIC, TSC, PCS or PQS.
        ///
        template <int N, class T>
        void interpolate_grid_to_particle_positions(const FFTWGrid<N> & grid,
                                                    const T * part,
                                                    size_t NumPart,
                                                    std::vector<FloatType> & interpolated_values,
                                                    std::string interpolation_method);

        /// @brief Interpolate a vector of grids to a set of positions given by the positions of particles.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class. Must have a get_pos() method.
        ///
        /// @param[in] grid_vec A vector of grids
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[out] interpolated_values_vec The interpolated values, one per grid per particle. Allocated in the
        /// method.
        /// @param[in] interpolation_method The interpolation method: NGP, CIC, TSC, PCS or PQS.
        ///
        template <int N, class T>
        void
        interpolate_grid_vector_to_particle_positions(const std::array<FFTWGrid<N>, N> & grid_vec,
                                                      const T * part,
                                                      size_t NumPart,
                                                      std::array<std::vector<FloatType>, N> & interpolated_values_vec,
                                                      std::string interpolation_method);

        /// Gather a grid vector and immediately consume each interpolated value, without output arrays.
        /// apply(index, value) receives an array<FloatType, N>, preserving the buffered gather's rounding.
        /// It returns a nonnegative double diagnostic; the result is its local maximum (or zero for no particles).
        /// Calls run concurrently under OpenMP: apply must be thread safe and must not throw. Only the
        /// calling particle's data should be modified. Grid halos must already have been communicated.
        template <int N, int ORDER, class T, class Apply>
        double interpolate_grid_vector_to_particle_positions_apply(const std::array<FFTWGrid<N>, N> & grid,
                                                                   const T * part,
                                                                   size_t NumPart,
                                                                   Apply && apply);

        template <int N, class T, class Apply>
        double interpolate_grid_vector_to_particle_positions_apply(const std::array<FFTWGrid<N>, N> & grid,
                                                                   const T * part,
                                                                   size_t NumPart,
                                                                   std::string interpolation_method,
                                                                   Apply && apply);

        /// @brief Assign particles to a grid to compute the over density field delta.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class. Must have a get_pos() method. If the particle has a get_mass method then this
        /// is used to weight the particle (we assign the particle with weight mass / mean_mass).
        ///
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[in] NumPartTot How many particles/positions we have in total over all tasks.
        /// @param[out] density The overdensity field.
        /// @param[in] density_assignment_method The assignment method: NGP, CIC, TSC, PCS or PQS.
        ///
        template <int N, class T>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid<N> & density,
                               std::string density_assignment_method);

        /// @brief Assign particles to a grid to compute the over density field delta.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class. Must have a get_pos() method. If the particle has a get_mass method then this
        /// is used to weight the particle (we assign the particle with weight mass / mean_mass).
        /// @tparam ORDER The order of the B-spline interpolation (1=NGP, 2=CIC, 3=TSC, 4=PCS, 5=PQS, ...). If larger
        /// than 5 then you must implement kernel<ORDER> yourself (a simple Mathematica calculation), see the source.
        ///
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[in] NumPartTot How many particles/positions we have in total over all tasks.
        /// @param[out] density The overdensity field.
        ///
        template <int N, int ORDER, class T>
        void particles_to_grid(const T * part, size_t NumPart, size_t NumPartTot, FFTWGrid<N> & density);

        /// Deposit at virtual positions x + assignment_offset without changing any particle.
        /// Offsets are in box units and must satisfy 0 <= offset[d] < 1/Nmesh. The x coordinate is
        /// left unwrapped for MPI halo deposition; other coordinates wrap periodically. With MPI,
        /// a positive x offset requires one extra right slice in addition to the assignment stencil.
        /// wrap_x may be enabled with one MPI task to wrap x before scaling to grid units (the generalized
        /// interlacing convention). The shifted coordinate is rounded to the particle's position type,
        /// matching an actual shift; this also preserves float-position arithmetic.
        template <int N, class T>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid<N> & density,
                               std::string density_assignment_method,
                               const std::array<double, N> & assignment_offset,
                               bool wrap_x = false);

        template <int N, int ORDER, class T>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid<N> & density,
                               const std::array<double, N> & assignment_offset,
                               bool wrap_x = false);

        namespace detail {
            template <int N, int ORDER, class T, bool SHIFTED>
            void particles_to_grid_impl(const T * part,
                                        size_t NumPart,
                                        size_t NumPartTot,
                                        FFTWGrid<N> & density,
                                        const std::array<double, N> & assignment_offset,
                                        bool wrap_x);
        }

        /// Internal method
        template <int N, class T>
        void particles_to_fourier_grid_interlacing(const T * part,
                                                   size_t NumPart,
                                                   size_t NumPartTot,
                                                   FFTWGrid<N> & density_grid_fourier,
                                                   std::string density_assignment_method);

        /// @brief Assign particles to grid to compute the over density.
        /// Do this for a normal grid and an interlaced grid and return
        /// the alias-corrected fourier transform of the density field in fourier space.
        /// This method does not deconvolve the window function
        ///
        /// @tparam T The particle class. Must have a get_pos() method. If the particle has a get_mass method then this
        /// is used to weight the particle (we assign the particle with weight mass / mean_mass).
        ///
        /// @param[in] part A pointer the first particle.
        /// @param[in] NumPart How many particles/positions we have that we want to interpolate the grid to.
        /// @param[in] NumPartTot How many particles/positions we have in total over all tasks.
        /// @param[out] density_grid_fourier The output density grid in fourier space (must be initialized)
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS).
        /// @param[in] interlacing If true use interlacing to reduce aliasing
        ///
        template <int N, class T>
        void particles_to_fourier_grid(const T * part,
                                       size_t NumPart,
                                       size_t NumPartTot,
                                       FFTWGrid<N> & density_grid_fourier,
                                       std::string density_assignment_method,
                                       bool interlacing);

        /// @brief Convolve a grid with a kernel
        ///
        /// @tparam N The dimension of the grid
        /// @tparam ORDER The width of the kernel in units of the cell-size. We consider the \f$ {\rm ORDER}^{\rm N} \f$
        /// closest grid-cells, so ORDER/2 cells to the left and right in each dimension.
        ///
        /// @param[in] grid_in The grid we want to convolve.
        /// @param[out] grid_out The in grid convolved with the kernel.
        /// @param[in] convolution_kernel A function taking in the distance to a given cell (in units of the cell size)
        /// and returns the kernel. For example the NGP kernel would be the function Prod_i ( |dx[i]| < 0.5 ), i.e. 1 if
        /// all positions are within half a grid cell of the cell center.
        ///
        template <int N, int ORDER>
        void convolve_grid_with_kernel(const FFTWGrid<N> & grid_in,
                                       FFTWGrid<N> & grid_out,
                                       const std::function<FloatType(std::array<double, N> &)> & convolution_kernel);

        //===================================================================================================
        //===================================================================================================

        /// Is the name a known density assignment / interpolation method (NGP, CIC, TSC, PCS, PQS)?
        inline bool is_known_interpolation_method(const std::string & method) {
            return method == "NGP" or method == "CIC" or method == "TSC" or method == "PCS" or method == "PQS";
        }

        template <int N, class T>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid<N> & density,
                               std::string density_assignment_method) {
            assert_mpi(is_known_interpolation_method(density_assignment_method),
                       ("[particles_to_grid] Unknown density assignment method [" + density_assignment_method +
                        "] (NGP, CIC, TSC, PCS or PQS)\n")
                           .c_str());
            if (density_assignment_method.compare("NGP") == 0)
                particles_to_grid<N, 1, T>(part, NumPart, NumPartTot, density);
            if (density_assignment_method.compare("CIC") == 0)
                particles_to_grid<N, 2, T>(part, NumPart, NumPartTot, density);
            if (density_assignment_method.compare("TSC") == 0)
                particles_to_grid<N, 3, T>(part, NumPart, NumPartTot, density);
            if (density_assignment_method.compare("PCS") == 0)
                particles_to_grid<N, 4, T>(part, NumPart, NumPartTot, density);
            if (density_assignment_method.compare("PQS") == 0)
                particles_to_grid<N, 5, T>(part, NumPart, NumPartTot, density);
        }

        template <int N, class T>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid<N> & density,
                               std::string density_assignment_method,
                               const std::array<double, N> & assignment_offset,
                               bool wrap_x) {
            assert_mpi(is_known_interpolation_method(density_assignment_method),
                       ("[particles_to_grid] Unknown density assignment method [" + density_assignment_method +
                        "] (NGP, CIC, TSC, PCS or PQS)\n")
                           .c_str());
            if (density_assignment_method.compare("NGP") == 0)
                particles_to_grid<N, 1, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (density_assignment_method.compare("CIC") == 0)
                particles_to_grid<N, 2, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (density_assignment_method.compare("TSC") == 0)
                particles_to_grid<N, 3, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (density_assignment_method.compare("PCS") == 0)
                particles_to_grid<N, 4, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (density_assignment_method.compare("PQS") == 0)
                particles_to_grid<N, 5, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
        }

        template <int N, class T>
        void interpolate_grid_to_particle_positions(const FFTWGrid<N> & grid,
                                                    const T * part,
                                                    size_t NumPart,
                                                    std::vector<FloatType> & interpolated_values,
                                                    std::string interpolation_method) {
            assert_mpi(is_known_interpolation_method(interpolation_method),
                       ("[interpolate_grid_to_particle_positions] Unknown interpolation method [" +
                        interpolation_method + "] (NGP, CIC, TSC, PCS or PQS)\n")
                           .c_str());
            if (interpolation_method.compare("NGP") == 0)
                interpolate_grid_to_particle_positions<N, 1, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("CIC") == 0)
                interpolate_grid_to_particle_positions<N, 2, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("TSC") == 0)
                interpolate_grid_to_particle_positions<N, 3, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("PCS") == 0)
                interpolate_grid_to_particle_positions<N, 4, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("PQS") == 0)
                interpolate_grid_to_particle_positions<N, 5, T>(grid, part, NumPart, interpolated_values);
        }

        /// @brief Get the interpolation order from a string holding the density_assignment_method (NGP, CIC, ...).
        /// Needed for the Fourier-space window function
        inline int interpolation_order_from_name(std::string density_assignment_method) {
            if (density_assignment_method.compare("NGP") == 0)
                return 1;
            if (density_assignment_method.compare("CIC") == 0)
                return 2;
            if (density_assignment_method.compare("TSC") == 0)
                return 3;
            if (density_assignment_method.compare("PCS") == 0)
                return 4;
            if (density_assignment_method.compare("PQS") == 0)
                return 5;
            assert_mpi(false, "[interpolation_order_from_name] Unknown density assignment method\n");
            return 0;
        }

        /// @brief Compute how many extra slices we need in the FFTWGrid for a given density assignment / interpolation
        /// method.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, TSC, PCS or PQS).
        /// @return The number of left and right slices.
        /// @details
        /// Example usage:
        ///
        /// auto nleftright = get_extra_slices_needed_for_density_assignment("CIC");
        ///
        /// FFTWGrid<N> grid (Nmesh, nleftright.first, nleftright.second);
        ///
        inline std::pair<int, int>
        get_extra_slices_needed_for_density_assignment(std::string density_assignment_method) {
            int p = 0;
            if (density_assignment_method.compare("NGP") == 0)
                p = 1;
            if (density_assignment_method.compare("CIC") == 0)
                p = 2;
            if (density_assignment_method.compare("TSC") == 0)
                p = 3;
            if (density_assignment_method.compare("PCS") == 0)
                p = 4;
            if (density_assignment_method.compare("PQS") == 0)
                p = 5;
            assert_mpi(p > 0, "[extra_slices_needed_density_assignment] Unknown density assignment method\n");
            if (p == 1)
                return {0, 0};
#ifdef CELLCENTERSHIFTED
            if (p % 2 == 1)
                return {p / 2, p / 2};
            if (p % 2 == 0)
                return {p / 2, p / 2};
#else
            if (p % 2 == 1)
                return {p / 2, p / 2 + 1};
            if (p % 2 == 0)
                return {p / 2 - 1, p / 2};
#endif
            return {0, 0};
        }

        /// @brief Compute how many extra slices we need in the FFTWGrid for a given density assignment order.
        /// @tparam ORDER The order of the B-spline assignment kernel (NGP=1, CIC=2, TSC=3, PCS=4, PQS=5, ...)
        /// @return The number of left and right slices.
        ///
        template <int ORDER>
        inline std::pair<int, int> get_extra_slices_needed_by_order() {
            if (ORDER == 1)
                return {0, 0};
#ifdef CELLCENTERSHIFTED
            if (ORDER % 2 == 1)
                return {ORDER / 2, ORDER / 2};
            if (ORDER % 2 == 0)
                return {ORDER / 2, ORDER / 2};
#else
            if (ORDER % 2 == 1)
                return {ORDER / 2, ORDER / 2 + 1};
            if (ORDER % 2 == 0)
                return {ORDER / 2 - 1, ORDER / 2};
#endif
            return {0, 0};
        }

        /// @brief Internal method. The B-spline interpolation kernels for a given order
        /// \f$ H^{(p)} = H * H * \ldots * H \f$ where H is the tophat \f$ H = [ |dx| < 0.5 ? 1 : 0 ] \f$
        /// and * is a convolution (easily computed with Mathematica)
        ///
        template <int ORDER>
        inline double kernel([[maybe_unused]] double x) {
            static_assert(ORDER > 0 and ORDER <= 5, "Error: kernel order is not implemented\n");
            return 0.0 / 0.0;
        }
        /// @brief The NGP kernel
        template <>
        inline double kernel<1>(double x) {
            return (x <= 0.5) ? 1.0 : 0.0;
        }
        /// @brief The CIC kernel
        template <>
        inline double kernel<2>(double x) {
            return (x < 1.0) ? 1.0 - x : 0.0;
        }
        /// @brief The TSC kernel
        template <>
        inline double kernel<3>(double x) {
            return (x < 0.5) ? 0.75 - x * x : (x < 1.5 ? 0.5 * (1.5 - x) * (1.5 - x) : 0.0);
        }
        /// @brief The PCS kernel
        template <>
        inline double kernel<4>(double x) {
            return (x < 1.0) ? 2.0 / 3.0 + x * x * (-1.0 + 0.5 * x) :
                               ((x < 2.0) ? (2 - x) * (2 - x) * (2 - x) / 6.0 : 0.0);
        }
        /// @brief The PQS kernel
        template <>
        inline double kernel<5>(double x) {
            return (x < 0.5) ?
                       115.0 / 192.0 + 0.25 * x * x * (x * x - 2.5) :
                       ((x < 1.5) ?
                            (55 + 4 * x * (5 - 2 * x * (15 + 2 * (-5 + x) * x))) / 96.0 :
                            ((x < 2.5) ? (5 - 2.0 * x) * (5 - 2.0 * x) * (5 - 2.0 * x) * (5 - 2.0 * x) / 384. : 0.0));
        }

        /// @brief Internal method. For communication between tasks needed when adding particles to grid
        template <int N>
        void add_contribution_from_extra_slices(FFTWGrid<N> & density, double background = -1.0);

        /// @brief This returns the a function giving the window function for a given density assignment method as
        /// function of the wave-vector in dimensionless units.
        /// @tparam N The dimension of the grid
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, ...) we used when making the
        /// density contrast.
        /// @param[in] Ngrid The grid size (used to set the nyquist frequency)
        ///
        template <int N>
        std::function<double(std::array<double, N> &)> get_window_function(std::string density_assignment_method,
                                                                           int Ngrid) {

            assert_mpi(Ngrid > 0, "[get_window_function_fourier] Ngrid must be positive\n");

            // The order of the method
            const int p = interpolation_order_from_name(density_assignment_method);

            // Just sinc to the power = order to the method
            const double knyquist = M_PI * Ngrid;
            std::function<double(std::array<double, N> &)> window_function = [=](std::array<double, N> & kvec) {
                double w = 1.0;
                for (int idim = 0; idim < N; idim++) {
                    const double koverkny = M_PI / 2. * (kvec[idim] / knyquist);
                    w *= koverkny == 0.0 ? 1.0 : std::sin(koverkny) / (koverkny);
                }
                // res = pow(w,p);
                double res = 1;
                for (int i = 0; i < p; i++)
                    res *= w;
                return res;
            };

            return window_function;
        }

        /// @brief The Fourier window of the density assignment, \f$ W(k) = \prod_i {\rm sinc}(\pi k_i / 2k_{\rm ny})^p
        /// \f$, from a table of the 1D factors (the wavevector components of a grid are \f$ 2\pi n \f$ with integer n).
        /// It does the same operations as get_window_function, so the values are identical, without the N sines per
        /// mode and the std::function call. Only for wavevectors of a grid of size Ngrid.
        template <int N>
        class WindowFunctionTable {
          public:
            WindowFunctionTable(std::string density_assignment_method, int Ngrid)
                : p(interpolation_order_from_name(density_assignment_method)), Ngrid(Ngrid), table(Ngrid + 1) {
                assert_mpi(Ngrid > 0, "[WindowFunctionTable] Ngrid must be positive\n");
                const double twopi = 2.0 * M_PI;
                const double knyquist = M_PI * Ngrid;
                for (int n = -Ngrid / 2; n <= Ngrid / 2; n++) {
                    const double k = twopi * double(n);
                    const double koverkny = M_PI / 2. * (k / knyquist);
                    table[n + Ngrid / 2] = koverkny == 0.0 ? 1.0 : std::sin(koverkny) / (koverkny);
                }
            }
            double operator()(const std::array<double, N> & kvec) const {
                double w = 1.0;
                for (int idim = 0; idim < N; idim++) {
                    // kvec = 2 pi n: round to the nearest integer (inline, std::lround is a library call)
                    const double n = kvec[idim] * (0.5 / M_PI);
                    w *= table[int(n + (n >= 0.0 ? 0.5 : -0.5)) + Ngrid / 2];
                }
                double res = 1;
                for (int i = 0; i < p; i++)
                    res *= w;
                return res;
            }

          private:
            int p;
            int Ngrid;
            std::vector<double> table;
        };

        /// @brief Deconvolves the density assignment kernel in Fourier space. We divide the fourier grid by the
        /// FFT of the density assignment kernels \f$ FFT[ H*H*H*...*H ] = FT[H]^p\f$.
        /// @tparam N The dimension of the grid
        /// @param[out] fourier_grid The Fourier grid of the density contrast that we will deconvolve.
        /// @param[in] density_assignment_method The density assignment method (NGP, CIC, ...) we used when making the
        /// density contrast.
        ///
        template <int N>
        void deconvolve_window_function_fourier(FFTWGrid<N> & fourier_grid, std::string density_assignment_method) {

            const auto Ngrid = fourier_grid.get_nmesh();
            const auto Local_nx = fourier_grid.get_local_nx();

            assert_mpi(Ngrid > 0, "[deconvolve_window_function_fourier] Ngrid must be positive\n");

            const WindowFunctionTable<N> window_function(density_assignment_method, Ngrid);

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && fourier_index : fourier_grid.get_fourier_range(islice, islice + 1)) {
                    auto kvec = fourier_grid.get_fourier_wavevector_from_index(fourier_index);
                    auto w = window_function(kvec);
                    auto value = fourier_grid.get_fourier_from_index(fourier_index);
                    fourier_grid.set_fourier_from_index(fourier_index, value / FML::GRID::FloatType(w));
                }
            }
        }

        //==============================================================================
        // Bin particles to grid using NGP, CIC, TSC, PCS, PQS, ...
        // Some of the methods require extra slices, see
        // get_extra_slices_needed_for_density_assignment
        //
        // NumPart: number of particles at the head of part
        // NumPartTot: total number of particles across tasks
        // NB: part.size() might not be equal to NumPart as we might have a buffer with
        //
        // Particles with get_mass are weighted by their mass (normalized by the mean mass), delta-f particles
        // (get_deltaf_weight) by their weight
        //==============================================================================

        template <int N, int ORDER, class T>
        void particles_to_grid(const T * part, size_t NumPart, size_t NumPartTot, FFTWGrid<N> & density) {
            detail::particles_to_grid_impl<N, ORDER, T, false>(part, NumPart, NumPartTot, density, {}, false);
        }

        template <int N, int ORDER, class T>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               FFTWGrid<N> & density,
                               const std::array<double, N> & assignment_offset,
                               bool wrap_x) {
            assert_mpi(not wrap_x or FML::NTasks == 1, "[particles_to_grid] Wrapped x offsets require one task");
            const int Nmesh = density.get_nmesh();
            assert_mpi(Nmesh > 0, "[particles_to_grid] Offset deposition needs an allocated grid");
            bool shifted = false;
            for (double offset : assignment_offset) {
                assert_mpi(std::isfinite(offset) and offset >= 0.0 and offset < 1.0 / Nmesh,
                           "[particles_to_grid] Assignment offsets must be in [0, 1/Nmesh)");
                shifted = shifted or offset != 0.0;
            }
            const auto nextra = get_extra_slices_needed_by_order<ORDER>();
            assert_mpi(FML::NTasks == 1 or assignment_offset[0] == 0.0 or
                           density.get_n_extra_slices_right() >= nextra.second + 1,
                       "[particles_to_grid] Positive x offset needs one additional right slice");
            if (shifted)
                detail::particles_to_grid_impl<N, ORDER, T, true>(part,
                                                                  NumPart,
                                                                  NumPartTot,
                                                                  density,
                                                                  assignment_offset,
                                                                  wrap_x);
            else
                particles_to_grid<N, ORDER, T>(part, NumPart, NumPartTot, density);
        }

        namespace detail {
            // Match the rounding of a stored position += offset, but only modify a local value.
            template <class Position>
            inline Position assignment_coordinate(Position coordinate, double offset, int dimension, bool wrap_x) {
                coordinate += offset;
                if (dimension > 0 or wrap_x)
                    coordinate = FML::PARTICLE::wrap_coordinate_periodic(coordinate);
                return coordinate;
            }
        } // namespace detail

        template <int N, int ORDER, class T, bool SHIFTED>
        void detail::particles_to_grid_impl(const T * part,
                                            size_t NumPart,
                                            size_t NumPartTot,
                                            FFTWGrid<N> & density,
                                            const std::array<double, N> & assignment_offset,
                                            bool wrap_x) {

            const auto nextra = get_extra_slices_needed_by_order<ORDER>();
            assert_mpi(density.get_n_extra_slices_left() >= nextra.first and
                           density.get_n_extra_slices_right() >= nextra.second,
                       "[particles_to_grid] Too few extra slices\n");

            //==========================================================
            // This is a generic method. You have to specify the kernel
            // and the corresponding width = the number of cells
            // the point gets distributed to in each dimension which
            // also corresponds to the order
            //==========================================================

            // For the kernel above we need to go kernel_width/2 cells to the left and right
            constexpr int widthtondim = FML::power(ORDER, N);

            // Info about the grid
            const auto Local_nx = density.get_local_nx();
            const auto Local_x_start = density.get_local_x_start();
            const int Nmesh = density.get_nmesh();
            // Particles may be up to spare_right cells to the right of the local domain if the grid has more extra
            // slices than the kernel needs (e.g. the half-cell shifted particles for interlacing)
            const int spare_right = density.get_n_extra_slices_right() - nextra.second;
            assert_mpi(FML::NTasks == 1 or Local_nx >= std::max(nextra.first, nextra.second + std::max(0, spare_right)),
                       "[particles_to_grid] Fewer slices on a task than extra slices needed by the kernel: the "
                       "contributions to the extra slices would reach beyond the neighbor task (use fewer tasks or a "
                       "larger grid)\n");

            // delta-f particles (get_deltaf_weight) only carry the perturbation: delta = norm * sum w_i W
            constexpr bool has_deltaf_weight = FML::PARTICLE::has_get_deltaf_weight<T>();

            // Set whole grid (also extra slices) to -1.0 (0 for delta-f particles)
            density.fill_real_grid(has_deltaf_weight ? 0.0 : -1.0);

            // Factor to normalize density to the mean density
            double norm_fac = std::pow((double)Nmesh, N) / double(NumPartTot);

            // Check if particles has a get_mass method and if so
            // compute the mean mass
            constexpr bool has_mass = FML::PARTICLE::has_get_mass<T>() and not has_deltaf_weight;
            static_assert(not(FML::PARTICLE::has_get_mass<T>() and has_deltaf_weight),
                          "[particles_to_grid] A particle cannot have both get_mass and get_deltaf_weight");
            if constexpr (has_mass) {
                double mean_mass = 0.0;
                for (size_t i = 0; i < NumPart; i++) {
                    mean_mass += FML::PARTICLE::GetMass(part[i]);
                }
                SumOverTasks(&mean_mass);
                mean_mass /= double(NumPartTot);
                norm_fac /= mean_mass;
            }

            // Add one particle to the grid. ATOMIC: atomic adds (a slow compare-and-swap loop for doubles), needed when
            // several threads can write to the same cell
            auto * density_raw = density.get_real_grid();
            auto add_particle = [&](size_t i, auto atomic_tag) {
                constexpr bool ATOMIC = decltype(atomic_tag)::value;

                // Particle position
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[i]);

                // Fetch mass if this is available
                double mass = 1.0;
                if constexpr (has_mass)
                    mass = FML::PARTICLE::GetMass(part[i]);
                if constexpr (has_deltaf_weight)
                    mass = FML::PARTICLE::GetDeltafWeight(part[i]);

                std::array<double, N> x;
                std::array<int, N> ix;
                [[maybe_unused]] std::array<int, N> ix_nbor;
                for (int idim = 0; idim < N; idim++) {
                    // Scale positions to be in [0, Nmesh]
                    if constexpr (SHIFTED)
                        x[idim] = assignment_coordinate(pos[idim], assignment_offset[idim], idim, wrap_x) * Nmesh;
                    else
                        x[idim] = pos[idim] * Nmesh;
                    // Grid-index for cell containing particle
                    ix[idim] = (int)x[idim];
                    // Distance relative to cell
                    x[idim] -= ix[idim];
                }

                // Periodic BC
                ix[0] -= int(Local_x_start);
                // NGP uses the cell ix directly: with one task (no extra slices to fold back) wrap x too (e.g. the
                // half-cell shifted particles for interlacing have x up to 1 + 1/(2Nmesh))
                if constexpr (ORDER == 1)
                    if (FML::NTasks == 1 and ix[0] >= Nmesh)
                        ix[0] -= Nmesh;
                // A particle in the local domain can be in the cell just outside it from round-off: xmin_domain =
                // ThisTask / NTasks is not exactly Local_x_start / Nmesh (e.g. a particle at x = 63/90 has
                // int(x * 90) = 62 with 10 tasks). Use the nearest cell in the domain (as in the interpolation)
                if (FML::NTasks > 1) {
                    if (ix[0] < 0) {
                        x[0] += ix[0];
                        ix[0] = 0;
                    } else if (ix[0] > int(Local_nx) - 1 + spare_right) {
                        x[0] += ix[0] - (int(Local_nx) - 1 + spare_right);
                        ix[0] = int(Local_nx) - 1 + spare_right;
                    }
                }
                for (int idim = 1; idim < N; idim++) {
                    if (ix[idim] == Nmesh)
                        ix[idim] = 0;
                }

                // If we are on the left or right of the cell determines how many cells
                // we have to go left and right
                std::array<int, N> xstart;
                if (ORDER % 2 == 0) {
                    for (int idim = 0; idim < N; idim++) {
                        xstart[idim] = -ORDER / 2 + 1;
#ifdef CELLCENTERSHIFTED
                        xstart[idim] = -ORDER / 2;
                        if (x[idim] > 0.5)
                            xstart[idim] += 1;
#endif
                    }
                } else {
#ifndef CELLCENTERSHIFTED
                    for (int idim = 0; idim < N; idim++) {
                        xstart[idim] = -ORDER / 2;
                        if (x[idim] > 0.5)
                            xstart[idim] += 1;
                    }
#else
                    // Nodes at the cell centers: the nearest node is the particle's own cell
                    for (int idim = 0; idim < N; idim++)
                        xstart[idim] = -ORDER / 2;
#endif
                }

                // The kernel weights and (wrapped) cell indices along each dimension. The weight of a cell is the
                // product of these (multiplied in the order idim = 0, 1, ..., as before, so the result is identical)
                [[maybe_unused]] std::array<std::array<double, ORDER>, N> w1d;
                [[maybe_unused]] std::array<std::array<int, ORDER>, N> i1d;
                if constexpr (ORDER > 1) {
                    for (int idim = 0; idim < N; idim++) {
                        for (int j = 0; j < ORDER; j++) {
                            const int go_left_right_or_stay = xstart[idim] + j;
#ifdef CELLCENTERSHIFTED
                            const double dx = std::fabs(-x[idim] + go_left_right_or_stay + 0.5);
#else
                            const double dx = std::fabs(-x[idim] + go_left_right_or_stay);
#endif
                            w1d[idim][j] = kernel<ORDER>(dx);
                            // Periodic BC for all but x (we have extra slices - XXX should assert that its not too
                            // large, but covered by boundscheck in FFTWGrid if this is turned on)! If only 1 task then
                            // we should also wrap x
                            int icoord = ix[idim] + go_left_right_or_stay;
                            if (idim > 0 or FML::NTasks == 1) {
                                if (icoord >= Nmesh)
                                    icoord -= Nmesh;
                                if (icoord < 0)
                                    icoord += Nmesh;
                            }
                            i1d[idim][j] = icoord;
                        }
                    }
                }

                // Loop over all nbor cells
                [[maybe_unused]] double sumweights = 0.0;
                for (int i = 0; i < widthtondim; i++) {
                    double w = 1.0;
                    std::array<int, N> icoord;
                    if constexpr (ORDER == 1) {
                        icoord = ix;
                    } else {
                        for (int idim = 0, n = 1; idim < N; idim++, n *= ORDER) {
                            const int j = i / n % ORDER;
                            w *= w1d[idim][j];
                            icoord[idim] = i1d[idim][j];
                        }
                    }

                    // Add particle to grid
                    // Old version when we did not use OMP: density.add_real(icoord, w * norm_fac * mass);
                    auto index = density.get_index_real(icoord);
                    double mass_to_add = w * norm_fac * mass;
                    if constexpr (ATOMIC) {
#ifdef USE_OMP
#pragma omp atomic
#endif
                        density_raw[index] += mass_to_add;
                    } else {
                        density_raw[index] += mass_to_add;
                    }

                    // Sum up and unsure weights sum to unity
                    sumweights += w;
                }

#ifdef DEBUG_INTERPOL
                // Check that the weights sum up to unity
                assert_mpi(
                    std::fabs(sumweights - 1.0) < 1e-3,
                    "[particles_to_grid] Possible problem with particles to grid: weights does not sum to unity!");
#endif
            };

            //==========================================================
            // One thread: the particles in order (the results do not depend on how the code is compiled)
            //
            // Several threads, without atomics: the local x-range is split into nchunks chunks of at least
            // 2 ORDER + 2 slices (so the cells written by particles in chunks c and c+2 never overlap; nchunks is even
            // with one task as the first and last chunk are then neighbors). The particles of the even chunks are added
            // in parallel, then those of the odd chunks, each chunk in the order of the particles. Every cell then gets
            // its contributions in the same order for any number of threads > 1 (deterministic; differs from the
            // one-thread result only by round-off from the order of the sums). The particles are not moved, only
            // their indices are grouped by chunk (4 bytes per particle). With too few chunks (thin slabs with many
            // tasks: fewer than 4 (2 ORDER + 2) slices per task, e.g. Nmesh 512 on 32 tasks) we fall back to atomic
            // adds, and the result then depends on the order of the additions (not bit-identical between runs).
            //==========================================================
#ifdef USE_OMP
            const int nthreads = omp_get_max_threads();
#else
            const int nthreads = 1;
#endif
            if (nthreads == 1) {
                for (size_t i = 0; i < NumPart; i++)
                    add_particle(i, std::false_type{});
            } else {
                const int Local_nx = density.get_local_nx();
                int nchunks = Local_nx / (2 * ORDER + 2);
                if (FML::NTasks == 1 and nchunks % 2 == 1)
                    nchunks--;
                const bool chunked = nchunks >= 4 and NumPart < size_t(std::numeric_limits<uint32_t>::max());
                if (not chunked) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < NumPart; i++)
                        add_particle(i, std::true_type{});
                } else {
                    // The chunk of each particle (from the x-slice it is in, clamped to the local range)
                    auto chunk_of = [&](size_t i) {
                        const auto * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[i]);
                        auto coordinate = pos[0];
                        if constexpr (SHIFTED)
                            coordinate = assignment_coordinate(coordinate, assignment_offset[0], 0, wrap_x);
                        int ix0 = int(coordinate * Nmesh) - int(Local_x_start);
                        ix0 = std::max(0, std::min(ix0, Local_nx - 1));
                        return int((long long)(ix0)*nchunks / Local_nx);
                    };
                    // Counting sort of the particle indices by chunk (stable: each chunk in particle order)
                    std::vector<size_t> start(nchunks + 1, 0);
                    std::vector<uint32_t> chunk(NumPart);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (size_t i = 0; i < NumPart; i++)
                        chunk[i] = uint32_t(chunk_of(i));
                    for (size_t i = 0; i < NumPart; i++)
                        start[chunk[i] + 1]++;
                    for (int c = 0; c < nchunks; c++)
                        start[c + 1] += start[c];
                    std::vector<uint32_t> order(NumPart);
                    {
                        std::vector<size_t> next(start.begin(), start.end() - 1);
                        for (size_t i = 0; i < NumPart; i++)
                            order[next[chunk[i]]++] = uint32_t(i);
                    }
                    std::vector<uint32_t>().swap(chunk);
                    for (int phase = 0; phase < 2; phase++) {
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                        for (int c = phase; c < nchunks; c += 2)
                            for (size_t k = start[c]; k < start[c + 1]; k++)
                                add_particle(order[k], std::false_type{});
                    }
                }
            }

            // Extra slices only relevant if we have more than 1 task
            if (FML::NTasks > 1)
                add_contribution_from_extra_slices<N>(density, has_deltaf_weight ? 0.0 : -1.0);
        }

        template <int N, int ORDER, class T>
        void
        interpolate_grid_vector_to_particle_positions(const std::array<FFTWGrid<N>, N> & grid_vec,
                                                      const T * part,
                                                      size_t NumPart,
                                                      std::array<std::vector<FloatType>, N> & interpolated_values_vec) {
            for (auto & values : interpolated_values_vec)
                if (values.size() < NumPart)
                    values.resize(NumPart);
            auto store = [&](size_t index, const std::array<FloatType, N> & value) {
                for (int dimension = 0; dimension < N; dimension++)
                    interpolated_values_vec[dimension][index] = value[dimension];
                return 0.0;
            };
            interpolate_grid_vector_to_particle_positions_apply<N, ORDER, T>(grid_vec, part, NumPart, store);
        }

        template <int N, int ORDER, class T, class Apply>
        double interpolate_grid_vector_to_particle_positions_apply(const std::array<FFTWGrid<N>, N> & grid_vec,
                                                                   const T * part,
                                                                   size_t NumPart,
                                                                   Apply && apply) {

            auto nextra = get_extra_slices_needed_by_order<ORDER>();
            assert_mpi(grid_vec.size() > 0,
                       "[interpolate_grid_to_particle_positions] Grid vector has to be already allocated!\n");
            for (auto & g : grid_vec) {
                assert_mpi(g.get_nmesh() > 0,
                           "[interpolate_grid_to_particle_positions] All grids has to be already allocated!\n");
                assert_mpi(g.get_nmesh() == grid_vec[0].get_nmesh(),
                           "[interpolate_grid_to_particle_positions] All grids has to have the same size!\n");
            }
            for (auto & g : grid_vec) {
                assert_mpi(g.get_n_extra_slices_left() >= nextra.first and
                               g.get_n_extra_slices_right() >= nextra.second,
                           "[interpolate_grid_to_particle_positions] Too few extra slices in some of the grids\n");
            }

            // We need to look at width^N cells in total
            constexpr int widthtondim = FML::power(ORDER, N);

            // Fetch grid information
            const auto Local_nx = grid_vec[0].get_local_nx();
            const auto Local_x_start = grid_vec[0].get_local_x_start();
            const int Nmesh = grid_vec[0].get_nmesh();
            const int spare_right = std::max(0, grid_vec[0].get_n_extra_slices_right() - nextra.second);
            assert_mpi(FML::NTasks == 1 or Local_nx >= std::max(nextra.first, nextra.second + spare_right),
                       "[interpolate_grid_to_particle_positions] Fewer slices on a task than extra slices needed by "
                       "the kernel: the extra slices cannot be filled from the neighbor task (use fewer tasks or a "
                       "larger grid)\n");

            double max_result = 0.0;

#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_result)
#endif
            for (size_t ind = 0; ind < NumPart; ind++) {

                // Positions in global grid in units of [Nmesh]
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[ind]);
                std::array<double, N> x;
                for (int idim = 0; idim < N; idim++)
                    x[idim] = pos[idim] * Nmesh;

                // Nearest grid-node in grid
                // Also do some sanity checks. Probably better to throw here if these tests kick in
                std::array<int, N> ix;
                for (int idim = 0; idim < N; idim++) {
                    ix[idim] = int(x[idim]);
                    if (idim == 0) {
                        // Particles may be up to spare_right cells outside the local domain if the grid has
                        // more extra slices than the kernel needs (e.g. when the particles have been shifted)
                        if (ix[0] >= (Local_x_start + Local_nx + spare_right))
                            ix[0] = int(Local_x_start + Local_nx + spare_right) - 1;
                        if (ix[0] < Local_x_start)
                            ix[0] = int(Local_x_start);
                    } else {
                        if (ix[idim] == Nmesh)
                            ix[idim] = Nmesh - 1;
                    }
                }

                // Positions to distance from neareste grid-node
                for (int idim = 0; idim < N; idim++) {
                    x[idim] -= ix[idim];
                }

                // From global ix to local ix
                ix[0] -= int(Local_x_start);

                // If we are on the left or right of the cell determines how many cells
                // we have to go left and right
                std::array<int, N> xstart;
                if (ORDER % 2 == 0) {
                    for (int idim = 0; idim < N; idim++) {
                        xstart[idim] = -ORDER / 2 + 1;
#ifdef CELLCENTERSHIFTED
                        xstart[idim] = -ORDER / 2;
                        if (x[idim] > 0.5)
                            xstart[idim] += 1;
#endif
                    }
                } else {
#ifndef CELLCENTERSHIFTED
                    for (int idim = 0; idim < N; idim++) {
                        xstart[idim] = -ORDER / 2;
                        if (x[idim] > 0.5)
                            xstart[idim] += 1;
                    }
#else
                    // Nodes at the cell centers: the nearest node is the particle's own cell
                    for (int idim = 0; idim < N; idim++)
                        xstart[idim] = -ORDER / 2;
#endif
                }

                // The kernel weights and cell indices along each dimension (periodic in all but x). The weight of a
                // cell is their product, multiplied in the order idim = 0, 1, ... as before (identical results)
                std::array<std::array<double, ORDER>, N> w1d;
                std::array<std::array<int, ORDER>, N> i1d;
                for (int idim = 0; idim < N; idim++) {
                    for (int j = 0; j < ORDER; j++) {
                        const int go_left_right_or_stay = ORDER == 1 ? 0 : xstart[idim] + j;
#ifdef CELLCENTERSHIFTED
                        const double dx = std::fabs(-x[idim] + go_left_right_or_stay + 0.5);
#else
                        const double dx = std::fabs(-x[idim] + go_left_right_or_stay);
#endif
                        // (NGP: the particle's own cell, weight 1 as in the scalar version; kernel<1>(dx) is 0 for
                        // dx > 0.5, i.e. for particles in the right half of their cell)
                        w1d[idim][j] = ORDER == 1 ? 1.0 : kernel<ORDER>(dx);
                        int icoord = ix[idim] + go_left_right_or_stay;
                        if (idim > 0) {
                            if (icoord >= Nmesh)
                                icoord -= Nmesh;
                            if (icoord < 0)
                                icoord += Nmesh;
                        }
                        i1d[idim][j] = icoord;
                    }
                }

                // Interpolation
                std::array<double, N> value;
                value.fill(0.0);
                [[maybe_unused]] double sumweight = 0;
                for (int i = 0; i < widthtondim; i++) {
                    double w = 1.0;
                    std::array<int, N> icoord;
                    for (int idim = 0, n = 1; idim < N; idim++, n *= ORDER) {
                        const int j = i / n % ORDER;
                        w *= w1d[idim][j];
                        icoord[idim] = i1d[idim][j];
                    }

                    // Add up
                    for (int idim = 0; idim < N; idim++)
                        value[idim] += grid_vec[idim].get_real(icoord) * w;
                    sumweight += w;
                }

#ifdef DEBUG_INTERPOL
                // Check that the weights sum up to unity
                assert_mpi(std::fabs(sumweight - 1.0) < 1e-3,
                           "[interpolate_grid_to_particle_positions] Possible problem with interpolation: weights does "
                           "not sum to unity!");
#endif

                // Preserve the precision boundary of the original buffered gather before applying an update.
                std::array<FloatType, N> interpolated_value;
                for (int idim = 0; idim < N; idim++)
                    interpolated_value[idim] = value[idim];
                max_result = std::max(max_result, apply(ind, interpolated_value));
            }
            return max_result;
        }

        template <int N, class T>
        void interpolate_grid_vector_to_particle_positions(const std::array<FFTWGrid<N>, N> & grid,
                                                           const T * part,
                                                           size_t NumPart,
                                                           std::array<std::vector<FloatType>, N> & interpolated_values,
                                                           std::string interpolation_method) {
            assert_mpi(is_known_interpolation_method(interpolation_method),
                       ("[interpolate_grid_vector_to_particle_positions] Unknown interpolation method [" +
                        interpolation_method + "] (NGP, CIC, TSC, PCS or PQS)\n")
                           .c_str());
            if (interpolation_method.compare("NGP") == 0)
                interpolate_grid_vector_to_particle_positions<N, 1, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("CIC") == 0)
                interpolate_grid_vector_to_particle_positions<N, 2, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("TSC") == 0)
                interpolate_grid_vector_to_particle_positions<N, 3, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("PCS") == 0)
                interpolate_grid_vector_to_particle_positions<N, 4, T>(grid, part, NumPart, interpolated_values);
            if (interpolation_method.compare("PQS") == 0)
                interpolate_grid_vector_to_particle_positions<N, 5, T>(grid, part, NumPart, interpolated_values);
        }

        template <int N, class T, class Apply>
        double interpolate_grid_vector_to_particle_positions_apply(const std::array<FFTWGrid<N>, N> & grid,
                                                                   const T * part,
                                                                   size_t NumPart,
                                                                   std::string interpolation_method,
                                                                   Apply && apply) {
            assert_mpi(is_known_interpolation_method(interpolation_method),
                       ("[interpolate_grid_vector_to_particle_positions_apply] Unknown interpolation method [" +
                        interpolation_method + "] (NGP, CIC, TSC, PCS or PQS)\n")
                           .c_str());
            if (interpolation_method == "NGP")
                return interpolate_grid_vector_to_particle_positions_apply<N, 1, T>(grid, part, NumPart, apply);
            if (interpolation_method == "CIC")
                return interpolate_grid_vector_to_particle_positions_apply<N, 2, T>(grid, part, NumPart, apply);
            if (interpolation_method == "TSC")
                return interpolate_grid_vector_to_particle_positions_apply<N, 3, T>(grid, part, NumPart, apply);
            if (interpolation_method == "PCS")
                return interpolate_grid_vector_to_particle_positions_apply<N, 4, T>(grid, part, NumPart, apply);
            return interpolate_grid_vector_to_particle_positions_apply<N, 5, T>(grid, part, NumPart, apply);
        }

        template <int N, int ORDER, class T>
        void interpolate_grid_to_particle_positions(const FFTWGrid<N> & grid,
                                                    const T * part,
                                                    size_t NumPart,
                                                    std::vector<FloatType> & interpolated_values) {

            auto nextra = get_extra_slices_needed_by_order<ORDER>();
            assert_mpi(grid.get_nmesh() > 0,
                       "[interpolate_grid_to_particle_positions] Grid has to be already allocated!\n");
            assert_mpi(grid.get_n_extra_slices_left() >= nextra.first and
                           grid.get_n_extra_slices_right() >= nextra.second,
                       "[interpolate_grid_to_particle_positions] Too few extra slices\n");

            // We need to look at width^N cells in total
            constexpr int widthtondim = FML::power(ORDER, N);

            // Fetch grid information
            const auto Local_nx = grid.get_local_nx();
            const auto Local_x_start = grid.get_local_x_start();
            const int Nmesh = grid.get_nmesh();
            const int spare_right = std::max(0, grid.get_n_extra_slices_right() - nextra.second);
            assert_mpi(FML::NTasks == 1 or Local_nx >= std::max(nextra.first, nextra.second + spare_right),
                       "[interpolate_grid_to_particle_positions] Fewer slices on a task than extra slices needed by "
                       "the kernel: the extra slices cannot be filled from the neighbor task (use fewer tasks or a "
                       "larger grid)\n");

            // Allocate memory needed
            interpolated_values.resize(NumPart);

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t ind = 0; ind < NumPart; ind++) {

                // Positions in global grid in units of [Nmesh]
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[ind]);
                std::array<double, N> x;
                for (int idim = 0; idim < N; idim++)
                    x[idim] = pos[idim] * Nmesh;

                // Nearest grid-node in grid
                // Also do some sanity checks. Probably better to throw here if these tests kick in
                std::array<int, N> ix;
                [[maybe_unused]] std::array<int, N> ix_nbor;
                for (int idim = 0; idim < N; idim++) {
                    ix[idim] = int(x[idim]);
                    if (idim == 0) {
                        // Particles may be up to spare_right cells outside the local domain if the grid has
                        // more extra slices than the kernel needs (e.g. when the particles have been shifted)
                        if (ix[0] >= (Local_x_start + Local_nx + spare_right))
                            ix[0] = int(Local_x_start + Local_nx + spare_right) - 1;
                        if (ix[0] < Local_x_start)
                            ix[0] = int(Local_x_start);
                    } else {
                        if (ix[idim] == Nmesh)
                            ix[idim] = Nmesh - 1;
                    }
                }

                // Positions to distance from neareste grid-node
                for (int idim = 0; idim < N; idim++) {
                    x[idim] -= ix[idim];
                }

                // From global ix to local ix
                ix[0] -= int(Local_x_start);

                // If we are on the left or right of the cell determines how many cells
                // we have to go left and right
                std::array<int, N> xstart;
                if (ORDER % 2 == 0) {
                    for (int idim = 0; idim < N; idim++) {
                        xstart[idim] = -ORDER / 2 + 1;
#ifdef CELLCENTERSHIFTED
                        xstart[idim] = -ORDER / 2;
                        if (x[idim] > 0.5)
                            xstart[idim] += 1;
#endif
                    }
                } else {
#ifndef CELLCENTERSHIFTED
                    for (int idim = 0; idim < N; idim++) {
                        xstart[idim] = -ORDER / 2;
                        if (x[idim] > 0.5)
                            xstart[idim] += 1;
                    }
#else
                    // Nodes at the cell centers: the nearest node is the particle's own cell
                    for (int idim = 0; idim < N; idim++)
                        xstart[idim] = -ORDER / 2;
#endif
                }

                // Interpolation
                FloatType value = 0;
                [[maybe_unused]] double sumweight = 0;
                for (int i = 0; i < widthtondim; i++) {
                    double w = 1.0;
                    std::array<int, N> icoord;
                    if constexpr (ORDER == 1) {
                        icoord = ix;
                    } else {
                        for (int idim = 0, n = 1; idim < N; idim++, n *= ORDER) {
                            int go_left_right_or_stay = xstart[idim] + (i / n % ORDER);
                            ix_nbor[idim] = ix[idim] + go_left_right_or_stay;
#ifdef CELLCENTERSHIFTED
                            double dx = std::fabs(-x[idim] + go_left_right_or_stay + 0.5);
#else
                            double dx = std::fabs(-x[idim] + go_left_right_or_stay);
#endif
                            w *= kernel<ORDER>(dx);
                        }

                        // Periodic BC
                        icoord[0] = ix_nbor[0];
                        for (int idim = 1; idim < N; idim++) {
                            icoord[idim] = ix_nbor[idim];
                            if (icoord[idim] >= Nmesh)
                                icoord[idim] -= Nmesh;
                            if (icoord[idim] < 0)
                                icoord[idim] += Nmesh;
                        }
                    }

                    // Add up
                    value += grid.get_real(icoord) * w;
                    sumweight += w;
                }

#ifdef DEBUG_INTERPOL
                // Check that the weights sum up to unity
                assert_mpi(std::fabs(sumweight - 1.0) < 1e-3,
                           "[interpolate_grid_to_particle_positions] Possible problem with interpolation: weights does "
                           "not sum to unity!");
#endif

                // Store the interpolated value
                interpolated_values[ind] = value;
            }
        }

        //=======================================================================
        // Communicate what we have added to the extra slices that belong
        // on neighbor tasks
        //=======================================================================
        template <int N>
        void add_contribution_from_extra_slices(FFTWGrid<N> & density, double background) {
            // The extra slices were initialized to the background value (-1 for delta, 0 for delta-f particles)

            auto Local_nx = density.get_local_nx();
            auto num_cells_slice = density.get_ntot_real_slice_alloc();
            int n_extra_left = density.get_n_extra_slices_left();
            int n_extra_right = density.get_n_extra_slices_right();
            ;

            std::vector<FloatType> buffer(num_cells_slice);

            // [1] Send to the right, receive from left
            for (int i = 0; i < n_extra_right; i++) {
                FloatType * extra_slice_right = density.get_real_grid_right() + num_cells_slice * i;
                FloatType * slice_left = density.get_real_grid() + num_cells_slice * i;
                FloatType * temp = buffer.data();

#ifdef USE_MPI
                MPI_Status status;

                int send_to = (ThisTask + 1) % NTasks;
                int recv_from = (ThisTask - 1 + NTasks) % NTasks;

                MPI_Sendrecv(&(extra_slice_right[0]),
                             int(sizeof(FloatType) * num_cells_slice),
                             MPI_CHAR,
                             send_to,
                             0,
                             &(temp[0]),
                             int(sizeof(FloatType) * num_cells_slice),
                             MPI_CHAR,
                             int(recv_from),
                             0,
                             FML::communicator,
                             &status);
#else
                temp = extra_slice_right;
#endif

                // Copy over data from temp
                for (int j = 0; j < num_cells_slice; j++) {
                    slice_left[j] += (temp[j] - background);
                }
            }

            // [2] Send to the left, receive from right
            for (int i = 1; i <= n_extra_left; i++) {
                FloatType * extra_slice_left = density.get_real_grid() - i * num_cells_slice;
                FloatType * slice_right = density.get_real_grid() + num_cells_slice * (Local_nx - i);
                FloatType * temp = buffer.data();

#ifdef USE_MPI
                MPI_Status status;

                int send_to = (ThisTask - 1 + NTasks) % NTasks;
                int recv_from = (ThisTask + 1) % NTasks;

                MPI_Sendrecv(&(extra_slice_left[0]),
                             int(sizeof(FloatType) * num_cells_slice),
                             MPI_CHAR,
                             send_to,
                             0,
                             &(temp[0]),
                             int(sizeof(FloatType) * num_cells_slice),
                             MPI_CHAR,
                             recv_from,
                             0,
                             FML::communicator,
                             &status);
#else
                temp = extra_slice_left;
#endif

                // Copy over data from temp
                for (int j = 0; j < num_cells_slice; j++) {
                    slice_right[j] += (temp[j] - background);
                }
            }
        }

        //=========================================================================================
        // This performs the convolution (grid * convolution_kernel)
        // The argument of the kernel is the number of cells we are away from the cell.
        // ORDER is the width of the kernel. We go through all ORDER^N cells
        // surrounding a given cell (for even ORDER we choose the cells to the right) and add up
        // the field-value at those cells times the kernel.
        // If the ORDER = 1 this only multiplies the whole grid by the constant conv_kernel(0,0,0,..)
        // If the kernel returns 1.0/ORDER^N then this is just a convolution with a tophat of size
        // R = ORDER / Nmesh
        //
        // Don't know if this will be useful for anything, but its just a copy of the same kind of
        // work done in the methods above so it comes for free. Not tested!
        //=========================================================================================

        template <int N, int ORDER>
        void convolve_grid_with_kernel(const FFTWGrid<N> & grid_in,
                                       FFTWGrid<N> & grid_out,
                                       const std::function<FloatType(std::array<double, N> &)> & convolution_kernel) {

            auto nextra = get_extra_slices_needed_by_order<ORDER>();
            assert_mpi(grid_in.get_n_extra_slices_left() >= nextra.first and
                           grid_in.get_n_extra_slices_right() >= nextra.second,
                       "[convolve_grid_with_kernel] Too few extra slices\n");
            assert_mpi(grid_in.get_nmesh() > 0, "[convolve_grid_with_kernel] Grid has to be already allocated!\n");

            // We need to look at width^N cells in total
            constexpr int widthtondim = FML::power(ORDER, N);
            std::array<int, N> xstart;
            if (ORDER % 2 == 0)
                xstart.fill(-ORDER / 2 + 1);
            else
                xstart.fill(-ORDER / 2);

            // Fetch grid information
            const int Nmesh = grid_in.get_nmesh();
            const auto Local_nx = grid_in.get_local_nx();
            const auto Local_x_start = grid_in.get_local_x_start();

            // Make outputgrid (this initializes it to zero)
            grid_out = FFTWGrid<N>(Nmesh, grid_in.get_n_extra_slices_left(), grid_in.get_n_extra_slices_right());

            // Loop over all cells in in-grid
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                std::array<double, N> dx;
                for (auto && real_index : grid_in.get_real_range(islice, islice + 1)) {

                    // Coordinate of cell
                    auto ix = grid_in.coord_from_index(real_index);

                    // Neighbor coord
                    [[maybe_unused]] std::array<int, N> ix_nbor;
                    ix_nbor[0] = ix[0];
                    for (int idim = 1; idim < N; idim++) {
                        ix_nbor[idim] = ix[idim] + 1;
                        if (ix_nbor[idim] >= Nmesh)
                            ix_nbor[idim] -= Nmesh;
                    }

                    // Interpolation
                    FloatType value = 0;
                    for (int i = 0; i < widthtondim; i++) {
                        std::array<int, N> icoord;
                        double w = 1.0;
                        if constexpr (ORDER == 1) {
                            icoord = ix;
                        } else {
                            for (int idim = 0, n = 1; idim < N; idim++, n *= ORDER) {
                                int go_left_right_or_stay = xstart[idim] + (i / n % ORDER);
                                ix_nbor[idim] = ix[idim] + go_left_right_or_stay;
                                dx[idim] = go_left_right_or_stay;
                            }
                            w = convolution_kernel(dx);

                            // Periodic BC
                            icoord[0] = ix_nbor[0];
                            for (int idim = 1; idim < N; idim++) {
                                icoord[idim] = ix_nbor[idim];
                                if (icoord[idim] >= Nmesh)
                                    icoord[idim] -= Nmesh;
                                if (icoord[idim] < 0)
                                    icoord[idim] += Nmesh;
                            }
                        }

                        // Add up
                        value += w * grid_in.get_real(icoord);
                    }

                    // Store the interpolated value
                    grid_out.set_real(ix, value);
                }
            }
        }

        template <int N, class T>
        void particles_to_fourier_grid_interlacing(const T * part,
                                                   size_t NumPart,
                                                   size_t NumPartTot,
                                                   FFTWGrid<N> & density_grid_fourier,
                                                   std::string density_assignment_method) {

            auto Ngrid = density_grid_fourier.get_nmesh();

            // Set how many extra slices we need for the density assignment to go smoothly
            // One extra slice in general as we need to shift the particle half a grid-cell
            auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            int nleft = nleftright.first;
            int nright = nleftright.second + 1;

            // If the grid has too few slices then we must reallocate it
            if (density_grid_fourier.get_n_extra_slices_left() < nleft or
                density_grid_fourier.get_n_extra_slices_right() < nright) {
                density_grid_fourier = FFTWGrid<N>(Ngrid, nleft, nright);
                density_grid_fourier.add_memory_label(
                    "FFTWGrid::particles_to_grid_interlacing::density_grid_fourier (reallocated)");
            }

            // Bin particles to grid
            particles_to_grid<N, T>(part, NumPart, NumPartTot, density_grid_fourier, density_assignment_method);

            // Deposit at virtual half-cell shifted positions; the particles remain untouched.
            const double shift = 1.0 / double(2 * Ngrid);
            std::array<double, N> assignment_offset;
            assignment_offset.fill(shift);
            FFTWGrid<N> density_grid_fourier2(Ngrid, nleft, nright);
            density_grid_fourier2.add_memory_label(
                "FFTWGrid::particles_to_fourier_grid_interlacing::density_grid_fourier2");
            particles_to_grid<N, T>(part,
                                    NumPart,
                                    NumPartTot,
                                    density_grid_fourier2,
                                    density_assignment_method,
                                    assignment_offset);

            // Fourier transform
            density_grid_fourier.fftw_r2c();
            density_grid_fourier2.fftw_r2c();

            // The mean of the two grids (alias cancellation)
            auto Local_nx = density_grid_fourier.get_local_nx();

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                const std::complex<FML::GRID::FloatType> I(0, 1);
                for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
                    // NB: on the Nyquist planes this is not the shift of the (real) grid (that is cos(k_d s), as in
                    // shift_grid_fourier), so the result is not Hermitian symmetric there. It is used for P(k), where
                    // only |delta(k)|^2 enters, so it is kept as it is (cos(k_N s) = 0 for the half-cell shift would
                    // drop the shifted grid on these planes and change the last P(k) bins by ~3%)
                    auto kvec = density_grid_fourier.get_fourier_wavevector_from_index(fourier_index);
                    auto ksum = kvec[0];
                    for (int idim = 1; idim < N; idim++)
                        ksum += kvec[idim];
                    auto norm = std::exp(I * FML::GRID::FloatType(ksum * shift));
                    auto grid1 = density_grid_fourier.get_fourier_from_index(fourier_index);
                    auto grid2 = density_grid_fourier2.get_fourier_from_index(fourier_index);
                    density_grid_fourier.set_fourier_from_index(fourier_index,
                                                                (grid1 + norm * grid2) / FML::GRID::FloatType(2.0));
                }
            }
        }

        template <int N, class T>
        void particles_to_fourier_grid(const T * part,
                                       size_t NumPart,
                                       size_t NumPartTot,
                                       FFTWGrid<N> & density_grid_fourier,
                                       std::string density_assignment_method,
                                       bool interlacing) {

            if (interlacing) {
                particles_to_fourier_grid_interlacing<N, T>(part,
                                                            NumPart,
                                                            NumPartTot,
                                                            density_grid_fourier,
                                                            density_assignment_method);
            } else {
                particles_to_grid<N, T>(part, NumPart, NumPartTot, density_grid_fourier, density_assignment_method);
                density_grid_fourier.fftw_r2c();
            }
        }

        //===================================================================================
        /// @brief Generate the set of shifts used for interlacing (alias reduction by averaging the density field
        /// computed from several copies of the particle distribution shifted by a fraction of a cell).
        ///
        /// The order is the number of grids (including the unshifted one). Supported orders:
        ///  - 1: no interlacing (only the zero shift)
        ///  - any order in 1D: shifts i/order * dx
        ///  - 2 in any dimension: 0 and (1/2,...,1/2) dx. Cancels the aliases (images at k + 2 k_Ny n) with odd
        ///    sum(n)
        ///  - 2^(NDIM-1) for NDIM >= 3 (e.g. 4 in 3D): all corners of the half-cell cube with an even number of
        ///    1/2's: 0, (1/2,1/2,0), (1/2,0,1/2), (0,1/2,1/2). Cancels all aliases except those with all n_i odd
        ///    (the (1,1,1) family which has the smallest amplitude)
        ///  - n^NDIM: a regular sub-lattice with n points per dimension (e.g. 8 = 2^3 in 3D cancels all aliases with
        ///    some odd n_i)
        ///  - 2^m >= 2^NDIM: the 2^NDIM corners of the half-cell cube, recursively refined along the diagonal
        /// The shifts are in [0, dx) in each coordinate.
        ///
        /// @tparam NDIM The dimension
        /// @param[in] interlacing_order The number of grids
        /// @param[in] dx The cell size
        //===================================================================================
        template <int NDIM>
        std::vector<std::array<double, NDIM>> generate_interlacing_shifts(int interlacing_order, double dx) {
            std::vector<std::array<double, NDIM>> shifts;

            // No interlacing: just the zero shift
            if (interlacing_order <= 1) {
                shifts.push_back(std::array<double, NDIM>{});
                return shifts;
            }

            // Any order in 1D
            if constexpr (NDIM == 1) {
                for (int i = 0; i < interlacing_order; i++)
                    shifts.push_back({i / double(interlacing_order) * dx});
                return shifts;
            }

            // Order 2 in any dimension: the body diagonal
            if (interlacing_order == 2) {
                std::array<double, NDIM> shift{};
                shifts.push_back(shift);
                shift.fill(0.5 * dx);
                shifts.push_back(shift);
                return shifts;
            }

            // Order 2^(NDIM-1) for NDIM >= 3: corners of the half-cell cube with an even number of 1/2's
            if (NDIM >= 3 and interlacing_order == FML::power(2, NDIM - 1)) {
                for (int i = 0; i < FML::power(2, NDIM); i++) {
                    std::array<double, NDIM> shift{};
                    int nhalf = 0;
                    for (int idim = 0; idim < NDIM; idim++) {
                        int bit = (i >> idim) & 1;
                        nhalf += bit;
                        shift[idim] = 0.5 * dx * bit;
                    }
                    if (nhalf % 2 == 0)
                        shifts.push_back(shift);
                }
                return shifts;
            }

            // Order n^NDIM: a regular sub-lattice
            const int n1d = int(std::round(std::pow(double(interlacing_order), 1.0 / NDIM)));
            if (FML::power(n1d, NDIM) == interlacing_order) {
                for (int i = 0; i < interlacing_order; i++) {
                    std::array<double, NDIM> shift{};
                    for (int idim = 0, n = 1; idim < NDIM; idim++, n *= n1d)
                        shift[idim] = ((i / n) % n1d) / double(n1d) * dx;
                    shifts.push_back(shift);
                }
                return shifts;
            }

            // Order 2^m with m >= NDIM: the 2^NDIM corners of the half-cell cube, then recursively add copies
            // shifted by dx/4, dx/8, ... along the diagonal
            int log2order = 0;
            while ((1 << (log2order + 1)) <= interlacing_order)
                log2order++;
            if ((1 << log2order) == interlacing_order and log2order >= NDIM) {
                for (int i = 0; i < FML::power(2, NDIM); i++) {
                    std::array<double, NDIM> shift{};
                    for (int idim = 0; idim < NDIM; idim++)
                        shift[idim] = 0.5 * dx * ((i >> idim) & 1);
                    shifts.push_back(shift);
                }
                for (int level = 1; level <= log2order - NDIM; level++) {
                    const double extra = dx / std::pow(2.0, level + 1);
                    const size_t nold = shifts.size();
                    for (size_t j = 0; j < nold; j++) {
                        auto shift = shifts[j];
                        for (int idim = 0; idim < NDIM; idim++)
                            shift[idim] += extra;
                        shifts.push_back(shift);
                    }
                }
                return shifts;
            }

            throw std::runtime_error("[generate_interlacing_shifts] Interlacing order " +
                                     std::to_string(interlacing_order) + " is not supported in " +
                                     std::to_string(NDIM) + "D");
        }

        //===================================================================================
        /// @brief Assign particles to a grid with interlacing of any (supported) order and return the density
        /// contrast in Fourier space: \f$ \delta(k) = \frac{1}{M}\sum_j e^{i k\cdot s_j} \delta_{s_j}(k) \f$ where
        /// \f$ \delta_{s}\f$ is the density of the particles shifted by \f$ s \f$. The window function is not
        /// deconvolved. Order 1 is the standard assignment, order 2 is the usual interlacing. See
        /// generate_interlacing_shifts for the orders available. Particle positions are never modified.
        ///
        /// @tparam N The dimension of the grid
        /// @tparam T The particle class
        /// @param[in] part Pointer to the first particle
        /// @param[in] NumPart Number of local particles
        /// @param[in] NumPartTot Total number of particles over all tasks
        /// @param[out] density_grid_fourier The density contrast in Fourier space (must have Nmesh set; it is
        /// reallocated if it has too few extra slices)
        /// @param[in] density_assignment_method NGP, CIC, TSC, PCS or PQS
        /// @param[in] interlacing_order The number of shifted grids
        //===================================================================================
        template <int N, class T>
        void particles_to_fourier_grid_interlaced(const T * part,
                                                  size_t NumPart,
                                                  size_t NumPartTot,
                                                  FFTWGrid<N> & density_grid_fourier,
                                                  std::string density_assignment_method,
                                                  int interlacing_order) {

            const int Ngrid = density_grid_fourier.get_nmesh();
            assert_mpi(Ngrid > 0, "[particles_to_fourier_grid_interlaced] Grid must be allocated\n");

            // One extra slice on the right as the particles are shifted by less than a cell without wrapping in x
            auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            const int nleft = nleftright.first;
            const int nright = nleftright.second + (interlacing_order > 1 ? 1 : 0);
            if (density_grid_fourier.get_n_extra_slices_left() < nleft or
                density_grid_fourier.get_n_extra_slices_right() < nright) {
                density_grid_fourier = FFTWGrid<N>(Ngrid, nleft, nright);
                density_grid_fourier.add_memory_label(
                    "FFTWGrid::particles_to_fourier_grid_interlaced::density_grid_fourier (reallocated)");
            }

            if (interlacing_order <= 1) {
                particles_to_grid<N, T>(part, NumPart, NumPartTot, density_grid_fourier, density_assignment_method);
                density_grid_fourier.fftw_r2c();
                return;
            }

            const auto shifts = generate_interlacing_shifts<N>(interlacing_order, 1.0 / double(Ngrid));
            const FloatType norm = FloatType(1.0 / double(shifts.size()));

            FFTWGrid<N> grid_shifted(Ngrid, nleft, nright);
            grid_shifted.add_memory_label("FFTWGrid::particles_to_fourier_grid_interlaced::grid_shifted");
            density_grid_fourier.fill_fourier_grid(0.0);
            density_grid_fourier.set_grid_status_real(false);
            for (auto & shift : shifts) {
                particles_to_grid<N, T>(part,
                                        NumPart,
                                        NumPartTot,
                                        grid_shifted,
                                        density_assignment_method,
                                        shift,
                                        FML::NTasks == 1);
                grid_shifted.fftw_r2c();

                // Virtual positions are offset by +s, giving delta(x - s): undo the Fourier phase
                FML::GRID::shift_grid_fourier<N>(grid_shifted, shift, +1.0);

                const auto Local_nx = density_grid_fourier.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && fourier_index : density_grid_fourier.get_fourier_range(islice, islice + 1)) {
                        auto value = density_grid_fourier.get_fourier_from_index(fourier_index);
                        value += grid_shifted.get_fourier_from_index(fourier_index) * norm;
                        density_grid_fourier.set_fourier_from_index(fourier_index, value);
                    }
                }
            }
        }

    } // namespace INTERPOLATION
} // namespace FML
#endif
