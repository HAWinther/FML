#ifndef FML_NUFFT_HEADER
#define FML_NUFFT_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>

namespace FML {

    //================================================================================
    /// Non-uniform fast Fourier transforms (NUFFT) between particles and an FFTWGrid.
    ///
    /// Type 1 (particles -> Fourier modes):
    /// \f[ F(k) = {\rm norm} \sum_p c_p e^{-i k\cdot x_p} \f]
    /// for all modes with \f$ |m_i| < N_{\rm mesh}/2 \f$ (\f$ k = 2\pi m \f$, positions in [0,1)).
    ///
    /// Type 2 (Fourier modes -> particles):
    /// \f[ f(x_p) = \sum_k F(k) e^{i k\cdot x_p} \f]
    /// summed over the same modes.
    ///
    /// Both are exact up to a relative error \f$ \sim \epsilon \f$ (in contrast to CIC/TSC/... which have
    /// aliasing and a window function). The algorithm is the standard one (as in FINUFFT, Barnett et al. 2019):
    /// spread with the "exponential of semicircle" kernel \f$ \phi(z) = e^{\beta(\sqrt{1-z^2}-1)} \f$ of width
    /// \f$ w \f$ cells onto a grid \f$ \sigma \f$ times finer, FFT, divide by the Fourier transform of the
    /// kernel and keep the modes of the coarse grid (and the reverse for type 2). The width is
    /// \f$ w \approx \log_{10}(1/\epsilon) + 1 \f$ for \f$ \sigma = 2 \f$, so the cost per particle is
    /// \f$ w^{\rm NDIM} \f$ (343 for \f$ \epsilon = 10^{-6} \f$ in 3D) and the fine grid has \f$ \sigma^{\rm NDIM} \f$
    /// times more cells than the output grid.
    ///
    /// Works with MPI (the particles must be on the task that owns their x-slab, as for the other assignment
    /// methods) and OpenMP. The fine grid size is rounded up to be even and divisible by the number of tasks.
    ///
    /// The conventions match FFTWGrid: fftw_r2c includes a factor \f$ 1/N_{\rm mesh}^{\rm NDIM} \f$ so
    /// particles_to_fourier_grid_nufft gives the same as particles_to_grid + fftw_r2c, but without window
    /// function and aliasing.
    //================================================================================
    namespace NUFFT {

        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;
        using FloatType = FML::GRID::FloatType;
        using ComplexType = FML::GRID::ComplexType;

        //================================================================================
        /// The exponential of semicircle (ES) spreading kernel and its Fourier transform.
        //================================================================================
        class ESKernel {
          public:
            double epsilon{1e-6}; ///< Requested relative accuracy
            double sigma{2.0};    ///< Upsampling factor of the fine grid
            int width{7};         ///< Kernel width in fine-grid cells
            double beta{16.1};    ///< Kernel shape parameter

            ESKernel() : ESKernel(1e-6, 2.0) {}
            ESKernel(double _epsilon, double _sigma = 2.0) : epsilon(_epsilon), sigma(_sigma) {
                assert_mpi(epsilon > 0.0 and epsilon < 1.0, "[NUFFT::ESKernel] epsilon must be in (0,1)\n");
                assert_mpi(sigma >= 1.25, "[NUFFT::ESKernel] sigma must be >= 1.25\n");
                // Width and beta as in FINUFFT (Barnett, Magland & af Klinteberg 2019)
                if (std::fabs(sigma - 2.0) < 1e-12) {
                    width = int(std::ceil(std::log10(10.0 / epsilon)));
                } else {
                    width = int(std::ceil(-std::log(epsilon) / (M_PI * std::sqrt(1.0 - 1.0 / sigma))));
                }
                width = std::max(2, std::min(16, width));
                beta = std::fabs(sigma - 2.0) < 1e-12 ? 2.30 * width : 0.97 * M_PI * (1.0 - 0.5 / sigma) * width;
                setup_quadrature();
            }

            /// The kernel as function of the distance x in fine-grid cells
            double operator()(double x) const {
                const double z = 2.0 * x / width;
                if (std::fabs(z) >= 1.0)
                    return 0.0;
                return std::exp(beta * (std::sqrt(1.0 - z * z) - 1.0));
            }

            /// Fourier transform \f$ \hat\phi(\kappa) = \int \phi(t) e^{-i\kappa t} dt \f$ with t in fine-grid cells
            double fourier(double kappa) const {
                double sum = 0.0;
                for (size_t i = 0; i < quad_x.size(); i++)
                    sum += quad_w[i] * quad_phi[i] * std::cos(kappa * quad_x[i]);
                return 2.0 * sum;
            }

            /// The size of the fine grid for a given output grid (>= sigma * nmesh, even and divisible by NTasks)
            int fine_nmesh(int nmesh) const {
                int nf = int(std::ceil(sigma * nmesh - 1e-10));
                while (nf % 2 != 0 or nf % FML::NTasks != 0)
                    nf++;
                return nf;
            }

            /// Number of extra slices needed on the fine grid (one more than the kernel half-width on both sides
            /// so that particles slightly outside the domain are fine)
            int extra_slices() const { return (width + 1) / 2 + 1; }

          private:
            std::vector<double> quad_x, quad_w, quad_phi;

            // Gauss-Legendre quadrature on [0, width/2] for the Fourier transform of the kernel
            void setup_quadrature() {
                const int nq = std::max(40, 4 * width);
                std::vector<double> x(nq), w(nq);
                for (int i = 0; i < nq; i++) {
                    // Newton iteration for the roots of P_nq starting from the Chebyshev guess
                    double z = std::cos(M_PI * (i + 0.75) / (nq + 0.5));
                    double dp = 1.0;
                    for (int it = 0; it < 100; it++) {
                        double p0 = 1.0, p1 = z;
                        for (int n = 2; n <= nq; n++) {
                            double p2 = ((2 * n - 1) * z * p1 - (n - 1) * p0) / n;
                            p0 = p1;
                            p1 = p2;
                        }
                        dp = nq * (z * p1 - p0) / (z * z - 1.0);
                        double dz = p1 / dp;
                        z -= dz;
                        if (std::fabs(dz) < 1e-15)
                            break;
                    }
                    x[i] = z;
                    w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
                }
                const double half = 0.5 * width;
                quad_x.resize(nq);
                quad_w.resize(nq);
                quad_phi.resize(nq);
                for (int i = 0; i < nq; i++) {
                    quad_x[i] = 0.5 * half * (x[i] + 1.0);
                    quad_w[i] = 0.5 * half * w[i];
                    quad_phi[i] = (*this)(quad_x[i]);
                }
            }
        };

        //================================================================================
        // Internal helpers
        //================================================================================

        /// @brief Add what was spread onto the extra slices to the neighbouring tasks (or wrap it around with one
        /// task) and zero the extra slices. The grid is assumed to start out as zero.
        template <int N>
        void add_extra_slices_to_neighbours(FFTWGrid<N> & grid) {
            const int nleft = grid.get_n_extra_slices_left();
            const int nright = grid.get_n_extra_slices_right();
            const auto Local_nx = grid.get_local_nx();
            const auto nslice = grid.get_ntot_real_slice_alloc();
            assert_mpi(Local_nx >= std::max(nleft, nright),
                       "[NUFFT] Too few slices per task for the kernel width. Use fewer tasks or a larger grid\n");
            std::vector<FloatType> buffer(nslice);

            // Right extra slice i belongs to slice i of the task on the right
            for (int i = 0; i < nright; i++) {
                FloatType * extra = grid.get_real_grid_right() + nslice * i;
                FloatType * target = grid.get_real_grid() + nslice * i;
#ifdef USE_MPI
                MPI_Sendrecv(extra, int(sizeof(FloatType) * nslice), MPI_CHAR, (FML::ThisTask + 1) % FML::NTasks, 0,
                             buffer.data(), int(sizeof(FloatType) * nslice), MPI_CHAR,
                             (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks, 0, FML::communicator, MPI_STATUS_IGNORE);
#else
                std::copy(extra, extra + nslice, buffer.begin());
#endif
                for (ptrdiff_t j = 0; j < nslice; j++)
                    target[j] += buffer[j];
                std::fill(extra, extra + nslice, FloatType(0));
            }

            // Left extra slice -i belongs to slice Local_nx - i of the task on the left
            for (int i = 1; i <= nleft; i++) {
                FloatType * extra = grid.get_real_grid() - nslice * i;
                FloatType * target = grid.get_real_grid() + nslice * (Local_nx - i);
#ifdef USE_MPI
                MPI_Sendrecv(extra, int(sizeof(FloatType) * nslice), MPI_CHAR,
                             (FML::ThisTask - 1 + FML::NTasks) % FML::NTasks, 0, buffer.data(),
                             int(sizeof(FloatType) * nslice), MPI_CHAR, (FML::ThisTask + 1) % FML::NTasks, 0,
                             FML::communicator, MPI_STATUS_IGNORE);
#else
                std::copy(extra, extra + nslice, buffer.begin());
#endif
                for (ptrdiff_t j = 0; j < nslice; j++)
                    target[j] += buffer[j];
                std::fill(extra, extra + nslice, FloatType(0));
            }
        }

        /// @brief Compute the kernel weights and the first cell index for a particle (fine grid units).
        /// For the x-direction the index is relative to the local slab.
        template <int N>
        inline void kernel_weights(const ESKernel & kernel,
                                   const double * pos,
                                   int nfine,
                                   ptrdiff_t local_x_start,
                                   std::array<int, N> & istart,
                                   std::array<double, N * 16> & weights) {
            const int w = kernel.width;
            for (int idim = 0; idim < N; idim++) {
                const double x = pos[idim] * nfine;
                const int j0 = int(std::ceil(x - 0.5 * w));
                istart[idim] = j0 - (idim == 0 ? int(local_x_start) : 0);
                for (int j = 0; j < w; j++)
                    weights[idim * w + j] = kernel(double(j0 + j) - x);
            }
        }

        /// @brief Spread values c_p onto the (zeroed) fine grid with the ES kernel
        template <int N, class T>
        void spread_to_fine_grid(const T * part,
                                 size_t NumPart,
                                 const std::vector<double> & values,
                                 const ESKernel & kernel,
                                 FFTWGrid<N> & fine) {
            const int nf = fine.get_nmesh();
            const int w = kernel.width;
            const int wtondim = FML::power(w, N);
            const auto Local_nx = fine.get_local_nx();
            const auto Local_x_start = fine.get_local_x_start();
            const int nleft = fine.get_n_extra_slices_left();
            const int nright = fine.get_n_extra_slices_right();
            FloatType * grid = fine.get_real_grid();
            fine.fill_real_grid(0.0);

            int out_of_bounds = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : out_of_bounds)
#endif
            for (size_t ip = 0; ip < NumPart; ip++) {
                const double * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[ip]);
                std::array<int, N> istart;
                std::array<double, N * 16> weights;
                kernel_weights<N>(kernel, pos, nf, Local_x_start, istart, weights);
                if (istart[0] < -nleft or istart[0] + w > Local_nx + nright) {
                    out_of_bounds = 1;
                    continue;
                }
                for (int i = 0; i < wtondim; i++) {
                    std::array<int, N> coord;
                    double weight = values[ip];
                    for (int idim = 0, n = 1; idim < N; idim++, n *= w) {
                        const int j = (i / n) % w;
                        weight *= weights[idim * w + j];
                        int c = istart[idim] + j;
                        if (idim > 0 or FML::NTasks == 1) {
                            const int nmax = idim > 0 ? nf : int(Local_nx);
                            c = (c % nmax + nmax) % nmax;
                        }
                        coord[idim] = c;
                    }
                    const auto index = fine.get_index_real(coord);
#ifdef USE_OMP
#pragma omp atomic
#endif
                    grid[index] += FloatType(weight);
                }
            }
            FML::MaxOverTasks(&out_of_bounds);
            assert_mpi(out_of_bounds == 0,
                       "[NUFFT::spread_to_fine_grid] A particle is outside the local domain (communicate the "
                       "particles first)\n");

            if (FML::NTasks > 1)
                add_extra_slices_to_neighbours<N>(fine);
        }

        /// @brief Interpolate the fine (real space, with boundaries communicated) grid to the particle positions
        template <int N, class T>
        void interpolate_from_fine_grid(const FFTWGrid<N> & fine,
                                        const T * part,
                                        size_t NumPart,
                                        const ESKernel & kernel,
                                        std::vector<FloatType> & result) {
            const int nf = fine.get_nmesh();
            const int w = kernel.width;
            const int wtondim = FML::power(w, N);
            const auto Local_nx = fine.get_local_nx();
            const auto Local_x_start = fine.get_local_x_start();
            const int nleft = fine.get_n_extra_slices_left();
            const int nright = fine.get_n_extra_slices_right();
            result.resize(NumPart);

            int out_of_bounds = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : out_of_bounds)
#endif
            for (size_t ip = 0; ip < NumPart; ip++) {
                const double * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[ip]);
                std::array<int, N> istart;
                std::array<double, N * 16> weights;
                kernel_weights<N>(kernel, pos, nf, Local_x_start, istart, weights);
                if (istart[0] < -nleft or istart[0] + w > Local_nx + nright) {
                    out_of_bounds = 1;
                    continue;
                }
                double sum = 0.0;
                for (int i = 0; i < wtondim; i++) {
                    std::array<int, N> coord;
                    double weight = 1.0;
                    for (int idim = 0, n = 1; idim < N; idim++, n *= w) {
                        const int j = (i / n) % w;
                        weight *= weights[idim * w + j];
                        int c = istart[idim] + j;
                        if (idim > 0)
                            c = (c % nf + nf) % nf;
                        coord[idim] = c;
                    }
                    sum += weight * fine.get_real(coord);
                }
                result[ip] = FloatType(sum);
            }
            FML::MaxOverTasks(&out_of_bounds);
            assert_mpi(out_of_bounds == 0,
                       "[NUFFT::interpolate_from_fine_grid] A particle is outside the local domain (communicate the "
                       "particles first)\n");
        }

        /// @brief The 1D table of 1/phi_hat for the integer wavenumbers m = -nf/2,...,nf/2 of the fine grid
        inline std::vector<double> inverse_kernel_fourier_table(const ESKernel & kernel, int nf) {
            std::vector<double> table(nf + 1);
            for (int m = -nf / 2; m <= nf / 2; m++)
                table[m + nf / 2] = 1.0 / kernel.fourier(2.0 * M_PI * m / double(nf));
            return table;
        }

        //================================================================================
        /// @brief Copy all Fourier modes with \f$ |m_i| < \min(N_{\rm src},N_{\rm dst})/2 \f$ from one grid to
        /// another of a different size, multiplying by factor(m). All other modes of dst are set to zero. Works with
        /// MPI (the modes are sent to the task that owns them in dst).
        ///
        /// @param[in] src The source grid (Fourier space)
        /// @param[out] dst The destination grid (allocated, Fourier space on output)
        /// @param[in] factor Function of the integer wavevector
        //================================================================================
        template <int N>
        void copy_fourier_modes(const FFTWGrid<N> & src,
                                FFTWGrid<N> & dst,
                                const std::function<double(const std::array<int, N> &)> & factor) {
            const int nsrc = src.get_nmesh();
            const int ndst = dst.get_nmesh();
            const int mmax = std::min(nsrc, ndst) / 2; // keep |m| < mmax
            const int ndst_per_task = ndst / FML::NTasks;

            dst.fill_fourier_grid(0.0);
            dst.set_grid_status_real(false);

            struct Mode {
                int coord[N];
                ComplexType value;
            };
            std::vector<std::vector<Mode>> send(FML::NTasks);
            const auto Local_nx = src.get_local_nx();
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && fourier_index : src.get_fourier_range(islice, islice + 1)) {
                    auto m = src.get_fourier_integer_wavevector_from_index(fourier_index);
                    bool keep = true;
                    for (int idim = 0; idim < N; idim++)
                        keep = keep and std::abs(m[idim]) < mmax;
                    if (not keep)
                        continue;
                    Mode mode;
                    for (int idim = 0; idim < N; idim++)
                        mode.coord[idim] = (m[idim] >= 0 or idim == N - 1) ? m[idim] : m[idim] + ndst;
                    mode.value = src.get_fourier_from_index(fourier_index) * FloatType(factor(m));
                    const int owner = FML::NTasks == 1 ? 0 : mode.coord[0] / ndst_per_task;
                    send[owner].push_back(mode);
                }
            }

            std::vector<Mode> recv;
#ifdef USE_MPI
            std::vector<int> nsend(FML::NTasks), nrecv(FML::NTasks), sdispl(FML::NTasks), rdispl(FML::NTasks);
            for (int t = 0; t < FML::NTasks; t++)
                nsend[t] = int(send[t].size() * sizeof(Mode));
            MPI_Alltoall(nsend.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, FML::communicator);
            std::vector<Mode> sendbuf;
            for (int t = 0; t < FML::NTasks; t++) {
                sdispl[t] = int(sendbuf.size() * sizeof(Mode));
                sendbuf.insert(sendbuf.end(), send[t].begin(), send[t].end());
            }
            size_t ntot = 0;
            for (int t = 0; t < FML::NTasks; t++) {
                rdispl[t] = int(ntot);
                ntot += nrecv[t];
            }
            recv.resize(ntot / sizeof(Mode));
            MPI_Alltoallv(sendbuf.data(), nsend.data(), sdispl.data(), MPI_CHAR, recv.data(), nrecv.data(),
                          rdispl.data(), MPI_CHAR, FML::communicator);
#else
            recv = send[0];
#endif
            const auto dst_x_start = dst.get_local_x_start();
            for (auto & mode : recv) {
                std::array<int, N> coord;
                for (int idim = 0; idim < N; idim++)
                    coord[idim] = mode.coord[idim];
                coord[0] -= int(dst_x_start);
                dst.set_fourier(coord, mode.value);
            }
        }

        //================================================================================
        /// @brief NUFFT type 1: compute
        /// \f$ F(k) = {\rm norm} \sum_p c_p e^{-i k\cdot x_p} \f$ for all modes of the output grid with
        /// \f$ |m_i| < N_{\rm mesh}/2 \f$ (the Nyquist planes are set to zero).
        ///
        /// @param[in] part Pointer to the first particle (positions in [0,1), on the right task)
        /// @param[in] NumPart Number of local particles
        /// @param[in] values The values c_p (size NumPart)
        /// @param[in] norm Overall normalization
        /// @param[out] out The output grid (Nmesh must be set). Fourier space on output.
        /// @param[in] kernel The spreading kernel (sets the accuracy)
        //================================================================================
        template <int N, class T>
        void nufft_type1(const T * part,
                         size_t NumPart,
                         const std::vector<double> & values,
                         double norm,
                         FFTWGrid<N> & out,
                         const ESKernel & kernel = ESKernel()) {
            const int nmesh = out.get_nmesh();
            assert_mpi(nmesh > 0, "[NUFFT::nufft_type1] Output grid must be allocated\n");
            assert_mpi(values.size() >= NumPart, "[NUFFT::nufft_type1] Too few values\n");
            const int nf = kernel.fine_nmesh(nmesh);
            const int nextra = kernel.extra_slices();

            FFTWGrid<N> fine(nf, nextra, nextra);
            fine.add_memory_label("FFTWGrid::NUFFT::nufft_type1::fine");
            spread_to_fine_grid<N, T>(part, NumPart, values, kernel, fine);
            fine.fftw_r2c();

            // r2c includes 1/nf^N, the kernel gives phi_hat(m) per dimension
            const auto inv = inverse_kernel_fourier_table(kernel, nf);
            const double prefactor = norm * std::pow(double(nf), N);
            copy_fourier_modes<N>(fine, out, [&](const std::array<int, N> & m) {
                double f = prefactor;
                for (int idim = 0; idim < N; idim++)
                    f *= inv[m[idim] + nf / 2];
                return f;
            });
        }

        //================================================================================
        /// @brief NUFFT type 2: compute \f$ f(x_p) = \sum_k F(k) e^{i k\cdot x_p} \f$ summing over the modes of
        /// the input grid with \f$ |m_i| < N_{\rm mesh}/2 \f$ (the Nyquist planes are ignored).
        ///
        /// @param[in] in The input grid in Fourier space
        /// @param[in] part Pointer to the first particle (positions in [0,1), on the right task)
        /// @param[in] NumPart Number of local particles
        /// @param[out] result The values at the particle positions
        /// @param[in] kernel The spreading kernel (sets the accuracy)
        //================================================================================
        template <int N, class T>
        void nufft_type2(const FFTWGrid<N> & in,
                         const T * part,
                         size_t NumPart,
                         std::vector<FloatType> & result,
                         const ESKernel & kernel = ESKernel()) {
            const int nmesh = in.get_nmesh();
            assert_mpi(nmesh > 0, "[NUFFT::nufft_type2] Input grid must be allocated\n");
            const int nf = kernel.fine_nmesh(nmesh);
            const int nextra = kernel.extra_slices();

            FFTWGrid<N> fine(nf, nextra, nextra);
            fine.add_memory_label("FFTWGrid::NUFFT::nufft_type2::fine");
            const auto inv = inverse_kernel_fourier_table(kernel, nf);
            copy_fourier_modes<N>(in, fine, [&](const std::array<int, N> & m) {
                double f = 1.0;
                for (int idim = 0; idim < N; idim++)
                    f *= inv[m[idim] + nf / 2];
                return f;
            });
            fine.fftw_c2r();
            fine.communicate_boundaries();
            interpolate_from_fine_grid<N, T>(fine, part, NumPart, kernel, result);
        }

        /// @brief The particle weights c_p = mass / mean mass (or 1 if the particles have no mass)
        template <class T>
        std::vector<double> particle_weights(const T * part, size_t NumPart, size_t NumPartTot) {
            std::vector<double> values(NumPart, 1.0);
            if constexpr (FML::PARTICLE::has_get_mass<T>()) {
                double mean_mass = 0.0;
                for (size_t i = 0; i < NumPart; i++) {
                    values[i] = FML::PARTICLE::GetMass(const_cast<T *>(part)[i]);
                    mean_mass += values[i];
                }
                FML::SumOverTasks(&mean_mass);
                mean_mass /= double(NumPartTot);
                for (auto & v : values)
                    v /= mean_mass;
            }
            return values;
        }

        //================================================================================
        /// @brief The density contrast of a set of particles in Fourier space computed with a NUFFT:
        /// \f$ \delta(k) = \frac{1}{N_{\rm part}} \sum_p \frac{m_p}{\bar{m}} e^{-ik\cdot x_p} \f$ for
        /// \f$ k \neq 0 \f$ and \f$ |m_i| < N_{\rm mesh}/2 \f$ (no window function, no aliasing).
        /// Same normalization as particles_to_grid followed by fftw_r2c.
        ///
        /// @param[in] part Pointer to the first particle
        /// @param[in] NumPart Number of local particles
        /// @param[in] NumPartTot Total number of particles
        /// @param[out] density_fourier The output grid (Nmesh must be set)
        /// @param[in] epsilon The requested accuracy
        /// @param[in] sigma The upsampling factor
        //================================================================================
        template <int N, class T>
        void particles_to_fourier_grid_nufft(const T * part,
                                             size_t NumPart,
                                             size_t NumPartTot,
                                             FFTWGrid<N> & density_fourier,
                                             double epsilon = 1e-6,
                                             double sigma = 2.0) {
            ESKernel kernel(epsilon, sigma);
            auto values = particle_weights(part, NumPart, NumPartTot);
            nufft_type1<N, T>(part, NumPart, values, 1.0 / double(NumPartTot), density_fourier, kernel);
            if (density_fourier.get_local_x_start() == 0)
                density_fourier.set_fourier_from_index(0, 0.0);
        }

        //================================================================================
        /// @brief The gradient of the potential at the particle positions computed with NUFFTs (no window
        /// functions, no aliasing, exact gradient \f$ ik \f$). Solves \f$ \nabla^2\Phi = {\rm norm}\cdot\delta \f$
        /// with the modes \f$ |m_i| < N_{\rm mesh}/2 \f$ and returns \f$ \nabla\Phi \f$ at the particle
        /// positions, i.e. the same quantity as compute_force_from_density_fourier (with continuous kernels)
        /// followed by interpolation to the particles.
        ///
        /// Memory: two grids of size \f$ (\sigma N_{\rm mesh})^{\rm NDIM} \f$.
        ///
        /// @param[in] part Pointer to the first particle
        /// @param[in] NumPart Number of local particles
        /// @param[in] NumPartTot Total number of particles
        /// @param[in] nmesh The resolution: modes with \f$ |m_i| < N_{\rm mesh}/2 \f$ are included
        /// @param[in] norm_poisson_equation The prefactor in the Poisson equation
        /// @param[out] force \f$ \nabla\Phi \f$ for each particle and direction
        /// @param[in] epsilon The requested accuracy
        /// @param[in] sigma The upsampling factor
        /// @param[in] filter Optional filter W(k) (k in units of 1/boxsize) multiplying the potential, e.g. a
        /// long-range split. Default none.
        //================================================================================
        template <int N, class T>
        void compute_force_nufft(const T * part,
                                 size_t NumPart,
                                 size_t NumPartTot,
                                 int nmesh,
                                 double norm_poisson_equation,
                                 std::array<std::vector<FloatType>, N> & force,
                                 double epsilon = 1e-6,
                                 double sigma = 2.0,
                                 std::function<double(double)> filter = nullptr) {
            ESKernel kernel(epsilon, sigma);
            const int nf = kernel.fine_nmesh(nmesh);
            const int nextra = kernel.extra_slices();
            const int mmax = nmesh / 2;

            // Spread and FFT
            FFTWGrid<N> potential(nf, nextra, nextra);
            potential.add_memory_label("FFTWGrid::NUFFT::compute_force_nufft::potential");
            auto values = particle_weights(part, NumPart, NumPartTot);
            spread_to_fine_grid<N, T>(part, NumPart, values, kernel, potential);
            potential.fftw_r2c();

            // delta(k) = nf^N / NumPartTot * B(k) / phi_hat(k) -> Phi(k) = -norm delta / k^2 and divide by
            // phi_hat(k) once more for the interpolation back to the particles
            const auto inv = inverse_kernel_fourier_table(kernel, nf);
            const double prefactor = std::pow(double(nf), N) / double(NumPartTot) * norm_poisson_equation;
            const auto Local_nx = potential.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && fourier_index : potential.get_fourier_range(islice, islice + 1)) {
                    auto m = potential.get_fourier_integer_wavevector_from_index(fourier_index);
                    double kmag2 = 0.0, f = prefactor;
                    bool keep = true;
                    for (int idim = 0; idim < N; idim++) {
                        keep = keep and std::abs(m[idim]) < mmax;
                        kmag2 += (2.0 * M_PI * m[idim]) * (2.0 * M_PI * m[idim]);
                        f *= inv[m[idim] + nf / 2] * inv[m[idim] + nf / 2];
                    }
                    if (not keep or kmag2 == 0.0) {
                        potential.set_fourier_from_index(fourier_index, 0.0);
                        continue;
                    }
                    f *= -1.0 / kmag2;
                    if (filter)
                        f *= filter(std::sqrt(kmag2));
                    potential.set_fourier_from_index(fourier_index,
                                                     potential.get_fourier_from_index(fourier_index) * FloatType(f));
                }
            }

            // Gradient, back to real space and interpolate
            for (int idim = 0; idim < N; idim++) {
                FFTWGrid<N> grad = potential;
                grad.add_memory_label("FFTWGrid::NUFFT::compute_force_nufft::grad");
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    const ComplexType I(0, 1);
                    for (auto && fourier_index : grad.get_fourier_range(islice, islice + 1)) {
                        auto m = grad.get_fourier_integer_wavevector_from_index(fourier_index);
                        auto value = grad.get_fourier_from_index(fourier_index);
                        grad.set_fourier_from_index(fourier_index, value * I * FloatType(2.0 * M_PI * m[idim]));
                    }
                }
                grad.fftw_c2r();
                grad.communicate_boundaries();
                interpolate_from_fine_grid<N, T>(grad, part, NumPart, kernel, force[idim]);
            }
        }

    } // namespace NUFFT
} // namespace FML
#endif
