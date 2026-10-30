#ifndef FML_SHEETINTERPOLATION_HEADER
#define FML_SHEETINTERPOLATION_HEADER

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>

namespace FML {
    namespace INTERPOLATION {

#ifdef USE_MPI
        namespace detail {
            /// Send send[t] to task t and return what all the tasks sent to us (in task order). The counts are in
            /// elements of type T (as a contiguous MPI type), so byte counts above 2^31 are fine
            template <class T>
            std::vector<T> alltoallv_elements(std::vector<std::vector<T>> & send) {
                static_assert(std::is_trivially_copyable_v<T>, "Sheet MPI records must be trivially copyable");
                FML::COMMUNICATION::require_count(send.size() == size_t(FML::NTasks) && sizeof(T) <= size_t(INT_MAX),
                                                  "[sheet exchange] Invalid peers or record extent");
                std::vector<FML::COMMUNICATION::Count> sizes(FML::NTasks);
                for (int t = 0; t < FML::NTasks; ++t)
                    sizes[t] = send[t].size();
                const auto outgoing = FML::COMMUNICATION::checked_int_layout(sizes);
                std::vector<int> nrecv(FML::NTasks);
                MPI_Alltoall(outgoing.counts.data(), 1, MPI_INT, nrecv.data(), 1, MPI_INT, FML::communicator);
                const auto incoming = FML::COMMUNICATION::checked_int_layout(nrecv);
                FML::COMMUNICATION::require_count(outgoing.total <= std::vector<T>().max_size() &&
                                                      incoming.total <= std::vector<T>().max_size(),
                                                  "[sheet exchange] Element storage exceeds vector capacity");
                std::vector<T> sendbuf;
                sendbuf.reserve(outgoing.total);
                for (int t = 0; t < FML::NTasks; ++t) {
                    sendbuf.insert(sendbuf.end(), send[t].begin(), send[t].end());
                    std::vector<T>().swap(send[t]);
                }
                const size_t ntot = incoming.total;
                std::vector<T> recv(ntot);
                MPI_Datatype type;
                MPI_Type_contiguous(int(sizeof(T)), MPI_BYTE, &type);
                MPI_Type_commit(&type);
                MPI_Alltoallv(sendbuf.data(),
                              outgoing.counts.data(),
                              outgoing.offsets.data(),
                              type,
                              recv.data(),
                              nrecv.data(),
                              incoming.offsets.data(),
                              type,
                              FML::communicator);
                MPI_Type_free(&type);
                return recv;
            }
        } // namespace detail
#endif

        //================================================================================
        // Spectral sheet interpolation (Hahn, Abel & Kaehler 2013; Stuecker et al 2020; List, Hahn & Rampf 2024).
        //
        // Particles that started on a regular lattice q_i sample the dark matter sheet x(q) = q + Psi(q). The
        // displacement field Psi(q) = x - q is a smooth function of q (until shell crossing) so it can be Fourier
        // interpolated to points in between the particles: we create "tracer" particles at
        //     x = q_i + s + Psi(q_i + s)
        // for the R^NDIM sub-lattice shifts s = (j_1,...,j_NDIM) / (R Npart_1D), j = 0,...,R-1, and assign all of
        // them to the grid. This suppresses the discreteness (shot noise and lattice) effects of the particles
        // at the cost of R^NDIM assignments. With R equal to the ratio of the force grid to the particle grid the
        // tracers fill the force grid one per cell (so the lattice pattern from a force grid finer than the
        // particle grid disappears).
        //
        // Caveat: after shell crossing Psi(q) has kinks and the Fourier interpolation rings (Gibbs). The method is
        // therefore most useful at early times / on large scales.
        //
        // The displacement field can come from the particles themselves (they need a Lagrangian coordinate,
        // get_q) or be supplied directly (e.g. a 1LPT displacement field computed from the initial conditions).
        //================================================================================

        /// A particle type for the sheet tracers (only a position)
        template <int N>
        struct SheetTracer {
            double pos[N];
            double * get_pos() { return pos; }
            constexpr int get_ndim() const { return N; }
        };

        /// @brief Move a vector of particles (any type with get_pos) to the task that owns their x-coordinate
        /// (the domain decomposition in FML::xmin_domain / xmax_domain).
        template <class T>
        void redistribute_particles_by_x(std::vector<T> & part) {
#ifdef USE_MPI
            if (FML::NTasks == 1)
                return;
            auto xmin_per_task = FML::GatherFromTasks(&FML::xmin_domain);
            std::vector<std::vector<T>> send(FML::NTasks);
            for (auto & p : part) {
                const double x = FML::PARTICLE::GetPos(p)[0];
                int owner =
                    int(std::upper_bound(xmin_per_task.begin(), xmin_per_task.end(), x) - xmin_per_task.begin()) - 1;
                owner = std::max(0, std::min(FML::NTasks - 1, owner));
                send[owner].push_back(p);
            }
            part = detail::alltoallv_elements(send);
#else
            (void)part;
#endif
        }

        //================================================================================
        /// @brief Compute the displacement field \f$ \Psi(q) = x - q \f$ on the particle lattice from a set of
        /// particles that started on a regular lattice with Npart_1D particles per dimension. The particles must
        /// have a Lagrangian coordinate (get_q). The lattice may be offset by a constant (e.g. cell centered); the
        /// offset (in units of the lattice spacing, in [0,1)) is returned. Grid cell i holds \f$ \Psi \f$ at
        /// \f$ q_i = (i + {\rm offset}) / N_{\rm part,1D} \f$. Displacements are wrapped to [-1/2,1/2) (box units).
        ///
        /// @param[in] part Pointer to the first particle
        /// @param[in] NumPart Number of local particles
        /// @param[in] Npart_1D Particles per dimension (the total number must be Npart_1D^N)
        /// @param[out] Psi The displacement field in real space (one grid per component, Nmesh = Npart_1D)
        /// @param[out] lattice_offset The offset of the lattice in units of the lattice spacing
        //================================================================================
        template <int N, class T>
        void displacement_field_from_particles(const T * part,
                                               size_t NumPart,
                                               int Npart_1D,
                                               std::array<FFTWGrid<N>, N> & Psi,
                                               std::array<double, N> & lattice_offset) {
            static_assert(FML::PARTICLE::has_get_q<T>(),
                          "displacement_field_from_particles needs particles with a Lagrangian coordinate (get_q)");
            const int Np = Npart_1D;

            // The lattice offset from the first particle on the first task that has particles
            std::array<double, N> offset;
            offset.fill(-1.0);
            if (NumPart > 0) {
                const double * q = FML::PARTICLE::GetLagrangianPos(const_cast<T *>(part)[0]);
                for (int idim = 0; idim < N; idim++) {
                    double u = q[idim] * Np;
                    offset[idim] = u - std::floor(u);
                    if (offset[idim] > 1.0 - 1e-8)
                        offset[idim] = 0.0;
                }
            }
            std::vector<std::array<double, N>> all_offsets = FML::GatherFromTasks(&offset);
            for (auto & o : all_offsets)
                if (o[0] >= 0.0) {
                    offset = o;
                    break;
                }
            lattice_offset = offset;

            for (int idim = 0; idim < N; idim++) {
                Psi[idim] = FFTWGrid<N>(Np);
                Psi[idim].add_memory_label("FFTWGrid::displacement_field_from_particles::Psi_" + std::to_string(idim));
            }
            const auto Local_nx = Psi[0].get_local_nx();
            const auto Local_x_start = Psi[0].get_local_x_start();
            const int np_per_task = Np / FML::NTasks;

            // Find the lattice node of each particle and the displacement
            struct Record {
                int coord[N];
                double psi[N];
            };
            std::vector<std::vector<Record>> send(FML::NTasks);
            int not_on_lattice = 0;
            for (size_t ip = 0; ip < NumPart; ip++) {
                const double * q = FML::PARTICLE::GetLagrangianPos(const_cast<T *>(part)[ip]);
                const double * x = FML::PARTICLE::GetPos(const_cast<T *>(part)[ip]);
                Record r;
                for (int idim = 0; idim < N; idim++) {
                    const double u = q[idim] * Np - offset[idim];
                    const double iu = std::round(u);
                    if (std::fabs(u - iu) > 1e-4)
                        not_on_lattice = 1;
                    r.coord[idim] = ((long long)(iu) % Np + Np) % Np;
                    double d = x[idim] - q[idim];
                    d -= std::round(d);
                    r.psi[idim] = d;
                }
                const int owner = FML::NTasks == 1 ? 0 : r.coord[0] / np_per_task;
                send[owner].push_back(r);
            }
            FML::MaxOverTasks(&not_on_lattice);
            assert_mpi(not_on_lattice == 0,
                       "[displacement_field_from_particles] The Lagrangian coordinates are not on a regular lattice\n");

            std::vector<Record> recv;
#ifdef USE_MPI
            recv = detail::alltoallv_elements(send);
#else
            recv = send[0];
#endif

            // Fill the grids and check that every node gets exactly one particle
            const size_t ncells_local = size_t(Local_nx) * size_t(FML::power(Np, N - 1));
            std::vector<char> filled(ncells_local, 0);
            int duplicates = 0;
            for (auto & r : recv) {
                std::array<int, N> coord;
                size_t ilocal = 0;
                for (int idim = 0; idim < N; idim++) {
                    coord[idim] = r.coord[idim] - (idim == 0 ? int(Local_x_start) : 0);
                    ilocal = ilocal * Np + coord[idim];
                }
                if (filled[ilocal])
                    duplicates = 1;
                filled[ilocal] = 1;
                for (int idim = 0; idim < N; idim++)
                    Psi[idim].set_real(coord, FloatType(r.psi[idim]));
            }
            long long nfilled = std::count(filled.begin(), filled.end(), char(1));
            FML::SumOverTasks(&nfilled);
            FML::MaxOverTasks(&duplicates);
            assert_mpi(duplicates == 0 and nfilled == (long long)FML::power(size_t(Np), N),
                       "[displacement_field_from_particles] Every lattice node must have exactly one particle\n");
        }

        //================================================================================
        /// @brief Create the tracers \f$ x = q_i + s + \Psi(q_i + s) \f$ for one shift s (box units). On return
        /// the tracers are on the task that owns their x-coordinate.
        ///
        /// @param[in] Psi_fourier The displacement field in Fourier space (Nmesh = Npart_1D)
        /// @param[in] lattice_offset The lattice offset in units of the lattice spacing
        /// @param[in] shift The shift s
        /// @param[out] tracers The tracers
        //================================================================================
        template <int N>
        void sheet_tracers(const std::array<FFTWGrid<N>, N> & Psi_fourier,
                           const std::array<double, N> & lattice_offset,
                           const std::array<double, N> & shift,
                           std::vector<SheetTracer<N>> & tracers) {
            const int Np = Psi_fourier[0].get_nmesh();
            const auto Local_nx = Psi_fourier[0].get_local_nx();
            const auto Local_x_start = Psi_fourier[0].get_local_x_start();
            const size_t ncells_local = size_t(Local_nx) * size_t(FML::power(Np, N - 1));
            tracers.resize(ncells_local);

            for (int idim = 0; idim < N; idim++) {
                FFTWGrid<N> psi = Psi_fourier[idim];
                psi.add_memory_label("FFTWGrid::sheet_tracers::psi");
                FML::GRID::shift_grid_fourier<N>(psi, shift, +1.0);
                psi.fftw_c2r();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : psi.get_real_range(islice, islice + 1)) {
                        auto coord = psi.get_coord_from_index(real_index);
                        size_t ilocal = 0;
                        for (int jdim = 0; jdim < N; jdim++)
                            ilocal = ilocal * Np + coord[jdim];
                        const int iglobal = coord[idim] + (idim == 0 ? int(Local_x_start) : 0);
                        double x = (iglobal + lattice_offset[idim]) / double(Np) + shift[idim] +
                                   psi.get_real_from_index(real_index);
                        x -= std::floor(x);
                        if (x >= 1.0)
                            x -= 1.0;
                        tracers[ilocal].pos[idim] = x;
                    }
                }
            }
            redistribute_particles_by_x(tracers);
        }

        //================================================================================
        /// @brief The density contrast in Fourier space from spectral sheet interpolation of a given displacement
        /// field: the average of the density fields of the R^NDIM tracer sets (R = refinement). The window function
        /// of the density assignment is not deconvolved.
        ///
        /// @param[in] Psi_fourier The displacement field in Fourier space (Nmesh = Npart_1D)
        /// @param[in] lattice_offset The lattice offset in units of the lattice spacing
        /// @param[in] refinement R (1 means just the particles)
        /// @param[out] density_fourier The output (Nmesh must be set)
        /// @param[in] density_assignment_method NGP, CIC, TSC, PCS or PQS, or NUFFT for the exact Fourier modes of
        /// the tracers (no window function or aliasing; modes |m_i| < Nmesh/2, see FML::NUFFT)
        /// @param[in] nufft_epsilon The accuracy of the NUFFT (if used)
        //================================================================================
        template <int N>
        void sheet_density_fourier(const std::array<FFTWGrid<N>, N> & Psi_fourier,
                                   const std::array<double, N> & lattice_offset,
                                   int refinement,
                                   FFTWGrid<N> & density_fourier,
                                   std::string density_assignment_method,
                                   double nufft_epsilon = 1e-6) {
            const int nmesh = density_fourier.get_nmesh();
            const int Np = Psi_fourier[0].get_nmesh();
            assert_mpi(nmesh > 0, "[sheet_density_fourier] Output grid must be allocated\n");
            assert_mpi(refinement >= 1, "[sheet_density_fourier] refinement must be >= 1\n");
            const int R = refinement;
            const int nshifts = FML::power(R, N);
            const size_t NumTracersTot = FML::power(size_t(Np), N);

            // Exact Fourier modes of the tracers with a NUFFT (accumulated in Fourier space)
            if (density_assignment_method == "NUFFT") {
                FML::NUFFT::ESKernel kernel(nufft_epsilon);
                FFTWGrid<N> density_shift(nmesh);
                density_shift.add_memory_label("FFTWGrid::sheet_density_fourier::density_shift");
                FFTWGrid<N> sum(nmesh,
                                density_fourier.get_n_extra_slices_left(),
                                density_fourier.get_n_extra_slices_right());
                sum.add_memory_label("FFTWGrid::sheet_density_fourier::sum");
                sum.fill_fourier_grid(0.0);
                sum.set_grid_status_real(false);
                std::vector<SheetTracer<N>> tracers;
                std::vector<double> ones;
                for (int ishift = 0; ishift < nshifts; ishift++) {
                    std::array<double, N> shift;
                    for (int idim = 0, n = 1; idim < N; idim++, n *= R)
                        shift[idim] = ((ishift / n) % R) / double(R * Np);
                    sheet_tracers<N>(Psi_fourier, lattice_offset, shift, tracers);
                    ones.assign(tracers.size(), 1.0);
                    FML::NUFFT::nufft_type1<N, SheetTracer<N>>(tracers.data(),
                                                               tracers.size(),
                                                               ones,
                                                               1.0 / double(NumTracersTot * nshifts),
                                                               density_shift,
                                                               kernel);
                    const auto Local_nx = sum.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                    for (int islice = 0; islice < Local_nx; islice++) {
                        for (auto && fourier_index : sum.get_fourier_range(islice, islice + 1)) {
                            sum.set_fourier_from_index(fourier_index,
                                                       sum.get_fourier_from_index(fourier_index) +
                                                           density_shift.get_fourier_from_index(fourier_index));
                        }
                    }
                }
                if (sum.get_local_x_start() == 0)
                    sum.set_fourier_from_index(0, 0.0);
                density_fourier = sum;
                return;
            }

            auto nleftright = get_extra_slices_needed_for_density_assignment(density_assignment_method);
            FFTWGrid<N> density_shift(nmesh, nleftright.first, nleftright.second);
            density_shift.add_memory_label("FFTWGrid::sheet_density_fourier::density_shift");
            // Keep the extra slices of the output grid (they are needed if it is used for interpolation later)
            FFTWGrid<N> sum(nmesh,
                            density_fourier.get_n_extra_slices_left(),
                            density_fourier.get_n_extra_slices_right());
            sum.add_memory_label("FFTWGrid::sheet_density_fourier::sum");

            std::vector<SheetTracer<N>> tracers;
            for (int ishift = 0; ishift < nshifts; ishift++) {
                std::array<double, N> shift;
                for (int idim = 0, n = 1; idim < N; idim++, n *= R)
                    shift[idim] = ((ishift / n) % R) / double(R * Np);
                sheet_tracers<N>(Psi_fourier, lattice_offset, shift, tracers);
                particles_to_grid<N, SheetTracer<N>>(tracers.data(),
                                                     tracers.size(),
                                                     NumTracersTot,
                                                     density_shift,
                                                     density_assignment_method);
                const auto Local_nx = sum.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : sum.get_real_range(islice, islice + 1)) {
                        auto value = sum.get_real_from_index(real_index);
                        value += density_shift.get_real_from_index(real_index) / FloatType(nshifts);
                        sum.set_real_from_index(real_index, value);
                    }
                }
            }
            std::vector<SheetTracer<N>>().swap(tracers);

            sum.fftw_r2c();
            density_fourier = sum;
        }

        //================================================================================
        /// @brief The density contrast in Fourier space from spectral sheet interpolation of the particles' own
        /// displacement field \f$ \Psi = x - q \f$ (particles must have get_q and must have started on a regular
        /// lattice). See sheet_density_fourier.
        //================================================================================
        template <int N, class T>
        void particles_to_fourier_grid_sheet(const T * part,
                                             size_t NumPart,
                                             int Npart_1D,
                                             int refinement,
                                             FFTWGrid<N> & density_fourier,
                                             std::string density_assignment_method,
                                             double nufft_epsilon = 1e-6) {
            std::array<FFTWGrid<N>, N> Psi;
            std::array<double, N> lattice_offset;
            displacement_field_from_particles<N, T>(part, NumPart, Npart_1D, Psi, lattice_offset);
            for (int idim = 0; idim < N; idim++)
                Psi[idim].fftw_r2c();
            sheet_density_fourier<N>(Psi,
                                     lattice_offset,
                                     refinement,
                                     density_fourier,
                                     density_assignment_method,
                                     nufft_epsilon);
        }

    } // namespace INTERPOLATION
} // namespace FML
#endif
