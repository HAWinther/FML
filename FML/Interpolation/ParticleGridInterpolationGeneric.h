#ifndef FML_PARTICLE_GRID_INTERPOLATION_GENERIC_H
#define FML_PARTICLE_GRID_INTERPOLATION_GENERIC_H

// =====================================================================================================================
// Density assignment and interpolation for any grid with the interface of FFTWGrid/GridInterface.h
// (e.g. PencilGrid<N>). FFTWGrid keeps its own (optimized) implementation in ParticleGridInterpolation.h: these
// overloads are chosen for the other grid types (an FFTWGrid<N>& parameter is more specialized than Grid<N>&).
//
// The particles must be in the domain of the grid (for a PencilGrid: get_domain().particle_domain()). An axis whose
// local range (get_local_range) is all of it is periodic within the task; on the others the kernel reaches into the
// extra cells, which are added to their owners (add_contribution_from_extra_cells) or must be filled before
// interpolating (communicate_boundaries).
// =====================================================================================================================

#include <FML/FFTWGrid/GridInterface.h>
// (the kernels; this header is also included at the end of it)
#include <FML/Interpolation/ParticleGridInterpolation.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace FML {
    namespace INTERPOLATION {

        namespace detail {
            // Does the task have only part of the axis (so the kernel uses the extra cells), or all of it (periodic)?
            template <class Grid>
            bool has_part_of_axis(const Grid & grid, int axis) {
                return grid.get_local_range(axis)[1] < grid.get_nmesh();
            }

            // The cell of a position and the first cell of the kernel along each axis, in local coordinates (the
            // axes split over the tasks: relative to the local range, clamped into it against round-off; the others:
            // periodic). x is returned as the distance from the cell (in units of cells)
            template <int N, int ORDER, class Grid, class Pos>
            void generic_kernel_cells(const Grid & grid,
                                      const Pos * pos,
                                      std::array<double, N> & x,
                                      std::array<int, N> & ix,
                                      std::array<int, N> & xstart,
                                      int beyond_right = 0) {
                const int Nmesh = grid.get_nmesh();
                for (int idim = 0; idim < N; idim++) {
                    x[idim] = pos[idim] * Nmesh;
                    ix[idim] = int(x[idim]);
                    if (has_part_of_axis(grid, idim)) {
                        const auto range = grid.get_local_range(idim);
                        int local = ix[idim] - int(range[0]);
                        // (beyond_right: the positions shifted by less than a cell can be in the next cell)
                        const int clamped = std::max(0, std::min(local, int(range[1]) - 1 + beyond_right));
                        x[idim] -= ix[idim] + (clamped - local);
                        ix[idim] = clamped;
                        // (only round-off may put a particle outside: see check_cells_outside_local_domain)
                        if (clamped != local)
                            check_cells_outside_local_domain(x[idim] < 0.0 ? -x[idim] : x[idim] - 1.0,
                                                             "[particles_to_grid / interpolation]");
                    } else {
                        if (ix[idim] >= Nmesh)
                            ix[idim] -= Nmesh;
                        x[idim] -= int(x[idim]);
                    }
                }
                for (int idim = 0; idim < N; idim++) {
#ifdef CELLCENTERSHIFTED
                    xstart[idim] = -ORDER / 2;
                    if (ORDER % 2 == 0 and x[idim] > 0.5)
                        xstart[idim] += 1;
#else
                    if (ORDER % 2 == 0)
                        xstart[idim] = -ORDER / 2 + 1;
                    else
                        xstart[idim] = -ORDER / 2 + (x[idim] > 0.5 ? 1 : 0);
#endif
                }
            }

            // The (local) cell and the weight of kernel cell number i (of ORDER^N)
            template <int N, int ORDER, class Grid>
            double generic_kernel_cell(const Grid & grid,
                                       int i,
                                       const std::array<double, N> & x,
                                       const std::array<int, N> & ix,
                                       const std::array<int, N> & xstart,
                                       std::array<int, N> & icoord) {
                const int Nmesh = grid.get_nmesh();
                double w = 1.0;
                for (int idim = 0, n = 1; idim < N; idim++, n *= ORDER) {
                    const int step = ORDER == 1 ? 0 : xstart[idim] + (i / n % ORDER);
#ifdef CELLCENTERSHIFTED
                    const double dx = std::fabs(-x[idim] + step + 0.5);
#else
                    const double dx = std::fabs(-x[idim] + step);
#endif
                    if constexpr (ORDER > 1)
                        w *= kernel<ORDER>(dx);
                    int c = ix[idim] + step;
                    if (not has_part_of_axis(grid, idim)) {
                        if (c >= Nmesh)
                            c -= Nmesh;
                        if (c < 0)
                            c += Nmesh;
                    }
                    icoord[idim] = c;
                }
                return w;
            }

            // The kernel of a particle as a separable stencil: the weights and the index offsets along each axis are
            // computed once (ORDER per axis) and combined over the ORDER^N cells. The index of a cell is affine in its
            // coordinates (FFTWGrid and PencilGrid), so the index of a cell is the base index plus one offset per
            // axis. Visits the cells in the order of generic_kernel_cell (the first axis fastest) with the same
            // weights (the product of the 1D weights from the first axis)
            template <int N, int ORDER>
            struct KernelStencil {
                std::array<std::array<double, ORDER>, N> w;
                std::array<std::array<IndexIntType, ORDER>, N> off;
                IndexIntType base;

                template <class Grid>
                KernelStencil(const Grid & grid,
                              const std::array<double, N> & x,
                              const std::array<int, N> & ix,
                              const std::array<int, N> & xstart) {
                    const int Nmesh = grid.get_nmesh();
                    std::array<std::array<int, ORDER>, N> cell;
                    for (int idim = 0; idim < N; idim++) {
                        const bool wrap = not has_part_of_axis(grid, idim);
                        for (int s = 0; s < ORDER; s++) {
                            const int step = ORDER == 1 ? 0 : xstart[idim] + s;
#ifdef CELLCENTERSHIFTED
                            const double dx = std::fabs(-x[idim] + step + 0.5);
#else
                            const double dx = std::fabs(-x[idim] + step);
#endif
                            if constexpr (ORDER > 1)
                                w[idim][s] = kernel<ORDER>(dx);
                            else
                                w[idim][s] = 1.0;
                            int c = ix[idim] + step;
                            if (wrap) {
                                if (c >= Nmesh)
                                    c -= Nmesh;
                                if (c < 0)
                                    c += Nmesh;
                            }
                            cell[idim][s] = c;
                        }
                    }
                    std::array<int, N> first;
                    for (int idim = 0; idim < N; idim++)
                        first[idim] = cell[idim][0];
                    base = grid.get_index_real(first);
                    for (int idim = 0; idim < N; idim++)
                        for (int s = 0; s < ORDER; s++) {
                            auto c = first;
                            c[idim] = cell[idim][s];
                            off[idim][s] = s == 0 ? 0 : grid.get_index_real(c) - base;
                        }
                }

                /// f(index, weight) for every cell of the kernel
                template <class F>
                void for_each(F && f) const {
                    if constexpr (N == 3) {
                        for (int s2 = 0; s2 < ORDER; s2++)
                            for (int s1 = 0; s1 < ORDER; s1++) {
                                const IndexIntType i12 = base + off[1][s1] + off[2][s2];
                                for (int s0 = 0; s0 < ORDER; s0++)
                                    f(i12 + off[0][s0], w[0][s0] * w[1][s1] * w[2][s2]);
                            }
                    } else {
                        for (int i = 0; i < FML::power(ORDER, N); i++) {
                            IndexIntType index = base;
                            double weight = 1.0;
                            for (int idim = 0, n = 1; idim < N; idim++, n *= ORDER) {
                                const int s = i / n % ORDER;
                                index += off[idim][s];
                                weight *= w[idim][s];
                            }
                            f(index, weight);
                        }
                    }
                }
            };

            template <int ORDER, class Grid>
            void check_extra_cells(const Grid & grid, int ndim, const char * name) {
                const auto nextra = get_extra_slices_needed_by_order<ORDER>();
                bool ok =
                    grid.get_n_extra_slices_left() >= nextra.first and grid.get_n_extra_slices_right() >= nextra.second;
                for (int idim = 0; idim < ndim; idim++)
                    if (has_part_of_axis(grid, idim))
                        ok = ok and grid.get_local_range(idim)[1] >= std::max(nextra.first, nextra.second);
                assert_mpi(ok,
                           (std::string("[") + name +
                            "] Too few extra cells for the kernel, or fewer local cells than extra cells on an axis")
                               .c_str());
            }

            // 1 if the grid has one more right extra cell than the kernel needs: then positions up to a cell beyond the
            // right edge of the local range are allowed (particles shifted by less than a cell that stay on their task)
            template <int ORDER, class Grid>
            int spare_right_cell(const Grid & grid) {
                return grid.get_n_extra_slices_right() >= get_extra_slices_needed_by_order<ORDER>().second + 1 ? 1 : 0;
            }
        } // namespace detail

        /// Density assignment (delta = rho / mean - 1; delta-f particles: the perturbation) for any grid with the
        /// interface (not FFTWGrid: it has its own). The grid needs the extra cells of the kernel
        /// (get_extra_slices_needed_by_order<ORDER>())
        /// Deposit at the virtual positions x + assignment_offset (0 <= offset < 1/Nmesh; the particles are not
        /// changed): wrapped periodically along the axes the task has all of, else into the extra cells (one more on the
        /// right than the kernel needs). wrap_x (one task with FFTWGrid) is not needed: the rule above covers it
        template <int N,
                  int ORDER,
                  class T,
                  template <int>
                  class Grid,
                  std::enable_if_t<not FML::GRID::is_fftwgrid_v<Grid<N>>, int> = 0>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               Grid<N> & density,
                               const std::array<double, N> & assignment_offset,
                               [[maybe_unused]] bool wrap_x = false) {
            static_assert(FML::GRID::has_fourier_grid_interface<Grid<N>>::value, "particles_to_grid: not a grid");
            detail::check_extra_cells<ORDER>(density, N, "particles_to_grid");
            bool shifted = false;
            for (double offset : assignment_offset) {
                assert_mpi(std::isfinite(offset) and offset >= 0.0 and offset < 1.0 / density.get_nmesh(),
                           "[particles_to_grid] Assignment offsets must be in [0, 1/Nmesh)");
                shifted = shifted or offset != 0.0;
            }
            if (shifted) {
                const auto nextra = get_extra_slices_needed_by_order<ORDER>();
                assert_mpi(density.get_n_extra_slices_right() >= nextra.second + 1,
                           "[particles_to_grid] A positive offset needs one additional right extra cell");
            }

            constexpr bool has_deltaf_weight = FML::PARTICLE::has_get_deltaf_weight<T>();
            constexpr bool has_mass = FML::PARTICLE::has_get_mass<T>() and not has_deltaf_weight;
            const int Nmesh = density.get_nmesh();
            assert_mpi(NumPartTot > 0, "[particles_to_grid] No particles (NumPartTot = 0): the density is undefined");
            double norm_fac = std::pow(double(Nmesh), N) / double(NumPartTot);
            if constexpr (has_mass) {
                double mean_mass = 0.0;
                for (size_t i = 0; i < NumPart; i++)
                    mean_mass += FML::PARTICLE::GetMass(part[i]);
                FML::SumOverTasks(&mean_mass);
                norm_fac /= mean_mass / double(NumPartTot);
            }

            // Deposit into the main and the extra cells, add the extra cells to their owners, subtract the mean
            const int beyond_right = detail::spare_right_cell<ORDER>(density);
            density.fill_real_grid(0.0);
            auto * grid = density.get_real_grid();

            // The (virtual) position of a particle: shifted by the offset, rounded to the position type as an actual
            // shift
            using Position = std::decay_t<decltype(FML::PARTICLE::GetPos(const_cast<T *>(part)[0])[0])>;
            auto position = [&](size_t ind) {
                const auto * pos0 = FML::PARTICLE::GetPos(const_cast<T *>(part)[ind]);
                std::array<Position, N> pos;
                for (int idim = 0; idim < N; idim++) {
                    pos[idim] = pos0[idim];
                    if (shifted) {
                        pos[idim] += assignment_offset[idim];
                        if (not detail::has_part_of_axis(density, idim))
                            pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(pos[idim]);
                    }
                }
                return pos;
            };

            // Add one particle to the grid (ATOMIC: when several threads can write to the same cell)
            auto add_particle = [&](size_t ind, auto atomic_tag) {
                constexpr bool ATOMIC = decltype(atomic_tag)::value;
                const auto pos = position(ind);
                double mass = 1.0;
                if constexpr (has_mass)
                    mass = FML::PARTICLE::GetMass(part[ind]);
                if constexpr (has_deltaf_weight)
                    mass = FML::PARTICLE::GetDeltafWeight(part[ind]);
                std::array<double, N> x;
                std::array<int, N> ix, xstart, icoord;
                detail::generic_kernel_cells<N, ORDER>(density, pos.data(), x, ix, xstart, beyond_right);
                (void)icoord;
                const detail::KernelStencil<N, ORDER> stencil(density, x, ix, xstart);
                stencil.for_each([&](IndexIntType index, double w) {
                    if constexpr (ATOMIC) {
#ifdef USE_OMP
#pragma omp atomic
#endif
                        grid[index] += FloatType(w * norm_fac * mass);
                    } else {
                        grid[index] += FloatType(w * norm_fac * mass);
                    }
                });
            };

            // One thread: the particles in order. Several threads, without atomics: an axis the task has all of
            // (periodic; the last one for a PencilGrid) is split into an even number nchunks >= 4 of chunks of at
            // least 2 ORDER + 2 cells, so the cells written by the particles in chunks c and c + 2 never overlap. The
            // particles of the even chunks are added in parallel, then those of the odd chunks, each chunk in the
            // order of the particles: every cell gets its contributions in the same order for any number of threads
            // > 1 (deterministic; differs from one thread only by round-off from the order of the sums). Atomic adds
            // only if there is no such axis or it is too short for 4 chunks
#ifdef USE_OMP
            const int nthreads = omp_get_max_threads();
#else
            const int nthreads = 1;
#endif
            int axis = -1;
            for (int idim = 0; idim < N; idim++)
                if (not detail::has_part_of_axis(density, idim))
                    axis = idim;
            int nchunks = axis < 0 ? 0 : Nmesh / (2 * ORDER + 2);
            if (nchunks % 2 == 1)
                nchunks--;
            if (nthreads == 1) {
                for (size_t ind = 0; ind < NumPart; ind++)
                    add_particle(ind, std::false_type{});
            } else if (nchunks < 4 or NumPart >= size_t(std::numeric_limits<uint32_t>::max())) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NumPart; ind++)
                    add_particle(ind, std::true_type{});
            } else {
                // Counting sort of the particle indices by the chunk of their cell along the axis (stable)
                std::vector<uint32_t> chunk(NumPart);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < NumPart; ind++) {
                    int icell = int(double(position(ind)[axis]) * Nmesh) % Nmesh;
                    if (icell < 0)
                        icell += Nmesh;
                    chunk[ind] = uint32_t((long long)(icell)*nchunks / Nmesh);
                }
                std::vector<size_t> start(nchunks + 1, 0);
                for (size_t ind = 0; ind < NumPart; ind++)
                    start[chunk[ind] + 1]++;
                for (int c = 0; c < nchunks; c++)
                    start[c + 1] += start[c];
                std::vector<uint32_t> order(NumPart);
                {
                    std::vector<size_t> next(start.begin(), start.end() - 1);
                    for (size_t ind = 0; ind < NumPart; ind++)
                        order[next[chunk[ind]]++] = uint32_t(ind);
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
            density.add_contribution_from_extra_cells();
            if constexpr (not has_deltaf_weight) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (IndexIntType irow = 0; irow < IndexIntType(density.get_local_nrows()); irow++)
                    for (auto && index : density.get_real_range_of_rows(irow, irow + 1))
                        density.set_real_from_index(index, density.get_real_from_index(index) - FloatType(1.0));
            }
            density.set_grid_status_real(true);
        }

        /// Density assignment of order ORDER (1 = NGP, ..., 5 = PQS) to a grid of any type without an offset (see above)
        template <int N, int ORDER, class T, template <int> class Grid>
        void particles_to_grid(const T * part, size_t NumPart, size_t NumPartTot, Grid<N> & density) {
            particles_to_grid<N, ORDER, T>(part, NumPart, NumPartTot, density, std::array<double, N>{}, false);
        }

        /// The same with the method by name and an offset (see above; declared in ParticleGridInterpolation.h)
        template <int N, class T, template <int> class Grid, std::enable_if_t<not FML::GRID::is_fftwgrid_v<Grid<N>>, int>>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               Grid<N> & density,
                               std::string density_assignment_method,
                               const std::array<double, N> & assignment_offset,
                               bool wrap_x) {
            const int order = interpolation_order_from_name(density_assignment_method);
            if (order == 1)
                particles_to_grid<N, 1, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (order == 2)
                particles_to_grid<N, 2, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (order == 3)
                particles_to_grid<N, 3, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (order == 4)
                particles_to_grid<N, 4, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
            if (order == 5)
                particles_to_grid<N, 5, T>(part, NumPart, NumPartTot, density, assignment_offset, wrap_x);
        }

        /// The same with the method by name (NGP, CIC, TSC, PCS, PQS)
        template <int N, class T, template <int> class Grid>
        void particles_to_grid(const T * part,
                               size_t NumPart,
                               size_t NumPartTot,
                               Grid<N> & density,
                               std::string density_assignment_method) {
            const int order = interpolation_order_from_name(density_assignment_method);
            if (order == 1)
                particles_to_grid<N, 1, T>(part, NumPart, NumPartTot, density);
            if (order == 2)
                particles_to_grid<N, 2, T>(part, NumPart, NumPartTot, density);
            if (order == 3)
                particles_to_grid<N, 3, T>(part, NumPart, NumPartTot, density);
            if (order == 4)
                particles_to_grid<N, 4, T>(part, NumPart, NumPartTot, density);
            if (order == 5)
                particles_to_grid<N, 5, T>(part, NumPart, NumPartTot, density);
        }

        /// Interpolation of a real grid to the particle positions for any grid with the interface (not FFTWGrid: it has
        /// its own). The extra cells must be filled (communicate_boundaries)
        template <int N, int ORDER, class T, template <int> class Grid>
        void interpolate_grid_to_particle_positions(const Grid<N> & grid,
                                                    const T * part,
                                                    size_t NumPart,
                                                    std::vector<FloatType> & interpolated_values) {
            static_assert(FML::GRID::has_fourier_grid_interface<Grid<N>>::value,
                          "interpolate_grid_to_particle_positions: not a grid");
            detail::check_extra_cells<ORDER>(grid, N, "interpolate_grid_to_particle_positions");
            interpolated_values.resize(NumPart);
            const int beyond_right = detail::spare_right_cell<ORDER>(grid);
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t ind = 0; ind < NumPart; ind++) {
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[ind]);
                std::array<double, N> x;
                std::array<int, N> ix, xstart, icoord;
                detail::generic_kernel_cells<N, ORDER>(grid, pos, x, ix, xstart, beyond_right);
                (void)icoord;
                FloatType value = 0;
                const detail::KernelStencil<N, ORDER> stencil(grid, x, ix, xstart);
                stencil.for_each(
                    [&](IndexIntType index, double w) { value += grid.get_real_from_index(index) * FloatType(w); });
                interpolated_values[ind] = value;
            }
        }

        /// The same with the method by name (NGP, CIC, TSC, PCS, PQS)
        template <int N, class T, template <int> class Grid>
        void interpolate_grid_to_particle_positions(const Grid<N> & grid,
                                                    const T * part,
                                                    size_t NumPart,
                                                    std::vector<FloatType> & interpolated_values,
                                                    std::string interpolation_method) {
            const int order = interpolation_order_from_name(interpolation_method);
            if (order == 1)
                interpolate_grid_to_particle_positions<N, 1, T>(grid, part, NumPart, interpolated_values);
            if (order == 2)
                interpolate_grid_to_particle_positions<N, 2, T>(grid, part, NumPart, interpolated_values);
            if (order == 3)
                interpolate_grid_to_particle_positions<N, 3, T>(grid, part, NumPart, interpolated_values);
            if (order == 4)
                interpolate_grid_to_particle_positions<N, 4, T>(grid, part, NumPart, interpolated_values);
            if (order == 5)
                interpolate_grid_to_particle_positions<N, 5, T>(grid, part, NumPart, interpolated_values);
        }

        /// Interpolate the N components of a vector field (e.g. the force) to the particle positions and call
        /// apply(index, value) for each particle (value: an array<FloatType, N>); returns the local maximum of what
        /// apply returns (as the FFTWGrid version). For any grid with the interface (not FFTWGrid: it has its own). The
        /// extra cells must be filled (communicate_boundaries)
        template <int N,
                  int ORDER,
                  class T,
                  template <int>
                  class Grid,
                  class Apply,
                  std::enable_if_t<not std::is_same_v<Grid<N>, FML::GRID::FFTWGrid<N>>, int> = 0>
        double interpolate_grid_vector_to_particle_positions_apply(const std::array<Grid<N>, N> & grid,
                                                                   const T * part,
                                                                   size_t NumPart,
                                                                   Apply && apply) {
            static_assert(FML::GRID::has_fourier_grid_interface<Grid<N>>::value,
                          "interpolate_grid_vector_to_particle_positions_apply: not a grid");
            detail::check_extra_cells<ORDER>(grid[0], N, "interpolate_grid_vector_to_particle_positions_apply");
            double result = 0.0;
            const int beyond_right = detail::spare_right_cell<ORDER>(grid[0]);
            // The components share the indices if they have the same extra cells (the index depends on them)
            bool same_layout = true;
            for (int c = 1; c < N; c++)
                same_layout = same_layout and grid[c].get_n_extra_slices_left() == grid[0].get_n_extra_slices_left() and
                              grid[c].get_n_extra_slices_right() == grid[0].get_n_extra_slices_right() and
                              grid[c].get_local_range(0) == grid[0].get_local_range(0) and
                              grid[c].get_local_range(1) == grid[0].get_local_range(1);
#ifdef USE_OMP
#pragma omp parallel for reduction(max : result)
#endif
            for (size_t ind = 0; ind < NumPart; ind++) {
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T *>(part)[ind]);
                std::array<double, N> x;
                std::array<int, N> ix, xstart, icoord;
                detail::generic_kernel_cells<N, ORDER>(grid[0], pos, x, ix, xstart, beyond_right);
                std::array<FloatType, N> value{};
                if (same_layout) {
                    const detail::KernelStencil<N, ORDER> stencil(grid[0], x, ix, xstart);
                    stencil.for_each([&](IndexIntType index, double w) {
                        for (int c = 0; c < N; c++)
                            value[c] += grid[c].get_real_from_index(index) * FloatType(w);
                    });
                } else {
                    for (int i = 0; i < FML::power(ORDER, N); i++) {
                        const double w = detail::generic_kernel_cell<N, ORDER>(grid[0], i, x, ix, xstart, icoord);
                        for (int c = 0; c < N; c++)
                            value[c] += grid[c].get_real(icoord) * FloatType(w);
                    }
                }
                result = std::max(result, double(apply(ind, value)));
            }
            return result;
        }

        /// The same with the method by name (NGP, CIC, TSC, PCS, PQS)
        template <int N,
                  class T,
                  template <int>
                  class Grid,
                  class Apply,
                  std::enable_if_t<not std::is_same_v<Grid<N>, FML::GRID::FFTWGrid<N>>, int> = 0>
        double interpolate_grid_vector_to_particle_positions_apply(const std::array<Grid<N>, N> & grid,
                                                                   const T * part,
                                                                   size_t NumPart,
                                                                   std::string interpolation_method,
                                                                   Apply && apply) {
            const int order = interpolation_order_from_name(interpolation_method);
            if (order == 1)
                return interpolate_grid_vector_to_particle_positions_apply<N, 1, T>(grid, part, NumPart, apply);
            if (order == 2)
                return interpolate_grid_vector_to_particle_positions_apply<N, 2, T>(grid, part, NumPart, apply);
            if (order == 3)
                return interpolate_grid_vector_to_particle_positions_apply<N, 3, T>(grid, part, NumPart, apply);
            if (order == 4)
                return interpolate_grid_vector_to_particle_positions_apply<N, 4, T>(grid, part, NumPart, apply);
            return interpolate_grid_vector_to_particle_positions_apply<N, 5, T>(grid, part, NumPart, apply);
        }

        /// Interpolate the N components of a vector field to the particle positions (for any grid with the interface,
        /// not FFTWGrid: it has its own). The extra cells must be filled (communicate_boundaries)
        template <int N,
                  int ORDER,
                  class T,
                  template <int>
                  class Grid,
                  std::enable_if_t<not std::is_same_v<Grid<N>, FML::GRID::FFTWGrid<N>>, int> = 0>
        void interpolate_grid_vector_to_particle_positions(const std::array<Grid<N>, N> & grid,
                                                           const T * part,
                                                           size_t NumPart,
                                                           std::array<std::vector<FloatType>, N> & interpolated_values) {
            for (int idim = 0; idim < N; idim++)
                interpolated_values[idim].resize(NumPart);
            interpolate_grid_vector_to_particle_positions_apply<N, ORDER, T>(
                grid, part, NumPart, [&](size_t i, const std::array<FloatType, N> & value) {
                    for (int idim = 0; idim < N; idim++)
                        interpolated_values[idim][i] = value[idim];
                    return 0.0;
                });
        }

        /// The same with the method by name (NGP, CIC, TSC, PCS, PQS)
        template <int N,
                  class T,
                  template <int>
                  class Grid,
                  std::enable_if_t<not std::is_same_v<Grid<N>, FML::GRID::FFTWGrid<N>>, int> = 0>
        void interpolate_grid_vector_to_particle_positions(const std::array<Grid<N>, N> & grid,
                                                           const T * part,
                                                           size_t NumPart,
                                                           std::array<std::vector<FloatType>, N> & interpolated_values,
                                                           std::string interpolation_method) {
            const int order = interpolation_order_from_name(interpolation_method);
            if (order == 1)
                interpolate_grid_vector_to_particle_positions<N, 1, T>(grid, part, NumPart, interpolated_values);
            if (order == 2)
                interpolate_grid_vector_to_particle_positions<N, 2, T>(grid, part, NumPart, interpolated_values);
            if (order == 3)
                interpolate_grid_vector_to_particle_positions<N, 3, T>(grid, part, NumPart, interpolated_values);
            if (order == 4)
                interpolate_grid_vector_to_particle_positions<N, 4, T>(grid, part, NumPart, interpolated_values);
            if (order == 5)
                interpolate_grid_vector_to_particle_positions<N, 5, T>(grid, part, NumPart, interpolated_values);
        }

    } // namespace INTERPOLATION
} // namespace FML

#endif
