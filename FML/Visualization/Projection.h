#ifndef FML_VISUALIZATION_PROJECTION_H
#define FML_VISUALIZATION_PROJECTION_H

// =====================================================================================================================
// Projections of distributed grids and particles to 2D images (Image.h), the same image on every task:
//
//   // A slab of thickness 0.1 (box units) through a 3D density grid (an FFTWGrid or a PencilGrid), seen along z
//   auto image = FML::VISUALIZATION::project_grid(delta, 2, 0.45, 0.1);
//   // The particles in the same slab, cloud in cell on 1024^2 pixels, in units of the mean (mass weighted if the
//   // particles have get_mass)
//   auto image2 = FML::VISUALIZATION::project_particles(part.get_particles_ptr(), part.get_npart(), 1024, 2, 0.45,
//                                                       0.1);
//   if (FML::ThisTask == 0)
//       FML::VISUALIZATION::write_png("slab.png", image2, "magma", FML::VISUALIZATION::ColorScale::automatic_log());
//
// Projecting along an axis: the image axes are the two other axes in order (along z: x to the right, y up; along y: x
// and z; along x: y and z). The slab [slab_start, slab_start + slab_thickness) is periodic (it may wrap past 1). The
// whole box: slab_start = 0, slab_thickness = 1 (the default). Collective: call on all tasks.
// =====================================================================================================================

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Visualization/Image.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace VISUALIZATION {
        namespace detail {
            /// The position of cell i of an nmesh grid along an axis (box units): FML's cells are at i / nmesh (the
            /// assignment and the Fourier transforms), at (i + 1/2) / nmesh with -DCELLCENTERSHIFTED
            inline double cell_position(int i, int nmesh) {
#ifdef CELLCENTERSHIFTED
                return (i + 0.5) / nmesh;
#else
                return double(i) / nmesh;
#endif
            }
            /// Is the coordinate x (box units, in [0,1)) in the periodic slab [start, start + thickness)?
            inline bool in_slab(double x, double start, double thickness) {
                if (thickness >= 1.0)
                    return true;
                double d = x - start;
                d -= std::floor(d);
                return d < thickness;
            }
            inline std::array<int, 2> image_axes(int N, int axis) {
                if (N == 2)
                    return {0, 1};
                if (axis == 0)
                    return {1, 2};
                if (axis == 1)
                    return {0, 2};
                return {0, 1};
            }
        } // namespace detail

        /// A projection of a real-space grid (FFTWGrid<N> or PencilGrid<N>, N = 2 or 3) along an axis: the sum or the
        /// mean (average = true) over the cells whose position (cell_position: i / Nmesh) is in the slab. An Nmesh x
        /// Nmesh image: pixel (ix, iy) is the cell at (ix, iy) / Nmesh. For N = 2 the image is the grid itself (axis
        /// and slab are ignored). Collective
        template <int N, template <int> class Grid>
        Image project_grid(const Grid<N> & grid,
                           int axis = N - 1,
                           double slab_start = 0.0,
                           double slab_thickness = 1.0,
                           bool average = true) {
            static_assert(N == 2 or N == 3, "[FML::VISUALIZATION::project_grid] Only for 2D and 3D grids");
            if (axis < 0 or axis >= N)
                throw std::runtime_error("[FML::VISUALIZATION::project_grid] The axis must be in [0, N)");
            if (not(slab_thickness > 0.0))
                throw std::runtime_error("[FML::VISUALIZATION::project_grid] The slab thickness must be positive");
            const int nmesh = grid.get_nmesh();
            const auto ax = detail::image_axes(N, axis);
            std::array<ptrdiff_t, N> offset;
            for (int d = 0; d < N; d++)
                offset[d] = grid.get_local_range(d)[0];

            // The number of cells along the axis that are in the slab (the same for every column)
            int ncells = 1;
            if (N == 3) {
                ncells = 0;
                for (int i = 0; i < nmesh; i++)
                    if (detail::in_slab(detail::cell_position(i, nmesh), slab_start, slab_thickness))
                        ncells++;
                if (ncells == 0)
                    throw std::runtime_error("[FML::VISUALIZATION::project_grid] No cell is in the slab: "
                                             "make it thicker than a cell");
            }

            Image image(nmesh, nmesh);
            for (auto && index : grid.get_real_range()) {
                auto coord = grid.get_coord_from_index(index);
                for (int d = 0; d < N; d++)
                    coord[d] += int(offset[d]);
                if (N == 3 and
                    not detail::in_slab(detail::cell_position(coord[axis], nmesh), slab_start, slab_thickness))
                    continue;
                image(coord[ax[0]], coord[ax[1]]) += double(grid.get_real_from_index(index));
            }
            FML::SumArrayOverTasks(image.data.data(), image.data.size());
            if (average and ncells > 1)
                for (auto & v : image.data)
                    v /= double(ncells);
            return image;
        }

        /// A projected density of particles (positions in [0,1)^N, N = 2 or 3) in a slab along an axis, on npix x npix
        /// pixels with NGP or CIC (periodic). Mass weighted if the particles have get_mass. In units of the mean pixel
        /// value (empty slab: zero). Collective
        template <class T>
        Image project_particles(const T * part,
                                size_t npart,
                                int npix,
                                int axis = 2,
                                double slab_start = 0.0,
                                double slab_thickness = 1.0,
                                const std::string & method = "CIC") {
            if (npix <= 0)
                throw std::runtime_error("[FML::VISUALIZATION::project_particles] npix must be positive");
            if (method != "NGP" and method != "CIC")
                throw std::runtime_error("[FML::VISUALIZATION::project_particles] Unknown method [" + method +
                                         "] (NGP or CIC)");
            if (not(slab_thickness > 0.0))
                throw std::runtime_error("[FML::VISUALIZATION::project_particles] The slab thickness must be positive");
            T tmp{};
            constexpr int N = FML::PARTICLE::GetNDIM(tmp);
            static_assert(N == 2 or N == 3, "[FML::VISUALIZATION::project_particles] Only for 2D and 3D particles");
            if (axis < 0 or axis >= N)
                throw std::runtime_error("[FML::VISUALIZATION::project_particles] The axis must be in [0, N)");
            const auto ax = detail::image_axes(N, axis);

            Image image(npix, npix);
            auto wrap = [npix](int i) { return i >= npix ? i - npix : (i < 0 ? i + npix : i); };
            for (size_t i = 0; i < npart; i++) {
                const auto * pos = FML::PARTICLE::GetPos(const_cast<T &>(part[i]));
                if (N == 3 and not detail::in_slab(double(pos[axis]), slab_start, slab_thickness))
                    continue;
                double mass = 1.0;
                if constexpr (FML::PARTICLE::has_get_mass<T>())
                    mass = double(FML::PARTICLE::GetMass(const_cast<T &>(part[i])));
                double x = double(pos[ax[0]]) * npix, y = double(pos[ax[1]]) * npix;
                if (method == "NGP") {
                    image(wrap(int(std::floor(x)) % npix), wrap(int(std::floor(y)) % npix)) += mass;
                } else {
                    x -= 0.5;
                    y -= 0.5;
                    const double fx = std::floor(x), fy = std::floor(y);
                    const double dx = x - fx, dy = y - fy;
                    const int ix = wrap(int(fx) % npix), iy = wrap(int(fy) % npix);
                    const int jx = wrap(ix + 1), jy = wrap(iy + 1);
                    image(ix, iy) += mass * (1 - dx) * (1 - dy);
                    image(jx, iy) += mass * dx * (1 - dy);
                    image(ix, jy) += mass * (1 - dx) * dy;
                    image(jx, jy) += mass * dx * dy;
                }
            }
            FML::SumArrayOverTasks(image.data.data(), image.data.size());
            const double mean = image.mean();
            if (mean > 0.0)
                for (auto & v : image.data)
                    v /= mean;
            return image;
        }
    } // namespace VISUALIZATION
} // namespace FML
#endif
