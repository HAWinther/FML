#ifndef FML_ANALYSIS_IMAGES_H
#define FML_ANALYSIS_IMAGES_H

#include <FML/Analysis/AnalysisContext.h>
#include <FML/Analysis/AnalysisGrid.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Visualization/Visualization.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// Pictures of fields at an output (SimulationCode and SnapshotAnalyzer, images = true): a slab of a field on a grid
// projected along an axis (the mean over the slab), coloured and written as a PNG (and optionally .npy) in the output
// folder: image_<field>_z<z>.png. FML/Visualization does the projection, colours and files. Here: the settings, the
// fields that come from the particles (the density and the velocity divergence) and the default look of each field.
// SimulationCode adds the fields that need the simulation (neutrinos, baryons - CDM, the fifth force).
//=============================================================================

namespace FML {
    namespace ANALYSIS {

        //=============================================================================
        /// The images: which fields, the grid, the slab and the colour map. Keys images_<name> (SimulationCode and
        /// SnapshotAnalyzer)
        //=============================================================================
        struct ImageSettings {
            static constexpr const char * prefix = "images_";
            std::vector<std::string> fields;
            int nmesh;
            int npix;
            int axis;
            double slab_center;
            double slab_thickness;
            std::string density_assignment_method;
            std::string colormap;
            bool npy;

            ImageSettings() {
                parameters([](const char *, auto & value, const auto & default_value, const char *) {
                    value = default_value;
                });
            }

            template <class F>
            void parameters(F && f) {
                f("fields",
                  fields,
                  std::vector<std::string>{"density"},
                  "The fields to draw, e.g. {\"density\", \"velocity_divergence\"}: density (1 + delta of the "
                  "particles, log colours), velocity_divergence (-div v / (a H) = f delta in linear theory); "
                  "SimulationCode also neutrinos (delta_nu), baryons_minus_cdm (delta_b - delta_c) and fifth_force "
                  "(|F_5th| / |F_Newton|)");
                f("nmesh", nmesh, 256, "Grid size of the fields");
                f("npix", npix, 0, "Pixels per side of the images (0: nmesh)");
                f("axis", axis, 2, "Projected along this axis (0, 1, 2: x, y, z)");
                f("slab_center", slab_center, 0.5, "Centre of the projected slab along the axis (units of the box)");
                f("slab_thickness",
                  slab_thickness,
                  0.1,
                  "Thickness of the projected slab (units of the box; 1: the whole box). The image is the mean over "
                  "it");
                f("density_assignment_method",
                  density_assignment_method,
                  std::string("CIC"),
                  "Density assignment method (NGP, CIC, TSC, PCS, PQS)");
                f("colormap",
                  colormap,
                  std::string(""),
                  "Colour map for all fields (magma, inferno, viridis, plasma, cividis, coolwarm, grey, cubehelix; "
                  "_r reverses). Empty: each field its own (density magma, signed fields coolwarm)");
                f("npy", npy, false, "Also write the projected values as .npy files (for numpy)");
            }

            void validate() const {
                if (fields.empty())
                    throw std::runtime_error("images_fields is empty");
                if (nmesh < 2 or (analysis_grid_is_fftwgrid<3> and nmesh % FML::NTasks != 0))
                    throw std::runtime_error("images_nmesh must be >= 2 and divisible by the number of MPI tasks");
                if (npix < 0)
                    throw std::runtime_error("images_npix must be >= 0 (0: images_nmesh)");
                if (axis < 0 or axis > 2)
                    throw std::runtime_error("images_axis must be 0, 1 or 2");
                if (not std::isfinite(slab_center))
                    throw std::runtime_error("images_slab_center must be a number");
                if (not(slab_thickness > 0.0 and slab_thickness <= 1.0))
                    throw std::runtime_error("images_slab_thickness must be in (0, 1] (units of the box)");
                if (slab_thickness * nmesh < 1.0)
                    throw std::runtime_error("images_slab_thickness must be at least one cell (1 / images_nmesh)");
                if (not FML::INTERPOLATION::is_known_interpolation_method(density_assignment_method))
                    throw std::runtime_error("images_density_assignment_method must be NGP, CIC, TSC, PCS or PQS");
                if (not colormap.empty())
                    FML::VISUALIZATION::colormap(colormap, 0.5); // throws for an unknown name
            }

            /// Throws if a field is not one of the allowed names
            void check_fields(const std::vector<std::string> & allowed, const std::string & why = "") const {
                for (auto & field : fields)
                    if (std::find(allowed.begin(), allowed.end(), field) == allowed.end()) {
                        std::string list;
                        for (auto & a : allowed)
                            list += (list.empty() ? "" : ", ") + a;
                        throw std::runtime_error("images_fields: [" + field + "] is not available here (" + list +
                                                 ")" + why);
                    }
            }

            bool wants(const std::string & field) const {
                return std::find(fields.begin(), fields.end(), field) != fields.end();
            }
        };

        /// The default colour map and scale of a field: the density (positive, large range) on a log scale, the signed
        /// fields symmetric about 0 with a diverging map, the fifth-force ratio linear
        inline std::pair<std::string, FML::VISUALIZATION::ColorScale> image_style(const std::string & field) {
            using FML::VISUALIZATION::ColorScale;
            if (field == "density")
                return {"magma", ColorScale::automatic_log(1.0, 99.9)};
            if (field == "fifth_force")
                return {"viridis", ColorScale::automatic_linear(0.5, 99.5)};
            return {"coolwarm", ColorScale::automatic_symmetric(99.0)};
        }

        /// The slab of the settings projected (the mean over the slab) and resized to npix. Collective
        template <class Grid>
        FML::VISUALIZATION::Image project_image(const Grid & real_grid, const ImageSettings & settings) {
            double start = settings.slab_center - 0.5 * settings.slab_thickness;
            start -= std::floor(start);
            auto image =
                FML::VISUALIZATION::project_grid(real_grid, settings.axis, start, settings.slab_thickness, true);
            if (settings.npix > 0 and settings.npix != image.nx)
                image = FML::VISUALIZATION::resize(image, settings.npix, settings.npix);
            return image;
        }

        /// Write the image of a field (and its .npy) in folder as image_<field>_z<redshift>.png, with the colours of
        /// the settings or image_style(field). Collective (task 0 writes). Returns the file name
        inline std::string write_field_image(const std::string & folder,
                                             const std::string & field,
                                             double redshift,
                                             const FML::VISUALIZATION::Image & image,
                                             const ImageSettings & settings) {
            auto [colormap, scale] = image_style(field);
            if (not settings.colormap.empty())
                colormap = settings.colormap;
            const std::string base =
                (folder.empty() ? "" : folder + "/") + "image_" + field + "_z" + redshift_label(redshift);
            if (FML::ThisTask == 0) {
                // A log scale needs positive values somewhere: otherwise linear
                if (scale.log and not(image.max() > 0.0))
                    scale = FML::VISUALIZATION::ColorScale::automatic_linear();
                FML::VISUALIZATION::write_png(base + ".png", image, colormap, scale);
                if (settings.npy)
                    FML::VISUALIZATION::write_npy(base + ".npy", image);
                std::cout << "Wrote " << base << ".png (" << field << ": min " << image.min() << " mean "
                          << image.mean() << " max " << image.max() << ")\n";
            }
            return base + ".png";
        }

        namespace detail {
            /// A particle with a position and a weight (for depositing a weighted quantity with particles_to_grid)
            struct WeightedPoint {
                double pos[3];
                double weight;
                double * get_pos() { return pos; }
                const double * get_pos() const { return pos; }
                double get_mass() const { return weight; }
                constexpr int get_ndim() const { return 3; }
            };

            /// sum_p w_p W(x - x_p) (any sign of w) on the grid in units of the mean particle number per cell, in
            /// real space. particles_to_grid gives sum w W / (<w> nbar) - 1 for positive weights, so we deposit
            /// w + c with c > 0 making them positive and subtract c times the number field
            template <class Grid>
            Grid deposit_weights(std::vector<WeightedPoint> & points,
                                 size_t ntot,
                                 int nmesh,
                                 const std::string & method,
                                 const Grid & number) {
                double shift = 0.0;
                for (auto & p : points)
                    shift = std::max(shift, std::fabs(p.weight));
                FML::MaxOverTasks(&shift);
                shift = shift > 0.0 ? 2.0 * shift : 1.0;
                double sum = 0.0;
                for (auto & p : points) {
                    p.weight += shift;
                    sum += p.weight;
                }
                FML::SumOverTasks(&sum);
                const double mean = sum / double(ntot);
                const auto nextra = FML::INTERPOLATION::get_extra_slices(method);
                auto grid = FML::GRID::make_grid<Grid>(nmesh, nextra.first, nextra.second);
                FML::INTERPOLATION::particles_to_grid(points.data(), points.size(), ntot, grid, method);
                for (auto && index : grid.get_real_range()) {
                    const double value = (double(grid.get_real_from_index(index)) + 1.0) * mean -
                                         shift * (double(number.get_real_from_index(index)) + 1.0);
                    grid.set_real_from_index(index, FML::GRID::FloatType(value));
                }
                for (auto & p : points)
                    p.weight -= shift;
                return grid;
            }
        } // namespace detail

        /// The density contrast delta of particles (mass weighted if they have masses) on an nmesh^3 grid in real
        /// space. Collective
        template <class Grid, class P>
        Grid particle_density_contrast(P * part, size_t npart, int nmesh, const std::string & method) {
            size_t ntot = npart;
            FML::SumOverTasks(&ntot);
            if (ntot == 0)
                throw std::runtime_error("[particle_density_contrast] No particles");
            const auto nextra = FML::INTERPOLATION::get_extra_slices(method);
            auto delta = FML::GRID::make_grid<Grid>(nmesh, nextra.first, nextra.second);
            FML::INTERPOLATION::particles_to_grid(part, npart, ntot, delta, method);
            return delta;
        }

        /// The velocity divergence theta = -div(v velocity_to_displacement) of particles in real space: with the
        /// factor converting a velocity to its redshift-space displacement in box units (v / (a H B) for a peculiar
        /// velocity v), theta = f delta in linear theory. The velocity field is the mass-weighted mean velocity
        /// (momentum / mass) of the cells; in cells without mass that of the fields smoothed with a Gaussian of two
        /// cells (zero if that has no mass either). Collective
        template <class Grid, class P>
        Grid particle_velocity_divergence(P * part,
                                          size_t npart,
                                          int nmesh,
                                          const std::string & method,
                                          double velocity_to_displacement) {
            static_assert(FML::PARTICLE::has_get_vel<P>(), "[particle_velocity_divergence] Particles need get_vel");
            size_t ntot = npart;
            FML::SumOverTasks(&ntot);
            if (ntot == 0)
                throw std::runtime_error("[particle_velocity_divergence] No particles");
            std::vector<detail::WeightedPoint> points(npart);
            for (size_t i = 0; i < npart; i++) {
                const auto * pos = FML::PARTICLE::GetPos(part[i]);
                for (int d = 0; d < 3; d++)
                    points[i].pos[d] = double(pos[d]);
                points[i].weight = 1.0;
            }
            // The number field (for the shift in deposit_weights) and the mass field M
            const auto nextra = FML::INTERPOLATION::get_extra_slices(method);
            auto number = FML::GRID::make_grid<Grid>(nmesh, nextra.first, nextra.second);
            FML::INTERPOLATION::particles_to_grid(points.data(), points.size(), ntot, number, method);
            for (size_t i = 0; i < npart; i++)
                points[i].weight = double(FML::PARTICLE::GetMass(part[i]));
            auto mass = detail::deposit_weights<Grid>(points, ntot, nmesh, method, number);

            // Cells without mass (voids at the resolution of the grid) take the velocity of the fields smoothed with a
            // Gaussian of two cells instead
            auto smoothed = [nmesh](const Grid & grid) {
                const double R = 2.0 / nmesh;
                auto out = grid;
                out.fftw_r2c();
                FML::GRID::multiply_fourier_by_function_of_k<3>(
                    out, [R](double k) { return std::exp(-0.5 * k * k * R * R); });
                out.fftw_c2r();
                return out;
            };
            const auto mass_smoothed = smoothed(mass);
            double mean_mass = 0.0;
            for (auto && index : mass.get_real_range())
                mean_mass += double(mass.get_real_from_index(index));
            FML::SumOverTasks(&mean_mass);
            mean_mass /= std::pow(double(nmesh), 3);
            const double min_mass = 1e-3 * mean_mass;

            // theta(k) = -sum_d i k_d v_d(k) (the layout of the momentum grids: the same indices)
            auto theta = FML::GRID::make_grid<Grid>(nmesh, nextra.first, nextra.second);
            theta.set_grid_status_real(false);
            for (auto && index : theta.get_fourier_range())
                theta.set_fourier_from_index(index, 0.0);
            for (int d = 0; d < 3; d++) {
                for (size_t i = 0; i < npart; i++)
                    points[i].weight = double(FML::PARTICLE::GetMass(part[i])) *
                                       double(FML::PARTICLE::GetVel(part[i])[d]) * velocity_to_displacement;
                auto momentum = detail::deposit_weights<Grid>(points, ntot, nmesh, method, number);
                const auto momentum_smoothed = smoothed(momentum);
                for (auto && index : momentum.get_real_range()) {
                    const double m = double(mass.get_real_from_index(index));
                    const double ms = double(mass_smoothed.get_real_from_index(index));
                    double v = 0.0;
                    if (m > min_mass)
                        v = double(momentum.get_real_from_index(index)) / m;
                    else if (ms > min_mass)
                        v = double(momentum_smoothed.get_real_from_index(index)) / ms;
                    momentum.set_real_from_index(index, FML::GRID::FloatType(v));
                }
                momentum.fftw_r2c();
                for (auto && index : theta.get_fourier_range()) {
                    std::array<double, 3> kvec;
                    double k2;
                    theta.get_fourier_wavevector_and_norm2_by_index(index, kvec, k2);
                    const auto value = theta.get_fourier_from_index(index) -
                                       FML::GRID::ComplexType(0.0, kvec[d]) * momentum.get_fourier_from_index(index);
                    theta.set_fourier_from_index(index, value);
                }
            }
            theta.fftw_c2r();
            return theta;
        }

    } // namespace ANALYSIS
} // namespace FML
#endif
