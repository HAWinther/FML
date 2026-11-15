#ifndef OUTPUTIMAGES_HEADER
#define OUTPUTIMAGES_HEADER

//=============================================================================
// images = true: PNG images of fields at every output, image_<field>_z<z>.png in the snapshot folder. Each is a slab
// of the field on an images_nmesh^3 grid projected along images_axis (the mean over the slab), coloured with
// FML/Visualization (settings and the particle fields: FML/Analysis/Images.h). The fields (images_fields):
//
//   density              1 + delta_cb of the CDM+baryon particles (log colours)
//   velocity_divergence  -div v / (a H) of the CDM+baryon particles: f delta in linear theory
//   baryons_minus_cdm    delta_b - delta_c (particle_baryons)
//   neutrinos            delta_nu: of the neutrino particles (particle_neutrinos), or linear theory on top of delta_cb
//                        (force_linear_massive_neutrinos: delta_cb(k) T_nu(k) / T_cb(k))
//   fifth_force          |F_5th| / |F_Newton|: the slab means of the two force magnitudes divided, with the fifth force
//                        of the gravity model (screened models: the method of their force) from delta_cb
//=============================================================================

#include "SimulationGrid.h"
#include "Species.h"
#include <FML/Analysis/Images.h>
#include <FML/NBody/NBody.h>
#include <FML/Smoothing/SmoothingFourier.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

template <int NDIM, class T>
class NBodySimulation;

template <int NDIM, class T>
void output_images(NBodySimulation<NDIM, T> & sim, double redshift, const std::string & snapshot_folder) {
    if constexpr (NDIM != 3) {
        FML::assert_mpi(false, "images are only implemented for NDIM = 3");
    } else {
        using Grid = SimulationGrid<3>;
        using FML::ANALYSIS::particle_density_contrast;
        const auto & settings = sim.get_parameters().images.image;
        const double a = 1.0 / (1.0 + redshift);
        const int nmesh = settings.nmesh;
        const std::string & method = settings.density_assignment_method;
        const double boxsize = sim.get_parameters().simulation.boxsize;

        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Images\n";
            std::cout << "# images_fields                    : ";
            for (auto & field : settings.fields)
                std::cout << field << " ";
            std::cout << "\n";
            std::cout << "# images_nmesh                     : " << nmesh << "\n";
            std::cout << "# Slab along axis " << settings.axis << " centred at " << settings.slab_center
                      << " of thickness " << settings.slab_thickness << " (units of the box)\n";
            std::cout << "#=====================================================\n";
        }

        auto write = [&](const std::string & field, const Grid & real_grid, double add = 0.0) {
            auto image = FML::ANALYSIS::project_image(real_grid, settings);
            for (auto & v : image.data)
                v += add;
            FML::ANALYSIS::write_field_image(snapshot_folder, field, redshift, image, settings);
        };
        // sum_i w_i g_i over the CDM+baryon species of a field g_i of the particles of each (real space)
        auto cb_sum = [&](auto && field_of) {
            Grid sum;
            bool first = true;
            sim.for_each_cb_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
                Grid g = field_of(particles);
                const double w = sim.cb_weight(info, a);
                if (first) {
                    for (auto && index : g.get_real_range())
                        g.set_real_from_index(index, FML::GRID::FloatType(w * g.get_real_from_index(index)));
                    sum = std::move(g);
                    first = false;
                } else {
                    for (auto && index : sum.get_real_range())
                        sum.set_real_from_index(index,
                                                sum.get_real_from_index(index) +
                                                    FML::GRID::FloatType(w * g.get_real_from_index(index)));
                }
            });
            return sum;
        };
        auto delta_of = [&](auto & particles) {
            return particle_density_contrast<Grid>(particles.get_particles_ptr(), particles.get_npart(), nmesh, method);
        };

        // delta_cb (real space) for the density, the linear neutrinos and the fifth force
        const bool need_delta_cb =
            settings.wants("density") or settings.wants("fifth_force") or
            (settings.wants("neutrinos") and not sim.has_neutrino_particles());
        Grid delta_cb;
        if (need_delta_cb)
            delta_cb = cb_sum(delta_of);

        if (settings.wants("density"))
            write("density", delta_cb, 1.0);

        if (settings.wants("velocity_divergence")) {
            // The code velocity a^2 dx/dt / H0 (x in box units) over a^2 H / H0 is the redshift-space displacement
            const double velocity_to_displacement = 1.0 / (a * a * sim.get_cosmology()->HoverH0_of_a(a));
            auto theta = cb_sum([&](auto & particles) {
                return FML::ANALYSIS::particle_velocity_divergence<Grid>(particles.get_particles_ptr(),
                                                                         particles.get_npart(),
                                                                         nmesh,
                                                                         method,
                                                                         velocity_to_displacement);
            });
            write("velocity_divergence", theta);
        }

        if (settings.wants("baryons_minus_cdm")) {
            Grid diff;
            bool have_cdm = false, have_baryons = false;
            sim.for_each_cb_species([&](auto & particles, const SIMCODE::SpeciesInfo & info) {
                const double sign = info.is(SIMCODE::ParticleSpecies::Baryon) ? 1.0 : -1.0;
                (sign > 0 ? have_baryons : have_cdm) = true;
                Grid delta = delta_of(particles);
                if (diff.get_nmesh() == 0) {
                    for (auto && index : delta.get_real_range())
                        delta.set_real_from_index(index, FML::GRID::FloatType(sign * delta.get_real_from_index(index)));
                    diff = std::move(delta);
                } else {
                    for (auto && index : diff.get_real_range())
                        diff.set_real_from_index(index,
                                                 diff.get_real_from_index(index) +
                                                     FML::GRID::FloatType(sign * delta.get_real_from_index(index)));
                }
            });
            FML::assert_mpi(have_cdm and have_baryons, "images: baryons_minus_cdm needs CDM and baryon particles");
            write("baryons_minus_cdm", diff);
        }

        if (settings.wants("neutrinos")) {
            Grid delta_nu;
            if (sim.has_neutrino_particles()) {
                // As the neutrino P(k): with force_neutrino_energy_density the energy-density contrast
                typename std::remove_reference_t<decltype(sim.get_neutrinos())>::ScopedEnergyWeights energy_weights(
                    sim.get_neutrinos(),
                    a,
                    sim.neutrino_energy_density());
                delta_nu = delta_of(sim.get_neutrino_particles());
            } else {
                // Linear theory on top of delta_cb: delta_nu(k) = delta_cb(k) T_nu(k, a) / T_cb(k, a), the ratio
                // tabulated in log k (k in h/Mpc)
                const double kmin = 2.0 * M_PI / boxsize, kmax = 1.01 * std::sqrt(3.0) * M_PI * nmesh / boxsize;
                const int ntab = 256;
                std::vector<double> ratio(ntab);
                for (int i = 0; i < ntab; i++) {
                    const double k = kmin * std::pow(kmax / kmin, i / double(ntab - 1));
                    ratio[i] = std::sqrt(sim.species_linear_ratio(SIMCODE::TransferSpecies::MassiveNeutrino, k, a));
                }
                auto ratio_of_k = [&](double kBox) {
                    if (not(kBox > 0.0))
                        return 0.0;
                    const double x = std::clamp(std::log(kBox / boxsize / kmin) / std::log(kmax / kmin), 0.0, 1.0) *
                                     (ntab - 1);
                    const int i = std::min(int(x), ntab - 2);
                    return ratio[i] + (x - i) * (ratio[i + 1] - ratio[i]);
                };
                delta_nu = delta_cb;
                delta_nu.fftw_r2c();
                FML::GRID::multiply_fourier_by_function_of_k<3>(delta_nu, ratio_of_k);
                delta_nu.fftw_c2r();
            }
            write("neutrinos", delta_nu);
        }

        if (settings.wants("fifth_force")) {
            const auto & grav = sim.get_gravity_model();
            const double H0Box = grav->H0_hmpc * boxsize;
            Grid delta = delta_cb;
            delta.fftw_r2c();
            Grid delta_ff;
            const double newton = grav->fifth_force_density_for_output(a, H0Box, delta, delta_ff);
            // |F_N|^2 and |F_5|^2 summed over the components: F_d(k) = i k_d / k^2 source(k) (the common factor
            // 1.5 OmegaM a cancels in the ratio)
            auto magnitude = [&](const Grid & source, double factor) {
                // (the same layout as the source: the indices of the cells are the same)
                auto out = FML::GRID::make_grid<Grid>(
                    source.get_nmesh(), source.get_n_extra_slices_left(), source.get_n_extra_slices_right());
                out.fill_real_grid(0.0);
                for (int d = 0; d < 3; d++) {
                    auto component = source;
                    for (auto && index : component.get_fourier_range()) {
                        std::array<double, 3> kvec;
                        double k2;
                        component.get_fourier_wavevector_and_norm2_by_index(index, kvec, k2);
                        const auto value = k2 > 0.0 ? component.get_fourier_from_index(index) *
                                                          FML::GRID::ComplexType(0.0, factor * kvec[d] / k2) :
                                                      FML::GRID::ComplexType(0.0);
                        component.set_fourier_from_index(index, value);
                    }
                    component.fftw_c2r();
                    for (auto && index : out.get_real_range()) {
                        const double f = component.get_real_from_index(index);
                        out.set_real_from_index(index, out.get_real_from_index(index) + FML::GRID::FloatType(f * f));
                    }
                }
                for (auto && index : out.get_real_range())
                    out.set_real_from_index(index, std::sqrt(out.get_real_from_index(index)));
                return out;
            };
            const auto image_newton = FML::ANALYSIS::project_image(magnitude(delta, newton), settings);
            auto image = FML::ANALYSIS::project_image(magnitude(delta_ff, 1.0), settings);
            for (size_t i = 0; i < image.data.size(); i++)
                image.data[i] = image_newton.data[i] > 0.0 ? image.data[i] / image_newton.data[i] : 0.0;
            FML::ANALYSIS::write_field_image(snapshot_folder, "fifth_force", redshift, image, settings);
        }
    }
}

#endif
