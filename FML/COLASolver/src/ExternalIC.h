#ifndef EXTERNALIC_HEADER
#define EXTERNALIC_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>

#include "Cosmology.h"
#include "GravityModel.h"
#include "SimulationParameters.h"

#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//=============================================================================
// Helpers for starting a simulation from particles in (gadget) files instead of
// generating the initial conditions: reading the particles and reconstructing
// the LPT displacement fields that COLA needs
//=============================================================================

namespace COLASOLVER {

    using GadgetHeader = FML::FILEUTILS::GADGET::GadgetHeader;

    /// Read particles from gadget files (all tasks read the same files and only keep the particles
    /// in their domain) and check that they are at the expected scale factor.
    /// Returns the particles (as FML::Vector so they can be moved into MPIParticles) and the header
    template <class T>
    std::pair<FML::Vector<T>, GadgetHeader> read_gadget_particles(const std::string & fileprefix,
                                                                  double allocation_factor,
                                                                  double expected_scale_factor,
                                                                  const std::string & caller) {
        FML::FILEUTILS::GADGET::GadgetReader g;
        FML::Vector<T> particles;
        const bool only_keep_part_in_domain = true;
        const bool verbose = false;
        g.read_gadget(fileprefix, particles, allocation_factor, only_keep_part_in_domain, verbose);

        auto header = g.get_header();
        if (FML::ThisTask == 0)
            FML::FILEUTILS::GADGET::print_header_info(header);

        // Check that the redshift in the file matched the initial redshift
        const std::string error = "[" + caller +
                                  "] The redshift in the gadgetfile we read does not match the initial redshift of "
                                  "the simulation";
        FML::assert_mpi(std::fabs(header.time - expected_scale_factor) < 1e-3, error.c_str());

        return {std::move(particles), header};
    }

    /// Wrap a coordinate (in units of the box) back into [0,1)
    inline void wrap_periodic(double & x) {
        if (x < 0.0)
            x += 1.0;
        if (x >= 1.0)
            x -= 1.0;
    }

    /// The displacement field Psi = D phi (from an LPT potential in fourier space) at the
    /// (Eulerian) positions of the particles
    template <int NDIM, class T>
    std::array<std::vector<FML::GRID::FloatType>, NDIM>
    displacement_field_at_particles(const FML::GRID::FFTWGrid<NDIM> & phi_fourier,
                                    FML::PARTICLE::MPIParticles<T> & part,
                                    const std::string & assignment_method) {
        std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> psi_vector;
        FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<NDIM>(phi_fourier, psi_vector);
        for (int idim = 0; idim < NDIM; idim++) {
            psi_vector[idim].communicate_boundaries();
        }
        std::array<std::vector<FML::GRID::FloatType>, NDIM> displacements;
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, T>(psi_vector,
                                                                                   part.get_particles_ptr(),
                                                                                   part.get_npart(),
                                                                                   displacements,
                                                                                   assignment_method);
        return displacements;
    }

    //=============================================================================
    /// Reconstruct the LPT displacements exactly from the positions and velocities of
    /// the particles. This always works for 1LPT initial conditions. For 2LPT
    /// (regular_q_grid_and_2LPT = true) it assumes the particles were displaced from
    /// a regular grid (i.e. it does not work for a glass)
    //=============================================================================
    template <int NDIM, class T>
    void exact_LPT_reconstruction(FML::PARTICLE::MPIParticles<T> & part,
                                  double scale_factor,
                                  double HoverH0,
                                  double f_1LPT,
                                  double f_2LPT,
                                  int cola_LPT_order,
                                  bool regular_q_grid_and_2LPT) {

        // v_code = a^2 E f Psi_code
        const double disp_factor = 1.0 / (scale_factor * scale_factor * HoverH0 * f_1LPT);
        const double f2_over_f1 = f_2LPT / f_1LPT;
        const double v_factor_2LPT = 1.0 / (1.0 - f2_over_f1);
        const int Npart_1D = std::round(std::pow(part.get_npart_total(), 1.0 / NDIM));
        auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t ind = 0; ind < part.get_npart(); ind++) {
            auto * pos = FML::PARTICLE::GetPos(part_ptr[ind]);
            auto * v = FML::PARTICLE::GetVel(part_ptr[ind]);
            std::array<double, NDIM> disp{}, disp2{}, q_vec{};
            for (int idim = 0; idim < NDIM; idim++) {
                disp[idim] = v[idim] * disp_factor;
                q_vec[idim] = pos[idim] - disp[idim];
                if (q_vec[idim] >= 1.0)
                    q_vec[idim] -= 1.0;
                if (q_vec[idim] < 0.0)
                    q_vec[idim] += 1.0;
            }

            // At this point we have:
            // q_vec == q + Psi2LPT(1 - f2LPT/f1LPT)
            // disp == Psi1LPT + (f2LPT/f1LPT) * Psi2LPT

            if (regular_q_grid_and_2LPT) {
                std::array<double, NDIM> q_exact;
                for (int idim = 0; idim < NDIM; idim++) {
                    // Compute exact q
                    q_exact[idim] = std::round(q_vec[idim] * Npart_1D) / double(Npart_1D);
                    // Compute exact D2
                    disp2[idim] = (q_vec[idim] - q_exact[idim]) * v_factor_2LPT;
                    if (disp2[idim] >= 0.5)
                        disp2[idim] -= 1.0;
                    if (disp2[idim] < -0.5)
                        disp2[idim] += 1.0;
                    // Compute exact D1
                    disp[idim] -= f2_over_f1 * disp2[idim];
                    if (disp[idim] >= 0.5)
                        disp[idim] -= 1.0;
                    if (disp[idim] < -0.5)
                        disp[idim] += 1.0;
                    // ...and finally wrap q_exact
                    if (q_exact[idim] >= 1.0)
                        q_exact[idim] -= 1.0;
                    if (q_exact[idim] < 0.0)
                        q_exact[idim] += 1.0;
                }
                q_vec = q_exact;
            }

            if constexpr (FML::PARTICLE::has_get_q<T>()) {
                auto * q = FML::PARTICLE::GetLagrangianPos(part_ptr[ind]);
                for (int idim = 0; idim < NDIM; idim++)
                    q[idim] = q_vec[idim];
            }
            if (cola_LPT_order >= 1)
                if constexpr (FML::PARTICLE::has_get_D_1LPT<T>()) {
                    auto * D = FML::PARTICLE::GetD_1LPT(part_ptr[ind]);
                    for (int idim = 0; idim < NDIM; idim++)
                        D[idim] = disp[idim];
                }
            if (cola_LPT_order >= 2)
                if constexpr (FML::PARTICLE::has_get_D_2LPT<T>()) {
                    auto * D2 = FML::PARTICLE::GetD_2LPT(part_ptr[ind]);
                    for (int idim = 0; idim < NDIM; idim++)
                        D2[idim] = disp2[idim];
                }
        }
    }

    /// Compute delta(k) from particles in gadget files (used to take the phases from them).
    /// The result is smoothed with the reconstruct_* settings
    template <int NDIM, class T>
    void delta_from_gadget_particles(FML::GRID::FFTWGrid<NDIM> & delta_fourier,
                                     const ICSettings & ic,
                                     double allocation_factor) {
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Reading phases from GADGET files [" + ic.reconstruct_gadgetfilepath + "]\n";
            std::cout << "#=====================================================\n";
        }

        auto gadget =
            read_gadget_particles<T>(ic.reconstruct_gadgetfilepath, allocation_factor, ic.a_initial(), "read_phases");

        // Move them into MPIParticles
        FML::PARTICLE::MPIParticles<T> temppart;
        temppart.move_from(std::move(gadget.first));

        FML::INTERPOLATION::particles_to_fourier_grid(temppart.get_particles_ptr(),
                                                      temppart.get_npart(),
                                                      temppart.get_npart_total(),
                                                      delta_fourier,
                                                      ic.reconstruct_assigment_method,
                                                      ic.reconstruct_interlacing);

        // Smoothing (should just use a sharpk filter to set modes beyond used to generate the IC to zero)
        // If we use a larger grid then the highest frequency modes, beyond kyquist of the grid used to
        // generate the IC, will just be noise and will lead to problems so these modes needs to be killed
        FML::GRID::smoothing_filter_fourier_space(delta_fourier,
                                                  ic.reconstruct_dimless_smoothing_scale,
                                                  ic.reconstruct_smoothing_filter);
    }

    //=============================================================================
    /// Read the particles from gadget files into part. With COLA we also reconstruct
    /// the LPT displacement fields (stored in the particles, or for scale-dependent
    /// COLA the LPT potentials phi_1LPT_ini_fourier and phi_2LPT_ini_fourier)
    //=============================================================================
    template <int NDIM, class T>
    void read_ic_from_gadget(FML::PARTICLE::MPIParticles<T> & part,
                             const SimulationParameters & params,
                             std::shared_ptr<Cosmology> cosmo,
                             std::shared_ptr<GravityModel<NDIM>> grav_ic,
                             FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                             FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier) {
        const auto & ic = params.ic;
        const auto & sim = params.simulation;
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Reading initial conditions from GADGET files [" + ic.reconstruct_gadgetfilepath + "]\n";
            std::cout << "#=====================================================\n";
        }

        auto gadget = read_gadget_particles<T>(ic.reconstruct_gadgetfilepath,
                                               params.particle.allocation_factor,
                                               ic.a_initial(),
                                               "read_ic");
        auto & externalpart = gadget.first;
        const double scale_factor = gadget.second.time;

        // Velocities we get from the gadget reader is peculiar km/s so scale to code units
        const double vel_norm = scale_factor / (100 * sim.boxsize);
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < externalpart.size(); i++) {
            auto * vel = FML::PARTICLE::GetVel(externalpart[i]);
            for (int idim = 0; idim < NDIM; idim++)
                vel[idim] *= vel_norm;
        }

        // Move them into MPIParticles
        part.move_from(std::move(externalpart));

        // Reconstruct the LPT fields exactly using the data in the gadget-files.
        // For 1LPT IC this always works. For 2LPT IC this assumes the q-grid the IC
        // was created on is a regular grid (i.e. does not work for a glass)
        // When using this with COLA we assume the COLA LPT order is the same as in the IC
        // If you have 1LPT IC and want 2LPT COLA then the D2LPT fields are put to zero!
        // Only works with scaleindependent COLA
        if (ic.reconstruct_exact) {
            if (FML::ThisTask == 0)
                std::cout << "Doing exact reconstruction of the LPT displacement fields\n";
            FML::assert_mpi(sim.use_scaledependent_cola == false,
                            "Exact 1LPT reconstruction only implemented for scaleindependent cola");
            const bool regular_q_grid_and_2LPT = ic.LPT_order > 1;
            exact_LPT_reconstruction<NDIM, T>(part,
                                              scale_factor,
                                              cosmo->HoverH0_of_a(scale_factor),
                                              grav_ic->get_f_1LPT(scale_factor),
                                              grav_ic->get_f_2LPT(scale_factor),
                                              sim.cola_LPT_order,
                                              regular_q_grid_and_2LPT);
            return;
        }

        // Assign particles to grid
        const auto nleftright =
            FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(ic.reconstruct_assigment_method);
        FML::GRID::FFTWGrid<NDIM> delta_fourier(ic.nmesh, nleftright.first, nleftright.second);
        FML::INTERPOLATION::particles_to_fourier_grid(part.get_particles_ptr(),
                                                      part.get_npart(),
                                                      part.get_npart_total(),
                                                      delta_fourier,
                                                      ic.reconstruct_assigment_method,
                                                      ic.reconstruct_interlacing);

        // Deconvolve window function
        FML::INTERPOLATION::deconvolve_window_function_fourier<NDIM>(delta_fourier, ic.reconstruct_assigment_method);

        // Smoothing (should just use a sharpk filter to set modes beyond Nmesh used to generate the IC to zero)
        // If we use a larger grid then the highest frequency modes, beyond knyuist of the grid used to
        // generate the IC, will just be noise and will lead to problems so these modes needs to be killed
        FML::GRID::smoothing_filter_fourier_space(delta_fourier,
                                                  ic.reconstruct_dimless_smoothing_scale,
                                                  ic.reconstruct_smoothing_filter);

        // In case of COLA we need to assign displacement fields to particles
        if (not sim.use_cola)
            return;

        // Has only implemented this for 1LPT and 2LPT
        FML::assert_mpi(FML::PARTICLE::has_get_D_3LPTa<T>() == false and FML::PARTICLE::has_get_D_3LPTb<T>() == false,
                        "3LPT not implemented for read_ic");

        // 1LPT displacement fields. We also set the (approximate) Lagrangian position q = x - Psi_1LPT
        // For scale-dependent COLA we store the potential
        if (FML::PARTICLE::has_get_D_1LPT<T>() and sim.cola_LPT_order >= 1) {
            if (FML::ThisTask == 0)
                std::cout << "Reconstructing 1LPT displacement field from the density field\n";

            FML::GRID::FFTWGrid<NDIM> temp;
            FML::GRID::FFTWGrid<NDIM> & phi_1LPT_fourier = sim.use_scaledependent_cola ? phi_1LPT_ini_fourier : temp;
            FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta_fourier, phi_1LPT_fourier);

            auto displacements =
                displacement_field_at_particles<NDIM, T>(phi_1LPT_fourier, part, ic.reconstruct_assigment_method);

            auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t ind = 0; ind < part.get_npart(); ind++) {
                auto * pos = FML::PARTICLE::GetPos(part_ptr[ind]);
                auto * D = FML::PARTICLE::GetD_1LPT(part_ptr[ind]);
                for (int idim = 0; idim < NDIM; idim++) {
                    D[idim] = displacements[idim][ind];
                }
                if constexpr (FML::PARTICLE::has_get_q<T>()) {
                    auto * q = FML::PARTICLE::GetLagrangianPos(part_ptr[ind]);
                    for (int idim = 0; idim < NDIM; idim++) {
                        q[idim] = pos[idim] - D[idim];
                        wrap_periodic(q[idim]);
                    }
                }
            }
        }

        // 2LPT displacement fields. Scale-independent COLA needs the displacement field in the particles
        // (NB: we should interpolate it to the Lagrangian positions, but we use the Eulerian position)
        // and we update the Lagrangian position q -= Psi_2LPT. For scale-dependent COLA we store the potential
        if (FML::PARTICLE::has_get_D_2LPT<T>() and sim.cola_LPT_order >= 2) {
            if (FML::ThisTask == 0)
                std::cout << "Reconstructing 2LPT displacement field from the density field\n";

            FML::GRID::FFTWGrid<NDIM> temp;
            FML::GRID::FFTWGrid<NDIM> & phi_2LPT_fourier = sim.use_scaledependent_cola ? phi_2LPT_ini_fourier : temp;
            FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_fourier, phi_2LPT_fourier);

            if (not sim.use_scaledependent_cola) {
                auto displacements =
                    displacement_field_at_particles<NDIM, T>(phi_2LPT_fourier, part, ic.reconstruct_assigment_method);

                auto * part_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t ind = 0; ind < part.get_npart(); ind++) {
                    auto * D = FML::PARTICLE::GetD_2LPT(part_ptr[ind]);
                    for (int idim = 0; idim < NDIM; idim++) {
                        D[idim] = displacements[idim][ind];
                    }
                    if constexpr (FML::PARTICLE::has_get_q<T>()) {
                        auto * q = FML::PARTICLE::GetLagrangianPos(part_ptr[ind]);
                        for (int idim = 0; idim < NDIM; idim++) {
                            q[idim] -= displacements[idim][ind];
                            wrap_periodic(q[idim]);
                        }
                    }
                }
            }
        }
    }

} // namespace COLASOLVER

#endif
