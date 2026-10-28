#ifndef EXTERNALIC_HEADER
#define EXTERNALIC_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/GadgetUtils/GadgetUtilsHDF5.h>
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

    /// What the header of the files we read must agree with (the simulation we run)
    struct ExpectedGadgetHeader {
        double scale_factor;
        double boxsize;  // Mpc/h
        double OmegaM;   // Total matter (what we write); OmegaCDM + Omegab is also accepted
        double Omega_cb; // OmegaCDM + Omegab
        double h;
    };

    inline ExpectedGadgetHeader expected_gadget_header(const SimulationParameters & params, const Cosmology & cosmo) {
        return {params.ic.a_initial(),
                params.simulation.boxsize,
                cosmo.get_OmegaM(),
                cosmo.get_OmegaCDM() + cosmo.get_Omegab(),
                cosmo.get_h()};
    }

    /// Check that the header of the files agrees with the simulation. The reader scales the positions by the BoxSize
    /// of the file (so a file in kpc/h, BoxSize = 1000 x simulation_boxsize, is fine) but nothing else would tell us
    /// that the box or the cosmology is not the one we simulate. Omega0 and HubbleParam are only checked when set
    inline void check_gadget_header(const GadgetHeader & header,
                                    const ExpectedGadgetHeader & expected,
                                    const std::string & fileprefix,
                                    const std::string & caller) {
        auto close = [](double x, double y, double rtol) { return std::fabs(x - y) <= rtol * std::fabs(y); };
        auto fail = [&](const std::string & what) {
            const std::string error =
                "[" + caller + "] The header of " + fileprefix + " does not match the simulation: " + what;
            FML::assert_mpi(false, error.c_str());
        };
        // Relative tolerance on a (the files store a in single or double precision): 1e-4 is dz = 2e-3 at z = 20
        if (not close(header.time, expected.scale_factor, 1e-4))
            fail("a = " + std::to_string(header.time) +
                 " in the file, the initial redshift ic_initial_redshift gives a = " +
                 std::to_string(expected.scale_factor));
        if (not(close(header.BoxSize, expected.boxsize, 1e-5) or
                close(header.BoxSize, 1000.0 * expected.boxsize, 1e-5)))
            fail("BoxSize = " + std::to_string(header.BoxSize) + " in the file, simulation_boxsize = " +
                 std::to_string(expected.boxsize) + " Mpc/h (or x1000 for kpc/h)");
        if (header.Omega0 > 0.0 and
            not(close(header.Omega0, expected.OmegaM, 1e-3) or close(header.Omega0, expected.Omega_cb, 1e-3)))
            fail("Omega0 = " + std::to_string(header.Omega0) +
                 " in the file, the cosmology has OmegaM = " + std::to_string(expected.OmegaM));
        if (header.HubbleParam > 0.0 and not close(header.HubbleParam, expected.h, 1e-3))
            fail("HubbleParam = " + std::to_string(header.HubbleParam) +
                 " in the file, cosmology_h = " + std::to_string(expected.h));
    }

    /// Read particles from gadget files (all tasks read the same files and only keep the particles
    /// in their domain) and check that they are at the expected scale factor, box and cosmology. The files can be
    /// binary (fileprefix.0, fileprefix.1, ...) or HDF5 (fileprefix.hdf5 or fileprefix.0.hdf5, ...)
    /// Returns the particles (as FML::Vector so they can be moved into MPIParticles) and the header
    template <class T>
    std::pair<FML::Vector<T>, GadgetHeader> read_gadget_particles(const std::string & fileprefix,
                                                                  double allocation_factor,
                                                                  const ExpectedGadgetHeader & expected,
                                                                  const std::string & caller) {
        FML::Vector<T> particles;
        const bool only_keep_part_in_domain = true;
        const bool verbose = false;
        GadgetHeader header;
        if (FML::FILEUTILS::GADGET::is_hdf5_snapshot(fileprefix)) {
#ifdef USE_HDF5
            if (FML::ThisTask == 0)
                std::cout << "# Reading GADGET HDF5 files\n";
            FML::FILEUTILS::GADGET::GadgetHDF5Reader g;
            g.read_gadget(fileprefix, particles, allocation_factor, only_keep_part_in_domain, verbose);
            header = g.get_header();
#else
            throw std::runtime_error("[" + caller + "] " + fileprefix +
                                     " is a GADGET HDF5 snapshot. Compile with USE_HDF5=true to read it");
#endif
        } else {
            FML::FILEUTILS::GADGET::GadgetReader g;
            g.read_gadget(fileprefix, particles, allocation_factor, only_keep_part_in_domain, verbose);
            header = g.get_header();
        }

        if (FML::ThisTask == 0)
            FML::FILEUTILS::GADGET::print_header_info(header);

        check_gadget_header(header, expected, fileprefix, caller);

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
    /// (Eulerian) positions of the particles. The interpolation with the assignment kernel multiplies each mode by
    /// the window W(k) again (the density was already deconvolved once), so we deconvolve it here too (without this
    /// the displacements were W Psi: P(k, z = 0) -16% at k = 0.25 h/Mpc with PCS, 32^3)
    template <int NDIM, class T>
    std::array<std::vector<FML::GRID::FloatType>, NDIM>
    displacement_field_at_particles(const FML::GRID::FFTWGrid<NDIM> & phi_fourier,
                                    FML::PARTICLE::MPIParticles<T> & part,
                                    const std::string & assignment_method) {
        auto phi_deconvolved = phi_fourier;
        FML::INTERPOLATION::deconvolve_window_function_fourier<NDIM>(phi_deconvolved, assignment_method);
        std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> psi_vector;
        FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<NDIM>(phi_deconvolved, psi_vector);
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

    /// The offset s (in units of the particle spacing, in [-1/2, 1/2)) of a regular lattice q = (i + s) / N from the
    /// approximate Lagrangian positions (q_vec = q + small 2LPT term). We take the circular mean of the
    /// fractional part of q_vec N over all particles: lattices are often offset (COLASolver offsets its lattice by
    /// half a force cell when force_nmesh > Npart_1D, other codes put particles at cell centres)
    template <int NDIM, class T>
    std::array<double, NDIM>
    estimate_lattice_offset(FML::PARTICLE::MPIParticles<T> & part, double disp_factor, int Npart_1D) {
        std::array<double, 2 * NDIM> sum{};
        auto * part_ptr = part.get_particles_ptr();
        for (size_t ind = 0; ind < part.get_npart(); ind++) {
            auto * pos = FML::PARTICLE::GetPos(part_ptr[ind]);
            auto * v = FML::PARTICLE::GetVel(part_ptr[ind]);
            for (int idim = 0; idim < NDIM; idim++) {
                const double theta = 2.0 * M_PI * Npart_1D * (pos[idim] - v[idim] * disp_factor);
                sum[2 * idim] += std::cos(theta);
                sum[2 * idim + 1] += std::sin(theta);
            }
        }
        FML::SumArrayOverTasks(sum.data(), 2 * NDIM);
        std::array<double, NDIM> offset;
        for (int idim = 0; idim < NDIM; idim++) {
            offset[idim] = std::atan2(sum[2 * idim + 1], sum[2 * idim]) / (2.0 * M_PI);
            // Below 1e-4 of a particle spacing this is the round-off of the positions in the file (and of the sum):
            // the lattice is at q = i / N
            if (std::fabs(offset[idim]) < 1e-4)
                offset[idim] = 0.0;
        }
        return offset;
    }

    //=============================================================================
    /// Reconstruct the LPT displacements exactly from the positions and velocities of
    /// the particles. This always works for 1LPT initial conditions. For 2LPT
    /// (regular_q_grid_and_2LPT = true) it assumes the particles were displaced from
    /// a regular grid (i.e. it does not work for a glass), possibly offset (we estimate the offset)
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
        // Assuming the lattice is at q = i / N gave wrong D1, D2 for an offset lattice (P(k) 0.9% off with
        // COLASolver's own fiducial offset and force_nmesh = 2 Npart_1D, a factor 2-8 with half a particle spacing)
        std::array<double, NDIM> lattice_offset{};
        if (regular_q_grid_and_2LPT) {
            lattice_offset = estimate_lattice_offset<NDIM, T>(part, disp_factor, Npart_1D);
            if (FML::ThisTask == 0) {
                std::cout << "# Lattice offset of the particles (in units of the particle spacing):";
                for (int idim = 0; idim < NDIM; idim++)
                    std::cout << " " << lattice_offset[idim];
                std::cout << "\n";
            }
        }
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
                    q_exact[idim] = (std::round(q_vec[idim] * Npart_1D - lattice_offset[idim]) + lattice_offset[idim]) /
                                    double(Npart_1D);
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
                                     const SimulationParameters & params,
                                     const Cosmology & cosmo) {
        const auto & ic = params.ic;
        if (FML::ThisTask == 0) {
            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "# Reading phases from GADGET files [" + ic.reconstruct_gadgetfilepath + "]\n";
            std::cout << "#=====================================================\n";
        }

        auto gadget = read_gadget_particles<T>(ic.reconstruct_gadgetfilepath,
                                               params.particle.allocation_factor,
                                               expected_gadget_header(params, cosmo),
                                               "read_phases");

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
                                               expected_gadget_header(params, *cosmo),
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
            // We get Psi from the velocities by dividing by the growth rate f(a). With scale-dependent growth (massive
            // neutrinos, modified gravity) the velocity is f(k) Psi and the displacements in the COLA frame would be
            // wrong (without COLA they are not used)
            FML::assert_mpi(not(sim.use_cola and grav_ic->is_growth_scaledependent()),
                            "ic_reconstruct_exact = true with COLA needs scale-independent growth (the velocities are "
                            "f(k) Psi). Use ic_reconstruct_exact = false");
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
