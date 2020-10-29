#ifndef COLA_HEADER
#define COLA_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>

#include "GravityModel.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using Spline = FML::INTERPOLATION::SPLINE::Spline;

//========================================================================
// This header contains all the methods for doing COLA, i.e.
// do the Kick and Drift step and changing the velocities from the COLA
// frame to the standard frame
//========================================================================

template <int NDIM, class T>
void cola_add_on_LPT_velocity(FML::PARTICLE::MPIParticles<T> & part,
                              std::shared_ptr<GravityModel<NDIM>> & grav,
                              double aini,
                              double a,
                              double sign = 1.0);

template <int NDIM, class T>
void cola_add_on_LPT_velocity_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                             std::shared_ptr<GravityModel<NDIM>> & grav,
                                             FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_3LPTa_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_3LPTb_ini_fourier,
                                             double H0Box,
                                             double aini,
                                             double a,
                                             double sign = 1.0);

template <int NDIM, class T>
void cola_kick_drift_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                    std::shared_ptr<GravityModel<NDIM>> & grav,
                                    FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_3LPTa_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_3LPTb_ini_fourier,
                                    double H0Box,
                                    double aini,
                                    double aold,
                                    double a,
                                    double delta_time_kick,
                                    [[maybe_unused]] double delta_time_drift);

template <int NDIM, class T>
void cola_kick_drift(FML::PARTICLE::MPIParticles<T> & part,
                     std::shared_ptr<GravityModel<NDIM>> & grav,
                     double aini,
                     double aold,
                     double a,
                     double delta_time_kick,
                     [[maybe_unused]] double delta_time_drift);

//========================================================================
// Add on LPT velocity
// In the COLA frame the initial velocity is zero, i.e. we have subtracted the
// velocity predicted by LPT. Here we add on the LPT velocity to the particles
//========================================================================
template <int NDIM, class T>
void cola_add_on_LPT_velocity(FML::PARTICLE::MPIParticles<T> & part,
                              std::shared_ptr<GravityModel<NDIM>> & grav,
                              double aini,
                              double a,
                              double sign) {

    if (FML::ThisTask == 0) {
        std::cout << "Adding on the LPT velocity to particles (COLA)\n";
    }

    auto cosmo = grav->cosmo;

    // 1LPT
    const double D1 = grav->get_D_1LPT(a);
    const double D1ini = grav->get_D_1LPT(aini);
    const double f1 = grav->get_f_1LPT(a) / D1;
    const double vfac_1LPT = sign * D1 / D1ini * f1 * a * a * cosmo->HoverH0_of_a(a);

    // 2LPT
    [[maybe_unused]] const double D2 = grav->get_D_2LPT(a);
    [[maybe_unused]] const double D2ini = grav->get_D_2LPT(aini);
    [[maybe_unused]] const double f2 = grav->get_f_2LPT(a) / D2;
    [[maybe_unused]] const double vfac_2LPT = sign * D2 / D2ini * f2 * a * a * cosmo->HoverH0_of_a(a);

    // 3LPTa
    [[maybe_unused]] const double D3a = grav->get_D_3LPTa(a);
    [[maybe_unused]] const double D3aini = grav->get_D_3LPTa(aini);
    [[maybe_unused]] const double f3a = grav->get_f_3LPTa(a) / D3a;
    [[maybe_unused]] const double vfac_3LPTa = sign * D3a / D3aini * f3a * a * a * cosmo->HoverH0_of_a(a);

    // 3LPTb
    [[maybe_unused]] const double D3b = grav->get_D_3LPTb(a);
    [[maybe_unused]] const double D3bini = grav->get_D_3LPTb(aini);
    [[maybe_unused]] const double f3b = grav->get_f_3LPTb(a) / D3b;
    [[maybe_unused]] const double vfac_3LPTb = sign * D3b / D3bini * f3b * a * a * cosmo->HoverH0_of_a(a);

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < part.get_npart(); i++) {
        auto & p = part[i];
        auto * vel = FML::PARTICLE::GetVel(p);

        if constexpr (FML::PARTICLE::has_get_D_1LPT<T>()) {
            auto * D1 = FML::PARTICLE::GetD_1LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] += D1[idim] * vfac_1LPT;
            }
        }

        if constexpr (FML::PARTICLE::has_get_D_2LPT<T>()) {
            auto * D2 = FML::PARTICLE::GetD_2LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] += D2[idim] * vfac_2LPT;
            }
        }

        if constexpr (FML::PARTICLE::has_get_D_3LPTa<T>()) {
            auto * D3a = FML::PARTICLE::GetD_3LPTa(p);
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] += D3a[idim] * vfac_3LPTa;
            }
        }

        if constexpr (FML::PARTICLE::has_get_D_3LPTb<T>()) {
            auto * D3b = FML::PARTICLE::GetD_3LPTb(p);
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] += D3b[idim] * vfac_3LPTb;
            }
        }
    }
}

// This method moves the particles using the COLA forces
// a is the scale factor for the new position
// aold is the scale factor for the old position
// aini is the scale factor for which displacement fields in particles are stored
template <int NDIM, class T>
void cola_kick_drift(FML::PARTICLE::MPIParticles<T> & part,
                     std::shared_ptr<GravityModel<NDIM>> & grav,
                     double aini,
                     double aold,
                     double a,
                     double delta_time_kick,
                     [[maybe_unused]] double delta_time_drift) {

    constexpr int LPT_order = (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>() and
                               FML::PARTICLE::has_get_D_3LPTa<T>() and FML::PARTICLE::has_get_D_3LPTb<T>()) ?
                                  3 :
                                  (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>() ?
                                       2 :
                                       (FML::PARTICLE::has_get_D_1LPT<T>() ? 1 : 0));

    if (FML::ThisTask == 0) {
        std::cout << "[Kick] + [Drift] COLA " << LPT_order << "LPT\n";
    }

    auto cosmo = grav->cosmo;
    const double norm_poisson = 1.5 * cosmo->get_OmegaM() * aold * grav->GeffOverG(aold);

    const double D1 = grav->get_D_1LPT(a);
    const double D1old = grav->get_D_1LPT(aold);
    const double D1ini = grav->get_D_1LPT(aini);
    const double fac1_pos = (D1 - D1old) / D1ini;
    const double fac1_vel = -norm_poisson * D1old / D1ini * delta_time_kick;

    [[maybe_unused]] const double D2 = grav->get_D_2LPT(a);
    [[maybe_unused]] const double D2old = grav->get_D_2LPT(aold);
    [[maybe_unused]] const double D2ini = grav->get_D_2LPT(aini);
    [[maybe_unused]] const double fac2_pos = (D2 - D2old) / D2ini;
    [[maybe_unused]] const double fac2_vel = -norm_poisson * (D2old - D1old * D1old) / D2ini * delta_time_kick;

    [[maybe_unused]] const double D3a = grav->get_D_3LPTa(a);
    [[maybe_unused]] const double D3aold = grav->get_D_3LPTa(aold);
    [[maybe_unused]] const double D3aini = grav->get_D_3LPTa(aini);
    [[maybe_unused]] const double fac3a_pos = (D3a - D3aold) / D3aini;
    [[maybe_unused]] const double fac3a_vel =
        -norm_poisson * (D3aold - 2.0 * D1old * D1old * D1old) / D3aini * delta_time_kick;

    [[maybe_unused]] const double D3b = grav->get_D_3LPTb(a);
    [[maybe_unused]] const double D3bold = grav->get_D_3LPTb(aold);
    [[maybe_unused]] const double D3bini = grav->get_D_3LPTb(aini);
    [[maybe_unused]] const double fac3b_pos = (D3b - D3bold) / D3bini;
    [[maybe_unused]] const double fac3b_vel =
        -norm_poisson * (D3bold + D1old * D1old * D1old - D1old * D2old) / D3bini * delta_time_kick;

    // Loop over all active particles
    const size_t np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < np; i++) {
        auto & p = part[i];
        auto * pos = FML::PARTICLE::GetPos(p);
        auto * vel = FML::PARTICLE::GetVel(p);

        if constexpr (LPT_order >= 1) {
            auto * D1 = FML::PARTICLE::GetD_1LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += D1[idim] * fac1_pos;
                vel[idim] += D1[idim] * fac1_vel;
            }
        }

        if constexpr (LPT_order >= 2) {
            auto * D2 = FML::PARTICLE::GetD_2LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += D2[idim] * fac2_pos;
                vel[idim] += D2[idim] * fac2_vel;
            }
        }

        if constexpr (LPT_order >= 3) {
            auto * D3a = FML::PARTICLE::GetD_3LPTa(p);
            auto * D3b = FML::PARTICLE::GetD_3LPTb(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += D3a[idim] * fac3a_pos;
                pos[idim] += D3b[idim] * fac3b_pos;
                vel[idim] += D3a[idim] * fac3a_vel;
                vel[idim] += D3b[idim] * fac3b_vel;
            }
        }

        // Periodic wrap
        for (int idim = 0; idim < NDIM; idim++) {
            if (pos[idim] >= 1.0)
                pos[idim] -= 1.0;
            if (pos[idim] < 0.0)
                pos[idim] += 1.0;
        }
    }
}

//====================================================================================
// Scaledependent methods
//====================================================================================

// This method kick's and drift's the particles one time-step
// Only 1LPT, though not hard to extend
// This method can be speed up a lot
template <int NDIM, class T>
void cola_kick_drift_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                    std::shared_ptr<GravityModel<NDIM>> & grav,
                                    FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_3LPTa_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_3LPTb_ini_fourier,
                                    double H0Box,
                                    double aini,
                                    double aold,
                                    double a,
                                    double delta_time_kick,
                                    [[maybe_unused]] double delta_time_drift) {

    constexpr bool print_timings = false;  
    FML::UTILS::Timings timer;
    timer.StartTiming("Scaledependent COLA");

    constexpr int LPT_order = (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>() and
                               FML::PARTICLE::has_get_D_3LPTa<T>() and FML::PARTICLE::has_get_D_3LPTb<T>()) ?
                                  3 :
                                  (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>() ?
                                       2 :
                                       (FML::PARTICLE::has_get_D_1LPT<T>() ? 1 : 0));

    if (FML::ThisTask == 0) {
        std::cout << "[Kick] + [Drift] Scaledependent COLA " << LPT_order << "LPT\n";
    }

    // Check that particles have the methods they need to use this method
    if constexpr (LPT_order == 1) {
        FML::assert_mpi(FML::PARTICLE::has_get_dDdloga_1LPT<T>(),
                        "Error in cola_kick_drift_scaledependent. Particle do not have a get_dDdloga_1LPT method\n");
        FML::assert_mpi(phi_1LPT_ini_fourier.get_nmesh() > 0,
                        "Error in cola_kick_drift_scaledependent. Initial 1LPT potential is not allocated\n");
    }
    if constexpr (LPT_order >= 2) {
        FML::assert_mpi(phi_2LPT_ini_fourier.get_nmesh() > 0,
                        "Error in cola_kick_drift_scaledependent. Initial 2LPT potential is not allocated\n");
    }
    if constexpr (LPT_order >= 3) {
        FML::assert_mpi(phi_3LPTa_ini_fourier.get_nmesh() > 0 and phi_3LPTb_ini_fourier.get_nmesh() > 0,
                        "Error in cola_kick_drift_scaledependent. Initial 2LPT potential is not allocated\n");
    }

    auto cosmo = grav->cosmo;
    const double OmegaM = cosmo->get_OmegaM();
    const std::string interpolation_method = "CIC";

    //======================================================================================
    // Swap positions
    //======================================================================================
    timer.StartTiming("Communication");
    FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
    part.communicate_particles();
    timer.EndTiming("Communication");

    //======================================================================================
    // Take the grid grid(kvec) and multiply it by func(k). FFT to get grid(x) and interpolate this to particle
    // positions and returns this vector for all particles
    //======================================================================================
    auto generate_displacements = [&](const FML::GRID::FFTWGrid<NDIM> & grid_fourier,
                                      std::array<std::vector<FML::GRID::FloatType>, NDIM> & result,
                                      std::function<double(double)> func) {
        timer.StartTiming("LPT potential -> Psi (FFTs)");
        std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> grid_vector_real;
        FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector_scaledependent<NDIM>(
            grid_fourier, grid_vector_real, func);
        for (int idim = 0; idim < NDIM; idim++) {
            grid_vector_real[idim].communicate_boundaries();
        }
        timer.EndTiming("LPT potential -> Psi (FFTs)");

        // Compute at particle positions (this would be faster if we could do direct assignment
        // which we can by using Lagrangian position (we know how this is generated...))
        timer.StartTiming("Interpolation");
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, T>(
            grid_vector_real, part.get_particles_ptr(), part.get_npart(), result, interpolation_method);
        timer.EndTiming("Interpolation");
    };

    // For 1LPT kick step: -1.5 * OmegaM * a * GeffG(k,a) * D1 / D1ini
    auto function_vel_1LPT = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        double factor = -1.5 * OmegaM * aold * grav->GeffOverG(aold, koverH0) * delta_time_kick;
        return factor * grav->source_factor_1LPT(aold, koverH0) * grav->get_D_1LPT(aold, koverH0) /
               grav->get_D_1LPT(aini, koverH0);
    };

    // For 1LPT drift step: (D1 - D1old) / D1ini
    auto function_pos_1LPT = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        return (grav->get_D_1LPT(a, koverH0) - grav->get_D_1LPT(aold, koverH0)) / grav->get_D_1LPT(aini, koverH0);
    };

    // For 2LPT kick step: -1.5 * OmegaM * a * GeffG(k,a) * (D2 - D1 * D1) / D2ini * delta_time_kick
    [[maybe_unused]] auto function_vel_2LPT = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        double factor = -1.5 * OmegaM * aold * grav->GeffOverG(aold, koverH0) * delta_time_kick;
        double D_1LPT = grav->get_D_1LPT(aold, koverH0);
        double D_2LPT = grav->get_D_2LPT(aold, koverH0);
        double D_2LPT_ini = grav->get_D_2LPT(aini, koverH0);
        return factor * grav->source_factor_2LPT(aold, koverH0) * (D_2LPT - D_1LPT * D_1LPT) / D_2LPT_ini;
    };

    // For 2LPT drift step: (D2 - D2old) / D2ini
    [[maybe_unused]] auto function_pos_2LPT = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        return (grav->get_D_2LPT(a, koverH0) - grav->get_D_2LPT(aold, koverH0)) / grav->get_D_2LPT(aini, koverH0);
    };

    // For 3LPT kick step
    [[maybe_unused]] auto function_vel_3LPTa = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        double factor = -1.5 * OmegaM * aold * grav->GeffOverG(aold, koverH0) * delta_time_kick;
        double D_1LPT = grav->get_D_1LPT(aold, koverH0);
        double D_3LPTa = grav->get_D_3LPTa(aold, koverH0);
        double D_3LPTa_ini = grav->get_D_3LPTa(aini, koverH0);
        return factor * grav->source_factor_3LPTa(aold, koverH0) * (D_3LPTa - 2.0 * D_1LPT * D_1LPT * D_1LPT) /
               D_3LPTa_ini;
    };
    [[maybe_unused]] auto function_vel_3LPTb = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        double factor = -1.5 * OmegaM * aold * grav->GeffOverG(aold, koverH0) * delta_time_kick;
        double D_1LPT = grav->get_D_1LPT(aold, koverH0);
        double D_2LPT = grav->get_D_2LPT(aold, koverH0);
        double D_3LPTb = grav->get_D_3LPTb(aold, koverH0);
        double D_3LPTb_ini = grav->get_D_3LPTb(aini, koverH0);
        return factor * grav->source_factor_3LPTb(aold, koverH0) *
               (D_3LPTb + D_1LPT * D_1LPT * D_1LPT - D_1LPT * D_2LPT) / D_3LPTb_ini;
    };

    // For 3LPT drift step
    [[maybe_unused]] auto function_pos_3LPTa = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        return (grav->get_D_3LPTa(a, koverH0) - grav->get_D_3LPTa(aold, koverH0)) / grav->get_D_3LPTa(aini, koverH0);
    };
    [[maybe_unused]] auto function_pos_3LPTb = [&](double kBox) {
        double koverH0 = kBox / H0Box;
        return (grav->get_D_3LPTb(a, koverH0) - grav->get_D_3LPTb(aold, koverH0)) / grav->get_D_3LPTb(aini, koverH0);
    };

    //======================================================================================
    // std::function is slow, make splines
    //======================================================================================
    const int Nmesh = phi_1LPT_ini_fourier.get_nmesh();
    const int npts = 4 * Nmesh;
    const double kmin = M_PI;
    const double kmax = 2.0 * M_PI * Nmesh / 2.0 * std::sqrt(double(NDIM));
    std::vector<double> k_vec(npts);
    std::vector<double> vel1(npts), pos1(npts);
    std::vector<double> vel2(npts), pos2(npts);
    std::vector<double> vel3a(npts), pos3a(npts);
    std::vector<double> vel3b(npts), pos3b(npts);

    for (int i = 0; i < npts; i++) {
        k_vec[i] = kmin + (kmax - kmin) * i / double(npts - 1);
        if constexpr (LPT_order >= 1) {
            pos1[i] = function_pos_1LPT(k_vec[i]);
            vel1[i] = function_vel_1LPT(k_vec[i]);
        }
        if constexpr (LPT_order >= 2) {
            pos2[i] = function_pos_2LPT(k_vec[i]);
            vel2[i] = function_vel_2LPT(k_vec[i]);
        }
        if constexpr (LPT_order >= 3) {
            pos3a[i] = function_pos_3LPTa(k_vec[i]);
            vel3a[i] = function_vel_3LPTa(k_vec[i]);
            pos3b[i] = function_pos_3LPTb(k_vec[i]);
            vel3b[i] = function_vel_3LPTb(k_vec[i]);
        }
    }

    Spline function_pos_1LPT_spline;
    Spline function_vel_1LPT_spline;
    if constexpr (LPT_order >= 1) {
        function_pos_1LPT_spline.create(k_vec, pos1);
        function_vel_1LPT_spline.create(k_vec, vel1);
    }

    Spline function_pos_2LPT_spline;
    Spline function_vel_2LPT_spline;
    if constexpr (LPT_order >= 2) {
        function_pos_2LPT_spline.create(k_vec, pos2);
        function_vel_2LPT_spline.create(k_vec, vel2);
    }

    Spline function_pos_3LPTa_spline;
    Spline function_pos_3LPTb_spline;
    Spline function_vel_3LPTa_spline;
    Spline function_vel_3LPTb_spline;
    if constexpr (LPT_order >= 3) {
        function_pos_3LPTa_spline.create(k_vec, pos3a);
        function_vel_3LPTa_spline.create(k_vec, vel3a);
        function_pos_3LPTb_spline.create(k_vec, pos3b);
        function_vel_3LPTb_spline.create(k_vec, vel3b);
    }

    //======================================================================================
    // Compute the full LPT force kick and velocity drift
    // We use D_1LPT and dD_1LPT_dloga as temporary storage for the kicks in 1LPT
    // We use D_1LPT and D_2LPT as temporary storage otherwise
    //======================================================================================

    auto temp_grid = phi_1LPT_ini_fourier;
    auto Local_nx = temp_grid.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        double kmag;
        std::array<double, NDIM> kvec;
        for (auto && fourier_index : temp_grid.get_fourier_range(islice, islice + 1)) {
            temp_grid.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
            auto delta_ini = phi_1LPT_ini_fourier.get_fourier_from_index(fourier_index);
            auto value = delta_ini * FML::GRID::FloatType(function_pos_1LPT_spline(kmag));
            if constexpr (LPT_order >= 2) {
                auto phi_2LPT = phi_2LPT_ini_fourier.get_fourier_from_index(fourier_index);
                value += phi_2LPT * FML::GRID::FloatType(function_pos_2LPT_spline(kmag));
            }
            if constexpr (LPT_order >= 3) {
                auto phi_3LPTa = phi_3LPTa_ini_fourier.get_fourier_from_index(fourier_index);
                auto phi_3LPTb = phi_3LPTb_ini_fourier.get_fourier_from_index(fourier_index);
                value += phi_3LPTa * FML::GRID::FloatType(function_pos_3LPTa_spline(kmag));
                value += phi_3LPTb * FML::GRID::FloatType(function_pos_3LPTb_spline(kmag));
            }
            temp_grid.set_fourier_from_index(fourier_index, value);
        }
    }

    std::array<std::vector<FML::GRID::FloatType>, NDIM> displacements;
    auto multiply_by_one = []([[maybe_unused]] double kBox) { return 1.0; };
    generate_displacements(temp_grid, displacements, multiply_by_one);

    // We store this in D_1LPT
    auto np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < np; ind++) {
        auto * D_pos = FML::PARTICLE::GetD_1LPT(part[ind]);
        for (int idim = 0; idim < NDIM; idim++)
            D_pos[idim] = displacements[idim][ind];
    }

#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        double kmag;
        std::array<double, NDIM> kvec;
        for (auto && fourier_index : temp_grid.get_fourier_range(islice, islice + 1)) {
            temp_grid.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
            auto delta_ini = phi_1LPT_ini_fourier.get_fourier_from_index(fourier_index);
            auto value = delta_ini * FML::GRID::FloatType(function_vel_1LPT_spline(kmag));
            if constexpr (LPT_order >= 2) {
                auto phi_2LPT = phi_2LPT_ini_fourier.get_fourier_from_index(fourier_index);
                value += phi_2LPT * FML::GRID::FloatType(function_vel_2LPT_spline(kmag));
            }
            if constexpr (LPT_order >= 3) {
                auto phi_3LPTa = phi_3LPTa_ini_fourier.get_fourier_from_index(fourier_index);
                auto phi_3LPTb = phi_3LPTb_ini_fourier.get_fourier_from_index(fourier_index);
                value += phi_3LPTa * FML::GRID::FloatType(function_vel_3LPTa_spline(kmag));
                value += phi_3LPTb * FML::GRID::FloatType(function_vel_3LPTb_spline(kmag));
            }
            temp_grid.set_fourier_from_index(fourier_index, value);
        }
    }
    generate_displacements(temp_grid, displacements, multiply_by_one);
    temp_grid.free();

    // If we have only 1LPT then we need dD_1LPT_dloga as temp storage
    // If we have 2LPT then we use D_2LPT as temp storage
    np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < np; ind++) {
        if constexpr (LPT_order == 1) {
            auto * D_vel = FML::PARTICLE::GetdDdloga_1LPT(part[ind]);
            for (int idim = 0; idim < NDIM; idim++)
                D_vel[idim] = displacements[idim][ind];
        }

        if constexpr (LPT_order >= 2) {
            auto * D_vel = FML::PARTICLE::GetD_2LPT(part[ind]);
            for (int idim = 0; idim < NDIM; idim++)
                D_vel[idim] = displacements[idim][ind];
        }
    }

    // Swap positions back
    timer.StartTiming("Communication");
    FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
    part.communicate_particles();
    timer.EndTiming("Communication");

    //================================================================
    // Do the cola_kick_drift step
    //================================================================
    np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < np; i++) {
        auto & p = part[i];
        auto * pos = FML::PARTICLE::GetPos(p);
        auto * vel = FML::PARTICLE::GetVel(p);

        if constexpr (LPT_order == 1) {
            // 1LPT: we are using D1 and dDdloga_1LPT
            auto * D_pos = FML::PARTICLE::GetD_1LPT(p);
            auto * D_vel = FML::PARTICLE::GetdDdloga_1LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += D_pos[idim];
                vel[idim] += D_vel[idim];
            }
        }

        if constexpr (LPT_order >= 2) {
            // 1LPT + 2LPT: we are using D1 and D2 as temp storage
            auto * D_pos = FML::PARTICLE::GetD_1LPT(p);
            auto * D_vel = FML::PARTICLE::GetD_2LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] += D_pos[idim];
                vel[idim] += D_vel[idim];
            }
        }

        // Periodic wrap
        for (int idim = 0; idim < NDIM; idim++) {
            if (pos[idim] >= 1.0)
                pos[idim] -= 1.0;
            if (pos[idim] < 0.0)
                pos[idim] += 1.0;
        }
    }

    timer.EndTiming("Scaledependent COLA");
    if (FML::ThisTask == 0 and print_timings)
        timer.PrintAllTimings();
}

template <int NDIM, class T>
void cola_add_on_LPT_velocity_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                             std::shared_ptr<GravityModel<NDIM>> & grav,
                                             FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_3LPTa_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_3LPTb_ini_fourier,
                                             double H0Box,
                                             double aini,
                                             double a,
                                             double sign) {

    constexpr int LPT_order = (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>() and
                               FML::PARTICLE::has_get_D_3LPTa<T>() and FML::PARTICLE::has_get_D_3LPTb<T>()) ?
                                  3 :
                                  (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>() ?
                                       2 :
                                       (FML::PARTICLE::has_get_D_1LPT<T>() ? 1 : 0));

    if (FML::ThisTask == 0) {
        std::cout << "Adding on the LPT velocity to particles (COLA " << LPT_order << "LPT scaledependent)\n";
    }

    if (not phi_1LPT_ini_fourier)
        FML::assert_mpi(false, "phi_1LPT_ini_fourier is not allocated");

    if constexpr (LPT_order == 1) {
        FML::assert_mpi(FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_dDdloga_1LPT<T>(),
                        "Error in cola_add_on_LPT_velocity_scaledependent. Particle do not have both get_D_1LPT "
                        "and get_dDdloga_1LPT methods\n");
    }

    if constexpr (LPT_order >= 2) {
        FML::assert_mpi(FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>(),
                        "Error in cola_add_on_LPT_velocity_scaledependent. Particle do not have both get_D_1LPT "
                        "and get_D_2LPT methods\n");
    }

    auto cosmo = grav->cosmo;
    const double vfactor = sign * a * a * cosmo->HoverH0_of_a(a);
    const std::string interpolation_method = "CIC";

    //======================================================================================
    // Swap positions
    //======================================================================================
    FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
    part.communicate_particles();

    //======================================================================================
    // Take the grid grid(kvec) and multiply it by func(k). FFT to get D grid(x) and interpolate this to
    // particle positions and returns this vector for all particles
    //======================================================================================
    auto generate_displacements = [&](const FML::GRID::FFTWGrid<NDIM> & grid_fourier,
                                      std::array<std::vector<FML::GRID::FloatType>, NDIM> & result,
                                      std::function<double(double)> func) {
        std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> grid_vector_real;
        FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector_scaledependent<NDIM>(
            grid_fourier, grid_vector_real, func);
        for (int idim = 0; idim < NDIM; idim++) {
            grid_vector_real[idim].communicate_boundaries();
        }
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, T>(
            grid_vector_real, part.get_particles_ptr(), part.get_npart(), result, interpolation_method);
    };

    auto function_vel_1LPT = [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return vfactor * grav->get_f_1LPT(a, koverH0) * grav->get_D_1LPT(a, koverH0) / grav->get_D_1LPT(aini, koverH0);
    };

    [[maybe_unused]] auto function_vel_2LPT = [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return vfactor * grav->get_f_2LPT(a, koverH0) * grav->get_D_2LPT(a, koverH0) / grav->get_D_2LPT(aini, koverH0);
    };

    [[maybe_unused]] auto function_vel_3LPTa = [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return vfactor * grav->get_f_3LPTa(a, koverH0) * grav->get_D_3LPTa(a, koverH0) /
               grav->get_D_3LPTa(aini, koverH0);
    };

    [[maybe_unused]] auto function_vel_3LPTb = [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return vfactor * grav->get_f_3LPTb(a, koverH0) * grav->get_D_3LPTb(a, koverH0) /
               grav->get_D_3LPTb(aini, koverH0);
    };

    //======================================================================================
    // std::function is slow, make splines
    //======================================================================================

    const int Nmesh = phi_1LPT_ini_fourier.get_nmesh();
    const int npts = 4 * Nmesh;
    const double kmin = M_PI;
    const double kmax = 2.0 * M_PI * Nmesh / 2.0 * std::sqrt(double(NDIM));
    std::vector<double> k_vec(npts);
    std::vector<double> vel1(npts), vel2(npts), vel3a(npts), vel3b(npts);

    for (int i = 0; i < npts; i++) {
        k_vec[i] = kmin + (kmax - kmin) * i / double(npts - 1);
        vel1[i] = function_vel_1LPT(k_vec[i]);
        if constexpr (LPT_order >= 2) {
            vel2[i] = function_vel_2LPT(k_vec[i]);
        }
        if constexpr (LPT_order >= 3) {
            vel3a[i] = function_vel_3LPTa(k_vec[i]);
            vel3b[i] = function_vel_3LPTb(k_vec[i]);
        }
    }

    Spline function_vel_1LPT_spline;
    function_vel_1LPT_spline.create(k_vec, vel1);

    Spline function_vel_2LPT_spline;
    if constexpr (LPT_order >= 2) {
        function_vel_2LPT_spline.create(k_vec, vel2);
    }
    Spline function_vel_3LPTa_spline;
    Spline function_vel_3LPTb_spline;
    if constexpr (LPT_order >= 3) {
        function_vel_3LPTa_spline.create(k_vec, vel3a);
        function_vel_3LPTb_spline.create(k_vec, vel3b);
    }

    //======================================================================================
    // Compute the total LPT potential
    //======================================================================================
    auto temp_grid = phi_1LPT_ini_fourier;
    auto Local_nx = temp_grid.get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int islice = 0; islice < Local_nx; islice++) {
        double kmag;
        std::array<double, NDIM> kvec;
        for (auto && fourier_index : temp_grid.get_fourier_range(islice, islice + 1)) {
            temp_grid.get_fourier_wavevector_and_norm_by_index(fourier_index, kvec, kmag);
            auto phi_1LPT = phi_1LPT_ini_fourier.get_fourier_from_index(fourier_index);
            auto value = phi_1LPT * FML::GRID::FloatType(function_vel_1LPT_spline(kmag));
            if constexpr (LPT_order >= 2) {
                auto phi_2LPT = phi_2LPT_ini_fourier.get_fourier_from_index(fourier_index);
                value += phi_2LPT * FML::GRID::FloatType(function_vel_2LPT_spline(kmag));
            }
            if constexpr (LPT_order >= 3) {
                auto phi_3LPTa = phi_3LPTa_ini_fourier.get_fourier_from_index(fourier_index);
                auto phi_3LPTb = phi_3LPTb_ini_fourier.get_fourier_from_index(fourier_index);
                value += phi_3LPTa * FML::GRID::FloatType(function_vel_3LPTa_spline(kmag));
                value += phi_3LPTb * FML::GRID::FloatType(function_vel_3LPTb_spline(kmag));
            }
            temp_grid.set_fourier_from_index(fourier_index, value);
        }
    }

    std::array<std::vector<FML::GRID::FloatType>, NDIM> displacements;
    auto multiply_by_one = []([[maybe_unused]] double kBox) { return 1.0; };
    generate_displacements(temp_grid, displacements, multiply_by_one);

    // We store this in D_1LPT
    auto np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < np; ind++) {
        auto * D_vel = FML::PARTICLE::GetD_1LPT(part[ind]);
        for (int idim = 0; idim < NDIM; idim++)
            D_vel[idim] = displacements[idim][ind];
    }

    //======================================================================================
    // Swap back
    //======================================================================================
    FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
    part.communicate_particles();

    //======================================================================================
    // Add on LPT velocity
    //======================================================================================
    np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < np; ind++) {
        auto * vel = FML::PARTICLE::GetVel(part[ind]);
        auto * dD = FML::PARTICLE::GetD_1LPT(part[ind]);
        for (int idim = 0; idim < NDIM; idim++) {
            vel[idim] += dD[idim];
        }
    }
}
#endif
