#ifndef COLA_HEADER
#define COLA_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timings.h>

#include "GravityModel.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using Spline = FML::INTERPOLATION::SPLINE::Spline;

//========================================================================
// This header contains all the methods for doing COLA, i.e.
// do the Kick and Drift step and changing the velocities from the COLA
// frame to the standard frame
//========================================================================

/// The highest LPT order whose COLA frame a particle type can store (the displacement fields D_1LPT, D_2LPT)
template <class T>
constexpr int cola_LPT_order_of() {
    if constexpr (FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>())
        return 2;
    else if constexpr (FML::PARTICLE::has_get_D_1LPT<T>())
        return 1;
    else
        return 0;
}

template <int NDIM, class T>
void cola_initialize_velocities(FML::PARTICLE::MPIParticles<T> & part) {
    auto np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < np; i++) {
        auto * vel = FML::PARTICLE::GetVel(part[i]);
        for (int idim = 0; idim < NDIM; idim++) {
            vel[idim] = 0.0;
        }
    }
}

template <int NDIM, class T>
void cola_add_on_LPT_velocity(FML::PARTICLE::MPIParticles<T> & part,
                              std::shared_ptr<GravityModel<NDIM>> & grav,
                              int LPT_order,
                              double aini,
                              double a,
                              double sign = 1.0);

template <int NDIM, class T>
void cola_add_on_LPT_velocity_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                             std::shared_ptr<GravityModel<NDIM>> & grav,
                                             int LPT_order,
                                             FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                             double H0Box,
                                             double aini,
                                             double a,
                                             double sign = 1.0,
                                             double lattice_offset = 0.0);

template <int NDIM, class T>
void cola_kick_drift_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                    std::shared_ptr<GravityModel<NDIM>> & grav,
                                    int LPT_order,
                                    FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                    double H0Box,
                                    double aini,
                                    double aold,
                                    double a,
                                    double delta_time_kick,
                                    [[maybe_unused]] double delta_time_drift,
                                    double lattice_offset = 0.0,
                                    bool do_drift = true);

template <int NDIM, class T>
void cola_kick_drift(FML::PARTICLE::MPIParticles<T> & part,
                     std::shared_ptr<GravityModel<NDIM>> & grav,
                     int LPT_order,
                     double aini,
                     double aold,
                     double a,
                     double delta_time_kick,
                     [[maybe_unused]] double delta_time_drift,
                     bool do_kick = true,
                     bool do_drift = true);

//========================================================================
// Add on LPT velocity
// In the COLA frame the initial velocity is zero, i.e. we have subtracted the
// velocity predicted by LPT. Here we add on the LPT velocity to the particles
//========================================================================
template <int NDIM, class T>
void cola_add_on_LPT_velocity(FML::PARTICLE::MPIParticles<T> & part,
                              std::shared_ptr<GravityModel<NDIM>> & grav,
                              int LPT_order,
                              double aini,
                              double a,
                              double sign) {
    if (LPT_order < 1)
        return;

    if (FML::ThisTask == 0) {
        std::cout << "Adding on the LPT velocity to particles (COLA)\n";
    }

    auto cosmo = grav->get_cosmo();

    // 1LPT
    const double D1 = grav->get_D_1LPT(a);
    const double D1ini = grav->get_D_1LPT(aini);
    const double f1 = grav->get_f_1LPT(a);
    double vfac_1LPT = sign * D1 / D1ini * f1 * a * a * cosmo->HoverH0_of_a(a);
    if (LPT_order < 1)
        vfac_1LPT = 0.0;

    // 2LPT
    [[maybe_unused]] const double D2 = grav->get_D_2LPT(a);
    [[maybe_unused]] const double D2ini = grav->get_D_2LPT(aini);
    [[maybe_unused]] const double f2 = grav->get_f_2LPT(a);
    [[maybe_unused]] double vfac_2LPT = sign * D2 / D2ini * f2 * a * a * cosmo->HoverH0_of_a(a);
    if (LPT_order < 2)
        vfac_2LPT = 0.0;

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
    }
}

// This method moves the particles using the COLA forces
// a is the scale factor for the new position
// aold is the scale factor for the old position
// aini is the scale factor for which displacement fields in particles are stored
// do_kick / do_drift: only update the velocities / positions (the two parts are independent, so a kick followed
// by a drift gives exactly the same result as doing both at once)
template <int NDIM, class T>
void cola_kick_drift(FML::PARTICLE::MPIParticles<T> & part,
                     std::shared_ptr<GravityModel<NDIM>> & grav,
                     int LPT_order,
                     double aini,
                     double aold,
                     double a,
                     double delta_time_kick,
                     [[maybe_unused]] double delta_time_drift,
                     bool do_kick,
                     bool do_drift) {
    if (LPT_order < 1)
        return;

    constexpr int LPT_order_particle = cola_LPT_order_of<T>();

    if (LPT_order_particle < LPT_order) {
        std::string error = "The particle type is compatible with " + std::to_string(LPT_order_particle) + "LPT, ";
        error +=
            "but we are asked to do " + std::to_string(LPT_order) + "LPT COLA. Lower the order or fix the particle\n";
        FML::assert_mpi(false, error.c_str());
    }

    if (FML::ThisTask == 0) {
        std::cout << "[Kick] + [Drift] COLA " << LPT_order << "LPT\n";
    }

    auto cosmo = grav->get_cosmo();
    // The LPT force as in the growth equations: rho_m(a) a^3 of the background (Omega_m_a3)
    const double norm_poisson = 1.5 * cosmo->Omega_m_a3(aold) * aold * grav->GeffOverG(aold);

    // The LPT force of each order as in the growth equations (GravityModel::compute_growth_factors), with the
    // source factors at k = 0 (e.g. 1 - f_nu for >= 2LPT with massive neutrinos and the nonlinear DGP source).
    // Without them the frame kick did not follow the LPT growth factors used for the drift
    const double source_1LPT = grav->source_factor_1LPT(aold);
    const double source_2LPT = grav->source_factor_2LPT(aold);
    const double nonlinear_2LPT = grav->nonlinear_source_factor_2LPT(aold);

    const double D1 = grav->get_D_1LPT(a);
    const double D1old = grav->get_D_1LPT(aold);
    const double D1ini = grav->get_D_1LPT(aini);
    double fac1_pos = (D1 - D1old) / D1ini;
    double fac1_vel = -norm_poisson * source_1LPT * D1old / D1ini * delta_time_kick;
    if (LPT_order < 1)
        fac1_pos = fac1_vel = 0.0;

    [[maybe_unused]] const double D2 = grav->get_D_2LPT(a);
    [[maybe_unused]] const double D2old = grav->get_D_2LPT(aold);
    [[maybe_unused]] const double D2ini = grav->get_D_2LPT(aini);
    [[maybe_unused]] double fac2_pos = (D2 - D2old) / D2ini;
    [[maybe_unused]] double fac2_vel =
        -norm_poisson * source_2LPT * (D2old - nonlinear_2LPT * D1old * D1old) / D2ini * delta_time_kick;
    if (LPT_order < 2)
        fac2_pos = fac2_vel = 0.0;

    // Loop over all active particles
    const size_t np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < np; i++) {
        auto & p = part[i];
        auto * pos = FML::PARTICLE::GetPos(p);
        auto * vel = FML::PARTICLE::GetVel(p);

        if (LPT_order >= 1) {
            auto * D1 = FML::PARTICLE::GetD_1LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                if (do_drift)
                    pos[idim] += D1[idim] * fac1_pos;
                if (do_kick)
                    vel[idim] += D1[idim] * fac1_vel;
            }
        }

        if (LPT_order >= 2) {
            auto * D2 = FML::PARTICLE::GetD_2LPT(p);
            for (int idim = 0; idim < NDIM; idim++) {
                if (do_drift)
                    pos[idim] += D2[idim] * fac2_pos;
                if (do_kick)
                    vel[idim] += D2[idim] * fac2_vel;
            }
        }

        // Periodic wrap
        for (int idim = 0; idim < NDIM; idim++)
            pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(pos[idim]);
    }
}

//====================================================================================
// Scaledependent methods
//====================================================================================

/// grid(k) = phi_1LPT(k) f_1(k) + phi_2LPT(k) f_2(k) up to LPT_order
/// (the grid must have the same layout as the potentials; k in units of 1/boxsize)
template <int NDIM>
void sum_of_LPT_potentials(FML::GRID::FFTWGrid<NDIM> & grid,
                           int LPT_order,
                           const FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                           const FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                           const Spline & f_1LPT,
                           const Spline & f_2LPT) {
    FML::GRID::for_each_fourier_mode<NDIM>(grid, [&](size_t fourier_index, double kmag) {
        auto phi_1LPT = phi_1LPT_ini_fourier.get_fourier_from_index(fourier_index);
        auto value = phi_1LPT * FML::GRID::FloatType(f_1LPT(kmag));
        if (LPT_order >= 2) {
            auto phi_2LPT = phi_2LPT_ini_fourier.get_fourier_from_index(fourier_index);
            value += phi_2LPT * FML::GRID::FloatType(f_2LPT(kmag));
        }
        grid.set_fourier_from_index(fourier_index, value);
    });
}

//========================================================================
/// Interpolate the LPT displacement grids of scale-dependent COLA to the Lagrangian positions q of the particles
/// (stored as the positions while they are swapped). The particle lattice may have been offset from the nodes of the
/// grids (ic_lattice_offset): the potentials are those of the lattice before the offset, so we interpolate at
/// q - lattice_offset, i.e. at the grid nodes when the IC grid has Npart_1D cells (exact with CIC; at the offset
/// positions CIC would smooth the displacements and the COLA frame would miss growth on small scales)
//========================================================================
template <int NDIM, class T>
void interpolate_LPT_grid_to_lagrangian_positions(const std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> & grid_vector_real,
                                                  FML::PARTICLE::MPIParticles<T> & part,
                                                  std::array<std::vector<FML::GRID::FloatType>, NDIM> & result,
                                                  const std::string & interpolation_method,
                                                  double lattice_offset) {
    if (lattice_offset == 0.0) {
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, T>(grid_vector_real,
                                                                                   part.get_particles_ptr(),
                                                                                   part.get_npart(),
                                                                                   result,
                                                                                   interpolation_method);
        return;
    }
    struct Point {
        double pos[NDIM];
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return NDIM; }
    };
    std::vector<Point> points(part.get_npart());
    auto * p = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < points.size(); i++) {
        const auto * q = FML::PARTICLE::GetPos(p[i]);
        for (int idim = 0; idim < NDIM; idim++) {
            points[i].pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(double(q[idim] - lattice_offset));
        }
    }
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, Point>(grid_vector_real,
                                                                                   points.data(),
                                                                                   points.size(),
                                                                                   result,
                                                                                   interpolation_method);
}

/// Run f with the particles at their Lagrangian positions (q swapped into the positions and the particles
/// communicated to the tasks of q), then swap back
template <class T, class F>
void with_particles_at_lagrangian_positions(FML::PARTICLE::MPIParticles<T> & part, F && f) {
    FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
    part.communicate_particles();
    f();
    FML::PARTICLE::swap_eulerian_and_lagrangian_positions(part.get_particles_ptr(), part.get_npart());
    part.communicate_particles();
}

/// The displacement field of sum_n f_n(k) phi_nLPT(k, zini) (n = 1, ..., LPT_order) at the Lagrangian positions of the
/// particles (they must be swapped: with_particles_at_lagrangian_positions)
template <int NDIM, class T>
std::array<std::vector<FML::GRID::FloatType>, NDIM>
lpt_displacement_at_lagrangian_positions(FML::PARTICLE::MPIParticles<T> & part,
                                         int LPT_order,
                                         const FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                         const FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                         const Spline & f_1LPT,
                                         const Spline & f_2LPT,
                                         double lattice_offset) {
    auto potential = phi_1LPT_ini_fourier;
    sum_of_LPT_potentials<NDIM>(potential, LPT_order, phi_1LPT_ini_fourier, phi_2LPT_ini_fourier, f_1LPT, f_2LPT);
    std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> grid_vector_real;
    std::function<double(double)> multiply_by_one = []([[maybe_unused]] double kBox) { return 1.0; };
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector_scaledependent<NDIM>(potential,
                                                                                        grid_vector_real,
                                                                                        multiply_by_one);
    potential.free();
    for (int idim = 0; idim < NDIM; idim++)
        grid_vector_real[idim].communicate_boundaries();
    std::array<std::vector<FML::GRID::FloatType>, NDIM> result;
    interpolate_LPT_grid_to_lagrangian_positions<NDIM, T>(grid_vector_real, part, result, "CIC", lattice_offset);
    return result;
}

/// A factor f(kBox) of the scale-dependent COLA step as a spline on 4 Nmesh points over the modes of the LPT grids
/// (pi to sqrt(NDIM) pi Nmesh; std::function is slow in the loop over the modes)
template <int NDIM>
Spline cola_factor_spline(int Nmesh, const std::function<double(double)> & f) {
    const int npts = 4 * Nmesh;
    const double kmin = M_PI;
    const double kmax = 2.0 * M_PI * Nmesh / 2.0 * std::sqrt(double(NDIM));
    std::vector<double> k_vec(npts), values(npts);
    for (int i = 0; i < npts; i++) {
        k_vec[i] = kmin + (kmax - kmin) * i / double(npts - 1);
        values[i] = f(k_vec[i]);
    }
    Spline spline;
    spline.create(k_vec, values);
    return spline;
}

/// Store a displacement field (one value per particle) in a particle field (D_1LPT, D_2LPT or dDdloga_1LPT)
template <int NDIM, class T, class Get>
void cola_store_in_particles(FML::PARTICLE::MPIParticles<T> & part,
                             const std::array<std::vector<FML::GRID::FloatType>, NDIM> & field,
                             Get && storage) {
    const size_t np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < np; ind++) {
        auto * D = storage(part[ind]);
        for (int idim = 0; idim < NDIM; idim++)
            D[idim] = field[idim][ind];
    }
}

/// Check that the particles and the stored potentials have what scale-dependent COLA of order LPT_order needs
template <int NDIM, class T>
void cola_check_scaledependent(int LPT_order,
                               const FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                               const FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                               const std::string & where) {
    constexpr int LPT_order_particle = cola_LPT_order_of<T>();
    FML::assert_mpi(phi_1LPT_ini_fourier.get_nmesh() > 0,
                    ("Error in " + where + ". Initial 1LPT potential is not allocated\n").c_str());
    if (LPT_order >= 2)
        FML::assert_mpi(phi_2LPT_ini_fourier.get_nmesh() > 0,
                        ("Error in " + where + ". Initial 2LPT potential is not allocated\n").c_str());
    if (LPT_order_particle == 1)
        FML::assert_mpi(
            FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_dDdloga_1LPT<T>(),
            ("Error in " + where + ". Particle do not have both get_D_1LPT and get_dDdloga_1LPT methods\n").c_str());
    if (LPT_order >= 2)
        FML::assert_mpi(
            FML::PARTICLE::has_get_D_1LPT<T>() and FML::PARTICLE::has_get_D_2LPT<T>(),
            ("Error in " + where + ". Particle do not have both get_D_1LPT and get_D_2LPT methods\n").c_str());
    if (LPT_order_particle < LPT_order) {
        std::string error = "The particle type is compatible with " + std::to_string(LPT_order_particle) + "LPT, ";
        error +=
            "but we are asked to do " + std::to_string(LPT_order) + "LPT COLA. Lower the order or fix the particle\n";
        FML::assert_mpi(false, error.c_str());
    }
}

// The particle storage used for the velocity kick in cola_kick_drift_scaledependent (dDdloga_1LPT for 1LPT
// particles, D_2LPT otherwise)
template <class T>
auto * cola_scaledependent_vel_storage(T & p, int LPT_order_particle) {
    if constexpr (FML::PARTICLE::has_get_D_2LPT<T>()) {
        if (LPT_order_particle != 1)
            return FML::PARTICLE::GetD_2LPT(p);
    }
    if constexpr (FML::PARTICLE::has_get_dDdloga_1LPT<T>()) {
        return FML::PARTICLE::GetdDdloga_1LPT(p);
    } else {
        // Not reached: the particle has D_2LPT or cola_kick_drift_scaledependent fails before
        return FML::PARTICLE::GetD_1LPT(p);
    }
}

// Complete the step of cola_kick_drift_scaledependent(..., do_drift = false): add the stored position displacement
template <int NDIM, class T>
void cola_drift_scaledependent(FML::PARTICLE::MPIParticles<T> & part, int LPT_order) {
    if (LPT_order < 1)
        return;
    constexpr int LPT_order_particle = cola_LPT_order_of<T>();
    const size_t np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < np; i++) {
        auto & p = part[i];
        auto * pos = FML::PARTICLE::GetPos(p);
        auto * D_pos = cola_scaledependent_vel_storage<T>(p, LPT_order_particle);
        for (int idim = 0; idim < NDIM; idim++)
            pos[idim] += D_pos[idim];

        // Periodic wrap
        for (int idim = 0; idim < NDIM; idim++)
            pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(pos[idim]);
    }
}

// This method kick's and drift's the particles one time-step (the 1LPT and 2LPT frames)
// With do_drift = false only the velocities are updated and the position displacement is stored in the particles
// (see cola_drift_scaledependent, which completes the step)
template <int NDIM, class T>
void cola_kick_drift_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                    std::shared_ptr<GravityModel<NDIM>> & grav,
                                    int LPT_order,
                                    FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                    FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                    double H0Box,
                                    double aini,
                                    double aold,
                                    double a,
                                    double delta_time_kick,
                                    [[maybe_unused]] double delta_time_drift,
                                    double lattice_offset,
                                    bool do_drift) {
    if (LPT_order < 1)
        return;
    constexpr int LPT_order_particle = cola_LPT_order_of<T>();
    if (FML::ThisTask == 0)
        std::cout << "[Kick] + [Drift] Scaledependent COLA " << LPT_order << "LPT\n";
    cola_check_scaledependent<NDIM, T>(LPT_order,
                                       phi_1LPT_ini_fourier,
                                       phi_2LPT_ini_fourier,
                                       "cola_kick_drift_scaledependent");

    // The LPT force as in the growth equations: rho_m(a) a^3 of the background
    const double OmegaM = grav->get_cosmo()->Omega_m_a3(aold);

    //======================================================================================
    // The factors of the LPT potentials phi_nLPT(k, zini) (k = kBox in units of 1/boxsize):
    //   drift: (D_n - D_n,old) / D_n,ini
    //   kick:  -1.5 OmegaM a GeffOverG(k,a) dt_kick S_n (D_1 / D_1,ini for 1LPT, (D_2 - nonlinear D_1^2) / D_2,ini
    //          for 2LPT) with S_n the source factor of the LPT equation
    //======================================================================================
    auto kick_factor = [&](double koverH0) {
        return -1.5 * OmegaM * aold * grav->GeffOverG(aold, koverH0) * delta_time_kick;
    };
    const int Nmesh = phi_1LPT_ini_fourier.get_nmesh();
    const Spline drift_1LPT = cola_factor_spline<NDIM>(Nmesh, [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return (grav->get_D_1LPT(a, koverH0) - grav->get_D_1LPT(aold, koverH0)) / grav->get_D_1LPT(aini, koverH0);
    });
    const Spline kick_1LPT = cola_factor_spline<NDIM>(Nmesh, [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return kick_factor(koverH0) * grav->source_factor_1LPT(aold, koverH0) * grav->get_D_1LPT(aold, koverH0) /
               grav->get_D_1LPT(aini, koverH0);
    });
    Spline drift_2LPT, kick_2LPT;
    if (LPT_order >= 2) {
        drift_2LPT = cola_factor_spline<NDIM>(Nmesh, [&](double kBox) {
            const double koverH0 = kBox / H0Box;
            return (grav->get_D_2LPT(a, koverH0) - grav->get_D_2LPT(aold, koverH0)) / grav->get_D_2LPT(aini, koverH0);
        });
        kick_2LPT = cola_factor_spline<NDIM>(Nmesh, [&](double kBox) {
            const double koverH0 = kBox / H0Box;
            const double D_1LPT = grav->get_D_1LPT(aold, koverH0);
            const double nonlinear = grav->nonlinear_source_factor_2LPT(aold, koverH0);
            return kick_factor(koverH0) * grav->source_factor_2LPT(aold, koverH0) *
                   (grav->get_D_2LPT(aold, koverH0) - nonlinear * D_1LPT * D_1LPT) / grav->get_D_2LPT(aini, koverH0);
        });
    }

    //======================================================================================
    // The drift and kick displacements at the Lagrangian positions, stored in the particles: the drift in D_1LPT,
    // the kick in dDdloga_1LPT (1LPT particles) or D_2LPT
    //======================================================================================
    with_particles_at_lagrangian_positions(part, [&]() {
        cola_store_in_particles<NDIM, T>(part,
                                         lpt_displacement_at_lagrangian_positions<NDIM, T>(part,
                                                                                           LPT_order,
                                                                                           phi_1LPT_ini_fourier,
                                                                                           phi_2LPT_ini_fourier,
                                                                                           drift_1LPT,
                                                                                           drift_2LPT,
                                                                                           lattice_offset),
                                         [](T & p) { return FML::PARTICLE::GetD_1LPT(p); });
        cola_store_in_particles<NDIM, T>(
            part,
            lpt_displacement_at_lagrangian_positions<NDIM, T>(part,
                                                              LPT_order,
                                                              phi_1LPT_ini_fourier,
                                                              phi_2LPT_ini_fourier,
                                                              kick_1LPT,
                                                              kick_2LPT,
                                                              lattice_offset),
            [](T & p) { return cola_scaledependent_vel_storage<T>(p, LPT_order_particle); });
    });

    //================================================================
    // The kick and drift
    //================================================================
    const size_t np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < np; i++) {
        auto & p = part[i];
        auto * pos = FML::PARTICLE::GetPos(p);
        auto * vel = FML::PARTICLE::GetVel(p);
        auto * D_pos = FML::PARTICLE::GetD_1LPT(p);
        auto * D_vel = cola_scaledependent_vel_storage<T>(p, LPT_order_particle);
        for (int idim = 0; idim < NDIM; idim++) {
            if (do_drift)
                pos[idim] += D_pos[idim];
            vel[idim] += D_vel[idim];
        }
        // Keep the position displacement for cola_drift_scaledependent: in the velocity storage as D1 is
        // overwritten when adding on the LPT velocity (cola_add_on_LPT_velocity_scaledependent)
        if (not do_drift) {
            for (int idim = 0; idim < NDIM; idim++)
                D_vel[idim] = D_pos[idim];
            continue;
        }

        // Periodic wrap
        for (int idim = 0; idim < NDIM; idim++)
            pos[idim] = FML::PARTICLE::wrap_coordinate_periodic(pos[idim]);
    }
}

/// Add sign times the LPT velocity a^2 H/H0 sum_n f_n(k,a) D_n(k,a) / D_n(k,aini) Psi_nLPT(k, zini) to the
/// particles (to go between the COLA frame and the full velocity)
template <int NDIM, class T>
void cola_add_on_LPT_velocity_scaledependent(FML::PARTICLE::MPIParticles<T> & part,
                                             std::shared_ptr<GravityModel<NDIM>> & grav,
                                             int LPT_order,
                                             FML::GRID::FFTWGrid<NDIM> & phi_1LPT_ini_fourier,
                                             FML::GRID::FFTWGrid<NDIM> & phi_2LPT_ini_fourier,
                                             double H0Box,
                                             double aini,
                                             double a,
                                             double sign,
                                             double lattice_offset) {
    if (LPT_order < 1)
        return;
    if (FML::ThisTask == 0)
        std::cout << "Adding on the LPT velocity to particles (COLA " << LPT_order << "LPT scaledependent)\n";
    cola_check_scaledependent<NDIM, T>(LPT_order,
                                       phi_1LPT_ini_fourier,
                                       phi_2LPT_ini_fourier,
                                       "cola_add_on_LPT_velocity_scaledependent");

    const double vfactor = sign * a * a * grav->get_cosmo()->HoverH0_of_a(a);
    const int Nmesh = phi_1LPT_ini_fourier.get_nmesh();
    const Spline vel_1LPT = cola_factor_spline<NDIM>(Nmesh, [&](double kBox) {
        const double koverH0 = kBox / H0Box;
        return vfactor * grav->get_f_1LPT(a, koverH0) * grav->get_D_1LPT(a, koverH0) / grav->get_D_1LPT(aini, koverH0);
    });
    Spline vel_2LPT;
    if (LPT_order >= 2)
        vel_2LPT = cola_factor_spline<NDIM>(Nmesh, [&](double kBox) {
            const double koverH0 = kBox / H0Box;
            return vfactor * grav->get_f_2LPT(a, koverH0) * grav->get_D_2LPT(a, koverH0) /
                   grav->get_D_2LPT(aini, koverH0);
        });

    // The velocity at the Lagrangian positions, stored in D_1LPT
    with_particles_at_lagrangian_positions(part, [&]() {
        cola_store_in_particles<NDIM, T>(part,
                                         lpt_displacement_at_lagrangian_positions<NDIM, T>(part,
                                                                                           LPT_order,
                                                                                           phi_1LPT_ini_fourier,
                                                                                           phi_2LPT_ini_fourier,
                                                                                           vel_1LPT,
                                                                                           vel_2LPT,
                                                                                           lattice_offset),
                                         [](T & p) { return FML::PARTICLE::GetD_1LPT(p); });
    });

    const size_t np = part.get_npart();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < np; ind++) {
        auto * vel = FML::PARTICLE::GetVel(part[ind]);
        auto * dD = FML::PARTICLE::GetD_1LPT(part[ind]);
        for (int idim = 0; idim < NDIM; idim++)
            vel[idim] += dD[idim];
    }
}

//========================================================================
/// Everything COLA needs during the simulation: the settings and (for
/// scale-dependent COLA) the LPT potentials at the initial time. This
/// selects between the scale-dependent and scale-independent methods
//========================================================================
template <int NDIM, class T>
class ColaState {
  public:
    bool enabled{false};        // Use COLA?
    bool scaledependent{false}; // Use the scale-dependent version?
    int LPT_order{2};           // The LPT order of COLA
    double aini{1.0};           // The initial scale factor
    double H0Box{1.0};          // H0 * boxsize
    double lattice_offset{0.0}; // The offset of the particle lattice (ic_lattice_offset) from the lattice of the
                                // stored LPT potentials (in units of the box)
    std::shared_ptr<GravityModel<NDIM>> grav;

    // The LPT potentials at the initial time (only allocated for scale-dependent COLA)
    FML::GRID::FFTWGrid<NDIM> phi_1LPT_ini_fourier;
    FML::GRID::FFTWGrid<NDIM> phi_2LPT_ini_fourier;

    /// A species on its own lattice, offset by shift (in each direction) from the lattice of the potentials above
    /// (the baryons with particle_baryon_lattice = offset). Its frame is computed from its own potentials: those of
    /// the density field shifted by it, f(q) -> f(q + shift), as for its initial conditions (the 2LPT potential of
    /// the shifted field), evaluated at its lattice nodes (q - lattice_offset - shift). Interpolating the potentials
    /// above to its positions between the nodes smoothed its frame: the baryon velocities at the initial redshift
    /// differed by 22% rms from those without COLA. Only for scale-dependent COLA (at most 2LPT)
    struct ShiftedSpecies {
        const FML::PARTICLE::MPIParticles<T> * part{nullptr};
        double shift{0.0};
        FML::GRID::FFTWGrid<NDIM> phi_1LPT_ini_fourier;
        FML::GRID::FFTWGrid<NDIM> phi_2LPT_ini_fourier;
    };
    std::vector<ShiftedSpecies> shifted_species;

    /// Add (sign = +1) or subtract (sign = -1) the LPT velocity at scale factor a, i.e. go from
    /// the COLA frame to the standard frame and back
    void add_on_LPT_velocity(FML::PARTICLE::MPIParticles<T> & part, double a, double sign) {
        if (scaledependent) {
            auto * sp = shifted_species_of(part);
            cola_add_on_LPT_velocity_scaledependent<NDIM, T>(part,
                                                             grav,
                                                             LPT_order,
                                                             sp ? sp->phi_1LPT_ini_fourier : phi_1LPT_ini_fourier,
                                                             sp ? sp->phi_2LPT_ini_fourier : phi_2LPT_ini_fourier,
                                                             H0Box,
                                                             aini,
                                                             a,
                                                             sign,
                                                             sp ? lattice_offset + sp->shift : lattice_offset);
        } else {
            cola_add_on_LPT_velocity<NDIM, T>(part, grav, LPT_order, aini, a, sign);
        }
    }

    /// The COLA kick and drift from aold to a
    void kick_drift(FML::PARTICLE::MPIParticles<T> & part,
                    double aold,
                    double a,
                    double delta_time_kick,
                    double delta_time_drift) {
        if (scaledependent) {
            auto * sp = shifted_species_of(part);
            cola_kick_drift_scaledependent<NDIM, T>(part,
                                                    grav,
                                                    LPT_order,
                                                    sp ? sp->phi_1LPT_ini_fourier : phi_1LPT_ini_fourier,
                                                    sp ? sp->phi_2LPT_ini_fourier : phi_2LPT_ini_fourier,
                                                    H0Box,
                                                    aini,
                                                    aold,
                                                    a,
                                                    delta_time_kick,
                                                    delta_time_drift,
                                                    sp ? lattice_offset + sp->shift : lattice_offset);
        } else {
            cola_kick_drift<NDIM, T>(part, grav, LPT_order, aini, aold, a, delta_time_kick, delta_time_drift);
        }
    }

    /// kick_drift in two parts: first the COLA kick (the velocities), then drift (the positions from aold to a).
    /// Between the two the particles are at aold, e.g. for the lightcone, which drifts them with the true velocity.
    /// The result is identical to kick_drift
    void kick(FML::PARTICLE::MPIParticles<T> & part, double aold, double a, double delta_time_kick) {
        if (scaledependent) {
            // This also computes the position displacement (stored in the particles until drift)
            auto * sp = shifted_species_of(part);
            cola_kick_drift_scaledependent<NDIM, T>(part,
                                                    grav,
                                                    LPT_order,
                                                    sp ? sp->phi_1LPT_ini_fourier : phi_1LPT_ini_fourier,
                                                    sp ? sp->phi_2LPT_ini_fourier : phi_2LPT_ini_fourier,
                                                    H0Box,
                                                    aini,
                                                    aold,
                                                    a,
                                                    delta_time_kick,
                                                    0.0,
                                                    sp ? lattice_offset + sp->shift : lattice_offset,
                                                    false);
        } else {
            cola_kick_drift<NDIM, T>(part, grav, LPT_order, aini, aold, a, delta_time_kick, 0.0, true, false);
        }
    }
    void drift(FML::PARTICLE::MPIParticles<T> & part, double aold, double a) {
        if (scaledependent) {
            cola_drift_scaledependent<NDIM, T>(part, LPT_order);
        } else {
            cola_kick_drift<NDIM, T>(part, grav, LPT_order, aini, aold, a, 0.0, 0.0, false, true);
        }
    }

  private:
    ShiftedSpecies * shifted_species_of(const FML::PARTICLE::MPIParticles<T> & part) {
        for (auto & sp : shifted_species)
            if (sp.part == &part)
                return &sp;
        return nullptr;
    }
};

//========================================================================
/// While this object exists the particles have their true velocities (the
/// LPT velocity at scale factor a is added on) instead of the COLA velocities.
/// Does nothing if COLA is not used
//========================================================================
template <int NDIM, class T>
class ScopedLPTVelocity {
  public:
    ScopedLPTVelocity(ColaState<NDIM, T> & cola,
                      FML::PARTICLE::MPIParticles<T> & part,
                      double a,
                      FML::UTILS::Timings * timer = nullptr)
        : cola(cola), part(part), a(a), timer(timer) {
        apply(+1.0);
    }
    ~ScopedLPTVelocity() { apply(-1.0); }
    ScopedLPTVelocity(const ScopedLPTVelocity &) = delete;
    ScopedLPTVelocity & operator=(const ScopedLPTVelocity &) = delete;

  private:
    ColaState<NDIM, T> & cola;
    FML::PARTICLE::MPIParticles<T> & part;
    double a;
    FML::UTILS::Timings * timer;

    void apply(double sign) {
        if (not cola.enabled)
            return;
        if (timer)
            timer->StartTiming("COLA output");
        cola.add_on_LPT_velocity(part, a, sign);
        if (timer)
            timer->EndTiming("COLA output");
    }
};

#endif
