#ifndef KICKDRIFT_HEADER
#define KICKDRIFT_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NBody/NBody.h>

#include "Neutrinos.h"
#include "Species.h"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

//=============================================================================
// The kick and drift of the particles of a species. How a species moves is described by its SpeciesInfo
// (Species.h): non-relativistic particles (CDM, baryons) use the standard leapfrog operators of the library
// (FML::NBODY::KickParticles / DriftParticles), relativistic particles (the neutrinos) the relativistic factors of
// Neutrinos.h. This is the place to add species-dependent dynamics (e.g. an extra drag on the dark matter in
// coupled dark energy, or a species-dependent force grid).
//=============================================================================

namespace COLASOLVER {

    /// The kick of a species: v -> velocity_scale v - F dt (times (1 + 2u^2) / sqrt(1 + u^2) at apos for
    /// relativistic species, u = |v| / (a c))
    template <int NDIM, class P>
    void kick_particles(FML::PARTICLE::MPIParticles<P> & part,
                        const SpeciesInfo & info,
                        std::array<FML::GRID::FFTWGrid<NDIM>, NDIM> & force_real,
                        double apos,
                        double delta_time,
                        double velocity_scale,
                        const std::string & interpolation_method,
                        double boxsize) {
        auto * p = part.get_particles_ptr();
        const size_t npart = part.get_npart();

        if (not info.relativistic) {
            if (velocity_scale != 1.0) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < npart; i++) {
                    auto * vel = FML::PARTICLE::GetVel(p[i]);
                    for (int idim = 0; idim < NDIM; idim++)
                        vel[idim] *= velocity_scale;
                }
            }
            FML::NBODY::KickParticles<NDIM>(force_real, part, delta_time, interpolation_method);
            return;
        }

        // Relativistic species
        namespace NU = COLASOLVER::NEUTRINOS;
        for (int idim = 0; idim < NDIM; idim++)
            force_real[idim].communicate_boundaries();
        std::array<std::vector<FML::GRID::FloatType>, NDIM> force;
        FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<NDIM, P>(
            force_real, p, npart, force, interpolation_method);
        const double inv_ac = 1.0 / (apos * NU::c_code(boxsize));
        double max_u = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_u)
#endif
        for (size_t i = 0; i < npart; i++) {
            auto * vel = FML::PARTICLE::GetVel(p[i]);
            double v2 = 0.0;
            for (int idim = 0; idim < NDIM; idim++) {
                vel[idim] *= velocity_scale;
                v2 += vel[idim] * vel[idim];
            }
            const double u2 = v2 * inv_ac * inv_ac;
            max_u = std::max(max_u, u2);
            const double R = NU::kick_factor(u2);
            for (int idim = 0; idim < NDIM; idim++)
                vel[idim] -= force[idim][i] * delta_time * R;
        }
        FML::MaxOverTasks(&max_u);
        if (FML::ThisTask == 0)
            std::cout << "[Kick " << info.name << "] max u = q/(amc) = " << std::sqrt(max_u) << "\n";
    }

    /// The drift of a species: x -> x + v dt (times 1 / sqrt(1 + u^2) with u at the middle of the drift for
    /// relativistic species). The particles are wrapped and communicated
    template <int NDIM, class P>
    void drift_particles(FML::PARTICLE::MPIParticles<P> & part,
                         const SpeciesInfo & info,
                         double apos,
                         double apos_new,
                         double delta_time,
                         double boxsize) {
        if (not info.relativistic) {
            FML::NBODY::DriftParticles<NDIM, P>(part, delta_time);
            return;
        }

        // Relativistic species
        namespace NU = COLASOLVER::NEUTRINOS;
        auto * p = part.get_particles_ptr();
        const size_t npart = part.get_npart();
        const double amid = 0.5 * (apos + apos_new);
        const double inv_ac = 1.0 / (amid * NU::c_code(boxsize));
        double max_disp = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_disp)
#endif
        for (size_t i = 0; i < npart; i++) {
            auto * pos = FML::PARTICLE::GetPos(p[i]);
            auto * vel = FML::PARTICLE::GetVel(p[i]);
            double v2 = 0.0;
            for (int idim = 0; idim < NDIM; idim++)
                v2 += vel[idim] * vel[idim];
            const double fac = delta_time * NU::drift_factor(v2 * inv_ac * inv_ac);
            for (int idim = 0; idim < NDIM; idim++) {
                const double disp = vel[idim] * fac;
                max_disp = std::max(max_disp, std::fabs(disp));
                double x = pos[idim] + disp;
                x -= std::floor(x);
                if (x >= 1.0)
                    x = 0.0;
                pos[idim] = x;
            }
        }
        FML::MaxOverTasks(&max_disp);
        if (FML::ThisTask == 0)
            std::cout << "[Drift " << info.name << "] Max displacement: " << max_disp << "\n";
        part.communicate_particles();
    }

} // namespace COLASOLVER

#endif
