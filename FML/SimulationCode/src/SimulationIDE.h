#ifndef SIMULATIONIDE_HEADER
#define SIMULATIONIDE_HEADER

#include "Simulation.h"

//=============================================================================
// Interacting dark energy (cosmology_model = IDE): the energy exchange Q = xi H rho_DE along the dark-matter
// 4-velocity (Valiviita, Majerotto & Maartens 2008, arXiv:0804.0232), with the rate perturbed by the local expansion
// of the dark matter, delta Q / Q = delta_DE - Psi + E, E = (theta_c - 3 Phi')/(3 Hp) (as in the Einstein-Boltzmann
// solver, FML/Cosmology/Perturbations/SpeciesPerturbation_DarkEnergyFluid.h). On sub-horizon scales (delta_DE = 0 for
// cs2 = 1, Psi negligible) the dark matter of a fluid element of physical volume V gains
//   dm/dt = Q V = xi rho_DE (H + theta_c/3) V = (xi/3) rho_DE dV/dt
// so a CDM particle gains xi/3 rho_DE times the change of its volume, and its momentum is not changed (the energy
// comes with the dark-matter velocity; the Euler equation is that of LCDM). In units of the mean particle mass today
// (Omega_s: the CDM, or the CDM+baryons as one fluid) with V relative to the mean comoving volume, s = 1/(1 + delta_n)
// (delta_n: the number density of the particles at the particle):
//   dM = xi D (s dlna + ds/3),   D = rho_DE a^3 / Omega_s
// The mean, s = 1, is the background mass factor Mbar(a) (cdm_mass_factor or cb_mass_factor, dMbar = xi D dlna), so
// between two updates (midpoint in time and in s)
//   M_new = M_old + (Mbar_new - Mbar_old) + xi D_mid [(s_mid - 1) dlna + (s_new - s_old)/3]
// The particles store M / Mbar (mass, ~1; the density assignment weights them by m/<m> and the background mass is in
// the Poisson source, Cosmology::clustering_mass_factor) and s (volume). delta_n: CIC on a grid of particle_Npart_1D
// cells (the particle lattice: no grid noise), interpolated back with CIC, so the mass perturbation is that of
// delta_n smoothed with the window W_CIC^2 of the particle grid, right for k << the particle Nyquist frequency k_N
// (xi = -0.1, w = -0.9, z = 49 -> 0: the extra growth of delta_cb over xi = 0, +10.3%, is 0.1 points too small for
// k < k_N/5 and 0.8 points for k < k_N/2; deconvolving W^2 fixes that in linear theory, but in the nonlinear regime it
// makes delta_n so noisy that the masses fluctuate by tens of percent).
// The mass cannot become negative (xi < 0: Q does not depend on rho_c, so the dark matter in a void can run out);
// 1 + delta_n >= 0.05. The first update sets the masses from linear theory, M/Mbar = 1 + m1 delta_n
// (GravityModel::ide_mass_growth; m1 ~ 0 at high redshift).
// Called before the force of every step (the particles are then at a). To linear order this gives
//   delta_c' = delta_n' (1 - q/3) - q delta_c,   q = xi rho_DE / rho_c
// the solver's equation; the CDM growth from z = 49 to 0 differs from that of the particle masses changing uniformly
// (no delta_c terms) by ~10% for xi = -0.1, w = -0.9.
//
// References:
//   Valiviita, Majerotto & Maartens 2008, JCAP 07, 020, arXiv:0804.0232   (the model Q = xi H rho_DE, its perturbations
//       and the large-scale instability for xi / (1 + w) > 0)
//   Other N-body implementations change the particle masses only with the background, m(a), with the Euler equation of
//   LCDM: Zhai, van de Bruck, Di Valentino, Li & Nunes 2026, arXiv:2606.11368 (RAMSES); ME-Gadget (Zhang et al. 2018,
//   arXiv:1811.01519; for its more general couplings also a velocity drag and an effective G). Here the mass also
//   follows the local expansion, as the perturbed rate of the Einstein-Boltzmann solver does
//=============================================================================

template <int NDIM, class T>
void NBodySimulation<NDIM, T>::update_ide_masses(double a) {
    if (not cosmo->is_ide() or a == ide_masses_a)
        return;
    if constexpr (FML::PARTICLE::has_get_mass<T>()) {
        timer.StartTiming("IDEMasses");
        const double aold = ide_masses_a;
        const bool first = not(aold > 0.0);
        const double xi = cosmo->ide_xi();
        const double dlna = first ? 0.0 : std::log(a / aold);
        const double amid = first ? a : std::sqrt(a * aold);

        for_each_cb_species([&](MPIParticles & particles, const SIMCODE::SpeciesInfo & info) {
            if (info.is(SIMCODE::ParticleSpecies::Baryon))
                return;
            const bool cdm_only = info.is(SIMCODE::ParticleSpecies::CDM);
            const double Omega_s = cosmo->get_OmegaCDM() + (cdm_only ? 0.0 : cosmo->get_Omegab());
            auto mbar = [&](double aa) { return cdm_only ? cosmo->cdm_mass_factor(aa) : cosmo->cb_mass_factor(aa); };
            const double mbar_new = mbar(a), mbar_old = first ? mbar_new : mbar(aold);
            const double D_mid = cosmo->ide_rho_de_a3(amid) / Omega_s;
            // The linear mass growth is that of the CDM+baryon fluid: the CDM alone has 1/f_c of it
            const double m1 = grav->ide_mass_growth(a) / (cdm_only ? cosmo->cdm_fraction_cb(a) : 1.0);

            auto * p = particles.get_particles_ptr();
            const size_t np = particles.get_npart();

            // The number density (the masses set to 1 for the assignment)
            std::vector<double> masses(np);
            for (size_t i = 0; i < np; i++) {
                masses[i] = FML::PARTICLE::GetMass(p[i]);
                FML::PARTICLE::SetMass(p[i], 1.0);
            }
            const std::string method = "CIC";
            const auto nleftright = FML::INTERPOLATION::get_extra_slices(method);
            FFTWGrid<NDIM> density(params.particle.Npart_1D, nleftright.first, nleftright.second);
            FML::INTERPOLATION::particles_to_grid(p, np, particles.get_npart_total(), density, method);
            density.communicate_boundaries();
            std::vector<FML::GRID::FloatType> delta_n;
            FML::INTERPOLATION::interpolate_grid_to_particle_positions(density, p, np, delta_n, method);
            density.free();

            // The mass cannot become negative (xi < 0: the dark matter in a void can run out; the CDM+baryon particles
            // keep their baryons)
            const double w_min = (cdm_only ? 0.0 : cosmo->get_Omegab() / Omega_s) / mbar_new;
            double max_change = 0.0, wmin = 1e30, wmax = -1e30;
            size_t nclamped = 0;
#ifdef USE_OMP
#pragma omp parallel for schedule(static) reduction(max : max_change, wmax) reduction(min : wmin)                      \
    reduction(+ : nclamped)
#endif
            for (size_t i = 0; i < np; i++) {
                const double s_new = 1.0 / std::max(1.0 + double(delta_n[i]), 0.05);
                double w;
                if (first or not(p[i].volume > 0.0)) {
                    w = 1.0 + m1 * (1.0 / s_new - 1.0);
                } else {
                    const double s_old = p[i].volume;
                    const double M_old = mbar_old * masses[i];
                    const double M_new = M_old + (mbar_new - mbar_old) +
                                         xi * D_mid * ((0.5 * (s_old + s_new) - 1.0) * dlna + (s_new - s_old) / 3.0);
                    w = M_new / mbar_new;
                    if (w < w_min) {
                        w = w_min;
                        nclamped++;
                    }
                    max_change = std::max(max_change, std::fabs(w - masses[i]));
                }
                wmin = std::min(wmin, w);
                wmax = std::max(wmax, w);
                FML::PARTICLE::SetMass(p[i], w);
                p[i].volume = s_new;
            }
            FML::MaxOverTasks(&max_change);
            FML::MaxOverTasks(&wmax);
            FML::MinOverTasks(&wmin);
            FML::SumOverTasks(&nclamped);
            if (FML::ThisTask == 0)
                std::cout << "# IDE: CDM" << (cdm_only ? "" : "+baryon") << " particle masses updated to a = " << a
                          << " (mean mass " << mbar_new << " of today's; m/<m> in [" << wmin << ", " << wmax
                          << "], largest change " << max_change << ", " << nclamped << " at the minimum)\n";
        });
        ide_masses_a = a;
        timer.EndTiming("IDEMasses");
    } else {
        throw std::runtime_error("cosmology_model = IDE needs particles with a mass (ParticleWithMass)");
    }
}

#endif
