#ifndef NEUTRINOPARTICLES_HEADER
#define NEUTRINOPARTICLES_HEADER

#include "TransferInput.h"
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Spline/Spline.h>

#include "ICFromParameters.h"
#include "Neutrinos.h"
#include "Particles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace SIMCODE {

    //=============================================================================
    /// Massive neutrinos as particles with the delta-f method (Elbers et al. 2021, arXiv:2010.07321). Owns the
    /// particles and does what is specific to them: the initial conditions (bulk LPT motion plus Fermi-Dirac
    /// thermal momenta), the delta-f weights (update_weights after every kick) and the (weighted) density field.
    /// The relativistic kick and drift are in KickDrift.h, the units and equations in Neutrinos.h.
    ///
    /// Neutrinos of different masses are one set of particles: all species have the same temperature and number
    /// density, so each particle is one of the species at random (species_of: from its ID only, the same for every
    /// number of tasks), with that mass in its thermal velocity (y = q c / (k_B T_nu0) is drawn as for one mass) and in
    /// its delta-f weight. Its weight also carries its mass relative to the mean mass, m_i / <m>, so the weighted
    /// density field is the mass-density contrast of all of them. The bulk part (the linear delta_nu and velocity)
    /// is that of all the neutrinos together for every particle (the transfer functions of their sum).
    //=============================================================================
    template <int NDIM>
    class NeutrinoParticles {
      public:
        using Particle = ParticleNeutrino<NDIM>;
        using Container = FML::PARTICLE::MPIParticles<Particle>;
        using TransferProvider = SIMCODE::TransferProvider;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using DVector = FML::INTERPOLATION::SPLINE::DVector;

        /// species: the massive neutrino species of the background (their masses, equal or not), Tnu0_eV: the
        /// temperature today, boxsize in Mpc/h, deltaf: use the delta-f weights (otherwise the weights are m_i / <m>,
        /// 1 for one mass, and the density is 1 + delta)
        void init(const NEUTRINOS::MassiveNeutrinoList & species, double _Tnu0_eV, double _boxsize, bool _deltaf) {
            if (species.empty())
                throw std::invalid_argument("NeutrinoParticles: no massive neutrinos");
            nu_species = species;
            masses.clear();
            for (auto & nu : species)
                masses.push_back(nu.mass_eV());
            mean_mass = 0.0;
            for (double m : masses) {
                if (not(m > 0.0))
                    throw std::invalid_argument("NeutrinoParticles: the masses must be > 0");
                mean_mass += m / double(masses.size());
            }
            single_mass = std::all_of(masses.begin(), masses.end(), [&](double m) { return m == masses.front(); });
            if (single_mass)
                mean_mass = masses.front();
            Tnu0_eV = _Tnu0_eV;
            boxsize = _boxsize;
            deltaf = _deltaf;
        }

        /// The species (index in the masses) of the particle with this ID, and its mass
        int species_of(long long int id) const {
            if (single_mass)
                return 0;
            return int(SIMCODE::NEUTRINOS::splitmix64(uint64_t(id) ^ 0x6E657574726E6FULL) % uint64_t(masses.size()));
        }
        double mass_of(long long int id) const { return masses[species_of(id)]; }
        const std::vector<double> & get_masses_eV() const { return masses; }

        Container & particles() { return part; }
        const Container & particles() const { return part; }

        /// The initial conditions. The bulk part is made like the CDM: delta_nu(k) = delta_cb(k) T_nu(k,zini) /
        /// T_cb(k,zini) from the same random field, 1LPT displacements and velocities v = a^2 H f_nu(k) Psi with f_nu
        /// from the transfer functions. On top of this each particle gets a thermal momentum q = y k_B T_nu0 / c in
        /// a random direction, with y drawn from the Fermi-Dirac distribution (the random numbers depend only on
        /// the seed and the particle ID). We store y and the linear delta_nu at the particle (for the weights)
        void generate(const FML::GRID::FFTWGrid<NDIM> & delta_cb_fourier,
                      const TransferProvider & transfers,
                      double zini,
                      double HoverH0_ini,
                      int Npart_1D,
                      double allocation_factor,
                      int random_seed,
                      long long int id_offset);

        /// The delta-f weights w = 1 - f0(y) / f_i where f_i = f0(y_ini) (1 + delta_ini) is the (conserved)
        /// phase-space density of the particle and y the current momentum in units of k_B T_nu0 / c. The weighted
        /// density (N_grid^3 / N_nu) sum w_i W(x - x_i) is then delta_nu (not 1 + delta_nu)
        void update_weights();

        /// The shot noise of the (weighted) density field in units of the box volume: sum w^2 / N^2 (1/N without
        /// the weights). Collective (all tasks must call it)
        double shotnoise();

        /// The mean energy per mass <E/m>(a) of the unperturbed neutrinos (all species; see
        /// NEUTRINOS::mean_energy_factor)
        double mean_energy_factor(double a) const { return SIMCODE::NEUTRINOS::mean_energy_factor(nu_species, a); }

        /// While this object lives the weights of the particles are w_i E_i / (m <E/m>(a)) with E_i / m =
        /// sqrt(1 + u_i^2) (several masses: w_i m_i/<m> E_i / (m_i <E/m>(a)) with the <E/m> of all species, so that
        /// the mass ratio carries over), so the (weighted) density field is the energy-density contrast
        /// delta_nu = delta rho_nu / rho_nu (as in CAMB/CLASS) instead of the mass-density contrast; also its shot
        /// noise. The original weights are restored (exactly) when it goes out of scope. Nothing if enabled = false
        class ScopedEnergyWeights {
          public:
            ScopedEnergyWeights(NeutrinoParticles & nu, double a, bool enabled) : nu(nu), enabled(enabled) {
                if (not enabled)
                    return;
                namespace NU = SIMCODE::NEUTRINOS;
                auto * p = nu.part.get_particles_ptr();
                const size_t npart = nu.part.get_npart();
                const double inv_ac = 1.0 / (a * NU::c_code(nu.boxsize));
                const double inv_mean = 1.0 / nu.mean_energy_factor(a);
                saved.resize(npart);
#ifdef USE_OMP
#pragma omp parallel for
#endif
                for (size_t i = 0; i < npart; i++) {
                    double v2 = 0.0;
                    for (int idim = 0; idim < NDIM; idim++)
                        v2 += p[i].vel[idim] * p[i].vel[idim];
                    saved[i] = p[i].weight;
                    p[i].weight = float(double(p[i].weight) * std::sqrt(1.0 + v2 * inv_ac * inv_ac) * inv_mean);
                }
            }
            ~ScopedEnergyWeights() {
                if (not enabled)
                    return;
                auto * p = nu.part.get_particles_ptr();
                for (size_t i = 0; i < saved.size(); i++)
                    p[i].weight = saved[i];
            }
            ScopedEnergyWeights(const ScopedEnergyWeights &) = delete;
            ScopedEnergyWeights & operator=(const ScopedEnergyWeights &) = delete;

          private:
            NeutrinoParticles & nu;
            bool enabled;
            std::vector<float> saved;
        };

        /// The neutrino density delta_nu(k) (the grid must be allocated). The k = 0 mode is set to zero
        void density_fourier(FML::GRID::FFTWGrid<NDIM> & density, const std::string & density_assignment_method);

      private:
        Container part;
        NEUTRINOS::MassiveNeutrinoList nu_species;
        std::vector<double> masses;
        double mean_mass{0.0};
        bool single_mass{true};
        double Tnu0_eV{0.0};
        double boxsize{1.0};
        bool deltaf{true};
    };

    template <int NDIM>
    void NeutrinoParticles<NDIM>::generate(const FML::GRID::FFTWGrid<NDIM> & delta_cb_fourier,
                                           const TransferProvider & transfers,
                                           double zini,
                                           double HoverH0_ini,
                                           int Npart_1D,
                                           double allocation_factor,
                                           int random_seed,
                                           long long int id_offset) {
        namespace NU = SIMCODE::NEUTRINOS;
        const double aini = 1.0 / (1.0 + zini);
        if (aini < 1.0 / (1.0 + SIMCODE::transfer_zmax(transfers)) * (1.0 - 1e-10))
            throw std::runtime_error("particle_neutrinos: the transfer functions only go up to z = " +
                                     std::to_string(SIMCODE::transfer_zmax(transfers)) + " (need zini)");

        // T_nu / T_cb and f_nu at zini
        const double kmin = transfers.range().kmin_hmpc;
        const double kmax = transfers.range().kmax_hmpc;
        const Spline ratio_spline = SIMCODE::spline_of_logk(
            kmin,
            kmax,
            [&](double logk) {
                const double k = std::exp(logk);
                return transfers.density_over_k_mpc2(SIMCODE::TransferSpecies::MassiveNeutrino, k, aini) /
                       transfers.density_over_k_mpc2(SIMCODE::TransferSpecies::CDMBaryon, k, aini);
            },
            "T_nu/T_cb(k,zini)");
        const Spline f_nu_spline = SIMCODE::spline_of_logk(
            kmin,
            kmax,
            [&](double logk) {
                return transfers.density_growth_rate(SIMCODE::TransferSpecies::MassiveNeutrino, std::exp(logk), aini);
            },
            "f_nu(k,zini)");
        if (FML::ThisTask == 0) {
            std::cout << "# Neutrino particles at z = " << zini << ": T_nu/T_cb, f_nu =";
            for (double k : {0.01, 0.1, 1.0})
                std::cout << " " << ratio_spline(std::log(k)) << ", " << f_nu_spline(std::log(k)) << " (k = " << k
                          << ")";
            std::cout << "\n";
        }
        auto clamp_logk = [&](double kBox) { return SIMCODE::clamped_logk(kBox / boxsize, kmin, kmax); };

        // The bulk part (1LPT) from delta_nu(k, zini)
        const double fac = aini * aini * HoverH0_ini;
        const std::vector<double> velocity_norms{fac * f_nu_spline(std::log(0.1)), 0.0, 0.0, 0.0};
        auto delta_nu = FML::NBODY::generate_species_from_delta_cb<NDIM, Particle>(
            part,
            delta_cb_fourier,
            [&](double kBox) { return ratio_spline(clamp_logk(kBox)); },
            [&](double kBox) { return fac * f_nu_spline(clamp_logk(kBox)); },
            velocity_norms,
            1,
            Npart_1D,
            allocation_factor,
            boxsize,
            zini,
            0.0,
            id_offset);

        // The linear delta_nu at the particles
        delta_nu.fftw_c2r();
        delta_nu.communicate_boundaries();
        std::vector<FML::GRID::FloatType> delta_at_particles;
        FML::INTERPOLATION::interpolate_grid_to_particle_positions<NDIM, Particle>(delta_nu,
                                                                                   part.get_particles_ptr(),
                                                                                   part.get_npart(),
                                                                                   delta_at_particles,
                                                                                   "CIC");
        delta_nu.free();

        // The thermal momenta (the velocity scale (T_nu0 / m_i) c of the particle's species)
        const double vth_norm = Tnu0_eV / mean_mass * NU::c_code(boxsize);
        const NU::FermiDiracSampler sampler;
        auto * p = part.get_particles_ptr();
        double min_delta = 1e100;
#ifdef USE_OMP
#pragma omp parallel for reduction(min : min_delta)
#endif
        for (size_t i = 0; i < part.get_npart(); i++) {
            p[i].delta_ini = float(delta_at_particles[i]);
            min_delta = std::min(min_delta, double(delta_at_particles[i]));
            NU::ParticleRNG rng(uint64_t(random_seed), uint64_t(p[i].id));
            const double y = sampler(rng.uniform());
            p[i].y_ini = float(y);
            const double cos_theta = 2.0 * rng.uniform() - 1.0;
            const double phi = 2.0 * M_PI * rng.uniform();
            const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
            const double dir[3] = {sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta};
            const double vth = single_mass ? vth_norm : Tnu0_eV / mass_of(p[i].id) * NU::c_code(boxsize);
            for (int idim = 0; idim < NDIM; idim++)
                p[i].vel[idim] += y * vth * (NDIM == 3 ? dir[idim] : (idim == 0 ? 1.0 : 0.0));
        }
        FML::MinOverTasks(&min_delta);
        if (min_delta <= -1.0)
            throw std::runtime_error("particle_neutrinos: delta_nu <= -1 in the initial conditions (start earlier)");
        update_weights();
        if (FML::ThisTask == 0)
            std::cout << "# Neutrino thermal velocity scale (T_nu0 / <m_nu>) c = " << vth_norm << " (code units), "
                      << vth_norm * 100.0 * boxsize
                      << " km/s today. Min delta_nu(zini) at the particles = " << min_delta << "\n";
    }

    template <int NDIM>
    void NeutrinoParticles<NDIM>::update_weights() {
        namespace NU = SIMCODE::NEUTRINOS;
        auto * p = part.get_particles_ptr();
        const size_t npart = part.get_npart();
        // Several masses: the weights carry m_i / <m>
        if (not deltaf) {
            for (size_t i = 0; i < npart; i++)
                p[i].weight = single_mass ? 1.0f : float(mass_of(p[i].id) / mean_mass);
            return;
        }
        const double y_over_v = 1.0 / NU::c_code(boxsize) * mean_mass / Tnu0_eV;
        double sum_w = 0.0, sum_w2 = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : sum_w, sum_w2)
#endif
        for (size_t i = 0; i < npart; i++) {
            double v2 = 0.0;
            for (int idim = 0; idim < NDIM; idim++)
                v2 += p[i].vel[idim] * p[i].vel[idim];
            const double mass_ratio = single_mass ? 1.0 : mass_of(p[i].id) / mean_mass;
            const double y = std::sqrt(v2) * y_over_v * mass_ratio;
            const double fi = NU::fermi_dirac(p[i].y_ini) * (1.0 + p[i].delta_ini);
            const double w = single_mass ? 1.0 - NU::fermi_dirac(y) / fi : (1.0 - NU::fermi_dirac(y) / fi) * mass_ratio;
            p[i].weight = float(w);
            sum_w += w;
            sum_w2 += w * w;
        }
        FML::SumOverTasks(&sum_w);
        FML::SumOverTasks(&sum_w2);
        const double N = double(part.get_npart_total());
        if (FML::ThisTask == 0)
            std::cout << "[Neutrinos] delta-f weights: mean " << sum_w / N << " rms " << std::sqrt(sum_w2 / N) << "\n";
    }

    template <int NDIM>
    double NeutrinoParticles<NDIM>::shotnoise() {
        const double N = double(part.get_npart_total());
        if (not deltaf and single_mass)
            return 1.0 / N;
        double sum_w2 = 0.0;
        auto * p = part.get_particles_ptr();
        for (size_t i = 0; i < part.get_npart(); i++)
            sum_w2 += double(p[i].weight) * double(p[i].weight);
        FML::SumOverTasks(&sum_w2);
        return sum_w2 / (N * N);
    }

    template <int NDIM>
    void NeutrinoParticles<NDIM>::density_fourier(FML::GRID::FFTWGrid<NDIM> & density,
                                                  const std::string & density_assignment_method) {
        FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                              part.get_npart(),
                                              part.get_npart_total(),
                                              density,
                                              density_assignment_method);
        density.fftw_r2c();
        // Without the delta-f weights the density is 1 + delta
        if (FML::ThisTask == 0)
            density.set_fourier_from_index(0, 0.0);
    }

} // namespace SIMCODE

#endif
