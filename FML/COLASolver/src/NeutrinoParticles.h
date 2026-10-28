#ifndef NEUTRINOPARTICLES_HEADER
#define NEUTRINOPARTICLES_HEADER

#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Spline/Spline.h>

#include "InitialConditions.h"
#include "Neutrinos.h"
#include "Particles.h"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace COLASOLVER {

    //=============================================================================
    /// Massive neutrinos as particles with the delta-f method (Elbers et al. 2021, arXiv:2010.07321). Owns the
    /// particles and does what is specific to them: the initial conditions (bulk LPT motion plus Fermi-Dirac
    /// thermal momenta), the delta-f weights (update_weights after every kick) and the (weighted) density field.
    /// The relativistic kick and drift are in KickDrift.h, the units and equations in Neutrinos.h.
    //=============================================================================
    template <int NDIM>
    class NeutrinoParticles {
      public:
        using Particle = ParticleNeutrino<NDIM>;
        using Container = FML::PARTICLE::MPIParticles<Particle>;
        using LinearTransferData = FML::FILEUTILS::LinearTransferData;
        using Spline = FML::INTERPOLATION::SPLINE::Spline;
        using DVector = FML::INTERPOLATION::SPLINE::DVector;

        /// mnu_eV: the mass of one neutrino, Tnu0_eV: the temperature today, boxsize in Mpc/h, deltaf: use the
        /// delta-f weights (otherwise all weights are 1 and the density is 1 + delta)
        void init(double _mnu_eV, double _Tnu0_eV, double _boxsize, bool _deltaf) {
            mnu_eV = _mnu_eV;
            Tnu0_eV = _Tnu0_eV;
            boxsize = _boxsize;
            deltaf = _deltaf;
        }

        Container & particles() { return part; }
        const Container & particles() const { return part; }
        void free() { part.free(); }

        /// The initial conditions. The bulk part is made like the CDM: delta_nu(k) = delta_cb(k) T_nu(k,zini) /
        /// T_cb(k,zini) from the same random field, 1LPT displacements and velocities v = a^2 H f_nu(k) Psi with f_nu
        /// from the transfer functions. On top of this each particle gets a thermal momentum q = y k_B T_nu0 / c in
        /// a random direction, with y drawn from the Fermi-Dirac distribution (the random numbers depend only on
        /// the seed and the particle ID). We store y and the linear delta_nu at the particle (for the weights)
        void generate(const FML::GRID::FFTWGrid<NDIM> & delta_cb_fourier,
                      const LinearTransferData & transferdata,
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

        /// The mean energy per mass <E/m>(a) of the unperturbed neutrinos (see NEUTRINOS::mean_energy_factor)
        double mean_energy_factor(double a) const {
            return COLASOLVER::NEUTRINOS::mean_energy_factor(a, mnu_eV, Tnu0_eV);
        }

        /// While this object lives the weights of the particles are w_i E_i / (m <E/m>(a)) with E_i / m =
        /// sqrt(1 + u_i^2), so the (weighted) density field is the energy-density contrast delta_nu = delta rho_nu /
        /// rho_nu (as in CAMB/CLASS) instead of the mass-density contrast; also its shot noise. The original weights
        /// are restored (exactly) when it goes out of scope. Does nothing if enabled = false
        class ScopedEnergyWeights {
          public:
            ScopedEnergyWeights(NeutrinoParticles & nu, double a, bool enabled) : nu(nu), enabled(enabled) {
                if (not enabled)
                    return;
                namespace NU = COLASOLVER::NEUTRINOS;
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
        double mnu_eV{0.0};
        double Tnu0_eV{0.0};
        double boxsize{1.0};
        bool deltaf{true};
    };

    template <int NDIM>
    void NeutrinoParticles<NDIM>::generate(const FML::GRID::FFTWGrid<NDIM> & delta_cb_fourier,
                                           const LinearTransferData & transferdata,
                                           double zini,
                                           double HoverH0_ini,
                                           int Npart_1D,
                                           double allocation_factor,
                                           int random_seed,
                                           long long int id_offset) {
        namespace NU = COLASOLVER::NEUTRINOS;
        const double aini = 1.0 / (1.0 + zini);
        if (aini < 1.0 / (1.0 + transferdata.get_zmax_splines()) * (1.0 - 1e-10))
            throw std::runtime_error("particle_neutrinos: the transfer functions only go up to z = " +
                                     std::to_string(transferdata.get_zmax_splines()) + " (need zini)");

        // T_nu / T_cb and f_nu at zini
        const double kmin = transferdata.get_kmin_hmpc_splines();
        const double kmax = transferdata.get_kmax_hmpc_splines();
        const int nk = 512;
        DVector logk(nk), ratio(nk), f_nu(nk);
        for (int i = 0; i < nk; i++) {
            logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
            const double k = std::exp(logk[i]);
            ratio[i] = transferdata.get_massive_neutrino_transfer_function(k, aini) /
                       transferdata.get_cdm_baryon_transfer_function(k, aini);
            f_nu[i] = transferdata.get_massive_neutrino_growth_rate(k, aini);
        }
        Spline ratio_spline(logk, ratio, "T_nu/T_cb(k,zini)");
        Spline f_nu_spline(logk, f_nu, "f_nu(k,zini)");
        if (FML::ThisTask == 0) {
            std::cout << "# Neutrino particles at z = " << zini << ": T_nu/T_cb, f_nu =";
            for (double k : {0.01, 0.1, 1.0})
                std::cout << " " << ratio_spline(std::log(k)) << ", " << f_nu_spline(std::log(k)) << " (k = " << k
                          << ")";
            std::cout << "\n";
        }
        auto clamp_logk = [&](double kBox) { return std::log(std::min(std::max(kBox / boxsize, kmin), kmax)); };

        // The bulk part (1LPT) from delta_nu(k, zini)
        const double fac = aini * aini * HoverH0_ini;
        const std::vector<double> velocity_norms{fac * f_nu_spline(std::log(0.1)), 0.0, 0.0, 0.0};
        auto delta_nu = generate_species_from_delta_cb<NDIM, Particle>(
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

        // The thermal momenta
        const double vth_norm = Tnu0_eV / mnu_eV * NU::c_code(boxsize);
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
            for (int idim = 0; idim < NDIM; idim++)
                p[i].vel[idim] += y * vth_norm * (NDIM == 3 ? dir[idim] : (idim == 0 ? 1.0 : 0.0));
        }
        FML::MinOverTasks(&min_delta);
        if (min_delta <= -1.0)
            throw std::runtime_error("particle_neutrinos: delta_nu <= -1 in the initial conditions (start earlier)");
        update_weights();
        if (FML::ThisTask == 0)
            std::cout << "# Neutrino thermal velocity scale (T_nu0 / m_nu) c = " << vth_norm << " (code units), "
                      << vth_norm * 100.0 * boxsize
                      << " km/s today. Min delta_nu(zini) at the particles = " << min_delta << "\n";
    }

    template <int NDIM>
    void NeutrinoParticles<NDIM>::update_weights() {
        namespace NU = COLASOLVER::NEUTRINOS;
        auto * p = part.get_particles_ptr();
        const size_t npart = part.get_npart();
        if (not deltaf) {
            for (size_t i = 0; i < npart; i++)
                p[i].weight = 1.0f;
            return;
        }
        const double y_over_v = 1.0 / NU::c_code(boxsize) * mnu_eV / Tnu0_eV;
        double sum_w = 0.0, sum_w2 = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : sum_w, sum_w2)
#endif
        for (size_t i = 0; i < npart; i++) {
            double v2 = 0.0;
            for (int idim = 0; idim < NDIM; idim++)
                v2 += p[i].vel[idim] * p[i].vel[idim];
            const double y = std::sqrt(v2) * y_over_v;
            const double fi = NU::fermi_dirac(p[i].y_ini) * (1.0 + p[i].delta_ini);
            const double w = 1.0 - NU::fermi_dirac(y) / fi;
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
        if (not deltaf)
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

} // namespace COLASOLVER

#endif
