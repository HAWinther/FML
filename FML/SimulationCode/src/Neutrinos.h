#ifndef SIMCODE_NEUTRINOS_HEADER
#define SIMCODE_NEUTRINOS_HEADER

#include <FML/Cosmology/Background/Species.h>
#include <FML/Units/Units.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

//=============================================================================
/// Helpers for massive neutrino particles (delta-f method, Elbers et al. 2021, arXiv:2010.07321)
///
/// Units: the code velocity of a neutrino particle is its comoving momentum q / (m H0 B) (B the boxsize), which is
/// the usual code velocity a^2 dx/dt / (H0 B) in the non-relativistic limit. With u = q / (a m c) = v / (a c_code)
/// (c_code = c / (H0 B)) the equations of motion are
///   dx/dt_drift = v / sqrt(1 + u^2)
///   dv/dt_kick  = -F (1 + 2u^2) / sqrt(1 + u^2)
/// where dt_drift, dt_kick and F are the same as for the CDM particles (the relativistic correction to the
/// geodesic equation in the Newtonian gauge, eq. 2.10 of Elbers et al. 2021).
/// The thermal momentum is q = y k_B T_nu0 / c with y drawn from the relativistic Fermi-Dirac distribution
/// y^2 / (e^y + 1) (BACKGROUND::FermiDiracMomentumSampler), so v_thermal = y c_code / M_today with M_today =
/// m c^2 / (k_B T_nu0) of the background's species (MassiveNeutrinoSpecies::M_today).
//=============================================================================
namespace SIMCODE {
    namespace NEUTRINOS {

        /// The speed of light in units of H0 B (B in Mpc/h)
        inline double c_code(double boxsize) { return FML::UTILS::SI::c_over_H0_mpch / boxsize; }

        /// The massive neutrino species of the background (Background/Species.h)
        using MassiveNeutrinoList = std::vector<FML::COSMOLOGY::BACKGROUND::MassiveNeutrinoSpecies>;

        /// The mean energy per mass <E/m>(a) of the unperturbed massive neutrinos (all species): their energy density
        /// over their mass density, sum_i m_i <E/m>_i / sum_i m_i with <E/m>_i = <sqrt(1 + u^2)> over the
        /// Fermi-Dirac distribution (u = y T_nu0 / (a m_i); -> 1 when non-relativistic). The same energy density as
        /// the background (MassiveNeutrinoSpecies::mean_energy_per_mass)
        inline double mean_energy_factor(const MassiveNeutrinoList & nus, double a) {
            if (nus.empty())
                return 1.0;
            const double x = std::log(a);
            if (std::all_of(nus.begin(), nus.end(), [&](auto & nu) { return nu.mass_eV() == nus.front().mass_eV(); }))
                return nus.front().mean_energy_per_mass(x);
            double num = 0.0, den = 0.0;
            for (auto & nu : nus) {
                num += nu.mass_eV() * nu.mean_energy_per_mass(x);
                den += nu.mass_eV();
            }
            return num / den;
        }

        /// The drift and kick factors for u = q / (a m c)
        inline double drift_factor(double u2) { return 1.0 / std::sqrt(1.0 + u2); }
        inline double kick_factor(double u2) { return (1.0 + 2.0 * u2) / std::sqrt(1.0 + u2); }

        /// A counter-based random generator (splitmix64): the numbers for a particle depend only on (seed, id)
        /// and not on the number of tasks or threads
        inline uint64_t splitmix64(uint64_t x) {
            x += 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        }
        struct ParticleRNG {
            uint64_t state;
            ParticleRNG(uint64_t seed, uint64_t id) : state(splitmix64(splitmix64(seed) ^ splitmix64(id + 1))) {}
            // Uniform in (0,1)
            double uniform() {
                state = splitmix64(state);
                return ((state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
            }
        };

    } // namespace NEUTRINOS
} // namespace SIMCODE

#endif
