#ifndef COLASOLVER_NEUTRINOS_HEADER
#define COLASOLVER_NEUTRINOS_HEADER

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
/// y^2 / (e^y + 1), so v_thermal = y (T_nu0 / m_nu) c_code.
//=============================================================================
namespace COLASOLVER {
    namespace NEUTRINOS {

        /// The speed of light in units of H0 B (B in Mpc/h)
        inline double c_code(double boxsize) { return 2997.92458 / boxsize; }

        /// The unperturbed (relativistic) Fermi-Dirac occupation number
        inline double fermi_dirac(double y) { return 1.0 / (std::exp(y) + 1.0); }

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

        /// Draws y from y^2 / (e^y + 1) on [0, ymax] by inverting the tabulated CDF
        class FermiDiracSampler {
          public:
            FermiDiracSampler(int n = 8192, double ymax = 30.0) : y(n), cdf(n) {
                // Trapezoidal integration on a fine grid
                for (int i = 0; i < n; i++)
                    y[i] = ymax * i / double(n - 1);
                cdf[0] = 0.0;
                for (int i = 1; i < n; i++) {
                    const double f0 = y[i - 1] * y[i - 1] * fermi_dirac(y[i - 1]);
                    const double f1 = y[i] * y[i] * fermi_dirac(y[i]);
                    cdf[i] = cdf[i - 1] + 0.5 * (f0 + f1) * (y[i] - y[i - 1]);
                }
                for (auto & c : cdf)
                    c /= cdf.back();
            }
            double operator()(double u) const {
                const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
                if (it == cdf.begin())
                    return 0.0;
                if (it == cdf.end())
                    return y.back();
                const size_t i = it - cdf.begin();
                const double t = (u - cdf[i - 1]) / (cdf[i] - cdf[i - 1]);
                return y[i - 1] + t * (y[i] - y[i - 1]);
            }

          private:
            std::vector<double> y, cdf;
        };

    } // namespace NEUTRINOS
} // namespace COLASOLVER

#endif
