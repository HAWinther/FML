#ifndef SIMULATION_SCALEDEPENDENTKICK_HEADER
#define SIMULATION_SCALEDEPENDENTKICK_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/NBody/TimeStepping.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"
#include "GravityModel.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace SIMCODE {

    /// FastPM with a kick factor per k (timestep_scaledependent_kick, experimental): Zel'dovich-consistent at every k
    /// for any model whose linear growth D(k, a) depends on scale (G_eff(k, a), massive neutrinos, ...).
    ///
    /// FastPM follows a linear mode x = q + D Psi exactly for any step size when the growth does not depend on k. If it
    /// does, the drift x += d v cannot be exact: d is one number per step and each mode needs its own D(k, a_n+1) -
    /// D(k, a_n). The kick can depend on k since the force is computed in Fourier space. With the velocity between two
    /// kicks the effective one
    ///
    ///   V_n+1/2(k) = (D(k, a_n+1) - D(k, a_n)) / d_n+1/2
    ///
    /// the common drift gives each mode its own growth, and the kick v = s v + kappa F (s: velocity_scale, kappa:
    /// delta_time_kick) needs the factor
    ///
    ///   R(k) = (V_after(k) - s V_before(k)) / (kappa G_f(k, a))
    ///
    /// on the force, where G_f(k, a) Psi is the force on the linear mode. V is the physical G_p(k, a) = a^3 E dD/da at
    /// the start and after a synchronizing step (no drift). G_f is the right hand side of the growth equation of the
    /// code (GravityModel::compute_growth_factors, LPTGrowthFactors):
    ///
    ///   D'' + (2 + dlogE/dloga) D' = 1.5 Omega_m(a) G_eff/G(a, k) S(a, k) D   =>   G_f(k, a) = 1.5 a Omega_m(a) E^2 a^3
    ///   G_eff/G(a, k) S(a, k) D(k, a)
    ///
    /// with S = source_factor_1LPT (massive neutrinos, interacting dark energy), so R only assumes that the force of
    /// the model on a linear mode is the one of its growth equation. R(k) multiplies the density of the force of the
    /// step (all of it: the force is linear in the density). Without scale-dependent growth R = 1 and nothing is done.
    ///
    /// Not with velocity friction (the friction term of the growth equation is not in the kick factor), the N-body
    /// gauge source, COLA, CDM and baryons or neutrinos as separate particles (one factor for one growth), or a
    /// short-range force (it is not multiplied by R).
    template <int NDIM>
    class ScaleDependentKick {
      public:
        using Step = FML::NBODY::TimeStepping::Step;

        ScaleDependentKick(std::shared_ptr<Cosmology> cosmo, std::shared_ptr<GravityModel<NDIM>> grav)
            : cosmo(cosmo), grav(grav) {
            if (grav->has_velocity_friction())
                throw std::runtime_error("timestep_scaledependent_kick: not with velocity friction");
        }

        /// Does it change anything (scale-dependent growth)?
        bool active() const { return grav->is_growth_scaledependent(); }

        /// Multiply the density (Fourier space, k in units of 1/boxsize) of the force of step s by R(k)
        void apply(FFTWGrid<NDIM> & density_fourier, const Step & s, double boxsize) const {
            if (not active())
                return;
            // R(k) tabulated in log k (h/Mpc) from half the fundamental mode to the corner of the grid
            const int nk = 400;
            const double kmin = M_PI / boxsize;
            const double kmax = 2.0 * M_PI / boxsize * density_fourier.get_nmesh() * std::sqrt(double(NDIM));
            std::vector<double> logk(nk), Rk(nk);
            for (int i = 0; i < nk; i++) {
                logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
                Rk[i] = R(std::exp(logk[i]), s);
            }
            FML::INTERPOLATION::SPLINE::Spline Rspline(logk, Rk, "scale-dependent kick factor");
            if (FML::ThisTask == 0)
                std::cout << "# Scale-dependent kick at a = " << s.apos << ": R(k = " << kmin << ", " << std::sqrt(kmin * kmax)
                          << ", " << kmax << " h/Mpc) = " << Rk.front() << " " << Rspline(0.5 * (logk.front() + logk.back()))
                          << " " << Rk.back() << "\n";
            FML::GRID::for_each_fourier_mode<NDIM>(density_fourier, [&](size_t fourier_index, double kmag) {
                if (kmag == 0.0)
                    return;
                const double k = std::min(std::max(kmag / boxsize, kmin), kmax);
                density_fourier.set_fourier_from_index(fourier_index,
                                                       density_fourier.get_fourier_from_index(fourier_index) *
                                                           FML::GRID::FloatType(Rspline(std::log(k))));
            });
        }

        /// After the drift of step s: remember it for the effective velocity before the next kick
        void after_step(const Step & s) {
            velocity_physical = is_synchronizing(s);
            prev_apos = s.apos;
            prev_apos_new = s.apos_new;
            prev_drift = s.delta_time_drift;
        }

        /// The factor for the kick of step s at k (h/Mpc)
        double R(double k, const Step & s) const {
            const double V_before = velocity_physical ? Gp(s.avel, k) : (D(prev_apos_new, k) - D(prev_apos, k)) / prev_drift;
            const double V_after =
                is_synchronizing(s) ? Gp(s.avel_new, k) : (D(s.apos_new, k) - D(s.apos, k)) / s.delta_time_drift;
            return (V_after - s.velocity_scale * V_before) / (s.delta_time_kick * Gf(s.apos, k));
        }

        /// The linear growth, its momentum a^3 E dD/da and its force (the right hand side of the growth equation)
        double D(double a, double k) const { return grav->get_D_1LPT(a, k / grav->H0_hmpc); }
        double Gp(double a, double k) const {
            // dlogD/dloga from D (get_f_1LPT gave NaN at some a with scale-dependent growth)
            const double h = 1e-4;
            const double f = (std::log(D(a * std::exp(h), k)) - std::log(D(a * std::exp(-h), k))) / (2.0 * h);
            return a * a * cosmo->HoverH0_of_a(a) * f * D(a, k);
        }
        double Gf(double a, double k) const {
            const double koverH0 = k / grav->H0_hmpc;
            return 1.5 * a * cosmo->Omega_m_a3(a) * grav->GeffOverG(a, koverH0) * grav->source_factor_1LPT(a, koverH0) *
                   D(a, k);
        }

      private:
        std::shared_ptr<Cosmology> cosmo;
        std::shared_ptr<GravityModel<NDIM>> grav;

        // The previous step: the velocities are physical at the start and after a synchronizing step
        bool velocity_physical{true};
        double prev_apos{0.0}, prev_apos_new{0.0}, prev_drift{0.0};

        // The last step of a KDK sequence only kicks (apos_new can differ from apos by round-off)
        static bool is_synchronizing(const Step & s) { return std::abs(s.apos_new / s.apos - 1.0) < 1e-10; }
    };

} // namespace SIMCODE
#endif
