#ifndef FML_STEPFROMBIGBANG_H
#define FML_STEPFROMBIGBANG_H
// =====================================================================================================================
// StepFromBigBang: initial conditions made by taking LPT-consistent time steps from (almost) the Big Bang.
//
// The particles start on the lattice with the Zel'dovich displacement and velocity at a_start ~ 0 and are moved to
// a_end with a few steps that are exact for the LPT growing mode for any step size:
//
//   bullfrog : drift-kick-drift with a kick chosen so that 2LPT is exact (BullFrog, Rampf, List & Hahn 2024),
//              one force per step, for growth that does not depend on scale.
//   lpt3     : exact to third order including the transverse term: two forces per step, one extra vector per
//              particle (get_sfbb_scratch), GR-like models with scale-independent growth.
//   hybrid   : scale-dependent FastPM steps (each mode follows its own D(k, a): a kick factor per k) to a_switch,
//              then BullFrog steps with the growth of a reference k. For massive neutrinos, f(R), any G_eff(k, a).
//
// The steps use only the force identity on the growing mode (force at the exact LPT configuration = sum of known
// multiples of the LPT fields), so they follow the growing mode of the particle system with the force that is used
// (e.g. near the particle Nyquist frequency), not continuous LPT, and there is no aliasing from products on a grid.
// See README.md and the derivation in Documentation (lpt3step notes).
//
// The caller provides
//   - a GrowthModel (Growth.h): the growth factors, their derivatives and the linear response kappa(a, k);
//   - a force function: compute the force F at the particles' current positions at scale factor a, with an optional
//     filter on the density in Fourier space, and add w_i F to the targets (position, velocity, scratch) with the given
//     weights. F is dv/dlna in code units (v = a^2 dx/dt / (H0 B)): for a linear mode F = kappa(a, k) D1(a, k) Psi1.
// =====================================================================================================================

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/NBody/NBody.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/StepFromBigBang/Growth.h>
#include <FML/Units/Units.h>

#include <array>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace FML::STEPFROMBIGBANG {

    /// Does the particle type have get_sfbb_scratch() (NDIM numbers, used by lpt3)?
    template <class P, class = void>
    struct has_sfbb_scratch : std::false_type {};
    template <class P>
    struct has_sfbb_scratch<P, std::void_t<decltype(std::declval<P &>().get_sfbb_scratch())>> : std::true_type {};

    /// Where the force function adds w F
    struct Target {
        enum Kind { Position, Velocity, Scratch } kind;
        double weight;
    };

    /// Add w_i F to the targets of one particle (for writing force functions)
    template <int NDIM, class P, class FT>
    void add_to_targets(P & p, const FT * F, const std::vector<Target> & targets) {
        for (const auto & t : targets) {
            if (t.weight == 0.0)
                continue;
            if (t.kind == Target::Position) {
                auto * x = p.get_pos();
                for (int d = 0; d < NDIM; d++)
                    x[d] += t.weight * F[d];
            } else if (t.kind == Target::Velocity) {
                auto * v = p.get_vel();
                for (int d = 0; d < NDIM; d++)
                    v[d] += t.weight * F[d];
            } else {
                if constexpr (has_sfbb_scratch<P>::value) {
                    auto * s = p.get_sfbb_scratch();
                    for (int d = 0; d < NDIM; d++)
                        s[d] += t.weight * F[d];
                } else {
                    throw std::runtime_error("[StepFromBigBang] Scratch target without get_sfbb_scratch");
                }
            }
        }
    }

    struct Settings {
        std::string scheme{"bullfrog"};   ///< bullfrog, lpt3 or hybrid
        double a_start{1e-7};             ///< the Zel'dovich start
        double a_end{0.05};               ///< where the IC are wanted
        int nsteps{1};                    ///< BullFrog / lpt3 steps (hybrid: after a_switch; 0 = none)
        double a_switch{1.0 / 151.0};     ///< hybrid: the scale-dependent steps end here
        int nsteps_scaledependent{1};     ///< hybrid: the scale-dependent steps
        double koverH0_ref{0.0};          ///< the k (c k / H0) of the scale-independent steps (0: large scales)
        double boxsize{1.0};              ///< Mpc/h (to convert kBox to k)
        double lpt3_first_s1{-1};         ///< lpt3: override s1/D1 of the first step (negative: the exact solution)
        double lpt3_predictor_frac{0.5};  ///< lpt3: the predictor at D0 + frac (D1 - D0)
        bool verbose{true};
    };

    template <int NDIM, class P>
    class StepFromBigBang {
      public:
        using Particles = FML::PARTICLE::MPIParticles<P>;
        using Filter = std::function<double(double)>;
        using ForceFunction =
            std::function<void(Particles &, double a, const Filter & filter_of_kBox, const std::vector<Target> &)>;

        StepFromBigBang(GrowthModel growth, ForceFunction force, Settings settings)
            : g(std::move(growth)), force(std::move(force)), s(std::move(settings)) {
            if (s.scheme != "bullfrog" and s.scheme != "lpt3" and s.scheme != "hybrid")
                throw std::invalid_argument("[StepFromBigBang] Unknown scheme " + s.scheme);
            if (not(s.a_start > 0.0 and s.a_start < s.a_end))
                throw std::invalid_argument("[StepFromBigBang] Need 0 < a_start < a_end");
            if (s.scheme == "lpt3") {
                if constexpr (not has_sfbb_scratch<P>::value)
                    throw std::invalid_argument("[StepFromBigBang] lpt3 needs get_sfbb_scratch() in the particle");
                if (g.scale_dependent)
                    throw std::invalid_argument("[StepFromBigBang] lpt3 needs scale-independent growth (use hybrid)");
                if (not g.gr_third_order)
                    throw std::invalid_argument("[StepFromBigBang] lpt3 needs the GR third-order force");
                if (s.nsteps < 1)
                    throw std::invalid_argument("[StepFromBigBang] lpt3 needs nsteps >= 1");
            }
            if (s.scheme == "bullfrog" and s.nsteps < 1)
                throw std::invalid_argument("[StepFromBigBang] bullfrog needs nsteps >= 1");
            if (s.scheme == "hybrid" and not(s.a_switch > s.a_start and s.a_switch <= s.a_end))
                throw std::invalid_argument("[StepFromBigBang] hybrid needs a_start < a_switch <= a_end");
            if (s.scheme == "hybrid" and s.nsteps_scaledependent < 1)
                throw std::invalid_argument("[StepFromBigBang] hybrid needs nsteps_scaledependent >= 1");
            if (s.scheme == "hybrid" and s.a_switch == s.a_end)
                s.nsteps = 0;
            if (s.scheme == "hybrid" and s.nsteps == 0 and s.a_switch != s.a_end)
                throw std::invalid_argument("[StepFromBigBang] hybrid with nsteps = 0 needs a_switch = a_end");
            if (s.a_start < g.a_min)
                throw std::invalid_argument("[StepFromBigBang] a_start below the growth model's a_min");
            build_schedule();
        }

        /// koverH0 of a dimensionless kBox
        double koverH0(double kBox) const { return kBox / s.boxsize * FML::UTILS::SI::c_over_H0_mpch; }

        /// Number of force evaluations run() will do
        int number_of_forces() const {
            if (s.scheme == "bullfrog")
                return s.nsteps;
            if (s.scheme == "lpt3")
                return 3 + 2 * (s.nsteps - 1) - (s.nsteps > 1 ? 1 : 0);
            return s.nsteps_scaledependent + s.nsteps;
        }

        //==============================================================================================================
        /// Make the start particles: the lattice with the Zel'dovich displacement at a_start of the field
        /// delta_fourier (the linear density at a_ref, which may depend on k through D1(a_ref, k)), and the velocity of
        /// the scheme (the growing-mode velocity, or for hybrid the effective velocity of the first scale-dependent
        /// drift). lattice_shift (box units) is added to every position (e.g. half a force cell).
        //==============================================================================================================
        void make_start_particles(Particles & part,
                                  int Npart_1D,
                                  double buffer_factor,
                                  const FML::GRID::FFTWGrid<NDIM> & delta_fourier,
                                  double a_ref,
                                  double lattice_shift = 0.0) const {
            const double a0 = s.a_start;
            // A copy with the extra slices NBodyInitialConditions needs (CIC)
            const auto nextra = FML::INTERPOLATION::get_extra_slices("CIC");
            FML::GRID::FFTWGrid<NDIM> delta(delta_fourier.get_nmesh(), nextra.first, nextra.second);
            delta.set_grid_status_real(false);
            FML::NUFFT::copy_fourier_modes<NDIM>(delta_fourier, delta, [](const std::array<int, NDIM> &) { return 1.0; });
            const double kref = s.koverH0_ref;
            FML::GRID::for_each_fourier_mode<NDIM>(delta, [&](size_t i, double kBox) {
                const double k = g.scale_dependent ? koverH0(kBox) : kref;
                delta.set_fourier_from_index(
                    i, delta.get_fourier_from_index(i) * FML::GRID::FloatType(g.D1(a0, k) / g.D1(a_ref, k)));
            });
            // velocity = norm(k) x displacement at a0
            std::function<double(double)> vnorm_of_kBox;
            if (s.scheme == "hybrid") {
                const double a1 = sd_a[1], d0 = sd_drift[0];
                vnorm_of_kBox = [this, a0, a1, d0](double kBox) {
                    const double k = koverH0(kBox);
                    return (g.D1(a1, k) - g.D1(a0, k)) / d0 / g.D1(a0, k);
                };
            } else {
                vnorm_of_kBox = [this, a0, kref](double) {
                    return g.vnorm(a0) * g.dD1(a0, kref) / g.D1(a0, kref);
                };
            }
            std::vector<FML::GRID::FFTWGrid<NDIM>> phi;
            FML::NBODY::NBodyInitialConditions<NDIM, P>(part,
                                                         Npart_1D,
                                                         buffer_factor,
                                                         delta,
                                                         phi,
                                                         1,
                                                         s.boxsize,
                                                         1.0 / a0 - 1.0,
                                                         std::vector<double>{vnorm_of_kBox(1e-10)},
                                                         vnorm_of_kBox);
            if (lattice_shift != 0.0) {
                for (auto & p : part) {
                    auto * x = p.get_pos();
                    for (int d = 0; d < NDIM; d++)
                        x[d] += lattice_shift;
                    // The Lagrangian position moves with the lattice (sheet interpolation uses x - q)
                    if constexpr (FML::PARTICLE::has_get_q<P>()) {
                        auto * q = FML::PARTICLE::GetLagrangianPos(p);
                        for (int d = 0; d < NDIM; d++) {
                            q[d] += lattice_shift;
                            q[d] -= std::floor(q[d]);
                        }
                    }
                }
                wrap_and_communicate(part);
            }
            if constexpr (has_sfbb_scratch<P>::value)
                for (auto & p : part)
                    for (int d = 0; d < NDIM; d++)
                        p.get_sfbb_scratch()[d] = 0.0;
        }

        //==============================================================================================================
        /// Move the particles from a_start to a_end
        //==============================================================================================================
        void run(Particles & part) {
            if (s.verbose and FML::ThisTask == 0) {
                std::cout << "# StepFromBigBang: " << s.scheme << " from z = " << 1.0 / s.a_start - 1.0 << " to z = "
                          << 1.0 / s.a_end - 1.0 << ", " << number_of_forces() << " force evaluations\n";
            }
            if (s.scheme == "bullfrog") {
                for (size_t i = 0; i + 1 < bf_a.size(); i++)
                    bullfrog_step(part, bf_a[i], bf_a[i + 1], i == 0);
            } else if (s.scheme == "lpt3") {
                lpt3_first_step(part, bf_a[0], bf_a[1]);
                for (size_t i = 1; i + 1 < bf_a.size(); i++)
                    lpt3_step(part, bf_a[i], bf_a[i + 1], i + 2 < bf_a.size());
            } else {
                scaledependent_steps(part);
                for (size_t i = 0; i + 1 < bf_a.size(); i++)
                    bullfrog_step(part, bf_a[i], bf_a[i + 1], false);
            }
        }

        const GrowthModel & growth() const { return g; }
        const Settings & settings() const { return s; }

      private:
        GrowthModel g;
        ForceFunction force;
        Settings s;
        std::vector<double> bf_a;       // bullfrog / lpt3 step times
        std::vector<double> sd_a;       // hybrid: the scale-dependent kick times a_0 .. a_N
        std::vector<double> sd_drift;   // hybrid: the drift factors int dlna / (a^2 E)

        //--------------------------------------------------------------------------------------------------------------
        // Step times: uniform in D1 (at koverH0_ref)
        //--------------------------------------------------------------------------------------------------------------
        void build_schedule() {
            const double k = s.koverH0_ref;
            auto uniform_in_D = [&](double a0, double a1, int n) {
                std::vector<double> a(n + 1);
                const double D0 = g.D1(a0, k), Dn = g.D1(a1, k);
                a[0] = a0;
                a[n] = a1;
                for (int i = 1; i < n; i++)
                    a[i] = g.a_of_D1(D0 + (Dn - D0) * i / double(n), k);
                return a;
            };
            if (s.scheme == "hybrid") {
                sd_a = uniform_in_D(s.a_start, s.a_switch, s.nsteps_scaledependent);
                sd_drift.resize(s.nsteps_scaledependent);
                for (int i = 0; i < s.nsteps_scaledependent; i++)
                    sd_drift[i] = drift_integral(sd_a[i], sd_a[i + 1]);
                bf_a = s.nsteps > 0 ? uniform_in_D(s.a_switch, s.a_end, s.nsteps) : std::vector<double>{};
            } else {
                bf_a = uniform_in_D(s.a_start, s.a_end, s.nsteps);
            }
        }

        /// int_{a0}^{a1} dlna / (a^2 E) (Simpson in ln a)
        double drift_integral(double a0, double a1) const {
            const int n = 2000;
            const double x0 = std::log(a0), h = (std::log(a1) - x0) / n;
            double sum = 0;
            for (int i = 0; i <= n; i++) {
                const double a = std::exp(x0 + i * h);
                const double w = (i == 0 or i == n) ? 1 : (i % 2 ? 4 : 2);
                sum += w / g.vnorm(a);
            }
            return sum * h / 3;
        }

        void wrap_and_communicate(Particles & part) const {
            for (auto & p : part) {
                auto * x = p.get_pos();
                for (int d = 0; d < NDIM; d++) {
                    x[d] -= std::floor(x[d]);
                    if (x[d] >= 1.0)
                        x[d] -= 1.0;
                }
            }
            part.communicate_particles();
        }

        static std::array<std::array<double, 2>, 2> inv2(double a, double b, double c, double d) {
            const double det = a * d - b * c;
            return {{{d / det, -b / det}, {-c / det, a / det}}};
        }
        static std::array<std::array<double, 3>, 3> inv3(const std::array<std::array<double, 3>, 3> & m) {
            std::array<std::array<double, 3>, 3> r;
            const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
            r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
            r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
            r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
            r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
            r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
            r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
            r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
            r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
            return r;
        }

        // The growing-mode rows over (Psi1, Psi2, Psi3a, Psi3b, Psi3c) at a (k = koverH0_ref)
        using Row = std::array<double, 5>;
        Row row_d(double a) const {
            const double k = s.koverH0_ref;
            return {g.D1(a, k), g.r2(a, k), g.r3a(a, k), g.r3b(a, k), g.r3c(a, k)};
        }
        Row row_u(double a) const {
            const double k = s.koverH0_ref;
            return {g.dD1(a, k), g.dr2(a, k), g.dr3a(a, k), g.dr3b(a, k), g.dr3c(a, k)};
        }
        /// The force (over kappa) at a configuration with coefficients K on the fields
        static Row row_force(const Row & K, double N = 1.0) {
            return {K[0],
                    K[1] + 7.0 / 3.0 * N * K[0] * K[0],
                    K[2] + 6.0 * K[0] * K[0] * K[0],
                    K[3] + 9.0 / 5.0 * K[0] * K[1] + 21.0 / 5.0 * K[0] * K[0] * K[0],
                    7.0 * K[0] * K[0] * K[0]};
        }

        void print_step(const std::string & name, double a0, double a1, const std::string & extra = "") const {
            if (s.verbose and FML::ThisTask == 0)
                std::cout << "#   " << name << " z = " << std::setprecision(6) << 1.0 / a0 - 1.0 << " -> "
                          << 1.0 / a1 - 1.0 << extra << "\n";
        }

        //--------------------------------------------------------------------------------------------------------------
        // BullFrog: exact for x = q + D1 Psi1 + r2 Psi2 (scale-independent growth at koverH0_ref). Drift to the
        // midpoint in D1, force there, then x1 and u1 (u = v / vnorm) as combinations of u0 and the force. The first
        // step starts from the Zel'dovich state (no second order): in the radiation era D1' is small and the
        // growing mode's r2' Psi2 is not negligible next to D1' Psi1, so the state must be described as it is.
        //--------------------------------------------------------------------------------------------------------------
        void bullfrog_step(Particles & part, double a0, double a1, bool from_zeldovich) {
            const double k = s.koverH0_ref;
            const double D10 = g.D1(a0, k), dD10 = g.dD1(a0, k);
            const double r20 = from_zeldovich ? 0.0 : g.r2(a0, k), dr20 = from_zeldovich ? 0.0 : g.dr2(a0, k);
            const double Dt = 0.5 * (D10 + g.D1(a1, k)), at = g.a_of_D1(Dt, k);
            const double al = (Dt - D10) / dD10;
            const double K1 = Dt, K2 = r20 + al * dr20, Nt = g.N(at, k);
            const auto Mi = inv2(dD10, dr20, K1, K2 + 7.0 / 3.0 * Nt * K1 * K1);
            const double cx0 = g.D1(a1, k) - D10, cx1 = g.r2(a1, k) - r20, cu0 = g.dD1(a1, k), cu1 = g.dr2(a1, k);
            const double w_u = cx0 * Mi[0][0] + cx1 * Mi[1][0], w_g = cx0 * Mi[0][1] + cx1 * Mi[1][1];
            const double m_u = cu0 * Mi[0][0] + cu1 * Mi[1][0], m_g = cu0 * Mi[0][1] + cu1 * Mi[1][1];
            const double vn0 = g.vnorm(a0), vn1 = g.vnorm(a1), kt = g.kappa(at, k);
            print_step("BullFrog", a0, a1);
            // Drift to the midpoint; the velocity to m_u u0
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * v = p.get_vel();
                for (int d = 0; d < NDIM; d++) {
                    x[d] += al * v[d] / vn0;
                    v[d] *= m_u * vn1 / vn0;
                }
            }
            wrap_and_communicate(part);
            const double c = (w_u - al) / m_u;
            force(part, at, nullptr, {{Target::Velocity, vn1 * m_g / kt}, {Target::Position, (w_g - c * m_g) / kt}});
            // The rest of the drift with the new velocity
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * v = p.get_vel();
                for (int d = 0; d < NDIM; d++)
                    x[d] += c * v[d] / vn1;
            }
            wrap_and_communicate(part);
        }

        //--------------------------------------------------------------------------------------------------------------
        // lpt3, first step from the Zel'dovich state at a0 ~ 0: forces at the Zel'dovich configuration with amplitude
        // c1 and at the corrected 2LPT configuration with c2 (and rho = r2 there); the positions are exact to third
        // order (incl. the transverse term) when (lambda1, lambda2, c1, c2) solve the four conditions on Psi2, Psi3a,
        // Psi3b, Psi3c (EdS: c1 = 7/54 D1, c2 = 7/18 D1). The force at the end gives the velocities (least squares over
        // the four third-order coefficients; exact in EdS) and is kept for the next step.
        //--------------------------------------------------------------------------------------------------------------
        void lpt3_first_step(Particles & part, double a0, double a1) {
            const double k = s.koverH0_ref;
            const double D10 = g.D1(a0, k), dD10 = g.dD1(a0, k), D1e = g.D1(a1, k);
            const Row t = row_d(a1);
            auto rho_of = [&](double c2) { return g.r2(g.a_of_D1(c2, k), k); };
            auto vecs = [&](double c1, double c2, double rho) {
                std::array<std::array<double, 4>, 2> v;
                v[0] = {7.0 / 3.0 * c1 * c1, 6 * c1 * c1 * c1, 21.0 / 5.0 * c1 * c1 * c1, 7 * c1 * c1 * c1};
                v[1] = {rho + 7.0 / 3.0 * c2 * c2,
                        6 * c2 * c2 * c2 + 18.0 / 7.0 * rho * c1,
                        9.0 / 5.0 * c2 * rho + 21.0 / 5.0 * c2 * c2 * c2 + 9.0 / 5.0 * rho * c1,
                        7 * c2 * c2 * c2};
                return v;
            };
            // Unknowns z = (lambda1, lambda2, c1, c2): residuals of the four conditions
            auto resid = [&](const std::array<double, 4> & z) {
                const double rho = rho_of(z[3]);
                auto v = vecs(z[2], z[3], rho);
                std::array<double, 4> r;
                for (int j = 0; j < 4; j++)
                    r[j] = (z[0] * v[0][j] + z[1] * v[1][j] - t[j + 1]) / t[j + 1];
                return r;
            };
            double c1, c2, l1, l2;
            if (s.lpt3_first_s1 > 0.0) {
                // Fixed s1: c2 = 7/18 D1 scaled, lambdas from the Psi2 and Psi3b conditions (no Psi3c exactness)
                c1 = s.lpt3_first_s1 * D1e;
                c2 = 7.0 / 18.0 * D1e;
                auto v = vecs(c1, c2, rho_of(c2));
                const auto Mi = inv2(v[0][0], v[1][0], v[0][2], v[1][2]);
                l1 = Mi[0][0] * t[1] + Mi[0][1] * t[3];
                l2 = Mi[1][0] * t[1] + Mi[1][1] * t[3];
            } else {
                // Newton from the EdS solution
                std::array<double, 4> z{-26244.0 / 2401.0, 972.0 / 343.0, 7.0 / 54.0 * D1e, 7.0 / 18.0 * D1e};
                for (int it = 0; it < 50; it++) {
                    auto r = resid(z);
                    double norm = 0;
                    for (auto x : r)
                        norm += x * x;
                    if (std::sqrt(norm) < 1e-13)
                        break;
                    std::array<std::array<double, 4>, 4> J;
                    for (int j = 0; j < 4; j++) {
                        auto zp = z;
                        const double h = 1e-7 * std::abs(z[j]);
                        zp[j] += h;
                        auto rp = resid(zp);
                        for (int i = 0; i < 4; i++)
                            J[i][j] = (rp[i] - r[i]) / h;
                    }
                    // Solve J dz = -r (Gaussian elimination with pivoting)
                    std::array<double, 4> b{-r[0], -r[1], -r[2], -r[3]};
                    for (int c = 0; c < 4; c++) {
                        int piv = c;
                        for (int i = c + 1; i < 4; i++)
                            if (std::abs(J[i][c]) > std::abs(J[piv][c]))
                                piv = i;
                        std::swap(J[c], J[piv]);
                        std::swap(b[c], b[piv]);
                        for (int i = c + 1; i < 4; i++) {
                            const double f = J[i][c] / J[c][c];
                            for (int j = c; j < 4; j++)
                                J[i][j] -= f * J[c][j];
                            b[i] -= f * b[c];
                        }
                    }
                    std::array<double, 4> dz;
                    for (int i = 3; i >= 0; i--) {
                        double sum = b[i];
                        for (int j = i + 1; j < 4; j++)
                            sum -= J[i][j] * dz[j];
                        dz[i] = sum / J[i][i];
                    }
                    for (int j = 0; j < 4; j++)
                        z[j] += dz[j];
                }
                auto r = resid(z);
                for (auto x : r)
                    if (std::abs(x) > 1e-8)
                        throw std::runtime_error("[StepFromBigBang] lpt3: the first-step conditions did not converge");
                l1 = z[0];
                l2 = z[1];
                c1 = z[2];
                c2 = z[3];
            }
            const double rho = rho_of(c2);
            const double a_c1 = g.a_of_D1(c1, k), a_c2 = g.a_of_D1(c2, k);
            const double k1 = g.kappa(a_c1, k), k2 = g.kappa(a_c2, k), ke = g.kappa(a1, k);

            // Velocity weights mu on (H1, H2, E = g_end - D1 Psi1): least squares over (Psi2, Psi3a, Psi3b, Psi3c)
            const Row de = row_d(a1), ue = row_u(a1);
            const Row fe = row_force(de);
            auto v = vecs(c1, c2, rho);
            std::array<std::array<double, 4>, 3> V{v[0], v[1], {fe[1], fe[2], fe[3], fe[4]}};
            std::array<double, 4> tv{ue[1], ue[2], ue[3], ue[4]}, sc{de[1], de[2], de[3], de[4]};
            std::array<std::array<double, 3>, 3> NM{};
            std::array<double, 3> rhs{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++)
                    for (int m = 0; m < 4; m++)
                        NM[i][j] += V[i][m] * V[j][m] / (sc[m] * sc[m]);
                for (int m = 0; m < 4; m++)
                    rhs[i] += V[i][m] * tv[m] / (sc[m] * sc[m]);
            }
            const auto NMi = inv3(NM);
            std::array<double, 3> mu{};
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    mu[i] += NMi[i][j] * rhs[j];
            double lsq = 0;
            for (int m = 0; m < 4; m++) {
                double sum = 0;
                for (int i = 0; i < 3; i++)
                    sum += mu[i] * V[i][m];
                lsq = std::max(lsq, std::abs(sum - tv[m]) / std::abs(tv[m]));
            }
            if (s.verbose and FML::ThisTask == 0)
                std::cout << "#   lpt3 first step: s1 = " << c1 / D1e << " D1, s2 = " << c2 / D1e
                          << " D1, lambda = (" << l1 << ", " << l2
                          << "), velocity least-squares residual " << lsq << "\n";
            print_step("lpt3 (first)", a0, a1);

            const double vn0 = g.vnorm(a0), vn1 = g.vnorm(a1);
            // a) the force at the Zel'dovich configuration c1 Psi1 into the scratch
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * vv = p.get_vel();
                auto * sc_ = p.get_sfbb_scratch();
                for (int d = 0; d < NDIM; d++) {
                    x[d] += (c1 - D10) / dD10 * vv[d] / vn0;
                    sc_[d] = 0.0;
                }
            }
            wrap_and_communicate(part);
            force(part, a_c1, nullptr, {{Target::Scratch, 1.0 / k1}});
            // b) H1 = g1 - c1 Psi1; move to the corrected 2LPT configuration; the rest of the position (W) into the
            //    scratch and the part of the velocity without the forces into v
            const double p2 = rho / (7.0 / 3.0 * c1 * c1);
            const double Wu = (D1e - c2 - l2 * c2) / dD10, WH = l1 - p2;
            const double Vu = (g.dD1(a1, k) - mu[1] * c2 - mu[2] * D1e) / dD10;
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * vv = p.get_vel();
                auto * sc_ = p.get_sfbb_scratch();
                for (int d = 0; d < NDIM; d++) {
                    const double u0 = vv[d] / vn0;
                    const double H1 = sc_[d] - c1 / dD10 * u0;
                    x[d] += (c2 - c1) / dD10 * u0 + p2 * H1;
                    sc_[d] = Wu * u0 + WH * H1;
                    vv[d] = vn1 * (Vu * u0 + mu[0] * H1);
                }
            }
            wrap_and_communicate(part);
            // c) the force at the corrected 2LPT configuration
            force(part, a_c2, nullptr, {{Target::Position, l2 / k2}, {Target::Velocity, vn1 * mu[1] / k2}});
            // d) the rest of the position
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * sc_ = p.get_sfbb_scratch();
                for (int d = 0; d < NDIM; d++) {
                    x[d] += sc_[d];
                    sc_[d] = 0.0;
                }
            }
            wrap_and_communicate(part);
            // e) the force at the end: the rest of the velocity, and kept for the next step
            force(part, a1, nullptr, {{Target::Velocity, vn1 * mu[2] / ke}, {Target::Scratch, 1.0 / ke}});
        }

        //--------------------------------------------------------------------------------------------------------------
        // lpt3, later steps: from (x0, u0, C0 = g(x0)): predictor at the exact 2LPT configuration at D1 = Dt (from u0
        // and C0, no Lagrangian coordinates), force there, and x1, u1 as combinations of u0, C0 and the force. Exact
        // for Psi1, Psi2 and the longitudinal third order (combined as 1/4 Psi3a + 3/4 Psi3b; exact in EdS); the
        // transverse part is not resolved (it needs q). Then the force at the end for the next step.
        //--------------------------------------------------------------------------------------------------------------
        void lpt3_step(Particles & part, double a0, double a1, bool need_end_force) {
            const double k = s.koverH0_ref;
            const Row d0 = row_d(a0), u0r = row_u(a0), C0r = row_force(d0);
            const double D10 = d0[0], Dt = D10 + s.lpt3_predictor_frac * (g.D1(a1, k) - D10);
            const double at = g.a_of_D1(Dt, k);
            const auto Pi = inv2(u0r[0], C0r[0], u0r[1], C0r[1]);
            const double r0 = Dt - D10, r1 = g.r2(at, k) - d0[1];
            const double al = Pi[0][0] * r0 + Pi[0][1] * r1, be = Pi[1][0] * r0 + Pi[1][1] * r1;
            Row K;
            for (int j = 0; j < 5; j++)
                K[j] = d0[j] + al * u0r[j] + be * C0r[j];
            const Row gr = row_force(K);
            const std::array<double, 3> om{0.25, 0.75, 0.0};
            auto reduce = [&](const Row & r) {
                return std::array<double, 3>{r[0], r[1], om[0] * r[2] + om[1] * r[3] + om[2] * r[4]};
            };
            const auto Ru = reduce(u0r), RC = reduce(C0r), Rg = reduce(gr);
            std::array<std::array<double, 3>, 3> M{{{Ru[0], RC[0], Rg[0]}, {Ru[1], RC[1], Rg[1]}, {Ru[2], RC[2], Rg[2]}}};
            const auto Mi = inv3(M);
            const Row d1 = row_d(a1), u1r = row_u(a1);
            Row dd;
            for (int j = 0; j < 5; j++)
                dd[j] = d1[j] - d0[j];
            const auto tx = reduce(dd), tu = reduce(u1r);
            std::array<double, 3> wx{}, wu{};
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                    wx[i] += Mi[i][j] * tx[j];
                    wu[i] += Mi[i][j] * tu[j];
                }
            const double vn0 = g.vnorm(a0), vn1 = g.vnorm(a1), kt = g.kappa(at, k), k1 = g.kappa(a1, k);
            print_step("lpt3", a0, a1);
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * v = p.get_vel();
                auto * sc_ = p.get_sfbb_scratch();
                for (int d = 0; d < NDIM; d++) {
                    const double u0 = v[d] / vn0, C0 = sc_[d];
                    x[d] += al * u0 + be * C0;
                    sc_[d] = (wx[0] - al) * u0 + (wx[1] - be) * C0;
                    v[d] = vn1 * (wu[0] * u0 + wu[1] * C0);
                }
            }
            wrap_and_communicate(part);
            force(part, at, nullptr, {{Target::Position, wx[2] / kt}, {Target::Velocity, vn1 * wu[2] / kt}});
            for (auto & p : part) {
                auto * x = p.get_pos();
                auto * sc_ = p.get_sfbb_scratch();
                for (int d = 0; d < NDIM; d++) {
                    x[d] += sc_[d];
                    sc_[d] = 0.0;
                }
            }
            wrap_and_communicate(part);
            if (need_end_force)
                force(part, a1, nullptr, {{Target::Scratch, 1.0 / k1}});
        }

        //--------------------------------------------------------------------------------------------------------------
        // hybrid: scale-dependent FastPM steps a_0 = a_start .. a_N = a_switch. Between kicks the velocity of mode k is
        // the effective V_n(k) = (D(k, a_n+1) - D(k, a_n)) / d_n (d_n the drift factor), so the common drift gives
        // each mode its own growth; the kick at a_n takes V_n-1 to V_n (to the growing-mode velocity at a_N) with a
        // factor per k on the force: K_n(k) = (V_n - V_n-1) / (kappa(a_n, k) D(k, a_n)). The start velocity is V_0
        // (make_start_particles), so there is no force at the lattice.
        //--------------------------------------------------------------------------------------------------------------
        void scaledependent_steps(Particles & part) {
            const int N = s.nsteps_scaledependent;
            auto V = [&](int n, double k) {
                if (n == N)
                    return g.vnorm(sd_a[N]) * g.dD1(sd_a[N], k);
                return (g.D1(sd_a[n + 1], k) - g.D1(sd_a[n], k)) / sd_drift[n];
            };
            for (int n = 1; n <= N; n++) {
                const double an = sd_a[n];
                print_step("scale-dependent FastPM", sd_a[n - 1], an);
                for (auto & p : part) {
                    auto * x = p.get_pos();
                    auto * v = p.get_vel();
                    for (int d = 0; d < NDIM; d++)
                        x[d] += sd_drift[n - 1] * v[d];
                }
                wrap_and_communicate(part);
                auto Kn = [&, n, an](double k) { return (V(n, k) - V(n - 1, k)) / (g.kappa(an, k) * g.D1(an, k)); };
                const double kref = s.koverH0_ref > 0 ? s.koverH0_ref : koverH0(1e-6);
                const double Kref = Kn(kref);
                Filter filter = [this, Kn, Kref](double kBox) { return Kn(koverH0(std::max(kBox, 1e-6))) / Kref; };
                force(part, an, filter, {{Target::Velocity, Kref}});
            }
        }
    };

} // namespace FML::STEPFROMBIGBANG
#endif
