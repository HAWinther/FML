#ifndef FML_PERTURBATION_SPTKERNELS_H
#define FML_PERTURBATION_SPTKERNELS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

//=====================================================================================================================
///
/// Standard perturbation theory kernels as functions of wave vectors (EdS time dependence), for the direct loop
/// integrals of TNS.h and EFTofLSS.h:
///
///  - F_n, G_n (n <= 3), the density and velocity-divergence kernels, from the recursion of Goroff et al. (1986)
///    (Bernardeau et al. 2002, eq. 43-44), symmetrized over the arguments.
///  - The redshift-space kernels Z_n (n <= 3) of a biased tracer with the Eulerian bias expansion
///    delta_g = b1 delta + b2/2 delta^2 + bG2 G2 + bGamma3 Gamma3 (the basis of CLASS-PT, Ivanov et al. 2020,
///    arXiv:1909.05277; Chudaykin et al. 2020, arXiv:2004.10607), as five components multiplying
///    {1, b1, b2, bG2, bGamma3}. They are built from the redshift-space mapping
///    delta_s(k) = sum_m (1/m!) FT[(-i k_z u_z)^m (1 + delta_g)], expanded order by order, so the third order kernel
///    is not typed in. Z2 is also given in closed form (Scoccimarro, Couchman & Frieman 1999) for speed.
///
/// The line of sight is a unit vector los (Z_n depend on it through the z-components along it).
/// G2(Phi) is the Galileon operator (d_i d_j Phi)^2 - (d^2 Phi)^2 with d^2 Phi = delta: its second order kernel is
/// sigma^2(k1, k2) = (k1.k2)^2/(k1 k2)^2 - 1. Gamma3 = G2(Phi_g) - G2(Phi_v) (d^2 Phi_v = theta).
///
//=====================================================================================================================

namespace FML::PERTURBATION::SPT {

    struct Vec3 {
        double x{0}, y{0}, z{0};
        Vec3 operator+(const Vec3 & o) const { return {x + o.x, y + o.y, z + o.z}; }
        Vec3 operator-(const Vec3 & o) const { return {x - o.x, y - o.y, z - o.z}; }
        Vec3 operator-() const { return {-x, -y, -z}; }
        Vec3 operator*(double a) const { return {a * x, a * y, a * z}; }
        double dot(const Vec3 & o) const { return x * o.x + y * o.y + z * o.z; }
        double norm2() const { return x * x + y * y + z * z; }
        double norm() const { return std::sqrt(norm2()); }
    };

    namespace detail {
        inline Vec3 sum(const Vec3 * q, int n) {
            Vec3 s;
            for (int i = 0; i < n; i++)
                s = s + q[i];
            return s;
        }
        /// |v|^2 small compared with the momenta it is made of: a total momentum that vanishes (q + (-q)); the
        /// kernels multiplying such a factor vanish (F_n, G_n of momenta summing to zero), so the term is skipped
        inline bool vanishes(const Vec3 & v, double scale2) { return v.norm2() <= 1e-28 * scale2; }

        /// Unsymmetrized F_n, G_n of the ordered momenta q[0..n) (the recursion)
        inline void fg_unsym(const Vec3 * q, int n, double & F, double & G) {
            if (n == 1) {
                F = G = 1.0;
                return;
            }
            F = G = 0.0;
            double scale2 = 0.0;
            for (int i = 0; i < n; i++)
                scale2 += q[i].norm2();
            for (int m = 1; m < n; m++) {
                const Vec3 k1 = sum(q, m), k2 = sum(q + m, n - m), k12 = k1 + k2;
                if (vanishes(k1, scale2) or vanishes(k2, scale2))
                    continue;
                double Fm, Gm, Fr, Gr;
                fg_unsym(q, m, Fm, Gm);
                fg_unsym(q + m, n - m, Fr, Gr);
                const double k1sq = k1.norm2(), k2sq = k2.norm2(), k1k2 = k1.dot(k2);
                const double alpha = k12.dot(k1) / k1sq;
                const double beta = k12.norm2() * k1k2 / (2.0 * k1sq * k2sq);
                const double norm = (2.0 * n + 3.0) * (n - 1.0);
                F += Gm * ((2.0 * n + 1.0) * alpha * Fr + 2.0 * beta * Gr) / norm;
                G += Gm * (3.0 * alpha * Fr + 2.0 * n * beta * Gr) / norm;
            }
        }

        /// Apply f to every permutation of q[0..n) and average (symmetrization)
        template <class Function>
        void symmetrize(const Vec3 * q, int n, Function && f) {
            std::array<int, 3> perm{0, 1, 2};
            std::array<Vec3, 3> p;
            int count = 0;
            do {
                for (int i = 0; i < n; i++)
                    p[i] = q[perm[i]];
                f(p.data());
                count++;
            } while (std::next_permutation(perm.begin(), perm.begin() + n));
            (void)count;
        }
        inline int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

        /// sigma^2(a, b) = (a.b)^2 / (a^2 b^2) - 1, the second order kernel of G2(Phi)
        inline double sigma2(const Vec3 & a, const Vec3 & b) {
            const double ab = a.dot(b);
            return ab * ab / (a.norm2() * b.norm2()) - 1.0;
        }
    } // namespace detail

    /// The symmetrized EdS kernels F_n and G_n of q[0..n) (n = 1, 2, 3)
    inline void FG(const Vec3 * q, int n, double & F, double & G) {
        F = G = 0.0;
        detail::symmetrize(q, n, [&](const Vec3 * p) {
            double f, g;
            detail::fg_unsym(p, n, f, g);
            F += f;
            G += g;
        });
        const double norm = 1.0 / detail::factorial(n);
        F *= norm;
        G *= norm;
    }
    inline double F2(const Vec3 & a, const Vec3 & b) {
        const double ab = a.dot(b), a2 = a.norm2(), b2 = b.norm2(), mu = ab / std::sqrt(a2 * b2);
        const double r = std::sqrt(a2 / b2);
        return 5.0 / 7.0 + 0.5 * mu * (r + 1.0 / r) + 2.0 / 7.0 * mu * mu;
    }
    inline double G2(const Vec3 & a, const Vec3 & b) {
        const double ab = a.dot(b), a2 = a.norm2(), b2 = b.norm2(), mu = ab / std::sqrt(a2 * b2);
        const double r = std::sqrt(a2 / b2);
        return 3.0 / 7.0 + 0.5 * mu * (r + 1.0 / r) + 4.0 / 7.0 * mu * mu;
    }

    /// Bias components: the kernels are sum_c Z[c] * bias_c with bias = {1, b1, b2, bG2, bGamma3}
    constexpr int NBIAS = 5;
    using BiasKernel = std::array<double, NBIAS>;
    enum BiasIndex { B_ONE = 0, B_B1 = 1, B_B2 = 2, B_BG2 = 3, B_BGAMMA3 = 4 };

    namespace detail {
        /// Everything Z_n needs about the subsets of the momenta (bit masks of q[0..n)): the summed vector and the
        /// symmetrized F, G of the momenta in the subset
        struct Subsets {
            int n{0};
            std::array<Vec3, 8> V;
            std::array<double, 8> F{}, G{};
            std::array<bool, 8> zero{};
            Subsets(const Vec3 * q, int n_) : n(n_) {
                double scale2 = 0.0;
                for (int i = 0; i < n; i++)
                    scale2 += q[i].norm2();
                for (int mask = 1; mask < (1 << n); mask++) {
                    std::array<Vec3, 3> sub;
                    int m = 0;
                    for (int i = 0; i < n; i++)
                        if (mask & (1 << i))
                            sub[m++] = q[i];
                    V[mask] = sum(sub.data(), m);
                    zero[mask] = vanishes(V[mask], scale2);
                    if (not zero[mask])
                        FG(sub.data(), m, F[mask], G[mask]);
                }
            }
        };
        inline int popcount(int mask) { return __builtin_popcount(unsigned(mask)); }

        /// The symmetrized real-space kernel of delta_g of the momenta in mask (empty mask: the 1 in 1 + delta_g),
        /// in bias components
        inline BiasKernel density_sym(const Subsets & S, const Vec3 * q, int mask) {
            BiasKernel K{};
            const int m = popcount(mask);
            if (m == 0) {
                K[B_ONE] = 1.0;
                return K;
            }
            K[B_B1] = S.F[mask];
            if (m == 2) {
                int i = __builtin_ctz(unsigned(mask)), j = __builtin_ctz(unsigned(mask & ~(1 << i)));
                K[B_B2] = 0.5;
                K[B_BG2] = sigma2(q[i], q[j]);
            } else if (m == 3) {
                // delta^2, G2(Phi), Gamma3 = G2(Phi_g) - G2(Phi_v) at third order: 2/3 sum over the splits into a
                // single momentum and a pair (the average of the compositions (1,2) and (2,1) over the orderings)
                for (int i = 0; i < 3; i++) {
                    const int pair = mask & ~(1 << i);
                    if (S.zero[pair])
                        continue; // F2 = G2 = 0 for a pair summing to zero
                    const double s = sigma2(q[i], S.V[pair]);
                    K[B_B2] += 2.0 / 3.0 * 0.5 * S.F[pair];
                    K[B_BG2] += 2.0 / 3.0 * s * S.F[pair];
                    K[B_BGAMMA3] += 2.0 / 3.0 * s * (S.F[pair] - S.G[pair]);
                }
            }
            return K;
        }
    } // namespace detail

    /// The symmetrized redshift-space kernel Z_n(q[0..n)) (n = 1, 2, 3) in bias components (growth rate f, line of
    /// sight los, a unit vector). The sum over ordered sequences of velocity blocks B_1..B_m and the density block D
    /// (the rest) of (prod |B_i|! |D|! / n!) (f k_z)^m / m! prod_i [(V_B.los / V_B^2) G_B] K_D
    inline BiasKernel Zn(const Vec3 * q, int n, double f, const Vec3 & los) {
        const detail::Subsets S(q, n);
        const int all = (1 << n) - 1;
        const double kz = S.V[all].dot(los);
        BiasKernel Z{};
        auto recurse = [&](auto && self, int used, int m, double weight) -> void {
            const int rest = all & ~used;
            if (rest != 0 or m >= 1) {
                const BiasKernel K = detail::density_sym(S, q, rest);
                const double pref = weight * detail::factorial(detail::popcount(rest)) * std::pow(f * kz, m) /
                                    detail::factorial(m) / detail::factorial(n);
                for (int c = 0; c < NBIAS; c++)
                    Z[c] += pref * K[c];
            }
            // The next velocity block: any nonempty subset of the rest
            for (int B = rest; B > 0; B = (B - 1) & rest) {
                if (S.zero[B])
                    continue; // G of momenta summing to zero vanishes
                const double vel = S.V[B].dot(los) / S.V[B].norm2() * S.G[B];
                self(self, used | B, m + 1, weight * detail::factorial(detail::popcount(B)) * vel);
            }
        };
        recurse(recurse, 0, 0, 1.0);
        return Z;
    }

    /// Z2 in closed form: b2/2 + bG2 sigma^2 + b1 F2 + f mu^2 G2 + f mu k/2 [mu1/k1 (b1 + f mu2^2) + mu2/k2 (b1 +
    /// f mu1^2)], mu = k.los/k, k = k1 + k2
    inline BiasKernel Z2(const Vec3 & k1, const Vec3 & k2, double f, const Vec3 & los) {
        const Vec3 k = k1 + k2;
        const double kz = k.dot(los), k1z = k1.dot(los), k2z = k2.dot(los);
        const double k1sq = k1.norm2(), k2sq = k2.norm2(), ksq = k.norm2();
        const double mu1sq = k1z * k1z / k1sq, mu2sq = k2z * k2z / k2sq, musq = kz * kz / ksq;
        BiasKernel Z{};
        const double a = 0.5 * f * kz; // f mu k / 2
        Z[B_ONE] = f * musq * G2(k1, k2) + a * f * (k1z / k1sq * mu2sq + k2z / k2sq * mu1sq);
        Z[B_B1] = F2(k1, k2) + a * (k1z / k1sq + k2z / k2sq);
        Z[B_B2] = 0.5;
        Z[B_BG2] = detail::sigma2(k1, k2);
        Z[B_BGAMMA3] = 0.0;
        return Z;
    }

} // namespace FML::PERTURBATION::SPT
#endif
