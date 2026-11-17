#ifndef FML_HYBRIDBIASONELOOPRSD_HEADER
#define FML_HYBRIDBIASONELOOPRSD_HEADER

#include <FML/ComputePowerSpectra/HybridBiasOneLoop.h>
#include <FML/PerturbationTheory/LPTRedshiftSpace.h>

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {

            //=====================================================================================================
            ///
            /// The one-loop LPT prediction for the hybrid bias spectra in redshift space, P_ab(k, nu) and the
            /// multipoles P_ab^l(k), of the operators 1, delta_L, delta_L^2 - <delta_L^2>, s^2 - <s^2>, nabla^2 delta_L
            /// (the measurement is compute_hybrid_bias_multipoles in HybridBiasRedshiftSpace.h).
            ///
            /// The ten spectra without nabla^2 delta are the components of the one-loop redshift-space LPT of Chen,
            /// Vlah, Castorina & White 2021 (PerturbationTheory/LPTRedshiftSpace.h, velocileptors' LPT_RSD) mapped to
            /// the basis as in real space (HybridBiasOneLoop.h): P_11 = [1], P_1d = [b1]/2, P_dd = [b1^2], P_1d2 =
            /// [b2], P_dd2 = [b1 b2], P_d2d2 = 4 [b2^2], P_1s2 = [bs]/2, P_ds2 = [b1 bs]/2, P_d2s2 = [b2 bs],
            /// P_s2s2 = [bs^2]. The five with nabla^2 delta = F(k) delta_L (F = -k^2) are the delta_L terms with F on
            /// the leg of the operator, at the same order (LPTRedshiftSpace::operator_components; the loop spectra
            /// with F on a leg from detail::loop_spectra). As for delta_L, (delta^2, nabla^2 delta) and (s^2,
            /// nabla^2 delta) keep only the terms of one-loop order (2 D xi_dL and 2 C^L:S; the real-space prediction
            /// of HybridBiasOneLoop.h uses the full Zel'dovich expressions for these two, which differ at two loops).
            ///
            /// Input: the linear P(k) at the redshift (k in h/Mpc, P in (Mpc/h)^3), already multiplied by the filter
            /// W^2 of the field the operators and the displacement are made from if any (the same filter for both;
            /// use a smooth filter: the FFTLog transforms ring for a sharp cut), and the growth rate f. The redshift-
            /// space displacements are those of LPTRedshiftSpace (scale-independent growth, EdS-like time dependence
            /// of the higher orders, as velocileptors). Large scales: P_11 = (1 + f nu^2)^2 P, P_1d = (1 + f nu^2) P,
            /// P_dd = P (times the Zel'dovich damping), P_1n = -k^2 (1 + f nu^2) P, P_dn = -k^2 P, P_nn = k^4 P.
            ///
            /// Units: those of P(k); nabla^2 delta in units of 1/k^2 (as HybridBiasSpectra::scale).
            ///
            //=====================================================================================================
            class HybridBiasOneLoopRSD {
              public:
                using DVector = std::vector<double>;
                using LPT = FML::PERTURBATION::LPTRedshiftSpace;

                HybridBiasOneLoopRSD() = default;
                HybridBiasOneLoopRSD(const DVector & k, const DVector & P, LPT::Options options = LPT::Options())
                    : lpt(k, P, options) {
                    // The loop spectra with F = -k^2 on a leg, on a coarser log grid, interpolated to the internal grid
                    const auto & kint = lpt.k_internal();
                    const auto & pint = lpt.p_internal();
                    const double kmin = kint.front(), kmax = kint.back();
                    const std::function<double(double)> Pfun = [&](double kk) {
                        if (kk < kmin or kk > kmax)
                            return 0.0;
                        return FML::PERTURBATION::detail::interp_clamped(std::log(kk), logk, pint);
                    };
                    logk.resize(kint.size());
                    for (size_t i = 0; i < kint.size(); i++)
                        logk[i] = std::log(kint[i]);
                    const std::function<double(double)> F = [](double kk) { return -kk * kk; };
                    const int nloop = 400;
                    DVector kl(nloop), lkl(nloop), Q5F(nloop), R1F(nloop), R2F(nloop);
                    for (int i = 0; i < nloop; i++) {
                        lkl[i] = std::log(kmin) + (std::log(kmax) - std::log(kmin)) * i / double(nloop - 1);
                        kl[i] = std::exp(lkl[i]) * (i == nloop - 1 ? 1.0 - 1e-12 : 1.0);
                    }
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
                    for (int i = 0; i < nloop; i++) {
                        if (not options.one_loop)
                            continue; // no loop terms: the Zel'dovich operator spectra
                        const auto L = detail::loop_spectra(kl[i], Pfun, kmin, kmax, F);
                        Q5F[i] = L.Q5F;
                        R1F[i] = L.R1F;
                        R2F[i] = L.R2F;
                    }
                    DVector q5(kint.size()), r1(kint.size()), r2(kint.size());
                    for (size_t i = 0; i < kint.size(); i++) {
                        q5[i] = FML::PERTURBATION::detail::interp_clamped(logk[i], lkl, Q5F);
                        r1[i] = FML::PERTURBATION::detail::interp_clamped(logk[i], lkl, R1F);
                        r2[i] = FML::PERTURBATION::detail::interp_clamped(logk[i], lkl, R2F);
                    }
                    oq = lpt.operator_qfunctions(F, q5, r1, r2);
                }

                /// The 15 spectra P_ab(k, nu) at the growth rate f: result[pair][ik], pairs as HybridBiasSpectra
                /// (pairs() gives them)
                std::vector<DVector> pkmu(double f, double nu, const DVector & k) {
                    const auto c = lpt.pk_components(f, nu, k);
                    const auto o = lpt.operator_components(oq, f, nu, k);
                    const auto pl = pairs();
                    std::vector<DVector> result(pl.size(), DVector(k.size(), 0.0));
                    for (size_t p = 0; p < pl.size(); p++) {
                        const int a = pl[p][0], b = pl[p][1];
                        for (size_t ik = 0; ik < k.size(); ik++) {
                            const auto & cc = c[ik];
                            const auto & oo = o[ik];
                            double v = 0.0;
                            if (b == NABLA2DELTA)
                                v = oo[a]; // (1,n), (d,n), (d2,n), (s2,n), (n,n) are components 0..4
                            else if (a == ONE and b == ONE)
                                v = cc[0];
                            else if (a == ONE and b == DELTA)
                                v = 0.5 * cc[1];
                            else if (a == ONE and b == DELTA2)
                                v = cc[3];
                            else if (a == ONE and b == S2)
                                v = 0.5 * cc[6];
                            else if (a == DELTA and b == DELTA)
                                v = cc[2];
                            else if (a == DELTA and b == DELTA2)
                                v = cc[4];
                            else if (a == DELTA and b == S2)
                                v = 0.5 * cc[7];
                            else if (a == DELTA2 and b == DELTA2)
                                v = 4.0 * cc[5];
                            else if (a == DELTA2 and b == S2)
                                v = cc[8];
                            else if (a == S2 and b == S2)
                                v = cc[9];
                            result[p][ik] = v;
                        }
                    }
                    return result;
                }

                /// The multipoles P_ab^l(k) = (2l+1) int_0^1 dnu P_ab(k, nu) L_l(nu) (Gauss-Legendre with nnu nodes in
                /// [0,1]): result[pair][ell][ik] for ell = 0..ellmax (odd ell zero)
                std::vector<std::vector<DVector>> multipoles(double f, const DVector & k, int ellmax = 4, int nnu = 8) {
                    if (ellmax < 0 or ellmax % 2 != 0)
                        throw std::invalid_argument("[HybridBiasOneLoopRSD::multipoles] ellmax must be even");
                    DVector x, w;
                    LPT::gauss_legendre(2 * nnu, x, w);
                    const int npairs = int(pairs().size());
                    std::vector<std::vector<DVector>> result(npairs,
                                                             std::vector<DVector>(ellmax + 1, DVector(k.size(), 0.0)));
                    for (size_t in = 0; in < x.size(); in++) {
                        if (x[in] <= 0.0)
                            continue;
                        const auto pk = pkmu(f, x[in], k);
                        for (int ell = 0; ell <= ellmax; ell += 2) {
                            const double fac = (2.0 * ell + 1.0) * w[in] * detail::legendre(ell, x[in]);
                            for (int p = 0; p < npairs; p++)
                                for (size_t ik = 0; ik < k.size(); ik++)
                                    result[p][ell][ik] += fac * pk[p][ik];
                        }
                    }
                    return result;
                }

                /// The pairs (a, b), a <= b, in the order of the results (as HybridBiasSpectra)
                static std::vector<std::array<int, 2>> pairs() {
                    std::vector<std::array<int, 2>> pl;
                    for (int a = 0; a < max_operators; a++)
                        for (int b = a; b < max_operators; b++)
                            pl.push_back({a, b});
                    return pl;
                }
                static std::vector<std::string> names() { return operator_names(); }

                LPT & lpt_model() { return lpt; }
                const LPT::OperatorQFunctions & nabla2_qfunctions() const { return oq; }

              private:
                LPT lpt;
                LPT::OperatorQFunctions oq;
                DVector logk;
            };

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
