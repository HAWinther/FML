#ifndef FML_HYBRIDBIASONELOOPMG_HEADER
#define FML_HYBRIDBIASONELOOPMG_HEADER

#include <FML/ComputePowerSpectra/HybridBiasOneLoop.h>

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {
        namespace HYBRIDBIAS {

            //=====================================================================================================
            ///
            /// One-loop LPT (CLEFT) prediction of the hybrid bias spectra in modified gravity with the full
            /// scale- and time-dependent LPT kernels (as in Aviles & Cervantes-Cota 2017, Aviles et al. 2018,
            /// Brando, Li & Koyama 2026): a general mu(k,a) model, Hu-Sawicki f(R) and normal-branch DGP, with or
            /// without screening (the chameleon M2, M3 and the Vainshtein terms of the scalar field equation).
            ///
            /// The quasi-static equations (Koyama, Taruya & Hiramatsu 2009): the Poisson equation -k^2 Phi = A(k,a)
            /// delta with A = 3/2 Omega_M(a) H^2 mu(k,a), and for f(R) and DGP the nonlinear source of the scalar
            ///   3 Pi(k) phi = 8 pi G rho delta - (1/2) M2 phi phi - (1/6) M3 phi phi phi
            ///   -k^2 Phi = A(k) delta - (k^2/a^2) NL(k) / (6 Pi(k)),   NL = (1/2) M2 phi phi + (1/6) M3 phi phi phi.
            /// The particles obey d^2x/dt^2 + 2H dx/dt = -grad_x Phi(x) with the potential sourced by the Eulerian
            /// density. With delta_E(k) = int d^3q e^(-ik.q) (e^(-ik.Psi(q)) - 1) and T Psi(q) = int_k (ik/k^2)
            /// Sigma(k) e^(ik.(q + Psi(q))) (T = d^2/dt^2 + 2H d/dt, Sigma = A delta_E + screening) the LPT kernels
            /// follow order by order without transforming the equations to Lagrangian space: the frame-lagging
            /// terms come from the e^(ik.Psi(q)) factor. The source at each order is a sum over the ways to split
            /// the legs between e^(ik.Psi(q)) and delta_E (set partitions). With Psi_n(K) = i [K/K^2 c_n + transverse]
            /// int delta(p_1)..delta(p_n) / n! (delta the linear field at the same time):
            ///   (T - A(K)) c_n = S_n.
            /// With a scale-dependent A the second order displacement has a transverse part (vorticity in Lagrangian
            /// space, the Eulerian flow stays irrotational): Psi_2 = i [(K/K^2) c_2 + w (p1 - p2)_perp] with
            /// T w = -(1/2) (A(p1) - A(p2)) (p1.p2)/(p1^2 p2^2) D1 D2. It enters the third order kernel and A^22,
            /// W^112, A^10 and U^11 at one loop. In GR c_2 = (3/7)(1 - mu^2) D_1 D_2 in EdS and w = 0.
            ///
            /// The ODEs are integrated in log a (RK4 from a = 1e-3, where the fifth force is negligible, with the
            /// growing mode of EdS as initial condition) for every node of the loop integrals (no interpolation of
            /// the kernels): for each k and loop momentum p, the second order kernels on the nodes of the angular
            /// integral (the Q-type, legs p and k - p, and the R-type, legs k and +-p) and the angle-integrated third
            /// order kernel c_3(k, p, -p). The loop functions are the correlators of the CLEFT q-functions (A^22,
            /// A^13, W^112, A^10, U^3, U^11, U^20, U^s2 and the nabla^2 delta versions) written with general
            /// kernels; the q-functions and the mu/q integrals are those of oneloop_hybrid_bias_spectra.
            ///
            /// Tests: with the EdS kernels (Omega_M = 1, GR) all 15 spectra agree with oneloop_hybrid_bias_spectra
            /// (velocileptors) to 1e-4; in LCDM the kernels give the known Omega_M^(-1/143) (2nd order, +0.9%) and
            /// ~+1.9% (3rd order) corrections; the second and third order kernels agree with the Eulerian SPT
            /// kernels of the same equations mapped to LPT (f(R) with and without screening, DGP, a mu(k,a) model)
            /// to 1e-5; the loop functions Q1, Q8, R1+2 agree with mgpt (Aviles et al.) to 0.1% for f(R) and
            /// unscreened DGP.
            ///
            /// Units: k in h/Mpc. Internally k in units of H0 (k/H0 = k [h/Mpc] * 2997.92458).
            ///
            //=====================================================================================================

            /// A modified gravity model on a flat LCDM background (matter and a cosmological constant)
            struct MGModel {
                enum Type { GR, MU, FOFR, DGP };
                Type type{GR};
                double OmegaM{0.3};
                /// MU: mu(k [h/Mpc], a) (no screening)
                std::function<double(double, double)> mu_of_k_a;
                /// FOFR: |f_R0| and the Hu-Sawicki n
                double fofr0{1e-5};
                double nfofr{1.0};
                /// DGP (normal branch): H0 r_c
                double H0rc{1.0};
                /// Include the screening (chameleon M2, M3 for f(R), Vainshtein for DGP)
                bool screening{true};

                std::string name() const {
                    switch (type) {
                        case GR:
                            return "GR";
                        case MU:
                            return "mu(k,a)";
                        case FOFR:
                            return "f(R) Hu-Sawicki (fR0 = " + std::to_string(fofr0) +
                                   ", n = " + std::to_string(nfofr) + (screening ? ", screening)" : ", no screening)");
                        case DGP:
                            return "nDGP (H0 rc = " + std::to_string(H0rc) +
                                   (screening ? ", screening)" : ", no screening)");
                    }
                    return "";
                }
            };

            /// Convenience constructors
            inline MGModel mgmodel_gr(double OmegaM) {
                MGModel m;
                m.type = MGModel::GR;
                m.OmegaM = OmegaM;
                return m;
            }
            inline MGModel mgmodel_mu(double OmegaM, std::function<double(double, double)> mu_of_k_hmpc_a) {
                MGModel m;
                m.type = MGModel::MU;
                m.OmegaM = OmegaM;
                m.mu_of_k_a = mu_of_k_hmpc_a;
                m.screening = false;
                return m;
            }
            inline MGModel mgmodel_fofr(double OmegaM, double fofr0, double nfofr = 1.0, bool screening = true) {
                MGModel m;
                m.type = MGModel::FOFR;
                m.OmegaM = OmegaM;
                m.fofr0 = fofr0;
                m.nfofr = nfofr;
                m.screening = screening;
                return m;
            }
            inline MGModel mgmodel_dgp(double OmegaM, double H0rc, bool screening = true) {
                MGModel m;
                m.type = MGModel::DGP;
                m.OmegaM = OmegaM;
                m.H0rc = H0rc;
                m.screening = screening;
                return m;
            }

            namespace detail {

                constexpr double c_over_H0_mpch = 2997.92458;

                using Vec3 = std::array<double, 3>;
                inline double dot3(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
                inline Vec3 add3(const Vec3 & a, const Vec3 & b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }

                /// Everything at one time (H0 = 1 units, k in units of H0)
                struct MGTime {
                    const MGModel * model{nullptr};
                    double a{1.0}, H2{1.0}, dlogHdx{0.0}, rhofac{1.0}; // rhofac = Omega_M a^-3 = 8 pi G rho / 3
                    // f(R): M1 = 3 m^2, M2, M3; DGP: beta and r_c
                    double m2{0.0}, M2{0.0}, M3{0.0}, beta{1.0}, rc{0.0};
                    bool screening{false};

                    void set(const MGModel & mod, double aa) {
                        model = &mod;
                        a = aa;
                        const double OmegaM = mod.OmegaM, OmegaL = 1.0 - OmegaM;
                        rhofac = OmegaM / (a * a * a);
                        H2 = rhofac + OmegaL;
                        dlogHdx = -1.5 * rhofac / H2;
                        screening = mod.screening and (mod.type == MGModel::FOFR or mod.type == MGModel::DGP);
                        if (mod.type == MGModel::FOFR) {
                            const double c = rhofac + 4.0 * OmegaL, c0 = OmegaM + 4.0 * OmegaL;
                            const double n = mod.nfofr, pexp = 1.0 / (n + 1.0);
                            const double Rbar = 3.0 * c;
                            const double fbar = mod.fofr0 * std::pow(c0 / c, n + 1.0);
                            // R(f_R) expanded around the background: M1 = dR/df, M2 = d^2R/df^2, M3 = d^3R/df^3
                            const double M1 = Rbar * pexp / fbar;
                            m2 = M1 / 3.0;
                            M2 = Rbar * pexp * (pexp + 1.0) / (fbar * fbar);
                            M3 = Rbar * pexp * (pexp + 1.0) * (pexp + 2.0) / (fbar * fbar * fbar);
                        }
                        if (mod.type == MGModel::DGP) {
                            rc = mod.H0rc;
                            beta = 1.0 + 2.0 * std::sqrt(H2) * rc * (1.0 + dlogHdx / 3.0);
                        }
                    }
                    /// mu(k) with k in units of H0
                    double mu(double k) const {
                        switch (model->type) {
                            case MGModel::GR:
                                return 1.0;
                            case MGModel::MU:
                                return model->mu_of_k_a(k / c_over_H0_mpch, a);
                            case MGModel::FOFR: {
                                const double k2a2 = k * k / (a * a);
                                return 1.0 + (1.0 / 3.0) * k2a2 / (k2a2 + m2);
                            }
                            case MGModel::DGP:
                                return 1.0 + 1.0 / (3.0 * beta);
                        }
                        return 1.0;
                    }
                    double A(double k) const { return 1.5 * rhofac * mu(k); }
                    /// Pi(k) (units H0^2): 3 Pi phi = 8 pi G rho delta at linear order
                    double Pi(double k) const {
                        const double k2a2 = k * k / (a * a);
                        if (model->type == MGModel::FOFR)
                            return k2a2 + m2;
                        if (model->type == MGModel::DGP)
                            return beta * k2a2;
                        return 0.0;
                    }
                    /// phi / delta at linear order: 8 pi G rho / (3 Pi)
                    double phihat(double k) const { return rhofac / Pi(k); }
                    /// M2(k1, k2)
                    double M2of(const Vec3 & k1, const Vec3 & k2) const {
                        if (model->type == MGModel::FOFR)
                            return M2;
                        if (model->type == MGModel::DGP) {
                            const double d = dot3(k1, k2);
                            return 2.0 * rc * rc / (a * a * a * a) * (dot3(k1, k1) * dot3(k2, k2) - d * d);
                        }
                        return 0.0;
                    }
                    /// (k^2/a^2) / (6 Pi(k)) (finite at k = 0 for DGP, zero for f(R))
                    double screen_prefactor(double k) const {
                        if (model->type == MGModel::DGP)
                            return 1.0 / (6.0 * beta);
                        if (k == 0.0)
                            return 0.0;
                        return k * k / (a * a) / (6.0 * Pi(k));
                    }
                };

                /// The source of the second order kernel c2 for legs p1, p2 (D1, D2 the linear growth at the legs)
                inline double source_c2(const MGTime & T, const Vec3 & p1, const Vec3 & p2, double D1, double D2) {
                    const double p1sq = dot3(p1, p1), p2sq = dot3(p2, p2), d = dot3(p1, p2);
                    const double Ksq = p1sq + p2sq + 2.0 * d;
                    const double K = std::sqrt(std::max(Ksq, 0.0));
                    const double Kp1 = p1sq + d, Kp2 = p2sq + d;
                    const double p1n = std::sqrt(p1sq), p2n = std::sqrt(p2sq);
                    const double AK = T.A(K), A1 = T.A(p1n), A2 = T.A(p2n);
                    double S = AK * (p1sq * p2sq - d * d) / (p1sq * p2sq) +
                               ((AK - A2) * Kp2 + (AK - A1) * Kp1) * d / (p1sq * p2sq);
                    S *= D1 * D2;
                    if (T.screening and K > 0.0)
                        S -= T.screen_prefactor(K) * T.M2of(p1, p2) * T.phihat(p1n) * T.phihat(p2n) * D1 * D2;
                    return S;
                }

                /// The source of the transverse part of the second order displacement. With a scale-dependent A(k) the
                /// force on the pair (p1, p2) is not along p1 + p2 and Psi_2 has a transverse part (vorticity in the
                /// Lagrangian displacement, not in the Eulerian flow): Psi_2 = i [(P/P^2) c2 + w (p1 - p2)_perp] with
                /// T w = -(1/2) (A(p1) - A(p2)) (p1.p2) / (p1^2 p2^2) D1 D2 (zero in GR)
                inline double source_w2(const MGTime & T, const Vec3 & p1, const Vec3 & p2, double D1, double D2) {
                    const double p1sq = dot3(p1, p1), p2sq = dot3(p2, p2);
                    const double dA = T.A(std::sqrt(p1sq)) - T.A(std::sqrt(p2sq));
                    if (dA == 0.0)
                        return 0.0;
                    return -0.5 * dA * dot3(p1, p2) / (p1sq * p2sq) * D1 * D2;
                }
                /// The transverse direction of a pair: (p1 - p2) - P (P.(p1 - p2)) / P^2 with P = p1 + p2
                inline Vec3 transverse_dir(const Vec3 & p1, const Vec3 & p2) {
                    const Vec3 P = add3(p1, p2);
                    const Vec3 d{p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]};
                    const double Psq = dot3(P, P);
                    if (Psq == 0.0)
                        return {0.0, 0.0, 0.0};
                    const double f = dot3(P, d) / Psq;
                    return {d[0] - f * P[0], d[1] - f * P[1], d[2] - f * P[2]};
                }

                /// The source of c3 for the three legs v[0..2] (D the linear growth at the legs, c2pair[i] the
                /// second order kernel of the pair without leg i, w2pair[i] the transverse part of the pair (v[i+1],
                /// v[i+2]) in that order). Set partitions of the legs between the e^(ik.Psi(q)) factor (Q) and delta_E
                /// (E)
                inline double source_c3(const MGTime & T,
                                        const Vec3 v[3],
                                        const double D[3],
                                        const double c2pair[3],
                                        const double w2pair[3]) {
                    const Vec3 K = add3(add3(v[0], v[1]), v[2]);
                    const double Ksq = dot3(K, K);
                    const double Kn = std::sqrt(Ksq);
                    double vsq[3], vn[3];
                    for (int i = 0; i < 3; i++) {
                        vsq[i] = dot3(v[i], v[i]);
                        vn[i] = std::sqrt(vsq[i]);
                    }
                    // pair sums P_bc (excluding leg a) and the transverse directions of the pairs
                    Vec3 P[3], U[3];
                    double Psq[3], Pn[3];
                    for (int aidx = 0; aidx < 3; aidx++) {
                        const int b = (aidx + 1) % 3, c = (aidx + 2) % 3;
                        P[aidx] = add3(v[b], v[c]);
                        Psq[aidx] = dot3(P[aidx], P[aidx]);
                        Pn[aidx] = std::sqrt(Psq[aidx]);
                        U[aidx] = transverse_dir(v[b], v[c]);
                    }
                    const double tiny = 1e-12 * (vn[0] + vn[1] + vn[2]);
                    const double AK = T.A(Kn);

                    // delta_E at second order for the pair without leg a (zero for a zero total momentum)
                    double dE2[3], Sscr2[3];
                    for (int aidx = 0; aidx < 3; aidx++) {
                        const int b = (aidx + 1) % 3, c = (aidx + 2) % 3;
                        if (Pn[aidx] <= tiny) {
                            dE2[aidx] = 0.0;
                            Sscr2[aidx] = 0.0;
                            continue;
                        }
                        dE2[aidx] =
                            c2pair[aidx] + dot3(P[aidx], v[b]) * dot3(P[aidx], v[c]) * D[b] * D[c] / (vsq[b] * vsq[c]);
                        Sscr2[aidx] = 0.0;
                        if (T.screening)
                            Sscr2[aidx] = -T.screen_prefactor(Pn[aidx]) * T.M2of(v[b], v[c]) * T.phihat(vn[b]) *
                                          T.phihat(vn[c]) * D[b] * D[c];
                    }

                    double S = 0.0;
                    // Q empty: delta_E with the partitions {a}{bc} and {1}{2}{3} (the single block is the linear term)
                    {
                        double dE3 = 1.0;
                        for (int i = 0; i < 3; i++)
                            dE3 *= dot3(K, v[i]) * D[i] / vsq[i];
                        for (int aidx = 0; aidx < 3; aidx++) {
                            if (Pn[aidx] <= tiny)
                                continue;
                            dE3 += dot3(K, v[aidx]) * D[aidx] / vsq[aidx] *
                                   (dot3(K, P[aidx]) * c2pair[aidx] / Psq[aidx] + w2pair[aidx] * dot3(K, U[aidx]));
                        }
                        S += AK * dE3;
                    }
                    // Q = {a}, E = {b, c}: k = P_bc
                    for (int aidx = 0; aidx < 3; aidx++) {
                        if (Pn[aidx] <= tiny)
                            continue;
                        const double Sig = T.A(Pn[aidx]) * dE2[aidx] + Sscr2[aidx];
                        const double qfac = -dot3(P[aidx], v[aidx]) * D[aidx] / vsq[aidx];
                        S += dot3(K, P[aidx]) / Psq[aidx] * Sig * qfac;
                    }
                    // Q = {a, b}, E = {c}: k = v_c
                    for (int c = 0; c < 3; c++) {
                        const int aidx = (c + 1) % 3, b = (c + 2) % 3;
                        double qfac = dot3(v[c], v[aidx]) * dot3(v[c], v[b]) * D[aidx] * D[b] / (vsq[aidx] * vsq[b]);
                        if (Pn[c] > tiny)
                            qfac -= dot3(v[c], P[c]) * c2pair[c] / Psq[c] + w2pair[c] * dot3(v[c], U[c]);
                        S += dot3(K, v[c]) / vsq[c] * T.A(vn[c]) * D[c] * qfac;
                    }
                    // The screening at third order
                    if (T.screening and Kn > 0.0) {
                        double NL3 = T.M3 * T.phihat(vn[0]) * T.phihat(vn[1]) * T.phihat(vn[2]) * D[0] * D[1] * D[2];
                        if (T.model->type != MGModel::FOFR)
                            NL3 = 0.0;
                        for (int aidx = 0; aidx < 3; aidx++) {
                            const int b = (aidx + 1) % 3, c = (aidx + 2) % 3;
                            const double M2a = T.M2of(v[aidx], P[aidx]);
                            if (M2a == 0.0)
                                continue;
                            // phi at second order for the pair: phihat delta_E2 - M2 phihat phihat D D / (3 Pi)
                            double phi2 = -T.M2of(v[b], v[c]) * T.phihat(vn[b]) * T.phihat(vn[c]) * D[b] * D[c] /
                                          (3.0 * T.Pi(Pn[aidx]));
                            if (Pn[aidx] > tiny)
                                phi2 += T.phihat(Pn[aidx]) * dE2[aidx];
                            NL3 += M2a * T.phihat(vn[aidx]) * D[aidx] * phi2;
                        }
                        S -= T.screen_prefactor(Kn) * NL3;
                    }
                    return S;
                }

                /// The loop functions at one k for every output time (see oneloop_hybrid_bias_spectra_mg)
                struct MGLoopFunctions {
                    // Q-type (legs p and k - p): weights P(p) P(|k-p|). A22 = a delta + b khat khat: G22 = k^2 b,
                    // G22A = k^2 a (from the transverse part of Psi_2)
                    double G22{0}, G22A{0}, U20{0}, Us2{0}, A10Qt{0}, A10Qkk{0}, A10QtF{0}, A10QkkF{0}, WQa{0}, WQb{0};
                    // R-type (legs k and p) without the external P(k): weights P(p)
                    double A10R1t{0}, A10R1kk{0}, A10R2t{0}, A10R2tF{0}, g{0}, gF{0}, WRa{0}, WRb{0}, I3{0};
                };

                /// Composite Gauss-Legendre on [lo, hi] with segments halving in size towards one end (for log
                /// singularities at that end) down to the given resolution; the last segment reaches the end
                inline void refined_nodes(double lo,
                                          double hi,
                                          bool towards_hi,
                                          double resolution,
                                          int order,
                                          std::vector<double> & x,
                                          std::vector<double> & w) {
                    std::vector<double> xs, ws;
                    const double len = hi - lo;
                    if (len <= 0.0)
                        return;
                    double frac = 1.0;
                    for (int level = 0; level < 60; level++) {
                        const bool last = (0.5 * frac * len <= resolution) or level == 59;
                        const double f0 = last ? 0.0 : 0.5 * frac;
                        // the segment between the distances frac and f0 from the end
                        double a = towards_hi ? hi - frac * len : lo + f0 * len;
                        double b = towards_hi ? hi - f0 * len : lo + frac * len;
                        composite_gl(a, b, 1, order, false, xs, ws);
                        x.insert(x.end(), xs.begin(), xs.end());
                        w.insert(w.end(), ws.begin(), ws.end());
                        if (last)
                            break;
                        frac = f0;
                    }
                }

                /// The settings of the MG loop integrals
                struct MGIntegrationSettings {
                    double aini{1e-3};          // Start of the growth ODEs
                    double dloga{0.05};         // RK4 step in log a
                    int order{6};               // Gauss-Legendre order of the segments
                    double segs_per_efold{2.0}; // Segments per e-fold of the radial/angular ranges
                    int nk_loop{80}; // Number of k (log spaced) the loop functions are computed at (3/4 below the cut)
                    int max_segs{8}; // Maximum number of segments of the angular integrals
                    double refine_to{1e-4}; // Refine the p-integral towards p = k down to this fraction of k
                    int refine_order{4};    // Gauss-Legendre order of the refinement levels
                };

                /// The loop functions for all output times at one k (k in h/Mpc)
                inline std::vector<MGLoopFunctions>
                mg_loop_functions_at_k(const MGModel & model,
                                       double k,
                                       const std::vector<double> & aout,
                                       const std::vector<std::function<double(double)>> & PL, // P_L(k, a_out) cut
                                       double kmin,
                                       double kcut,
                                       const MGIntegrationSettings & set) {
                    const int nout = int(aout.size());
                    std::vector<MGLoopFunctions> res(nout);
                    const double kH = k * c_over_H0_mpch; // units of H0
                    const bool do_R = (k < kcut);

                    // The p-nodes: [kmin, kcut] split at k and |kcut - k|, refined towards p = k
                    std::vector<double> px, pw;
                    {
                        std::vector<double> xs, ws;
                        auto add_plain = [&](double lo, double hi) {
                            if (hi <= lo)
                                return;
                            const int nseg = std::min(4 * set.max_segs,
                                                      std::max(2, int(set.segs_per_efold * std::log(hi / lo)) + 1));
                            composite_gl(lo, hi, nseg, set.order, true, xs, ws);
                            px.insert(px.end(), xs.begin(), xs.end());
                            pw.insert(pw.end(), ws.begin(), ws.end());
                        };
                        std::vector<double> br{kmin, kcut};
                        if (k > kmin and k < kcut)
                            br.push_back(k);
                        if (std::abs(kcut - k) > kmin and std::abs(kcut - k) < kcut)
                            br.push_back(std::abs(kcut - k));
                        std::sort(br.begin(), br.end());
                        for (size_t b = 0; b + 1 < br.size(); b++) {
                            const double lo = br[b], hi = br[b + 1];
                            if (hi <= lo)
                                continue;
                            const bool hi_is_k = std::abs(hi - k) < 1e-12 * k, lo_is_k = std::abs(lo - k) < 1e-12 * k;
                            if (hi_is_k and lo < 0.5 * k) {
                                add_plain(lo, 0.5 * k);
                                refined_nodes(0.5 * k, hi, true, set.refine_to * k, set.refine_order, px, pw);
                            } else if (hi_is_k) {
                                refined_nodes(lo, hi, true, set.refine_to * k, set.refine_order, px, pw);
                            } else if (lo_is_k and hi > 2.0 * k) {
                                refined_nodes(lo, 2.0 * k, false, set.refine_to * k, set.refine_order, px, pw);
                                add_plain(2.0 * k, hi);
                            } else if (lo_is_k) {
                                refined_nodes(lo, hi, false, set.refine_to * k, set.refine_order, px, pw);
                            } else {
                                add_plain(lo, hi);
                            }
                        }
                    }

                    // The time steps (landing on the outputs)
                    std::vector<double> xout(nout);
                    for (int i = 0; i < nout; i++)
                        xout[i] = std::log(aout[i]);
                    std::vector<int> order_out(nout);
                    for (int i = 0; i < nout; i++)
                        order_out[i] = i;
                    std::sort(order_out.begin(), order_out.end(), [&](int i, int j) { return xout[i] < xout[j]; });

                    for (size_t ip = 0; ip < px.size(); ip++) {
                        const double p = px[ip];
                        const double pH = p * c_over_H0_mpch;
                        // Q-nodes: s' = |k - p| in [max(|k-p|, kmin), min(k+p, kcut)]
                        std::vector<double> qx, qw, rx, rw;
                        {
                            const double lo = std::max(std::abs(k - p), kmin), hi = std::min(k + p, kcut);
                            if (hi > lo) {
                                const int nseg = std::min(set.max_segs,
                                                          std::max(1, int(set.segs_per_efold * std::log(hi / lo)) + 1));
                                composite_gl(lo, hi, nseg, set.order, true, qx, qw);
                            }
                        }
                        if (do_R) {
                            const double lo = std::max(std::abs(k - p), 1e-8 * k), hi = k + p;
                            const int nseg =
                                std::min(set.max_segs, std::max(1, int(set.segs_per_efold * std::log(hi / lo)) + 1));
                            composite_gl(lo, hi, nseg, set.order, true, rx, rw);
                        }
                        const int nQ = int(qx.size()), nR = int(rx.size());
                        if (nQ == 0 and nR == 0)
                            continue;

                        // Geometry of the nodes (units of H0)
                        const Vec3 kv{0.0, 0.0, kH};
                        // Q: legs p1 = p, p2 = k - p, angle of p to k from |k - p|
                        std::vector<Vec3> Qp1(nQ), Qp2(nQ);
                        for (int j = 0; j < nQ; j++) {
                            const double s = qx[j];
                            const double mu = std::max(-1.0, std::min(1.0, (k * k + p * p - s * s) / (2.0 * k * p)));
                            Qp1[j] = {pH * std::sqrt(1.0 - mu * mu), 0.0, pH * mu};
                            Qp2[j] = {-Qp1[j][0], 0.0, kH - Qp1[j][2]};
                        }
                        // R: legs k and p with |k + p| = s
                        std::vector<Vec3> Rp(nR), Rm(nR);
                        for (int j = 0; j < nR; j++) {
                            const double s = rx[j];
                            const double mu = std::max(-1.0, std::min(1.0, (s * s - k * k - p * p) / (2.0 * k * p)));
                            Rp[j] = {pH * std::sqrt(1.0 - mu * mu), 0.0, pH * mu};
                            Rm[j] = {-Rp[j][0], 0.0, -Rp[j][2]};
                        }

                        // State: D(k), D(p) (+ derivatives); per Q node: c2(p, k-p), D(|k-p|), w(p, k-p); per R node:
                        // c2(k,p), c2(k,-p), w(k,p), w(-p,k); the angle-integrated c3 (weights of the R nodes)
                        const int iDk = 0, iDp = 2, iQ = 4, iR = iQ + 6 * nQ, iC3 = iR + 8 * nR, nvar = iC3 + 2;
                        std::vector<double> y(nvar), k1v(nvar), k2v(nvar), k3v(nvar), k4v(nvar), yt(nvar);
                        std::vector<double> smag(nR);
                        for (int j = 0; j < nR; j++) {
                            const Vec3 smv = add3(kv, Rm[j]);
                            smag[j] = std::sqrt(dot3(smv, smv));
                        }

                        auto rhs = [&](double x, const std::vector<double> & Y, std::vector<double> & dY) {
                            MGTime T;
                            T.set(model, std::exp(x));
                            const double damp = 2.0 + T.dlogHdx, iH2 = 1.0 / T.H2;
                            auto lin = [&](int i, double Ktot, double S) {
                                dY[i] = Y[i + 1];
                                dY[i + 1] = -damp * Y[i + 1] + (T.A(Ktot) * Y[i] + S) * iH2;
                            };
                            auto free = [&](int i, double S) {
                                dY[i] = Y[i + 1];
                                dY[i + 1] = -damp * Y[i + 1] + S * iH2;
                            };
                            const double Dk = Y[iDk], Dp = Y[iDp];
                            lin(iDk, kH, 0.0);
                            lin(iDp, pH, 0.0);
                            for (int j = 0; j < nQ; j++) {
                                const int i = iQ + 6 * j;
                                const double Ds = Y[i + 2];
                                lin(i, kH, source_c2(T, Qp1[j], Qp2[j], Dp, Ds));
                                lin(i + 2, qx[j] * c_over_H0_mpch, 0.0);
                                free(i + 4, source_w2(T, Qp1[j], Qp2[j], Dp, Ds));
                            }
                            double S3 = 0.0;
                            for (int j = 0; j < nR; j++) {
                                const int i = iR + 8 * j;
                                lin(i, rx[j] * c_over_H0_mpch, source_c2(T, kv, Rp[j], Dk, Dp));
                                lin(i + 2, smag[j], source_c2(T, kv, Rm[j], Dk, Dp));
                                free(i + 4, source_w2(T, kv, Rp[j], Dk, Dp));
                                free(i + 6, source_w2(T, Rm[j], kv, Dp, Dk));
                                // third order for the legs (k, p, -p): pairs without leg 0 = (p,-p) (zero), without
                                // leg 1 = (-p, k), without leg 2 = (k, p)
                                const Vec3 legs[3] = {kv, Rp[j], Rm[j]};
                                const double Dl[3] = {Dk, Dp, Dp};
                                const double c2p[3] = {0.0, Y[i + 2], Y[i]};
                                const double w2p[3] = {0.0, Y[i + 6], Y[i + 4]};
                                S3 += rw[j] * rx[j] * source_c3(T, legs, Dl, c2p, w2p);
                            }
                            lin(iC3, kH, S3);
                        };

                        // Initial conditions (EdS growing modes): D = a, c2 = S2 / (3.5 H^2), c3 = S3 / (9 H^2),
                        // w = 0 (the fifth force is negligible at aini)
                        {
                            const double ai = set.aini;
                            MGTime T;
                            T.set(model, ai);
                            y.assign(nvar, 0.0);
                            y[iDk] = y[iDk + 1] = ai;
                            y[iDp] = y[iDp + 1] = ai;
                            for (int j = 0; j < nQ; j++) {
                                const int i = iQ + 6 * j;
                                y[i + 2] = y[i + 3] = ai;
                                y[i] = source_c2(T, Qp1[j], Qp2[j], ai, ai) / (3.5 * T.H2);
                                y[i + 1] = 2.0 * y[i];
                            }
                            double S3 = 0.0;
                            for (int j = 0; j < nR; j++) {
                                const int i = iR + 8 * j;
                                y[i] = source_c2(T, kv, Rp[j], ai, ai) / (3.5 * T.H2);
                                y[i + 1] = 2.0 * y[i];
                                y[i + 2] = source_c2(T, kv, Rm[j], ai, ai) / (3.5 * T.H2);
                                y[i + 3] = 2.0 * y[i + 2];
                                const Vec3 legs[3] = {kv, Rp[j], Rm[j]};
                                const double Dl[3] = {ai, ai, ai};
                                const double c2p[3] = {0.0, y[i + 2], y[i]};
                                const double w2p[3] = {0.0, 0.0, 0.0};
                                S3 += rw[j] * rx[j] * source_c3(T, legs, Dl, c2p, w2p);
                            }
                            y[iC3] = S3 / (9.0 * T.H2);
                            y[iC3 + 1] = 3.0 * y[iC3];
                        }

                        // Integrate to each output and accumulate the loop functions
                        double x = std::log(set.aini);
                        for (int io : order_out) {
                            const double xend = xout[io];
                            const int nsteps = std::max(1, int(std::ceil((xend - x) / set.dloga)));
                            const double h = (xend - x) / nsteps;
                            for (int st = 0; st < nsteps and h > 0.0; st++) {
                                rhs(x, y, k1v);
                                for (int i = 0; i < nvar; i++)
                                    yt[i] = y[i] + 0.5 * h * k1v[i];
                                rhs(x + 0.5 * h, yt, k2v);
                                for (int i = 0; i < nvar; i++)
                                    yt[i] = y[i] + 0.5 * h * k2v[i];
                                rhs(x + 0.5 * h, yt, k3v);
                                for (int i = 0; i < nvar; i++)
                                    yt[i] = y[i] + h * k3v[i];
                                rhs(x + h, yt, k4v);
                                for (int i = 0; i < nvar; i++)
                                    y[i] += h / 6.0 * (k1v[i] + 2.0 * k2v[i] + 2.0 * k3v[i] + k4v[i]);
                                x += h;
                            }
                            x = xend;

                            // Accumulate (vectors in h/Mpc from here). The second order displacement of a pair (p1, p2)
                            // per D1 D2 is L2 = (P/P^2) l2 + tau u (u the transverse direction, tau = w / (D1 D2))
                            auto & L = res[io];
                            const auto & P = PL[io];
                            const double Pp = P(p);
                            if (Pp == 0.0)
                                continue;
                            const double Dk = y[iDk], Dp = y[iDp];
                            const double wp = pw[ip] * p / k / (4.0 * M_PI * M_PI);
                            const double Fp = -p * p;
                            const double c2u = c_over_H0_mpch * c_over_H0_mpch; // w in units of (Mpc/h)^2
                            const Vec3 kh{0.0, 0.0, k};
                            const Vec3 zhat{0.0, 0.0, 1.0};
                            auto scale = [](const Vec3 & v, double f) { return Vec3{v[0] * f, v[1] * f, v[2] * f}; };
                            for (int j = 0; j < nQ; j++) {
                                const int i = iQ + 6 * j;
                                const double s = qx[j];
                                const double Ps = P(s);
                                if (Ps == 0.0)
                                    continue;
                                const double Ds = y[i + 2];
                                const double l2 = y[i] / (Dp * Ds);
                                const double tau = y[i + 4] * c2u / (Dp * Ds);
                                const Vec3 p1 = scale(Qp1[j], 1.0 / c_over_H0_mpch),
                                           p2 = scale(Qp2[j], 1.0 / c_over_H0_mpch);
                                const Vec3 u = transverse_dir(p1, p2);
                                Vec3 L2 = scale(kh, l2 / (k * k));
                                for (int d = 0; d < 3; d++)
                                    L2[d] += tau * u[d];
                                const double w = wp * qw[j] * s * Pp * Ps;
                                const double mu12 = dot3(p1, p2) / (p * s);
                                // A22: (1/2) L2 L2: trace and khat khat
                                const double t22 = 0.5 * dot3(L2, L2), kk22 = 0.5 * L2[2] * L2[2];
                                L.G22 += w * k * k * 0.5 * (3.0 * kk22 - t22);
                                L.G22A += w * k * k * 0.5 * (t22 - kk22);
                                // U20, Us2: only khat.L2 survives the angular average
                                const double l2eff = k * L2[2];
                                L.U20 += w * l2eff;
                                L.Us2 += w * l2eff * (mu12 * mu12 - 1.0 / 3.0);
                                // A10 with the delta leg p1 and the Psi1 leg p2: [p2_i L2_j + p2_j L2_i] / p2^2
                                const double t = 2.0 * dot3(p2, L2) / (s * s) * k * k;
                                const double kk = 2.0 * p2[2] * L2[2] / (s * s) * k * k;
                                L.A10Qt += w * t;
                                L.A10Qkk += w * kk;
                                L.A10QtF += w * Fp * t;
                                L.A10QkkF += w * Fp * kk;
                                // W: p1_i p2_j L2_k / (p1^2 p2^2): khat khat khat and the symmetrized khat delta (times
                                // k^3)
                                const double k3 = k * k * k / (p * p * s * s);
                                L.WQa += w * p1[2] * p2[2] * L2[2] * k3;
                                L.WQb +=
                                    w * (p1[2] * dot3(p2, L2) + p2[2] * dot3(p1, L2) + L2[2] * dot3(p1, p2)) / 3.0 * k3;
                            }
                            for (int j = 0; j < nR; j++) {
                                const int i = iR + 8 * j;
                                const double s = rx[j];
                                const double l2 = y[i] / (Dk * Dp);
                                const double tau = y[i + 4] * c2u / (Dk * Dp);
                                const Vec3 pv = scale(Rp[j], 1.0 / c_over_H0_mpch);
                                const Vec3 sv = add3(kh, pv);
                                const Vec3 u = transverse_dir(kh, pv);
                                Vec3 L2 = scale(sv, l2 / (s * s));
                                for (int d = 0; d < 3; d++)
                                    L2[d] += tau * u[d];
                                const double w = wp * rw[j] * s * Pp;
                                // A10: T_R1 (delta leg k, Psi1 leg p): [p_i L2_j + p_j L2_i] / p^2; T_R2 (delta leg p,
                                // Psi1 leg k): [k_i L2_j + k_j L2_i] / k^2 (trace = khat khat)
                                L.A10R1t += w * 2.0 * dot3(pv, L2) / (p * p) * k * k;
                                L.A10R1kk += w * 2.0 * pv[2] * L2[2] / (p * p) * k * k;
                                L.A10R2t += w * 2.0 * dot3(kh, L2);
                                L.A10R2tF += w * Fp * 2.0 * dot3(kh, L2);
                                // U11: g = int P(p) khat.L2
                                L.g += w * L2[2];
                                L.gF += w * Fp * L2[2];
                                // W: k_i p_j L2_k / (k^2 p^2) (times k^3)
                                const double k3 = k / (p * p);
                                L.WRa += w * k * pv[2] * L2[2] * k3;
                                L.WRb +=
                                    w * (k * dot3(pv, L2) + pv[2] * dot3(kh, L2) + L2[2] * dot3(kh, pv)) / 3.0 * k3;
                                (void)zhat;
                            }
                            if (do_R) {
                                // l3 = c3 / (D(k) D(p)^2) integrated over the nodes
                                L.I3 += wp * Pp * y[iC3] / (Dk * Dp * Dp);
                            }
                        }
                    }
                    return res;
                }
            } // namespace detail

            /// Settings of the modified gravity loop integrals
            using MGOneLoopSettings = detail::MGIntegrationSettings;

            //=====================================================================================================
            /// @brief The linear growth factor D(k,a) of the model (normalized to D = a at early times, k in h/Mpc)
            //=====================================================================================================
            inline double
            mg_linear_growth(const MGModel & model, double k, double a, double aini = 1e-3, double dloga = 0.005) {
                using namespace detail;
                const double kH = k * c_over_H0_mpch;
                double x = std::log(aini), y[2] = {aini, aini};
                const int n = std::max(1, int(std::ceil((std::log(a) - x) / dloga)));
                const double h = (std::log(a) - x) / n;
                auto f = [&](double xx, const double * Y, double * dY) {
                    MGTime T;
                    T.set(model, std::exp(xx));
                    dY[0] = Y[1];
                    dY[1] = -(2.0 + T.dlogHdx) * Y[1] + T.A(kH) / T.H2 * Y[0];
                };
                double k1[2], k2[2], k3[2], k4[2], yt[2];
                for (int s = 0; s < n; s++) {
                    f(x, y, k1);
                    for (int i = 0; i < 2; i++)
                        yt[i] = y[i] + 0.5 * h * k1[i];
                    f(x + 0.5 * h, yt, k2);
                    for (int i = 0; i < 2; i++)
                        yt[i] = y[i] + 0.5 * h * k2[i];
                    f(x + 0.5 * h, yt, k3);
                    for (int i = 0; i < 2; i++)
                        yt[i] = y[i] + h * k3[i];
                    f(x + h, yt, k4);
                    for (int i = 0; i < 2; i++)
                        y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
                    x += h;
                }
                return y[0];
            }

            /// The loop functions of the modified gravity prediction on a grid in k for several output times
            struct MGLoopTables {
                std::vector<double> kgrid;                            // log spaced below and above the cut
                int n1{0};                                            // number of grid points below the cut
                std::vector<std::vector<detail::MGLoopFunctions>> lf; // [ik][output]
                std::vector<double> aout;
            };

            //=====================================================================================================
            /// @brief The loop functions of oneloop_hybrid_bias_spectra_mg for several output times at once (the
            /// expensive part; the kernels are integrated once from early times through all the outputs)
            //=====================================================================================================
            inline MGLoopTables mg_loop_tables(const MGModel & model,
                                               const std::vector<double> & aout,
                                               const std::vector<std::function<double(double)>> & pofk,
                                               const std::function<double(double)> & W,
                                               const ZeldovichOptions & options,
                                               const MGOneLoopSettings & settings = {}) {
                using namespace detail;
                if (options.kmax_integration <= options.kmin_integration)
                    throw std::runtime_error("[mg_loop_tables] Set kmax_integration (the cutoff)");
                if (aout.size() != pofk.size())
                    throw std::runtime_error("[mg_loop_tables] Need one P(k) per output");
                const int nout = int(aout.size());
                const double kmin = options.kmin_integration, kcut = options.kmax_integration;
                std::vector<std::function<double(double)>> PW(nout);
                for (int io = 0; io < nout; io++) {
                    PW[io] = [&, io](double k) {
                        if (k < kmin or k > kcut)
                            return 0.0;
                        const double w = W(k);
                        return pofk[io](k) * w * w;
                    };
                }
                MGLoopTables tab;
                tab.aout = aout;
                const int n1 = std::max(8, int(settings.nk_loop * 0.75)), n2 = std::max(4, settings.nk_loop - n1);
                tab.n1 = n1;
                const double k0 = kmin, k1 = kcut * (1.0 - 1e-6), k2 = kcut * (1.0 + 1e-6), k3 = 2.0 * kcut;
                for (int i = 0; i < n1; i++)
                    tab.kgrid.push_back(k0 * std::pow(k1 / k0, i / double(n1 - 1)));
                for (int i = 0; i < n2; i++)
                    tab.kgrid.push_back(k2 * std::pow(k3 / k2, i / double(n2 - 1)));
                const int ng = int(tab.kgrid.size());
                tab.lf.resize(ng);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int i = 0; i < ng; i++)
                    tab.lf[i] = mg_loop_functions_at_k(model, tab.kgrid[i], aout, PW, kmin, kcut, settings);
                return tab;
            }

            //=====================================================================================================
            /// @brief The one-loop prediction for output number io of the tables (pofk the linear P(k) at that output,
            /// the same as given to mg_loop_tables)
            //=====================================================================================================
            inline std::vector<std::vector<double>>
            oneloop_hybrid_bias_spectra_mg_from_tables(const MGLoopTables & tab,
                                                       int io,
                                                       const std::function<double(double)> & pofk,
                                                       const std::function<double(double)> & W,
                                                       const std::vector<double> & kvalues,
                                                       const ZeldovichOptions & options,
                                                       std::vector<std::string> * names = nullptr,
                                                       std::vector<std::array<int, 2>> * pairs = nullptr) {
                using namespace detail;
                if (options.noperators < 1 or options.noperators > 5)
                    throw std::runtime_error("[oneloop_hybrid_bias_spectra_mg] noperators must be 1, ..., 5");
                if (io < 0 or io >= int(tab.aout.size()))
                    throw std::runtime_error("[oneloop_hybrid_bias_spectra_mg] No such output in the tables");
                const double kmin = options.kmin_integration, kcut = options.kmax_integration;
                auto P = [&](double k) {
                    if (k < kmin or k > kcut)
                        return 0.0;
                    const double w = W(k);
                    return pofk(k) * w * w;
                };
                const auto & kgrid = tab.kgrid;
                const auto & lf = tab.lf;
                const int ng = int(kgrid.size()), n1 = tab.n1;
                LoopFill fill = [&](double k, double * G, double & zerolag_integrand) {
                    zerolag_integrand = 0.0;
                    if (k <= kgrid.front() or k >= kgrid.back())
                        return;
                    // Cubic (4-point Lagrange) interpolation in log k within the part of the grid below or above the
                    // cut
                    const int i =
                        std::min(ng - 2, int(std::upper_bound(kgrid.begin(), kgrid.end(), k) - kgrid.begin()) - 1);
                    if (i == n1 - 1)
                        return; // in the gap at the cut
                    const int lo = (i < n1) ? 0 : n1, hi = (i < n1) ? n1 - 1 : ng - 1;
                    const int i0 = std::max(lo, std::min(hi - 3, i - 1));
                    double wl[4];
                    const double lk = std::log(k);
                    for (int a = 0; a < 4; a++) {
                        wl[a] = 1.0;
                        for (int b = 0; b < 4; b++)
                            if (b != a)
                                wl[a] *= (lk - std::log(kgrid[i0 + b])) /
                                         (std::log(kgrid[i0 + a]) - std::log(kgrid[i0 + b]));
                    }
                    auto I = [&](double MGLoopFunctions::* m) {
                        double v = 0.0;
                        for (int a = 0; a < 4; a++)
                            v += wl[a] * (lf[i0 + a][io].*m);
                        return v;
                    };
                    const double Pk = P(k), Fk = -k * k;
                    G[G13] = 0.5 * Pk * I(&MGLoopFunctions::I3);
                    G[GU3] = -G[G13];
                    G[GU3N] = Fk * G[GU3];
                    G[G22] = I(&MGLoopFunctions::G22);
                    G[G22A] = I(&MGLoopFunctions::G22A);
                    G[GU20] = -I(&MGLoopFunctions::U20);
                    G[GUS2] = -I(&MGLoopFunctions::Us2);
                    // W112 = alpha (khat delta + perm) + beta khat khat khat (times k^3)
                    const double ca = 3.0 * (2.0 * I(&MGLoopFunctions::WQa) - 4.0 * Pk * I(&MGLoopFunctions::WRa));
                    const double cb = 3.0 * (2.0 * I(&MGLoopFunctions::WQb) - 4.0 * Pk * I(&MGLoopFunctions::WRb));
                    const double al = 0.5 * (cb - ca), be = ca - 3.0 * al;
                    G[GT] = -be / 3.0;
                    G[GVS] = al + be / 5.0;
                    // A10 = T_R1 - T_Q - T_R2 = a delta + b khat khat (times k^2)
                    auto a10 = [&](double t, double kk, double & GA, double & GB) {
                        const double a = 0.5 * (t - kk), b = 0.5 * (3.0 * kk - t);
                        GA = -0.5 * (a + b / 3.0);
                        GB = -b / 6.0;
                    };
                    const double R1t = I(&MGLoopFunctions::A10R1t), R1kk = I(&MGLoopFunctions::A10R1kk);
                    const double R2t = I(&MGLoopFunctions::A10R2t), R2tF = I(&MGLoopFunctions::A10R2tF);
                    a10(Pk * (R1t - R2t) - I(&MGLoopFunctions::A10Qt),
                        Pk * (R1kk - R2t) - I(&MGLoopFunctions::A10Qkk),
                        G[G10A],
                        G[G10B]);
                    a10(Fk * Pk * R1t - Pk * R2tF - I(&MGLoopFunctions::A10QtF),
                        Fk * Pk * R1kk - Pk * R2tF - I(&MGLoopFunctions::A10QkkF),
                        G[G10AN],
                        G[G10BN]);
                    // U11
                    const double g = I(&MGLoopFunctions::g), gF = I(&MGLoopFunctions::gF);
                    G[GU11] = -2.0 * k * Pk * g;
                    G[GU11DN] = -k * (Pk * gF + Fk * Pk * g);
                    G[GU11NN] = -2.0 * k * Fk * Pk * gF;
                    // X10(q = 0) = 0
                    zerolag_integrand = G[G10A];
                };
                return oneloop_from_loop_functions(pofk, W, kvalues, options, fill, names, pairs);
            }

            //=====================================================================================================
            /// @brief The one-loop LPT prediction of the hybrid bias spectra in modified gravity with the full
            /// scale-dependent kernels (see the top of this file), for several output times at once
            ///
            /// @param[in] model The gravity model (mgmodel_gr, mgmodel_mu, mgmodel_fofr, mgmodel_dgp)
            /// @param[in] aout The scale factors of the outputs
            /// @param[in] pofk The linear power spectrum P(k, a) of the model at each output (k in h/Mpc). The
            /// Lagrangian operators are built from this field (as in a simulation that uses the linear field of the
            /// model)
            /// @param[in] W The filter of the linear field (operators and displacement), e.g. a sharp cut
            /// @param[in] kvalues The k-values to compute the spectra at
            /// @param[in] options As for oneloop_hybrid_bias_spectra (kmax_integration = the cut must be set)
            /// @param[in] settings The accuracy settings of the loop integrals
            /// @param[out] names, pairs As for zeldovich_hybrid_bias_spectra
            ///
            /// @return result[output][pair][ik]
            //=====================================================================================================
            inline std::vector<std::vector<std::vector<double>>>
            oneloop_hybrid_bias_spectra_mg(const MGModel & model,
                                           const std::vector<double> & aout,
                                           const std::vector<std::function<double(double)>> & pofk,
                                           const std::function<double(double)> & W,
                                           const std::vector<double> & kvalues,
                                           const ZeldovichOptions & options,
                                           const MGOneLoopSettings & settings = {},
                                           std::vector<std::string> * names = nullptr,
                                           std::vector<std::array<int, 2>> * pairs = nullptr) {
                const int nout = int(aout.size());
                if (nout == 0 or kvalues.empty())
                    return std::vector<std::vector<std::vector<double>>>(nout);
                const auto tab = mg_loop_tables(model, aout, pofk, W, options, settings);
                std::vector<std::vector<std::vector<double>>> result(nout);
                for (int io = 0; io < nout; io++)
                    result[io] = oneloop_hybrid_bias_spectra_mg_from_tables(tab,
                                                                            io,
                                                                            pofk[io],
                                                                            W,
                                                                            kvalues,
                                                                            options,
                                                                            names,
                                                                            pairs);
                return result;
            }

        } // namespace HYBRIDBIAS
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML
#endif
