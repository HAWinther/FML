#ifndef FML_COSMOLOGY_SPECTRA_LENSING_H
#define FML_COSMOLOGY_SPECTRA_LENSING_H

// =============================================================================
// Weak gravitational lensing of the CMB power spectra (full-sky correlation-function method of
// Challinor & Lewis 2005, astro-ph/0502425, as implemented in CLASS).
//
// Given the unlensed C_ell^TT, EE, TE (BB) and the lensing potential C_ell^phiphi for
// ell = 0..L, the lensed correlation functions are
//   xi~(beta)  = sum_l (2l+1)/(4pi) C_l^TT [X000^2 d^l_00 + 8/(l(l+1)) C_gl,2 X'000^2 d^l_1-1
//                                           + C_gl,2^2 (X'000^2 d^l_00 + X220^2 d^l_2-2)]
// and similarly xi~_+ (EE+BB, d_22), xi~_- (EE-BB, d_2-2), xi~_X (TE, d_20), with
//   C_gl(beta)   = sum_l (2l+1)/(4pi) l(l+1) C_l^phiphi d^l_11(beta),
//   C_gl,2(beta) = sum_l (2l+1)/(4pi) l(l+1) C_l^phiphi d^l_1-1(beta),
//   sigma^2(beta) = C_gl(0) - C_gl(beta),   X000 = exp(-l(l+1) sigma^2 / 4), ...
// (second order in C_gl,2, optional first order in C_gl). The lensed spectra follow from
//   C~_l^TT = 2 pi int dmu xi~(mu) d^l_00(mu)   (Gauss-Legendre in mu = cos beta),
//   C~^EE + C~^BB with d_22, C~^EE - C~^BB with d_2-2, C~^TE with d_20.
// Only the difference xi~ - xi is transformed and added to the unlensed spectra, which keeps the
// quadrature error small. Lensing converts E into B, so C~_l^BB is produced even if C_l^BB = 0.
//
// The Wigner d-functions d^l_{m m'}(beta) are computed by the standard three-term recurrence in l.
//
// Usage:
//   Lensing::Result r = Lensing::lens(cl_tt, cl_ee, cl_te, cl_bb, cl_pp, lmax_out);
// All C_l are dimensionless, indexed by ell (entries 0 and 1 are ignored), and must extend to
// L = cl_pp.size() - 1 > lmax_out; L - lmax_out ~ 500 gives ~1e-3 accuracy at lmax_out = 2500.
// cl_ee, cl_te, cl_bb may be empty (temperature only / no primordial B).
// =============================================================================

#include <vector>

namespace FML::COSMOLOGY {

    namespace Lensing {

        struct Options {
            int c2_order = 2; // order in C_gl,2 (0, 1 or 2)
            int c0_order = 0; // 1: include the first order in C_gl (more accurate, slower)
            // Gauss-Legendre points: L + n_mu_extra. N = L (as in CLASS) treats the input as band-limited at L;
            // more points resolve the sharp truncation of the input C_l at L, which changes the lensed
            // spectra near lmax_out unless L - lmax_out is large.
            int n_mu_extra = 0;
        };

        struct Result {
            std::vector<double> tt, ee, te, bb; // lensed C_l for l = 0..lmax_out (0 and 1 set to 0)
        };

        Result lens(const std::vector<double> & cl_tt,
                    const std::vector<double> & cl_ee,
                    const std::vector<double> & cl_te,
                    const std::vector<double> & cl_bb,
                    const std::vector<double> & cl_pp,
                    int lmax_out,
                    const Options & opt = Options());

        // d^l_{m m'}(beta) for l = 0..lmax at mu = cos(beta), by recurrence in l (zero for l < max(|m|,|m'|))
        void wigner_d(int m, int mp, double mu, int lmax, std::vector<double> & d);
    } // namespace Lensing

} // namespace FML::COSMOLOGY

#endif
