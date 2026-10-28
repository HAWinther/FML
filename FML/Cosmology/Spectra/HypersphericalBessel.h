#ifndef FML_COSMOLOGY_SPECTRA_HYPERSPHERICALBESSEL_H
#define FML_COSMOLOGY_SPECTRA_HYPERSPHERICALBESSEL_H

// =============================================================================
// Hyperspherical Bessel functions Phi^nu_ell(chi): the radial eigenfunctions of the Laplacian in
// a curved FRW universe. They replace the spherical Bessel functions j_ell(k r) in the
// line-of-sight integrals when K != 0.
//
// Variables (dimensionless):
//   chi = sqrt|K| * comoving distance,   nu = q / sqrt|K|,   q^2 = k^2 + K,
// where -k^2 is the Laplacian eigenvalue and K = -Omega_K H0^2/c^2. sign = +1 (closed, K > 0),
// -1 (open, K < 0) or 0 (flat: then Phi^nu_ell(chi) = j_ell(nu chi) for any scaling of chi).
// With sin_K(chi) = sin(chi), sinh(chi) or chi and cot_K = sin_K'/sin_K, they satisfy
//   Phi'' + 2 cot_K Phi' + [nu^2 - sign - ell(ell+1)/sin_K^2] Phi = 0,
//   Phi_0 = sin(nu chi) / (nu sin_K chi),
//   Phi_1 = [sin(nu chi) cot_K / nu - cos(nu chi)] / (sin_K sqrt(nu^2 - sign)),
//   sqrt(nu^2 - sign (l+1)^2) Phi_{l+1} = (2l + 1) cot_K Phi_l - sqrt(nu^2 - sign l^2) Phi_{l-1},
//   dPhi_l/dchi = l cot_K Phi_l - sqrt(nu^2 - sign (l+1)^2) Phi_{l+1},
// and reduce to j_ell(nu chi) when nu -> infinity with nu chi fixed. In a closed universe nu must
// be an integer >= 3 (as for the perturbation modes), chi is in [0, pi], and Phi_ell = 0 for
// ell >= nu.
//
// Method (Kosowsky 1998; Tram 2017): upward recurrence from Phi_0, Phi_1 in the oscillating
// region ell < ell_tp(chi), where ell_tp(ell_tp + 1) = (nu^2 - sign) sin_K^2 is the turning
// point, and Miller's downward recurrence above it, matched to the upward solution. In a closed
// universe the downward recurrence starts exactly at ell = nu - 1 (Phi_nu = 0), so it is exact.
//
// Usage:
//   HypersphericalBessel hb(-1, nu);          // open universe, mode nu
//   std::vector<double> phi;
//   hb.column(chi, lmax, phi);                // phi[ell] = Phi^nu_ell(chi), ell = 0..lmax
// =============================================================================

#include <vector>

namespace FML::COSMOLOGY {

    class HypersphericalBessel {
      public:
        // lmax_hint: the recurrence coefficients sqrt(nu^2 - sign l^2) and their inverses are
        // tabulated up to this ell (plus the Miller margin) for speed; larger ell still work.
        HypersphericalBessel(int sign, double nu, int lmax_hint = 0);

        // Phi_0 .. Phi_lmax at chi; out is resized to lmax + 1
        void column(double chi, int lmax, std::vector<double> & out) const;
        // The same plus the derivatives dPhi_ell/dchi, ell = 0..lmax
        void column_with_derivative(double chi, int lmax, std::vector<double> & phi, std::vector<double> & dphi) const;
        // Single value (computes the whole column up to ell; use column() for many ell)
        double operator()(int ell, double chi) const;

        // sin_K(chi), cot_K(chi) for this geometry
        double sinK(double chi) const;
        double cotK(double chi) const;
        // Turning point ell_tp(chi): Phi_ell is oscillating for ell < ell_tp and decays rapidly above
        double turning_point(double chi) const;

        int sign() const { return sgn; }
        double nu() const { return nu_; }

      private:
        int sgn;
        double nu_;
        // sqrt(nu^2 - sign l^2), the coefficient of the recurrence (0 for l >= nu when closed)
        double coef(int l) const;
        std::vector<double> coef_tab, inv_coef_tab; // tabulated coef(l) and 1/coef(l)
        double coef_fast(int l) const { return l < int(coef_tab.size()) ? coef_tab[l] : coef(l); }
        double inv_coef_fast(int l) const { return l < int(inv_coef_tab.size()) ? inv_coef_tab[l] : 1.0 / coef(l); }
    };

} // namespace FML::COSMOLOGY

#endif
