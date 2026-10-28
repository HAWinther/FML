#ifndef FML_COSMOLOGY_SPECTRA_TWOPOINT_H
#define FML_COSMOLOGY_SPECTRA_TWOPOINT_H

// =============================================================================
// Two-point functions: angular power spectra <-> angular correlation functions on the sphere, and
// the three-dimensional correlation function of a power spectrum. Pure numerics; PowerSpectrum
// uses these for the CMB and the matter field (see PowerSpectrum::get_cmb_correlation,
// get_xi_matter, get_angular_cl_matter).
//
// Angular correlation of a spin-0 field (CMB temperature, projected galaxy density):
//   w(theta) = sum_l (2l+1)/(4 pi) C_l B_l^2 P_l(cos theta),   B_l = exp(-l(l+1) sigma_b^2 / 2)
// with a Gaussian beam of full width at half maximum fwhm = sqrt(8 ln 2) sigma_b (0 = no beam).
//
// CMB polarization (Q, U measured along / perpendicular to the great circle joining the two points):
//   xi_+(theta) = <Q Q> + <U U> = sum_l (2l+1)/(4 pi) (C_l^EE + C_l^BB) B_l^2 d^l_22(theta)
//   xi_-(theta) = <Q Q> - <U U> = sum_l (2l+1)/(4 pi) (C_l^EE - C_l^BB) B_l^2 d^l_2-2(theta)
//   xi_X(theta) = <T Q>         = sum_l (2l+1)/(4 pi) C_l^TE B_l^2 d^l_20(theta)
// (the conventions of Challinor & Lewis 2005, also used in Lensing.h; the sign of xi_X depends on
// the convention for Q).
//
// 3D correlation function of a power spectrum P(k) (Fourier convention delta(x) = int d^3k/(2pi)^3 ...):
//   xi(r) = int dk k^2/(2 pi^2) P(k) j_0(kr) e^{-k^2 R^2} = 1/(2 pi^2 r) int dk k P(k) sin(kr) e^{-k^2 R^2}
// evaluated with Filon's method: k P(k) e^{-k^2 R^2} is interpolated linearly between the tabulated
// points and the product with sin(kr) is integrated exactly, so the step only has to resolve P(k), not
// the oscillation (any r on one table). R is an optional Gaussian smoothing scale.
//   sigma_R^2 = int dk k^2/(2 pi^2) P(k) W_TH(kR)^2,  W_TH(y) = 3 (sin y - y cos y)/y^3  (top hat)
// =============================================================================

#include <functional>
#include <vector>

namespace FML::COSMOLOGY {

    namespace TwoPoint {

        // ---------------------------------------------------------------------------
        // Sphere. cl is indexed by ell (cl[ell]); multipoles below ell_min are left out (the monopole and
        // dipole of the CMB are not cosmological). theta in radians.
        // ---------------------------------------------------------------------------
        double beam(int ell, double fwhm); // B_l for a Gaussian beam (fwhm in radians)

        std::vector<double> angular_correlation(const std::vector<double> & cl,
                                                const std::vector<double> & theta,
                                                double fwhm = 0.0,
                                                int ell_min = 2);

        struct CMBCorrelation {
            std::vector<double> theta;               // [rad]
            std::vector<double> TT, plus, minus, TQ; // C^TT, xi_+, xi_-, xi_X (units of the input C_l)
        };
        // ee, te, bb may be empty (temperature only / no B modes)
        CMBCorrelation cmb_correlation(const std::vector<double> & tt,
                                       const std::vector<double> & ee,
                                       const std::vector<double> & te,
                                       const std::vector<double> & bb,
                                       const std::vector<double> & theta,
                                       double fwhm = 0.0);

        // Inverse transform: C_l = 2 pi int_{-1}^{1} w(theta) P_l(cos theta) dcos(theta), for a function w
        // given as a callable of theta; Gauss-Legendre with n points (exact for band-limited w with
        // n > lmax(w) + l_max / 2). Used to test angular_correlation.
        std::vector<double> legendre_transform(const std::function<double(double)> & w, int l_max, int n);

        // ---------------------------------------------------------------------------
        // Three dimensions. k (increasing, > 0) and P tabulated on the same grid; units are free
        // (r in 1/[k], xi dimensionless if P is in [k]^-3).
        // ---------------------------------------------------------------------------
        std::vector<double> xi_of_r(const std::vector<double> & k,
                                    const std::vector<double> & P,
                                    const std::vector<double> & r,
                                    double R_smooth = 0.0);
        double sigma_R(const std::vector<double> & k, const std::vector<double> & P, double R);

        // ---------------------------------------------------------------------------
        // Redshift distribution of a tracer (projected galaxy/matter field). n(z) need not be normalised;
        // the projection kernel is W(chi) = n(z) dz/dchi normalised to int W dchi = 1 (as in CLASS).
        // ---------------------------------------------------------------------------
        struct RedshiftWindow {
            std::function<double(double)> n; // dN/dz (arbitrary normalisation)
            double z_min = 0.0, z_max = 0.0; // support
            double bias = 1.0;               // linear bias b: delta_g = b Delta_m
            static RedshiftWindow gaussian(double z0, double sigma_z, double bias = 1.0); // +-6 sigma
            // Top hat with Gaussian edges of width sigma_edge (sharp edges make the exact C_l noisy)
            static RedshiftWindow tophat(double z1, double z2, double sigma_edge = 0.01, double bias = 1.0);
        };
    } // namespace TwoPoint

} // namespace FML::COSMOLOGY

#endif
