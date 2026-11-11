#ifndef FML_COSMOLOGY_SPECTRA_EISENSTEINHU_H
#define FML_COSMOLOGY_SPECTRA_EISENSTEINHU_H

#include <cmath>

//=====================================================================================================================
///
/// The Eisenstein & Hu (1998, arXiv:astro-ph/9709112) fits to the matter transfer function of cold dark matter and
/// baryons (no neutrinos): with the baryon acoustic oscillations (eqs. 16-24) and the no-wiggle fit with zero baryon
/// oscillations (eqs. 26-31). Equal to colossus' eisenstein98 and eisenstein98_zb to rounding
/// (PerturbationTheory/tests/test_nowiggle.cpp). Header only, no dependencies: the one copy in FML, used for smooth
/// shapes and extrapolations (Cosmology/Spectra, HMcode-2020, FieldLevel, PerturbationTheory/NoWiggle.h).
///
/// k in h/Mpc. T(k -> 0) = 1.
///
//=====================================================================================================================
namespace FML::COSMOLOGY {

    /// The cosmology of the Eisenstein & Hu fits
    struct EisensteinHuParameters {
        double Omega_m{0.3111}; // total matter (CDM + baryons; the fits have no neutrinos)
        double Omega_b{0.049};
        double h{0.6766};
        double TCMB{2.7255};
        double n_s{0.9665}; // for the power spectrum shapes
    };

    /// The Eisenstein & Hu (1998) transfer function with baryon oscillations (eqs. 16-24), k in h/Mpc
    inline double eisenstein_hu_transfer(double k, const EisensteinHuParameters & p) {
        const double kM = k * p.h; // 1/Mpc
        const double th = p.TCMB / 2.7, om = p.Omega_m * p.h * p.h, ob = p.Omega_b * p.h * p.h;
        const double fb = ob / om, fc = 1 - fb;
        const double zeq = 2.50e4 * om / std::pow(th, 4), keq = 7.46e-2 * om / (th * th);
        const double b1 = 0.313 * std::pow(om, -0.419) * (1 + 0.607 * std::pow(om, 0.674));
        const double b2 = 0.238 * std::pow(om, 0.223);
        const double zd = 1291 * std::pow(om, 0.251) / (1 + 0.659 * std::pow(om, 0.828)) * (1 + b1 * std::pow(ob, b2));
        auto R = [&](double z) { return 31.5 * ob / std::pow(th, 4) * 1e3 / z; };
        const double Rd = R(zd), Req = R(zeq);
        const double s = 2. / (3 * keq) * std::sqrt(6 / Req) *
                         std::log((std::sqrt(1 + Rd) + std::sqrt(Rd + Req)) / (1 + std::sqrt(Req)));
        const double ksilk = 1.6 * std::pow(ob, 0.52) * std::pow(om, 0.73) * (1 + std::pow(10.4 * om, -0.95));
        const double q = kM / (13.41 * keq);
        const double a1 = std::pow(46.9 * om, 0.670) * (1 + std::pow(32.1 * om, -0.532));
        const double a2 = std::pow(12.0 * om, 0.424) * (1 + std::pow(45.0 * om, -0.582));
        const double alpha_c = std::pow(a1, -fb) * std::pow(a2, -fb * fb * fb);
        const double bb1 = 0.944 / (1 + std::pow(458 * om, -0.708)), bb2 = std::pow(0.395 * om, -0.0266);
        const double beta_c = 1 / (1 + bb1 * (std::pow(fc, bb2) - 1));
        auto T0 = [&](double alpha, double beta) {
            const double L = std::log(M_E + 1.8 * beta * q);
            const double C = 14.2 / alpha + 386 / (1 + 69.9 * std::pow(q, 1.08));
            return L / (L + C * q * q);
        };
        const double ks = kM * s;
        const double f = 1 / (1 + std::pow(ks / 5.4, 4));
        const double Tc = f * T0(1, beta_c) + (1 - f) * T0(alpha_c, beta_c);
        const double y = (1 + zeq) / (1 + zd), sy = std::sqrt(1 + y);
        const double G = y * (-6 * sy + (2 + 3 * y) * std::log((sy + 1) / (sy - 1)));
        const double alpha_b = 2.07 * keq * s * std::pow(1 + Rd, -0.75) * G;
        const double beta_node = 8.41 * std::pow(om, 0.435);
        const double beta_b = 0.5 + fb + (3 - 2 * fb) * std::sqrt(std::pow(17.2 * om, 2) + 1);
        const double stilde = s / std::cbrt(1 + std::pow(beta_node / ks, 3));
        const double x = kM * stilde, j0 = x < 1e-4 ? 1 - x * x / 6 : std::sin(x) / x;
        const double Tb = (T0(1, 1) / (1 + std::pow(ks / 5.2, 2)) +
                           alpha_b / (1 + std::pow(beta_b / ks, 3)) * std::exp(-std::pow(kM / ksilk, 1.4))) *
                          j0;
        return fb * Tb + fc * Tc;
    }

    /// The Eisenstein & Hu (1998) no-wiggle transfer function (eqs. 26-31), k in h/Mpc
    inline double eisenstein_hu_transfer_nowiggle(double k, const EisensteinHuParameters & p) {
        const double kM = k * p.h;
        const double th = p.TCMB / 2.7, om = p.Omega_m * p.h * p.h, ob = p.Omega_b * p.h * p.h, fb = ob / om;
        const double s = 44.5 * std::log(9.83 / om) / std::sqrt(1 + 10 * std::pow(ob, 0.75));
        const double alpha = 1 - 0.328 * std::log(431 * om) * fb + 0.38 * std::log(22.3 * om) * fb * fb;
        const double gamma = p.Omega_m * p.h * (alpha + (1 - alpha) / (1 + std::pow(0.43 * kM * s, 4)));
        const double q = k * th * th / gamma;
        const double L = std::log(2 * M_E + 1.8 * q), C = 14.2 + 731 / (1 + 62.5 * q);
        return L / (L + C * q * q);
    }

    /// The power spectrum shapes k^n_s T(k)^2 (with and without the wiggles; arbitrary normalization), k in h/Mpc
    inline double eisenstein_hu_power_shape(double k, const EisensteinHuParameters & p) {
        return std::pow(k, p.n_s) * std::pow(eisenstein_hu_transfer(k, p), 2);
    }
    inline double eisenstein_hu_power_shape_nowiggle(double k, const EisensteinHuParameters & p) {
        return std::pow(k, p.n_s) * std::pow(eisenstein_hu_transfer_nowiggle(k, p), 2);
    }

} // namespace FML::COSMOLOGY

#endif
