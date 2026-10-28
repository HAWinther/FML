#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using namespace Constants;

    RecombinationHistory::RecombinationHistory(const BackgroundCosmology & cosmo_in, RecombinationOptions options_in)
        : cosmo(cosmo_in), options(options_in), Yp(cosmo_in.get_params().helium_fraction()) {
        reion = options.reionization && cosmo.get_params().reionization;
        if (options.x_end != 0.0)
            throw std::invalid_argument("RecombinationHistory: x_end must be 0 (tau is integrated back from today)");
        if (cosmo.get_x_start() > options.x_start || cosmo.get_x_end() < options.x_end)
            std::cerr << "RecombinationHistory: warning: x-range extends outside the background solution\n";
    }

    // =============================================================================
    // Number densities and temperature
    // =============================================================================

    double RecombinationHistory::nb_of_x(double x) const {
        const double H0 = cosmo.get_H0();
        const double rho_c0 = 3.0 * H0 * H0 / (8.0 * M_PI * G);
        return cosmo.rhoB(x) * rho_c0 / m_H;
    }

    double RecombinationHistory::nH_of_x(double x) const { return (1.0 - Yp) * nb_of_x(x); }

    double RecombinationHistory::Tb_of_x(double x) const {
        if (bool(log_Tb_spline))
            return std::exp(log_Tb_spline(x));
        return cosmo.TCMB_of_x(x);
    }

    // =============================================================================
    // Saha equilibrium (T_b = T_gamma in this regime), with F = (m_e k T / (2 pi hbar^2))^{3/2}:
    //   n_HII   n_e / n_HI   = F exp(-epsilon_0 / kT)          = S_H
    //   n_HeII  n_e / n_HeI  = 4 F exp(-chi_0 / kT)             = S_1   (statistical weights)
    //   n_HeIII n_e / n_HeII = F exp(-chi_1 / kT)               = S_2
    //   n_e = n_H x_H + n_He (x_HeII + 2 x_HeIII)
    // For given n_e: x_H = S_H/(S_H + n_e), x_HeII = r1/(1 + r1 + r1 r2), x_HeIII = r1 r2/(...),
    // r1 = S_1/n_e, r2 = S_2/n_e. n_e is found by root finding in ln n_e (the right-hand side
    // decreases monotonically with n_e).
    // =============================================================================

    RecombinationHistory::SahaResult RecombinationHistory::saha(double x, double xH_fixed) const {
        const double kT = k_b * cosmo.TCMB_of_x(x);
        const double F = std::pow(m_e * kT / (2.0 * M_PI * hbar * hbar), 1.5);
        const double S_H = F * std::exp(-epsilon_0 / kT);
        const double S_1 = 4.0 * F * std::exp(-xhi0 / kT);
        const double S_2 = F * std::exp(-xhi1 / kT);
        const double nH = nH_of_x(x), nHe = f_He() * nH;

        auto fractions = [&](double ne, SahaResult & r) {
            r.xH = xH_fixed >= 0.0 ? xH_fixed : S_H / (S_H + ne);
            if (nHe > 0.0) {
                const double r1 = S_1 / ne, r2 = S_2 / ne;
                // Divide through by the largest term to avoid overflow when r1, r2 are huge
                if (r1 * r2 > 1.0 && r2 > 1.0) {
                    const double d = 1.0 / (r1 * r2) + 1.0 / r2 + 1.0;
                    r.xHeII = (1.0 / r2) / d;
                    r.xHeIII = 1.0 / d;
                } else {
                    const double d = 1.0 + r1 + r1 * r2;
                    r.xHeII = r1 / d;
                    r.xHeIII = r1 * r2 / d;
                }
            } else {
                r.xHeII = r.xHeIII = 0.0;
            }
            return nH * r.xH + nHe * (r.xHeII + 2.0 * r.xHeIII);
        };

        SahaResult r{};
        const double ne_max = nH + 2.0 * nHe;
        const double ne_min = 1e-30 * nH;
        auto g = [&](double log_ne) {
            const double ne = std::exp(log_ne);
            return ne - fractions(ne, r);
        };
        double ne;
        if (g(std::log(ne_min)) >= 0.0) {
            ne = ne_min; // essentially neutral
        } else {
            ne = std::exp(Utils::find_root(g, std::log(ne_min), std::log(ne_max) + 1e-12, 1e-14, 1e-13));
        }
        fractions(ne, r);
        r.Xe = ne / nH;
        return r;
    }

    double RecombinationHistory::Xe_saha_of_x(double x) const { return saha(x).Xe; }

    // =============================================================================
    // Reionization (CAMB parametrization)
    // =============================================================================

    void RecombinationHistory::reionize(double x, double Xe_rec, double dXe_rec, double & Xe, double & dXe) const {
        if (!reion) {
            Xe = Xe_rec;
            dXe = dXe_rec;
            return;
        }
        const auto & p = cosmo.get_params();
        const double fHe = f_He();
        const double z = std::exp(-x) - 1.0;

        // Hydrogen + first helium ionization: tanh in y = (1+z)^{3/2}
        const double y = std::pow(1.0 + z, 1.5), y_re = std::pow(1.0 + p.z_reion, 1.5);
        const double dy = 1.5 * std::sqrt(1.0 + p.z_reion) * p.delta_z_reion;
        const double u = (y_re - y) / dy;
        const double TH = 0.5 * (1.0 + std::tanh(u));
        const double dTH = 0.5 / std::pow(std::cosh(std::min(u, 300.0)), 2) * 1.5 * y / dy; // du/dx = 1.5 y / dy

        // Second helium ionization: tanh in z
        const double uHe = (p.z_He_reion - z) / p.delta_z_He_reion;
        const double THe = 0.5 * (1.0 + std::tanh(uHe));
        const double dTHe =
            0.5 / std::pow(std::cosh(std::min(std::fabs(uHe), 300.0)), 2) * (1.0 + z) / p.delta_z_He_reion;

        Xe = Xe_rec + (1.0 + fHe - Xe_rec) * TH + fHe * THe;
        dXe = dXe_rec * (1.0 - TH) + (1.0 + fHe - Xe_rec) * dTH + fHe * dTHe;
    }

    // =============================================================================
    // Peebles:
    //   dX_e/dx = C_r/H [ beta (1 - X_e) - n_H alpha2 X_e^2 ]
    //   alpha2  = F (64 pi / sqrt(27 pi)) (alpha hbar / (m_e c))^2 c sqrt(eps0/kT) phi2,  phi2 = 0.448 ln(eps0/kT)
    //   beta    = alpha2 (m_e kT / (2 pi hbar^2))^{3/2} exp(-eps0/kT)
    //   beta2   = beta exp(3 eps0 / (4 kT))   (combined exponent to avoid overflow)
    //   C_r     = (L_2s1s + L_alpha) / (L_2s1s + L_alpha + beta2)
    //   L_alpha = H (3 eps0 / (hbar c))^3 / ((8 pi)^2 n_1s),  n_1s = (1 - X_e) n_H
    // =============================================================================

    double RecombinationHistory::dXedx_peebles(double x, double Xe, double Tb) const {
        const double kT = k_b * Tb;
        const double y = epsilon_0 / kT;
        const double H = cosmo.H_of_x(x);
        const double nH = nH_of_x(x);

        const double phi2 = 0.448 * std::log(y);
        const double lambda_e = alpha * hbar / (m_e * c); // alpha * reduced Compton wavelength
        const double alpha2 = options.peebles_fudge * 64.0 * M_PI / std::sqrt(27.0 * M_PI) * lambda_e * lambda_e * c *
                              std::sqrt(y) * phi2;
        const double saha_factor = std::pow(m_e * kT / (2.0 * M_PI * hbar * hbar), 1.5);
        const double beta = alpha2 * saha_factor * std::exp(-y);
        const double beta2 = alpha2 * saha_factor * std::exp(-0.25 * y);

        const double n1s = std::max(1.0 - Xe, 1e-30) * nH;
        const double k_alpha = 3.0 * epsilon_0 / (hbar * c);
        const double Lambda_alpha = H * k_alpha * k_alpha * k_alpha / (64.0 * M_PI * M_PI * n1s);
        const double Cr = (lambda_2s1s + Lambda_alpha) / (lambda_2s1s + Lambda_alpha + beta2);

        return Cr / H * (beta * (1.0 - Xe) - nH * alpha2 * Xe * Xe);
    }

    // =============================================================================
    // Baryon temperature (Compton heating by CMB photons + adiabatic cooling):
    //   dT_b/dx = -2 T_b + (Gamma_C / H) (T_gamma - T_b)
    //   Gamma_C = (8 sigma_T a_r T_gamma^4 / (3 m_e c)) X_e / (1 + f_He + X_e),
    //   a_r = pi^2 k_b^4 / (15 hbar^3 c^3),  f_He = n_He / n_H (all helium atoms, also without helium ionization)
    // =============================================================================

    double RecombinationHistory::dTbdx(double x, double Xe, double Tb) const {
        const double Tg = cosmo.TCMB_of_x(x);
        const double a_r = M_PI * M_PI * std::pow(k_b, 4) / (15.0 * std::pow(hbar * c, 3));
        const double Gamma_C = 8.0 * sigma_T * a_r * std::pow(Tg, 4) / (3.0 * m_e * c) * Xe / (1.0 + f_He_atoms() + Xe);
        return -2.0 * Tb + Gamma_C / cosmo.H_of_x(x) * (Tg - Tb);
    }


    // =============================================================================
    // RECFAST 1.5 (Seager, Sasselov & Scott 1999; Wong, Moss & Scott 2008; Rubino-Martin et al. 2010), with the
    // constants and fitting formulae of the RECFAST code (as in CAMB and CLASS). Variables x_H = n_HII/n_H,
    // x_He = n_HeII/n_He, the total X_e = x_H + f_He x_He; z-derivatives are converted with d/dx = -(1+z) d/dz.
    //   dx_H/dz = [X_e x_H n_H a_H - b_H (1 - x_H) e^{-h nu_Lya/kT}] C_H / (H (1+z))
    //   C_H = (1 + K_H L_2s n_1s) / (1/F_H + K_H L_2s n_1s / F_H + K_H b_H n_1s),  n_1s = n_H (1 - x_H),
    //   K_H = lambda_Lya^3/(8 pi H) [1 + Gaussian corrections in ln(1+z)],  a_H = 1e-19 F_H... (Pequignot et al.)
    // and the analogous He I singlet equation with the Sobolev escape probability of the 2^1P line, its
    // reduction by hydrogen continuum absorption (fudge F_He) and the triplet (2^3P) channel.
    // =============================================================================

    namespace recfast {
        constexpr double h_P = 2.0 * M_PI * hbar;
        constexpr double L_H_ion = 1.096787737e7, L_H_alpha = 8.225916453e6; // inverse wavelengths [1/m]
        constexpr double L_He1_ion = 1.98310772e7, L_He2_ion = 4.389088863e7;
        constexpr double L_He_2s = 1.66277434e7, L_He_2p = 1.71134891e7;
        constexpr double L_He_2Pt = 1.690871466e7, L_He_2St = 1.5985597526e7, L_He2St_ion = 3.8454693845e6;
        constexpr double A2P_s = 1.798287e9, A2P_t = 177.58; // Einstein A of 2^1P, 2^3P [1/s]
        constexpr double sigma_He_2Ps = 1.436289e-22,
                         sigma_He_2Pt = 1.484872e-22;            // H photoionization at those lines [m^2]
        constexpr double Lambda_H = 8.2245809, Lambda_He = 51.3; // two-photon 2s-1s rates [1/s]
        constexpr double not4 = 3.9715;                          // m_He / m_H
        constexpr double m_H_recfast = 1.673575e-27;
        // case-B fit (Pequignot, Petitjean & Boisson 1991) and He I fits (Verner & Ferland 1996, triplets)
        constexpr double a_PPB = 4.309, b_PPB = -0.6166, c_PPB = 0.6703, d_PPB = 0.5300;
        const double a_VF = std::pow(10.0, -16.744), b_VF = 0.711, T_0 = std::pow(10.0, 0.477121),
                     T_1 = std::pow(10.0, 5.114);
        const double a_trip = std::pow(10.0, -16.306), b_trip = 0.761;
        // Gaussian correction to K_H (RECFAST 1.5)
        constexpr double AGauss1 = -0.14, AGauss2 = 0.079, zGauss1 = 7.28, zGauss2 = 6.73, wGauss1 = 0.18,
                         wGauss2 = 0.33;
        // derived constants
        constexpr double CR = 2.0 * M_PI * (m_e / h_P) * (k_b / h_P); // (2 pi m_e k T/h^2)/T
        constexpr double CDB = h_P * c * (L_H_ion - L_H_alpha) / k_b; // ionization from n = 2 [K]
        constexpr double CDB_He = h_P * c * (L_He1_ion - L_He_2s) / k_b;
        constexpr double CL = h_P * c * L_H_alpha / k_b, CL_He = h_P * c * L_He_2s / k_b,
                         CL_Het = h_P * c * L_He_2St / k_b;
        constexpr double CDB_He2s2p = h_P * c * (L_He_2p - L_He_2s) / k_b;
        const double CK = std::pow(1.0 / L_H_alpha, 3) / (8.0 * M_PI),
                     CK_He = std::pow(1.0 / L_He_2p, 3) / (8.0 * M_PI);
        constexpr double x_H_trigger = 0.995, x_He_trigger = 0.995, x_He_trigger_small = 5e-9, z_switch_late = 800.0;
        constexpr double x_H_limit_KHe = 0.9999999, x_H_limit_CfHe_t = 0.99999, max_exp_boltz = 680.0;
    } // namespace recfast

    double RecombinationHistory::dxHdx_recfast(double x, double xH, double Xe, double Tb) const {
        using namespace recfast;
        const double z = std::exp(-x) - 1.0, H = cosmo.H_of_x(x), nH = nH_of_x(x);
        const double t4 = Tb / 1e4;
        const double Rdown = 1e-19 * a_PPB * std::pow(t4, b_PPB) / (1.0 + c_PPB * std::pow(t4, d_PPB));
        const double Rup = Rdown * std::pow(CR * Tb, 1.5) * std::exp(-CDB / Tb);
        const double lz = std::log(1.0 + z);
        const double K = CK / H *
                         (1.0 + AGauss1 * std::exp(-std::pow((lz - zGauss1) / wGauss1, 2)) +
                          AGauss2 * std::exp(-std::pow((lz - zGauss2) / wGauss2, 2)));
        const double n1s = nH * (1.0 - xH), F = options.recfast_fudge_H;
        const double C = (xH < x_H_trigger || z < z_switch_late) ?
                             (1.0 + K * Lambda_H * n1s) / (1.0 / F + K * Lambda_H * n1s / F + K * Rup * n1s) :
                             1.0;
        const double dxdz = (Xe * xH * nH * Rdown - Rup * (1.0 - xH) * std::exp(-CL / Tb)) * C / (H * (1.0 + z));
        return -(1.0 + z) * dxdz;
    }

    double RecombinationHistory::dxHedx_recfast(double x, double xHe, double xH, double Xe, double Tb) const {
        using namespace recfast;
        if (xHe < 1e-15)
            return 0.0;
        const double z = std::exp(-x) - 1.0, H = cosmo.H_of_x(x), nH = nH_of_x(x);
        const double fHe = f_He(), nHe = fHe * nH;
        const double sq0 = std::sqrt(Tb / T_0), sq1 = std::sqrt(Tb / T_1);
        const double Rdown_He = a_VF / (sq0 * std::pow(1.0 + sq0, 1.0 - b_VF) * std::pow(1.0 + sq1, 1.0 + b_VF));
        const double Rup_He = 4.0 * Rdown_He * std::pow(CR * Tb, 1.5) * std::exp(-CDB_He / Tb);

        int flag = 6;
        if (xHe < x_He_trigger_small || xHe > x_He_trigger)
            flag = 0;
        else if (z < z_switch_late)
            flag = 2;

        double K_He, CfHe_t = 0.0, Rdown_trip = 0.0, Rup_trip = 0.0;
        if (flag == 0) {
            K_He = CK_He / H;
        } else {
            // Sobolev escape of the 2^1P photons
            const double tau_s = A2P_s * CK_He * 3.0 * nHe * (1.0 - xHe) / H;
            const double p_s = (1.0 - std::exp(-tau_s)) / tau_s;
            K_He = 1.0 / (A2P_s * p_s * 3.0 * nHe * (1.0 - xHe));
            if (xH < x_H_limit_KHe) { // continuum opacity of neutral hydrogen (Doppler width)
                const double Doppler = c * L_He_2p * std::sqrt(2.0 * k_b * Tb / (m_H_recfast * not4 * c * c));
                const double gamma_2Ps = 3.0 * A2P_s * fHe * (1.0 - xHe) * c * c /
                                         (std::sqrt(M_PI) * sigma_He_2Ps * 8.0 * M_PI * Doppler * (1.0 - xH)) /
                                         std::pow(c * L_He_2p, 2);
                const double AHcon = A2P_s / (1.0 + 0.36 * std::pow(gamma_2Ps, options.recfast_fudge_He));
                K_He = 1.0 / ((A2P_s * p_s + AHcon) * 3.0 * nHe * (1.0 - xHe));
            }
            if (flag >= 3) { // triplets
                Rdown_trip = a_trip / (sq0 * std::pow(1.0 + sq0, 1.0 - b_trip) * std::pow(1.0 + sq1, 1.0 + b_trip));
                Rup_trip =
                    Rdown_trip * std::exp(-h_P * c * L_He2St_ion / (k_b * Tb)) * std::pow(CR * Tb, 1.5) * 4.0 / 3.0;
                const double tau_t = A2P_t * nHe * (1.0 - xHe) * 3.0 / (8.0 * M_PI * H * std::pow(L_He_2Pt, 3));
                const double p_t = (1.0 - std::exp(-tau_t)) / tau_t;
                const double CL_PSt = h_P * c * (L_He_2Pt - L_He_2St) / k_b;
                if (xH >= x_H_limit_CfHe_t) {
                    CfHe_t = A2P_t * p_t * std::exp(-CL_PSt / Tb);
                } else {
                    const double Doppler = c * L_He_2Pt * std::sqrt(2.0 * k_b * Tb / (m_H_recfast * not4 * c * c));
                    const double gamma_2Pt = 3.0 * A2P_t * fHe * (1.0 - xHe) * c * c /
                                             (std::sqrt(M_PI) * sigma_He_2Pt * 8.0 * M_PI * Doppler * (1.0 - xH)) /
                                             std::pow(c * L_He_2Pt, 2);
                    const double AHcon = A2P_t / (1.0 + 0.66 * std::pow(gamma_2Pt, 0.9)) / 3.0;
                    CfHe_t = (A2P_t * p_t + AHcon) * std::exp(-CL_PSt / Tb);
                }
                CfHe_t = CfHe_t / (Rup_trip + CfHe_t);
            }
        }
        const double Boltz = std::exp(std::min(CDB_He2s2p / Tb, max_exp_boltz));
        const double n1 = nHe * (1.0 - xHe);
        const double C_He = (1.0 + K_He * Lambda_He * n1 * Boltz) / (1.0 + K_He * (Lambda_He + Rup_He) * n1 * Boltz);
        double dxdz =
            (Xe * xHe * nH * Rdown_He - Rup_He * (1.0 - xHe) * std::exp(-CL_He / Tb)) * C_He / (H * (1.0 + z));
        if (flag >= 3)
            dxdz += (Xe * xHe * nH * Rdown_trip - (1.0 - xHe) * 3.0 * Rup_trip * std::exp(-CL_Het / Tb)) * CfHe_t /
                    (H * (1.0 + z));
        return -(1.0 + z) * dxdz;
    }

    // Phases (as RECFAST): (A) Saha for H, He I, He II until helium starts recombining (x_He < 0.995) or x_H < 0.995;
    // (B) hydrogen in Saha (n_e = n_H x_H + n_He x_He), ODE for x_He and T_b; (C) ODE for x_H, x_He and T_b.
    void RecombinationHistory::solve_Xe_recfast(const std::vector<double> & x_array,
                                                std::vector<double> & Xe,
                                                std::vector<double> & dXedx,
                                                std::vector<double> & Tb,
                                                size_t & i_ode) {
        using namespace recfast;
        const size_t n = x_array.size();
        const double fHe = f_He();
        for (size_t i = 0; i < n; i++)
            Tb[i] = cosmo.TCMB_of_x(x_array[i]);
        const bool evolve_Tb = options.evolve_Tb;

        // (A) Saha
        size_t iA = n;
        double xHe_A = 0.0;
        for (size_t i = 0; i < n; i++) {
            const auto r = saha(x_array[i]);
            Xe[i] = r.Xe;
            xHe_A = r.xHeII + r.xHeIII;
            if (r.xH < x_H_trigger || (fHe > 0.0 && xHe_A < x_He_trigger)) {
                iA = i;
                break;
            }
        }
        if (iA == 0)
            throw std::runtime_error("RecombinationHistory: recombination has started at x_start");
        if (iA == n) {
            i_ode = n;
            return;
        }

        // hydrogen Saha with the helium electrons: x_H^2 + x_H (S/n_H + f_He x_He) - S/n_H = 0, S = (2 pi m_e
        // kT/h^2)^{3/2} e^{-E/kT}
        auto xH_saha = [&](double x, double xHe) {
            const double T = cosmo.TCMB_of_x(x);
            const double S = std::pow(CR * T, 1.5) * std::exp(-h_P * c * L_H_ion / (k_b * T)) / nH_of_x(x);
            const double b = S + fHe * xHe;
            return 0.5 * (std::sqrt(b * b + 4.0 * S) - b);
        };

        // (B) from the last Saha point
        size_t i0 = iA - 1;
        double xHe0 = fHe > 0.0 ? saha(x_array[i0]).xHeII : 0.0, Tb0 = Tb[i0];
        size_t iB = i0;
        if (fHe > 0.0 && xH_saha(x_array[i0], xHe0) >= x_H_trigger) {
            // integrate helium until the Saha x_H drops below the trigger
            std::vector<double> xs(x_array.begin() + i0, x_array.end());
            ODEFunction rhs = [&](double x, const double * y, double * dydx) {
                const double xH = xH_saha(x, y[0]), Xet = xH + fHe * y[0], T = evolve_Tb ? y[1] : cosmo.TCMB_of_x(x);
                dydx[0] = dxHedx_recfast(x, y[0], xH, Xet, T);
                if (evolve_Tb)
                    dydx[1] = dTbdx(x, Xet, y[1]);
                return GSL_SUCCESS;
            };
            // stop at the first grid point where x_H(Saha) < trigger: integrate on chunks of the grid
            std::vector<double> y0 = {xHe0};
            if (evolve_Tb)
                y0.push_back(Tb0);
            size_t j = 0;
            const size_t chunk = 200;
            while (true) {
                const size_t j1 = std::min(j + chunk, xs.size() - 1);
                std::vector<double> xc(xs.begin() + j, xs.begin() + j1 + 1);
                ODESolver ode(1e-6, 1e-14, options.recfast_relerr);
                ode.set_finite_difference_jacobian(true);
                ode.solve(rhs, xc, y0, gsl_odeiv2_step_msbdf);
                const auto yHe = ode.get_data_by_component(0);
                std::vector<double> yT = evolve_Tb ? ode.get_data_by_component(1) : std::vector<double>();
                bool stop = false;
                size_t m = 1;
                for (; m < xc.size(); m++) {
                    const size_t i = i0 + j + m;
                    const double xH = xH_saha(xc[m], yHe[m]);
                    Xe[i] = xH + fHe * yHe[m];
                    if (evolve_Tb)
                        Tb[i] = yT[m];
                    if (xH < x_H_trigger) {
                        stop = true;
                        break;
                    }
                }
                if (stop) {
                    iB = i0 + j + m;
                    xHe0 = yHe[m];
                    Tb0 = evolve_Tb ? yT[m] : Tb[iB];
                    break;
                }
                j = j1;
                y0 = {yHe.back()};
                if (evolve_Tb)
                    y0.push_back(yT.back());
                if (j1 == xs.size() - 1) {
                    i_ode = n;
                    return;
                }
            }
        } else if (fHe > 0.0) {
            xHe0 = saha(x_array[i0]).xHeII;
        }

        // (C) full system from iB (state there from phase B or Saha)
        const double xH0 = (iB == i0) ? saha(x_array[i0]).xH : xH_saha(x_array[iB], xHe0);
        std::vector<double> xs(x_array.begin() + iB, x_array.end());
        ODEFunction rhs = [&](double x, const double * y, double * dydx) {
            const double xHe = fHe > 0.0 ? y[1] : 0.0, Xet = y[0] + fHe * xHe;
            const double T = evolve_Tb ? y[2] : cosmo.TCMB_of_x(x);
            dydx[0] = dxHdx_recfast(x, y[0], Xet, T);
            dydx[1] = fHe > 0.0 ? dxHedx_recfast(x, xHe, y[0], Xet, T) : 0.0;
            dydx[2] = evolve_Tb ? dTbdx(x, Xet, y[2]) : 0.0;
            return GSL_SUCCESS;
        };
        ODESolver ode(1e-6, 1e-14, options.recfast_relerr);
        ode.set_finite_difference_jacobian(true);
        ode.solve(rhs, xs, {xH0, xHe0, Tb0}, gsl_odeiv2_step_msbdf);
        const auto yH = ode.get_data_by_component(0), yHe = ode.get_data_by_component(1),
                   yT = ode.get_data_by_component(2);
        for (size_t m = 0; m < xs.size(); m++) {
            const size_t i = iB + m;
            Xe[i] = yH[m] + fHe * yHe[m];
            if (evolve_Tb)
                Tb[i] = yT[m];
            double d[3];
            const double y[3] = {yH[m], yHe[m], yT[m]};
            rhs(xs[m], y, d);
            dXedx[i] = d[0] + fHe * d[1];
        }
        i_ode = iB + 1; // the first point of phase C keeps the Saha/phase-B value; use the spline derivative there
    }

    // =============================================================================
    // Solve
    // =============================================================================

    void RecombinationHistory::solve() {
        Utils::Timer timer("RecombinationHistory::solve");
        if (!cosmo.is_solved())
            throw std::runtime_error("RecombinationHistory::solve: background not solved");

        const auto x_array = Utils::linspace(options.x_start, options.x_end, options.n_points);
        solve_Xe(x_array);
        solve_tau(x_array);
        solve_sound_horizon(x_array);
        solved = true;
        timer.print();
    }

    void RecombinationHistory::solve_Xe(const std::vector<double> & x_array) {
        const size_t n = x_array.size();
        std::vector<double> Xe(n), dXedx(n, 0.0), Tb(n);
        if (options.model == RecombinationOptions::Model::Recfast) {
            size_t i_ode = n;
            solve_Xe_recfast(x_array, Xe, dXedx, Tb, i_ode); // dX_e/dx for i >= i_ode
            finish_Xe(x_array, Xe, dXedx, Tb, i_ode);
            return;
        }

        // Saha regime (hydrogen and helium); switch when the hydrogen fraction drops below the limit
        std::vector<double> xH(n);
        size_t i_switch = n;
        for (size_t i = 0; i < n; i++) {
            const auto r = saha(x_array[i]);
            Xe[i] = r.Xe;
            xH[i] = r.xH;
            if (r.xH < options.Xe_saha_limit) {
                i_switch = i;
                break;
            }
        }
        if (i_switch == 0)
            throw std::runtime_error("RecombinationHistory: x_H below the Saha limit at x_start");

        // Peebles regime: start from the last Saha point so X_e is continuous.
        // With T_b evolution, y = (X_e, T_b); the Compton coupling is very stiff (Gamma_C/H ~ 1e6),
        // so an implicit method is used.
        for (size_t i = 0; i < n; i++)
            Tb[i] = cosmo.TCMB_of_x(x_array[i]);
        if (i_switch < n) {
            const size_t i0 = i_switch - 1;
            std::vector<double> x_peebles(x_array.begin() + i0, x_array.end());
            const bool evolve_Tb = options.evolve_Tb;
            ODEFunction rhs = [&](double x, const double * y, double * dydx) {
                const double Tb_x = evolve_Tb ? y[1] : cosmo.TCMB_of_x(x);
                dydx[0] = dXedx_peebles(x, y[0], Tb_x);
                if (evolve_Tb)
                    dydx[1] = dTbdx(x, y[0], y[1]);
                return GSL_SUCCESS;
            };
            std::vector<double> y_ic = {xH[i0]};
            if (evolve_Tb)
                y_ic.push_back(Tb[i0]);

            ODESolver ode(1e-6, 1e-14, 1e-10);
            ode.set_finite_difference_jacobian(true);
            ode.solve(rhs, x_peebles, y_ic, evolve_Tb ? gsl_odeiv2_step_msbdf : gsl_odeiv2_step_rk8pd);
            // Peebles gives x_H; the remaining helium ionization is added from Saha with x_H fixed
            // (it is ~1e-4 at the switch and decays quickly).
            const auto xH_p = ode.get_data_by_component(0);
            const auto dxH_p = ode.get_derivative_data_by_component(0);
            std::vector<double> XeHe(x_peebles.size(), 0.0);
            bool he_done = !options.helium;
            for (size_t j = 0; j < x_peebles.size(); j++) {
                if (!he_done) {
                    const auto r = saha(x_peebles[j], xH_p[j]);
                    XeHe[j] = f_He() * (r.xHeII + 2.0 * r.xHeIII);
                    if (j > 0 && XeHe[j] < 1e-12 * xH_p[j])
                        he_done = true;
                }
            }
            // dXeHe/dx by finite differences on the grid (it is tiny and smooth)
            for (size_t j = 1; j < x_peebles.size(); j++) {
                const size_t jm = j - 1, jp = std::min(j + 1, x_peebles.size() - 1);
                const double dXeHe = (XeHe[jp] - XeHe[jm]) / (x_peebles[jp] - x_peebles[jm]);
                Xe[i0 + j] = xH_p[j] + XeHe[j];
                dXedx[i0 + j] = dxH_p[j] + dXeHe;
            }
            if (evolve_Tb) {
                const auto Tb_p = ode.get_data_by_component(1);
                for (size_t j = 1; j < x_peebles.size(); j++)
                    Tb[i0 + j] = Tb_p[j];
            }
        }

        finish_Xe(x_array, Xe, dXedx, Tb, i_switch);
    }

    void RecombinationHistory::finish_Xe(const std::vector<double> & x_array,
                                         std::vector<double> & Xe,
                                         std::vector<double> & dXedx,
                                         const std::vector<double> & Tb,
                                         size_t i_switch) {
        const size_t n = x_array.size();
        if (options.evolve_Tb) {
            std::vector<double> log_Tb(n);
            for (size_t i = 0; i < n; i++)
                log_Tb[i] = std::log(Tb[i]);
            log_Tb_spline.create(x_array, log_Tb, "log_Tb");
        }

        // d X_e / dx in the Saha regime (i < i_switch) from a spline of ln X_e (smooth there)
        {
            // (the spline extends a few points into the ODE regime so that its end condition does not affect i_switch -
            // 1;
            //  for the Peebles model it stops at the switch as before)
            const size_t m =
                (options.model == RecombinationOptions::Model::Recfast) ? std::min(n, i_switch + 20) : i_switch;
            std::vector<double> xs(x_array.begin(), x_array.begin() + m), lx(m);
            for (size_t i = 0; i < m; i++)
                lx[i] = std::log(Xe[i]);
            if (i_switch >= 3) {
                Spline1D s(xs, lx);
                for (size_t i = 0; i < i_switch; i++)
                    dXedx[i] = Xe[i] * s.deriv_x(xs[i]);
            }
        }

        // Reionization, and d ln X_e / dx (exact in the Peebles and reionization parts)
        std::vector<double> log_Xe(n);
        dlogXedx.assign(n, 0.0);
        for (size_t i = 0; i < n; i++) {
            double Xt, dXt;
            reionize(x_array[i], Xe[i], dXedx[i], Xt, dXt);
            log_Xe[i] = std::log(Xt);
            dlogXedx[i] = dXt / Xt;
        }
        log_Xe_spline.create(x_array, log_Xe, "log_Xe");
    }

    void RecombinationHistory::solve_tau(const std::vector<double> & x_array) {
        const size_t n = x_array.size();

        // Integrate backwards from x = 0 where tau = 0
        std::vector<double> x_rev(x_array.rbegin(), x_array.rend());
        ODEFunction rhs = [&](double x, const double *, double * dydx) {
            dydx[0] = -c * sigma_T * ne_of_x(x) / cosmo.H_of_x(x);
            return GSL_SUCCESS;
        };
        ODESolver ode(-1e-5, 1e-14, 1e-11);
        ode.solve(rhs, x_rev, {0.0}, gsl_odeiv2_step_rk8pd);
        auto tau_rev = ode.get_data_by_component(0);
        std::vector<double> tau(tau_rev.rbegin(), tau_rev.rend());

        // tau' (analytic), tau'' = tau' (d ln X_e/dx - 3 - d ln H/dx) = tau' (d ln X_e/dx - 2 - Hp'/Hp)
        std::vector<double> dtau(n), ddtau(n), g(n), dg(n);
        for (size_t i = 0; i < n; i++) {
            const double x = x_array[i];
            dtau[i] = -c * sigma_T * ne_of_x(x) / cosmo.H_of_x(x);
            ddtau[i] = dtau[i] * (dlogXedx[i] - 2.0 - cosmo.dHpdx_of_x(x) / cosmo.Hp_of_x(x));
            const double e = std::exp(-tau[i]);
            g[i] = -dtau[i] * e;
            dg[i] = (dtau[i] * dtau[i] - ddtau[i]) * e;
        }

        tau_spline.create(x_array, tau, "tau");
        ddtauddx_spline.create(x_array, ddtau, "ddtauddx");
        g_spline.create(x_array, g, "g_tilde");
        dgdx_spline.create(x_array, dg, "dgdx_tilde");
    }

    void RecombinationHistory::solve_sound_horizon(const std::vector<double> & x_array) {
        ODEFunction rhs = [&](double x, const double *, double * dydx) {
            dydx[0] = cs_of_x(x) / cosmo.Hp_of_x(x);
            return GSL_SUCCESS;
        };
        // Initial condition: c_s ~ const early on, so s = (c_s / c) eta  (eta has units of length)
        const double x0 = x_array.front();
        ODESolver ode(1e-3, 0.0, 1e-10);
        ode.solve(rhs, x_array, {cs_of_x(x0) / c * cosmo.eta_of_x(x0)}, gsl_odeiv2_step_rk8pd);
        sound_horizon_spline.create(x_array, ode.get_data_by_component(0), "sound_horizon");
    }

    void RecombinationHistory::require_solved(const char * func) const {
        if (!solved)
            throw std::runtime_error(std::string("RecombinationHistory::") + func + ": call solve() first");
    }

    // =============================================================================
    // Getters
    // =============================================================================

    double RecombinationHistory::Xe_of_x(double x) const {
        if (!log_Xe_spline)
            require_solved("Xe_of_x");
        return std::exp(log_Xe_spline(x));
    }

    double RecombinationHistory::ne_of_x(double x) const { return Xe_of_x(x) * nH_of_x(x); }

    double RecombinationHistory::tau_of_x(double x) const {
        require_solved("tau_of_x");
        return tau_spline(x);
    }

    double RecombinationHistory::dtaudx_of_x(double x) const { return -c * sigma_T * ne_of_x(x) / cosmo.H_of_x(x); }

    double RecombinationHistory::ddtauddx_of_x(double x) const {
        require_solved("ddtauddx_of_x");
        return ddtauddx_spline(x);
    }

    double RecombinationHistory::g_tilde_of_x(double x) const {
        require_solved("g_tilde_of_x");
        return g_spline(x);
    }

    double RecombinationHistory::dgdx_tilde_of_x(double x) const {
        require_solved("dgdx_tilde_of_x");
        return dgdx_spline(x);
    }

    double RecombinationHistory::ddgddx_tilde_of_x(double x) const {
        require_solved("ddgddx_tilde_of_x");
        return dgdx_spline.deriv_x(x);
    }

    double RecombinationHistory::R_of_x(double x) const { return 4.0 * cosmo.rhoGamma(x) / (3.0 * cosmo.rhoB(x)); }

    double RecombinationHistory::cs_of_x(double x) const {
        const double R = R_of_x(x);
        return c * std::sqrt(R / (3.0 * (1.0 + R)));
    }

    double RecombinationHistory::sound_horizon_of_x(double x) const {
        require_solved("sound_horizon_of_x");
        return sound_horizon_spline(x);
    }

    // =============================================================================
    // Characteristic epochs
    // =============================================================================

    double RecombinationHistory::x_last_scattering() const {
        return Utils::find_root([&](double x) { return tau_of_x(x) - 1.0; }, -10.0, -3.0);
    }

    double RecombinationHistory::x_visibility_peak() const {
        // Coarse scan for the maximum, then refine with g' = 0
        double x_max = -7.0, g_max = 0.0;
        for (double x = -10.0; x <= -3.0; x += 0.01) {
            const double g = g_tilde_of_x(x);
            if (g > g_max) {
                g_max = g;
                x_max = x;
            }
        }
        return Utils::find_root([&](double x) { return dgdx_tilde_of_x(x); }, x_max - 0.01, x_max + 0.01);
    }

    double RecombinationHistory::x_Xe_half() const {
        return Utils::find_root([&](double x) { return Xe_of_x(x) - 0.5; }, -10.0, -3.0);
    }

    double RecombinationHistory::tau_reionization() const { return tau_of_x(-std::log(51.0)); }

    double RecombinationHistory::tau_reionization_model(double z_reion) const {
        const auto & p = cosmo.get_params();
        const double fHe = f_He();
        const double y_re = std::pow(1.0 + z_reion, 1.5), dy = 1.5 * std::sqrt(1.0 + z_reion) * p.delta_z_reion;
        auto f = [&](double x) { // -dtau/dx = c n_e sigma_T / H
            const double z = std::exp(-x) - 1.0;
            const double TH = 0.5 * (1.0 + std::tanh((y_re - std::pow(1.0 + z, 1.5)) / dy));
            const double THe = 0.5 * (1.0 + std::tanh((p.z_He_reion - z) / p.delta_z_He_reion));
            const double Xe = (1.0 + fHe) * TH + fHe * THe;
            return c * Xe * nH_of_x(x) * sigma_T / cosmo.H_of_x(x);
        };
        // split at the reionization redshifts so the adaptive integrator sees both steps
        const double x_re = -std::log1p(z_reion), x_He = -std::log1p(p.z_He_reion), x0 = -std::log(51.0);
        std::vector<double> b = {x0, 0.0};
        for (double xb : {x_re, x_He})
            if (xb > x0 && xb < 0.0)
                b.push_back(xb);
        std::sort(b.begin(), b.end());
        double tau = 0.0;
        for (size_t i = 0; i + 1 < b.size(); i++)
            tau += Utils::integrate(f, b[i], b[i + 1], 0.0, 1e-10);
        return tau;
    }

    double RecombinationHistory::x_Xe_half_saha() const {
        return Utils::find_root([&](double x) { return Xe_saha_of_x(x) - 0.5; }, -10.0, -3.0);
    }

    // =============================================================================
    // Output
    // =============================================================================

    void RecombinationHistory::info() const {
        require_solved("info");
        const auto f = std::cout.flags();
        std::cout << std::setprecision(6);
        const bool rf = options.model == RecombinationOptions::Model::Recfast;
        std::cout << "\nRecombinationHistory (" << (rf ? "RECFAST" : "Peebles") << ", Yp = " << Yp
                  << (options.helium ? ", helium" : ", hydrogen only")
                  << (reion ? ", reionization at z = " + std::to_string(cosmo.get_params().z_reion).substr(0, 5) :
                              ", no reionization")
                  << ", " << (options.evolve_Tb ? "T_b evolved" : "T_b = T_gamma")
                  << (!rf && options.peebles_fudge != 1.0 ? ", fudge = " + std::to_string(options.peebles_fudge) : "")
                  << "):\n";
        auto print_epoch = [&](const char * name, double x) {
            std::cout << "  " << std::left << std::setw(28) << name << "x = " << std::setw(11) << x
                      << "  z = " << std::setw(10) << std::exp(-x) - 1.0 << "  t = " << cosmo.t_of_x(x) / yr << " yr\n";
        };
        print_epoch("X_e = 0.5 (Saha only):", x_Xe_half_saha());
        print_epoch(rf ? "X_e = 0.5 (RECFAST):" : "X_e = 0.5 (Saha+Peebles):", x_Xe_half());
        print_epoch("Last scattering (tau = 1):", x_last_scattering());
        print_epoch("Visibility peak:", x_visibility_peak());

        const double x_ls = x_last_scattering();
        std::cout << "  X_e today                   = " << Xe_of_x(0.0) << (reion ? " (reionized)" : " (freeze-out)")
                  << "\n";
        std::cout << "  X_e at z = 30 (freeze-out)  = " << Xe_of_x(-std::log(31.0)) << "\n";
        if (reion)
            std::cout << "  Optical depth tau(z = 50)   = " << tau_reionization() << "\n";
        std::cout << "  Sound horizon at tau = 1    = " << sound_horizon_of_x(x_ls) / Mpc << " Mpc\n";
        std::cout << "  Integral of g over x        = " << g_spline.integral(options.x_start, options.x_end) << "\n\n";
        std::cout.flags(f);
    }

    void RecombinationHistory::output(const std::string & filename, double x_min, double x_max, int n_points) const {
        require_solved("output");
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("RecombinationHistory::output: cannot open " + filename);
        fp << "# Recombination history. ne in 1/m^3, Tb in K, cs in units of c, s in Mpc\n"
           << "# x  Xe  ne  tau  dtaudx  ddtauddx  g  dgdx  ddgddx  Tb  cs  s  Xe_saha\n";
        fp << std::scientific << std::setprecision(10);
        for (double x : Utils::linspace(x_min, x_max, n_points)) {
            fp << x << " " << Xe_of_x(x) << " " << ne_of_x(x) << " " << tau_of_x(x) << " " << dtaudx_of_x(x) << " "
               << ddtauddx_of_x(x) << " " << g_tilde_of_x(x) << " " << dgdx_tilde_of_x(x) << " " << ddgddx_tilde_of_x(x)
               << " " << Tb_of_x(x) << " " << cs_of_x(x) / c << " " << sound_horizon_of_x(x) / Mpc << " "
               << Xe_saha_of_x(x) << "\n";
        }
    }

} // namespace FML::COSMOLOGY
