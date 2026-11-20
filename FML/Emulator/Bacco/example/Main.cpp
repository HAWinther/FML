#include <FML/Emulator/Bacco/Bacco.h>

#include <iomanip>
#include <iostream>

// baccoemu (Angulo, Zennaro, Arico, Pellejero Ibanez, Burger; MIT licence): neural-network emulators of the linear
// and nonlinear matter power spectrum, of its baryonic suppression and of the Lagrangian bias expansion. See Bacco.h /
// README.md, and cite the papers when using them.
//
// The data files are found next to Bacco.h (data/, made by export_bacco.py); if you move them, call
// FML::EMULATOR::BACCO::set_data_dir("path/to/data");

using namespace FML::EMULATOR::BACCO;

int main() {
    Parameters p;
    p.Omega_cold = 0.31; // Omega_cdm + Omega_b (or p.Omega_m)
    p.Omega_b = 0.049;
    p.sigma8_cold = 0.81; // or p.A_s
    p.h = 0.67;
    p.n_s = 0.965;
    p.M_nu = 0.06; // eV
    p.w0 = -1.0;
    p.wa = 0.0;
    // Baryons (Arico et al. 2021), log10 of masses in Msun/h
    p.M_c = 14.0;
    p.eta = -0.3;
    p.beta = -0.22;
    p.M1_z0_cen = 10.5;
    p.theta_out = 0.25;
    p.theta_inn = -0.86;
    p.M_inn = 13.4;

    if (not data_available(true)) { // the networks are not distributed with FML
        std::cout << "The baccoemu data is missing: run FML/Emulator/Bacco/get_data.sh first\n";
        return 1;
    }

    MatterPowerSpectrum::Options opt;
    opt.nonlinear_model = "Arico2023"; // wider ranges than the default Angulo2021
    MatterPowerSpectrum emu(opt);

    const double a = 1.0 / 1.5;
    std::vector<double> k = {0.01, 0.03, 0.1, 0.3, 1.0, 3.0};
    std::cout << "# sigma8_cold = " << emu.sigma8(p) << ", sigma8_tot = " << emu.sigma8(p, false)
              << ", A_s = " << emu.A_s(p) << "\n";
    auto plin = emu.linear_pk(p, a, k, false);              // total matter
    auto pnl = emu.nonlinear_pk(p, a, k, false);            // total matter
    auto pnl_bar = emu.nonlinear_pk(p, a, k, false, true);  // with the baryonic boost
    std::cout << "# z = 0.5, total matter\n# k (h/Mpc)   P_lin   P_nl   P_nl with baryons   (Mpc/h)^3\n";
    for (size_t i = 0; i < k.size(); i++)
        std::cout << std::setw(8) << k[i] << " " << std::setw(12) << plin[i] << " " << std::setw(12) << pnl[i] << " "
                  << std::setw(12) << pnl_bar[i] << "\n";

    // The baryonic suppression of the bispectrum (Burger et al. 2025) for equilateral triangles
    MatterBispectrum bk;
    auto r = bk.baryonic_boost(p, a, {0.1, 1.0, 5.0}, {0.1, 1.0, 5.0}, {0.1, 1.0, 5.0});
    std::cout << "\n# B_baryons / B_dmo, equilateral, k = 0.1, 1, 5 h/Mpc: " << r.boost[0] << " " << r.boost[1] << " "
              << r.boost[2] << "\n";

    // Galaxies: the second order Lagrangian bias expansion (b1, b2, bs2, blaplacian are Lagrangian), here with the
    // coevolution relations for galaxies (Zennaro et al. 2022) for b2, bs2 and blaplacian
    const double b1 = 0.5;
    std::array<double, 4> bias = {b1, COEVOLUTION::gal_b2L_b1L(b1), COEVOLUTION::gal_bs2L_b1L(b1),
                                  COEVOLUTION::gal_blL_b1L(b1)};
    std::vector<double> kg = {0.02, 0.05, 0.1, 0.2, 0.4};
    LbiasExpansion lb;
    auto g = lb.galaxy_real_pk(bias, p, a, kg);
    LbiasExpansionRSD lbz;
    LbiasExpansionRSD::RSDParameters rsd;
    rsd.f_sat = 0.1;      // satellite fraction and
    rsd.lambda_FoG = 2.5; // the fingers of god damping scale (h/Mpc)
    auto ml = lbz.galaxy_pk(bias, rsd, p, a, kg);
    std::cout << "\n# Galaxies, z = 0.5, b1 = " << b1 << "\n# k (h/Mpc)   P_gg   P_gm   P_0   P_2   P_4 (redshift space)\n";
    for (size_t i = 0; i < kg.size(); i++)
        std::cout << std::setw(8) << kg[i] << " " << std::setw(12) << g.auto_pk[i] << " " << std::setw(12)
                  << g.cross_pk[i] << " " << std::setw(12) << ml[0][i] << " " << std::setw(12) << ml[1][i] << " "
                  << std::setw(12) << ml[2][i] << "\n";
}
