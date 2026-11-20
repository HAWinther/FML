// SPDX-License-Identifier: GPL-3.0-or-later
// Part of FML's port of e-MANTIS (GPL-3.0-or-later, see ../COPYING and ../README.md).
#include <FML/Emulator/EMantis/EMantis.h>

#include <iomanip>
#include <iostream>

// e-MANTIS (Saez-Casares et al.; arXiv:2303.08899, arXiv:2410.05226): emulators of the nonlinear matter power spectrum and the halo mass function
// in f(R) gravity and wCDM. See EMantis.h / README.md, and cite the e-MANTIS papers when using them.
//
// The data files are found next to EMantis.h; if you move them, call
// FML::EMULATOR::EMANTIS::set_data_dir("path/to/data");

using namespace FML::EMULATOR::EMANTIS;

int main() {
    // f(R) cosmology: sigma8 is sigma8_lcdm, the sigma8 of LCDM with the same A_s (or give p.A_s instead)
    Parameters p;
    p.Omega_b = 0.049;
    p.Omega_m = 0.31;
    p.h = 0.68;
    p.n_s = 0.965;
    p.sigma8 = 0.81;
    p.logfR0 = 5.0; // |fR0| = 1e-5

    // B(k) = P_f(R)(k) / P_LCDM(k) at z = 0.5. Multiply a LCDM nonlinear P(k) (e.g. EuclidEmulator2 or HMcode) by it
    NonLinearMGBoostEmulator fr("fR");
    const double a = 1.0 / 1.5;
    std::vector<double> k = {0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 9.0};
    auto boost = fr.boost(p, a, k);
    std::cout << "# f(R) boost at z = 0.5, |fR0| = 1e-5\n# k (h/Mpc)    B(k)\n";
    for (size_t i = 0; i < k.size(); i++)
        std::cout << std::setw(10) << k[i] << " " << std::setw(12) << boost[i] << "\n";

    // The halo mass function (200c) with its emulation uncertainty
    HMFEmulator hmf("fR", "200c");
    auto r = hmf.range("mass_halo", a);
    std::vector<double> mass;
    for (int i = 0; i < 6; i++)
        mass.push_back(r.first * std::pow(r.second / r.first, i / 5.0));
    std::vector<double> std;
    auto n = hmf.hmf(mass, p, a, false, &std);
    std::cout << "\n# HMF (200c) at z = 0.5\n# M (Msun/h)    HMF    std\n";
    for (size_t i = 0; i < mass.size(); i++)
        std::cout << std::setw(12) << mass[i] << " " << std::setw(12) << n[i] << " " << std::setw(12) << std[i] << "\n";

    // wCDM: P_nl / P_lin
    Parameters q = p;
    q.w = -0.9;
    NonLinearBoostEmulator wcdm("wCDM");
    auto bw = wcdm.boost(q, a, {0.1, 1.0, 5.0});
    std::cout << "\n# wCDM (w = -0.9) P_nl/P_lin at k = 0.1, 1, 5 h/Mpc: " << bw[0] << " " << bw[1] << " " << bw[2] << "\n";
    return 0;
}
