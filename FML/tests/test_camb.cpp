//==========================================================
// Tests for FML::FILEUTILS::LinearTransferData (CAMBUtils/CAMBReader.h) with the CAMB tables in
// ../SimulationCode/input/camb_data_lcdm_nu0.4_zmax200 (0.4 eV, 60 redshifts up to z = 200) against CAMB itself
// (P_cb and P_total / P_cb computed with camb 2.0.5 for the same cosmology):
//   - P_cb(k, z) and the total-matter P(k, z) (the neutrino weight was wrong at high z)
//   - P(k) above the largest k of the tables falls (it grew as k^ns)
//   - growth rates outside the redshift range of the tables (were extrapolated), the redshift range getters
//   - an infofile not ordered in redshift (aborted in GSL)
// The header is also included in test_camb_odr.cpp (it must be usable from several translation units)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/CAMBUtils/CAMBReader.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace FML::FILEUTILS;

int camb_odr_check(); // defined in test_camb_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(3);
    os << x;
    return os.str();
}

const std::string tables = "../SimulationCode/input/camb_data_lcdm_nu0.4_zmax200";
const std::string infofile_repo = "../SimulationCode/input/transfer_infofile_lcdm_nu0.4_zmax200.txt";

// Write an infofile pointing to the tables (in the original order or with neighbouring entries swapped)
void write_infofile(const std::string & filename, bool shuffled) {
    std::ifstream in(infofile_repo);
    std::string path;
    int n;
    in >> path >> n;
    std::vector<std::pair<std::string, std::string>> entries(n);
    for (auto & e : entries)
        in >> e.first >> e.second;
    std::ofstream out(filename);
    out << tables << " " << n << "\n";
    for (int i = 0; i < n; i++) {
        auto & e = entries[shuffled ? (i % 2 == 0 ? std::min(i + 1, n - 1) : i - 1) : i];
        out << e.first << "  " << e.second << "\n";
    }
}

LinearTransferData make(const std::string & infofile) {
    const double h = 0.671, OmegaM = 0.3175, Omegab = 0.049, OmegaMNu = 0.0095;
    LinearTransferData td(Omegab, OmegaM - Omegab - OmegaMNu, OmegaMNu, 0.05, 2.215e-9, 0.966, h);
    td.read_transfer(infofile);
    return td;
}

int main() {
    check(camb_odr_check() == 1, "CAMBReader.h can be included in several translation units");

    const std::string info = "test_camb_info_" + std::to_string(FML::ThisTask) + ".txt";
    const std::string info_rev = "test_camb_info_rev_" + std::to_string(FML::ThisTask) + ".txt";
    write_infofile(info, false);
    write_infofile(info_rev, true);

    auto td = make(info);

    // {z, k (h/Mpc), P_cb (Mpc/h)^3, P_total / P_cb} from camb
    const double ref[9][4] = {{0, 0.01, 22139.13306, 0.990603017},
                              {0, 0.1, 4927.908073, 0.9544797854},
                              {0, 1.0, 59.50767141, 0.9415706892},
                              {51.196, 0.01, 13.95134958, 0.9601131397},
                              {51.196, 0.1, 3.332207649, 0.9404881986},
                              {51.196, 1.0, 0.04061055403, 0.9395204385},
                              {200, 0.01, 0.9569841095, 0.9465158295},
                              {200, 0.1, 0.2462068485, 0.9235346698},
                              {200, 1.0, 0.003001024966, 0.9231194976}};
    double err_cb = 0.0, err_tot = 0.0;
    for (auto & r : ref) {
        const double a = 1.0 / (1.0 + r[0]), k = r[1];
        const double pcb = td.get_cdm_baryon_power_spectrum(k, a);
        const double ptot = td.get_total_power_spectrum(k, a);
        err_cb = std::max(err_cb, std::fabs(pcb / r[2] - 1.0));
        err_tot = std::max(err_tot, std::fabs(ptot / pcb / r[3] - 1.0));
    }
    check(err_cb < 3e-3, "P_cb(k, z = 0, 51, 200) = CAMB (max error " + num(err_cb) + ")");
    check(err_tot < 3e-4,
          "P_total / P_cb = CAMB (max error " + num(err_tot) + ", was 0.2% at z = 51 and 2% at z = 200)");

    // Beyond the largest k of the tables
    const double kmax = td.get_kmax_hmpc_splines();
    const double p1 = td.get_cdm_baryon_power_spectrum(kmax, 1.0), p2 = td.get_cdm_baryon_power_spectrum(5 * kmax, 1.0);
    check(p2 < p1 and p2 > 0.0,
          "P(k) above kmax = " + num(kmax) + " h/Mpc falls (P(5 kmax)/P(kmax) = " + num(p2 / p1) +
              ", was 4.7: T was constant)");

    // Redshift range
    check(td.get_zmin_splines() == 0.0 and std::fabs(td.get_zmax_splines() - 200.0) < 1e-10,
          "redshift range of the tables [" + num(td.get_zmin_splines()) + ", " + num(td.get_zmax_splines()) + "]");

    // Growth rate beyond the tables: the value at the end (not an extrapolation)
    const double f200 = td.get_cdm_baryon_growth_rate(0.1, 1.0 / 201.0);
    const double f332 = td.get_cdm_baryon_growth_rate(0.1, 1.0 / 333.0);
    check(f332 == f200 and f200 > 0.9 and f200 < 1.0,
          "growth rate beyond the tables = the value at z = 200 (" + num(f332) + ", was 1.07)");

    // Infofile not ordered in redshift
    bool ok = true;
    try {
        auto td_rev = make(info_rev);
        for (double z : {0.0, 1.0, 49.0})
            for (double k : {0.01, 0.1, 1.0})
                ok = ok and td_rev.get_cdm_baryon_transfer_function(k, 1.0 / (1.0 + z)) ==
                                td.get_cdm_baryon_transfer_function(k, 1.0 / (1.0 + z));
    } catch (...) {
        ok = false;
    }
    check(ok, "infofile not ordered in redshift = ordered infofile (aborted in GSL before)");

    std::remove(info.c_str());
    std::remove(info_rev.c_str());

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All CAMB reader tests passed\n" :
                                   std::to_string(fails) + " CAMB reader test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
