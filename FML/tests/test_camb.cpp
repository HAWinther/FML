//==========================================================
// Tests for the CAMB reader of FML/TransferProviders (CAMBFileTransferProvider) with the CAMB tables in
// ../SimulationCode/input/camb_data_lcdm_nu0.4_zmax200 (0.4 eV, 60 redshifts up to z = 200) against CAMB itself
// (P_cb and P_total / P_cb computed with camb 2.0.5 for the same cosmology):
//   - P_cb(k, z) and the total-matter P(k, z) (CAMB's total column)
//   - P(k) above the largest k of the tables falls (power-law extrapolation of T)
//   - growth rates outside the redshift range of the tables (the value at the end), the redshift range
//   - an infofile not ordered in redshift
// The comparison with CLASS and the gauge conversion are in TransferProviders/tests (they need generated data)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/TransferProviders/CAMBFileTransferProvider.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;
using FML::TESTS::num;

using namespace FML::INITIALCONDITIONS;

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

CAMBFileTransferProvider make(const std::string & infofile) {
    CAMBFileTransferProvider::Settings s;
    s.infofile = infofile;
    s.h = 0.671;
    s.Omegab = 0.049;
    s.OmegaMNu = 0.0095;
    s.OmegaCDM = 0.3175 - s.Omegab - s.OmegaMNu;
    s.A_s = 2.215e-9;
    s.n_s = 0.966;
    s.k_pivot_mpc = 0.05;
    s.range_policy = RangePolicy::SourceExtrapolation;
    return CAMBFileTransferProvider(s);
}

int main() {
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
        const double pcb = td.power(Species::CDMBaryon, k, a);
        const double ptot = td.power(Species::TotalMatter, k, a);
        err_cb = std::max(err_cb, std::fabs(pcb / r[2] - 1.0));
        err_tot = std::max(err_tot, std::fabs(ptot / pcb / r[3] - 1.0));
    }
    check_local(err_cb < 3e-3, "P_cb(k, z = 0, 51, 200) = CAMB (max error " + num(err_cb, 3) + ")");
    check_local(err_tot < 3e-4, "P_total / P_cb = CAMB (max error " + num(err_tot, 3) + ")");

    // Beyond the largest k of the tables
    const double kmax = td.range().kmax_hmpc;
    const double p1 = td.power(Species::CDMBaryon, kmax, 1.0), p2 = td.power(Species::CDMBaryon, 5 * kmax, 1.0);
    check_local(p2 < p1 and p2 > 0.0,
                "P(k) above kmax = " + num(kmax, 3) + " h/Mpc falls (P(5 kmax)/P(kmax) = " + num(p2 / p1, 3) + ")");

    // Redshift range
    const double zmin = 1.0 / td.range().amax - 1.0, zmax = 1.0 / td.range().amin - 1.0;
    check_local(std::fabs(zmin) < 1e-12 and std::fabs(zmax - 200.0) < 1e-10,
                "redshift range of the tables [" + num(zmin, 3) + ", " + num(zmax, 3) + "]");

    // Growth rate beyond the tables: the value at the end (not an extrapolation)
    const double f200 = td.density_growth_rate(Species::CDMBaryon, 0.1, 1.0 / 201.0);
    const double f332 = td.density_growth_rate(Species::CDMBaryon, 0.1, 1.0 / 333.0);
    check_local(f332 == f200 and f200 > 0.9 and f200 < 1.0,
                "growth rate beyond the tables = the value at z = 200 (" + num(f332, 3) + ")");

    // Infofile not ordered in redshift
    bool ok = true;
    try {
        auto td_rev = make(info_rev);
        for (double z : {0.0, 1.0, 49.0})
            for (double k : {0.01, 0.1, 1.0})
                ok = ok and td_rev.density_transfer(Species::CDMBaryon, k, 1.0 / (1.0 + z)) ==
                                td.density_transfer(Species::CDMBaryon, k, 1.0 / (1.0 + z));
    } catch (...) {
        ok = false;
    }
    check_local(ok, "infofile not ordered in redshift = ordered infofile");

    std::remove(info.c_str());
    std::remove(info_rev.c_str());

    return FML::TESTS::summary("CAMB reader");
}
