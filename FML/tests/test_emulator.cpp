//==========================================================
// Tests for FML::EMULATOR::EUCLIDEMULATOR2 (Emulator/EuclidEmulator2.h):
//   - the boost for a reference cosmology at z = 0 and 1 (values of the original code; ~1 on large scales)
//   - a missing data file and too many redshifts throw (segfault / stack overflow before)
//   - copies of the Cosmology class own their GSL objects (double free before) and the emulator cannot be copied
// The header is also included in test_emulator_odr.cpp (it must be usable from several translation units)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Emulator/EuclidEmulator2.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

using namespace FML::EMULATOR::EUCLIDEMULATOR2;

int emulator_odr_check(); // defined in test_emulator_odr.cpp

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}

double boost_at(const std::pair<std::vector<double>, std::vector<double>> & b, double k) {
    size_t i = 0;
    while (b.first[i] < k)
        i++;
    return b.second[i];
}

int main() {
    check(emulator_odr_check() == 1, "EuclidEmulator2.h can be included in several translation units");

    // Reference values (k = 0.0150042, 0.304633, 1.00213, 5.02027 h/Mpc)
    {
        EuclidEmulator ee2(0.049, 0.315, 0.06, 0.966, 0.674, -1.0, 0.0, 2.1e-9);
        const double ref[2][4] = {{0.99194673605028083, 1.5359928473359081, 5.7588427213863813, 28.809772623082434},
                                  {0.99421122363744574, 1.2467343786121599, 3.111326554129056, 18.59839146099004}};
        double maxerr = 0.0;
        for (int iz = 0; iz < 2; iz++) {
            auto b = ee2.compute_boost(double(iz));
            int ik = 0;
            for (double k : {0.01, 0.3, 1.0, 5.0})
                maxerr = std::max(maxerr, std::fabs(boost_at(b, k) / ref[iz][ik++] - 1.0));
        }
        check(maxerr < 1e-10, "boost at z = 0, 1 = reference values");

        // Too many redshifts
        bool thrown = false;
        try {
            ee2.compute_boost(std::vector<double>(200, 0.5), 200);
        } catch (std::exception &) {
            thrown = true;
        }
        check(thrown, "more than 101 redshifts throws (buffer overflow before)");
    }

    // Missing data file
    {
        const std::string path = path_to_ee2_datafile;
        set_path_to_ee2_data("this_file_does_not_exist.dat");
        bool thrown = false;
        try {
            EuclidEmulator ee2(0.049, 0.315, 0.06, 0.966, 0.674, -1.0, 0.0, 2.1e-9);
        } catch (std::exception &) {
            thrown = true;
        }
        set_path_to_ee2_data(path);
        check(thrown, "a missing data file throws (segfault before)");
    }

    // Copies
    {
        auto csm = std::make_unique<Cosmology>(0.049, 0.315, 0.06, 0.966, 0.674, -1.0, 0.0, 2.1e-9);
        Cosmology copy = *csm;
        const double ref = csm->compute_step_number(1.0);
        csm.reset();
        Cosmology copy2(0.05, 0.3, 0.0, 0.96, 0.7, -1.0, 0.0, 2.0e-9);
        copy2 = copy;
        check(copy.compute_step_number(1.0) == ref and copy2.compute_step_number(1.0) == ref,
              "copies of Cosmology own their GSL objects (double free before)");
        check(not std::is_copy_constructible_v<EuclidEmulator>, "EuclidEmulator cannot be copied (double free before)");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All emulator tests passed\n" : std::to_string(fails) + " emulator test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
