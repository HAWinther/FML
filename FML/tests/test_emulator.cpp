//==========================================================
// Tests for FML::EMULATOR::EUCLIDEMULATOR2 (Emulator/EuclidEmulator2.h):
//   - the boost for a reference cosmology at z = 0 and 1 (values of the original code; ~1 on large scales)
//   - a missing data file and too many redshifts throw (segfault / stack overflow before)
//   - copies of the Cosmology class own their GSL objects (double free before) and the emulator cannot be copied
//   - the f(R) fitting function (Emulator/FofrFittingFunction.h): the values of the original code (C/fit.c and
//     Python/fit_lin.py of github.com/HAWinther/FofrFittingFunction), the limits and the clamping
// The headers are also included in test_emulator_odr.cpp (they must be usable from several translation units)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Emulator/EuclidEmulator2.h>
#include <FML/Emulator/FofrFittingFunction.h>
#include <FML/Global/Global.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <FML/tests/TestHelpers.h>

using FML::TESTS::check_local;

using namespace FML::EMULATOR::EUCLIDEMULATOR2;

int emulator_odr_check(); // defined in test_emulator_odr.cpp


double boost_at(const std::pair<std::vector<double>, std::vector<double>> & b, double k) {
    size_t i = 0;
    while (b.first[i] < k)
        i++;
    return b.second[i];
}

int main() {
    check_local(emulator_odr_check() == 1, "EuclidEmulator2.h can be included in several translation units");

    // Reference values (k = 0.0150042, 0.304633, 1.00213, 5.02027 h/Mpc)
    {
        EuclidEmulator ee2(0.049, 0.315, 0.06, 0.966, 0.674, -1.0, 0.0, 2.1e-9);
        const double ref[2][4] = {{0.99194673605028083, 1.5359928473359081, 5.7588427213863813, 28.809772623082434},
                                  {0.99421122363262493, 1.246734379379111, 3.1113265612912757, 18.598391486773103}};
        double maxerr = 0.0;
        for (int iz = 0; iz < 2; iz++) {
            auto b = ee2.compute_boost(double(iz));
            int ik = 0;
            for (double k : {0.01, 0.3, 1.0, 5.0})
                maxerr = std::max(maxerr, std::fabs(boost_at(b, k) / ref[iz][ik++] - 1.0));
        }
        check_local(maxerr < 1e-10, "boost at z = 0, 1 = reference values");

        // Too many redshifts
        bool thrown = false;
        try {
            ee2.compute_boost(std::vector<double>(200, 0.5), 200);
        } catch (std::exception &) {
            thrown = true;
        }
        check_local(thrown, "more than 101 redshifts throws (buffer overflow before)");
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
        check_local(thrown, "a missing data file throws (segfault before)");
    }

    // Copies
    {
        auto csm = std::make_unique<Cosmology>(0.049, 0.315, 0.06, 0.966, 0.674, -1.0, 0.0, 2.1e-9);
        Cosmology copy = *csm;
        const double ref = csm->compute_step_number(1.0);
        csm.reset();
        Cosmology copy2(0.05, 0.3, 0.0, 0.96, 0.7, -1.0, 0.0, 2.0e-9);
        copy2 = copy;
        check_local(copy.compute_step_number(1.0) == ref and copy2.compute_step_number(1.0) == ref,
                    "copies of Cosmology own their GSL objects (double free before)");
        check_local(not std::is_copy_constructible_v<EuclidEmulator>,
                    "EuclidEmulator cannot be copied (double free before)");
    }

    // The f(R) fitting function (Winther et al. 2019) against the original code
    {
        namespace FIT = FML::EMULATOR::FOFRFITTINGFUNCTION;
        // z, |fR0|, k [h/Mpc], P_f(R)/P_LCDM (C/fit.c and Python/fit_lin.py)
        const double nonlinear_ref[][4] = {{0, 1e-05, 1, 1.17205501084086},
                                           {0.5, 1e-06, 3, 1.02219466187104},
                                           {1, 5e-05, 0.3, 1.10079964711245},
                                           {2, 2e-05, 5, 1.11997793795586},
                                           {0, 7e-06, 10, 1.21708492505093},
                                           {0, 0.0001, 0.1, 1.17918622867476}};
        const double linear_ref[][4] = {
            {0, 1e-05, 0.1, 1.07750324211026}, {1, 1e-06, 1, 1.09700603286284}, {2, 0.0001, 0.05, 1.01266220331046}};
        double err = 0.0, err_lin = 0.0;
        for (const auto & r : nonlinear_ref)
            err = std::max(err, std::fabs(FIT::nonlinear_boost(r[0], r[1], r[2]) / r[3] - 1));
        for (const auto & r : linear_ref)
            err_lin = std::max(err_lin, std::fabs(FIT::linear_boost(r[0], r[1], r[2]) / r[3] - 1));
        check_local(err < 1e-13, "f(R) nonlinear fitting function = the original code");
        check_local(err_lin < 1e-13, "f(R) linear fitting function = the original (Python) code");
        check_local(std::fabs(FIT::nonlinear_boost(0.0, 1e-5, 1e-4) - 1) < 1e-3 and
                        std::fabs(FIT::linear_boost(0.0, 1e-5, 1e-4) - 1) < 1e-3,
                    "f(R) fitting functions -> 1 on large scales");
        check_local(FIT::nonlinear_boost(0.0, 1e-5, 30.0) == FIT::nonlinear_boost(0.0, 1e-5, 10.0) and
                        FIT::nonlinear_boost(5.0, 1e-5, 1.0) == FIT::nonlinear_boost(3.0, 1e-5, 1.0) and
                        FIT::nonlinear_boost(0.0, 1e-3, 1.0) == FIT::nonlinear_boost(0.0, 1e-4, 1.0) and
                        FIT::nonlinear_boost(0.0, -1e-5, 1.0) == FIT::nonlinear_boost(0.0, 1e-5, 1.0),
                    "f(R) fitting function: the closest value outside the fitted range; the sign of fR0 is ignored");
        // The linear fit for effective k < 2 (C/fit_lin.c set k = 1 there with the integer abs)
        check_local(FIT::linear_boost(0.0, 1e-5, 0.5) < FIT::linear_boost(0.0, 1e-5, 1.5),
                    "f(R) linear fitting function increases with k (also for effective k < 2)");
    }

    return FML::TESTS::summary("emulator");
}
