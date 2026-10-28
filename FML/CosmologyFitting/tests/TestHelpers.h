#ifndef TESTHELPERS_H
#define TESTHELPERS_H

// Minimal test helpers shared by the unit tests

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace Test {
    inline int n_checks = 0, n_failed = 0;

    // Relative error |value - expected| / max(|expected|, 1)  (absolute for |expected| < 1)
    inline void check_close(const std::string & what, double value, double expected, double tol) {
        n_checks++;
        const double err = std::fabs(value - expected) / std::max(std::fabs(expected), 1.0);
        const bool ok = err <= tol; // also false for NaN
        if (!ok)
            n_failed++;
        std::cout << (ok ? "  [ OK ] " : "  [FAIL] ") << std::left << std::setw(52) << what << std::scientific
                  << std::setprecision(3) << " err = " << err << " (tol " << tol << ")" << std::defaultfloat;
        if (!ok)
            std::cout << "  value = " << std::setprecision(10) << value << ", expected = " << expected;
        std::cout << "\n";
    }

    // Pure relative error |value/expected - 1|
    inline void check_rel(const std::string & what, double value, double expected, double tol) {
        n_checks++;
        const double err = std::fabs(value / expected - 1.0);
        const bool ok = err <= tol;
        if (!ok)
            n_failed++;
        std::cout << (ok ? "  [ OK ] " : "  [FAIL] ") << std::left << std::setw(52) << what << std::scientific
                  << std::setprecision(3) << " rel = " << err << " (tol " << tol << ")" << std::defaultfloat;
        if (!ok)
            std::cout << "  value = " << std::setprecision(10) << value << ", expected = " << expected;
        std::cout << "\n";
    }

    inline void check(const std::string & what, bool ok) {
        n_checks++;
        if (!ok)
            n_failed++;
        std::cout << (ok ? "  [ OK ] " : "  [FAIL] ") << what << "\n";
    }

    inline int summary() {
        std::cout << "\n" << (n_checks - n_failed) << " / " << n_checks << " checks passed\n";
        return n_failed == 0 ? 0 : 1;
    }
} // namespace Test

#endif
