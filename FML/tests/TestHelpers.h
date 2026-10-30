#ifndef FML_TESTS_TESTHELPERS_HEADER
#define FML_TESTS_TESTHELPERS_HEADER

// Shared helpers for the FML test programs: pass/fail checks, a summary with the exit code, tolerance comparisons
// and exception checks. Only the tests include this header; the library does not depend on it. Everything is inline
// so a test may include it from several translation units.
//
//   check(ok, label)        collective: every task must call it; fails on every task if it fails on any task
//   check_local(ok, label)  no communication: may be called on some tasks only (e.g. inside if (ThisTask == 0))
//   check_close / check_rel tolerance checks that also print the error (collective)
//   return summary("name"); at the end of main (collective): prints the result on task 0, returns the exit code

#include <FML/Global/Global.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace FML {
    namespace TESTS {

        namespace detail {
            inline int n_checks_collective = 0; // Collective checks made (the same on every task)
            inline int n_checks_local = 0;      // Local checks made on this task
            inline int n_failed_collective = 0; // Failed collective checks (the same on every task)
            inline int n_failed_local = 0;      // Failed local checks on this task
        } // namespace detail

        /// Collective check: every task must call it (with the same label). The check fails on every task if ok is
        /// false on any task. Printed on task 0. Returns the combined result.
        inline bool check(bool ok, const std::string & label) {
            int bad = ok ? 0 : 1;
            FML::MaxOverTasks(&bad);
            ++detail::n_checks_collective;
            detail::n_failed_collective += bad;
            if (FML::ThisTask == 0)
                std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
            return bad == 0;
        }

        /// Local check without communication: it may be called on a subset of the tasks. Printed on task 0; a
        /// failure on another task is printed by that task. Returns ok.
        inline bool check_local(bool ok, const std::string & label) {
            ++detail::n_checks_local;
            if (!ok)
                ++detail::n_failed_local;
            if (FML::ThisTask == 0)
                std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << '\n';
            else if (!ok)
                std::cout << "[FAIL] (task " << FML::ThisTask << ") " << label << '\n';
            return ok;
        }

        /// Number of failed checks seen by this task so far (collective failures plus this task's local ones)
        inline int failures() { return detail::n_failed_collective + detail::n_failed_local; }

        /// Collective: prints the result on task 0 and returns the exit code for main (0 if every check passed on
        /// every task, otherwise 1). Local checks and failures are summed over the tasks.
        inline int summary(const std::string & name) {
            int failed = detail::n_failed_local, checks = detail::n_checks_local;
            FML::SumOverTasks(&failed);
            FML::SumOverTasks(&checks);
            failed += detail::n_failed_collective;
            checks += detail::n_checks_collective;
            if (FML::ThisTask == 0) {
                if (failed == 0)
                    std::cout << "All " << name << " tests passed (" << checks << " checks)\n";
                else
                    std::cout << name << ": " << failed << " of " << checks << " checks FAILED\n";
            }
            return failed == 0 ? 0 : 1;
        }

        /// A number as text, for check labels
        inline std::string num(double x, int precision = 6) {
            std::ostringstream os;
            os << std::setprecision(precision) << x;
            return os.str();
        }

        /// |a - b| / max(floor, |a|, |b|): the relative difference, absolute when both are below floor. Infinite
        /// if either value is not finite.
        inline double relative_difference(double a, double b, double floor = 1.0) {
            if (!std::isfinite(a) || !std::isfinite(b))
                return INFINITY;
            return std::abs(a - b) / std::max({floor, std::abs(a), std::abs(b)});
        }

        /// relative_difference(a, b, floor) <= tolerance (false for non-finite values)
        inline bool near(double a, double b, double tolerance, double floor = 1.0) {
            return relative_difference(a, b, floor) <= tolerance;
        }

        /// Collective check of |value - expected| / max(|expected|, 1) <= tolerance; the label shows the error
        inline bool check_close(double value, double expected, double tolerance, const std::string & label) {
            const double err = std::fabs(value - expected) / std::max(std::fabs(expected), 1.0);
            const bool ok = err <= tolerance; // also false for NaN
            std::ostringstream os;
            os << std::left << std::setw(52) << label << std::scientific << std::setprecision(3) << " err = " << err
               << " (tol " << tolerance << ")";
            if (!ok)
                os << std::defaultfloat << std::setprecision(10) << "  value = " << value
                   << ", expected = " << expected;
            return check(ok, os.str());
        }

        /// Collective check of |value / expected - 1| <= tolerance; the label shows the error
        inline bool check_rel(double value, double expected, double tolerance, const std::string & label) {
            const double err = std::fabs(value / expected - 1.0);
            const bool ok = err <= tolerance; // also false for NaN
            std::ostringstream os;
            os << std::left << std::setw(52) << label << std::scientific << std::setprecision(3) << " rel = " << err
               << " (tol " << tolerance << ")";
            if (!ok)
                os << std::defaultfloat << std::setprecision(10) << "  value = " << value
                   << ", expected = " << expected;
            return check(ok, os.str());
        }

        /// True if f() throws an exception of type E (or derived from it)
        template <class E = std::exception, class F>
        bool throws(F && f) {
            try {
                f();
            } catch (const E &) {
                return true;
            }
            return false;
        }

        /// True if f() throws an exception of type E whose message contains text
        template <class E = std::exception, class F>
        bool throws_with(F && f, const std::string & text) {
            try {
                f();
            } catch (const E & e) {
                return std::string(e.what()).find(text) != std::string::npos;
            }
            return false;
        }

        /// Run f with std::cout silenced
        template <class F>
        void quietly(F && f) {
            std::ostringstream sink;
            auto * old = std::cout.rdbuf(sink.rdbuf());
            try {
                f();
            } catch (...) {
                std::cout.rdbuf(old);
                throw;
            }
            std::cout.rdbuf(old);
        }

        /// Minimal particle with a position only
        template <int NDIM, class T = double>
        struct PositionParticle {
            T pos[NDIM];
            T * get_pos() { return pos; }
            constexpr int get_ndim() const { return NDIM; }
        };

        /// Minimal particle with a position and a velocity
        template <int NDIM, class T = double>
        struct PositionVelocityParticle {
            T pos[NDIM];
            T vel[NDIM];
            T * get_pos() { return pos; }
            T * get_vel() { return vel; }
            constexpr int get_ndim() const { return NDIM; }
        };

    } // namespace TESTS
} // namespace FML

#endif
