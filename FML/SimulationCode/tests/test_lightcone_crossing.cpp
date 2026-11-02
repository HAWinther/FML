// Unit tests of the lightcone crossing (src/Lightcone/LightconeCrossing.h)
#include "../src/Lightcone/LightconeCrossing.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

static int nfail = 0;
static void check(bool ok, const char * what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (not ok)
        nfail++;
}

int main() {
    // r_lc(a) linear in a: the chord is exact
    const auto r_lin = [](double a) { return 1.0 - a; };
    // r_lc(a) = 1/a - 1: the chord is not exact
    const auto r_inv = [](double a) { return 1.0 / a - 1.0; };

    // The step and the fractions of a partial step
    {
        auto s = make_lightcone_step(0.4, 0.6, 0.1, 2.0, r_lin);
        check(s.a_old == 0.4 and s.a_new == 0.6 and s.frac_old == 0.0 and s.frac_new == 1.0, "full step");
        s = make_lightcone_step(0.4, 0.8, 0.5, 0.7, r_lin);
        check(std::abs(s.frac_old - 0.25) < 1e-14 and std::abs(s.frac_new - 0.75) < 1e-14 and s.a_old == 0.5 and
                  s.a_new == 0.7,
              "partial step fractions");
        check(std::abs(s.r_lc_old_squared - 0.25) < 1e-14 and std::abs(s.r_lc_new_squared - 0.09) < 1e-14,
              "partial step radii");
    }

    // A particle at rest at r = 0.5 crosses r_lc = 1 - a at a = 0.5
    {
        const auto s = make_lightcone_step(0.4, 0.6, 0.1, 2.0, r_lin);
        double f = -1, a = -1;
        const bool c = lightcone_crossing<3>({0.3, 0.4, 0.0}, {0.0, 0.0, 0.0}, s, r_lin, f, a);
        check(c and std::abs(a - 0.5) < 1e-14 and std::abs(f - 0.5) < 1e-14, "particle at rest, chord exact");

        // Outside the lightcone at the start and inside it at the end
        check(not lightcone_crossing<3>({0.61, 0.0, 0.0}, {0.0, 0.0, 0.0}, s, r_lin, f, a), "outside at start");
        check(not lightcone_crossing<3>({0.39, 0.0, 0.0}, {0.0, 0.0, 0.0}, s, r_lin, f, a), "inside at end");
    }

    // A radially moving particle: |x0 + f dx| = r_lc(a) is linear in f, so the chord is exact
    {
        const auto s = make_lightcone_step(0.4, 0.6, 0.1, 2.0, r_lin);
        double f = -1, a = -1;
        // r(f) = 0.45 + 0.1 f, r_lc(f) = 0.6 - 0.2 f: f = 0.5
        const bool c = lightcone_crossing<2>({0.45, 0.0}, {0.1, 0.0}, s, r_lin, f, a);
        check(c and std::abs(f - 0.5) < 1e-14 and std::abs(a - 0.5) < 1e-14, "moving particle, chord exact");
    }

    // The lightcone ends inside the step and the chord puts the crossing after it: bisection finds the crossing
    {
        const double a_true = 0.57;
        const double r = r_inv(a_true);
        const auto s = make_lightcone_step(0.4, 0.8, 0.55, 0.6, r_inv);
        double f = -1, a = -1;
        const bool c = lightcone_crossing<3>({r, 0.0, 0.0}, {0.0, 0.0, 0.0}, s, r_inv, f, a);
        check(c and std::abs(a - a_true) < 1e-10 and f >= s.frac_old and f <= s.frac_new, "partial step, bisection");

        // In the same full step the chord is used (approximate)
        const auto sfull = make_lightcone_step(0.4, 0.8, 0.1, 2.0, r_inv);
        const bool cfull = lightcone_crossing<3>({r, 0.0, 0.0}, {0.0, 0.0, 0.0}, sfull, r_inv, f, a);
        const double a_chord = 0.4 + 0.4 * (r_inv(0.4) - r) / (r_inv(0.4) - r_inv(0.8));
        check(cfull and std::abs(a - a_chord) < 1e-14, "full step, chord");
    }

    // Random particles: a crossing iff inside r_lc_old at a_old and outside r_lc_new at a_new, and the crossing (in a
    // partial step) inside [frac_old, frac_new] with |x + f dx| = r_lc(a)
    {
        std::mt19937 gen(1234);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        const auto s = make_lightcone_step(0.3, 0.9, 0.4, 0.8, r_inv);
        int ncross = 0, nbad = 0;
        for (int i = 0; i < 20000; i++) {
            std::array<double, 3> x, dx;
            for (int d = 0; d < 3; d++) {
                x[d] = 2.0 * u(gen);
                dx[d] = 0.2 * u(gen);
            }
            double r2o = 0, r2n = 0;
            for (int d = 0; d < 3; d++) {
                r2o += std::pow(x[d] + dx[d] * s.frac_old, 2);
                r2n += std::pow(x[d] + dx[d] * s.frac_new, 2);
            }
            const bool expected = r2o <= s.r_lc_old_squared and r2n > s.r_lc_new_squared;
            double f = -1, a = -1;
            const bool c = lightcone_crossing<3>(x, dx, s, r_inv, f, a);
            if (c != expected)
                nbad++;
            if (c) {
                ncross++;
                if (f < s.frac_old or f > s.frac_new or std::abs(a - (s.apos + (s.apos_new - s.apos) * f)) > 1e-14)
                    nbad++;
            }
        }
        check(nbad == 0 and ncross > 100, "random particles, partial step");
    }

    if (nfail > 0) {
        std::printf("%d test(s) FAILED\n", nfail);
        return EXIT_FAILURE;
    }
    std::printf("All lightcone crossing tests passed\n");
    return EXIT_SUCCESS;
}
