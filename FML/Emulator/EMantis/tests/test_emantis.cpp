// =============================================================================
// Tests of the e-MANTIS emulators (EMantis.h). Run with `make run` in this folder.
//
//   * Against the e-MANTIS Python package (EMantisReference.h, from make_emantis_reference.py): sigma8 (LCDM, wCDM),
//     the f(R) boost (between and at nodes, A_s input, low-k extrapolation, clamped parameters, default bins),
//     the wCDM boost and the HMF with its std for all 16 data files (and extrapolation to high masses), and the
//     range errors (the same parameter as in Python)
//   * The GP sums cancel strongly, so the agreement is ~1e-10 (a few 1e-8 when A_s is converted to sigma8): the
//     floating point reproducibility of the emulators themselves
//   * Consistency: A_s input = sigma8 input with the emulated sigma8, continuity at the nodes, B >= 1 and the
//     low-k rule, the generic GPEmulator, errors
// =============================================================================

#include "EMantisReference.h"
#include <FML/Emulator/EMantis/EMantis.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <map>
#include <memory>

using namespace FML::EMULATOR::EMANTIS;
using FML::TESTS::check;
using FML::TESTS::num;

static double max_rel_diff(const std::vector<double> & a, const std::vector<double> & b) {
    if (a.size() != b.size())
        return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        d = std::max(d, std::abs(a[i] / b[i] - 1.0));
    return d;
}

template <class F>
static bool throws_type(F && f, int type) {
    try {
        f();
    } catch (EmulationRangeError &) {
        return type == 0;
    } catch (std::invalid_argument &) {
        return type == 1;
    } catch (std::runtime_error &) {
        return type == 2;
    }
    return false;
}

int main() {
    NonLinearMGBoostEmulator fr("fR");
    NonLinearBoostEmulator wc("wCDM");
    Sigma8Emulator s8l("LCDM"), s8w("wCDM");
    std::map<std::string, std::unique_ptr<HMFEmulator>> hmfs;

    //=========================================================================
    // Against Python
    //=========================================================================
    std::map<std::string, double> maxerr, maxerr_std;
    std::map<std::string, int> nbad;
    for (auto & c : EMantisReference::cases) {
        Parameters p;
        p.Omega_b = c.Omega_b;
        p.Omega_m = c.Omega_m;
        p.h = c.h;
        p.n_s = c.n_s;
        p.sigma8 = c.sigma8;
        p.A_s = c.A_s;
        p.w = c.w;
        p.logfR0 = c.logfR0;
        const std::string label = c.kind + (c.kind == "hmf" ? " " + c.model + " " + c.extra : "") +
                                  (std::isnan(c.A_s) ? "" : " (A_s)");
        std::vector<double> out, sd;
        std::string err;
        try {
            if (c.kind == "s8")
                out = {(c.model == "LCDM" ? s8l : s8w).sigma8(p)};
            else if (c.kind == "fr")
                out = fr.boost(p, c.a, c.x, c.flag1, c.flag2);
            else if (c.kind == "wc")
                out = wc.boost(p, c.a, c.x);
            else {
                const std::string key = c.model + c.extra;
                if (not hmfs.count(key)) {
                    const auto pos = c.extra.find(':');
                    hmfs[key] = std::make_unique<HMFEmulator>(c.model, c.extra.substr(0, pos), c.extra.substr(pos + 1) == "1");
                }
                out = hmfs[key]->hmf(c.x, p, c.a, c.flag1, &sd);
            }
        } catch (EmulationRangeError & e) {
            err = e.param;
        }
        if (not c.error.empty() or not err.empty()) {
            if (err != c.error) {
                nbad[c.kind]++;
                std::cout << "  " << label << ": Python error '" << c.error << "', C++ '" << err << "'\n";
            }
            continue;
        }
        const double tol = std::isnan(c.A_s) ? 1e-8 : 3e-7;
        const double e = max_rel_diff(out, c.expected);
        const double es = c.expected_std.empty() ? 0.0 : max_rel_diff(sd, c.expected_std);
        const std::string group = c.kind + (std::isnan(c.A_s) ? "" : " (A_s)");
        maxerr[group] = std::max(maxerr[group], e);
        maxerr_std[group] = std::max(maxerr_std[group], es);
        if (not(e < tol) or not(es < 1e-6)) {
            nbad[c.kind]++;
            std::cout << "  " << label << " at a = " << c.a << ": rel. error " << num(e) << " (std " << num(es) << ")\n";
        }
    }
    for (auto & kv : maxerr)
        std::cout << "  " << kv.first << ": max rel. error " << num(kv.second) << ", std " << num(maxerr_std[kv.first]) << "\n";
    check(nbad["s8"] == 0, "sigma8 (LCDM, wCDM) = Python");
    check(nbad["fr"] == 0, "f(R) boost = Python (nodes, A_s, low k, clamping, default bins, range errors)");
    check(nbad["wc"] == 0, "wCDM P_nl/P_lin = Python");
    check(nbad["hmf"] == 0, "HMF and its std = Python, all 16 mass definitions/corrections");

    //=========================================================================
    // Consistency
    //=========================================================================
    Parameters p;
    p.Omega_b = 0.049;
    p.Omega_m = 0.31;
    p.h = 0.68;
    p.n_s = 0.965;
    p.logfR0 = 5.3;
    p.A_s = 2.1e-9;
    std::vector<double> k = {0.01, 0.05, 0.2, 1.0, 5.0};
    {
        auto b1 = fr.boost(p, 0.6, k);
        Parameters q = p;
        q.sigma8 = s8l.sigma8(p);
        q.A_s = NAN;
        auto b2 = fr.boost(q, 0.6, k);
        check(max_rel_diff(b1, b2) == 0.0, "A_s input = sigma8 input with the emulated sigma8");
    }
    {
        Parameters q = p;
        q.sigma8 = 0.81;
        q.A_s = NAN;
        double jump = 0.0;
        for (double a : fr.aexp_nodes()) {
            auto b0 = fr.boost(q, a, k);
            for (double da : {-1e-10, 1e-10}) {
                if (a + da < 0.25 or a + da > 1.0)
                    continue;
                jump = std::max(jump, max_rel_diff(fr.boost(q, a + da, k), b0));
            }
        }
        check(jump < 1e-8, "the boost is continuous at the nodes (max jump " + num(jump) + ")");

        bool ge1 = true, lowk = true;
        const double k0 = fr.range("k").first;
        for (double logfR0 : {4.0, 5.0, 6.0, 7.0})
            for (double a : {0.25, 0.4, 0.77, 1.0}) {
                q.logfR0 = logfR0;
                auto b = fr.boost(q, a, {1e-4, 1e-3, k0});
                for (double x : b)
                    ge1 = ge1 and x >= 1.0;
                lowk = lowk and b[0] <= b[2] and b[1] <= b[2];
            }
        check(ge1 and lowk, "B >= 1, and B(k < k0) <= B(k0)");

        // The generic emulator gives the same as the wrapper
        auto & g = fr.emulator();
        auto b1 = g.predict(std::map<std::string, double>{{"Omega_b", q.Omega_b},
                                                          {"Omega_m", q.Omega_m},
                                                          {"h", q.h},
                                                          {"sigma8_lcdm", q.sigma8},
                                                          {"n_s", q.n_s},
                                                          {"logfR0", q.logfR0}},
                            0.5,
                            {0.1, 1.0});
        auto b2 = fr.boost(q, 0.5, {0.1, 1.0});
        check(max_rel_diff(b1, b2) == 0.0 and g.param_names().size() == 6 and fr.kbins().size() == 90 and
                  fr.aexp_nodes().size() == 22,
              "GPEmulator: parameters by name, bins and nodes");
    }
    {
        auto & h = *hmfs.begin()->second;
        Parameters q = p;
        q.A_s = NAN;
        q.sigma8 = 0.8;
        auto r = h.range("mass_halo", 0.7);
        std::vector<double> sd;
        auto n = h.hmf({r.first * 2, r.second / 2}, q, 0.7, false, &sd);
        check(n[0] > n[1] and sd[0] > 0 and sd[1] > 0 and sd[0] < n[0] and sd[1] < n[1], "HMF decreases with mass, 0 < std < HMF");
    }

    //=========================================================================
    // Errors
    //=========================================================================
    {
        Parameters q = p; // both sigma8 and A_s
        q.sigma8 = 0.8;
        check(throws_type([&] { fr.boost(q, 0.5); }, 1), "giving both sigma8 and A_s throws");
        Parameters r = p;
        r.A_s = NAN;
        check(throws_type([&] { fr.boost(r, 0.5); }, 1), "giving neither sigma8 nor A_s throws");
        Parameters s = p;
        s.A_s = 1e-8;
        check(throws_type([&] { fr.boost(s, 0.5); }, 0), "A_s outside the sigma8 emulator range throws EmulationRangeError");
        check(throws_type([] { HMFEmulator("fR", "300c"); }, 1), "unknown mass definition throws");
        check(throws_type([] { GPEmulator("no_such_file.txt"); }, 2), "missing data file throws");
        check(throws_type([&] { std::vector<double> sd; fr.emulator().predict(std::vector<double>(6, 0.5), 0.5, {}, false, false, &sd); }, 1),
              "the std needs with_std at load");
    }

    return FML::TESTS::summary("EMantis");
}
