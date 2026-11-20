// =============================================================================
// Tests of the baccoemu port (Bacco.h). Run with `make run` in this folder.
//
//   * Against the baccoemu Python package (BaccoReference.h, from make_bacco_reference.py, with the networks
//     evaluated in float64): sigma8 / sigma12 / A_s, the linear P(k) (cold and total, default bins and given k),
//     the no-wiggle and BAO-smeared P(k), the nonlinear boost and P_nl (Angulo2021 and Arico2023, cold and total
//     matter with neutrinos, no-wiggle or smeared-BAO emulator, given linear P(k)), the Arico2021 and Burger2025
//     baryonic boosts, the bispectrum baryonic boost, and the range errors (the same parameter as in Python)
//   * The agreement is ~1e-15 (baccoemu itself evaluates in float32, which moves its results by up to ~1.5e-6)
//   * Consistency: A_s input = sigma8_cold input, Omega_m input = Omega_cold input, errors
// =============================================================================

#include "BaccoReference.h"
#include <FML/Emulator/Bacco/Bacco.h>
#include <FML/tests/TestHelpers.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>

using namespace FML::EMULATOR::BACCO;
using FML::TESTS::check;

static double max_rel_diff(const std::vector<double> & a, const std::vector<double> & b) {
    if (a.size() != b.size())
        return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        d = std::max(d, std::abs(a[i] / b[i] - 1.0));
    return d;
}

static double max_scaled_diff(const std::vector<double> & a, const std::vector<double> & b, double scale) {
    if (a.size() != b.size())
        return INFINITY;
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        d = std::max(d, std::abs(a[i] - b[i]) / scale);
    return d;
}

static Parameters params(const std::vector<double> & v) {
    Parameters p;
    double * f[] = {&p.Omega_cold, &p.Omega_m, &p.Omega_b, &p.sigma8_cold, &p.A_s, &p.h, &p.n_s, &p.M_nu, &p.w0,
                    &p.wa, &p.M_c, &p.eta, &p.beta, &p.M1_z0_cen, &p.theta_out, &p.theta_inn, &p.M_inn};
    for (size_t i = 0; i < v.size(); i++)
        *f[i] = v[i];
    return p;
}

int main() {
    // Without the data (not distributed with FML): a MissingDataError that says how to get it
    {
        const std::string dir = data_dir;
        set_data_dir("/nonexistent/bacco/data");
        bool ok = false;
        try {
            MatterPowerSpectrum e;
        } catch (MissingDataError & e) {
            ok = std::string(e.what()).find("get_data.sh") != std::string::npos;
        }
        data_dir = dir;
        check(ok, "missing data: MissingDataError pointing to get_data.sh");
    }
    if (not data_available(true)) {
        std::cout << "The baccoemu data is not in " << data_dir << ": run ../get_data.sh to test the emulators\n";
        return FML::TESTS::summary("Bacco (without data)");
    }

    // The emulators, by options string
    std::map<std::string, std::unique_ptr<MatterPowerSpectrum>> emus;
    auto get = [&](const std::string & opts) -> const MatterPowerSpectrum & {
        auto & e = emus[opts];
        if (not e) {
            MatterPowerSpectrum::Options o;
            o.nonlinear_model = opts.find("Arico2023") != std::string::npos ? "Arico2023" : "Angulo2021";
            o.smeared_bao = opts.find("smeared_bao") != std::string::npos;
            o.no_wiggles = not o.smeared_bao;
            o.baryonic_model = opts.find("Burger2025") != std::string::npos ? "Burger2025" : "Arico2021";
            auto t0 = std::chrono::steady_clock::now();
            e = std::make_unique<MatterPowerSpectrum>(o);
            auto t1 = std::chrono::steady_clock::now();
            std::cout << "Loaded [" << opts << "] in " << std::chrono::duration<double>(t1 - t0).count() << " s\n";
        }
        return *e;
    };
    MatterBispectrum bk;
    std::unique_ptr<LbiasExpansion> lb;
    std::unique_ptr<LbiasExpansionRSD> lbz;
    auto flat = [](const std::vector<std::vector<double>> & v) {
        std::vector<double> r;
        for (auto & x : v)
            r.insert(r.end(), x.begin(), x.end());
        return r;
    };

    //=========================================================================
    // Against Python
    //=========================================================================
    std::map<std::string, double> maxerr;
    std::map<std::string, int> ncases, nbad;
    for (auto & c : BaccoReference::cases) {
        const auto p = params(c.params);
        const std::string label = c.kind + (c.options.empty() ? "" : " [" + c.options + "]");
        std::vector<double> got;
        std::string err;
        try {
            const auto & E = get(c.options);
            if (c.kind == "sigma8")
                got = {E.sigma8(p, c.cold)};
            else if (c.kind == "sigma12")
                got = {E.sigma12(p, c.cold)};
            else if (c.kind == "A_s")
                got = {E.A_s(p)};
            else if (c.kind == "linear")
                got = E.linear_pk(p, c.a, c.k, c.cold);
            else if (c.kind == "no_wiggles")
                got = E.no_wiggles_pk(p, c.a, c.k);
            else if (c.kind == "smeared_bao")
                got = E.smeared_bao_pk(p, c.a, c.k);
            else if (c.kind == "boost")
                got = E.nonlinear_boost(p, c.a, c.k, c.cold, c.k_lin, c.pk_lin);
            else if (c.kind == "pk_nl")
                got = E.nonlinear_pk(p, c.a, c.k, c.cold);
            else if (c.kind == "pk_nl_baryons")
                got = E.nonlinear_pk(p, c.a, c.k, c.cold, true);
            else if (c.kind == "baryon")
                got = E.baryonic_boost(p, c.a, c.k);
            else if (c.kind.rfind("lbias", 0) == 0) {
                if (not lb) {
                    auto t0 = std::chrono::steady_clock::now();
                    lb = std::make_unique<LbiasExpansion>();
                    std::cout << "Loaded LbiasExpansion in "
                              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
                }
                if (c.kind == "lbias_lpt")
                    got = flat(lb->lpt_pk(p, c.a, c.k));
                else if (c.kind == "lbias_bao")
                    got = lb->smeared_bao_pk(p, c.a, c.k);
                else if (c.kind == "lbias_pnn")
                    got = flat(lb->nonlinear_pnn(p, c.a, c.k));
                else if (c.kind == "lbias_galaxy") {
                    auto r = lb->galaxy_real_pk({c.extra[0], c.extra[1], c.extra[2], c.extra[3]}, p, c.a, c.k);
                    got = flat({r.auto_pk, r.cross_pk});
                }
            } else if (c.kind.rfind("rsd", 0) == 0) {
                if (not lbz) {
                    auto t0 = std::chrono::steady_clock::now();
                    lbz = std::make_unique<LbiasExpansionRSD>();
                    std::cout << "Loaded LbiasExpansionRSD in "
                              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s\n";
                }
                auto flat3 = [&](const LbiasExpansionRSD::Multipoles & m) {
                    return flat({flat(m[0]), flat(m[1]), flat(m[2])});
                };
                LbiasExpansionRSD::RSDParameters r;
                std::array<double, 4> b{};
                if (c.extra.size() == 10) {
                    b = {c.extra[0], c.extra[1], c.extra[2], c.extra[3]};
                    r.f_sat = c.extra[4];
                    r.lambda_FoG = c.extra[5];
                    r.epsilon_1 = c.extra[6];
                    r.epsilon_2 = c.extra[7];
                    r.epsilon_3 = c.extra[8];
                    r.mean_num_dens = c.extra[9];
                }
                if (c.kind == "rsd_lpt")
                    got = flat3(lbz->lpt_pk(p, c.a, c.k));
                else if (c.kind == "rsd_bao") {
                    auto m = lbz->smeared_bao_pk(p, c.a, c.k);
                    got = flat({m[0], m[1], m[2]});
                } else if (c.kind == "rsd_pnn")
                    got = flat3(lbz->nonlinear_pnn(p, c.a, c.k));
                else if (c.kind == "rsd_galaxy") {
                    auto m = lbz->galaxy_pk(b, r, p, c.a, c.k);
                    got = flat({m[0], m[1], m[2]});
                } else if (c.kind == "rsd_galaxy_2d")
                    got = flat(lbz->galaxy_2d_pk(b, r, p, c.a, c.k));
            } else if (c.kind == "bispectrum") {
                auto r = bk.baryonic_boost(p, c.a, c.k1, c.k2, c.k3);
                got = r.boost;
                std::vector<double> kflat;
                for (auto & t : r.k)
                    kflat.insert(kflat.end(), t.begin(), t.end());
                if (max_rel_diff(kflat, c.k_out) > 0)
                    err = "triangles differ";
            }
        } catch (EmulationRangeError & e) {
            err = e.param;
        } catch (std::exception & e) {
            err = std::string("exception: ") + e.what();
        }
        ncases[label]++;
        if (not c.error.empty() or not err.empty()) {
            if (err != c.error) {
                nbad[label]++;
                std::cout << "  " << label << ": error '" << err << "', expected '" << c.error << "'\n";
            }
            continue;
        }
        // Bias spectra cross zero: relative to the largest |value| of the case (of the whole output: the reference
        // holds only every stride-th value, to keep BaccoReference.h small)
        const bool bias_case = c.kind.rfind("lbias", 0) == 0 or c.kind.rfind("rsd", 0) == 0;
        double scale = 0.0;
        for (double x : got)
            scale = std::max(scale, std::abs(x));
        std::vector<double> kept;
        for (size_t i = 0; i < got.size(); i += c.stride)
            kept.push_back(got[i]);
        const double d = got.size() != c.size ? INFINITY
                         : bias_case          ? max_scaled_diff(kept, c.expected, scale)
                                              : max_rel_diff(kept, c.expected);
        maxerr[label] = std::max(maxerr[label], d);
        // Burger2025 in Python takes float32 inputs: the default k bins are rounded to float32
        const double tol = (c.kind == "baryon" and c.options == "Burger2025" and c.k.empty()) ? 1e-7
                           : bias_case                                                   ? 1e-10
                                                                                         : 1e-12;
        if (not(d < tol)) {
            nbad[label]++;
            std::cout << "  " << label << " a = " << c.a << " cold = " << c.cold << ": max rel diff " << d << "\n";
        }
    }
    for (auto & [label, n] : ncases) {
        std::ostringstream s;
        s << "Python: " << label << " (" << n << " cases, max rel diff " << maxerr[label] << ")";
        check(nbad[label] == 0, s.str());
    }

    //=========================================================================
    // Consistency
    //=========================================================================
    const auto & E = get("");
    Parameters p;
    p.Omega_cold = 0.30;
    p.Omega_b = 0.048;
    p.sigma8_cold = 0.80;
    p.h = 0.68;
    p.n_s = 0.96;
    p.M_nu = 0.1;
    std::vector<double> k = {0.01, 0.1, 0.5, 1.0, 3.0};
    {
        Parameters q = p;
        q.sigma8_cold = NAN;
        q.A_s = E.A_s(p);
        check(std::abs(E.sigma8(q) - 0.80) < 1e-12, "sigma8_cold(A_s(sigma8_cold)) = sigma8_cold");
        check(max_rel_diff(E.nonlinear_pk(q, 0.8, k), E.nonlinear_pk(p, 0.8, k)) < 1e-12,
              "A_s input = sigma8_cold input");
        Parameters r = p;
        r.Omega_cold = NAN;
        r.Omega_m = p.Omega_cold + p.M_nu / 93.14 / (p.h * p.h);
        check(max_rel_diff(E.nonlinear_pk(r, 0.8, k, false), E.nonlinear_pk(p, 0.8, k, false)) < 1e-12,
              "Omega_m input = Omega_cold input");
    }
    {
        auto b = E.nonlinear_boost(p, 1.0, {1e-3, 5e-3});
        check(b[0] == 1.0 and b[1] == 1.0, "nonlinear boost = 1 below the bins");
        auto pnl = E.nonlinear_pk(p, 1.0, k), pl = E.linear_pk(p, 1.0, k);
        check(std::abs(pnl[0] / pl[0] - 1) < 0.01 and pnl[4] > 3 * pl[4], "P_nl ~ P_lin on large scales, > on small");
    }
    auto throws = [](auto && f) {
        try {
            f();
        } catch (std::exception &) {
            return true;
        }
        return false;
    };
    check(throws([&] { E.linear_pk(p, 1.0, {1e-6}); }), "linear_pk: k outside the bins throws");
    check(throws([&] { E.nonlinear_pk(p, 1.0, {100.0}); }), "nonlinear: k above the bins throws");
    check(throws([&] {
              Parameters q = p;
              q.A_s = 1e-9;
              E.nonlinear_pk(q, 1.0, k);
          }),
          "inconsistent sigma8_cold and A_s throws");
    check(throws([&] {
              Parameters q = p;
              q.Omega_m = 0.5;
              E.linear_pk(q, 1.0);
          }),
          "inconsistent Omega_cold and Omega_m throws");
    check(throws([&] { E.baryonic_boost(p, 1.0, k); }), "baryonic boost without the baryon parameters throws");

    // Timing
    {
        auto t0 = std::chrono::steady_clock::now();
        const int n = 20;
        for (int i = 0; i < n; i++)
            E.nonlinear_pk(p, 0.5 + 0.02 * i, k);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "nonlinear_pk: " << 1e3 * std::chrono::duration<double>(t1 - t0).count() / n << " ms\n";
    }

    return FML::TESTS::summary("Bacco");
}
