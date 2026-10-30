#include <FML/Global/Global.h>
#include <FML/PerturbationTheory/FastOneLoop.h>
#include <array>
#include <chrono>
#include <iostream>
using namespace FML::PERTURBATION;
int failures = 0;
void check(bool ok, const std::string & label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool near(double a, double b, double relative = 1e-6, double absolute = 1e-15) {
    return std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}
// Eisenstein & Hu (1998) no-wiggle LCDM-like spectrum [k in h/Mpc, P in (Mpc/h)^3], h = 0.67, Om = 0.31,
// Ob = 0.049, ns = 0.965, Tcmb = 2.7255
double eisenstein_hu(double k) {
    const double h = .67, om = .31, ob = .049, ns = .965, theta = 2.7255 / 2.7;
    const double wm = om * h * h, wb = ob * h * h, fb = ob / om;
    const double s = 44.5 * std::log(9.83 / wm) / std::sqrt(1 + 10 * std::pow(wb, .75));
    const double ag = 1 - .328 * std::log(431 * wm) * fb + .38 * std::log(22.3 * wm) * fb * fb;
    const double gamma = om * h * (ag + (1 - ag) / (1 + std::pow(.43 * k * h * s, 4)));
    const double q = k * theta * theta / gamma;
    const double l0 = std::log(2 * std::exp(1.) + 1.8 * q), c0 = 14.2 + 731 / (1 + 62.5 * q);
    const double t = l0 / (l0 + c0 * q * q);
    return 4e6 * std::pow(k, ns) * t * t;
}
// Fully symmetrized EdS F3(a, b, c) from the standard recursion (independent of the analytic P13 kernel)
using V3 = std::array<double, 3>;
V3 add(V3 a, V3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
double dot3(V3 a, V3 b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double alpha(V3 a, V3 b) { return dot3(add(a, b), a) / dot3(a, a); }
double beta(V3 a, V3 b) {
    V3 s = add(a, b);
    return dot3(s, s) * dot3(a, b) / (2 * dot3(a, a) * dot3(b, b));
}
double f2s(V3 a, V3 b) { return (5 * (alpha(a, b) + alpha(b, a)) + 4 * beta(a, b)) / 14; }
double g2s(V3 a, V3 b) { return (3 * (alpha(a, b) + alpha(b, a)) + 8 * beta(a, b)) / 14; }
double f3u(V3 a, V3 b, V3 c) {
    V3 bc = add(b, c), ab = add(a, b);
    double t = 0;
    if (dot3(bc, bc) > 1e-300)
        t += 7 * alpha(a, bc) * f2s(b, c) + 2 * beta(a, bc) * g2s(b, c);
    if (dot3(ab, ab) > 1e-300)
        t += g2s(a, b) * (7 * alpha(ab, c) + 2 * beta(ab, c));
    return t / 18;
}
double f3s(V3 a, V3 b, V3 c) {
    return (f3u(a, b, c) + f3u(a, c, b) + f3u(b, a, c) + f3u(b, c, a) + f3u(c, a, b) + f3u(c, b, a)) / 6;
}
Vector grid(size_t n) {
    Vector k(n);
    for (size_t i = 0; i < n; ++i)
        k[i] = std::exp(std::log(1e-5) + double(i) / (n - 1) * std::log(1e7));
    return k;
}
int main() {
    check(near(p13_kernel(1), -88) && near(p13_kernel(1e-8), -168) && near(p13_kernel(1e8), -488. / 5) &&
              near(p13_kernel(1 - 1e-7), p13_kernel(1 + 1e-7), 1e-5),
          "P13 kernel is stable at unity and both analytic asymptotes");
    Spectrum power = [](double q) { return q * std::exp(-q * q); };
    auto k = grid(1024);
    Vector p;
    for (double q : k)
        p.push_back(power(q));
    FastEdS workspace;
    auto start = std::chrono::steady_clock::now();
    auto fast = workspace.evaluate(k, p, .1, 2.1);
    double fast_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::vector<size_t> indices;
    Vector output;
    for (double wanted : {.1, .2, .5, 1., 2.}) {
        size_t i = size_t(std::lower_bound(fast.k.begin(), fast.k.end(), wanted) - fast.k.begin());
        indices.push_back(i);
        output.push_back(fast.k[i]);
    }
    auto reference = direct_one_loop(output, power, k.front(), k.back());
    bool ok = true;
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t j = indices[i];
        ok &= near(fast.p22[j], reference.p22[i], .012) && near(fast.p13[j], reference.p13[i], 2e-5) &&
              near(fast.total[j], fast.linear[j] + fast.loop[j]);
    }
    check(ok && fast.validation_k.size() == 3 && fast.max_reference_relative_error < .02,
          "FFTLog P22 and FFT P13 agree with independent direct q,p quadrature for a Gaussian-cutoff spectrum");
#ifdef USE_OMP
    std::vector<std::unique_ptr<FastEdS>> independent;
    for (int i = 0; i < 2; ++i) {
        independent.push_back(std::make_unique<FastEdS>());
        independent.back()->evaluate(k, p, .1, 2.1);
    }
    int concurrent_bad = 0;
#pragma omp parallel for reduction(+ : concurrent_bad)
    for (int i = 0; i < 2; ++i) {
        try {
            auto prediction = independent[i]->evaluate(k, p, .1, 2.1);
            if (!near(prediction.p13[indices[2]], fast.p13[indices[2]], 1e-10))
                ++concurrent_bad;
        } catch (...) {
            ++concurrent_bad;
        }
    }
    check(concurrent_bad == 0,
          "separately preplanned FFT workspaces execute concurrently without shared mutable state");
#endif
    auto builds = workspace.convolution_plan_builds();
    Vector scaled = p;
    for (auto & x : scaled)
        x *= 3;
    auto amplitude = workspace.evaluate(k, scaled, .1, 2.1);
    ok = true;
    for (size_t j : indices)
        ok &= near(amplitude.p22[j], 9 * fast.p22[j], .012) && near(amplitude.p13[j], 9 * fast.p13[j], 2e-8) &&
              near(amplitude.linear[j], 3 * fast.linear[j]);
    check(ok && workspace.convolution_plan_builds() == builds,
          "growth/amplitude scaling is quadratic in the loops and reuses convolution plans");
    Spectrum shaped = [](double q) { return q / std::pow(1 + q * q / .04, 2) * std::exp(-q * q / 100); };
    Vector shape;
    for (double q : k)
        shape.push_back(shaped(q));
    auto other = workspace.evaluate(k, shape, .1, 1.1);
    Vector second;
    for (size_t i : {size_t(0), other.k.size() / 2, other.k.size() - 1})
        second.push_back(other.k[i]);
    auto shape_ref = direct_one_loop(second, shaped, k.front(), k.back());
    ok = true;
    for (size_t i = 0; i < second.size(); ++i) {
        size_t j = size_t(std::lower_bound(other.k.begin(), other.k.end(), second[i]) - other.k.begin());
        ok &= near(other.p22[j], shape_ref.p22[i], .02) && near(other.p13[j], shape_ref.p13[i], .001);
    }
    check(ok, "a smooth turnover/UV-cutoff spectrum agrees with direct quadrature");
    auto refined_k = grid(2048);
    Vector refined_p;
    for (double q : refined_k)
        refined_p.push_back(power(q));
    FastEdS refined_workspace;
    auto refined = refined_workspace.evaluate(refined_k, refined_p, .2, 2.1);
    Vector refined_points;
    for (size_t i : {size_t(0), refined.k.size() / 2, refined.k.size() - 1})
        refined_points.push_back(refined.k[i]);
    auto refined_ref = direct_one_loop(refined_points, power, refined_k.front(), refined_k.back());
    ok = true;
    for (size_t i = 0; i < refined_points.size(); ++i) {
        size_t j = size_t(std::lower_bound(refined.k.begin(), refined.k.end(), refined_points[i]) - refined.k.begin());
        ok &= near(refined.p22[j], refined_ref.p22[i], .01) && near(refined.p13[j], refined_ref.p13[i], 1e-5);
    }
    check(ok, "resolution and logarithmic-padding change retains direct-reference agreement in the tested interior");
    auto low = direct_one_loop({.005, .01}, power, 1e-6, 20);
    double coefficient = 9 / (196 * detail::pi * detail::pi) * std::sqrt(detail::pi) / (2 * std::sqrt(2.));
    check(near(low.p22[0] / std::pow(.005, 4), coefficient, .003) &&
              near(low.p22[1] / std::pow(.01, 4), coefficient, .003),
          "direct P22 approaches the independently derived k^4 mass-conservation limit");
    double p13_limit = -61 / (1260 * detail::pi * detail::pi) * .005 * .005 * power(.005);
    check(near(low.p13[0], p13_limit, .002), "direct P13 approaches the analytic displacement-variance limit");
    DirectOptions fine;
    fine.order = 20;
    fine.log_step = .08;
    auto converged = direct_one_loop(output, power, k.front(), k.back(), fine);
    ok = true;
    for (size_t i = 0; i < output.size(); ++i)
        ok &= near(reference.p22[i], converged.p22[i], 2e-4) && near(reference.p13[i], converged.p13[i], 2e-4);
    check(ok, "direct quadrature converges under higher order and finer segmentation");
    Kernels custom;
    custom.name = "scaled test kernels";
    custom.f2 = [](double q, double p, double mu) { return 2 * eds_f2(q, p, mu); };
    custom.f3_collapsed = [](double k, double q, double) {
        double r = q / k;
        return 3 * p13_kernel(r) / (3024 * r * r);
    };
    auto modified = direct_one_loop(output, power, k.front(), k.back(), {}, custom);
    ok = true;
    for (size_t i = 0; i < output.size(); ++i)
        ok &= near(modified.p22[i], 4 * reference.p22[i], 1e-8) && near(modified.p13[i], 3 * reference.p13[i], 1e-8);
    check(ok && modified.kernels == custom.name,
          "custom scale-dependent kernel callbacks stay in direct quadrature with the correct F2/F3 normalization");
    bool rejected = false;
    try {
        workspace.evaluate(k, p, .01, .08);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    check(rejected, "an inaccurate low-k FFTLog tail fails direct-reference checks instead of returning a prediction");
    auto bad = k;
    bad[500] *= 1.001;
    rejected = false;
    try {
        workspace.evaluate(bad, p, .1, 2);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "nonlogarithmic input is rejected");
    rejected = false;
    Kernels incomplete;
    incomplete.f2 = custom.f2;
    try {
        direct_one_loop(output, power, 1e-5, 100, {}, incomplete);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "incomplete custom kernel sets are rejected rather than silently using EdS");
    start = std::chrono::steady_clock::now();
    auto all_direct = direct_one_loop(fast.k, power, k.front(), k.back());
    double direct_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ok = true;
    for (size_t i = 0; i < fast.k.size(); ++i)
        ok &= near(fast.p22[i], all_direct.p22[i], .012) && near(fast.p13[i], all_direct.p13[i], 2e-5);
    check(ok, "every requested Gaussian-spectrum mode agrees with direct quadrature, beyond the built-in spot checks");
    {
        // LCDM-like spectrum on 1e-5..50 h/Mpc: direct quadrature against FAST-PT 4.1.1 (one_loop_dd, n_pad = 1024,
        // C_window = 0.75, no extrapolation) for the same tabulated spectrum, and the fast path against direct
        Vector lk(2048), lp(2048);
        for (size_t i = 0; i < lk.size(); ++i) {
            lk[i] = std::exp(std::log(1e-5) + double(i) / (lk.size() - 1) * std::log(50 / 1e-5));
            lp[i] = eisenstein_hu(lk[i]);
        }
        auto table = [&](double q) {
            if (q < lk.front() || q > lk.back())
                return 0.;
            auto hi = std::lower_bound(lk.begin(), lk.end(), q);
            if (hi == lk.begin())
                return lp.front();
            const size_t j = size_t(hi - lk.begin());
            const double t = std::log(q / lk[j - 1]) / std::log(lk[j] / lk[j - 1]);
            return lp[j - 1] + t * (lp[j] - lp[j - 1]);
        };
        const Vector kref{.02, .05, .1, .2, .5, 1.};
        const Vector fastpt{-2.406801e+02, -3.222807e+02, 2.156807e+02, 1.064071e+03, 1.117305e+03, 6.290851e+02};
        auto d = direct_one_loop(kref, table, lk.front(), lk.back());
        bool agree = true;
        for (size_t i = 0; i < kref.size(); ++i)
            agree &= std::abs(d.loop[i] - fastpt[i]) < 1e-3 * std::abs(fastpt[i]) + 1e-5 * d.linear[i];
        check(agree, "LCDM-like spectrum: direct P22 + P13 agree with FAST-PT to 1e-3 for k = 0.02-1 h/Mpc");
        FastEdS lcdm;
        bool fast_ok = true;
        try {
            auto f = lcdm.evaluate(lk, lp, .02, 1.);
            auto all = direct_one_loop(f.k, table, lk.front(), lk.back());
            for (size_t i = 0; i < f.k.size(); ++i)
                fast_ok &= near(f.p22[i], all.p22[i], 1e-3) && near(f.p13[i], all.p13[i], 1e-4);
        } catch (const std::exception &) {
            fast_ok = false;
        }
        check(fast_ok, "LCDM-like spectrum: tapered fast P22 and P13 agree with direct for every mode in 0.02-1 h/Mpc");
        FastOptions untapered;
        untapered.taper = 0;
        bool rejected = false;
        try {
            lcdm.evaluate(lk, lp, .02, 1., untapered);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        check(rejected, "LCDM-like spectrum without the taper: the ringing P22 is caught by the reference checks");
        // Custom-kernel path with the EdS F3 from the recursion reproduces the analytic EdS P13 kernel
        Kernels eds;
        eds.name = "EdS kernels from the recursion";
        eds.f2 = [](double q, double p, double mu) { return eds_f2(q, p, mu); };
        eds.f3_collapsed = [](double kk, double q, double mu) {
            V3 a{0, 0, kk}, b{q * std::sqrt(1 - mu * mu), 0, q * mu};
            return f3s(a, b, V3{-b[0], -b[1], -b[2]});
        };
        DirectOptions fine;
        fine.order = 32;
        const Vector ks{.05, .2, 1.};
        auto analytic = direct_one_loop(ks, table, lk.front(), lk.back(), fine);
        auto recursion = direct_one_loop(ks, table, lk.front(), lk.back(), fine, eds);
        bool same = true;
        for (size_t i = 0; i < ks.size(); ++i)
            same &= near(recursion.p13[i], analytic.p13[i], 1e-5) && near(recursion.p22[i], analytic.p22[i], 1e-12);
        check(same, "custom kernels: EdS F3 from the recursion reproduces the analytic P13 kernel");
    }
    if (!FML::ThisTask)
        std::cout << "Timing: " << fast.k.size() << " modes; checked fast=" << fast_time << " s; direct=" << direct_time
                  << " s; ratio=" << direct_time / fast_time << "; FFT buffers=" << workspace.fft_buffer_bytes()
                  << " bytes\n"
                  << (failures ? "Perturbation tests FAILED" : "All perturbation tests passed") << '\n';
    return failures ? 1 : 0;
}
