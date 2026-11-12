#include <FML/PerturbationTheory/CorrelationMultipoles.h>
#include <FML/Global/Global.h>
#include <FML/tests/TestHelpers.h>
#include "CorrelationReference.h"
#include <iostream>
#include <memory>
using namespace FML::PERTURBATION;
using FML::TESTS::check;
using V = std::vector<double>;
using Rows = std::array<V, 3>;
V loggrid(double a, double b, int n) {
    V v(n);
    for (int i = 0; i < n; ++i)
        v[i] = std::exp(std::log(a) + (std::log(b) - std::log(a)) * i / (n - 1));
    return v;
}
double maxerror(const Rows & a, const Rows & b) {
    double error = 0;
    for (int l = 0; l < 3; ++l)
        for (size_t i = 0; i < a[l].size(); ++i)
            error = std::max(error, std::abs(a[l][i] - b[l][i]));
    return error;
}
template <class F>
bool rejects(F f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}
int main() {
    const double pi = double(correlation_reference::pi);
    const V k = loggrid(1e-5, 20, 4096), s = loggrid(.4, 7, 21);
    Rows p;
    for (int il = 0; il < 3; ++il)
        for (double q : k)
            p[il].push_back(std::pow(q, 2 * il) * std::exp(-3 * q * q));
    const auto saved = p;
    CorrelationWorkspace workspace(k, s);
    auto actual = workspace.evaluate(p);
    Rows analytic;
    for (int il = 0; il < 3; ++il)
        for (double r : s) {
            const int ell = 2 * il;
            analytic[il].push_back((il == 1 ? -1 : 1) * std::pow(r, ell) * std::exp(-r * r / 12) /
                                   (std::pow(2, ell + 3) * std::pow(pi, 1.5) * std::pow(3, ell + 1.5)));
        }
    check(maxerror(actual.xi, analytic) < 2e-7,
          "Gaussian analytic transforms: phase +,-,+ and 1/(2 pi^2) normalization");
    auto direct = correlation_reference::direct(k, p, s, .1, 0);
    if (!FML::ThisTask)
        std::cout << "Gaussian interpolation errors: direct-analytic=" << maxerror(direct, analytic)
                  << " FFT-direct=" << maxerror(actual.xi, direct) << "\n";
    check(maxerror(direct, analytic) < 5e-8 && maxerror(actual.xi, direct) < 2e-7,
          "independent finite-band Simpson integration agrees with analytic and FFT transforms");
    CorrelationOptions opt;
    opt.damping_radius = 2;
    CorrelationWorkspace damped(k, s, opt);
    auto smoothed = damped.evaluate(p);
    for (int il = 0; il < 3; ++il)
        for (size_t i = 0; i < s.size(); ++i) {
            const int ell = 2 * il;
            const double r = s[i];
            analytic[il][i] = (il == 1 ? -1 : 1) * std::pow(r, ell) * std::exp(-r * r / 28) /
                              (std::pow(2, ell + 3) * std::pow(pi, 1.5) * std::pow(7, ell + 1.5));
        }
    check(maxerror(smoothed.xi, analytic) < 2e-7, "explicit Gaussian regulator has the analytic length-scale response");
    // A constant monopole is a contact term only in the infinite, unfiltered transform.
    // With explicit Gaussian smoothing it has this finite-width analytic profile.
    Rows contact;
    for (auto & v : contact)
        v.assign(k.size(), 0);
    contact[0].assign(k.size(), 3);
    auto noise = damped.evaluate(contact);
    Rows noise_expected;
    for (auto & v : noise_expected)
        v.assign(s.size(), 0);
    for (size_t i = 0; i < s.size(); ++i)
        noise_expected[0][i] = 3 * std::exp(-s[i] * s[i] / 16) / (64 * std::pow(pi, 1.5));
    check(maxerror(noise.xi, noise_expected) < 2e-7,
          "constant stochastic power is retained with its declared filtered profile");
    Rows scaled = p;
    for (auto & v : scaled)
        for (auto & x : v)
            x *= -2.5;
    auto again = workspace.evaluate(scaled);
    for (auto & v : actual.xi)
        for (auto & x : v)
            x *= -2.5;
    check(maxerror(again.xi, actual.xi) < 1e-14 && workspace.evaluations() == 2,
          "signed spectra and amplitude linearity with repeated fixed FFT setup");
    Rows zero;
    for (auto & v : zero)
        v.assign(k.size(), 0);
    const auto empty = workspace.evaluate(zero);
    check(maxerror(empty.xi, zero) == 0, "zero power transforms to zero on requested common separations");
    V kn = loggrid(.001, 3, 601);
    for (size_t i = 1; i + 1 < kn.size(); ++i)
        kn[i] *= std::exp(.001 * std::sin(double(i)));
    V sn{2, 7, 20, 40, 80};
    Rows signed_p;
    for (int il = 0; il < 3; ++il)
        for (double q : kn)
            signed_p[il].push_back((il + 1) * std::cos(5 * q + il) * std::exp(-q * q));
    opt.damping_radius = .7;
    opt.taper_fraction = .15;
    opt.fft_samples = 8192;
    CorrelationWorkspace nonuniform(kn, sn, opt);
    auto signed_xi = nonuniform.evaluate(signed_p);
    auto reference = correlation_reference::direct(kn, signed_p, sn, .15, .7);
    auto refined = correlation_reference::direct(kn, signed_p, sn, .15, .7, .05);
    check(maxerror(reference, refined) < 1e-9 && maxerror(signed_xi.xi, refined) < 2e-7,
          "nonuniform signed input and tapered finite support agree with refined oscillatory quadrature");
    opt.fft_samples = 512;
    CorrelationWorkspace coarse(kn, sn, opt);
    opt.fft_samples = 4096;
    CorrelationWorkspace fine(kn, sn, opt);
    auto xc = coarse.evaluate(signed_p), xf = fine.evaluate(signed_p);
    check(maxerror(xf.xi, refined) < maxerror(xc.xi, refined) && maxerror(xf.xi, refined) < 5e-7,
          "internal FFT sampling refinement reduces finite-band transform error");
    // Length units: k -> k/2, P -> 8P, s -> 2s leaves dimensionless xi invariant.
    V ku = k, su = s;
    Rows pu = p;
    for (auto & x : ku)
        x /= 2;
    for (auto & x : su)
        x *= 2;
    for (auto & v : pu)
        for (auto & x : v)
            x *= 8;
    CorrelationWorkspace units(ku, su);
    auto xu = units.evaluate(pu);
    auto original = workspace.evaluate(p);
    check(maxerror(xu.xi, original.xi) < 1e-13 && xu.s == su,
          "physical length conversion preserves dimensionless correlation multipoles");
    auto moved = std::move(units);
    auto xm = moved.evaluate(pu);
    check(maxerror(xm.xi, xu.xi) < 1e-15 && p == saved, "workspace move preserves transforms without modifying inputs");
    check(rejects([&] {
              Rows none;
              units.evaluate(none);
          }),
          "moved-from workspace is rejected safely");
    bool owner_only = true;
    if (!FML::ThisTask) {
        CorrelationWorkspace local(k, s);
        owner_only = maxerror(local.evaluate(p).xi, original.xi) < 1e-14;
    }
    check(owner_only, "local transform may run on just one MPI rank");
    check(rejects([&] {
              V bad = k;
              bad[3] = bad[2];
              CorrelationWorkspace w(bad, s);
          }),
          "duplicate input k rejected");
    check(rejects([&] { CorrelationWorkspace w(k, {0, 1}); }), "zero separation rejected");
    check(rejects([&] { CorrelationWorkspace w(k, {2, 1}); }), "unordered separations rejected");
    check(rejects([&] { CorrelationWorkspace w(k, {1e-20}); }), "separation extrapolation rejected");
    check(rejects([&] {
              auto o = opt;
              o.taper_fraction = .6;
              CorrelationWorkspace w(k, s, o);
          }),
          "overlapping taper widths rejected");
    check(rejects([&] {
              auto o = opt;
              o.fft_samples = 127;
              CorrelationWorkspace w(k, s, o);
          }),
          "odd FFT width rejected");
    check(rejects([&] {
              auto o = opt;
              o.damping_radius = NAN;
              CorrelationWorkspace w(k, s, o);
          }),
          "nonfinite damping rejected");
    check(rejects([&] {
              Rows bad = p;
              bad[1].pop_back();
              workspace.evaluate(bad);
          }),
          "mismatched multipole lengths rejected");
    check(rejects([&] {
              Rows bad = p;
              bad[2][2] = NAN;
              workspace.evaluate(bad);
          }),
          "nonfinite power rejected");
#ifdef USE_OMP
    std::vector<std::unique_ptr<CorrelationWorkspace>> workers;
    for (int i = 0; i < 2; ++i)
        workers.push_back(std::make_unique<CorrelationWorkspace>(k, s));
    int bad = 0;
#pragma omp parallel for reduction(+ : bad)
    for (int i = 0; i < 2; ++i)
        if (maxerror(workers[i]->evaluate(p).xi, original.xi) > 1e-14)
            ++bad;
    check(bad == 0, "separately prepared correlation workspaces execute concurrently");
#endif
    return FML::TESTS::summary("correlation multipoles");
}
