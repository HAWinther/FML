#include <FML/PerturbationTheory/CorrelationMultipoles.h>
#include <FML/PerturbationTheory/EPTRedshiftSpace.h>
#include <FML/PerturbationTheory/EFTofLSS.h>
#include <FML/PerturbationTheory/TNS.h>
#include <FML/Global/Global.h>
#include <FML/tests/TestHelpers.h>
#include "CorrelationReference.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
using namespace FML::PERTURBATION;
using FML::TESTS::check;
using V = std::vector<double>;
using Rows = std::array<V, 3>;
namespace {
    V k, p;
    V s{25, 50, 80, 110, 140};
    CorrelationOptions options;
    double error(const Rows & a, const Rows & b) {
        double diff = 0, scale = 0;
        for (int l = 0; l < 3; ++l)
            for (size_t j = 0; j < s.size(); ++j) {
                diff = std::max(diff, std::abs(a[l][j] - b[l][j]));
                scale = std::max(scale, std::abs(b[l][j]));
            }
        return diff / std::max(scale, 1e-20);
    }
    void compare(const std::string & label, const V & kv, const Rows & power, const CorrelationMultipoles & fast) {
        const auto ref = correlation_reference::direct(kv, power, s, options.taper_fraction, options.damping_radius);
        const auto refined =
            correlation_reference::direct(kv, power, s, options.taper_fraction, options.damping_radius, .05);
        const double delta = error(fast.xi, refined);
        if (!FML::ThisTask)
            std::cout << label << " correlation max error / largest |xi| = " << delta << '\n';
        check(error(ref, refined) < 1e-6 && delta < 5e-5,
              label + " multipoles to xi agree with refined independent finite-band quadrature");
    }
} // namespace
int main(int argc, char ** argv) {
    std::ifstream input("reference/lptrsd_linear_pk.txt");
    if (!input)
        throw std::runtime_error("Run from PerturbationTheory/tests: missing linear reference");
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        double x, y;
        std::istringstream row(line);
        row >> x >> y;
        k.push_back(x);
        p.push_back(y);
    }
    V knw, pnw;
    std::ifstream nw("reference/rept_nowiggle_pk.txt");
    if (!nw)
        throw std::runtime_error("Missing no-wiggle reference");
    while (std::getline(nw, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        double x, y;
        std::istringstream row(line);
        row >> x >> y;
        knw.push_back(x);
        pnw.push_back(y);
    }
    if (knw != k)
        throw std::runtime_error("No-wiggle input grid differs");
    V kv(96);
    for (size_t i = 0; i < kv.size(); ++i)
        kv[i] = std::exp(std::log(.003) + std::log(.6 / .003) * i / (kv.size() - 1));
    options.fft_samples = 8192;
    options.damping_radius = 4;
    options.taper_fraction = .1;
    CorrelationWorkspace workspace(kv, s, options);
    LPTRedshiftSpace::Options lopt;
    lopt.N = 512;
    lopt.jn = 6;
    lopt.nmax = 6;
    LPTRedshiftSpace lpt(k, p, lopt);
    LPTRedshiftSpace::Bias lb;
    lb.b1 = .8;
    lb.b2 = .2;
    lb.bs = -.1;
    lb.alpha0 = 2;
    auto lp = lpt.multipoles(.75, kv, lb, 1.02, .98);
    compare("LPT (with AP)", lp.k, {lp.p0, lp.p2, lp.p4}, workspace.evaluate_spectrum(lp));
    auto bad = lp;
    bad.k[1] *= 1.0001;
    bool rejected = false;
    try {
        workspace.evaluate_spectrum(bad);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "model adapter rejects a changed k grid");
    EPTRedshiftSpace::Options eopt;
    eopt.kmin = kv.front();
    eopt.kmax = kv.back();
    eopt.nk = int(kv.size());
    // Explicit no-wiggle input: this test is of the prediction-to-correlation interface, not the split algorithm.
    EPTRedshiftSpace ept(k, p, pnw, eopt);
    EPTRedshiftSpace::Bias eb = lb;
    eb.b1 = 1.8;
    auto ep = ept.multipoles(.75, eb, 1.02, .98);
    CorrelationWorkspace eworkspace(ep.k, s, options);
    compare("IR EPT (with AP)", ep.k, {ep.p0, ep.p2, ep.p4}, eworkspace.evaluate_spectrum(ep));
    TNS tns(k, p, kv);
    TNS::Parameters tp;
    tp.f = .75;
    tp.b1 = 1.8;
    tp.b2 = .2;
    tp.sigma_v = 3;
    auto tpower = tns.multipoles(tp);
    compare("TNS", kv, tpower, workspace.evaluate(tpower));
    EFTofLSS::Options eftopt;
    eftopt.nowiggle = EFTofLSS::Options::NoWiggleMethod::Given;
    eftopt.Pnw = pnw;
    eftopt.r_bao = 110;
    EFTofLSS eft(k, p, kv, .75, eftopt);
    EFTofLSS::Bias fb;
    fb.b1 = 1.8;
    fb.b2 = .2;
    fb.c0 = 2;
    const auto fp = eft.multipoles(fb);
    compare("EFTofLSS", kv, fp, workspace.evaluate(fp));
    check(workspace.evaluations() == 3 && eworkspace.evaluations() == 1,
          "shared workspace handles multiple prediction models without rebuilding FFT setup");
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        auto start = std::chrono::steady_clock::now();
        double checksum = 0;
        for (int i = 0; i < 100; ++i)
            checksum += workspace.evaluate(fp).xi[0][0];
        double cached = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 100;
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) {
            CorrelationWorkspace fresh(kv, s, options);
            checksum += fresh.evaluate(fp).xi[0][0];
        }
        double fresh = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 20;
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i)
            checksum += correlation_reference::direct(kv, fp, s, .1, 4)[0][0];
        double direct = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 10;
        if (!FML::ThisTask)
            std::cout << "Correlation timing seconds/call: reused=" << cached << " fresh=" << fresh
                      << " direct=" << direct << " checksum=" << checksum << '\n';
    }
    return FML::TESTS::summary("correlation model adapters");
}
