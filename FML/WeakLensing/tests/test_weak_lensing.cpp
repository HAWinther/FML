#include <FML/WeakLensing/WeakLensing.h>
#include "Reference.h"
#include <FML/Global/Global.h>
#ifdef USE_FFTW
#include <FML/WeakLensing/FlatSky.h>
#endif
#include <iostream>
#include <string>
using namespace FML::WEAKLENSING;
static int failures = 0, checks = 0;
static void check(bool ok, const std::string & label) {
    ++checks;
    failures += !ok;
    if (!ok || FML::ThisTask == 0)
        std::cout << (ok ? "PASS " : "FAIL ") << label << "\n";
}
static void close(double x, double expected, double tolerance, const std::string & label) {
    check(std::isfinite(x) && std::abs(x - expected) <= tolerance * std::max(1., std::abs(expected)), label);
}
template <class F>
static void rejects(F && f, const std::string & label) {
    bool rejected = false;
    try {
        f();
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, label);
}
static void scalar_tests() {
    FlatGeometry toy{2. / 3, 1., [](double) { return 0.; }};
    const auto source = Sources::plane(2.3);
    close(source.efficiency(0), 1., 1e-15, "source normalization");
    close(source.efficiency(1), 1 - 1 / 2.3, 1e-15, "single-source distance ratio");
    close(source.efficiency(3), 0., 0, "foreground sources do not lens");
    const Sources mixture({4., 2., 4.}, {1., 2., 1.});
    close(mixture.efficiency(1), .625, 1e-15, "weighted duplicate source planes");
    close(mixture.input_normalization(), 4., 0, "original normalization retained");
    std::vector<double> chi, pdf;
    for (int i = 0; i <= 1000; ++i) {
        chi.push_back(1. + 2. * i / 1000);
        pdf.push_back(1.);
    }
    const auto broad = Sources::from_distance_density(chi, pdf);
    const double lens = 1.5;
    close(broad.efficiency(lens),
          (3 - lens - lens * std::log(3 / lens)) / 2,
          2e-7,
          "uniform source distribution analytic integral");
    const auto nz = Sources::from_redshift_density({1., 2., 3.}, {1., 2., 1.}, [](double z) { return 10 * z; });
    const auto nchi = Sources::from_distance_density({10., 20., 30.}, {.1, .2, .1});
    close(nz.efficiency(15), nchi.efficiency(15), 1e-15, "n(z) quadrature measure and distance Jacobian");
    close(nz.input_normalization(), 3., 1e-15, "trapezoid distribution normalization");
    for (int n : {2, 3, 8, 16, 32, 64})
        close(Quadrature(n).integrate([](double) { return 1.; }, 0., 1.),
              1.,
              2e-12,
              "Gauss rule constant order " + std::to_string(n));
    close(Quadrature(4).integrate([](double x) { return std::pow(x, 7); }, 0., 1.),
          .125,
          2e-14,
          "Gauss polynomial independent oracle");
    BornPlan plan({0., 1., 3.}, {source, Sources::plane(4.1)}, toy);
    close(plan.weight(0, 0) + plan.weight(0, 1),
          2.3 * 2.3 / 6,
          2e-14,
          "source inside finite shell: exact integrated weight");
    close(plan.weight(1, 0) + plan.weight(1, 1), 4.5 - 9 / 4.1, 2e-14, "partial lens interval normalization");
    const std::size_t pixels = 3 + FML::ThisTask;
    BornAccumulator accumulator(plan, pixels);
    rejects([&] { accumulator.maps(); }, "incomplete projection rejected");
    std::vector<double> values(pixels);
    for (std::size_t i = 0; i < pixels; ++i)
        values[i] = double(i + FML::ThisTask) - 1;
    rejects([&] { accumulator.add_shell(1, values.data(), values.size()); }, "out-of-order shell rejected");
    auto bad = values;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { accumulator.add_shell(0, bad.data(), bad.size()); }, "NaN shell rejected before mutation");
    rejects([&] { accumulator.add_shell(0, values.data(), values.size() - 1); }, "pixel mismatch rejected");
    accumulator.add_shell(0, values.data(), values.size());
    rejects([&] { accumulator.add_shell(0, values.data(), values.size()); }, "duplicate shell rejected");
    accumulator.add_shell(1, values.data(), values.size());
    for (std::size_t i = 0; i < pixels; ++i)
        close(accumulator.maps()[0][i], values[i] * 2.3 * 2.3 / 6, 3e-14, "local Born pixels and atomic failed update");
    BornAccumulator empty(plan, 0);
    empty.add_shell(0, static_cast<double *>(nullptr), 0);
    empty.add_shell(1, static_cast<double *>(nullptr), 0);
    check(empty.complete() && empty.maps()[0].empty(), "empty local pixel ownership");
    FlatGeometry zero{0., 1., [](double) { return 0.; }};
    close(zero.kernel(source, 1), 0., 0, "zero matter gives zero lensing");
    FlatGeometry evolving{2. / 3, 1., [](double x) { return x; }};
    close(evolving.kernel(source, 1), 2 * (1 - 1 / 2.3), 1e-15, "scale factor in kernel");

    const auto w4 = convergence_kernel(Sources::plane(4.), toy), w2 = convergence_kernel(Sources::plane(2.), toy);
    const PowerSpectrum power{[](double, double) { return 2.; }, 0., 1e8, 0., 0.};
    const std::vector<double> multipoles = {10., 30., 100.};
    const auto auto_cl = limber_spectrum(multipoles, w4, w4, power, toy, {0., 2., 4.});
    for (double cl : auto_cl)
        close(cl, 8. / 3, 2e-13, "constant P source-plane analytic Ckk");
    const auto cross = limber_spectrum(multipoles, w2, w4, power, toy, {0., 2., 4.});
    const auto reverse = limber_spectrum(multipoles, w4, w2, power, toy, {0., 2., 4.});
    for (std::size_t i = 0; i < cross.size(); ++i) {
        close(cross[i], 5. / 3, 2e-13, "analytic tomographic cross spectrum");
        close(cross[i], reverse[i], 1e-15, "tomographic symmetry");
    }
    const RadialSelection foreground({1., 3.}, {1., 1.});
    close(foreground(2), .5, 1e-15, "continuous foreground selection");
    close(foreground(.5), 0., 0, "selection outside support");
    const auto gk = limber_spectrum({30.}, foreground, w4, power, toy, {0., 1., 3., 4.}, 32);
    close(gk[0], std::log(3.) - .5, 2e-13, "analytic galaxy-convergence spectrum");
    const PowerSpectrum slope{[](double k, double) { return k; }, 0., 1e8, 0., 0.};
    const auto moving =
        limber_spectrum({9.5, 19.5}, [](double x) { return x; }, [](double x) { return x; }, slope, toy, {1., 2.}, 16);
    close(moving[0], 10 * std::log(2.), 2e-13, "extended Limber ell plus half and k units");
    close(moving[1] / moving[0], 2., 2e-13, "ell dependence of provider");
    rejects([&] { limber_spectrum({100.}, w4, w4, PowerSpectrum{power.value, 0., 1., 0., 0.}, toy, {1., 2.}); },
            "power k coverage enforced");
    rejects([&] { limber_spectrum({10.}, w4, w4, power, evolving, {1., 2.}); }, "power redshift coverage enforced");
    const auto zero_cl = limber_spectrum(
        {100.},
        [](double) { return 0.; },
        w4,
        PowerSpectrum{[](double, double) -> double { throw std::logic_error("must not query"); }, 0., 1., 0., 0.},
        toy,
        {0., 4.});
    close(zero_cl[0], 0., 0, "zero kernel does not query power");

    const double amplitude = std::sqrt(2 * std::acos(-1.)) * std::erf(3 / std::sqrt(2.));
    const Function xi = [](double r) { return std::exp(-r * r / 2); };
    for (double r : {0., .5, 2.5})
        close(projected_galaxy_matter(r, 3., xi, 8),
              amplitude * std::exp(-r * r / 2),
              2e-13,
              "finite-Pi Gaussian projected correlation");
    const Function wp = [&](double r) { return amplitude * std::exp(-r * r / 2); };
    close(excess_surface_density(1., 2., wp, 8),
          2 * amplitude * (2 * (-std::expm1(-.5)) - std::exp(-.5)),
          3e-13,
          "Gaussian excess surface density");
    close(excess_surface_density(2., 1e25, [](double) { return 7.; }), 0., 0., "constant sheet cancels in DeltaSigma");
    close(inverse_sigma_critical_comoving(4., .5, Sources::plane(10.), 2.),
          7.2,
          1e-15,
          "comoving critical-density redshift factor");
    close(inverse_sigma_critical_comoving(4., .5, Sources::plane(3.), 2.),
          0.,
          0,
          "foreground source has zero shear efficiency");
    close(physical_surface_density(2., .5), 4.5, 1e-15, "physical surface density conversion");
    close(2 * inverse_sigma_critical_comoving(4., .5, Sources::plane(10.), 2.),
          physical_surface_density(2., .5) * 2. * 4. / 1.5 * .6,
          1e-15,
          "physical/comoving shear invariance");
    const auto radial_error = [&](int count) {
        std::vector<double> edges;
        for (int i = 0; i <= count; ++i)
            edges.push_back(2.3 * i / count);
        BornAccumulator a(BornPlan(edges, {source}, toy), 1);
        for (int i = 0; i < count; ++i) {
            const double value = std::pow((edges[i] + edges[i + 1]) / 2, 2);
            a.add_shell(i, &value, 1);
        }
        return std::abs(a.maps()[0][0] - std::pow(2.3, 4) / 20);
    };
    check(radial_error(16) < radial_error(4) / 10, "radial shell representation converges on refinement");
    const double dh = 2997.92458;
    const FlatGeometry eds{1., dh, [=](double x) { return 1 / std::pow(1 - x / (2 * dh), 2) - 1; }};
    const auto a = convergence_kernel(Sources::plane(1200), eds), b = convergence_kernel(Sources::plane(2200), eds);
    const RadialSelection triangle({200., 800., 1000.}, {0., 1., 0.});
    const auto mm = [](double k, double z) {
        return 2000 * std::pow(k / .1, .8) * std::exp(-std::pow(k / .8, 2)) / (1 + z) / (1 + z);
    };
    const PowerSpectrum matter{mm, 0., 1e9, 0., 4.};
    const PowerSpectrum gm{[&](double k, double z) { return (1.4 + .2 * z + .1 * k) * mm(k, z); }, 0., 1e9, 0., 4.};
    std::vector<double> edges;
    for (int i = 0; i <= 220; ++i)
        edges.push_back(10. * i);
    for (const auto & row : lensing_reference) {
        const auto ca = limber_spectrum({row[0]}, a, a, matter, eds, edges, 16);
        const auto cab = limber_spectrum({row[0]}, a, b, matter, eds, edges, 16);
        const auto cg = limber_spectrum({row[0]}, triangle, b, gm, eds, edges, 16);
        check(std::abs(ca[0] / row[1] - 1) < 2e-8, "independent SciPy Ckk physical-unit reference");
        check(std::abs(cab[0] / row[2] - 1) < 2e-8, "independent SciPy tomographic cross reference");
        check(std::abs(cg[0] / row[3] - 1) < 2e-8, "independent SciPy scale-dependent Pgm reference");
    }
    rejects([] { Sources({0.}, {1.}); }, "observer source rejected");
    rejects([] { Sources({2.}, {-1.}); }, "negative source mass rejected");
    rejects([] { Sources({2.}, {0.}); }, "empty source probability rejected");
    rejects([] { Sources::from_distance_density({1., 1.}, {1., 1.}); }, "nonmonotone distribution rejected");
    rejects([&] { BornPlan({0., 2., 1.}, {source}, toy); }, "nonmonotone shells rejected");
    rejects([] { Quadrature(1); }, "invalid quadrature rejected");
    rejects([&] { excess_surface_density(0., 1., wp); }, "zero surface-density radius rejected");
}
#ifdef USE_FFTW
static void shear_tests() {
    const int n = FML::NTasks == 3 ? 15 : 16;
    const double tx = .2, ty = .35, pi = std::acos(-1.);
    const double tolerance = sizeof(FML::GRID::FloatType) == 4 ? 2e-5 : 3e-12;
    for (const auto mode : std::vector<std::array<int, 2>>{{2, 0}, {0, 2}, {1, 2}, {-1, 2}}) {
        AngularGrid kappa(n);
        kappa.fill_real_grid(
            [&](std::array<double, 2> & x) { return 2. + .3 * std::cos(2 * pi * (mode[0] * x[0] + mode[1] * x[1])); });
        const auto shear = shear_from_convergence(kappa, tx, ty);
        const double lx = mode[0] / tx, ly = mode[1] / ty, den = lx * lx + ly * ly;
        double error = 0, preserved = 0;
        for (int ix = 0; ix < kappa.get_local_nx(); ++ix)
            for (int iy = 0; iy < n; ++iy) {
                const auto x = kappa.get_real_position({ix, iy});
                const double signal = .3 * std::cos(2 * pi * (mode[0] * x[0] + mode[1] * x[1]));
                error =
                    std::max(error,
                             std::abs(double(shear.gamma1.get_real({ix, iy})) - signal * (lx * lx - ly * ly) / den));
                error = std::max(error, std::abs(double(shear.gamma2.get_real({ix, iy})) - signal * 2 * lx * ly / den));
                preserved = std::max(preserved, std::abs(double(kappa.get_real({ix, iy})) - (2 + signal)));
            }
        FML::MaxOverTasks(&error);
        FML::MaxOverTasks(&preserved);
        close(error, 0., tolerance, "Fourier mode shear incl. sign/aspect/DC");
        close(preserved, 0., tolerance, "input convergence preserved");
        auto one = shear.gamma1, two = shear.gamma2;
        one.fftw_r2c();
        two.fftw_r2c();
        double bmode = 0;
        for (auto index : one.get_fourier_range()) {
            const auto m = one.get_fourier_integer_wavevector_from_index(index);
            const double x = m[0] / tx, y = m[1] / ty, norm = x * x + y * y;
            if (norm == 0 || (n % 2 == 0 && (std::abs(m[0]) == n / 2 || m[1] == n / 2)))
                continue;
            const std::complex<double> b =
                -(2 * x * y / norm) * std::complex<double>(one.get_fourier_from_index(index)) +
                ((x * x - y * y) / norm) * std::complex<double>(two.get_fourier_from_index(index));
            bmode = std::max(bmode, std::abs(b));
        }
        FML::MaxOverTasks(&bmode);
        close(bmode, 0., tolerance, "Born shear has no B mode");
    }
    AngularGrid dc(n);
    dc.fill_real_grid(2.);
    const auto zero = shear_from_convergence(dc, tx, ty);
    double maxzero = 0;
    for (auto i : zero.gamma1.get_real_range())
        maxzero = std::max(maxzero,
                           std::max(std::abs(double(zero.gamma1.get_real_from_index(i))),
                                    std::abs(double(zero.gamma2.get_real_from_index(i)))));
    FML::MaxOverTasks(&maxzero);
    close(maxzero, 0., tolerance, "mass-sheet monopole removed");
    if (n % 2 == 0) {
        AngularGrid nyquist(n);
        nyquist.fill_real_grid([&](std::array<double, 2> & x) {
            return std::cos(2 * pi * (n / 2 * x[0] + x[1])) + std::cos(2 * pi * (x[0] + n / 2 * x[1]));
        });
        const auto filtered = shear_from_convergence(nyquist, tx, ty);
        double error = 0;
        for (auto i : filtered.gamma1.get_real_range())
            error = std::max(error,
                             std::max(std::abs(double(filtered.gamma1.get_real_from_index(i))),
                                      std::abs(double(filtered.gamma2.get_real_from_index(i)))));
        FML::MaxOverTasks(&error);
        close(error, 0., tolerance, "Nyquist lines explicitly removed");
    }
    rejects([&] { shear_from_convergence(dc, -1., ty); }, "invalid angular widths rejected collectively");
    if (FML::NTasks > 1)
        rejects([&] { shear_from_convergence(dc, FML::ThisTask == 0 ? tx : tx * 2, ty); },
                "inconsistent rank geometry rejected collectively");
    if (FML::ThisTask == 0)
        dc.set_real({0, 0}, std::numeric_limits<FML::GRID::FloatType>::quiet_NaN());
    rejects([&] { shear_from_convergence(dc, tx, ty); }, "nonfinite rank pixels rejected collectively");
}
#endif
int main() {
    scalar_tests();
#ifdef USE_FFTW
    shear_tests();
#endif
    FML::MaxOverTasks(&failures);
    if (FML::ThisTask == 0)
        std::cout << "WeakLensing: " << checks << " checks on root, " << failures << " failures\n";
    return failures ? 1 : 0;
}
