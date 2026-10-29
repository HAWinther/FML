// Toy flat geometry and Gaussian spectra/profiles: a runnable API demonstration,
// not a calibrated cosmological prediction. FML initializes itself automatically.
#include <FML/WeakLensing/WeakLensing.h>
#include <FML/Global/Global.h>
#ifdef USE_FFTW
#include <FML/WeakLensing/FlatSky.h>
#endif
#include <iostream>
#include <iomanip>
using namespace FML::WEAKLENSING;
int main() {
    const double dh = 2997.92458;
    const FlatGeometry geometry{.3, dh, [=](double chi) { return chi / dh; }};
    const std::vector<Sources> bins = {
        Sources::plane(1600.),
        Sources::from_redshift_density({.4, .6, .8, 1.}, {0., 1., 1., 0.}, [=](double z) { return dh * z; })};
    std::vector<double> edges;
    for (int j = 0; j <= 64; ++j)
        edges.push_back(3000. * j / 64);
    for (const auto & bin : bins)
        for (double chi : bin.distances())
            edges.push_back(chi);
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    BornPlan plan(edges, bins, geometry);
#ifdef USE_FFTW
    const int n = FML::NTasks == 3 ? 15 : 16;
    AngularGrid kappa(n);
    const auto local_pixels = std::size_t(kappa.get_local_nx()) * n;
#else
    const std::size_t local_pixels = 32;
#endif
    BornAccumulator projected(plan, local_pixels);
    std::vector<double> shell(local_pixels);
    for (std::size_t j = 0; j < plan.shells(); ++j) {
        const double growth = 1 / (1 + geometry.redshift((edges[j] + edges[j + 1]) / 2));
        for (std::size_t i = 0; i < local_pixels; ++i) {
#ifdef USE_FFTW
            const auto x = kappa.get_real_position({int(i / n), int(i % n)});
            const double phase = 2 * std::acos(-1.) * (x[0] + 2 * x[1]);
#else
            const double phase = 2 * std::acos(-1.) * (i + local_pixels * FML::ThisTask) / (local_pixels * FML::NTasks);
#endif
            shell[i] = .1 * growth * std::cos(phase);
        }
        projected.add_shell(j, shell.data(), shell.size());
    }
#ifdef USE_FFTW
    for (std::size_t i = 0; i < local_pixels; ++i)
        kappa.set_real({int(i / n), int(i % n)}, FML::GRID::FloatType(projected.maps()[0][i]));
    const auto shear = shear_from_convergence(kappa, .1, .15);
    double peak = 0;
    for (auto i : shear.gamma1.get_real_range())
        peak = std::max(peak, std::abs(double(shear.gamma1.get_real_from_index(i))));
    FML::MaxOverTasks(&peak);
    if (FML::ThisTask == 0)
        std::cout << "# bin 0 peak |gamma1| " << peak << "\n";
#endif
    const PowerSpectrum pmm{[](double k, double z) { return 1000 * std::exp(-k * k) / (1 + z) / (1 + z); },
                            0.,
                            1e8,
                            0.,
                            2.};
    const auto w0 = convergence_kernel(bins[0], geometry), w1 = convergence_kernel(bins[1], geometry);
    const RadialSelection foreground({300., 700., 1000.}, {0., 1., 0.});
    auto panels = edges;
    for (double chi : foreground.distances())
        panels.push_back(chi);
    std::sort(panels.begin(), panels.end());
    panels.erase(std::unique(panels.begin(), panels.end()), panels.end());
    const std::vector<double> ell = {30., 100., 300.};
    const auto c00 = limber_spectrum(ell, w0, w0, pmm, geometry, panels);
    const auto c01 = limber_spectrum(ell, w0, w1, pmm, geometry, panels);
    // Toy unbiased foreground: Pgm=Pmm; an actual galaxy model supplies Pgm.
    const auto cg0 = limber_spectrum(ell, foreground, w0, pmm, geometry, panels);
    if (FML::ThisTask == 0) {
        std::cout << std::setprecision(10) << "# Toy model: ell Ckk_00 Ckk_01 Cgk_0\n";
        for (std::size_t i = 0; i < ell.size(); ++i)
            std::cout << ell[i] << " " << c00[i] << " " << c01[i] << " " << cg0[i] << "\n";
        const Function xi = [](double r) { return std::exp(-r * r / 2); };
        const Function wp = [&](double r) { return projected_galaxy_matter(r, 6., xi, 8); };
        std::cout << "# R [Mpc/h], wgm [Mpc/h], DeltaSigma_com [(Msun/h)/(Mpc/h)^2]\n";
        for (double r : {.2, .5, 1., 2.})
            std::cout << r << " " << wp(r) << " " << excess_surface_density(r, 8.3e10, wp, 8) << "\n";
    }
}
