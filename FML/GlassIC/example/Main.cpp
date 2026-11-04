#include <FML/GlassIC/GlassIC.h>
#include <FML/GlassIC/LoadIO.h>
#include <iostream>
using namespace FML::GLASSIC;
// Analytic example spectrum only, not a calibrated cosmology or transfer solution.
struct ToyProvider : FML::INITIALCONDITIONS::TransferProvider {
    double h() const override { return .7; }
    FML::INITIALCONDITIONS::TransferRange range() const override { return {1e-5, 100, .01, 1}; }
    FML::INITIALCONDITIONS::Gauge density_gauge() const override {
        return FML::INITIALCONDITIONS::Gauge::SynchronousCDM;
    }
    bool has_density(FML::INITIALCONDITIONS::Species s) const override {
        return s == FML::INITIALCONDITIONS::Species::CDMBaryon;
    }
    double density_transfer(FML::INITIALCONDITIONS::Species, double k, double a) const override {
        return a * std::exp(-k * k);
    }
    double primordial_power(double) const override { return 100; }
};
int main(int argc, char ** argv) {
    auto load = poisson_load(4096, 12345, 100.);
    Options o;
    o.mesh = 48;
    o.max_steps = 1000;
    const auto result = relax(load, o);
    if (FML::ThisTask == 0)
        std::cout << "Glass status (0 converged, 1 max steps, 2 nonfinite): " << int(result.status) << " steps "
                  << result.steps << " force RMS " << result.residuals.force_rms << " velocity RMS "
                  << result.residuals.velocity_rms << '\n';
    // An exhausted run is inspectable; require convergence yourself before production ICs.
    if (result.status == Status::NonFinite)
        return 1;
    if (argc > 1)
        save_load(load, argv[1]); // Optional NEW prefix: persist unperturbed glass before applying ICs.
    Grid delta(o.mesh, 1, 1);
    ToyProvider provider;
    density_from_provider(delta, provider, FML::INITIALCONDITIONS::Species::CDMBaryon, .02, load.boxsize, 54321, true);
    // EdS f=1; a H L converts box displacement to peculiar velocity in km/s for H in km/s/(Mpc/h).
    const double a = .02, H = 100 * std::pow(a, -1.5), velocity_factor = a * H * load.boxsize;
    apply_3lpt(load, delta, velocity_factor, FML::COSMOLOGY::LPT::Dealiasing::Padding);
    if (FML::ThisTask == 0)
        std::cout << "Applied shared-provider 3LPT at glass q; IDs and q retained, velocity in km/s.\n";
}
