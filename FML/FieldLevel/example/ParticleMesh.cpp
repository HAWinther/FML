#include <FML/FieldLevel/ParticleMesh.h>
#include <FML/FieldLevel/Frontend.h>
#include <iomanip>
#include <iostream>
int main() {
    namespace FF = FML::FIELDLEVEL::FRAMEWORK;
    try {
        FML::FIELDLEVEL::FRONTEND::initialize(8, 16);
        FF::Configuration c;
        c.mesh = 8; c.density_mesh = 16; c.boxsize = 200;
        c.initial_cutoff = c.bias_cutoff = c.likelihood_cutoff = .08;
        c.evolution = FF::Evolution::Linear; // Initial-field generator, not the PM evolution.
        FF::Spectrum spectrum;
        spectrum.evaluate = [](double k, const auto &) { return FF::Power{1000 * std::exp(-100 * k * k), {}}; };
        FF::PMSettings options;
        options.background.omega_m = .3;
        options.scale_factors.clear();
        for (int j = 0; j <= 32; ++j) options.scale_factors.push_back(.05 * std::pow(20., j / 32.));
        FF::ParticleMeshForwardT<FML::FIELDLEVEL::FRONTEND::Grid> model(c, spectrum, options);
        auto workspace = model.workspace();
        const auto & result = model.evolve(model.white_noise(91), model.initial_parameters(), workspace);
        if (FML::ThisTask == 0) {
            std::cout << std::setprecision(12) << "a mass_error density_rms mean_px mean_py mean_pz\n";
            for (const auto & d : result.diagnostics)
                std::cout << d.a << ' ' << d.mass_error << ' ' << d.density_rms << ' '
                          << d.mean_momentum[0] << ' ' << d.mean_momentum[1] << ' ' << d.mean_momentum[2] << '\n';
        }
        return 0;
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0) std::cerr << e.what() << '\n';
        return 1;
    }
}
