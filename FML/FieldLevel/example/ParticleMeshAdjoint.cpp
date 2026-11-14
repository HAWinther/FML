#include <FML/FieldLevel/ParticleMeshAdjoint.h>
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
        c.evolution = FF::Evolution::Linear;
        FF::Spectrum spectrum;
        spectrum.parameters = {{"log_amplitude", 0., 0., 1.}};
        spectrum.evaluate = [](double k, const auto & p) {
            return FF::Power{1000 * std::exp(p[0] - 100 * k * k), {1.}};
        };
        FF::PMSettings options;
        options.scale_factors = {.1, .2, .3, .4, .5, .6, .7, .8, 1.};
        using Grid = FML::FIELDLEVEL::FRONTEND::Grid;
        FF::ParticleMeshForwardT<Grid> model(c, spectrum, options);
        auto work = model.workspace();
        const auto & result = model.evolve(model.white_noise(91), model.initial_parameters(), work);
        // A smooth density statistic, not a likelihood: its white-field and
        // spectrum sensitivities demonstrate the real-space cotangent convention.
        FF::PMCotangent cotangent;
        cotangent.matter_density.resize(result.matter_density.size());
        double statistic = 0;
        for (size_t q = 0; q < result.matter_density.size(); ++q) {
            auto cell = result.density_layout.global_coordinate(q);
            cotangent.matter_density[q] = std::sin(2 * M_PI * cell[0] / c.density_mesh) /
                                         result.density_layout.global_cells;
            statistic += cotangent.matter_density[q] * result.matter_density[q];
        }
        statistic = model.sum(statistic);
        FF::ParticleMeshAdjointT<Grid> adjoint(model);
        const auto g = adjoint.pullback(cotangent, work, {4});
        double norm = 0; for (double x : g.white) norm += x * x;
        norm = std::sqrt(model.sum(norm));
        if (FML::ThisTask == 0)
            std::cout << std::setprecision(14) << "Density statistic: " << statistic
                      << "\nWhite-gradient norm: " << norm
                      << "\nd statistic / d log_amplitude (global): " << g.spectrum[0]
                      << "\nReplayed steps / retained checkpoint states: " << g.report.replayed_steps
                      << " / " << g.report.peak_checkpoint_states
                      << "\nCheckpoint payload bytes on rank 0: " << g.report.checkpoint_payload_bytes << '\n';
        return 0;
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0) std::cerr << e.what() << '\n';
        return 1;
    }
}
