// Small periodic demonstration of joint field/bias/noise/BAO inference, not a calibrated survey analysis.
#include <FML/FieldLevel/Framework.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (FML::NTasks != 1) {
        if (FML::ThisTask == 0)
            std::cerr << "[FieldFramework] Requires a single-rank FML communicator\n";
        return 1;
    }
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    const std::string evolution = argc > 1 ? argv[1] : "2lpt";
    c.evolution = evolution == "linear" ? FF::Evolution::Linear :
                  evolution == "1lpt"   ? FF::Evolution::Zeldovich :
                                          FF::Evolution::LPT2;
    FF::Model model(c, FF::Spectrum::bao(FML::FIELDLEVEL::BAOSpectrum::eisenstein_hu(200)));
    auto workspace = model.workspace();
    auto parameters = model.initial_parameters();
    parameters[0] = 1.02;
    const auto truth = model.white_noise(100);
    auto data = model.mock(truth, parameters, 200, workspace);
    FF::Posterior posterior(model, data);
    auto initial = model.initial_parameters();
    auto estimate = FF::map(posterior, FML::FIELDLEVEL::Field(model.cells(), 0), initial, false, 40, 1e-6);
    FML::MCMC::HMCOptions options;
    options.warmup = 80;
    options.samples = argc > 2 ? std::atoi(argv[2]) : 200;
    options.n_leapfrog = 4;
    options.seed = 300;
    std::ofstream fields;
    const std::string prefix = argc > 3 ? argv[3] : "";
    if (!prefix.empty()) {
        fields.open(prefix + ".white_fields");
        fields << "# draw cell white_noise\n";
        if (!fields)
            throw std::runtime_error("Cannot open field stream");
    }
    auto chain = FF::sample(posterior,
                            estimate.white,
                            initial,
                            options,
                            true,
                            [&](const auto & field, const auto &, size_t draw) {
                                if (fields.is_open() && draw % 20 == 0)
                                    for (size_t q = 0; q < field.size(); ++q)
                                        fields << draw << ' ' << q << ' ' << std::setprecision(17) << field[q] << '\n';
                            });
    std::cout << "Periodic toy joint inference; MAP status " << int(estimate.status) << ", acceptance "
              << chain.acceptance << ", divergences " << chain.divergences << '\n';
    for (size_t j = 0; j < parameters.size(); ++j) {
        double mean = 0;
        for (const auto & row : chain.recorded)
            mean += row[j];
        mean /= chain.recorded.size();
        std::cout << model.parameters()[j].name << " truth " << parameters[j] << " sample mean " << mean << '\n';
    }
    if (!prefix.empty()) {
        std::ofstream output(prefix + ".parameters");
        for (const auto & parameter : model.parameters())
            output << parameter.name << ' ';
        output << '\n';
        for (const auto & row : chain.recorded) {
            for (auto value : row)
                output << std::setprecision(17) << value << ' ';
            output << '\n';
        }
        FML::FIELDLEVEL::Field final(chain.last.begin(), chain.last.begin() + model.cells());
        std::vector<double> p(chain.last.begin() + model.cells(), chain.last.end());
        // Choose a H f L explicitly; 30 is a demonstration velocity conversion, not a cosmology prediction.
        const auto reconstruction = model.reconstruct(final, p, workspace, 30.);
        std::ofstream out(prefix + ".reconstruction");
        out << "# cell delta_initial delta_matter ux_q uy_q uz_q\n";
        for (size_t q = 0; q < model.cells(); ++q)
            out << q << ' ' << reconstruction.initial_density[q] << ' ' << reconstruction.matter_density[q] << ' '
                << reconstruction.velocity[0][q] << ' ' << reconstruction.velocity[1][q] << ' '
                << reconstruction.velocity[2][q] << '\n';
        if (!output || !out || !fields)
            throw std::runtime_error("Framework output failed");
    }
}
