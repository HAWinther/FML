#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using Clock = std::chrono::steady_clock;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 2 || FML::NTasks != 1)
            throw std::invalid_argument("Serial metric_probe PREFIX required");
        const std::string prefix = argv[1];
        FF::EulerianLPTConfiguration c;
        c.mesh = 4;
        c.product_mesh = 8;
        c.density_mesh = 16;
        c.boxsize = 100;
        const double kf = 2 * M_PI / c.boxsize;
        c.initial_cutoff = 1.7 * kf;
        c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        std::array<FF::Parameter, 4> bias = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                              {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                              {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
        FF::DistributedEulerianLPTModel model(c,
                                              FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }),
                                              bias);
        const auto p = model.initial_parameters();
        FF::Modes data;
        {
            auto w = model.workspace();
            data = model.mock(model.white_noise(4101), p, 1000., 4102, w);
        }
        const auto target = FF::fixed_noise_posterior(model, data, 1000.);
        {
            auto w = model.workspace();
            Field g;
            std::vector<double> gp;
            const double U = target.evaluate(model.white_noise(4103), p, w, g, gp);
            std::ofstream out(prefix + ".target.json");
            out << std::setprecision(17) << "{\"potential\":" << U << ",\"field_gradient\":[";
            for (size_t j = 0; j < g.size(); ++j) {
                if (j)
                    out << ',';
                out << g[j];
            }
            out << "],\"scalar_gradient\":[";
            for (size_t j = 0; j < gp.size(); ++j) {
                if (j)
                    out << ',';
                out << gp[j];
            }
            out << "],\"data\":[";
            for (size_t j = 0; j < data.size(); ++j) {
                if (j)
                    out << ',';
                out << '[' << data[j].real() << ',' << data[j].imag() << ']';
            }
            out << "]}\n";
            out.close();
            if (!out)
                throw std::runtime_error("Cannot write target reference");
        }
        const int waves[3][3] = {{1, 0, 0}, {1, 1, 0}, {0, 0, 1}};
        for (int chain = 0; chain < 4; ++chain) {
            FML::MCMC::HMCOptions o;
            o.warmup = 16;
            o.samples = 128;
            o.n_leapfrog = 8;
            o.step_size = .15;
            o.seed = 4110 + chain;
            o.jitter = false;
            o.adapt_step_size = false;
            o.adapt_mass = false;
            std::ofstream trace(prefix + ".chain" + std::to_string(chain) + ".csv");
            trace << std::setprecision(17) << "draw,c0,s0,c1,s1,c2,s2\n";
            const auto start = Clock::now();
            const auto result = FF::sample(target,
                                           model.white_noise(4103 + chain),
                                           p,
                                           o,
                                           false,
                                           [&](const Field & s, const auto &, size_t draw) {
                                               trace << draw;
                                               for (const auto & wave : waves) {
                                                   double cosine = 0, sine = 0;
                                                   for (size_t q = 0; q < s.size(); ++q) {
                                                       const auto x = model.layout().global_coordinate(q);
                                                       double phase = 0;
                                                       for (int d = 0; d < 3; ++d)
                                                           phase += 2 * M_PI * wave[d] * x[d] / 4.;
                                                       cosine += s[q] * std::cos(phase);
                                                       sine += s[q] * std::sin(phase);
                                                   }
                                                   trace << ',' << std::sqrt(2. / 64) * cosine << ','
                                                         << std::sqrt(2. / 64) * sine;
                                               }
                                               trace << '\n';
                                           });
            trace.close();
            if (!trace)
                throw std::runtime_error("Cannot write metric trace");
            const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            std::ofstream metadata(prefix + ".chain" + std::to_string(chain) + ".json");
            metadata << std::setprecision(17) << "{\"seconds\":" << seconds << ",\"acceptance\":" << result.acceptance
                     << ",\"divergences\":" << result.divergences << ",\"samples\":" << result.recorded.size() << "}\n";
            metadata.close();
            if (!metadata)
                throw std::runtime_error("Cannot write metric metadata");
            std::cout << "Completed metric chain " << chain << " in " << seconds << " s\n" << std::flush;
        }
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
