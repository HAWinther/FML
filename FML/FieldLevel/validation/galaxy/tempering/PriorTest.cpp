#include "MockSettings.h"
#include "PriorPredictive.h"
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 2 || FML::NTasks != 1)
            throw std::invalid_argument("tempering_prior_test OUTPUT.csv, serial only");
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
        auto invalid = model.parameters();
        invalid[0].prior = FF::Prior::Gaussian;
        bool rejected = false;
        try {
            MOCK::prior_parameters(invalid, 1);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("Unsupported scalar prior accepted");
        std::ofstream out(argv[1]);
        out << std::setprecision(17) << "sample";
        for (const auto & p : model.parameters())
            out << ',' << p.name;
        for (int j = 0; j < 64; ++j)
            out << ",white" << j;
        for (int j = 0; j < 64; ++j)
            out << ",noise" << j;
        out << '\n';
        for (unsigned k = 0; k < 1024; ++k) {
            MOCK::Seeds seeds;
            std::istringstream input("FML_MOCK_2 101 " + std::to_string(20001 + 8 * k) + " " +
                                     std::to_string(20002 + 8 * k) + " " + std::to_string(20003 + 8 * k) + " " +
                                     std::to_string(20007 + 8 * k));
            seeds.read(input, 4);
            auto p = MOCK::prior_parameters(model.parameters(), seeds.generation_key(1));
            const auto white = model.white_noise(seeds.generation_key(0));
            const auto noise = model.white_noise(seeds.generation_key(2));
            if (p != MOCK::prior_parameters(model.parameters(), seeds.generation_key(1)) ||
                white != model.white_noise(seeds.generation_key(0)))
                throw std::runtime_error("Nonreproducible prior draw");
            if (white.size() != 64 || noise.size() != 64)
                throw std::runtime_error("Incomplete latent prior");
            out << k;
            for (double x : p)
                out << ',' << x;
            for (double x : white)
                out << ',' << x;
            for (double x : noise)
                out << ',' << x;
            out << '\n';
        }
        out.close();
        if (!out)
            throw std::runtime_error("Cannot write prior reference draws");
        std::cout << "PASS: 1024 complete prior/noise draws; exact reproducibility and unsupported-prior rejection\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
