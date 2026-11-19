#ifndef FML_VALIDATION_GAUSS_NEWTON_FIXTURE_H
#define FML_VALIDATION_GAUSS_NEWTON_FIXTURE_H
#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <fstream>
namespace GN {
    namespace FF = FML::FIELDLEVEL::FRAMEWORK;
    using Field = FML::FIELDLEVEL::Field;
    using Model = FF::DistributedEulerianLPTModel;
    struct Frame {
        unsigned ensemble, draw;
        std::vector<double> frozen, parameters;
        Field white;
        FF::Modes prediction;
        double quadratic;
    };
    struct Fixture {
        FF::Modes data;
        std::vector<Frame> frames;
        explicit Fixture(const std::string & path) {
            std::ifstream in(path);
            std::string tag;
            unsigned mid, nm, nf, trials, cases;
            if (!(in >> tag >> mid >> nm >> nf >> trials >> cases) || tag != "FML_STABILITY_1" || mid != 203 ||
                nf != 12 || nm < 1 || nm > 64 || cases > 8)
                throw std::invalid_argument("Expected frozen bounded dataset-203 fixture");
            for (unsigned j = 0; j < cases; ++j) {
                std::string name;
                int current, leaps;
                double step;
                if (!(in >> name >> current >> step >> leaps))
                    throw std::invalid_argument("Missing case");
            }
            data.resize(nm);
            auto modes = [&](FF::Modes & v) {
                for (auto & z : v) {
                    double re, im;
                    if (!(in >> re >> im) || !std::isfinite(re) || !std::isfinite(im))
                        throw std::invalid_argument("Invalid modes");
                    z = {re, im};
                }
            };
            modes(data);
            for (unsigned j = 0; j < nf; ++j) {
                Frame f;
                if (!(in >> f.ensemble >> f.draw) || f.ensemble > 3 ||
                    (f.draw != 0 && f.draw != 1024 && f.draw != 2047))
                    throw std::invalid_argument("Unexpected frame");
                f.frozen.resize(5);
                f.parameters.resize(5);
                f.white.resize(64);
                f.prediction.resize(nm);
                for (auto * v : {&f.frozen, &f.parameters, &f.white})
                    for (auto & x : *v)
                        if (!(in >> x) || !std::isfinite(x))
                            throw std::invalid_argument("Invalid frame vector");
                modes(f.prediction);
                if (!(in >> f.quadratic) || !std::isfinite(f.quadratic))
                    throw std::invalid_argument("Invalid Q");
                frames.push_back(std::move(f));
            }
            std::string extra;
            if (in >> extra)
                throw std::invalid_argument("Extra fixture input");
            for (unsigned e = 0; e < 4; ++e)
                for (unsigned draw : {0u, 1024u, 2047u}) {
                    size_t count = 0;
                    for (const auto & f : frames)
                        count += f.ensemble == e && f.draw == draw;
                    if (count != 1)
                        throw std::invalid_argument("Missing/duplicate frame");
                }
        }
    };
    inline Model model() {
        FF::EulerianLPTConfiguration c;
        c.mesh = 4;
        c.product_mesh = 8;
        c.density_mesh = 16;
        c.boxsize = 100;
        const double kf = 2 * M_PI / 100;
        c.initial_cutoff = 1.7 * kf;
        c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        std::array<FF::Parameter, 4> biases = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                                {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                                {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                                {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
        return Model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), biases);
    }
} // namespace GN
#endif
