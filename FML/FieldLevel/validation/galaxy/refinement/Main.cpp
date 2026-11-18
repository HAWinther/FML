#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using Model = FF::DistributedEulerianLPTModel;
using FML::FIELDLEVEL::Field;
namespace FS = std::filesystem;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 8 || std::string(argv[1]) != "predict" || FML::NTasks != 1)
            throw std::invalid_argument(
                "refinement_predict predict INPUT DATA OUTPUT PRODUCT DENSITY EPSILON; serial only");
        FF::EulerianLPTConfiguration c;
        c.mesh = 4;
        size_t used;
        c.product_mesh = std::stoi(argv[5], &used);
        if (used != std::string(argv[5]).size())
            throw std::invalid_argument("Invalid product mesh");
        c.density_mesh = std::stoi(argv[6], &used);
        if (used != std::string(argv[6]).size())
            throw std::invalid_argument("Invalid density mesh");
        c.nufft_epsilon = std::stod(argv[7], &used);
        if (used != std::string(argv[7]).size() || !std::isfinite(c.nufft_epsilon) || c.nufft_epsilon <= 0 ||
            c.nufft_epsilon > 1e-4 || c.product_mesh < 8 || c.product_mesh > 24 || c.product_mesh % 2 ||
            c.density_mesh < c.product_mesh || c.density_mesh > 64 || c.density_mesh % 2)
            throw std::invalid_argument("Refinement controls exceed frozen-fixture bounds");
        c.boxsize = 100;
        const double kf = 2 * M_PI / c.boxsize;
        c.initial_cutoff = 1.7 * kf;
        c.matter_band = {1.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {1.6 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        std::array<FF::Parameter, 4> priors = {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                                {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                                {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                                {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
        Model model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), priors);
        auto work = model.workspace();
        std::ifstream datafile(argv[3]);
        size_t count;
        std::string tag;
        if (!(datafile >> tag >> count) || tag != "FML_MODES_1" || count != model.modes().size())
            throw std::runtime_error("Frozen mode count differs");
        FF::Modes data(count);
        for (size_t j = 0; j < count; ++j) {
            size_t id;
            double k, weight, re, im;
            const auto & mode = model.modes()[j];
            if (!(datafile >> id >> k >> weight >> re >> im) || id != j || std::abs(k - mode.k) > 1e-14 ||
                weight != mode.weight || !std::isfinite(re) || !std::isfinite(im))
                throw std::runtime_error("Frozen mode order/weight/data differs");
            data[j] = {re, im};
        }
        std::string extra;
        if (datafile >> extra)
            throw std::runtime_error("Extra frozen mode fields");
        std::ifstream input(argv[2]);
        size_t rows, offset;
        if (!(input >> tag >> rows >> offset) || tag != "FML_STATES_1" || rows < 1 || rows > 256 ||
            offset > 65536 - rows)
            throw std::runtime_error("Invalid bounded state batch");
        std::vector<Field> white(rows, Field(64));
        std::vector<std::vector<double>> parameters(rows, std::vector<double>(5));
        for (size_t j = 0; j < rows; ++j) {
            size_t id;
            if (!(input >> id) || id != offset + j)
                throw std::runtime_error("State index differs");
            for (auto & v : white[j])
                if (!(input >> v) || !std::isfinite(v))
                    throw std::runtime_error("Invalid field");
            for (auto & v : parameters[j])
                if (!(input >> v) || !std::isfinite(v))
                    throw std::runtime_error("Invalid parameter");
            if (!model.in_support(parameters[j]))
                throw std::runtime_error("State outside scalar support");
        }
        if (input >> extra)
            throw std::runtime_error("Extra state fields");
        const FS::path final = FS::absolute(argv[4]);
        if (FS::exists(final))
            throw std::runtime_error("Prediction output already exists");
        FS::create_directories(final.parent_path());
        const FS::path temporary = final.string() + ".tmp";
        std::ofstream out(temporary);
        out << std::setprecision(17) << "draw,quadratic";
        for (size_t j = 0; j < count; ++j)
            out << ",re" << j << ",im" << j;
        out << '\n';
        const auto start = std::chrono::steady_clock::now();
        for (size_t j = 0; j < rows; ++j) {
            const auto & pred = model.predict(white[j], parameters[j], work);
            double q = 0;
            for (size_t k = 0; k < count; ++k) {
                if (!std::isfinite(pred[k].real()) || !std::isfinite(pred[k].imag()))
                    throw std::runtime_error("Nonfinite prediction");
                q += .5 * model.modes()[k].weight * std::norm(pred[k] - data[k]) * model.volume() / 1000.;
            }
            if (!std::isfinite(q) || q < 0)
                throw std::runtime_error("Nonfinite likelihood quadratic");
            out << offset + j << ',' << q;
            for (auto z : pred)
                out << ',' << z.real() << ',' << z.imag();
            out << '\n';
        }
        out.close();
        if (!out)
            throw std::runtime_error("Cannot publish predictions");
        FS::rename(temporary, final);
        std::cout << "REFINEMENT " << rows << " states, " << count << " modes, "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " seconds\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
