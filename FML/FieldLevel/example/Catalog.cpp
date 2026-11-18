#include <FML/FieldLevel/PeriodicCatalog.h>
#include <FML/FieldLevel/EulerianBias.h>
#ifdef FML_FIELDLEVEL_PENCIL_GRID
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <fstream>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
#ifdef FML_FIELDLEVEL_PENCIL_GRID
using Grid = FML::GRID::PencilGrid<3>;
#else
using Grid = FML::GRID::FFTWGrid<3>;
#endif
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        int good = argc >= 7 && argc <= 9;
        FML::MinOverTasks(&good);
        if (!good)
            throw std::invalid_argument("Usage: catalog_to_modes INPUT.csv PREFIX NMESH BOXSIZE KMAX cube|sphere "
                                        "[BATCH_ROWS=100000] [EPSILON=1e-8]");
        int n = 0;
        double cutoff = 0;
        size_t batch_size = 0;
        FF::PeriodicCatalogSettings config;
        FML::FIELDLEVEL::EULERIAN::BandGeometry geometry{};
        FML::COMMUNICATION::collective_try(
            [&] {
                size_t consumed;
                n = std::stoi(argv[3], &consumed);
                if (consumed != std::string(argv[3]).size())
                    throw std::invalid_argument("Invalid mesh");
                config.boxsize = std::stod(argv[4], &consumed);
                if (consumed != std::string(argv[4]).size())
                    throw std::invalid_argument("Invalid box");
                cutoff = std::stod(argv[5], &consumed);
                if (consumed != std::string(argv[5]).size())
                    throw std::invalid_argument("Invalid cutoff");
                const std::string band = argv[6];
                if (band != "cube" && band != "sphere")
                    throw std::invalid_argument("Use cube or sphere");
                geometry = band == "cube" ? FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube :
                                            FML::FIELDLEVEL::EULERIAN::BandGeometry::Sphere;
                batch_size = argc >= 8 ? std::stoull(argv[7], &consumed) : 100000;
                if (argc >= 8 && (argv[7][0] == '-' || consumed != std::string(argv[7]).size()))
                    throw std::invalid_argument("Invalid batch size");
                config.epsilon = argc >= 9 ? std::stod(argv[8], &consumed) : 1e-8;
                if (argc >= 9 && consumed != std::string(argv[8]).size())
                    throw std::invalid_argument("Invalid epsilon");
                if (n < 4 || n % 2 || n > INT_MAX / 128 || !std::isfinite(config.boxsize) || config.boxsize <= 0 ||
                    !std::isfinite(cutoff) || cutoff <= 0 || cutoff >= M_PI * n / config.boxsize || batch_size == 0 ||
                    batch_size > std::vector<FF::CatalogEntry>().max_size())
                    throw std::invalid_argument("Invalid catalog geometry/buffer");
            },
            "Catalog CLI controls");
        for (double x : {double(n), config.boxsize, cutoff, double(geometry), config.epsilon}) {
            double lo = x, hi = x;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            if (lo != hi)
                throw std::invalid_argument("Catalog CLI controls differ across ranks");
        }
        FML::COMMUNICATION::Count lo = batch_size, hi = batch_size;
        FML::MinOverTasks(&lo);
        FML::MaxOverTasks(&hi);
        if (lo != hi)
            throw std::invalid_argument("Catalog CLI batch sizes differ across ranks");
        config.max_input_rows = batch_size;
        config.max_received_rows = std::max(batch_size, config.max_received_rows);
        FML::Domain domain = FML::Domain::uniform_slab(FML::NTasks);
#ifdef FML_FIELDLEVEL_PENCIL_GRID
        const int px = FML::NTasks % 2 ? 1 : 2, py = FML::NTasks / px;
        if (px > n || py > n / 2 + 1)
            throw std::invalid_argument("Too many pencil ranks for this mesh");
        std::vector<double> x(px + 1), y(py + 1);
        for (int p = 0; p <= px; ++p)
            x[p] = double(p * n / px) / n;
        for (int p = 0; p <= py; ++p)
            y[p] = double(p * n / py) / n;
        domain = FML::Domain::pencil(x, y);
#else
        good = n % FML::NTasks == 0;
        FML::MinOverTasks(&good);
        if (!good)
            throw std::invalid_argument("Slab mesh must be divisible by rank count");
#endif
        auto prototype = FML::GRID::make_grid<Grid>(n, 0, 0, domain);
        FF::PeriodicCatalogT<Grid> catalog(prototype, config);
        std::ifstream input;
        std::istream * source = nullptr;
        size_t line_number = 0;
        bool stream_done = false;
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    if (std::string(argv[1]) == "-")
                        source = &std::cin;
                    else {
                        input.open(argv[1]);
                        if (!input)
                            throw std::runtime_error("Cannot open catalog");
                        source = &input;
                    }
                }
            },
            "Open catalog");
        for (;;) {
            std::vector<FF::CatalogEntry> batch;
            FML::COMMUNICATION::collective_try(
                [&] {
                    if (FML::ThisTask != 0)
                        return;
                    std::string line;
                    while (!stream_done && batch.size() < batch_size && std::getline(*source, line)) {
                        ++line_number;
                        if (source == &std::cin && line == "#FML_CATALOG_END") {
                            stream_done = true;
                            break;
                        }
                        if (line.empty() || line[0] == '#')
                            continue;
                        if (line == "x,y,z" || line == "x,y,z,weight")
                            continue;
                        FF::CatalogEntry e;
                        if (line.find(',') != std::string::npos) {
                            if (line.back() == ',')
                                throw std::runtime_error("Missing CSV value at row " + std::to_string(line_number));
                            std::istringstream csv(line);
                            std::string token;
                            std::vector<double> values;
                            while (std::getline(csv, token, ',')) {
                                size_t used;
                                double value = std::stod(token, &used);
                                for (; used < token.size(); ++used)
                                    if (!std::isspace(static_cast<unsigned char>(token[used])))
                                        throw std::runtime_error("Invalid CSV value at row " +
                                                                 std::to_string(line_number));
                                values.push_back(value);
                            }
                            if (values.size() != 3 && values.size() != 4)
                                throw std::runtime_error("Expected 3 or 4 columns at row " +
                                                         std::to_string(line_number));
                            for (int d = 0; d < 3; ++d)
                                e.position[d] = values[d];
                            if (values.size() == 4)
                                e.weight = values[3];
                        } else {
                            std::istringstream row(line);
                            if (!(row >> e.position[0] >> e.position[1] >> e.position[2]))
                                throw std::runtime_error("Invalid catalog row " + std::to_string(line_number));
                            row >> std::ws;
                            if (!row.eof() && !(row >> e.weight))
                                throw std::runtime_error("Invalid weight at row " + std::to_string(line_number));
                            row >> std::ws;
                            if (!row.eof())
                                throw std::runtime_error("Extra data at row " + std::to_string(line_number));
                        }
                        batch.push_back(e);
                    }
                    if (source->bad())
                        throw std::runtime_error("Catalog read failed");
                },
                "Read catalog batch");
            unsigned long long rows = batch.size();
#ifdef USE_MPI
            MPI_Bcast(&rows, 1, MPI_UNSIGNED_LONG_LONG, 0, FML::communicator);
#endif
            if (rows == 0)
                break;
            catalog.add_batch(batch);
        }
        const auto stats = catalog.finish();
        const auto & grid = catalog.fourier_grid();
        FML::FIELDLEVEL::EULERIAN::SharpKBand band{cutoff, geometry};
        FML::COMMUNICATION::collective_try(
            [&] {
                std::ofstream modes(std::string(argv[2]) + ".rank" + std::to_string(FML::ThisTask) + ".modes.csv");
                modes << std::setprecision(17) << "nx,ny,nz,k,weight,delta_re,delta_im\n";
                for (auto i : grid.get_fourier_range()) {
                    const auto k = grid.get_fourier_integer_wavevector_from_index(i);
                    if (!band.contains(k, config.boxsize, n))
                        continue;
                    double k2 = 0;
                    for (int v : k)
                        k2 += double(v) * v;
                    const auto value = grid.get_fourier_from_index(i);
                    modes << k[0] << ',' << k[1] << ',' << k[2] << ',' << 2 * M_PI * std::sqrt(k2) / config.boxsize
                          << ',' << FML::GRID::fourier_mode_multiplicity(k[2], n) << ',' << value.real() << ','
                          << value.imag() << '\n';
                }
                if (!modes)
                    throw std::runtime_error("Failed mode output");
                if (FML::ThisTask == 0) {
                    std::ofstream meta(std::string(argv[2]) + ".json");
                    meta << std::setprecision(17) << "{\n  \"rows\": " << stats.rows
                         << ",\n  \"boxsize\": " << config.boxsize << ",\n  \"mesh\": " << n
                         << ",\n  \"fine_mesh\": " << catalog.fine_mesh()
                         << ",\n  \"kernel_width\": " << catalog.kernel_width()
                         << ",\n  \"epsilon\": " << config.epsilon << ",\n  \"ranks\": " << FML::NTasks
                         << ",\n  \"cutoff\": " << cutoff << ",\n  \"geometry\": \"" << argv[6]
                         << "\",\n  \"effective_count\": " << stats.effective_count
                         << ",\n  \"poisson_power\": " << stats.poisson_power
                         << ",\n  \"mode_variance\": " << stats.mode_variance << "\n}\n";
                    if (!meta)
                        throw std::runtime_error("Failed catalog metadata output");
                }
            },
            "Save catalog modes");
    } catch (const std::exception & e) {
        std::cerr << "catalog_to_modes: " << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
    return 0;
}
