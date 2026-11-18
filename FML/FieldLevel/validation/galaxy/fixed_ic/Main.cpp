#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/FixedIC.h>
#include <FML/MCMC/Diagnostics.h>
#ifdef FML_FIELDLEVEL_PENCIL_GRID
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
#ifdef FML_FIELDLEVEL_PENCIL_GRID
using Grid = FML::GRID::PencilGrid<3>;
#else
using Grid = FML::GRID::FFTWGrid<3>;
#endif
using Model = FF::DistributedEulerianLPTModelT<Grid>;
static constexpr double noise_power = 2.;
static std::array<FF::Parameter, 4> biases() {
    return {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
             {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
             {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
             {"b_laplacian", 3., 0, 1, -20, 20, FF::Prior::Uniform}}};
}
static FF::EulerianLPTConfiguration configuration(int n, int nd, int width) {
    FF::EulerianLPTConfiguration c;
    c.mesh = n;
    c.density_mesh = nd;
    c.kernel_width = width;
    c.boxsize = 100;
    double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 2.4 * kf;
    c.matter_band = {2.5 * kf, E::BandGeometry::Cube};
    c.likelihood_band = {2.1 * kf, E::BandGeometry::Cube};
    return c;
}
static FML::Domain domain(int n) {
#ifdef FML_FIELDLEVEL_PENCIL_GRID
    int px = FML::NTasks % 2 == 0 ? 2 : 1, py = FML::NTasks / px;
    std::vector<double> ex(px + 1), ey(py + 1);
    for (int p = 0; p <= px; ++p)
        ex[p] = double(p * n / px) / n;
    for (int p = 0; p <= py; ++p)
        ey[p] = double(p * n / py) / n;
    return FML::Domain::pencil(ex, ey);
#else
    (void)n;
    return FML::Domain::uniform_slab(FML::NTasks);
#endif
}
// A fixed band-limited white realization. The 1/sqrt(N^3) Fourier scaling
// keeps the physical linear field identical when the particle/product mesh changes.
static FML::FIELDLEVEL::Field initial_field(const Model & m) {
    auto s = m.white_noise(0);
    std::fill(s.begin(), s.end(), 0.);
    int n = m.configuration().mesh;
    for (int nx = -2; nx <= 2; ++nx)
        for (int ny = -2; ny <= 2; ++ny)
            for (int nz = -2; nz <= 2; ++nz) {
                int n2 = nx * nx + ny * ny + nz * nz;
                bool positive = nx > 0 || (nx == 0 && ny > 0) || (nx == 0 && ny == 0 && nz > 0);
                if (!positive || n2 >= 2.4 * 2.4)
                    continue;
                uint64_t id = ((nx + 3) * 7 + ny + 3) * 7 + nz + 3;
                FML::RANDOM::COUNTER_BASED::Stream rng(6201, id, 0x47414c41u, 0);
                double a = rng.normal(), b = rng.normal();
                for (size_t q = 0; q < s.size(); ++q) {
                    auto x = m.layout().global_coordinate(q);
                    double phase = 2 * M_PI * (nx * x[0] + ny * x[1] + nz * x[2]) / n;
                    s[q] += std::sqrt(2. / m.layout().global_cells) * (a * std::cos(phase) - b * std::sin(phase));
                }
            }
    return s;
}
static std::vector<std::string> split(std::string line) {
    std::vector<std::string> fields;
    std::istringstream in(line);
    std::string field;
    while (std::getline(in, field, ','))
        fields.push_back(field);
    return fields;
}
static FF::Modes load_data(const Model & m, const std::string & path) {
    std::map<std::array<int, 3>, FML::FIELDLEVEL::Complex> lookup;
    FML::COMMUNICATION::collective_try(
        [&] {
            std::ifstream in(path);
            if (!in)
                throw std::runtime_error("Cannot open data file");
            std::string line;
            std::getline(in, line);
            while (std::getline(in, line)) {
                auto f = split(line);
                if (f.size() != 9)
                    throw std::runtime_error("Invalid data row");
                auto key = std::array<int, 3>{std::stoi(f[0]), std::stoi(f[1]), std::stoi(f[2])};
                if (!lookup.emplace(key, FML::FIELDLEVEL::Complex(std::stod(f[5]), std::stod(f[6]))).second)
                    throw std::runtime_error("Duplicate data mode");
            }
        },
        "[Galaxy pilot] Read fixed data");
    FF::Modes data;
    FML::COMMUNICATION::collective_try(
        [&] {
            for (const auto & mode : m.modes())
                data.push_back(lookup.at({mode.n[0], mode.n[1], mode.n[2]}));
        },
        "[Galaxy pilot] Map local modes");
    m.validate_modes(data);
    return data;
}
static void prepare(const std::string & prefix) {
    if (FML::NTasks != 1)
        throw std::invalid_argument("Prepare uses one task");
    auto c = configuration(16, 48, 8);
    Model m(c, FF::sigma8_spectrum([](double k) { return 2000 * std::exp(-5 * k); }), biases(), domain(c.mesh));
    auto w = m.workspace();
    auto s = initial_field(m);
    auto p = m.initial_parameters();
    p[0] = 1.;
    p[1] = 1.8;
    p[2] = .4;
    p[3] = -.25;
    p[4] = 3.;
    auto signal = m.predict(s, p, w);
    auto data = m.mock(s, p, noise_power, 6202, w);
    std::ofstream out(prefix + ".data.csv");
    out << std::setprecision(17) << "nx,ny,nz,k,weight,data_re,data_im,signal_re,signal_im\n";
    for (size_t j = 0; j < data.size(); ++j) {
        const auto & mode = m.modes()[j];
        out << mode.n[0] << ',' << mode.n[1] << ',' << mode.n[2] << ',' << mode.k << ',' << mode.weight << ','
            << data[j].real() << ',' << data[j].imag() << ',' << signal[j].real() << ',' << signal[j].imag() << '\n';
    }
    if (!out)
        throw std::runtime_error("Failed to save mock");
    std::ofstream info(prefix + ".problem.csv");
    info << std::setprecision(17)
         << "boxsize,initial_kf,matter_kf,likelihood_kf,noise_power,n,nd,width,ic_seed,noise_seed";
    for (int j = 0; j < 5; ++j)
        info << ",truth" << j << ",lower" << j << ",upper" << j;
    info << '\n'
         << c.boxsize << ',' << c.initial_cutoff * c.boxsize / (2 * M_PI) << ','
         << c.matter_band.cutoff * c.boxsize / (2 * M_PI) << ',' << c.likelihood_band.cutoff * c.boxsize / (2 * M_PI)
         << ',' << noise_power << ',' << c.mesh << ',' << c.density_mesh << ',' << c.kernel_width << ",6201,6202";
    for (int j = 0; j < 5; ++j)
        info << ',' << p[j] << ',' << m.parameters()[j].lower << ',' << m.parameters()[j].upper;
    info << '\n';
    if (!info)
        throw std::runtime_error("Failed to save problem metadata");
}
static void run(int argc, char ** argv) {
    if (argc != 11 && argc != 12)
        throw std::invalid_argument(
            "chain DATA PREFIX N ND WIDTH SEED ALPHA WARMUP DRAWS; predict DATA PREFIX N ND WIDTH ALPHA B1 B2 BK BLAP");
    const bool predict_only = std::string(argv[1]) == "predict";
    const bool joint = std::string(argv[1]) == "chain-joint";
    if ((predict_only && argc != 12) || (!predict_only && argc != 11))
        throw std::invalid_argument("Wrong argument count");
    auto c = configuration(std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]));
    Model m(c, FF::sigma8_spectrum([](double k) { return 2000 * std::exp(-5 * k); }), biases(), domain(c.mesh));
    auto w = m.workspace();
    auto s = initial_field(m);
    auto data = load_data(m, argv[2]);
    auto p = m.initial_parameters();
    if (predict_only) {
        for (int j = 0; j < 5; ++j)
            p[j] = std::stod(argv[7 + j]);
        const auto prediction = m.predict(s, p, w);
        // Small validation export only: retain each rank's owned modes in its own file.
        std::ofstream out(std::string(argv[3]) + ".rank" + std::to_string(FML::ThisTask) + ".modes.csv");
        out << std::setprecision(17) << "nx,ny,nz,k,weight,data_re,data_im,model_re,model_im\n";
        for (size_t j = 0; j < prediction.size(); ++j) {
            auto mode = m.modes()[j];
            out << mode.n[0] << ',' << mode.n[1] << ',' << mode.n[2] << ',' << mode.k << ',' << mode.weight << ','
                << data[j].real() << ',' << data[j].imag() << ',' << prediction[j].real() << ',' << prediction[j].imag()
                << '\n';
        }
        if (!out)
            throw std::runtime_error("Failed prediction export");
        return;
    }
    FF::FixedICOptions options;
    options.amplitude_step = .025;
    if (joint)
        options.bias_sweeps = 0;
    options.seed = std::stoull(argv[7]);
    options.warmup = std::stoi(argv[9]);
    options.samples = std::stoi(argv[10]);
    std::array<double, 5> start = {std::stod(argv[8]), p[1], p[2], p[3], p[4]};
    std::array<double, 4> reference = {p[1], p[2], p[3], p[4]};
    std::vector<double> weights;
    for (auto mode : m.modes())
        weights.push_back(mode.weight);
    auto provider = [&](double alpha) {
        p[0] = alpha;
        m.predict(s, p, w);
        return FF::FixedNoiseLinearBiasLikelihood<4>(m.bias_modes(w),
                                                     data,
                                                     weights,
                                                     noise_power / m.volume(),
                                                     reference);
    };
    const auto before = std::chrono::steady_clock::now();
    auto result =
        joint ? FF::sample_fixed_ic_joint_amplitude<4>(provider, m.parameters()[0], biases(), start, options) :
                FF::sample_fixed_ic_amplitude<4>(provider, m.parameters()[0], biases(), {1, 2, 2, 1}, start, options);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();
    if (FML::ThisTask == 0) {
        std::ofstream out(std::string(argv[3]) + ".chain.csv");
        out << std::setprecision(17) << "draw,alpha,b_delta,b_delta2,b_K2,b_laplacian,potential\n";
        for (size_t i = 0; i < result.draws.size(); ++i) {
            out << i;
            for (auto x : result.draws[i])
                out << ',' << x;
            out << ',' << result.potential[i] << '\n';
        }
        if (!out)
            throw std::runtime_error("Failed chain export");
        std::ofstream meta(std::string(argv[3]) + ".meta.csv");
        meta << std::setprecision(17)
             << "n,nd,width,seed,warmup,draws,amplitude_step,bias_sweeps,acceptance,evaluations,seconds,joint_proposal,"
                "support_rejections\n"
             << c.mesh << ',' << c.density_mesh << ',' << c.kernel_width << ',' << options.seed << ',' << options.warmup
             << ',' << options.samples << ',' << options.amplitude_step << ',' << options.bias_sweeps << ','
             << result.sampling_acceptance << ',' << result.forward_evaluations << ',' << seconds << ',' << int(joint)
             << ',' << result.support_rejections << '\n';
        if (!meta)
            throw std::runtime_error("Failed metadata export");
        std::cout << "CHAIN " << argv[3] << " seconds=" << seconds << " acceptance=" << result.sampling_acceptance
                  << std::endl;
    }
}
static void diagnose(int argc, char ** argv) {
    if (argc < 5 || FML::NTasks != 1)
        throw std::invalid_argument("diagnose OUTPUT CHAIN1 CHAIN2 [...], one task");
    std::array<std::vector<std::vector<double>>, 5> values;
    for (int f = 3; f < argc; ++f) {
        std::ifstream in(argv[f]);
        if (!in)
            throw std::runtime_error("Missing chain");
        std::string line;
        std::getline(in, line);
        std::array<std::vector<double>, 5> chain;
        while (std::getline(in, line)) {
            auto row = split(line);
            if (row.size() != 7)
                throw std::runtime_error("Invalid chain row");
            for (int j = 0; j < 5; ++j)
                chain[j].push_back(std::stod(row[j + 1]));
        }
        for (int j = 0; j < 5; ++j)
            values[j].push_back(std::move(chain[j]));
    }
    std::ofstream out(argv[2]);
    out << std::setprecision(17) << "parameter,status,rhat,bulk_ess,tail_ess\n";
    const char * names[] = {"alpha", "b_delta", "b_delta2", "b_K2", "b_laplacian"};
    for (int j = 0; j < 5; ++j) {
        auto d = FML::MCMC::diagnose_chains(values[j]);
        out << names[j] << ',' << FML::MCMC::diagnostic_status_name(d.status) << ',' << d.rhat << ',' << d.ess_bulk
            << ',' << d.ess_tail << '\n';
    }
    if (!out)
        throw std::runtime_error("Failed diagnostics export");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc == 3 && std::string(argv[1]) == "prepare")
            prepare(argv[2]);
        else if (argc >= 2 && std::string(argv[1]) == "diagnose")
            diagnose(argc, argv);
        else if (argc >= 2 && (std::string(argv[1]) == "chain" || std::string(argv[1]) == "chain-joint" ||
                               std::string(argv[1]) == "predict"))
            run(argc, argv);
        else
            throw std::invalid_argument("Use prepare, chain, predict or diagnose");
    } catch (const std::exception & e) {
        std::cerr << "Galaxy fixed-IC pilot: " << e.what() << std::endl;
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
    return 0;
}
