#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/Frontend.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/resource.h>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using Grid = FML::FIELDLEVEL::FRONTEND::Grid;
using Model = FF::DistributedEulerianLPTModelT<Grid>;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
static void barrier() {
#ifdef USE_MPI
    MPI_Barrier(FML::communicator);
#endif
}
static auto priors() {
    return std::array<FF::Parameter, 4>{{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
                                         {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
                                         {"b_laplacian", 7., 0, 1, -100, 100, FF::Prior::Uniform}}};
}
static Field white(const Model & m, uint64_t seed) {
    auto grid = FML::GRID::make_grid<Grid>(m.configuration().mesh, 0, 0, m.domain());
    const double gain = 1 / std::sqrt(double(m.layout().global_cells));
    for (auto i : grid.get_fourier_range()) {
        auto n = grid.get_fourier_integer_wavevector_from_index(i);
        const double k = 2 * M_PI * std::sqrt(double(n[0]) * n[0] + double(n[1]) * n[1] + double(n[2]) * n[2]) /
                         m.configuration().boxsize;
        Complex value = 0;
        if (k > 0 && k < m.configuration().initial_cutoff) {
            int sign = 1;
            for (int d = 0; d < 3; ++d)
                if (n[d]) {
                    if (n[d] < 0) {
                        for (auto & v : n)
                            v = -v;
                        sign = -1;
                    }
                    break;
                }
            const uint64_t id = (uint64_t(n[0] + 16) * 33 + n[1] + 16) * 33 + n[2] + 16;
            FML::RANDOM::COUNTER_BASED::Stream rng(seed, id, 0x4e554d53u);
            const double re = rng.normal(), im = rng.normal();
            value = gain * Complex(re, sign * im) / std::sqrt(2.);
        }
        grid.set_fourier_from_index(i, value);
    }
    grid.set_grid_status_real(false);
    grid.fftw_c2r();
    Field result;
    FML::COMMUNICATION::collective_try(
        [&] {
            result.reserve(m.cells());
            for (auto i : grid.get_real_range())
                result.push_back(grid.get_real_from_index(i));
        },
        "Numerics source");
    return result;
}
static FF::Modes read_data(const Model & m, const std::string & path) {
    FF::Modes data;
    FML::COMMUNICATION::collective_try(
        [&] {
            std::ifstream file(path);
            if (!file)
                throw std::runtime_error("Missing frozen data");
            std::map<std::array<int, 3>, Complex> all;
            std::string line;
            std::getline(file, line);
            while (std::getline(file, line)) {
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream row(line);
                std::array<int, 3> n;
                double re, im;
                std::string extra;
                if (!(row >> n[0] >> n[1] >> n[2] >> re >> im) || (row >> extra) || !std::isfinite(re) ||
                    !std::isfinite(im) || !all.emplace(n, Complex(re, im)).second)
                    throw std::runtime_error("Invalid frozen data row");
            }
            for (const auto & mode : m.modes())
                data.push_back(all.at({mode.n[0], mode.n[1], mode.n[2]}));
        },
        "Read numerical data");
    return data;
}
static void vector(std::ostream & out, const std::vector<double> & v) {
    out << '[';
    for (size_t j = 0; j < v.size(); ++j) {
        if (j)
            out << ',';
        out << v[j];
    }
    out << ']';
}
static double scaled_error(double a, double b) { return std::abs(a - b) / std::max({1., std::abs(a), std::abs(b)}); }
static FF::Modes prediction(const Model & m, const std::vector<double> & p, Model::Workspace & w) {
    const auto columns = m.bias_modes(w);
    FF::Modes out(m.modes().size(), 0);
    for (size_t j = 0; j < out.size(); ++j)
        for (int a = 0; a < 4; ++a)
            out[j] += p[a + 1] * columns[a][j];
    return out;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        std::string action, prefix, datafile;
        FF::EulerianLPTConfiguration c;
        FML::COMMUNICATION::collective_try(
            [&] {
                if (argc != 8)
                    throw std::invalid_argument("generate|assess PREFIX MESH PRODUCT DENSITY EPSILON DATAFILE-or-dash");
                action = argv[1];
                prefix = argv[2];
                c.mesh = std::stoi(argv[3]);
                c.product_mesh = std::stoi(argv[4]);
                c.density_mesh = std::stoi(argv[5]);
                c.nufft_epsilon = std::stod(argv[6]);
                datafile = argv[7];
                if (action != "generate" && action != "assess")
                    throw std::invalid_argument("Unknown action");
            },
            "Numerical screen input");
        FML::FIELDLEVEL::FRONTEND::initialize(c.mesh, c.density_mesh);
        c.boxsize = 100;
        const double kf = 2 * M_PI / c.boxsize;
        c.initial_cutoff = 2.7 * kf;
        c.matter_band = {2.8 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.likelihood_band = {2.5 * kf, FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};
        c.matter_assignment = FF::MatterAssignment::PSWF;
        const auto start = Clock::now();
        Model m(c, FF::sigma8_spectrum([](double k) {  return 20000  *  std::exp(-5  *  k); }), priors());
        auto s = white(m, 1901), direction = white(m, 1902);
        auto p = m.initial_parameters();
        const double noise = 1000.;
        if (action == "generate") {
            auto w = m.workspace();
            const auto data = m.mock(s, p, noise, 1903, w);
            FML::COMMUNICATION::collective_try(
                [&] {
                    std::ofstream out(prefix + ".rank" + std::to_string(FML::ThisTask) + ".data.csv");
                    out << std::setprecision(17) << "nx,ny,nz,re,im\n";
                    for (size_t j = 0; j < data.size(); ++j) {
                        const auto & n = m.modes()[j].n;
                        out << n[0] << ',' << n[1] << ',' << n[2] << ',' << data[j].real() << ',' << data[j].imag()
                            << '\n';
                    }
                    if (!out)
                        throw std::runtime_error("Cannot write data");
                },
                "Write frozen data");
            return 0;
        }
        const auto data = read_data(m, datafile);
        auto post = FF::fixed_noise_posterior(m, data, noise, FF::InitialFieldPrior::ConditionOnField);
        const double setup = m.maximum(elapsed(start));
        std::vector<double> potentials, scalar_gradients, forward_times, gradient_times;
        double field_fd = 0, alpha_fd = 0, cold = 0, field_analytic = 0;
        {
            auto w = m.workspace();
            Field gs;
            std::vector<double> gp;
            barrier();
            auto t = Clock::now();
            m.predict(s, p, w);
            cold = m.maximum(elapsed(t));
            for (int state = 0; state < 3; ++state) {
                p[0] = .9 + .1 * state;
                const double U = post.evaluate(s, p, w, gs, gp);
                potentials.push_back(U);
                scalar_gradients.insert(scalar_gradients.end(), gp.begin(), gp.end());
                const auto columns = m.bias_modes(w);
                const auto pred = prediction(m, p, w);
                auto gg = FML::GRID::make_grid<Grid>(c.mesh, 0, 0, m.domain());
                size_t q = 0;
                for (auto i : gg.get_real_range())
                    gg.set_real_from_index(i, gs[q++]);
                gg.set_grid_status_real(true);
                gg.fftw_r2c();
                FML::COMMUNICATION::collective_try(
                    [&] {
                        std::ofstream out(prefix + ".state" + std::to_string(state) + ".rank" +
                                          std::to_string(FML::ThisTask) + ".csv");
                        out << std::setprecision(17)
                            << "nx,ny,nz,weight,k,re,im,c0re,c0im,c1re,c1im,c2re,c2im,c3re,c3im,gre,gim\n";
                        for (size_t j = 0; j < pred.size(); ++j) {
                            const auto & mode = m.modes()[j];
                            const auto g = std::sqrt(double(m.layout().global_cells)) *
                                           Complex(gg.get_fourier_from_index(mode.fourier_index));
                            out << mode.n[0] << ',' << mode.n[1] << ',' << mode.n[2] << ',' << mode.weight << ','
                                << mode.k << ',' << pred[j].real() << ',' << pred[j].imag();
                            for (int a = 0; a < 4; ++a)
                                out << ',' << columns[a][j].real() << ',' << columns[a][j].imag();
                            out << ',' << g.real() << ',' << g.imag() << '\n';
                        }
                        if (!out)
                            throw std::runtime_error("Cannot write state modes");
                    },
                    "Write screen state");
                if (state == 1) {
                    double dot = 0;
                    for (size_t q = 0; q < s.size(); ++q)
                        dot += gs[q] * direction[q];
                    field_analytic = m.sum(dot);
                }
            }
            p[0] = 1.;
            const double h = 1e-4;
            auto a = s, b = s;
            for (size_t q = 0; q < s.size(); ++q) {
                a[q] += h * direction[q];
                b[q] -= h * direction[q];
            }
            field_fd = scaled_error(field_analytic, (post.value(a, p, w) - post.value(b, p, w)) / (2 * h));
            auto pa = p, pb = p;
            pa[0] += h;
            pb[0] -= h;
            const double Ua = post.value(s, pa, w);
            const auto ma = prediction(m, pa, w);
            const double Ub = post.value(s, pb, w);
            const auto mb = prediction(m, pb, w);
            alpha_fd = scaled_error(scalar_gradients[5], (Ua - Ub) / (2 * h));
            FML::COMMUNICATION::collective_try(
                [&] {
                    std::ofstream out(prefix + ".jacalpha.rank" + std::to_string(FML::ThisTask) + ".csv");
                    out << std::setprecision(17) << "nx,ny,nz,re,im\n";
                    for (size_t j = 0; j < ma.size(); ++j) {
                        const auto & n = m.modes()[j].n;
                        const auto v = (ma[j] - mb[j]) / (2 * h);
                        out << n[0] << ',' << n[1] << ',' << n[2] << ',' << v.real() << ',' << v.imag() << '\n';
                    }
                    if (!out)
                        throw std::runtime_error("Cannot write alpha Jacobian");
                },
                "Write alpha Jacobian");
            for (int repeat = 0; repeat < 3; ++repeat) {
                barrier();
                t = Clock::now();
                m.predict(s, p, w);
                forward_times.push_back(m.maximum(elapsed(t)));
                barrier();
                t = Clock::now();
                post.evaluate(s, p, w, gs, gp);
                gradient_times.push_back(m.maximum(elapsed(t)));
            }
        }
        auto free = FF::fixed_noise_posterior(m, data, noise);
        FML::MCMC::HMCOptions o;
        o.warmup = 0;
        o.samples = 2;
        o.n_leapfrog = 2;
        o.step_size = .0001;
        o.seed = 5150;
        o.jitter = false;
        o.adapt_step_size = false;
        o.adapt_mass = false;
        barrier();
        const auto hmc_start = Clock::now();
        const auto chain = FF::sample(free, s, p, o);
        const double hmc = m.maximum(elapsed(hmc_start));
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage))
            throw std::runtime_error("Cannot measure RSS");
#ifdef __APPLE__
        const double rss = double(usage.ru_maxrss) / (1024 * 1024);
#else
        const double rss = double(usage.ru_maxrss) / 1024;
#endif
        const double rss_max = m.maximum(rss), rss_sum = m.sum(rss);
        double modes = 0;
        for (const auto & mode : m.modes())
            modes += mode.weight;
        modes = m.sum(modes);
        FML::COMMUNICATION::collective_try(
            [&] {
                if (FML::ThisTask == 0) {
                    std::ofstream out(prefix + ".json");
                    out << std::setprecision(17) << "{\"mesh\":" << c.mesh << ",\"product\":" << m.product_mesh()
                        << ",\"density\":" << c.density_mesh << ",\"epsilon\":" << c.nufft_epsilon
                        << ",\"kernel_width\":" << m.assignment_width() << ",\"ranks\":" << FML::NTasks
                        << ",\"real_modes\":" << modes << ",\"setup_seconds\":" << setup
                        << ",\"cold_forward_seconds\":" << cold << ",\"forward_seconds\":";
                    vector(out, forward_times);
                    out << ",\"gradient_seconds\":";
                    vector(out, gradient_times);
                    out << ",\"hmc_two_draws_seconds\":" << hmc << ",\"hmc_divergences\":" << chain.divergences
                        << ",\"hmc_acceptance\":" << chain.acceptance << ",\"rank_peak_rss_max_mib\":" << rss_max
                        << ",\"sum_rank_peak_rss_mib\":" << rss_sum << ",\"field_fd_error\":" << field_fd
                        << ",\"alpha_fd_error\":" << alpha_fd << ",\"potential\":";
                    vector(out, potentials);
                    out << ",\"scalar_gradients\":";
                    vector(out, scalar_gradients);
                    out << "}\n";
                    out.close();
                    if (!out)
                        throw std::runtime_error("Cannot write screen metrics");
                }
            },
            "Write screen metrics");
    } catch (const std::exception & e) {
        if (FML::ThisTask == 0)
            std::cerr << e.what() << '\n';
        return 1;
    }
}
