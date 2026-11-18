#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <FML/FieldLevel/FixedIC.h>
#include <fstream>
#include <sstream>
#include <iomanip>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
struct Profile {
    std::array<double, 4> bias{};
    double value = HUGE_VAL, kkt = HUGE_VAL;
};
static Profile profile(const std::array<FF::Modes, 4> & columns,
                       const FF::Modes & data,
                       const std::vector<double> & weights,
                       double variance,
                       const std::array<FF::Parameter, 4> & priors,
                       const FF::FixedNoiseLinearBiasLikelihood<4> & cache) {
    std::array<double, 4> center{}, scale{}, lower{}, upper{};
    for (int a = 0; a < 4; ++a) {
        center[a] = .5 * (priors[a].lower + priors[a].upper);
        scale[a] = priors[a].scale();
        lower[a] = (priors[a].lower - center[a]) / scale[a];
        upper[a] = (priors[a].upper - center[a]) / scale[a];
    }
    long double h[4][4] = {}, rhs[4] = {}, largest = 0;
    for (size_t k = 0; k < data.size(); ++k) {
        auto residual = data[k];
        for (int a = 0; a < 4; ++a)
            residual -= center[a] * columns[a][k];
        for (int a = 0; a < 4; ++a) {
            const auto ca = scale[a] * columns[a][k];
            rhs[a] += (long double)(weights[k] / variance) * std::real(std::conj(ca) * residual);
            for (int b = 0; b < 4; ++b)
                h[a][b] += (long double)(weights[k] / variance) * std::real(std::conj(ca) * (scale[b] * columns[b][k]));
        }
    }
    for (int a = 0; a < 4; ++a)
        largest = std::max(largest, std::abs(h[a][a]));
    Profile best;
    for (int pattern = 0; pattern < 81; ++pattern) {
        int code = pattern, flag[4], free[4], m = 0;
        long double x[4] = {};
        for (int a = 0; a < 4; ++a) {
            flag[a] = code % 3;
            code /= 3;
            if (flag[a] == 0)
                free[m++] = a;
            else
                x[a] = flag[a] == 1 ? lower[a] : upper[a];
        }
        long double matrix[4][5] = {};
        for (int i = 0; i < m; ++i) {
            const int a = free[i];
            matrix[i][m] = rhs[a];
            for (int b = 0; b < 4; ++b)
                if (flag[b])
                    matrix[i][m] -= h[a][b] * x[b];
            for (int j = 0; j < m; ++j)
                matrix[i][j] = h[a][free[j]];
        }
        bool valid = true;
        for (int j = 0; j < m; ++j) {
            int pivot = j;
            for (int i = j + 1; i < m; ++i)
                if (std::abs(matrix[i][j]) > std::abs(matrix[pivot][j]))
                    pivot = i;
            if (std::abs(matrix[pivot][j]) <= 128 * std::numeric_limits<long double>::epsilon() * largest) {
                valid = false;
                break;
            }
            for (int k = j; k <= m; ++k)
                std::swap(matrix[j][k], matrix[pivot][k]);
            const long double diagonal = matrix[j][j];
            for (int k = j; k <= m; ++k)
                matrix[j][k] /= diagonal;
            for (int i = 0; i < m; ++i)
                if (i != j) {
                    const long double factor = matrix[i][j];
                    for (int k = j; k <= m; ++k)
                        matrix[i][k] -= factor * matrix[j][k];
                }
        }
        if (!valid)
            continue;
        for (int i = 0; i < m; ++i)
            x[free[i]] = matrix[i][m];
        std::array<double, 4> b{};
        for (int a = 0; a < 4; ++a) {
            valid &= std::isfinite(x[a]) && x[a] >= lower[a] - 1e-9 && x[a] <= upper[a] + 1e-9;
            b[a] = std::clamp(center[a] + scale[a] * double(x[a]), priors[a].lower, priors[a].upper);
        }
        if (!valid)
            continue;
        const double value = cache.value(b);
        if (value < best.value) {
            best.bias = b;
            best.value = value;
        }
    }
    if (!std::isfinite(best.value))
        throw std::runtime_error("No feasible bias profile");
    const auto gradient = cache.gradient(best.bias);
    double violation = 0, normalization = 1;
    for (int a = 0; a < 4; ++a) {
        const double g = gradient[a] * scale[a];
        double v = std::abs(g);
        if (best.bias[a] <= priors[a].lower + 1e-8 * scale[a])
            v = std::max(0., -g);
        else if (best.bias[a] >= priors[a].upper - 1e-8 * scale[a])
            v = std::max(0., g);
        violation = std::max(violation, v);
        normalization = std::max(normalization, double(std::abs(rhs[a])));
        long double hx = 0;
        for (int b = 0; b < 4; ++b)
            hx += h[a][b] * (best.bias[b] - center[b]) / scale[b];
        normalization = std::max(normalization, double(std::abs(hx)));
    }
    best.kkt = violation / normalization;
    if (best.kkt > 1e-8)
        throw std::runtime_error("Bias profile fails convex KKT certificate");
    return best;
}
static std::array<FF::Parameter, 4> biases() {
    return {{{"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform},
             {"b_delta2", .3, 0, 1, -4, 4, FF::Prior::Uniform},
             {"b_K2", -.2, 0, 1, -4, 4, FF::Prior::Uniform},
             {"b_laplacian", 7, 0, 1, -100, 100, FF::Prior::Uniform}}};
}
static void verify_profiler() {
    const auto prior = biases();
    std::vector<double> w(4, 1);
    std::array<FF::Modes, 4> c;
    for (auto & v : c)
        v.resize(4);
    for (int a = 0; a < 4; ++a)
        c[a][a] = 1.;
    FF::Modes d = {7., -8., 2., 150.};
    FF::FixedNoiseLinearBiasLikelihood<4> cache(c, d, w, 1., {0, 0, 0, 0});
    const auto p = profile(c, d, w, 1., prior, cache);
    const std::array<double, 4> expected = {4, -4, 2, 100};
    for (int a = 0; a < 4; ++a)
        if (std::abs(p.bias[a] - expected[a]) > 1e-10)
            throw std::runtime_error("Independent clipped solution mismatch");
    for (int a = 0; a < 4; ++a) {
        std::fill(c[a].begin(), c[a].end(), 0.);
        c[a][a] = 2.;
        c[a][(a + 1) % 4] = .4;
    }
    const std::array<double, 4> interior = {1, .2, -.3, 7};
    std::fill(d.begin(), d.end(), 0.);
    for (int a = 0; a < 4; ++a)
        for (int k = 0; k < 4; ++k)
            d[k] += interior[a] * c[a][k];
    FF::FixedNoiseLinearBiasLikelihood<4> coupled(c, d, w, 1., {0, 0, 0, 0});
    const auto q = profile(c, d, w, 1., prior, coupled);
    for (int a = 0; a < 4; ++a)
        if (std::abs(q.bias[a] - interior[a]) > 1e-10)
            throw std::runtime_error("Independent coupled solution mismatch");
    for (auto & v : c)
        std::fill(v.begin(), v.end(), 0.);
    FF::FixedNoiseLinearBiasLikelihood<4> flat(c, d, w, 1., {0, 0, 0, 0});
    const auto z = profile(c, d, w, 1., prior, flat);
    if (z.kkt != 0)
        throw std::runtime_error("Flat profile mismatch");
    std::cout << "PROFILE_REFERENCE_CHECK 3 passed\n";
}
template <class Target, class Work>
static double
endpoint_gradient_check(const Target & target, const Field & white, const std::vector<double> & p, Work & work) {
    Field gradient;
    std::vector<double> gp;
    const double u = target.evaluate(white, p, work, gradient, gp);
    if (!std::isfinite(u))
        throw std::runtime_error("Nonfinite endpoint target");
    double worst = 0;
    auto compare = [&](double fd, double analytic) {
        if (!std::isfinite(fd) || !std::isfinite(analytic))
            throw std::runtime_error("Nonfinite endpoint derivative comparison");
        const double e = std::abs(fd - analytic) / std::max({1., std::abs(fd), std::abs(analytic)});
        worst = std::max(worst, e);
        if (e > 3e-5)
            throw std::runtime_error("Endpoint field/scalar finite difference failed");
    };
    for (int direction = 0; direction < 3; ++direction) {
        Field d(white.size());
        double norm = 0;
        for (size_t q = 0; q < d.size(); ++q) {
            d[q] = std::cos(double(q + 1) * (direction + 2));
            norm += d[q] * d[q];
        }
        norm = std::sqrt(norm);
        auto plus = white, minus = white;
        double analytic = 0;
        const double h = 2e-5;
        for (size_t q = 0; q < d.size(); ++q) {
            d[q] /= norm;
            plus[q] += h * d[q];
            minus[q] -= h * d[q];
            analytic += gradient[q] * d[q];
        }
        compare((target.value(plus, p, work) - target.value(minus, p, work)) / (2 * h), analytic);
    }
    for (size_t j = 0; j < p.size(); ++j) {
        auto plus = p, minus = p;
        const double h = 1e-5 * std::max(1., std::abs(p[j]));
        plus[j] += h;
        minus[j] -= h;
        compare((target.value(white, plus, work) - target.value(white, minus, work)) / (2 * h), gp[j]);
    }
    return worst;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 3 || FML::NTasks != 1)
            throw std::invalid_argument("geometry_probe POINTS.csv RESULTS.csv, serial only");
        verify_profiler();
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
        const auto prior = biases();
        FF::DistributedEulerianLPTModel model(c,
                                              FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }),
                                              prior);
        auto work = model.workspace();
        const auto initial = model.initial_parameters();
        const auto data = model.mock(model.white_noise(4101), initial, 1000, 4102, work);
        const auto target = FF::fixed_noise_posterior(model, data, 1000.);
        const auto weights = target.likelihood().weights();
        const double variance = 1000. / model.volume();
        std::ifstream input(argv[1]);
        if (!input)
            throw std::runtime_error("Missing diagnostic points");
        std::ofstream output(argv[2]);
        output << std::setprecision(17)
               << "path,index,t,alpha,u_direct,u_profile,white_prior,quadratic_direct,quadratic_profile,kkt,profile_"
                  "full_error,b_delta,b_delta2,b_K2,b_laplacian\n";
        std::string line;
        std::getline(input, line);
        size_t count = 0;
        double worst_kkt = 0, worst_full = 0, worst_gradient = 0;
        int gradient_states = 0;
        while (std::getline(input, line)) {
            std::vector<std::string> v;
            std::istringstream stream(line);
            std::string value;
            while (std::getline(stream, value, ','))
                v.push_back(value);
            if (v.size() != 72)
                throw std::runtime_error("Wrong geometry row dimension");
            std::vector<double> p(5);
            Field white(64);
            for (int j = 0; j < 5; ++j)
                p[j] = std::stod(v[3 + j]);
            for (int j = 0; j < 64; ++j)
                white[j] = std::stod(v[8 + j]);
            const auto prediction = model.predict(white, p, work);
            double white_prior = 0, quadratic = 0, normalization = 0;
            for (double x : white)
                white_prior += .5 * x * x;
            for (size_t k = 0; k < data.size(); ++k) {
                quadratic += .5 * weights[k] * std::norm(prediction[k] - data[k]) / variance;
                normalization += .5 * weights[k] * std::log(variance);
            }
            const double u = target.value(white, p, work);
            const auto columns = model.bias_modes(work);
            FF::FixedNoiseLinearBiasLikelihood<4> cache(columns, data, weights, variance, {0, 0, 0, 0});
            const auto optimum = profile(columns, data, weights, variance, prior, cache);
            auto profile_p = p;
            std::copy(optimum.bias.begin(), optimum.bias.end(), profile_p.begin() + 1);
            const double profile_u = target.value(white, profile_p, work);
            const double error =
                std::abs(profile_u - (white_prior + optimum.value)) / std::max(1., std::abs(profile_u));
            if (error > 1e-10 ||
                std::abs(u - (white_prior + quadratic + normalization)) > 1e-10 * std::max(1., std::abs(u)) ||
                profile_u > u + 1e-8)
                throw std::runtime_error("Profile/full-target mismatch");
            worst_kkt = std::max(worst_kkt, optimum.kkt);
            worst_full = std::max(worst_full, error);
            output << v[0] << ',' << v[1] << ',' << v[2] << ',' << p[0] << ',' << u - normalization << ','
                   << profile_u - normalization << ',' << white_prior << ',' << quadratic << ','
                   << optimum.value - normalization << ',' << optimum.kkt << ',' << error;
            for (double b : optimum.bias)
                output << ',' << b;
            output << '\n';
            if ((v[0] == "white-to0" && v[1] == "0") || (v[0].rfind("white-to", 0) == 0 && v[1] == "100")) {
                worst_gradient = std::max(worst_gradient, endpoint_gradient_check(target, white, p, work));
                ++gradient_states;
            }
            ++count;
        }
        if (input.bad())
            throw std::runtime_error("Geometry input read failed");
        output.close();
        if (!output)
            throw std::runtime_error("Geometry result write failed");
        std::cout << std::setprecision(17) << "PROFILE_AUDIT_CHECK " << count << ' ' << worst_kkt << ' ' << worst_full
                  << '\n';
        if (gradient_states != 4)
            throw std::runtime_error("Endpoint gradient check count changed");
        std::cout << "ENDPOINT_GRADIENT_CHECK " << gradient_states * 8 << ' ' << worst_gradient << '\n';
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
