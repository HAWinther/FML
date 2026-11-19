#include "Fixture.h"
#include "Metric.h"
#include <iomanip>
#include <iostream>
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    try {
        if (argc != 3 || FML::NTasks != 1)
            throw std::invalid_argument("gn_build FIXTURE OUTPUT_PREFIX, serial only");
        GN::Fixture fixture(argv[1]);
        auto model = GN::model();
        auto work = model.workspace();
        auto target = GN::FF::fixed_noise_posterior(model, fixture.data, 1000.);
        if (model.modes().size() != fixture.data.size())
            throw std::invalid_argument("Mode layout changed");
        const double variance = 1000. / model.volume();
        std::ofstream rows(std::string(argv[2]) + ".rows.csv"), checks(std::string(argv[2]) + ".checks.csv");
        rows << std::setprecision(17) << "ensemble,draw,row";
        for (size_t j = 0; j < 64; ++j)
            rows << ",x" << j;
        rows << '\n';
        checks << std::setprecision(17)
               << "ensemble,draw,prediction_error,quadratic_error,gradient_reconstruction_error,fd_1e4,fd_1e5\n";
        for (const auto & f : fixture.frames) {
            const auto prediction = model.predict(f.white, f.parameters, work);
            double pe = 0, q = 0;
            for (size_t j = 0; j < prediction.size(); ++j) {
                pe = std::max(pe, std::abs(prediction[j] - f.prediction[j]));
                q += .5 * target.likelihood().weights()[j] * std::norm(prediction[j] - fixture.data[j]) / variance;
            }
            const double qe = std::abs(q - f.quadratic) / std::max(1., std::abs(q));
            if (pe > 1e-10 || qe > 1e-11)
                throw std::runtime_error("Original prediction/Q changed");
            std::vector<GN::Field> jacobian;
            std::vector<double> residual;
            for (size_t j = 0; j < prediction.size(); ++j)
                for (int part = 0; part < 2; ++part) {
                    const double weight = target.likelihood().weights()[j];
                    if (!(weight > 0))
                        throw std::runtime_error("Unexpected zero mode weight");
                    GN::FF::Modes bar(prediction.size(), 0.);
                    bar[j] = part ? FML::FIELDLEVEL::Complex(0., 1. / std::sqrt(weight * variance)) :
                                    FML::FIELDLEVEL::Complex(1. / std::sqrt(weight * variance), 0.);
                    model.predict(f.white, f.parameters, work);
                    GN::Field row;
                    std::vector<double> gp;
                    model.pullback(f.parameters, bar, work, row, gp);
                    rows << f.ensemble << ',' << f.draw << ',' << jacobian.size();
                    for (double x : row) {
                        if (!std::isfinite(x))
                            throw std::runtime_error("Nonfinite Jacobian");
                        rows << ',' << x;
                    }
                    rows << '\n';
                    jacobian.push_back(row);
                    const auto r = prediction[j] - fixture.data[j];
                    residual.push_back(std::sqrt(weight / variance) * (part ? r.imag() : r.real()));
                }
            GN::Field gradient;
            std::vector<double> gp;
            target.evaluate(f.white, f.parameters, work, gradient, gp);
            GN::Field rebuilt = f.white;
            for (size_t row = 0; row < jacobian.size(); ++row)
                for (size_t j = 0; j < 64; ++j)
                    rebuilt[j] += jacobian[row][j] * residual[row];
            double ge = 0;
            for (size_t j = 0; j < 64; ++j)
                ge = std::max(ge,
                              std::abs(rebuilt[j] - gradient[j]) /
                                  std::max({1., std::abs(rebuilt[j]), std::abs(gradient[j])}));
            if (ge > 1e-9)
                throw std::runtime_error("Adjoint rows do not rebuild posterior gradient");
            std::array<double, 2> fd = {0, 0};
            std::mt19937_64 rng(941201 + f.ensemble * 4096 + f.draw);
            std::normal_distribution<double> normal;
            for (unsigned direction = 0; direction < 4; ++direction) {
                GN::Field v(64);
                double norm = 0;
                for (double & x : v) {
                    x = normal(rng);
                    norm += x * x;
                }
                for (double & x : v)
                    x /= std::sqrt(norm);
                for (size_t hi = 0; hi < 2; ++hi) {
                    const double h = hi ? 1e-5 : 1e-4;
                    auto a = f.white, b = f.white;
                    for (size_t j = 0; j < 64; ++j) {
                        a[j] += h * v[j];
                        b[j] -= h * v[j];
                    }
                    const auto plus = model.predict(a, f.parameters, work),
                               minus = model.predict(b, f.parameters, work);
                    for (size_t j = 0; j < prediction.size(); ++j) {
                        const auto delta =
                            (plus[j] - minus[j]) / (2 * h) * std::sqrt(target.likelihood().weights()[j] / variance);
                        for (size_t part = 0; part < 2; ++part) {
                            const double actual = part ? delta.imag() : delta.real(),
                                         expected = GN::Metric::dot(jacobian[2 * j + part], v);
                            fd[hi] = std::max(fd[hi],
                                              std::abs(actual - expected) /
                                                  std::max({1., std::abs(actual), std::abs(expected)}));
                        }
                    }
                }
            }
            if (fd[0] > 1e-5 || fd[1] > 1e-5)
                throw std::runtime_error("Independent directional Jacobian reference failed");
            checks << f.ensemble << ',' << f.draw << ',' << pe << ',' << qe << ',' << ge << ',' << fd[0] << ',' << fd[1]
                   << '\n';
        }
        rows.close();
        checks.close();
        if (!rows || !checks)
            throw std::runtime_error("Cannot save Jacobian controls");
        std::cout << "PASS: 12 original states, exact-gradient reconstruction and 96 directional finite differences\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
