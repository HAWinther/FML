// Portable CIC local derivatives and separate continuity/one-sided knot tests.
#include <FML/FieldLevel/FieldLevelBAO.h>
#include <FML/tests/TestHelpers.h>
#include "GradientConvergence.h"
#include <iomanip>
using namespace FML::FIELDLEVEL;
using FML::TESTS::check_local;
static Field deterministic(size_t n, double scale, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    Field f(n);
    for (auto & x : f) {
        const double u = (double(rng() >> 11) + .5) / 9007199254740992.;
        const double v = (double(rng() >> 11) + .5) / 9007199254740992.;
        x = scale * std::sqrt(-2 * std::log(u)) * std::cos(2 * M_PI * v);
    }
    return f;
}
static void local_cic(Model model, double centering) {
    Settings s;
    s.nmesh = 8;
    s.boxsize = 200;
    s.kmax = .09;
    s.model = model;
    s.assignment = Assignment::CIC;
    s.centering = centering;
    s.b2 = .2;
    s.bG2 = -.1;
    s.growth_rate = model == Model::LPT ? .5 : 0;
    FieldLevelBAO fl(s, BAOSpectrum::eisenstein_hu(200));
    const auto x = deterministic(fl.ncells(), .01, 31), v = deterministic(fl.ncells(), .01, 81);
    auto truth = deterministic(fl.ncells(), .02, 93);
    fl.set_data(fl.predict(truth, .97));
    auto psi = fl.displacement(x, 1.03);
    double margin = 1;
    for (size_t p = 0; p < x.size(); ++p)
        for (int d = 0; d < 3; ++d) {
            const double coordinate = .5 + 2 * s.nmesh * psi[d][p];
            const double fraction = coordinate - std::floor(coordinate);
            margin = std::min({margin, fraction, 1 - fraction});
        }
    check_local(margin > .4, "CIC fixture stays away from grid knots");
    Field g;
    double gb;
    fl.potential_and_gradient(x, 1.03, g, gb);
    double dot = 0;
    for (size_t p = 0; p < x.size(); ++p)
        dot += g[p] * v[p];
    const auto a = TESTING::derivative_sweep(
        [&](double h) {
            auto y = x;
            for (size_t p = 0; p < x.size(); ++p)
                y[p] += h * v[p];
            return fl.potential(y, 1.03);
        },
        dot,
        .1,
        "CIC field model=" + std::to_string(int(model)) + " c=" + std::to_string(centering));
    const auto b = TESTING::derivative_sweep([&](double h) { return fl.potential(x, 1.03 + h); }, gb, .01, "CIC beta");
    check_local(a.resolved(), "CIC local field derivative has a resolved step interval");
    check_local(b.resolved(), "CIC local beta derivative has a resolved step interval");
}
static void crossing() {
    Settings s;
    s.nmesh = 8;
    s.nmesh_density = 16;
    s.boxsize = 200;
    s.kmax = .09;
    s.model = Model::LPT1;
    s.assignment = Assignment::CIC;
    s.centering = 0;
    s.bias = 1;
    FieldLevelBAO fl(s, BAOSpectrum::eisenstein_hu(200));
    Field delta(fl.ncells());
    // At q_x=1/4, Psi_x=-A/(2 pi)=-half a density cell: a known CIC knot.
    for (size_t p = 0; p < delta.size(); ++p)
        delta[p] = M_PI / s.nmesh_density * std::cos(2 * M_PI * double(p / 64) / s.nmesh);
    std::vector<double> inverse(fl.modes().size());
    for (size_t i = 0; i < inverse.size(); ++i)
        inverse[i] = 1 / fl.effective_transfer(fl.modes()[i], 1.);
    const auto x = fl.filter(delta, inverse);
    auto truth = x;
    for (auto & z : truth)
        z *= .7;
    fl.set_data(fl.predict(truth, 1.));
    auto value = [&](double h) {
        auto y = x;
        for (auto & z : y)
            z *= 1 + h;
        return fl.potential(y, 1.);
    };
    const double u = value(0);
    std::vector<double> left, right, jump;
    for (double h : {1e-3, 2.5e-4, 6.25e-5, 1.5625e-5}) {
        left.push_back((u - value(-h)) / h);
        right.push_back((value(h) - u) / h);
        jump.push_back(std::max(std::abs(value(h) - u), std::abs(value(-h) - u)));
        std::cout << "CIC_KNOT h=" << h << " left=" << left.back() << " right=" << right.back()
                  << " continuity=" << jump.back() << '\n';
    }
    check_local(jump.back() < .03 * jump.front(), "CIC potential is continuous through a cell crossing");
    const double scale = std::max({1., std::abs(left.back()), std::abs(right.back())});
    check_local(std::abs(left[3] - left[2]) < 1e-3 * scale && std::abs(right[3] - right[2]) < 1e-3 * scale,
                "CIC one-sided derivatives converge at the knot");
    check_local(std::abs(left.back() - right.back()) > 1e-3 * scale,
                "CIC knot has distinct one-sided derivatives; no smooth-gradient assertion");
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (FML::NTasks != 1)
        return 1;
    std::cout << std::setprecision(14);
    for (auto model : {Model::LPT1, Model::LPT})
        for (double c : {0., -1.})
            local_cic(model, c);
    crossing();
    auto f = [](double h) { return std::exp(h); };
    check_local(TESTING::derivative_sweep(f, 1., .1, "sweep positive control").resolved(),
                "step-sweep accepts correct derivative");
    check_local(!TESTING::derivative_sweep(f, 1.001, .1, "sweep negative control").resolved(),
                "step-sweep rejects a biased derivative");
    return FML::TESTS::summary("BAO portable gradients");
}
