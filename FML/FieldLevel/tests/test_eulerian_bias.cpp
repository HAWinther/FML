#include <FML/FieldLevel/EulerianBias.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <limits>
#include <string>
namespace E = FML::FIELDLEVEL::EULERIAN;
using FML::TESTS::check;
using Field = std::vector<double>;
static double dot(const Field & a, const Field & b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
        s += a[i] * b[i];
    FML::SumOverTasks(&s);
    return s;
}
template <class Grid>
void exercise(const FML::Domain & domain, E::BandGeometry geometry, const std::string & label, double boxsize = 100.) {
    auto grid = FML::GRID::make_grid<Grid>(16, 0, 0, domain);
    E::QuadraticBias<Grid> model(grid, boxsize, {2.5 * 2 * M_PI / boxsize, geometry});
    auto w = model.workspace(grid);
    using Columns = typename E::QuadraticBias<Grid>::Columns;
    Field matter, direction;
    Columns expected, cotangent;
    const std::array<std::array<int, 3>, 3> waves = {{{1, 0, 0}, {0, 1, 1}, {2, 2, 2}}};
    const bool cubic = geometry == E::BandGeometry::Cube;
    double variance = (.7 * .7 + .4 * .4 + (cubic ? .2 * .2 : 0)) / 2;
    for (auto i : grid.get_real_range()) {
        auto c = grid.get_coord_from_index(i);
        std::array<double, 3> x;
        for (int d = 0; d < 3; ++d)
            x[d] = (c[d] + grid.get_local_range(d)[0]) * 2 * M_PI / 16;
        const double f[3] = {.7 * std::cos(x[0]), .4 * std::sin(x[1] + x[2]), .2 * std::cos(2 * (x[0] + x[1] + x[2]))};
        matter.push_back(3.1 + f[0] + f[1] + f[2] + .13 * std::cos(5 * x[0]) + .06 * std::cos(8 * x[0]));
        direction.push_back(.7 * std::cos(x[0]) + .6 * std::sin(x[1] + x[2]) + .2 + std::sin(x[0] + 2 * x[2]) +
                            .3 * std::cos(3 * x[1]) + .1 * std::sin(x[0]));
        double delta = f[0] + f[1] + (cubic ? f[2] : 0), lap = 0, tidal2 = 0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                double tidal = 0;
                for (int m = 0; m < (cubic ? 3 : 2); ++m) {
                    double n2 = 0;
                    for (int v : waves[m])
                        n2 += v * v;
                    tidal += (double(waves[m][a] * waves[m][b]) / n2 - (a == b ? 1. / 3 : 0.)) * f[m];
                    if (a == 0 && b == 0)
                        lap -= n2 * f[m];
                }
                tidal2 += tidal * tidal;
            }
        const double value[4] = {delta,
                                 delta * delta - variance,
                                 tidal2 - 2 * variance / 3,
                                 lap * std::pow(2 * M_PI / boxsize, 2)};
        for (int a = 0; a < 4; ++a) {
            expected[a].push_back(value[a]);
            cotangent[a].push_back(.7 * std::cos(x[0]) + .5 * std::cos(2 * x[0]) + .3 * std::cos(2 * (x[1] + x[2])) +
                                   2 + std::sin((a + 1) * x[0] + x[1]) + .3 * std::cos(x[1] + 2 * x[2]));
        }
    }
    check(FML::TESTS::throws([&] { model.pullback(cotangent, w); }), label + " reverse requires tape");
    const auto columns = model.evaluate(matter, w);
    for (int a = 0; a < 4; ++a) {
        double err = 0, sum = 0;
        for (size_t q = 0; q < matter.size(); ++q) {
            err = std::max(err, std::abs(columns[a][q] - expected[a][q]));
            sum += columns[a][q];
        }
        FML::MaxOverTasks(&err);
        FML::SumOverTasks(&sum);
        check(err < 2e-13 && std::abs(sum) < 2e-10,
              label + " analytic two/three-wave column " + std::to_string(a) + " and zero mean");
    }
    std::array<double, 4> adjoints;
    for (int a = 0; a < 4; ++a) {
        Columns isolated;
        for (auto & c : isolated)
            c.assign(matter.size(), 0.);
        isolated[a] = cotangent[a];
        adjoints[a] = dot(model.pullback(isolated, w), direction);
    }
    for (double h : {1e-3, 1e-4}) {
        auto plus = matter, minus = matter;
        for (size_t q = 0; q < matter.size(); ++q) {
            plus[q] += h * direction[q];
            minus[q] -= h * direction[q];
        }
        const auto cp = model.evaluate(plus, w);
        const auto cm = model.evaluate(minus, w);
        for (int a = 0; a < 4; ++a) {
            Field d(matter.size());
            for (size_t q = 0; q < d.size(); ++q)
                d[q] = (cp[a][q] - cm[a][q]) / (2 * h);
            const double difference = dot(d, cotangent[a]);
            check(std::abs(adjoints[a]) > .01 && FML::TESTS::near(difference, adjoints[a], 2e-9),
                  label + " column " + std::to_string(a) + " nonzero directional derivative h=" + std::to_string(h));
        }
    }
    model.evaluate(matter, w);
    Columns constants;
    for (int a = 0; a < 4; ++a)
        constants[a].assign(matter.size(), a + 1.);
    auto zero = model.pullback(constants, w);
    check(dot(zero, zero) < 1e-24, label + " global mean-subtraction transpose");
    auto foreign = model.workspace(grid);
    E::QuadraticBias<Grid> another(grid, boxsize, {2.5 * 2 * M_PI / boxsize, geometry});
    check(FML::TESTS::throws([&] { another.evaluate(matter, foreign); }), label + " foreign tape rejected");
    auto moved = std::move(foreign);
    check(FML::TESTS::throws([&] { model.evaluate(matter, foreign); }), label + " moved-from tape rejected");
    model.evaluate(matter, moved);
    auto malformed = cotangent;
    if (FML::ThisTask == FML::NTasks - 1)
        malformed[2].pop_back();
    check(FML::TESTS::throws([&] { model.pullback(malformed, moved); }),
          label + " rank-local malformed cotangent rejected");
    auto bad = matter;
    if (FML::ThisTask == FML::NTasks - 1)
        bad.pop_back();
    check(FML::TESTS::throws([&] { model.evaluate(bad, w); }), label + " rank-local shape rejected collectively");
    check(FML::TESTS::throws([&] { model.pullback(cotangent, w); }), label + " failed evaluate invalidates tape");
    bad = matter;
    if (FML::ThisTask == FML::NTasks - 1)
        bad[0] = std::numeric_limits<double>::quiet_NaN();
    check(FML::TESTS::throws([&] { model.evaluate(bad, w); }), label + " rank-local nonfinite rejected collectively");
    check(FML::TESTS::throws([&] {
              E::QuadraticBias<Grid> invalid(grid, boxsize, {4.01 * 2 * M_PI / boxsize, geometry});
          }),
          label + " unresolved quadratic products rejected");
#ifdef USE_MPI
    if (FML::NTasks > 1)
        check(FML::TESTS::throws([&] {
                  E::QuadraticBias<Grid> invalid(grid,
                                                 boxsize,
                                                 {(FML::ThisTask ? 2.4 : 2.5) * 2 * M_PI / boxsize, geometry});
              }),
              label + " inconsistent bands rejected collectively");
#endif
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    for (auto geometry : {E::BandGeometry::Sphere, E::BandGeometry::Cube}) {
        E::SharpKBand band{2 * M_PI * 2 / 100, geometry};
        check(!band.contains({2, 0, 0}, 100, 16) && band.contains({1, 0, 0}, 100, 16), "strict sharp band boundary");
        check(!band.contains({0, 0, 0}, 100, 16) && !band.contains({8, 0, 0}, 100, 16),
              "DC and Nyquist planes excluded");
        check(band.contains({1, 1, 1}, 100, 16), "interior oblique mode retained");
        band.cutoff = 2 * M_PI * 2.5 / 100;
        check(band.contains({2, 2, 2}, 100, 16) == (geometry == E::BandGeometry::Cube),
              "cube and sphere distinguish corner modes");
    }
    for (auto geometry : {E::BandGeometry::Sphere, E::BandGeometry::Cube}) {
        std::string name = geometry == E::BandGeometry::Cube ? "cube" : "sphere";
        exercise<FML::GRID::FFTWGrid<3>>(FML::Domain::uniform_slab(FML::NTasks), geometry, "slab " + name);
#ifdef USE_MPI
        for (int px = 1; px <= FML::NTasks; ++px) {
            if (FML::NTasks % px)
                continue;
            int py = FML::NTasks / px;
            std::vector<double> ex(px + 1), ey(py + 1);
            for (int p = 0; p <= px; ++p)
                ex[p] = double(p) / px;
            for (int p = 0; p <= py; ++p)
                ey[p] = double(p) / py;
            exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(ex, ey),
                                               geometry,
                                               "pencil " + std::to_string(px) + "x" + std::to_string(py) + " " + name);
        }
#endif
    }
    return FML::TESTS::summary("Eulerian bias");
}
