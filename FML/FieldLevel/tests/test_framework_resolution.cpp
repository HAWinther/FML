// Compare the same low-band initial density field on different particle/FFT meshes.
#include <FML/FieldLevel/Framework.h>
#include <FML/tests/TestHelpers.h>
#include <map>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using Key = std::array<int, 3>;
static std::map<Key, FML::FIELDLEVEL::Complex> prediction(int mesh) {
    FF::Configuration c;
    c.mesh = mesh;
    c.density_mesh = 2 * mesh;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    FF::Model model(c, FF::Spectrum::fixed([](double) { return 200.; }));
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[1] = .2;
    p[2] = -.1;
    p[3] = 10;
    FML::FIELDLEVEL::Field white(model.cells());
    const double T = std::sqrt(200. * model.cells() / model.volume());
    for (size_t q = 0; q < white.size(); ++q) {
        const int x = int(q / (mesh * mesh)), y = int(q / mesh % mesh), z = int(q % mesh);
        const double k = 2 * M_PI / mesh;
        white[q] =
            (.08 * std::cos(k * x) + .06 * std::cos(k * y) + .025 * std::cos(k * (x + y)) + .02 * std::cos(k * z)) / T;
    }
    const auto values = model.predict(white, p, w);
    std::map<Key, FML::FIELDLEVEL::Complex> result;
    for (size_t i = 0; i < values.size(); ++i) {
        const auto & m = model.modes()[i];
        result[{m.n[0], m.n[1], m.n[2]}] = values[i];
    }
    return result;
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    if (FML::NTasks != 1)
        return 1;
    const auto coarse = prediction(8), fine = prediction(16);
    double error = 0, norm = 0;
    for (const auto & item : coarse) {
        error += std::norm(item.second - fine.at(item.first));
        norm += std::norm(fine.at(item.first));
    }
    const double relative = std::sqrt(error / norm);
    std::cout << std::setprecision(14) << "MESH_REFINEMENT relative tracer-field change " << relative << '\n';
    FML::TESTS::check_local(
        relative < 1e-3,
        "fixed physical initial field/tracer prediction converges across meshes with distinct cutoffs and RSD");
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    FF::Spectrum spectrum;
    spectrum.parameters = {{"log_amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    spectrum.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    FF::Model model(c, spectrum);
    auto work = model.workspace();
    auto params = model.initial_parameters();
    auto white = model.white_noise(91);
    auto data = model.mock(white, params, 92, work);
    FF::Posterior posterior(model, data);
    FML::FIELDLEVEL::Field gradient, scratch;
    std::vector<double> gp, tmp;
    const double U = posterior.evaluate(white, params, work, gradient, gp);
    for (size_t j = 0; j < 2; ++j) {
        auto plus = params, minus = params;
        plus[j] += 1e-5;
        minus[j] -= 1e-5;
        const double numeric = (posterior.evaluate(white, plus, work, scratch, tmp) -
                                posterior.evaluate(white, minus, work, scratch, tmp)) /
                               2e-5;
        FML::TESTS::check_local(std::abs(gp[j] - numeric) / std::max({1., std::abs(gp[j]), std::abs(numeric)}) < 2e-6,
                                "non-BAO amplitude/tilt spectrum derivatives use the generic prior interface");
    }
    FF::GaussianFieldLikelihood likelihood(model.modes(), c.likelihood_cutoff, model.volume(), data);
    FF::BasicPosterior<FF::Model> composed(model, std::move(likelihood));
    const double explicitU = composed.evaluate(white, params, work, scratch, tmp);
    FML::TESTS::check_local(std::abs(explicitU - U) < 1e-12,
                            "explicit likelihood composition matches the convenience posterior");
    auto bc = c;
    bc.initial_cutoff = bc.bias_cutoff = bc.likelihood_cutoff = .11;
    auto bao_spectrum = FF::Spectrum::bao(FML::FIELDLEVEL::BAOSpectrum::eisenstein_hu(200));
    bao_spectrum.parameters[0].prior = FF::Prior::Uniform;
    FF::Model bao(bc, bao_spectrum);
    auto bw = bao.workspace();
    auto bp = bao.initial_parameters();
    bp[0] = 1.02;
    const auto bs = bao.white_noise(81), bt = bao.white_noise(82);
    const auto bd = bao.mock(bt, bp, 83, bw);
    FF::Posterior bpost(bao, bd);
    FML::FIELDLEVEL::Settings ls;
    ls.nmesh = bc.mesh;
    ls.nmesh_density = bc.density_mesh;
    ls.boxsize = bc.boxsize;
    ls.kmax = .11;
    ls.centering = 0;
    ls.model = FML::FIELDLEVEL::Model::LPT;
    ls.lpt_order = 2;
    ls.growth_rate = bc.growth_rate;
    FML::FIELDLEVEL::FieldLevelBAO legacy(ls, FML::FIELDLEVEL::BAOSpectrum::eisenstein_hu(200));
    legacy.set_data(bd);
    const double ref = legacy.potential(bs, bp[0]) - legacy.potential(bt, 1.);
    const double first = bpost.evaluate(bs, bp, bw, scratch, tmp);
    auto priorp = bp;
    priorp[0] = 1.;
    const double second = bpost.evaluate(bt, priorp, bw, scratch, tmp);
    FML::TESTS::check_local(
        std::abs(first - second - ref) < 1e-9,
        "bounded-uniform BAO prior reproduces the original posterior up to parameter-independent constants");
    auto invalid_prior = FF::Parameter{"b1", 1, 0, 1, -HUGE_VAL, HUGE_VAL, FF::Prior::Uniform};
    FML::TESTS::check_local(FML::TESTS::throws([&] { FF::Model invalid(c, spectrum, {}, {invalid_prior}); }),
                            "improper unbounded uniform priors are rejected");
    auto duplicate = FF::Parameter{"b1", 1, 0, 1};
    FML::TESTS::check_local(FML::TESTS::throws([&] { FF::Model invalid(c, spectrum, {}, {duplicate, duplicate}); }),
                            "duplicate named prior overrides are rejected");
    return FML::TESTS::summary("field framework resolution");
}
