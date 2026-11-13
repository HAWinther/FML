#include <FML/FieldLevel/DistributedLPT.h>
#include <FML/tests/TestHelpers.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Complex;
using FML::FIELDLEVEL::Field;
using Reconstruction = FF::Model::Reconstruction;
using FML::TESTS::check_local;
template <class V>
static double error(const V & a, const V & b) {
    if (a.size() != b.size())
        return HUGE_VAL;
    double result = 0;
    for (size_t q = 0; q < a.size(); ++q)
        result = std::max(result, std::abs(a[q] - b[q]));
    return result;
}
static double physical_error(const Reconstruction & a, const Reconstruction & b) {
    double e = std::max(error(a.initial_density, b.initial_density), error(a.matter_density, b.matter_density));
    for (int c = 0; c < 3; ++c)
        e = std::max({e, error(a.displacement[c], b.displacement[c]), error(a.velocity[c], b.velocity[c])});
    return e;
}
static double norm(const Field & x) {
    double s = 0;
    for (double v : x)
        s += v * v;
    return s;
}
static double relative(double a, double b) { return std::abs(a - b) / std::max({1., std::abs(a), std::abs(b)}); }
static std::string label(FF::Evolution e) {
    return e == FF::Evolution::Linear ? "linear" : e == FF::Evolution::Zeldovich ? "1lpt" : "2lpt";
}
template <class Forward>
static void run_case(FF::Configuration c, const FF::Spectrum & spectrum, int argc, char ** argv) {
    const auto name = label(c.evolution);
    auto lc = c;
    lc.evolution = FF::Evolution::Linear;
    FF::DistributedLinearModel metadata(lc, spectrum);
    FF::FourierResponse response;
    for (const auto & m : metadata.modes())
        response.multiplier.push_back(m.n[0] == 0 ? 0 : .9 + .02 * m.n[0] * m.n[0]);
    Forward model(c, spectrum, response);
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[2] = 1.4;
    if (c.evolution != FF::Evolution::Linear) {
        p[3] = .2;
        p[4] = -.15;
        p[5] = 12;
    }
    const auto s = model.white_noise(41);
    const auto data = model.mock(model.white_noise(42), p, 43, w);
    FF::BasicPosterior<Forward> posterior(model, data);
    Field gs, tmp;
    std::vector<double> gp, ptmp;
    const double before = posterior.evaluate(s, p, w, gs, gp);
    const auto prediction = model.predict(s, p, w);
    constexpr double vf = 17.3;
    const auto out = model.reconstruct(s, p, w, vf);
    bool shape = out.initial_density.size() == model.cells() && out.matter_density.size() == model.cells() &&
                 out.tracer.size() == model.modes().size();
    for (int a = 0; a < 3; ++a)
        shape &= out.displacement[a].size() == model.cells() && out.velocity[a].size() == model.cells();
    check_local(shape, name + " reconstruction owns local field slabs/modes");
    check_local(model.maximum(error(out.tracer, prediction)) < 1e-12,
                name + " reconstructed tracer matches biased/RSD/response prediction");
    check_local(FML::TESTS::throws([&] { model.pullback(p, data, w, tmp, ptmp); }),
                name + " reconstruction invalidates the old pullback cache");
    const double after = posterior.evaluate(s, p, w, tmp, ptmp);
    check_local(relative(before, after) < 1e-13 && model.maximum(error(gs, tmp)) < 1e-12 &&
                    model.maximum(error(gp, ptmp)) < 1e-10,
                name + " posterior/adjoint restored by fresh prediction after reconstruction");
    double initial_mean = 0, matter_mean = 0;
    for (double x : out.initial_density)
        initial_mean += x;
    for (double x : out.matter_density)
        matter_mean += x;
    check_local(std::abs(model.sum(initial_mean)) < 1e-10 && std::abs(model.sum(matter_mean)) < 1e-10,
                name + " initial/matter outputs are zero-mean contrasts");
    if (FML::NTasks == 1) {
        FF::Model serial(c, spectrum, response);
        auto sw = serial.workspace();
        const auto reference = serial.reconstruct(s, p, sw, vf);
        check_local(physical_error(out, reference) < 1e-11 && error(out.tracer, reference.tracer) < 1e-12,
                    name + " all distributed reconstruction products match the existing serial API");
    }
    auto biased = p;
    biased[2] = 2.2;
    biased.back() += .5;
    if (c.evolution != FF::Evolution::Linear) {
        biased[3] = -.4;
        biased[4] = .3;
        biased[5] = -20;
    }
    const auto b = model.reconstruct(s, biased, w, vf);
    check_local(model.maximum(physical_error(out, b)) < 1e-12 && model.maximum(error(out.tracer, b.tracer)) > 1e-6,
                name + " physical products are independent of tracer bias/noise");
    auto no_rsd = c;
    no_rsd.growth_rate = 0;
    Forward real_space(no_rsd, spectrum, response);
    auto rw = real_space.workspace();
    const auto real = real_space.reconstruct(s, p, rw, vf);
    check_local(model.maximum(physical_error(out, real)) < 1e-12 &&
                    model.maximum(error(out.tracer, real.tracer)) > 1e-6,
                name + " physical products are real-space while tracer retains RSD");
    FF::FourierResponse zeros;
    zeros.multiplier.assign(model.modes().size(), 0);
    Forward blinded(c, spectrum, zeros);
    auto zw = blinded.workspace();
    const auto z = blinded.reconstruct(s, p, zw, vf);
    double zmax = 0;
    for (auto x : z.tracer)
        zmax = std::max(zmax, std::abs(x));
    check_local(model.maximum(zmax) == 0 && model.maximum(physical_error(out, z)) < 1e-12,
                name + " zero observation response leaves physical outputs intact");
    const auto zero_velocity = model.reconstruct(s, p, w, 0);
    const auto scaled = model.reconstruct(s, p, w, -2 * vf);
    double velocity_error = 0;
    for (int a = 0; a < 3; ++a)
        for (size_t q = 0; q < s.size(); ++q)
            velocity_error = std::max({velocity_error,
                                       std::abs(zero_velocity.velocity[a][q]),
                                       std::abs(scaled.velocity[a][q] + 2 * out.velocity[a][q])});
    check_local(model.maximum(velocity_error) < 1e-12 &&
                    model.maximum(error(out.matter_density, scaled.matter_density)) < 1e-12,
                name + " supplied velocity conversion scales only the velocity samples");
    if (c.evolution != FF::Evolution::LPT2) {
        double e = 0;
        for (int a = 0; a < 3; ++a)
            for (size_t q = 0; q < s.size(); ++q)
                e = std::max(e, std::abs(out.velocity[a][q] - vf * out.displacement[a][q]));
        check_local(model.maximum(e) < 1e-13, name + " first-order velocity equals supplied factor times displacement");
    }
    if (c.evolution == FF::Evolution::Linear)
        check_local(error(out.initial_density, out.matter_density) == 0,
                    "linear reconstructed matter equals initial density");
    const auto uniform = model.reconstruct(Field(model.cells(), 0), p, w, vf);
    double uniform_error = 0;
    for (double x : uniform.initial_density)
        uniform_error = std::max(uniform_error, std::abs(x));
    for (double x : uniform.matter_density)
        uniform_error = std::max(uniform_error, std::abs(x));
    for (auto x : uniform.tracer)
        uniform_error = std::max(uniform_error, std::abs(x));
    for (int a = 0; a < 3; ++a) {
        for (double x : uniform.displacement[a])
            uniform_error = std::max(uniform_error, std::abs(x));
        for (double x : uniform.velocity[a])
            uniform_error = std::max(uniform_error, std::abs(x));
    }
    check_local(model.maximum(uniform_error) < 1e-12,
                name + " uniform initial state reconstructs zero contrasts/motion");
    check_local(FML::TESTS::throws([&] {
                    auto bad = s;
                    if (FML::ThisTask == 0)
                        bad[0] = std::numeric_limits<double>::quiet_NaN();
                    model.reconstruct(bad, p, w, vf);
                }),
                name + " one-rank nonfinite field fails collectively during reconstruction");
    check_local(FML::TESTS::throws([&] {
                    double bad = vf;
                    if (FML::ThisTask == 0)
                        bad = HUGE_VAL;
                    model.reconstruct(s, p, w, bad);
                }),
                name + " invalid one-rank velocity conversion fails collectively");
    if (FML::NTasks > 1)
        check_local(FML::TESTS::throws([&] {
                        const double factor = FML::ThisTask == 0 ? vf + 1 : vf;
                        model.reconstruct(s, p, w, factor);
                    }),
                    name + " inconsistent replicated velocity conversions rejected");
    if (argc == 2)
        FML::COMMUNICATION::collective_try(
            [&] {
                std::ofstream f(std::string(argv[1]) + "-" + name + ".rank" + std::to_string(FML::ThisTask) + ".txt");
                if (!f)
                    throw std::runtime_error("Cannot open reconstruction comparison output");
                f << std::setprecision(17);
                const size_t offset = size_t(model.layout().local_start) * c.mesh * c.mesh;
                for (size_t q = 0; q < model.cells(); ++q) {
                    f << "F " << offset + q << ' ' << out.initial_density[q] << ' ' << out.matter_density[q];
                    for (int a = 0; a < 3; ++a)
                        f << ' ' << out.displacement[a][q];
                    for (int a = 0; a < 3; ++a)
                        f << ' ' << out.velocity[a][q];
                    f << '\n';
                }
                for (size_t i = 0; i < out.tracer.size(); ++i) {
                    const auto & m = model.modes()[i];
                    f << "M " << m.n[0] << ' ' << m.n[1] << ' ' << m.n[2] << ' ' << out.tracer[i].real() << ' '
                      << out.tracer[i].imag() << '\n';
                }
                f.close();
                if (!f)
                    throw std::runtime_error("Cannot write reconstruction comparison output");
            },
            "Reconstruction comparison output");
    const double power = model.sum(norm(out.initial_density)), matter = model.sum(norm(out.matter_density));
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "RECON_RESULT " << name << ' ' << power << ' ' << matter << '\n';
}
// Exact two-wave displacement and velocity fields. The matter-density reference
// independently sums particles in Fourier space, then evaluates the retained band.
template <class Forward>
static void manufactured(FF::Evolution evolution) {
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .08;
    c.bias_cutoff = .07;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    c.evolution = evolution;
    auto spectrum = FF::Spectrum::fixed([](double) { return 200.; });
    Forward model(c, spectrum);
    auto w = model.workspace();
    auto p = model.initial_parameters();
    p[0] = 1;
    const double A = .5, B = .4, vf = 17.3, T = std::sqrt(200. * 512 / model.volume());
    Field white(model.cells());
    for (size_t q = 0; q < white.size(); ++q) {
        const size_t id = size_t(model.layout().local_start) * 64 + q;
        white[q] = (A * std::cos(2 * M_PI * double(id / 64) / 8) + B * std::cos(2 * M_PI * double(id % 8) / 8)) / T;
    }
    const auto out = model.reconstruct(white, p, w, vf);
    double e = 0, matter_error = 0;
    struct Mode {
        int n[3];
        double weight;
        Complex value;
    };
    std::vector<Mode> reference;
    if (evolution != FF::Evolution::Linear) {
        for (int ix = 0; ix < 8; ++ix)
            for (int iy = 0; iy < 8; ++iy)
                for (int iz = 0; iz <= 4; ++iz) {
                    const int x = ix <= 4 ? ix : ix - 8, y = iy <= 4 ? iy : iy - 8;
                    const double k = 2 * M_PI / c.boxsize * std::sqrt(double(x * x + y * y + iz * iz));
                    if (k == 0 || k >= std::max(c.initial_cutoff, c.likelihood_cutoff))
                        continue;
                    Complex density = 0;
                    for (int qx = 0; qx < 8; ++qx)
                        for (int qy = 0; qy < 8; ++qy)
                            for (int qz = 0; qz < 8; ++qz) {
                                const double ax = 2 * M_PI * qx / 8, az = 2 * M_PI * qz / 8;
                                double px = -A / (2 * M_PI) * std::sin(ax), pz = -B / (2 * M_PI) * std::sin(az);
                                if (evolution == FF::Evolution::LPT2) {
                                    px -= 3. / 7 * A * B / (4 * M_PI) * std::sin(ax) * std::cos(az);
                                    pz -= 3. / 7 * A * B / (4 * M_PI) * std::cos(ax) * std::sin(az);
                                }
                                density += std::polar(1.,
                                                      -2 * M_PI *
                                                          (x * (double(qx) / 8 + px) + y * double(qy) / 8 +
                                                           iz * (double(qz) / 8 + pz))) /
                                           512.;
                            }
                    reference.push_back({{x, y, iz}, iz == 0 ? 1. : 2., density});
                }
    }
    for (size_t q = 0; q < white.size(); ++q) {
        const size_t id = size_t(model.layout().local_start) * 64 + q;
        const int x = int(id / 64), y = int(id / 8 % 8), z = int(id % 8);
        const double ax = 2 * M_PI * x / 8, az = 2 * M_PI * z / 8, initial = A * std::cos(ax) + B * std::cos(az);
        const double px1 = -A / (2 * M_PI) * std::sin(ax), pz1 = -B / (2 * M_PI) * std::sin(az);
        const double px2 =
            evolution == FF::Evolution::LPT2 ? -3. / 7 * A * B / (4 * M_PI) * std::sin(ax) * std::cos(az) : 0;
        const double pz2 =
            evolution == FF::Evolution::LPT2 ? -3. / 7 * A * B / (4 * M_PI) * std::cos(ax) * std::sin(az) : 0;
        e = std::max({e,
                      std::abs(out.initial_density[q] - initial),
                      std::abs(out.displacement[0][q] - px1 - px2),
                      std::abs(out.displacement[1][q]),
                      std::abs(out.displacement[2][q] - pz1 - pz2),
                      std::abs(out.velocity[0][q] - vf * (px1 + 2 * px2)),
                      std::abs(out.velocity[1][q]),
                      std::abs(out.velocity[2][q] - vf * (pz1 + 2 * pz2))});
        double matter = initial;
        if (evolution != FF::Evolution::Linear) {
            matter = 0;
            for (const auto & m : reference)
                matter += m.weight *
                          std::real(m.value * std::polar(1., 2 * M_PI * (m.n[0] * x + m.n[1] * y + m.n[2] * z) / 8.));
        }
        matter_error = std::max(matter_error, std::abs(out.matter_density[q] - matter));
    }
    const double physical = model.maximum(e), density = model.maximum(matter_error);
    check_local(physical < 1e-12,
                label(evolution) + " analytic initial density and real-space displacement/velocity fields");
    check_local(density < 1e-4,
                label(evolution) +
                    " real-space matter agrees with independent particle Fourier sum (assignment tolerance)");
    if (FML::ThisTask == 0)
        std::cout << std::setprecision(17) << "RECON_ANALYTIC " << label(evolution) << ' ' << physical << ' ' << density
                  << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    FF::Configuration c;
    c.mesh = 8;
    c.density_mesh = 16;
    c.boxsize = 100;
    c.initial_cutoff = .115;
    c.bias_cutoff = .08;
    c.likelihood_cutoff = .10;
    c.growth_rate = .6;
    FF::Spectrum spectrum;
    spectrum.parameters = {{"log_amplitude", .1, 0, 1}, {"tilt", .2, 0, 1}};
    spectrum.evaluate = [](double k, const auto & p) {
        return FF::Power{200 * std::exp(p[0]) * std::pow(k / .1, p[1]), {1., std::log(k / .1)}};
    };
    c.evolution = FF::Evolution::Linear;
    run_case<FF::DistributedLinearModel>(c, spectrum, argc, argv);
    manufactured<FF::DistributedLinearModel>(c.evolution);
    c.evolution = FF::Evolution::Zeldovich;
    run_case<FF::DistributedLPTModel>(c, spectrum, argc, argv);
    manufactured<FF::DistributedLPTModel>(c.evolution);
    c.evolution = FF::Evolution::LPT2;
    run_case<FF::DistributedLPTModel>(c, spectrum, argc, argv);
    manufactured<FF::DistributedLPTModel>(c.evolution);
    return FML::TESTS::summary("distributed field reconstruction");
}
