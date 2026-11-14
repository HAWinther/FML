#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/FieldLevel/Frontend.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using Model = FF::PeriodicPMModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
using Posterior = FF::PeriodicPMPosteriorT<FML::FIELDLEVEL::FRONTEND::Grid>;
static void require(bool valid, const char * text) {
    int bad = !valid; FML::MaxOverTasks(&bad); if (bad) throw std::runtime_error(text);
}
static std::ofstream output(const std::string & name) {
    std::ofstream file;
    FML::COMMUNICATION::collective_try([&] {
        if (FML::ThisTask == 0) { file.open(name); if (!file) throw std::runtime_error("Cannot open output " + name); file << std::setprecision(17); }
    }, "PM HMC output");
    return file;
}
struct Problem {
    Model model;
    Model::Workspace work;
    std::vector<double> parameters;
    std::array<Field, 2> basis;
    FF::Modes data;
    Posterior posterior;
    static FF::Configuration configuration() {
        FF::Configuration c; c.mesh = 4; c.density_mesh = 8; c.boxsize = 1;
        c.initial_cutoff = c.bias_cutoff = 7.; c.likelihood_cutoff = 15.; c.evolution = FF::Evolution::Linear; return c;
    }
    static FF::Spectrum spectrum() {
        return {{{"log_amplitude", 0., 0., .5, -1., 1.}}, [](double, const auto & p) { return FF::Power{.02 * std::exp(p[0]), {1.}}; }};
    }
    static FF::PMSettings settings() { FF::PMSettings s; s.scale_factors = {.2,.3,.4,.5,.6}; return s; }
    static FF::Parameter noise() { const double x = std::log(.001); return {"log_noise_power", x, x, .5, -14., -3.}; }
    Field field(const std::vector<double> & z) const {
        Field white(model.cells());
        for (size_t q = 0; q < white.size(); ++q) white[q] = z[0] * basis[0][q] + z[1] * basis[1][q];
        return white;
    }
    static double inner(const Field & a, const Field & b) { double s = 0; for (size_t q = 0; q < a.size(); ++q) s += a[q] * b[q]; return s; }
    explicit Problem() : model(configuration(), spectrum(), settings(), noise(), {3}), work(model.workspace()),
                         parameters(model.initial_parameters()), data(make_data()), posterior(model, data) {}
    FF::Modes make_data() {
        for (auto & b : basis) b.resize(model.cells());
        for (size_t q = 0; q < model.cells(); ++q) {
            const double phase = 2 * M_PI * model.layout().global_coordinate(q)[0] / 4.;
            basis[0][q] = std::sqrt(2. / model.layout().global_cells) * std::cos(phase);
            basis[1][q] = std::sqrt(2. / model.layout().global_cells) * std::sin(phase);
        }
        require(std::abs(model.sum(inner(basis[0], basis[0])) - 1) < 1e-13 &&
                std::abs(model.sum(inner(basis[1], basis[1])) - 1) < 1e-13 &&
                std::abs(model.sum(inner(basis[0], basis[1]))) < 1e-13, "Conditional basis is not orthonormal");
        return model.mock(field({1.4,-.9}), parameters, 92, work);
    }
    double evaluate(const std::vector<double> & z, std::vector<double> & gradient) {
        Field g; std::vector<double> gp;
        const double value = posterior.evaluate(field(z), parameters, work, g, gp);
        gradient = {model.sum(inner(g, basis[0])), model.sum(inner(g, basis[1]))}; return value;
    }
    double value(const std::vector<double> & z) {
        const auto white = field(z);
        const auto & predicted = model.predict(white, parameters, work);
        FF::Modes g; double noise_gradient;
        double value = posterior.likelihood().evaluate(predicted, parameters.back(), g, noise_gradient);
        value = model.sum(value + .5 * inner(white, white));
        for (size_t j = 0; j < parameters.size(); ++j) value += model.parameters()[j].penalty(parameters[j]);
        return value;
    }
};
static void reference(Problem & p, const std::string & prefix, int n, double extent) {
    require(n >= 9 && n <= 201 && n % 2 == 1 && extent >= 4 && extent <= 9, "Reference needs odd grid size 9..201");
    auto file = output(prefix + ".grid.csv"); if (FML::ThisTask == 0) file << "z0,z1,potential\n";
    for (int i = 0; i < n; ++i) {
        const double x = -extent + 2 * extent * i / (n - 1);
        for (int j = 0; j < n; ++j) {
            const double y = -extent + 2 * extent * j / (n - 1), u = p.value({x,y});
            if (FML::ThisTask == 0) file << x << ',' << y << ',' << u << '\n';
        }
        if (FML::ThisTask == 0 && i % 20 == 0) std::cout << "REFERENCE_ROW " << i << '/' << n << std::endl;
    }
    for (const auto & z : {std::vector<double>{1.4,-.9}, {-.3,.7}, {2.,-1.2}}) {
        std::vector<double> gradient; const double full = p.evaluate(z, gradient), direct = p.value(z);
        require(std::abs(full - direct) < 1e-10, "Quadrature and full posterior values differ");
    }
}
static double leapfrog(Problem & p, std::vector<double> & z, std::vector<double> & momentum, double eps, int steps) {
    std::vector<double> g; double u = p.evaluate(z, g);
    for (int i = 0; i < steps; ++i) {
        for (int d = 0; d < 2; ++d) { momentum[d] -= .5 * eps * g[d]; z[d] += eps * momentum[d]; }
        u = p.evaluate(z, g);
        for (int d = 0; d < 2; ++d) momentum[d] -= .5 * eps * g[d];
    }
    return u;
}
static void sweep(Problem & p, const std::string & prefix) {
    auto file = output(prefix + ".energy.csv"); if (FML::ThisTask == 0) file << "epsilon,probe,delta_h,reversal_error\n";
    const std::vector<std::vector<double>> starts{{1.51,-1.03},{1.03,-.53},{.53,.37},{-.83,.73}};
    for (int count : {2,4,8,16,32}) for (size_t j = 0; j < starts.size(); ++j) {
        const double eps = .8 / count;
        auto z = starts[j]; std::vector<double> momentum{.7 + .13*j, -.4 + .21*j}, initial = momentum;
        const double old_h = p.value(z) + .5 * (momentum[0]*momentum[0] + momentum[1]*momentum[1]);
        const double u = leapfrog(p, z, momentum, eps, count);
        const double delta = u + .5*(momentum[0]*momentum[0] + momentum[1]*momentum[1]) - old_h;
        for (auto & x : momentum) x = -x;
        leapfrog(p, z, momentum, eps, count);
        double error = 0; for (int d = 0; d < 2; ++d) error = std::max({error, std::abs(z[d]-starts[j][d]), std::abs(momentum[d]+initial[d])});
        if (FML::ThisTask == 0) file << eps << ',' << j << ',' << delta << ',' << error << '\n';
        require(std::isfinite(delta) && error < 1e-8, "Leapfrog probe nonfinite or not reversible");
    }
}
static void chain(Problem & p, const std::string & prefix, int id, double eps, int leap, int warm, int draws, bool adapt) {
    const std::vector<std::vector<double>> starts{{.3,-.3},{2.,-1.5},{1.8,.5},{-.5,-1.8}};
    FML::MCMC::HMCOptions o; o.warmup = warm; o.samples = draws; o.n_leapfrog = leap;
    o.jitter = true; o.step_size = eps; o.adapt_step_size = adapt; o.adapt_mass = false;
    o.target_acceptance = .85; o.seed = 1000 + id; o.divergence_threshold = 10.;
    FML::MCMC::HMCMass mass; mass.diagonal = {1.,1.};
    auto result = FML::MCMC::run_hmc([&](const auto & z, auto & g) { return p.evaluate(z,g); }, starts[id % 4], mass, o);
    require(result.recorded.size() == size_t(draws) && result.completed_samples == draws, "Conditional HMC recording incomplete");
    auto file = output(prefix + ".chain.csv"), meta = output(prefix + ".meta.csv");
    if (FML::ThisTask == 0) {
        file << "draw,z0,z1,potential\n";
        for (size_t i = 0; i < result.recorded.size(); ++i) file << i << ',' << result.recorded[i][0] << ',' << result.recorded[i][1] << ',' << result.potential[i] << '\n';
        meta << "acceptance,divergences,epsilon,warmup,draws,leapfrog,adapt,seed\n" << result.acceptance << ',' << result.divergences << ',' << result.step_size << ',' << warm << ',' << draws << ',' << leap << ',' << adapt << ',' << o.seed << '\n';
    }
}
static void full(Problem & p, const std::string & prefix, bool joint, int draws) {
    auto start = p.model.white_noise(95); for (auto & x : start) x *= .05;
    const auto signal = p.field({1.4,-.9}); for (size_t q = 0; q < start.size(); ++q) start[q] += signal[q];
    FML::MCMC::HMCOptions o; o.warmup = 16; o.samples = draws; o.n_leapfrog = 4;
    o.step_size = .06; o.adapt_step_size = o.adapt_mass = false; o.seed = 700 + joint; o.divergence_threshold = 10.;
    std::vector<std::vector<double>> trace;
    const auto result = FF::sample(p.posterior, start, p.parameters, o, joint,
        [&](const Field & white, const auto & parameters, size_t) {
            double dc = 0; for (double x : white) dc += x;
            trace.push_back({p.model.sum(Problem::inner(white,p.basis[0])), p.model.sum(Problem::inner(white,p.basis[1])),
                p.model.sum(dc) / std::sqrt(double(p.model.layout().global_cells)), p.model.sum(Problem::inner(white,white)), parameters[0],parameters[1]});
        });
    require(trace.size() == size_t(draws) && result.recorded.size() == size_t(draws), "Full HMC sink lost draws/rejections");
    bool finite = std::isfinite(result.acceptance); for (double u : result.potential) finite &= std::isfinite(u);
    require(finite, "Full HMC returned nonfinite summary");
    auto file = output(prefix + ".full.csv"), meta = output(prefix + ".meta.csv"), last = output(prefix + ".snapshot");
    // Global canonical final white field: test/validation output only.
    Field final(p.model.layout().global_cells,0.);
    for (size_t q = 0; q < p.model.cells(); ++q) final[p.model.layout().global_cell_id(q)] = result.last[q];
    FML::SumArrayOverTasks(final.data(),final.size());
    if (FML::ThisTask == 0) {
        file << "draw,z0,z1,dc,white_norm,log_amplitude,log_noise,potential\n";
        for (size_t i = 0; i < trace.size(); ++i) { file << i; for (double x : trace[i]) file << ',' << x; file << ',' << result.potential[i] << '\n'; }
        meta << "acceptance,divergences,epsilon,warmup,draws,joint\n" << result.acceptance << ',' << result.divergences << ',' << result.step_size << ',' << o.warmup << ',' << draws << ',' << joint << '\n';
        for (double x : final) last << x << '\n';
        for (size_t j = p.model.cells(); j < result.last.size(); ++j) last << result.last[j] << '\n';
    }
}
int main(int argc, char ** argv) {
    try {
        FML::FIELDLEVEL::FRONTEND::initialize(4,8);
        std::string mode, prefix; int n=0,id=0,leap=0,warm=0,draws=0,adapt=0; double eps=0;
        FML::COMMUNICATION::collective_try([&] {
            if (argc < 3) throw std::invalid_argument("pm_hmc reference|sweep|chain|full PREFIX ...");
            mode=argv[1];prefix=argv[2];
            auto integer=[](const char * s) { size_t used; int v=std::stoi(s,&used); if (used!=std::string(s).size()) throw std::invalid_argument("Invalid integer"); return v; };
            if (mode=="reference" && (argc==4 || argc==5)) { n=integer(argv[3]);eps=6.;if(argc==5){size_t used;eps=std::stod(argv[4],&used);if(used!=std::string(argv[4]).size())throw std::invalid_argument("Invalid extent");} }
            else if(mode=="sweep" && argc==3) {}
            else if(mode=="chain" && argc==9) { id=integer(argv[3]);size_t used;eps=std::stod(argv[4],&used);if(used!=std::string(argv[4]).size())throw std::invalid_argument("Invalid step");leap=integer(argv[5]);warm=integer(argv[6]);draws=integer(argv[7]);adapt=integer(argv[8]); }
            else if(mode=="full" && argc==5) {id=integer(argv[3]);draws=integer(argv[4]);}
            else throw std::invalid_argument("Invalid mode/argument count");
        },"PM HMC input");
        Problem p;
        const int code=mode=="reference"?0:mode=="sweep"?1:mode=="chain"?2:3;
        p.model.controls({double(code),double(n),double(id),eps,double(leap),double(warm),double(draws),double(adapt)});
        require(id>=0 && id<100 && warm>=0 && warm<=4096 && draws>=0 && draws<=4096 &&
                (code!=2 || (eps>0 && leap>0 && leap<=128 && draws>0 && (adapt==0||adapt==1))) &&
                (code!=3 || (id<=1 && draws>0)), "Invalid bounded HMC controls");
        if(code==0) reference(p,prefix,n,eps); else if(code==1)sweep(p,prefix); else if(code==2)chain(p,prefix,id,eps,leap,warm,draws,adapt);else full(p,prefix,id,draws);
        return 0;
    } catch(const std::exception & e) { if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1; }
}
