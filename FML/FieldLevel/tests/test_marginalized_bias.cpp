// Exact analytic bias marginalization for the Eulerian model: gradients against finite differences, and
// the Gaussian identities that make the marginal exact (conditional mean minimizes the joint target,
// conditional covariance is the inverse joint Hessian in b, marginal differences equal joint differences
// plus the log-determinant change).
#include <FML/FieldLevel/MarginalizedBias.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace E = FML::FIELDLEVEL::EULERIAN;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;

template <class Grid>
void exercise(const FML::Domain & domain, const std::string & label, bool compact, bool pswf) {
    FF::EulerianLPTConfiguration c;
    c.mesh = compact ? 4 : 8;
    c.product_mesh = compact ? 8 : 0;
    c.density_mesh = 16;
    c.boxsize = 100;
    const double kf = 2 * M_PI / c.boxsize;
    c.initial_cutoff = 1.7 * kf;
    c.matter_band = {1.8 * kf, E::BandGeometry::Cube};
    c.likelihood_band = {1.6 * kf, E::BandGeometry::Cube};
    c.matter_assignment = pswf ? FF::MatterAssignment::PSWF : FF::MatterAssignment::Smooth;
    const std::array<FF::Parameter, 4> gaussian{{{"b_delta", 2, 2, 1.155},
                                                 {"b_delta2", 0, 0, 2.309},
                                                 {"b_K2", 0, 0, 2.309},
                                                 {"b_laplacian", 0, 0, 57.7}}};
    using Model = FF::DistributedEulerianLPTModelT<Grid>;
    Model model(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), gaussian, domain);
    auto w = model.workspace();
    const double noise = 1000;
    const std::vector<double> truth = {1, 1.6, .3, -.2, 7};
    const auto data = model.mock(model.white_noise(4101), truth, noise, 4102, w);
    FF::BiasMarginalizedPosterior<Grid> marginal(model, data, noise);
    const auto joint = FF::fixed_noise_posterior(model, data, noise);
    auto s = model.white_noise(77);
    for (auto & x : s)
        x *= .9;
    const std::vector<double> theta = {1.07};

    // Gradient against central differences along a random field direction and theta.
    Field gs;
    std::vector<double> gt;
    const double U = marginal.evaluate(s, theta, w, gs, gt);
    check(std::isfinite(U) && gs.size() == s.size() && gt.size() == 1, label + " finite marginal target");
    const auto direction = model.white_noise(78);
    double slope = 0;
    for (size_t q = 0; q < s.size(); ++q)
        slope += gs[q] * direction[q];
    slope = model.sum(slope) + gt[0] * .3;
    double best = HUGE_VAL;
    for (double h : {1e-3, 3e-4, 1e-4, 3e-5}) {
        auto plus = s, minus = s;
        for (size_t q = 0; q < s.size(); ++q) {
            plus[q] += h * direction[q];
            minus[q] -= h * direction[q];
        }
        Field t1;
        std::vector<double> t2;
        const double up = marginal.evaluate(plus, {theta[0] + .3 * h}, w, t1, t2);
        const double down = marginal.evaluate(minus, {theta[0] - .3 * h}, w, t1, t2);
        best = std::min(best, std::abs((up - down) / (2 * h) - slope) / std::max(1., std::abs(slope)));
    }
    check(best < 2e-6, label + " marginal gradient matches finite differences (" + FML::TESTS::num(best) + ")");

    // Conditional mean minimizes the joint target in b; conditional covariance inverts its b-Hessian.
    const auto conditional = marginal.conditional_bias(s, theta, w);
    auto p = theta;
    p.insert(p.end(), conditional.mean.begin(), conditional.mean.end());
    Field js;
    std::vector<double> jp;
    joint.evaluate(s, p, w, js, jp);
    double stationarity = 0;
    for (int a = 0; a < 4; ++a)
        stationarity = std::max(stationarity, std::abs(jp[1 + a]) * gaussian[a].sigma);
    check(stationarity < 1e-8, label + " conditional mean is the joint minimum in b");
    double hessian_error = 0;
    const double eps = 1e-4;
    for (int b = 0; b < 4; ++b) {
        auto pp = p, pm = p;
        pp[1 + b] += eps * gaussian[b].sigma;
        pm[1 + b] -= eps * gaussian[b].sigma;
        std::vector<double> gplus, gminus;
        joint.evaluate(s, pp, w, js, gplus);
        joint.evaluate(s, pm, w, js, gminus);
        // (H * Cov) e_b should equal e_b: H column b from differences, Cov from the conditional.
        for (int a = 0; a < 4; ++a) {
            double hc = 0;
            for (int d = 0; d < 4; ++d) {
                auto qp = p, qm = p;
                qp[1 + d] += eps * gaussian[d].sigma;
                qm[1 + d] -= eps * gaussian[d].sigma;
                std::vector<double> g1, g2;
                joint.evaluate(s, qp, w, js, g1);
                joint.evaluate(s, qm, w, js, g2);
                const double H_ad = (g1[1 + a] - g2[1 + a]) / (2 * eps * gaussian[d].sigma);
                hc += H_ad * conditional.covariance[d * 4 + b];
            }
            hessian_error = std::max(hessian_error, std::abs(hc - (a == b ? 1. : 0.)));
        }
        (void)gplus;
        (void)gminus;
    }
    check(hessian_error < 1e-6, label + " conditional covariance inverts the joint bias Hessian (" +
                                    FML::TESTS::num(hessian_error) + ")");

    // Marginal difference between two fields = joint minimum difference + 1/2 log det A difference.
    auto s2 = s;
    for (size_t q = 0; q < s2.size(); ++q)
        s2[q] += .2 * direction[q];
    const double U2 = marginal.evaluate(s2, theta, w, gs, gt);
    const auto c2 = marginal.conditional_bias(s2, theta, w);
    auto p2 = theta;
    p2.insert(p2.end(), c2.mean.begin(), c2.mean.end());
    const double J1 = joint.evaluate(s, p, w, js, jp), J2 = joint.evaluate(s2, p2, w, js, jp);
    auto logdet = [](const auto & chol) {
        double v = 0;
        for (int i = 0; i < 4; ++i)
            v += 2 * std::log(chol[i * 4 + i]);
        return v;
    };
    // log det A = -log det Cov.
    const double expected = (J2 - J1) - .5 * (logdet(c2.cholesky) - logdet(conditional.cholesky));
    check(std::abs((U2 - U) - expected) < 1e-8 * std::max(1., std::abs(expected)),
          label + " marginal differences equal joint minima plus log-determinant change");

    // Bounded (uniform) bias priors are rejected: the marginal would not be exact.
    std::array<FF::Parameter, 4> uniform = gaussian;
    uniform[0] = {"b_delta", 1.6, 0, 1, 0, 4, FF::Prior::Uniform};
    Model bounded(c, FF::sigma8_spectrum([](double k) { return 20000 * std::exp(-5 * k); }), uniform, domain);
    check(FML::TESTS::throws([&] { FF::BiasMarginalizedPosterior<Grid> invalid(bounded, data, noise); }),
          label + " bounded bias priors rejected");

    // Short HMC on the marginal target: runs, finite, no divergences.
    FML::MCMC::HMCOptions o;
    o.warmup = 20;
    o.samples = 20;
    o.n_leapfrog = 8;
    o.step_size = .05;
    o.target_acceptance = .8;
    o.seed = 5;
    const auto result = FF::sample_bias_marginalized(marginal, s, theta, o);
    bool finite = result.completed_samples == 20;
    for (double u : result.potential)
        finite &= std::isfinite(u);
    check(finite && result.divergences == 0, label + " marginal HMC runs without divergences");
}

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    using Slab = FML::GRID::FFTWGrid<3>;
    exercise<Slab>(FML::default_particle_domain, "slab compact pswf", true, true);
    exercise<Slab>(FML::default_particle_domain, "slab shared smooth", false, false);
#ifdef USE_MPI
    const int px = FML::NTasks % 2 == 0 ? 2 : 1;
    std::vector<double> xe, ye;
    for (int i = 0; i <= px; ++i)
        xe.push_back(double(i) / px);
    for (int i = 0; i <= FML::NTasks / px; ++i)
        ye.push_back(double(i) / (FML::NTasks / px));
    exercise<FML::GRID::PencilGrid<3>>(FML::Domain::pencil(xe, ye), "pencil compact pswf", true, true);
#endif
    return FML::TESTS::summary("bias marginalization");
}
