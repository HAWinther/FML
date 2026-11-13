#include <FML/SPH/SPH.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
#include <iomanip>
using namespace FML::SPH;
using FML::TESTS::check;
namespace C = FML::COMMUNICATION;
template <int D>
std::vector<Particle<D>> lattice(int n, bool irregular = false) {
    size_t count = 1;
    for (int d = 0; d < D; ++d)
        count *= n;
    std::vector<Particle<D>> p;
    for (size_t id = 0; id < count; ++id) {
        if (int(id % FML::NTasks) != FML::ThisTask)
            continue;
        Particle<D> q;
        q.id = id;
        q.mass = 1. / count;
        q.energy = 1 + .15 * std::sin(.79 * id);
        size_t k = id;
        for (int d = 0; d < D; ++d) {
            q.position[d] = (k % n + .5) / n;
            k /= n;
            if (irregular)
                q.position[d] += .08 / n * std::sin(.53 * (id + 1) * (d + 1));
            q.velocity[d] = irregular ? .1 * std::sin(.37 * (id + 1) * (d + 2)) : 0;
        }
        if (irregular)
            q.mass *= D == 1 ? 1 + .45 * (id % 2 ? 1 : -1) : 1 + .25 * std::cos(.67 * id);
        p.push_back(q);
    }
    return p;
}
template <int D>
Options options(int n) {
    Options o;
    o.h_min = .45 / n;
    o.h_max = 1.7 / n;
    o.eta = 1.2;
    o.density_tolerance = 2e-12;
    o.max_message_bytes = 512;
    return o;
}
template <int D>
void kernel_tests() {
    CubicSpline<D> kernel;
    const int n = 10000;
    double integral = 0;
    const double area = D == 1 ? 2 : D == 2 ? 2 * M_PI : 4 * M_PI;
    for (int j = 0; j <= n; ++j) {
        double r = 2. * j / n;
        double y = kernel(r, 1).w * std::pow(r, D - 1) * area;
        integral += (j == 0 || j == n ? 1 : j % 2 ? 4 : 2) * y;
    }
    integral *= 2. / n / 3;
    check(std::abs(integral - 1) < 1e-11, "kernel normalized in D=" + std::to_string(D));
    double error = 0;
    for (double r : {.13, .9, 1., 1.6, 1.99}) {
        auto k = kernel(r, 1);
        const double eps = 1e-6;
        error = std::max(error, std::abs(k.dr - (kernel(r + eps, 1).w - kernel(r - eps, 1).w) / (2 * eps)));
        error = std::max(error, std::abs(k.dh - (kernel(r, 1 + eps).w - kernel(r, 1 - eps).w) / (2 * eps)));
    }
    check(error < 1e-8, "radial/h derivatives independently differenced in D=" + std::to_string(D));
    check(kernel(2, 1).w == 0 && kernel(2.01, 1).dr == 0 && kernel(0, 1).dr == 0, "kernel support/central gradient");
}
template <int D>
void conservative_tests(int n) {
    auto o = options<D>(n);
    if constexpr (D == 1)
        o.h_max = 2.5 / n;
    o.conductivity_alpha = .2;
    Solver<D> solver(lattice<D>(n, true), o);
    auto result = solver.evaluate();
    const auto & p = solver.particles();
    struct Record {
        Particle<D> p;
        Thermodynamics gas;
    };
    std::vector<Record> local;
    for (size_t i = 0; i < p.size(); ++i)
        local.push_back({p[i], result.gas[i]});
    const auto full = C::gather_records_checked(local, true);
    double rho_error = 0, force_error = 0, du_error = 0, root_error = 0, power = 0;
    Vector<D> momentum{};
    unsigned long long union_pairs = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        const auto & a = p[i];
        const auto & g = result.gas[i];
        double rho = 0, dh = 0, du = 0;
        Vector<D> acc{};
        for (const auto & r : full) {
            const auto & b = r.p;
            const auto & q = r.gas;
            auto dx = FML::SPH::detail::separation<D>(a.position, b.position, o.boxsize);
            double r2 = FML::SPH::detail::dot<D>(dx, dx), distance = std::sqrt(r2);
            auto wi = CubicSpline<D>{}(distance, a.h), wj = CubicSpline<D>{}(distance, b.h);
            rho += b.mass * wi.w;
            dh += b.mass * wi.dh;
            if (distance == 0 || a.id == b.id)
                continue;
            if (distance >= 2 * a.h && distance < 2 * b.h)
                ++union_pairs;
            double dot = 0;
            for (int d = 0; d < D; ++d)
                dot += (a.velocity[d] - b.velocity[d]) * dx[d];
            const double havg = (a.h + b.h) / 2, rhoavg = (g.density + q.density) / 2;
            double visc = 0;
            if (dot < 0) {
                double mu = havg * dot / (r2 + o.viscosity_epsilon * havg * havg);
                visc = (-o.viscosity_alpha * (g.sound_speed + q.sound_speed) * mu / 2 + o.viscosity_beta * mu * mu) /
                       rhoavg * (g.limiter + q.limiter) / 2;
            }
            const double fi = g.pressure / (g.omega * g.density * g.density),
                         fj = q.pressure / (q.omega * q.density * q.density);
            for (int d = 0; d < D; ++d)
                acc[d] -= b.mass * (fi * wi.dr + fj * wj.dr + visc * (wi.dr + wj.dr) / 2) * dx[d] / distance;
            du += b.mass * (fi * wi.dr + visc * (wi.dr + wj.dr) / 4) * dot / distance;
            du += o.conductivity_alpha * b.mass / rhoavg * (a.energy - b.energy) *
                  std::sqrt(std::abs(g.pressure - q.pressure) / rhoavg) * (wi.dr + wj.dr) / 2;
        }
        rho_error = std::max(rho_error, std::abs(rho - g.density));
        root_error = std::max(root_error, std::abs(std::pow(a.h, D) * rho / std::pow(o.eta, D) / a.mass - 1));
        rho_error = std::max(rho_error, std::abs(1 + a.h * dh / (D * rho) - g.omega));
        for (int d = 0; d < D; ++d) {
            force_error = std::max(force_error, std::abs(acc[d] - result.acceleration[i][d]));
            momentum[d] += a.mass * result.acceleration[i][d];
        }
        du_error = std::max(du_error, std::abs(du - result.energy_rate[i]));
        power += a.mass * (FML::SPH::detail::dot<D>(a.velocity, result.acceleration[i]) + result.energy_rate[i]);
    }
    FML::MaxOverTasks(&rho_error);
    FML::MaxOverTasks(&root_error);
    FML::MaxOverTasks(&force_error);
    FML::MaxOverTasks(&du_error);
    FML::SumOverTasks(&union_pairs);
    FML::SumOverTasks(&power);
    for (auto & x : momentum)
        FML::SumOverTasks(&x);
    double momentum_error = 0;
    for (auto x : momentum)
        momentum_error = std::max(momentum_error, std::abs(x));
    check(rho_error < 2e-11 && root_error < 3e-12, "all-pair density/grad-h root in D=" + std::to_string(D));
    check(force_error < 3e-10 && du_error < 3e-10, "cell forces/heat agree with all-pair variable-support reference");
    check(union_pairs > 0, "fixture exercises r>2h_i but r<2h_j support union");
    check(momentum_error < 3e-12 && std::abs(power) < 3e-12, "semi-discrete momentum and total energy conservation");
    check(result.timestep > 0 && std::isfinite(result.timestep), "finite shared CFL/acceleration/energy timestep");
    // Independent variational force check at fixed entropy, with viscosity/conductivity disabled.
    auto p0 = solver.particles();
    for (auto & x : p0)
        x.velocity.fill(0);
    o.viscosity_alpha = o.viscosity_beta = o.conductivity_alpha = 0;
    Solver<D> adiabatic(p0, o);
    auto base = adiabatic.evaluate();
    struct Entropy {
        std::uint64_t id;
        double value;
    };
    std::vector<Entropy> entropy;
    FML::HYDRODYNAMICS::IdealGas eos(o.gamma);
    for (size_t i = 0; i < adiabatic.particles().size(); ++i)
        entropy.push_back(
            {adiabatic.particles()[i].id, eos.entropy_proxy(base.gas[i].density, adiabatic.particles()[i].energy)});
    auto all_entropy = C::gather_records_checked(entropy, true);
    std::sort(all_entropy.begin(), all_entropy.end(), [](auto a, auto b) { return a.id < b.id; });
    double directional = 0;
    for (size_t i = 0; i < adiabatic.particles().size(); ++i)
        for (int d = 0; d < D; ++d)
            directional -= adiabatic.particles()[i].mass * base.acceleration[i][d] *
                           std::sin(.33 * (adiabatic.particles()[i].id + 1) * (d + 1));
    FML::SumOverTasks(&directional);
    auto energy = [&](double eps) {
        auto moved = adiabatic.particles();
        for (auto & x : moved)
            for (int d = 0; d < D; ++d) {
                x.position[d] += eps * std::sin(.33 * (x.id + 1) * (d + 1));
                x.position[d] -= std::floor(x.position[d]);
            }
        Solver<D> shifted(moved, o);
        auto state = shifted.evaluate();
        double u = 0;
        for (size_t i = 0; i < shifted.particles().size(); ++i) {
            const auto & x = shifted.particles()[i];
            auto it = std::lower_bound(all_entropy.begin(), all_entropy.end(), x.id, [](auto e, auto id) {
                return e.id < id;
            });
            u += x.mass * it->value * std::pow(state.gas[i].density, o.gamma - 1) / (o.gamma - 1);
        }
        FML::SumOverTasks(&u);
        return u;
    };
    double fd = (energy(2e-6) - energy(-2e-6)) / 4e-6;
    check(std::abs(fd - directional) < 2e-6 * std::max(1., std::abs(fd)),
          "pressure is finite-difference gradient of adaptive fixed-entropy energy");
    if (FML::ThisTask == 0)
        std::cout << "REFERENCE " << D << " density=" << rho_error << " root=" << root_error
                  << " acceleration=" << force_error << " heat=" << du_error << " momentum=" << momentum_error
                  << " energy_rate=" << power << " variational=" << std::abs(fd - directional)
                  << " support_union=" << union_pairs << "\n";
}
void transport_and_safety() {
    auto o = options<3>(8);
    auto p = lattice<3>(8);
    for (auto & q : p) {
        q.energy = 1;
        q.velocity = {2.3, -.4, .1};
    }
    Solver<3> gas(p, o);
    const auto before = gas.diagnostics();
    gas.step(.2);
    const auto after = gas.diagnostics();
    double error = 0;
    for (const auto & q : gas.particles()) {
        size_t k = q.id;
        for (int d = 0; d < 3; ++d) {
            double expected = (k % 8 + .5) / 8 + .2 * std::array<double, 3>{2.3, -.4, .1}[d];
            k /= 8;
            expected -= std::floor(expected);
            error = std::max(error, std::abs(q.position[d] - expected));
        }
        error = std::max(error, std::abs(q.energy - 1));
    }
    FML::MaxOverTasks(&error);
    check(error < 2e-12 && std::abs(before.gas_energy() - after.gas_energy()) < 1e-12,
          "uniform flow wraps/migrates across arbitrary slabs without heating");
    check(FML::TESTS::throws([&] {
              auto bad = p;
              if (FML::ThisTask == 0 && !bad.empty())
                  bad[0].energy = -1;
              Solver<3> fail(bad, o);
          }),
          "one-owner invalid energy rejects collectively");
    check(FML::TESTS::throws([&] {
              auto bad = o;
              if (FML::ThisTask == 0)
                  bad.gamma = 1;
              Solver<3> fail(p, bad);
          }),
          "one-rank invalid EOS rejects collectively");
    if (FML::NTasks > 1)
        check(FML::TESTS::throws([&] {
                  auto bad = o;
                  if (FML::ThisTask == 0)
                      bad.courant = .23;
                  Solver<3> fail(p, bad);
              }),
              "rank-inconsistent options reject collectively");
    check(FML::TESTS::throws([&] {
              auto bad = p;
              Particle<3> q;
              q.id = UINT64_MAX;
              q.position = {.1, .2, .3};
              if (FML::ThisTask == 0) {
                  bad.push_back(q);
                  bad.push_back(q);
              }
              Solver<3> fail(bad, o);
          }),
          "duplicate IDs rejected without catalogue all-gather");
    check(FML::TESTS::throws([&] {
              auto bad = p;
              Particle<3> q;
              q.id = UINT64_MAX;
              q.position = {.1, .2, .3};
              if (FML::ThisTask == 0 || (FML::NTasks > 1 && FML::ThisTask == 1))
                  bad.push_back(q);
              if (FML::NTasks == 1)
                  bad.push_back(q);
              Solver<3> fail(bad, o);
          }),
          "cross-rank duplicate IDs rejected by hash partition");
    Solver<3> empty({}, o);
    check(empty.evaluate().timestep == std::numeric_limits<double>::infinity() && empty.diagnostics().mass == 0,
          "globally empty gas is a valid collective no-work state");
    std::vector<Particle<3>> isolated;
    if (FML::ThisTask == 0) {
        Particle<3> q;
        q.position = {.2, .2, .2};
        isolated.push_back(q);
    }
    Solver<3> sparse(isolated, o);
    check(FML::TESTS::throws([&] { sparse.evaluate(); }), "unbracketed smoothing root rejects rather than clamping");
    auto few = o;
    few.density_iterations = 1;
    Solver<3> exhausted(p, few);
    check(FML::TESTS::throws([&] { exhausted.evaluate(); }), "density iteration exhaustion is explicit");
    auto exp = [](double t) { return Background{std::exp(t), 1}; };
    Solver<3> expanding(p, o, exp);
    auto initial = expanding.diagnostics();
    auto original_particles = expanding.particles();
    check(FML::TESTS::throws([&] { expanding.step(1); }), "negative predictor energy rejects the full step");
    auto final = expanding.diagnostics();
    bool unchanged = original_particles.size() == expanding.particles().size();
    for (size_t i = 0; i < original_particles.size(); ++i) {
        const auto & a = original_particles[i];
        const auto & b = expanding.particles()[i];
        unchanged &=
            a.id == b.id && a.position == b.position && a.velocity == b.velocity && a.energy == b.energy && a.h == b.h;
    }
    check(unchanged && expanding.time() == 0 && initial.gas_energy() == final.gas_energy() &&
              initial.momentum == final.momentum,
          "failed RK2 step preserves particles and epoch");
    auto wrong = [](double, const auto &, auto & force) {
        if (FML::ThisTask == 0)
            force.clear();
    };
    check(FML::TESTS::throws([&] { gas.evaluate(wrong); }),
          "one-owner external-force shape error rejects collectively");
    auto nan = [](double, const auto &, auto & force) {
        if (FML::ThisTask == 0 && !force.empty())
            force[0][0] = NAN;
    };
    check(FML::TESTS::throws([&] { gas.evaluate(nan); }), "one-owner nonfinite external force rejects collectively");
    if (FML::NTasks > 1) {
        Solver<3>::ExternalForce cb;
        if (FML::ThisTask == 0)
            cb = wrong;
        check(FML::TESTS::throws([&] { gas.evaluate(cb); }), "external callback presence agrees before callbacks run");
    }
    // A callback's acceleration is added once, and enters the acceleration timestep.
    auto constant = [](double, const auto &, auto & force) {
        for (auto & a : force)
            a = {.2, 0, 0};
    };
    Solver<3> accelerated(p, o);
    accelerated.step(.01, constant);
    double dv = 0;
    for (const auto & q : accelerated.particles())
        dv = std::max(dv, std::abs(q.velocity[0] - 2.302));
    FML::MaxOverTasks(&dv);
    check(dv < 2e-12, "external gravity/source acceleration is added once in physical units");
    // Bounded communication is exercised above with 512-byte messages.
    auto local = lattice<3>(16);
    auto narrow = options<3>(16);
    Solver<3> selective(local, narrow);
    auto state = selective.evaluate();
    unsigned long long ghosts = state.density_ghosts, global = selective.particles().size();
    FML::SumOverTasks(&ghosts);
    FML::SumOverTasks(&global);
    check(FML::NTasks == 1 || ghosts < global * static_cast<unsigned long long>(FML::NTasks - 1),
          "slab exchange selects support ghosts instead of replicating all particles");
}
void dissipation_test() {
    auto p = lattice<1>(32, true);
    auto o = options<1>(32);
    o.h_max = 2.5 / 32;
    o.viscosity_alpha = o.viscosity_beta = o.conductivity_alpha = 0;
    Solver<1> inviscid(p, o);
    auto a = inviscid.evaluate();
    o.viscosity_alpha = 1;
    o.viscosity_beta = 2;
    Solver<1> viscous(p, o);
    auto b = viscous.evaluate();
    double min_heat = 0, viscous_work = 0;
    for (size_t i = 0; i < inviscid.particles().size(); ++i) {
        const auto & q = inviscid.particles()[i];
        min_heat = std::min(min_heat, b.energy_rate[i] - a.energy_rate[i]);
        viscous_work += q.mass * q.velocity[0] * (b.acceleration[i][0] - a.acceleration[i][0]);
    }
    FML::MinOverTasks(&min_heat);
    FML::SumOverTasks(&viscous_work);
    check(min_heat > -1e-12 && viscous_work < 0, "viscosity heats approaching pairs and dissipates kinetic energy");
    o.viscosity_alpha = o.viscosity_beta = 0;
    o.conductivity_alpha = .3;
    Solver<1> conductive(p, o);
    auto c = conductive.evaluate();
    double heat = 0, entropy = 0;
    for (size_t i = 0; i < inviscid.particles().size(); ++i) {
        const auto & q = inviscid.particles()[i];
        double du = c.energy_rate[i] - a.energy_rate[i];
        heat += q.mass * du;
        entropy += q.mass * du / q.energy;
    }
    FML::SumOverTasks(&heat);
    FML::SumOverTasks(&entropy);
    check(std::abs(heat) < 1e-12 && entropy > 0, "conductivity conserves thermal energy and increases entropy");
    auto shear = lattice<3>(8);
    for (auto & q : shear) {
        q.energy = 1;
        q.velocity = {0, .1 * std::sin(2 * M_PI * q.position[0]), 0};
    }
    auto opt = options<3>(8);
    Solver<3> sheared(shear, opt);
    auto s = sheared.evaluate();
    double limiter = 0;
    for (const auto & g : s.gas)
        limiter = std::max(limiter, g.limiter);
    FML::MaxOverTasks(&limiter);
    check(limiter < 1e-10, "Balsara limiter suppresses viscosity in a divergence-free periodic shear");
    for (auto & q : shear)
        q.velocity = {-.1 * std::sin(2 * M_PI * q.position[0]), 0, 0};
    Solver<3> compression(shear, opt);
    auto rates = compression.evaluate();
    double active = 1;
    for (const auto & g : rates.gas)
        active = std::min(active, g.limiter);
    FML::MinOverTasks(&active);
    check(active > .99, "Balsara limiter retains viscosity in irrotational compression");
}
void valid_empty_rank_test() {
    auto p = lattice<3>(4);
    for (auto & q : p) {
        q.energy = 1;
        for (auto & x : q.position)
            x = .01 + .005 * x;
    }
    Options o;
    o.h_min = 1e-5;
    o.h_max = .01;
    o.max_message_bytes = 512;
    Solver<3> cluster(p, o);
    auto state = cluster.evaluate();
    unsigned long long empty = cluster.particles().empty();
    FML::SumOverTasks(&empty);
    check(std::isfinite(state.timestep) && state.timestep > 0 && (FML::NTasks == 1 || empty > 0),
          "valid clustered gas solves forces/timestep with spatially empty MPI ranks");
}
void units_and_background_test() {
    auto p = lattice<1>(32, true);
    auto o = options<1>(32);
    o.h_max = 2.5 / 32;
    Solver<1> base(p, o);
    auto a = base.evaluate();
    for (auto & q : p)
        q.mass *= 1e160;
    Solver<1> scaled(p, o);
    auto b = scaled.evaluate();
    double error = 0;
    for (size_t i = 0; i < base.particles().size(); ++i) {
        error = std::max(error, std::abs(a.acceleration[i][0] - b.acceleration[i][0]));
        error = std::max(error, std::abs(a.energy_rate[i] - b.energy_rate[i]));
        error = std::max(error, std::abs(base.particles()[i].h - scaled.particles()[i].h));
    }
    FML::MaxOverTasks(&error);
    check(error < 2e-10,
          "hydrodynamic rates are invariant under extreme mass-unit rescaling without rho-squared overflow");
    auto q = lattice<3>(8, true);
    auto opts = options<3>(8);
    opts.viscosity_alpha = opts.viscosity_beta = 0;
    Solver<3> physical(q, opts);
    auto r = physical.evaluate();
    Solver<3> comoving(q, opts, [](double) { return Background{2, .3}; });
    auto c = comoving.evaluate();
    double difference = 0;
    for (size_t i = 0; i < physical.particles().size(); ++i) {
        const auto & x = physical.particles()[i];
        for (int d = 0; d < 3; ++d)
            difference =
                std::max(difference, std::abs(c.acceleration[i][d] - (r.acceleration[i][d] / 2 - .3 * x.velocity[d])));
        difference =
            std::max(difference,
                     std::abs(c.energy_rate[i] - (r.energy_rate[i] / 2 - 3 * (opts.gamma - 1) * .3 * x.energy)));
    }
    FML::MaxOverTasks(&difference);
    check(difference < 1e-12, "nonuniform comoving hydro scales by 1/a with explicit Hubble drag/cooling");
}
void cosmological_viscosity_test() {
    auto p = lattice<3>(8);
    for (auto & q : p) {
        q.energy = 1;
        q.velocity = {-.01 * std::sin(2 * M_PI * q.position[0]), 0, 0};
    }
    auto o = options<3>(8);
    auto expansion = [](double) { return Background{1, 1}; };
    Solver<3> dissipative(p, o, expansion);
    auto with = dissipative.evaluate();
    o.viscosity_alpha = o.viscosity_beta = 0;
    Solver<3> ideal(p, o, expansion);
    auto without = ideal.evaluate();
    double error = 0;
    for (size_t i = 0; i < ideal.particles().size(); ++i) {
        error = std::max(error, std::abs(with.energy_rate[i] - without.energy_rate[i]));
        for (int d = 0; d < 3; ++d)
            error = std::max(error, std::abs(with.acceleration[i][d] - without.acceleration[i][d]));
    }
    FML::MaxOverTasks(&error);
    check(error < 1e-12,
          "Hubble separation prevents spurious viscosity for peculiar compression with physically receding pairs");
}
void expansion_test() {
    auto o = options<3>(8);
    o.viscosity_alpha = o.viscosity_beta = 0;
    auto evolve = [&](double dt) {
        auto p = lattice<3>(8);
        for (auto & q : p) {
            q.energy = 1;
            q.velocity = {.1, 0, 0};
        }
        Solver<3> s(p, o, [](double t) { return Background{std::exp(.2 * t), .2}; });
        for (int i = 0; i < int(std::round(.1 / dt)); ++i)
            s.step(dt);
        double error = 0;
        auto rates = s.evaluate();
        for (size_t i = 0; i < s.particles().size(); ++i) {
            auto & q = s.particles()[i];
            error = std::max(error, std::abs(q.energy - std::exp(-.4 * .1)));
            error = std::max(error, std::abs(q.velocity[0] - .1 * std::exp(-.2 * .1)));
        }
        FML::MaxOverTasks(&error);
        return error;
    };
    const double e1 = evolve(.02), e2 = evolve(.01);
    check(e1 / e2 > 3.7 && e1 / e2 < 4.3 && e2 < 2e-7,
          "comoving homogeneous Hubble drag/adiabatic cooling are second order");
    if (FML::ThisTask == 0)
        std::cout << "EXPANSION " << e1 << ' ' << e2 << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(17);
    try {
        FML::HYDRODYNAMICS::IdealGas eos(1.4);
        check(std::abs(eos.specific_energy(2, eos.pressure(2, 3)) - 3) < 1e-14 &&
                  std::abs(eos.sound_speed_squared(2, 3) - 1.68) < 1e-14,
              "shared ideal-gas pressure/energy/sound-speed conventions");
        kernel_tests<1>();
        kernel_tests<2>();
        kernel_tests<3>();
        conservative_tests<1>(24);
        conservative_tests<2>(8);
        conservative_tests<3>(8);
        transport_and_safety();
        dissipation_test();
        valid_empty_rank_test();
        units_and_background_test();
        cosmological_viscosity_test();
        expansion_test();
        return FML::TESTS::summary("SPH");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
}
