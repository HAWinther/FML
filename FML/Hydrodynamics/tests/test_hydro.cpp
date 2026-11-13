#include <FML/Hydrodynamics/FiniteVolume.h>
#include <FML/Hydrodynamics/ExactRiemann.h>
#include <FML/tests/TestHelpers.h>
#include <iostream>
#include <iomanip>
using namespace FML::HYDRODYNAMICS;
using FML::TESTS::check;

double maximum(double x) {
    FML::MaxOverTasks(&x);
    return x;
}
void riemann_tests() {
    IdealGas eos(1.4);
    Primitive<3> q{1.3, {.3, -.2, .1}, .7};
    auto u = conserved<3>(q, eos);
    auto p = primitive<3>(u, eos);
    check(std::abs(p.pressure - q.pressure) < 1e-14 && p.velocity == q.velocity,
          "Euler primitive/conserved conversions share ideal-gas conventions");
    double error = 0;
    for (auto method : {FluxMethod::Rusanov, FluxMethod::HLLE, FluxMethod::HLLC})
        for (int axis = 0; axis < 3; ++axis) {
            auto result = riemann<3>(q, q, eos, axis, method);
            auto expected = physical_flux<3>(q, u, axis);
            for (int j = 0; j < 6; ++j)
                error = std::max(error, std::abs(result.flux[j] - expected[j]));
            check(result.valid, "equal-state Riemann result is representable");
        }
    check(error < 1e-13, "all fluxes reproduce analytic uniform Euler flux in every axis");
    Primitive<1> l{1, {0}, 1}, r{2, {0}, 1};
    auto contact = riemann<1>(l, r, eos), diffusive = riemann<1>(l, r, eos, 0, FluxMethod::HLLE);
    check(contact.valid && contact.method == FluxMethod::HLLC && contact.flux[0] == 0 &&
              std::abs(contact.flux[1] - 1) < 1e-14 && contact.flux[2] == 0 && diffusive.flux[0] != 0 &&
              std::abs(diffusive.interface_velocity) < 1e-14 && diffusive.flux[3] == 0,
          "HLLC resolves stationary contact absent in HLLE");
    l.velocity[0] = r.velocity[0] = .7;
    auto advected = riemann<1>(l, r, eos);
    auto fl = physical_flux<1>(l, conserved<1>(l, eos), 0);
    error = 0;
    for (int j = 0; j < 4; ++j)
        error = std::max(error, std::abs(advected.flux[j] - fl[j]));
    check(error < 1e-13, "HLLC advected contact equals the correct upwind physical flux");
    Primitive<3> a{1, {0, .4, .2}, 1}, b{1, {0, -.3, -.1}, 1};
    auto shear = riemann<3>(a, b, eos);
    check(shear.valid && shear.flux[0] == 0 && shear.flux[2] == 0 && shear.flux[3] == 0 && shear.flux[4] == 0,
          "stationary tangential velocity contact carries no spurious shear flux");
    l = {1, {5}, 1};
    r = {.8, {4}, .7};
    for (auto method : {FluxMethod::HLLE, FluxMethod::HLLC}) {
        auto f = riemann<1>(l, r, eos, 0, method);
        auto ex = physical_flux<1>(l, conserved<1>(l, eos), 0);
        error = 0;
        for (int j = 0; j < 4; ++j)
            error = std::max(error, std::abs(f.flux[j] - ex[j]));
        check(error < 1e-12, "supersonic flux selects upwind state");
    }
    ExactRiemann sod({1, {0}, 1}, {.125, {0}, .1}, eos);
    check(std::abs(sod.star_pressure() - .30313017805064685) < 1e-12 &&
              std::abs(sod.star_velocity() - .9274526200489499) < 1e-12,
          "exact Sod pressure and contact speed match independent reference constants");
    auto left = sod.sample(-10), right = sod.sample(10);
    check(left.density == 1 && right.density == .125, "exact far fields preserve input states");
    ExactRiemann mirror({.125, {0}, .1}, {1, {0}, 1}, eos);
    error = 0;
    for (int j = -50; j <= 50; ++j) {
        auto x = sod.sample(j * .05), y = mirror.sample(-j * .05);
        error = std::max({error,
                          std::abs(x.density - y.density),
                          std::abs(x.pressure - y.pressure),
                          std::abs(x.velocity[0] + y.velocity[0])});
    }
    check(error < 1e-12, "exact shock/rarefaction solution respects left-right reflection");
    ExactRiemann vacuum({1, {-8}, 1}, {1, {8}, 1}, eos);
    auto empty = vacuum.sample(0);
    auto vflux = riemann<1>({1, {-8}, 1}, {1, {8}, 1}, eos);
    check(vacuum.has_vacuum() && empty.density == 0 && empty.pressure == 0 && vflux.valid &&
              vflux.method == FluxMethod::HLLE && vflux.fallback == FluxFallback::Vacuum,
          "generated vacuum has exact reference and explicit HLLC-to-HLLE fallback");
    auto expansion = riemann<1>({1, {-2}, .4}, {1, {2}, .4}, eos);
    ExactRiemann rare({1, {-2}, .4}, {1, {2}, .4}, eos);
    check(expansion.valid && expansion.method == FluxMethod::HLLE && expansion.fallback != FluxFallback::None &&
              !rare.has_vacuum() && rare.star_pressure() > 0,
          "strong two-rarefaction problem rejects negative HLLC star pressure");
    check(!riemann<1>({-1, {0}, 1}, {1, {0}, 1}, eos).valid && !riemann<1>({1, {NAN}, 1}, {1, {0}, 1}, eos).valid,
          "invalid Riemann input is reported explicitly");
    Primitive<1> fast{1, {1e8}, 1};
    auto high = conserved<1>(fast, eos);
    DualEnergy dual;
    dual.enabled = true;
    auto recovered = recover<1>(high, eos, dual);
    check(recovered.valid && recovered.used_auxiliary && std::abs(recovered.state.pressure - 1) < 1e-14,
          "dual-energy pressure avoids subtracting dominant kinetic energy");
    // Deterministic broad positive-state sample; check forward-Euler two-cell updates independently.
    unsigned long long random = 123, failures = 0, fallbacks = 0;
    auto uniform = [&] {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        return double(random >> 11) * 0x1p-53;
    };
    for (int k = 0; k < 512; ++k) {
        Primitive<1> x{std::exp(10 * uniform() - 5), {20 * uniform() - 10}, std::exp(10 * uniform() - 5)},
            y{std::exp(10 * uniform() - 5), {20 * uniform() - 10}, std::exp(10 * uniform() - 5)};
        for (auto method : {FluxMethod::Rusanov, FluxMethod::HLLE}) {
            auto f = riemann<1>(x, y, eos, 0, method);
            auto ux = conserved<1>(x, eos), uy = conserved<1>(y, eos);
            auto fx = physical_flux<1>(x, ux, 0), fy = physical_flux<1>(y, uy, 0);
            double dt = .1 / f.max_speed;
            for (int j = 0; j < 4; ++j) {
                ux[j] -= dt * (f.flux[j] - fx[j]);
                uy[j] -= dt * (fy[j] - f.flux[j]);
            }
            failures += !f.valid || !recover<1>(ux, eos).valid || !recover<1>(uy, eos).valid;
        }
        auto f = riemann<1>(x, y, eos);
        failures += !f.valid;
        fallbacks += f.fallback != FluxFallback::None;
    }
    check(failures == 0 && fallbacks > 0,
          "512 independent state pairs exercise positive low-order updates and HLLC fallback");
}
void halo_tests() {
    for (auto bc : {Boundary::Periodic, Boundary::Outflow, Boundary::Reflecting}) {
        CartesianGrid<1> grid({3}, {1}, {{{bc, bc}}}, 1000, 64);
        std::vector<Conserved<1>> local(grid.layout().cells);
        for (size_t i = 0; i < local.size(); ++i) {
            int g = int(grid.layout().global_index(i));
            local[i] = {1. + g, 10. + g, 100. + g, 5. + g};
        }
        grid.exchange(local);
        double error = 0;
        const int map[7] = {1, 0, 0, 1, 2, 2, 1};
        const int signs[7] = {-1, -1, 1, 1, 1, -1, -1};
        for (size_t i = 0; i < grid.padded().size(); ++i) {
            int g = grid.layout().start + int(i) - 2, source = 0, sign = 1;
            if (bc == Boundary::Periodic)
                source = (g % 3 + 3) % 3;
            if (bc == Boundary::Outflow)
                source = std::clamp(g, 0, 2);
            if (bc == Boundary::Reflecting) {
                source = map[g + 2];
                sign = signs[g + 2];
            }
            const auto & x = grid.padded()[i];
            error = std::max({error,
                              std::abs(x[0] - (1 + source)),
                              std::abs(x[1] - sign * (10 + source)),
                              std::abs(x[2] - (100 + source))});
        }
        check(maximum(error) == 0, "two-layer halo maps match independent periodic/outflow/reflection tables");
    }
    CartesianGrid<1> one({1}, {1}, {{{Boundary::Reflecting, Boundary::Reflecting}}}, 1000, 64);
    std::vector<Conserved<1>> onecell(one.layout().cells, Conserved<1>{1, 2, 10, 3});
    one.exchange(onecell);
    double error = 0;
    for (size_t i = 0; i < one.padded().size(); ++i) {
        int g = int(i) - 2;
        double expected = std::abs(g) % 2 ? -2 : 2;
        error = std::max(error, std::abs(one.padded()[i][1] - expected));
    }
    check(maximum(error) == 0, "one-cell reflecting domain handles repeated reflections and empty ranks");
}
void corner_halo_test() {
    std::array<std::array<Boundary, 2>, 3> b = {{{Boundary::Periodic, Boundary::Periodic},
                                                 {Boundary::Reflecting, Boundary::Reflecting},
                                                 {Boundary::Outflow, Boundary::Outflow}}};
    CartesianGrid<3> grid({2, 3, 1}, {1, 1, 1}, b, 10000, 64);
    std::vector<Conserved<3>> state(grid.layout().cells);
    for (size_t i = 0; i < state.size(); ++i) {
        double id = grid.layout().global_index(i);
        state[i] = {1 + id, 10 + id, 20 + id, 30 + id, 100 + id, 3 + id};
    }
    grid.exchange(state);
    double error = 0;
    const int mapping[7] = {1, 0, 0, 1, 2, 2, 1}, signs[7] = {-1, -1, 1, 1, 1, -1, -1};
    for (size_t i = 0; i < grid.padded().size(); ++i) {
        auto p = FML::HYDRODYNAMICS::detail::coordinates<3>(i, grid.padded_shape());
        int x = (grid.layout().start + p[0] - 2) % 2;
        if (x < 0)
            x += 2;
        int y = mapping[p[1]], id = x * 3 + y;
        const auto & u = grid.padded()[i];
        error = std::max({error,
                          std::abs(u[0] - (1 + id)),
                          std::abs(u[1] - (10 + id)),
                          std::abs(u[2] - signs[p[1]] * (20 + id)),
                          std::abs(u[3] - (30 + id)),
                          std::abs(u[4] - (100 + id))});
    }
    check(maximum(error) == 0,
          "3D corner halos combine periodic, reflecting and outflow boundaries without transverse sign errors");
}
template <int D>
void uniform_test() {
    Options<D> o;
    o.shape.fill(3);
    o.shape[0] = 1;
    o.max_message_bytes = 64;
    Primitive<D> q;
    for (int d = 0; d < D; ++d)
        q.velocity[d] = .2 * (d + 1);
    for (auto method : {FluxMethod::Rusanov, FluxMethod::HLLE, FluxMethod::HLLC}) {
        o.flux = method;
        FiniteVolume<D> s(o, [&](auto) { return q; });
        auto old = s.cells();
        for (int i = 0; i < 3; ++i)
            s.step(.001);
        double error = 0;
        for (size_t i = 0; i < old.size(); ++i)
            for (int j = 0; j < D + 3; ++j)
                error = std::max(error, std::abs(old[i][j] - s.cells()[i][j]));
        check(maximum(error) < 2e-14, "uniform multidimensional flow preserved with one x cell and all fluxes");
        auto totals = s.diagnostics();
        check(std::abs(totals.mass - 1) < 1e-13, "owned-cell integral is independent of MPI slab widths/empty ranks");
    }
}
void evolution_tests() {
    auto error = [](int n) {
        Options<1> o;
        o.shape = {n};
        o.gamma = 1.4;
        double sinc = std::sin(M_PI / n) / (M_PI / n);
        FiniteVolume<1> s(o, [&](auto x) { return Primitive<1>{1 + .2 * sinc * std::sin(2 * M_PI * x[0]), {.7}, 1}; });
        auto before = s.diagnostics();
        while (s.time() < .15)
            s.advance(.15 - s.time());
        auto q = s.primitive_cells();
        double e = 0;
        for (size_t i = 0; i < q.size(); ++i)
            e += std::abs(q[i].density -
                          (1 + .2 * sinc * std::sin(2 * M_PI * (s.layout().center(i)[0] - .7 * s.time())))) /
                 n;
        FML::SumOverTasks(&e);
        auto after = s.diagnostics();
        check(std::abs(after.mass - before.mass) < 2e-13 && std::abs(after.energy - before.energy) < 2e-13,
              "periodic FV mass and energy conserved under nonuniform evolution");
        return e;
    };
    const double coarse = error(32), fine = error(64);
    check(fine < .45 * coarse, "limited second-order advection improves under mesh refinement");
    if (FML::ThisTask == 0)
        std::cout << "ADVECTION " << coarse << ' ' << fine << '\n';
    Options<1> wall;
    wall.shape = {32};
    wall.boundaries = {{{Boundary::Reflecting, Boundary::Reflecting}}};
    FiniteVolume<1> closed(wall, [](auto) { return Primitive<1>{1, {.4}, 1}; });
    auto b = closed.diagnostics();
    while (closed.time() < .03)
        closed.advance(.03 - closed.time());
    auto a = closed.diagnostics();
    check(std::abs(a.mass - b.mass) < 1e-13 && std::abs(a.energy - b.energy) < 1e-13 && a.momentum[0] < b.momentum[0],
          "reflecting walls conserve mass/energy and supply wall momentum impulse");
    Options<1> cold;
    cold.shape = {8};
    cold.gamma = 1.4;
    FiniteVolume<1> gas(cold, [](auto) { return Primitive<1>{1, {0}, 1e-6}; });
    auto old = gas.cells();
    auto gravity = [](double, const auto &, const auto &, auto & force) {
        for (auto & f : force)
            f[0] = 1000;
    };
    check(FML::TESTS::throws([&] { gas.step(.01, gravity); }),
          "fixed excessive source step rejects after conservative low-order fallback");
    check(gas.cells() == old && gas.time() == 0, "failed step preserves all conserved cells and epoch");
    double dt = gas.advance(.1, gravity);
    auto q = gas.primitive_cells();
    double err = 0;
    for (const auto & x : q)
        err = std::max({err, std::abs(x.velocity[0] - 1000 * dt), std::abs(x.pressure - 1e-6)});
    check(maximum(err) < 1e-11 && gas.statistics().rejected_attempts > 0 && gas.statistics().low_order_stages > 0,
          "adaptive positivity retries recover cold gravity kick with correct work and no clipping");
    Options<1> high;
    high.shape = {32};
    high.gamma = 1.4;
    high.dual.enabled = true;
    FiniteVolume<1> moving(high, [](auto x) { return Primitive<1>{x[0] < .5 ? 1. : 2., {1e8}, 1}; });
    double end = 2e-9;
    while (moving.time() < end)
        moving.advance(end - moving.time());
    err = 0;
    for (const auto & x : moving.primitive_cells())
        err = std::max(err, std::abs(x.pressure - 1));
    check(maximum(err) < 1e-4 && moving.statistics().dual_cells > 0 && std::isfinite(moving.statistics().energy_repair),
          "high-Mach contact retains pressure using explicit accounted dual-energy recovery");
    Options<2> transverse;
    transverse.shape = {32, 1};
    transverse.lengths = {1, 1e8};
    transverse.gamma = 1.4;
    transverse.dual.enabled = true;
    transverse.flux = FluxMethod::HLLE;
    FiniteVolume<2> stationary(transverse, [](auto x) { return Primitive<2>{x[0] < .5 ? 1. : 2., {0, 1e8}, 1}; });
    while (stationary.time() < .02)
        stationary.advance(.02 - stationary.time());
    double transverse_error = 0;
    for (const auto & q : stationary.primitive_cells())
        transverse_error = std::max(transverse_error, std::abs(q.pressure - 1));
    check(maximum(transverse_error) < 1e-8,
          "HLLE dual-energy transport keeps constant pressure across a high-Mach tangential contact");
    FiniteVolume<1> repaired(high, [](auto) { return Primitive<1>{1, {1e8}, 1}; });
    auto altered = repaired.cells();
    for (auto & u : altered)
        u[2] -= 4;
    repaired.set_state(altered);
    check(std::abs(repaired.statistics().initial_energy_repair - 4) < 1e-13 && repaired.statistics().energy_repair == 0,
          "replacement-state dual repair is reported separately from accepted-step repairs");
    if (FML::ThisTask == 0)
        std::cout << "DUAL " << err << ' ' << moving.statistics().energy_repair << ' '
                  << moving.statistics().absolute_energy_repair << '\n';
}
void safety_and_sources() {
    Options<1> o;
    o.shape = {8};
    check(FML::TESTS::throws([&] {
              auto p = o;
              if (FML::ThisTask == 0)
                  p.gamma = 1;
              FiniteVolume<1> bad(p);
          }),
          "one-rank invalid EOS fails collectively");
    if (FML::NTasks > 1)
        check(FML::TESTS::throws([&] {
                  auto p = o;
                  if (FML::ThisTask == 0)
                      p.courant = .2;
                  FiniteVolume<1> bad(p);
              }),
              "inconsistent rank configurations fail collectively");
    check(FML::TESTS::throws([&] {
              auto p = o;
              p.boundaries[0][0] = Boundary::Outflow;
              FiniteVolume<1> bad(p);
          }),
          "one-sided periodic boundary rejected");
    check(FML::TESTS::throws([&] {
              FiniteVolume<1> bad(o, [](auto) {
                  Primitive<1> q;
                  if (FML::ThisTask == 0)
                      q.pressure = -1;
                  return q;
              });
          }),
          "one-owner nonphysical initialization fails collectively");
    FiniteVolume<1> s(o);
    auto wrong = [](double, const auto &, const auto &, auto & force) {
        if (FML::ThisTask == 0)
            force.clear();
    };
    check(FML::TESTS::throws([&] { s.advance(.01, wrong); }), "one-owner gravity shape error fails collectively");
    auto nan = [](double, const auto &, const auto &, auto & force) {
        if (FML::ThisTask == 0 && !force.empty())
            force[0][0] = NAN;
    };
    check(FML::TESTS::throws([&] { s.advance(.01, nan); }), "nonfinite gravity callback output fails collectively");
    auto state = s.cells();
    if (FML::ThisTask == 0 && !state.empty())
        state[0][0] = -1;
    check(FML::TESTS::throws([&] { s.set_state(state); }),
          "one-owner replacement shape/state is validated before mutation");
    auto expand = [](double dt) {
        Options<3> opts;
        opts.shape = {4, 3, 2};
        FiniteVolume<3> s(
            opts,
            [](auto) {
                return Primitive<3>{1, {.1, 0, 0}, 1};
            },
            [](double t) {
                return Expansion{std::exp(.2 * t), .2};
            });
        for (int i = 0; i < int(std::round(.1 / dt)); ++i)
            s.step(dt);
        double e = 0;
        for (const auto & p : s.primitive_cells()) {
            e = std::max(e, std::abs(p.pressure - std::exp(-.4 * .1)));
            e = std::max(e, std::abs(p.velocity[0] - .1 * std::exp(-.2 * .1)));
        }
        return maximum(e);
    };
    const double e1 = expand(.02), e2 = expand(.01);
    check(e2 < 2e-7 && e1 / e2 > 3.7 && e1 / e2 < 4.3,
          "comoving homogeneous drag/cooling reproduce analytic second-order solution");
    if (FML::ThisTask == 0)
        std::cout << "EXPANSION " << e1 << ' ' << e2 << '\n';
}
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    std::cout << std::setprecision(17);
    try {
        riemann_tests();
        halo_tests();
        corner_halo_test();
        uniform_test<1>();
        uniform_test<2>();
        uniform_test<3>();
        evolution_tests();
        safety_and_sources();
        return FML::TESTS::summary("finite-volume hydrodynamics");
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(FML::communicator, 1);
#endif
        return 1;
    }
}
