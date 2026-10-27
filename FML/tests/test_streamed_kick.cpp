// Streamed kicks against the previous buffered gather/update, with exact velocity and diagnostic comparisons.
#include <FML/NBody/NBody.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace I = FML::INTERPOLATION;
int failures = 0;
void check(bool ok, const std::string & label) {
    int bad = !ok; FML::MaxOverTasks(&bad); failures += bad;
    if (FML::ThisTask == 0) std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}

template <int N, class Position, class Velocity>
struct Particle {
    Position pos[N];
    Velocity vel[N];
    uint64_t tag;
    const Position * get_pos() const { return pos; }
    Velocity * get_vel() { return vel; }
    constexpr int get_ndim() const { return N; }
};

template <int N, class P>
std::vector<P> make_particles(bool empty_rank, bool all_empty) {
    std::vector<P> p(all_empty || (empty_rank && FML::NTasks > 1 && FML::ThisTask == 0) ? 0 : 1200);
    using Pos = std::remove_cv_t<std::remove_reference_t<decltype(P{}.pos[0])>>;
    std::mt19937 rng(253 + FML::ThisTask);
    std::uniform_real_distribution<double> u(0,1);
    for (size_t i = 0; i < p.size(); i++) {
        for (int d = 0; d < N; d++) {
            const Pos lo = d == 0 ? Pos(FML::xmin_domain) : Pos(0);
            const Pos hi = d == 0 ? Pos(FML::xmax_domain) : Pos(1);
            Pos x = Pos(double(lo) + (double(hi) - lo) * u(rng));
            if (i % 7 == 0) x = lo;
            if (i % 7 == 1 || x >= hi) x = std::nextafter(hi, lo);
            p[i].pos[d] = x;
            p[i].vel[d] = 2.0 * u(rng) - 1.0;
        }
        p[i].tag = 1234567 + i;
    }
    return p;
}

template <int N>
std::array<FML::GRID::FFTWGrid<N>, N> make_force(const std::string & method) {
    auto halo = I::get_extra_slices_needed_for_density_assignment(method);
    std::array<FML::GRID::FFTWGrid<N>, N> force;
    for (int d = 0; d < N; d++) {
        force[d] = FML::GRID::FFTWGrid<N>(24, halo.first, halo.second);
        force[d].fill_real_grid([d](std::array<double,N> & x) {
            double value = 0.17 * (d + 1);
            for (int j = 0; j < N; j++)
                value += std::sin(2 * M_PI * (j + 1) * x[j]) + 0.2 * std::cos(2 * M_PI * x[j] * (d + 1));
            return FML::GRID::FloatType(value);
        });
    }
    return force;
}

template <int N>
uint64_t grid_hash(const std::array<FML::GRID::FFTWGrid<N>,N> & force) {
    uint64_t hash = 1469598103934665603ull;
    for (const auto & grid : force)
        for (auto i : grid.get_real_range()) {
            const auto value = grid.get_real_from_index(i);
            const auto * bytes = reinterpret_cast<const unsigned char *>(&value);
            for (size_t j = 0; j < sizeof(value); j++) hash = (hash ^ bytes[j]) * 1099511628211ull;
        }
    return hash;
}

// The old two-pass kick is retained here as the reference. It uses the public buffered gather API.
template <int N, class P>
double buffered_kick(std::array<FML::GRID::FFTWGrid<N>,N> & force, std::vector<P> & p,
                     double dt, const std::string & method) {
    if (dt == 0) return 0;
    for (auto & grid : force) grid.communicate_boundaries();
    std::array<std::vector<FML::GRID::FloatType>, N> values;
    I::interpolate_grid_vector_to_particle_positions<N, P>(force, p.data(), p.size(), values, method);
    double max_change = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(max : max_change)
#endif
    for (size_t i = 0; i < p.size(); i++)
        for (int d = 0; d < N; d++) {
            double dv = -values[d][i] * dt;
            max_change = std::max(max_change, std::abs(dv));
            p[i].vel[d] += dv;
        }
    FML::MaxOverTasks(&max_change);
    return max_change;
}

template <int N, class P>
void run_case(const std::string & label, bool empty_rank = false, bool all_empty = false) {
    const auto original = make_particles<N,P>(empty_rank, all_empty);
    bool velocities = true, positions = true, diagnostics = true, grids = true;
    double worst = 0;
    for (const std::string method : {"NGP", "CIC", "TSC", "PCS", "PQS"}) {
        auto force = make_force<N>(method);
        const uint64_t before = grid_hash<N>(force);
        for (double dt : {0.0, 0.125, -0.333, 1e-12}) {
            auto expected = original, actual = original;
            const double max_change = buffered_kick<N>(force, expected, dt, method);
            std::ostringstream output, expected_output;
            expected_output.copyfmt(std::cout);
            if (dt != 0 && FML::ThisTask == 0) expected_output << "[Kick] Max delta_vel : " << max_change << '\n';
            auto * previous = std::cout.rdbuf(output.rdbuf());
            FML::NBODY::KickParticles<N,P>(force, actual.data(), actual.size(), dt, method);
            std::cout.rdbuf(previous);
            diagnostics = diagnostics && output.str() == expected_output.str();
            for (size_t i = 0; i < actual.size(); i++) {
                velocities = velocities && std::memcmp(actual[i].vel, expected[i].vel, sizeof(actual[i].vel)) == 0;
                positions = positions && std::memcmp(actual[i].pos, original[i].pos, sizeof(actual[i].pos)) == 0 &&
                            actual[i].tag == original[i].tag;
                for (int d = 0; d < N; d++) worst = std::max(worst, std::abs(double(actual[i].vel[d]) - expected[i].vel[d]));
            }
            grids = grids && grid_hash<N>(force) == before;
        }
    }
    FML::MaxOverTasks(&worst);
    check(velocities, label + " velocities match buffered kicks bit-for-bit");
    check(positions, label + " positions and metadata preserved bit-for-bit");
    check(diagnostics, label + " maximum-velocity-change diagnostics unchanged");
    check(grids, label + " force-grid interiors unchanged");
    if (FML::ThisTask == 0) std::cout << "  maximum velocity difference: " << worst << '\n';
}

struct ContainerParticle : Particle<3,double,double> {
    double * get_pos() { return pos; }
};

void test_container() {
    using P = ContainerParticle;
    auto original = make_particles<3,P>(true, false);
    auto force = make_force<3>("CIC");
    FML::PARTICLE::MPIParticles<P> part;
    part.create(original.data(), original.size(), original.size()+16, FML::xmin_domain, FML::xmax_domain, false);
    std::vector<P> expected;
    if (part.get_npart()) expected.assign(part.get_particles_ptr(), part.get_particles_ptr() + part.get_npart());
    buffered_kick<3>(force, expected, 0.25, "CIC");
    FML::NBODY::KickParticles<3,P>(force, part, 0.25, "CIC");
    bool same = part.get_npart() == expected.size();
    const auto * p = part.get_particles_ptr();
    for (size_t i = 0; same && i < expected.size(); i++)
        same = std::memcmp(p[i].vel, expected[i].vel, sizeof(p[i].vel)) == 0;
    check(same, "MPIParticles wrapper matches buffered kicks, including an empty rank");
    // A zero step remains a no-op even with no allocated force grids and a null particle pointer.
    std::array<FML::GRID::FFTWGrid<3>,3> unallocated;
    FML::NBODY::KickParticles<3,P>(unallocated, static_cast<P*>(nullptr), 0, 0, "CIC");
    check(true, "zero timestep accepts null particles and unallocated grids as before");
}

int main() {
    run_case<2, Particle<2,double,double>>("2D double");
    run_case<3, Particle<3,double,double>>("3D double");
    run_case<3, Particle<3,float,float>>("3D float position/velocity");
    run_case<3, Particle<3,float,double>>("3D float position/double velocity", true);
    run_case<3, Particle<3,double,float>>("3D double position/float velocity", true);
    run_case<3, Particle<3,double,double>>("all ranks empty", false, true);
    test_container();
    if (FML::ThisTask == 0) std::cout << "Streamed-kick failures: " << failures << '\n';
    return failures ? 1 : 0;
}
