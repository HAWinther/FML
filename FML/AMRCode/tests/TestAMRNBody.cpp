#include "../AMRNBody.h"

#include <cstring>
#include <random>

//==========================================================================================
// Tests of AMRNBody:
//
//  deposit    Mass assignment on a refined grid: the total mass is conserved, the result does not
//             depend on the number of tasks (checksum), and a uniform lattice gives delta = 0
//  blob       Force from a Gaussian blob of particles (refined around it) against the exact
//             radial force M(<r) / (4 pi r^2) - r / 3 (the -r/3 from the uniform background of
//             the periodic Poisson equation; the image terms vanish to second order by symmetry)
//  zeldovich  A plane wave evolved from z = 49 to just before shell crossing, against the exact
//             Zel'dovich solution, with and without subcycling (refining on the pancake)
//
// Run: mpirun -np N ./testamrnbody [deposit|blob|zeldovich|all]
//==========================================================================================

using namespace FML::NBODY::AMR;
using Grid = AMRNBodySimulation::Grid;
using Particles = AMRNBodySimulation::Particles;

static int nfail = 0;
static void check(bool ok, const std::string & what) {
    nfail += not ok;
    if (FML::ThisTask == 0)
        std::cout << (ok ? "  OK    " : "  FAIL  ") << what << "\n";
}

// Deterministic uniform deviates per (particle, stream), independent of the number of tasks
static double uniform(uint64_t i, uint64_t stream) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + stream * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return (double(x >> 11) + 0.5) / double(1ull << 53);
}

// Particles with ids i = ThisTask, ThisTask + NTasks, ... made by f(i, particle)
static void make_particles(Particles & part, long long ntot, const std::function<void(long long, AMRParticle &)> & f) {
    std::vector<AMRParticle> v;
    for (long long i = FML::ThisTask; i < ntot; i += FML::NTasks) {
        AMRParticle p{};
        p.id = i;
        f(i, p);
        v.push_back(p);
    }
    part.create(v, size_t(3 * ntot / FML::NTasks + 1000), FML::PARTICLE::Input::Distributed);
}

//==========================================================================================
void test_deposit() {
    if (FML::ThisTask == 0)
        std::cout << "\nDeposit\n";
    for (int cic = 0; cic < 2; cic++) {
        // Clustered particles: half uniform, half in a small cube (forces refinement)
        const long long ntot = 200000;
        Particles part;
        make_particles(part, ntot, [](long long i, AMRParticle & p) {
            for (int d = 0; d < 3; d++)
                p.Pos[d] = i % 2 ? uniform(i, d) : 0.3 + 0.08 * uniform(i, d);
        });
        Grid grid(4, 9, {true, true, true});
        part.set_domain(grid.get_domain());
        refine_on_particle_count(grid, part, 8.0, 9);
        const double m = 1.0 / double(ntot);
        auto mass = assign_mass_to_leaves(grid, part, m, cic == 1);
        double tot = 0.0, sumsq = 0.0;
        for (IndexInt i = 0; i < grid.get_n_local(); i++) {
            tot += mass[i];
            sumsq += mass[i] * mass[i] * double(grid.get_cell(i).key % 997);
        }
        FML::SumOverTasks(&tot);
        FML::SumOverTasks(&sumsq);
        auto nl = grid.get_n_global();
        if (FML::ThisTask == 0)
            std::cout << "  " << (cic ? "CIC" : "NGP") << ": " << nl << " leaves, total mass - 1 = " << tot - 1.0
                      << ", checksum " << std::setprecision(15) << sumsq << std::setprecision(6) << "\n";
        check(std::abs(tot - 1.0) < 1e-12, std::string(cic ? "CIC" : "NGP") + " mass conserved");
    }

    // A lattice with one particle per cell centre on a refined grid: delta = 0 with CIC on the uniform part.
    // (on a refined region a lattice is not uniform at the finer level, so only check the base level)
    {
        const int n1 = 32;
        const long long ntot = (long long)n1 * n1 * n1;
        Particles part;
        make_particles(part, ntot, [n1](long long i, AMRParticle & p) {
            p.Pos[0] = (i % n1 + 0.5) / n1;
            p.Pos[1] = ((i / n1) % n1 + 0.5) / n1;
            p.Pos[2] = (i / (n1 * n1) + 0.5) / n1;
        });
        Grid grid(5, 5, {true, true, true});
        part.set_domain(grid.get_domain());
        auto mass = assign_mass_to_leaves(grid, part, 1.0 / double(ntot), true);
        double maxdev = 0.0;
        for (IndexInt i = 0; i < grid.get_n_local(); i++)
            maxdev = std::max(maxdev, std::abs(mass[i] / std::ldexp(1.0, -15) - 1.0));
        FML::MaxOverTasks(&maxdev);
        check(maxdev < 1e-12, "lattice gives delta = 0 (max |delta| = " + std::to_string(maxdev) + ")");
    }
}

//==========================================================================================
void test_blob() {
    if (FML::ThisTask == 0)
        std::cout << "\nForce of a Gaussian blob\n";
    const long long ntot = 1 << 18;
    const double sigma = 0.02;
    const std::array<double, 3> c0{0.5, 0.5, 0.5};
    // Exact radial force -dphi/dr for D^2 phi = rho - 1, rho = Gaussian of total mass 1
    auto exact = [&](double r) {
        const double x = r / (std::sqrt(2.0) * sigma);
        const double M = std::erf(x) - std::sqrt(2.0 / M_PI) * (r / sigma) * std::exp(-x * x);
        return -(M / (4.0 * M_PI * r * r) - r / 3.0);
    };
    for (int lmax : {5, 7, 9}) {
        Particles part;
        make_particles(part, ntot, [&](long long i, AMRParticle & p) {
            for (int d = 0; d < 3; d++) {
                // Box-Muller
                const double u1 = uniform(i, 2 * d), u2 = uniform(i, 2 * d + 1);
                p.Pos[d] = c0[d] + sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            }
        });
        AMRNBodyParameters par;
        par.levelmin = 5;
        par.levelmax = lmax;
        par.n_refine = 32.0;
        par.mg_epsilon = 1e-8;
        AMRNBodySimulation sim(par, part, 1.0);
        // Error of the radial force relative to the exact one, in shells (particles at r in [r1, r2])
        const std::vector<double> rb{0.01, 0.02, 0.04, 0.08};
        std::vector<double> err2(rb.size() - 1, 0.0), ref2(rb.size() - 1, 0.0), tang2(rb.size() - 1, 0.0);
        for (size_t i = 0; i < part.get_npart(); i++) {
            double x[3], r = 0.0;
            for (int d = 0; d < 3; d++) {
                x[d] = part[i].Pos[d] - c0[d];
                r += x[d] * x[d];
            }
            r = std::sqrt(r);
            for (size_t b = 0; b + 1 < rb.size(); b++) {
                if (r < rb[b] or r >= rb[b + 1])
                    continue;
                double fr = 0.0, f2 = 0.0;
                for (int d = 0; d < 3; d++) {
                    fr += part[i].Acc[d] * x[d] / r;
                    f2 += part[i].Acc[d] * part[i].Acc[d];
                }
                const double fe = exact(r);
                err2[b] += (fr - fe) * (fr - fe);
                ref2[b] += fe * fe;
                tang2[b] += std::max(f2 - fr * fr, 0.0);
            }
        }
        auto nl = sim.leaves_per_level();
        int finest = 0;
        for (int l = 0; l < int(nl.size()); l++)
            if (nl[l] > 0)
                finest = l;
        for (size_t b = 0; b + 1 < rb.size(); b++) {
            FML::SumOverTasks(&err2[b]);
            FML::SumOverTasks(&ref2[b]);
            FML::SumOverTasks(&tang2[b]);
        }
        const auto nleaves = sim.grid.get_n_global(); // collective
        if (FML::ThisTask == 0) {
            std::cout << "  levels 5-" << finest << " (finest cell " << std::ldexp(1.0, -finest) / sigma << " sigma), "
                      << nleaves << " leaves. rms radial force error / exact force:";
            for (size_t b = 0; b + 1 < rb.size(); b++)
                std::cout << "  r/sigma in [" << rb[b] / sigma << "," << rb[b + 1] / sigma << "): " << std::setprecision(3)
                          << std::sqrt(err2[b] / ref2[b]);
            std::cout << std::setprecision(6) << "\n";
        }
        // With levels to 9 (finest cell 0.1 sigma) the force must be accurate at r > sigma/2 (the particle noise of
        // 2^18 particles limits it to about 1%)
        if (lmax == 9)
            for (size_t b = 0; b + 1 < rb.size(); b++)
                check(std::sqrt(err2[b] / ref2[b]) < (b == 0 ? 0.05 : 0.03),
                      "blob force accurate in shell " + std::to_string(b));
    }
}

//==========================================================================================
// Plane wave along x: x = q - D(a) b sin(2 pi q) / (2 pi), exact until shell crossing at D(a) b = 1.
// The lattice is at the cell corners q = i/n (as FML's initial conditions): a lattice at the cell centres sits on
// the kinks of the cloud-in-cell weights, which makes the assignment non-linear in small displacements
//==========================================================================================
struct ZeldovichResult {
    double errx, errv, maxdy;
    std::vector<double> x; // Final x of the particles with id < n1 (to compare runs)
};

ZeldovichResult run_zeldovich(int levelmin, int levelmax, double n_refine, bool subcycling, int nsteps, bool verbose) {
    const double OmegaM = 0.3;
    LCDM cosmo(OmegaM);
    const double aini = 0.02, across = 1.0, aend = 0.8;
    const double b = 1.0 / cosmo.D1(across);
    const int n1 = 1 << levelmin;
    const long long ntot = (long long)n1 * n1 * n1;
    auto exact_x = [&](double q, double a) { return q - cosmo.D1(a) * b * std::sin(2 * M_PI * q) / (2 * M_PI); };
    auto exact_v = [&](double q, double a) {
        return -a * a * cosmo.E(a) * cosmo.f1(a) * cosmo.D1(a) * b * std::sin(2 * M_PI * q) / (2 * M_PI);
    };
    Particles part;
    make_particles(part, ntot, [&](long long i, AMRParticle & p) {
        const double q[3] = {double(i % n1) / n1, double((i / n1) % n1) / n1, double(i / (n1 * n1)) / n1};
        for (int d = 0; d < 3; d++)
            p.Pos[d] = q[d];
        p.Pos[0] = exact_x(q[0], aini);
        p.Vel[0] = exact_v(q[0], aini);
    });
    AMRNBodyParameters par;
    par.OmegaM = OmegaM;
    par.levelmin = levelmin;
    par.levelmax = levelmax;
    par.n_refine = n_refine;
    par.subcycling = subcycling;
    par.mg_epsilon = 1e-8;
    AMRNBodySimulation sim(par, part, aini);
    long long kicks = 0, nsub = 0;
    for (int s = 1; s <= nsteps; s++) {
        sim.step(aini * std::pow(aend / aini, double(s) / nsteps));
        kicks += sim.last.nkicks;
        nsub += sim.last.nsub;
    }
    ZeldovichResult res{0.0, 0.0, 0.0, std::vector<double>(n1, 0.0)};
    double ex2 = 0.0, ev2 = 0.0;
    const double disp = cosmo.D1(aend) * b / (2 * M_PI);
    const double vamp = aend * aend * cosmo.E(aend) * cosmo.f1(aend) * disp;
    for (size_t i = 0; i < part.get_npart(); i++) {
        const long long id = part[i].id;
        const double q0 = double(id % n1) / n1;
        double dx = part[i].Pos[0] - exact_x(q0, aend);
        dx -= std::round(dx);
        ex2 += dx * dx;
        const double dv = part[i].Vel[0] - exact_v(q0, aend);
        ev2 += dv * dv;
        double dy = part[i].Pos[1] - double((id / n1) % n1) / n1;
        dy -= std::round(dy);
        res.maxdy = std::max(res.maxdy, std::abs(dy));
        if (id < n1)
            res.x[id] = q0 + dx; // Unwrapped error-carrying position
    }
    FML::SumOverTasks(&ex2);
    FML::SumOverTasks(&ev2);
    FML::MaxOverTasks(&res.maxdy);
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, res.x.data(), n1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    res.errx = std::sqrt(ex2 / double(ntot)) / disp;
    res.errv = std::sqrt(ev2 / double(ntot)) / vamp;
    auto nl = sim.leaves_per_level();
    int finest = 0;
    for (int l = 0; l < int(nl.size()); l++)
        if (nl[l] > 0)
            finest = l;
    if (FML::ThisTask == 0 and verbose)
        std::cout << "  " << n1 << "^3 particles, levels " << levelmin << "-" << finest << ", subcycling " << subcycling
                  << ": rms error x " << res.errx << ", v " << res.errv << " (relative to the amplitudes), max |dy| "
                  << res.maxdy << ", " << nsub << " sub-steps, " << double(kicks) / double(ntot) << " kicks per particle\n";
    return res;
}

void test_zeldovich() {
    if (FML::ThisTask == 0)
        std::cout << "\nZel'dovich plane wave (z = 49 to D b = 0.88, shell crossing at D b = 1)\n";
    // Uniform grids: second order convergence of the error (the wave is resolved by 32 and 64 cells)
    auto u5 = run_zeldovich(5, 5, 8.0, false, 40, true);
    auto u6 = run_zeldovich(6, 6, 8.0, false, 40, true);
    check(u5.errx < 0.03 and u6.errx < 0.012 and u5.errx / u6.errx > 2.0, "uniform grids: the error converges with resolution");
    // Refined (more than 0.25 particles per cell: level 6 everywhere, 7 on the pancake), with and without subcycling
    auto r0 = run_zeldovich(5, 8, 0.25, false, 40, true);
    auto r1 = run_zeldovich(5, 8, 0.25, true, 40, true);
    double d2 = 0.0;
    for (size_t i = 0; i < r0.x.size(); i++)
        d2 += (r0.x[i] - r1.x[i]) * (r0.x[i] - r1.x[i]);
    LCDM cosmo(0.3);
    const double diff = std::sqrt(d2 / double(r0.x.size())) / (cosmo.D1(0.8) / cosmo.D1(1.0) / (2 * M_PI));
    if (FML::ThisTask == 0)
        std::cout << "  subcycling vs not: rms difference in x " << diff << " (relative to the amplitude)\n";
    check(r0.errx < 0.035 and r1.errx < 0.035 and r1.maxdy < 1e-4, "refined: Zel'dovich solution reproduced");
    check(diff < 0.005, "subcycling agrees with the finest common step");
}

int main(int argc, char ** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    if (which == "deposit" or which == "all")
        test_deposit();
    if (which == "blob" or which == "all")
        test_blob();
    if (which == "zeldovich" or which == "all")
        test_zeldovich();
    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail ? std::to_string(nfail) + " tests FAILED" : "All tests passed") << "\n";
    return nfail ? 1 : 0;
}
