#include "../TreePM.h"

#include <iomanip>
#include <iostream>
#include <string>

//==========================================================================================
// Tests of TreePMCode:
//
//  blob       Force of a Gaussian blob of particles against the exact periodic radial force M(<r) / (4 pi r^2) - r / 3
//             (the -r/3 from the uniform background of the periodic Poisson equation); tree against an exact tree
//             (opening angle 0) and the cell list
//  zeldovich  A plane wave from z = 49 to just before shell crossing against the exact Zel'dovich solution
//  twofluid   CDM and baryons as two sets of particles on a plane wave: LCDM (both on the Zel'dovich solution) and
//             coupled quintessence (beta = 0.3: the split of CDM and baryons against the linear two-fluid growth)
//  gadget     Individual (GADGET-2) time steps: the plane wave against Zel'dovich and 160 common steps, two fluids
//
// Run: mpirun -np N ./testtreepm [blob|zeldovich|twofluid|gadget|all]
//==========================================================================================

using namespace FML::NBODY::TREEPM;
using Particles = TreePMSimulation::Particles;

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

// Particles with ids i = ThisTask, ThisTask + NTasks, ... made by f(i, particle), on the slab domain
static void make_particles(Particles & part, long long ntot, const std::function<void(long long, TreePMParticle &)> & f) {
    std::vector<TreePMParticle> v;
    for (long long i = FML::ThisTask; i < ntot; i += FML::NTasks) {
        TreePMParticle p{};
        p.id = i;
        p.mass = 1.0;
        f(i, p);
        for (int d = 0; d < 3; d++) {
            p.Pos[d] -= std::floor(p.Pos[d]);
            if (p.Pos[d] >= 1.0)
                p.Pos[d] = 0.0;
        }
        v.push_back(p);
    }
    part.create(v, size_t(3 * ntot / FML::NTasks + 1000), FML::PARTICLE::Input::Distributed);
}

//==========================================================================================
// Gaussian blob
//==========================================================================================
struct BlobErrors {
    std::vector<double> rel;          // rms radial force error / exact in shells
    std::vector<std::array<double, 3>> acc; // per particle (ordered by id) for comparisons
};

BlobErrors blob_force(const std::string & backend, double opening, int Nmesh, bool verbose, long long ntot = 1 << 17) {
    const double sigma = 0.02;
    const std::array<double, 3> c0{0.5, 0.5, 0.5};
    auto exact = [&](double r) {
        const double x = r / (std::sqrt(2.0) * sigma);
        const double M = std::erf(x) - std::sqrt(2.0 / M_PI) * (r / sigma) * std::exp(-x * x);
        return -(M / (4.0 * M_PI * r * r) - r / 3.0);
    };
    Particles part;
    make_particles(part, ntot, [&](long long i, TreePMParticle & p) {
        for (int d = 0; d < 3; d++) {
            const double u1 = uniform(i, 2 * d), u2 = uniform(i, 2 * d + 1);
            p.Pos[d] = c0[d] + sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        }
    });
    TreePMParameters par;
    par.Nmesh = Nmesh;
    par.Npart_1D = 64;
    par.backend = backend;
    par.opening_angle = opening;
    par.cutoff_tolerance = 1e-6;
    TreePMSimulation sim(par, part, 1.0);
    const std::vector<double> rb{0.01, 0.02, 0.04, 0.08, 0.16};
    std::vector<double> err2(rb.size() - 1, 0.0), ref2(rb.size() - 1, 0.0);
    BlobErrors out;
    out.acc.assign(ntot, {0, 0, 0});
    for (size_t i = 0; i < part.get_npart(); i++) {
        double x[3], r = 0.0;
        for (int d = 0; d < 3; d++) {
            x[d] = part[i].Pos[d] - c0[d];
            r += x[d] * x[d];
        }
        r = std::sqrt(r);
        for (int d = 0; d < 3; d++)
            out.acc[part[i].id][d] = part[i].Acc[d];
        for (size_t b = 0; b + 1 < rb.size(); b++) {
            if (r < rb[b] or r >= rb[b + 1])
                continue;
            double fr = 0.0;
            for (int d = 0; d < 3; d++)
                fr += part[i].Acc[d] * x[d] / r;
            const double fe = exact(r);
            err2[b] += (fr - fe) * (fr - fe);
            ref2[b] += fe * fe;
        }
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, out.acc.data(), int(3 * ntot), MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    for (size_t b = 0; b + 1 < rb.size(); b++) {
        FML::SumOverTasks(&err2[b]);
        FML::SumOverTasks(&ref2[b]);
        out.rel.push_back(std::sqrt(err2[b] / ref2[b]));
    }
    if (FML::ThisTask == 0 and verbose) {
        std::cout << "  " << ntot << " particles, " << backend << " (opening " << opening << "), PM " << Nmesh
                  << "^3: rms radial force error / exact:";
        for (size_t b = 0; b + 1 < rb.size(); b++)
            std::cout << "  r/sigma [" << rb[b] / sigma << "," << rb[b + 1] / sigma << "): " << std::setprecision(3)
                      << out.rel[b];
        std::cout << std::setprecision(6) << "\n";
    }
    return out;
}

static double rel_diff(const std::vector<std::array<double, 3>> & a, const std::vector<std::array<double, 3>> & b) {
    double d2 = 0, r2 = 0;
    for (size_t i = 0; i < a.size(); i++)
        for (int d = 0; d < 3; d++) {
            d2 += (a[i][d] - b[i][d]) * (a[i][d] - b[i][d]);
            r2 += b[i][d] * b[i][d];
        }
    return std::sqrt(d2 / r2);
}

void test_blob() {
    if (FML::ThisTask == 0)
        std::cout << "\nForce of a Gaussian blob (sigma = 0.02)\n";
    // Accuracy against the exact force (2^17 particles: about 1% particle noise)
    const auto tree = blob_force("tree", 0.5, 64, true);
    for (size_t b = 0; b < tree.rel.size(); b++)
        check(tree.rel[b] < (b == 0 ? 0.05 : 0.03), "blob force accurate in shell " + std::to_string(b));
    // The backends and the PM grid on a small blob (an exact tree is a direct sum here)
    const long long nsmall = 1 << 13;
    const auto exact_tree = blob_force("tree", 0.0, 64, false, nsmall);
    const auto cell = blob_force("cell", 0.0, 64, false, nsmall);
    const auto approx = blob_force("tree", 0.5, 64, false, nsmall);
    const auto coarse = blob_force("tree", 0.0, 32, false, nsmall);
    const double dc = rel_diff(cell.acc, exact_tree.acc), dt = rel_diff(approx.acc, exact_tree.acc),
                 dm = rel_diff(coarse.acc, exact_tree.acc);
    if (FML::ThisTask == 0)
        std::cout << "  " << nsmall << " particles, rms difference to the exact tree: cell list " << dc
                  << ", tree (opening 0.5) " << dt << ", PM 32^3 instead of 64^3 " << dm << "\n";
    check(dc < 1e-10, "cell list = exact tree");
    check(dt < 5e-3, "tree with opening 0.5 within 0.5% of the exact tree");
    check(dm < 5e-3, "the force does not depend on the PM grid (the split moves with it)");
}

//==========================================================================================
// Plane wave along x: x = q - D(a) b sin(2 pi q) / (2 pi), exact until shell crossing at D(a) b = 1. With CDM and
// baryons (two fluids) both species start on the same wave, and with coupled quintessence they follow the linear
// two-fluid growth (small amplitude). The softening is 1/4 of the particle spacing: near shell crossing the sheets of
// particles come closer than the spacing within a sheet, and with GADGET-like softening (1/30) the discrete lattice
// forces between them are not smoothed (velocities 20% off the continuous solution at D b = 0.88, 2.5% with 1/4)
//==========================================================================================
struct WaveResult {
    double errx{0}, errv{0};
    double amp[2]{0, 0}, lin[2]{0, 0};
    std::vector<double> x;
};

WaveResult run_wave(int n1, int nsteps, bool baryons, const std::string & model, double beta, double Db_end,
                    const std::string & backend = "tree", double softening = 0.25, double asmth = 1.25,
                    const std::string & timestep_method = "fixed", int * finest_bin = nullptr) {
    TreePMParameters par;
    par.OmegaM = 0.3;
    par.OmegaB = 0.05;
    par.gravity_model = model;
    par.cq_lambda = 0.5;
    par.cq_beta = beta;
    par.baryons = baryons;
    par.Nmesh = 2 * n1;
    par.Npart_1D = n1;
    par.backend = backend;
    par.softening = softening;
    par.asmth = asmth;
    par.timestep_method = timestep_method;
    const Cosmology cosmo = make_cosmology(par);
    const double aini = 0.02, aend = 0.8;
    const double b = Db_end / cosmo.D1(aend);
    const long long N = (long long)n1 * n1 * n1, ntot = baryons ? 2 * N : N;

    // The linear two-fluid growth from aini (RK4 in log a): {D_c, D_c', D_b, D_b'}
    auto rhs = [&](double x, const std::array<double, 4> & y) {
        const double a = std::exp(x), fc = cosmo.cdm_fraction(a);
        const double om = 1.5 * cosmo.OmegaM_of_a(a) * cosmo.poisson_factor(a, Fluid::CDM), h = 2.0 + cosmo.dlogEdloga(a);
        const double sb = om * (fc * y[0] + (1.0 - fc) * y[2]), sc = sb + om * cosmo.cdm_fifth_force(a) * y[0];
        return std::array<double, 4>{y[1], -(h + cosmo.friction(a, Fluid::CDM)) * y[1] + sc, y[3], -h * y[3] + sb};
    };
    const double D0 = cosmo.D1(aini), f0 = cosmo.f1(aini);
    std::array<double, 4> y{D0, f0 * D0, D0, f0 * D0};
    {
        const int n = 2000;
        const double x1 = std::log(aini), dx = (std::log(aend) - x1) / n;
        for (int i = 0; i < n; i++) {
            const double x = x1 + i * dx;
            auto add = [](const std::array<double, 4> & p, const std::array<double, 4> & q, double c) {
                std::array<double, 4> r;
                for (int k = 0; k < 4; k++)
                    r[k] = p[k] + c * q[k];
                return r;
            };
            auto k1 = rhs(x, y), k2 = rhs(x + 0.5 * dx, add(y, k1, 0.5 * dx)), k3 = rhs(x + 0.5 * dx, add(y, k2, 0.5 * dx)),
                 k4 = rhs(x + dx, add(y, k3, dx));
            for (int k = 0; k < 4; k++)
                y[k] += dx / 6.0 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]);
        }
    }
    // One fluid: the exact Zel'dovich solution with the model's growth
    const double Dend[2] = {baryons ? y[0] : cosmo.D1(aend), baryons ? y[2] : cosmo.D1(aend)};
    const double fend[2] = {baryons ? y[1] / y[0] : cosmo.f1(aend), baryons ? y[3] / y[2] : cosmo.f1(aend)};

    Particles part;
    make_particles(part, ntot, [&](long long i, TreePMParticle & p) {
        const long long j = i % N;
        p.species = int(i / N);
        const double q[3] = {double(j % n1) / n1, double((j / n1) % n1) / n1, double(j / (n1 * n1)) / n1};
        for (int d = 0; d < 3; d++)
            p.Pos[d] = q[d];
        const double s = std::sin(2 * M_PI * q[0]) / (2 * M_PI);
        p.Pos[0] = q[0] - D0 * b * s;
        p.Vel[0] = -aini * aini * cosmo.E(aini) * f0 * D0 * b * s;
    });
    TreePMSimulation sim(par, part, aini);
    int finest = 0;
    for (int st = 1; st <= nsteps; st++) {
        sim.step(aini * std::pow(aend / aini, double(st) / nsteps));
        finest = std::max(finest, sim.last.finest_bin);
    }
    if (finest_bin)
        *finest_bin = finest;

    WaveResult r;
    r.x.assign(n1, 0.0);
    double num[2] = {0, 0}, den[2] = {0, 0}, ex2[2] = {0, 0}, ev2[2] = {0, 0}, vden[2] = {0, 0};
    for (size_t i = 0; i < part.get_npart(); i++) {
        const int sp = part[i].species;
        const long long j = part[i].id % N;
        const double q0 = double(j % n1) / n1;
        double dx = part[i].Pos[0] - q0;
        dx -= std::round(dx);
        const double s = -b * std::sin(2 * M_PI * q0) / (2 * M_PI);
        num[sp] += dx * s;
        den[sp] += s * s;
        ex2[sp] += (dx - Dend[sp] * s) * (dx - Dend[sp] * s);
        const double vex = aend * aend * cosmo.E(aend) * fend[sp] * Dend[sp] * s;
        ev2[sp] += (part[i].Vel[0] - vex) * (part[i].Vel[0] - vex);
        vden[sp] += vex * vex;
        if (part[i].id < n1)
            r.x[part[i].id] = q0 + dx;
    }
    for (int sp = 0; sp < 2; sp++) {
        FML::SumOverTasks(&num[sp]);
        FML::SumOverTasks(&den[sp]);
        FML::SumOverTasks(&ex2[sp]);
        FML::SumOverTasks(&ev2[sp]);
        FML::SumOverTasks(&vden[sp]);
    }
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, r.x.data(), n1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
    const double D1end = cosmo.D1(aend);
    for (int sp = 0; sp < (baryons ? 2 : 1); sp++) {
        r.amp[sp] = num[sp] / den[sp] / D1end;
        r.lin[sp] = Dend[sp] / D1end;
    }
    r.errx = std::sqrt(ex2[0] / den[0]) / Dend[0];
    r.errv = std::sqrt(ev2[0] / vden[0]);
    if (FML::ThisTask == 0)
        std::cout << "  " << model << (baryons ? " (CDM + baryons)" : "") << ", " << timestep_method << ", " << n1 << "^3, " << nsteps
                  << " steps, D b = " << Db_end << ": rms error x " << r.errx << ", v " << r.errv << "; D_c / D = " << r.amp[0]
                  << " (expected " << r.lin[0] << ")"
                  << (baryons ? ", D_b / D = " + std::to_string(r.amp[1]) + " (expected " + std::to_string(r.lin[1]) + ")" : "")
                  << "\n";
    return r;
}

void test_zeldovich() {
    if (FML::ThisTask == 0)
        std::cout << "\nZel'dovich plane wave (z = 49 to D b = 0.88; shell crossing at D b = 1)\n";
    auto w = run_wave(32, 40, false, "LCDM", 0.0, 0.88);
    check(w.errx < 0.03 and w.errv < 0.04 and std::abs(w.amp[0] - 1.0) < 3e-3, "plane wave follows the Zel'dovich solution");
    auto c = run_wave(32, 40, false, "LCDM", 0.0, 0.88, "cell");
    double d2 = 0;
    for (size_t i = 0; i < w.x.size(); i++)
        d2 += (w.x[i] - c.x[i]) * (w.x[i] - c.x[i]);
    const double diff = std::sqrt(d2 / w.x.size()) / (0.88 / (2 * M_PI));
    if (FML::ThisTask == 0)
        std::cout << "  tree vs cell list: rms difference in x " << diff << " of the amplitude\n";
    check(diff < 1e-3, "tree and cell list give the same evolution");
    // A checksum of the positions (compare across numbers of tasks)
    double sum = 0;
    for (auto x : w.x)
        sum += x * x;
    if (FML::ThisTask == 0)
        std::cout << "  checksum of the final x of the first row: " << std::setprecision(15) << sum << std::setprecision(6)
                  << "\n";
}

void test_two_fluid() {
    // Softening of one particle spacing: CDM and baryons start from the same lattice, and a force that resolves the
    // pairs (softening 1/4 of the spacing) pulls each baryon back to its CDM partner, which damps the CDM-baryon split
    // the fifth force builds up (D_b / D_c = 0.916 instead of 0.785 here, the same with r_s = 2 cells); with the force
    // smoothed over the spacing the split is that of the two-fluid equations
    if (FML::ThisTask == 0)
        std::cout << "\nCDM and baryons as two sets of particles (plane wave, D b = 0.15 at a = 0.8, softening 1 spacing)\n";
    auto l = run_wave(16, 40, true, "LCDM", 0.0, 0.15, "tree", 1.0);
    check(std::abs(l.amp[1] / l.amp[0] - 1.0) < 1e-6, "LCDM: both species move together");
    auto c = run_wave(16, 40, true, "CoupledQuintessence", 0.3, 0.15, "tree", 1.0);
    const double split = c.amp[1] / c.amp[0], split_lin = c.lin[1] / c.lin[0];
    if (FML::ThisTask == 0)
        std::cout << "  D_b / D_c = " << split << " (linear two-fluid " << split_lin << "); relative to LCDM: D_c "
                  << c.amp[0] / l.amp[0] << " (" << c.lin[0] << "), D_b " << c.amp[1] / l.amp[1] << " (" << c.lin[1] << ")\n";
    // (the softened force at the scale of the wave, 0.91 of the growth here, does not cancel exactly in the ratio)
    check(std::abs(c.amp[0] / l.amp[0] / c.lin[0] - 1.0) < 1e-2 and std::abs(c.amp[1] / l.amp[1] / c.lin[1] - 1.0) < 1e-2,
          "coupled quintessence: each species follows the linear two-fluid growth (relative to LCDM, to 1%)");
    check(std::abs(split / split_lin - 1.0) < 2e-3, "coupled quintessence: the baryon-CDM split agrees with linear theory");
    auto d = run_wave(16, 40, true, "CovariantDarkScattering", 0.0, 0.15, "tree", 1.0);
    const double dsplit = d.amp[1] / d.amp[0], dsplit_lin = d.lin[1] / d.lin[0];
    if (FML::ThisTask == 0)
        std::cout << "  covariant dark scattering: D_b / D_c = " << dsplit << " (linear two-fluid " << dsplit_lin << ")\n";
    check(std::abs(dsplit / dsplit_lin - 1.0) < 2e-3, "covariant dark scattering: the baryon-CDM split agrees with linear theory");
}

//==========================================================================================
// GADGET-2 time steps: the plane wave with 40 PM steps and individual short-range steps (the sheets near shell crossing
// get finer bins) against the Zel'dovich solution and the run with 160 common steps, and the two-fluid wave
//==========================================================================================
void test_gadget_steps() {
    if (FML::ThisTask == 0)
        std::cout << "\nIndividual (GADGET-2) time steps\n";
    int finest = 0;
    auto g = run_wave(32, 40, false, "LCDM", 0.0, 0.88, "tree", 0.25, 1.25, "gadget", &finest);
    auto f = run_wave(32, 160, false, "LCDM", 0.0, 0.88, "tree", 0.25, 1.25, "fixed");
    double d2 = 0;
    for (size_t i = 0; i < g.x.size(); i++)
        d2 += (g.x[i] - f.x[i]) * (g.x[i] - f.x[i]);
    const double diff = std::sqrt(d2 / g.x.size()) / (0.88 / (2 * M_PI));
    if (FML::ThisTask == 0)
        std::cout << "  finest bin " << finest << "; rms difference in x to 160 common steps " << diff << " of the amplitude\n";
    check(finest >= 1, "the sheets near shell crossing get finer short-range bins");
    check(g.errx < 0.03 and g.errv < 0.04 and std::abs(g.amp[0] - 1.0) < 3e-3, "plane wave follows the Zel'dovich solution");
    check(diff < 3e-3, "individual steps agree with 160 common steps");
    auto l = run_wave(16, 40, true, "LCDM", 0.0, 0.15, "tree", 1.0, 1.25, "gadget");
    auto c = run_wave(16, 40, true, "CoupledQuintessence", 0.3, 0.15, "tree", 1.0, 1.25, "gadget");
    const double split = c.amp[1] / c.amp[0], split_lin = c.lin[1] / c.lin[0];
    if (FML::ThisTask == 0)
        std::cout << "  two fluids: D_b / D_c = " << split << " (linear two-fluid " << split_lin << ")\n";
    check(std::abs(l.amp[1] / l.amp[0] - 1.0) < 1e-6 and std::abs(split / split_lin - 1.0) < 2e-3,
          "two fluids with individual steps: the baryon-CDM split agrees with linear theory");
}

int main(int argc, char ** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    if (which == "blob" or which == "all")
        test_blob();
    if (which == "zeldovich" or which == "all")
        test_zeldovich();
    if (which == "twofluid" or which == "all")
        test_two_fluid();
    if (which == "gadget" or which == "all")
        test_gadget_steps();
    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail ? std::to_string(nfail) + " tests FAILED" : "All tests passed") << "\n";
    return nfail ? 1 : 0;
}
