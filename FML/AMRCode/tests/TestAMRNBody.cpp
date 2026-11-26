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
//  cq         Coupled quintessence / Fading Dark Sector and covariant dark scattering: LCDM limit, plane wave with
//             the model's growth
//  twofluid   CDM and baryons as two sets of particles: an LCDM plane wave (both species on the Zel'dovich
//             solution), and with coupled quintessence (beta = 0.3) the split of CDM and baryons against the linear
//             two-fluid growth
//
// Run: mpirun -np N ./testamrnbody [deposit|blob|zeldovich|cq|twofluid|all]
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

// gravity_model = CoupledQuintessence (lambda, beta): the same with that background, G_eff and friction (in one
// dimension the force is linear in the displacement until shell crossing for any scale-independent Poisson equation,
// so the Zel'dovich solution with the model's growth factor is exact). lcdm_growth: compare with the LCDM growth instead
ZeldovichResult run_zeldovich(int levelmin, int levelmax, double n_refine, bool subcycling, int nsteps, bool verbose,
                              const std::string & model = "LCDM", double lambda = 0.0, double beta = 0.0,
                              bool lcdm_growth = false, double ds_w = -0.98, double cds_alpha = 1.0) {
    const double OmegaM = 0.3;
    AMRNBodyParameters par;
    par.OmegaM = OmegaM;
    par.gravity_model = model;
    par.cq_lambda = lambda;
    par.cq_beta = beta;
    par.ds_w = ds_w;
    par.cds_alpha = cds_alpha;
    const Cosmology cosmo = lcdm_growth ? Cosmology(OmegaM) : make_cosmology(par);
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
        std::cout << "  "
                  << (model == "LCDM" ? ""
                      : model == "CovariantDarkScattering"
                          ? model + " (w " + std::to_string(ds_w) + ", alpha " + std::to_string(cds_alpha) +
                                (lcdm_growth ? ", LCDM growth" : "") + "): "
                          : model + " (lambda " + std::to_string(lambda) + ", beta " + std::to_string(beta) +
                                (lcdm_growth ? ", LCDM growth" : "") + "): ")
                  << n1 << "^3 particles, levels " << levelmin << "-" << finest << ", subcycling " << subcycling
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

//==========================================================================================
// Coupled quintessence / the Fading Dark Sector: lambda = beta = 0 is LCDM (no radiation: the same background), and
// the plane wave with a strong coupling (beta = 0.3: G_eff/G = 1.18 for the CDM) follows the model's growth factor
//==========================================================================================
void test_coupled_quintessence() {
    if (FML::ThisTask == 0)
        std::cout << "\nCoupled quintessence / Fading Dark Sector\n";
    const Cosmology lcdm(0.3), zero = Cosmology::coupled_quintessence(0.3, 0.049, 0.0, 0.0);
    double emax = 0.0;
    for (double a : {0.02, 0.1, 0.3, 0.6, 1.0}) {
        emax = std::max(emax, std::abs(zero.E(a) / lcdm.E(a) - 1.0));
        emax = std::max(emax, std::abs(zero.D1(a) / lcdm.D1(a) - 1.0));
        emax = std::max(emax, std::abs(zero.f1(a) / lcdm.f1(a) - 1.0));
        emax = std::max(emax, std::abs(zero.D2(a) / lcdm.D2(a) - 1.0));
        emax = std::max(emax, std::abs(zero.kick_factor(a, 1.1 * a) / lcdm.kick_factor(a, 1.1 * a) - 1.0));
        emax = std::max(emax, std::abs(zero.drift_factor(a, 1.1 * a) / lcdm.drift_factor(a, 1.1 * a) - 1.0));
    }
    if (FML::ThisTask == 0)
        std::cout << "  lambda = beta = 0 vs LCDM: max relative difference (E, D1, f1, D2, kick, drift) " << emax << "\n";
    check(emax < 1e-6, "coupled quintessence with lambda = beta = 0 is LCDM");

    // The Fading Dark Sector best fit (c = -0.85, c' = 0.05): G_eff/G of the CDM 1 + 2 c'^2 = 1.005
    AMRNBodyParameters par;
    par.gravity_model = "FadingDarkSector";
    const Cosmology fds = make_cosmology(par);
    if (FML::ThisTask == 0)
        std::cout << "  FDS best fit: E/E_LCDM(a = 1) " << fds.E(1.0) / lcdm.E(1.0) << ", D/D_LCDM(a = 1) "
                  << fds.D1(1.0) / lcdm.D1(1.0) << ", dphi/dlna(a = 1) " << fds.dphidlna(1.0) << ", Gamma(a = 1) "
                  << fds.friction(1.0) << ", P(a = 1) " << fds.poisson_factor(1.0) << "\n";
    check(fds.poisson_factor(1.0) > 1.0 and fds.poisson_factor(1.0) < 1.005, "FDS: G_eff/G of the one fluid in (1, 1 + 2 c'^2)");

    // Covariant dark scattering: alpha = 0, w = -1 is LCDM; the linear growth of the N-body equations (two fluids) for
    // alpha = 1, w = -0.98 from z = 49 to 0 relative to alpha = 0 is that of SimulationCode (SPECIES.md: CDM -9.13%,
    // baryons -0.32%, CDM+baryons -7.77%, with OmegaCDM = 0.2685, Omegab = 0.049)
    {
        const Cosmology z0 = Cosmology::covariant_dark_scattering(0.3, 0.049, -1.0, 0.0);
        double e = 0.0;
        for (double a : {0.02, 0.1, 0.3, 0.6, 1.0}) {
            e = std::max(e, std::abs(z0.E(a) / lcdm.E(a) - 1.0));
            e = std::max(e, std::abs(z0.D1(a) / lcdm.D1(a) - 1.0));
            e = std::max(e, std::abs(z0.kick_factor(a, 1.1 * a) / lcdm.kick_factor(a, 1.1 * a) - 1.0));
        }
        check(e < 1e-6, "covariant dark scattering with alpha = 0, w = -1 is LCDM");
        const double OM = 0.3175, OB = 0.049, aini = 1.0 / 50.0;
        auto grow = [&](const Cosmology & c) {
            auto rhs = [&](double x, const std::array<double, 4> & y) {
                const double a = std::exp(x), fc = c.cdm_fraction(a);
                const double om = 1.5 * c.OmegaM_of_a(a), h = 2.0 + c.dlogEdloga(a), sb = om * (fc * y[0] + (1 - fc) * y[2]);
                return std::array<double, 4>{y[1], -(h + c.friction(a, Fluid::CDM)) * y[1] + sb, y[3], -h * y[3] + sb};
            };
            const double D0 = c.D1(aini), f0 = c.f1(aini);
            std::array<double, 4> y{D0, f0 * D0, D0, f0 * D0};
            const int n = 4000;
            const double x1 = std::log(aini), dx = -x1 / n;
            for (int i = 0; i < n; i++) {
                const double x = x1 + i * dx;
                auto add = [](const std::array<double, 4> & p, const std::array<double, 4> & q, double t) {
                    std::array<double, 4> r;
                    for (int k = 0; k < 4; k++)
                        r[k] = p[k] + t * q[k];
                    return r;
                };
                auto k1 = rhs(x, y), k2 = rhs(x + dx / 2, add(y, k1, dx / 2)), k3 = rhs(x + dx / 2, add(y, k2, dx / 2)),
                     k4 = rhs(x + dx, add(y, k3, dx));
                for (int k = 0; k < 4; k++)
                    y[k] += dx / 6 * (k1[k] + 2 * k2[k] + 2 * k3[k] + k4[k]);
            }
            return y;
        };
        const auto c1 = Cosmology::covariant_dark_scattering(OM, OB, -0.98, 1.0), c0 = Cosmology::covariant_dark_scattering(OM, OB, -0.98, 0.0);
        const auto y1 = grow(c1), y0 = grow(c0);
        const double fc = c1.cdm_fraction(1.0);
        const double dc = 100 * (y1[0] / y0[0] - 1), db = 100 * (y1[2] / y0[2] - 1),
                     dcb = 100 * ((fc * y1[0] + (1 - fc) * y1[2]) / (fc * y0[0] + (1 - fc) * y0[2]) - 1);
        if (FML::ThisTask == 0)
            std::cout << "  Covariant dark scattering (alpha = 1, w = -0.98), two-fluid growth z = 49 -> 0: CDM " << dc
                      << "%, baryons " << db << "%, CDM+baryons " << dcb << "% (SimulationCode: -9.13, -0.32, -7.77); "
                      << "Gamma today " << c1.friction(1.0, Fluid::CDM) << "\n";
        check(std::abs(dc + 9.13) < 0.02 and std::abs(db + 0.32) < 0.02 and std::abs(dcb + 7.77) < 0.02,
              "covariant dark scattering: the linear growth is that of SimulationCode");
        auto cds = run_zeldovich(5, 5, 8.0, false, 40, true, "CovariantDarkScattering", 0.0, 0.0, false, -0.98, 1.0);
        auto cdsw = run_zeldovich(5, 5, 8.0, false, 40, true, "CovariantDarkScattering", 0.0, 0.0, true, -0.98, 1.0);
        // The drag acts late (Gamma ~ 2 per ln a at a = 0.8), so the velocities tell the models apart
        check(cds.errx < 0.03 and cds.errv < 0.04 and cdsw.errv > 3.0 * cds.errv,
              "covariant dark scattering: one fluid follows the Zel'dovich solution with the model's growth");
    }

    auto cq = run_zeldovich(5, 5, 8.0, false, 40, true, "CoupledQuintessence", 0.5, 0.3);
    auto wrong = run_zeldovich(5, 5, 8.0, false, 40, true, "CoupledQuintessence", 0.5, 0.3, true);
    check(cq.errx < 0.03 and cq.maxdy < 1e-4, "coupled quintessence: Zel'dovich solution with the model's growth reproduced");
    check(wrong.errx > 3.0 * cq.errx, "coupled quintessence: the LCDM growth does not fit (the test sees the model)");
}

//==========================================================================================
// CDM and baryons as two sets of particles (species 0, 1) on the same lattice, both on a small plane wave along x
// with the growing mode of one fluid at aini. The linear two-fluid growth (x = log a)
//   D_c'' + (2 + dlogE/dx + Gamma_c) D_c' = (3/2) Omega_m(a) P(a) [(1 + 2 beta^2) f_c D_c + f_b D_b]
//   D_b'' + (2 + dlogE/dx) D_b'          = (3/2) Omega_m(a) P(a) [f_c D_c + f_b D_b]
// (P = rho_m(a) a^3 / OmegaM, Gamma_c = -beta dphi/dlna) is the expected displacement of each species (exact in one
// dimension for one fluid; for two fluids to first order, so the amplitude is small: D b = 0.15 at the end)
//==========================================================================================
struct TwoFluidResult {
    double amp[2];     // Measured displacement amplitudes (CDM, baryons) relative to the one-fluid D b
    double lin[2];     // Linear two-fluid prediction (the same normalization)
    double errx[2];    // rms deviation from the linear two-fluid solution, relative to the amplitude
};

TwoFluidResult run_twofluid_wave(const std::string & model, double lambda, double beta, int level, int nsteps,
                                 double ds_w = -0.98, double cds_alpha = 1.0) {
    const double OmegaM = 0.3, OmegaB = 0.05;
    AMRNBodyParameters par;
    par.OmegaM = OmegaM;
    par.OmegaB = OmegaB;
    par.gravity_model = model;
    par.cq_lambda = lambda;
    par.cq_beta = beta;
    par.ds_w = ds_w;
    par.cds_alpha = cds_alpha;
    par.baryons = true;
    par.levelmin = level;
    par.levelmax = level;
    par.subcycling = false;
    par.mg_epsilon = 1e-8;
    const Cosmology cosmo = make_cosmology(par);
    const double aini = 0.02, aend = 0.8;
    const double b = 0.15 / cosmo.D1(aend);

    // Linear two-fluid growth from aini (RK4 in log a): y = {D_c, D_c', D_b, D_b'}
    auto rhs = [&](double x, const std::array<double, 4> & y) {
        const double a = std::exp(x);
        const double fc = cosmo.cdm_fraction(a);
        const double om = 1.5 * cosmo.OmegaM_of_a(a) * cosmo.poisson_factor(a, Fluid::CDM);
        const double h = 2.0 + cosmo.dlogEdloga(a);
        const double src_b = om * (fc * y[0] + (1.0 - fc) * y[2]);
        const double src_c = src_b + om * cosmo.cdm_fifth_force(a) * y[0];
        return std::array<double, 4>{y[1], -(h + cosmo.friction(a, Fluid::CDM)) * y[1] + src_c, y[3], -h * y[3] + src_b};
    };
    auto grow = [&](double a1, double a2, std::array<double, 4> y) {
        const int n = 2000;
        const double x1 = std::log(a1), dx = (std::log(a2) - x1) / n;
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
        return y;
    };
    const double D0 = cosmo.D1(aini), f0 = cosmo.f1(aini);
    const std::array<double, 4> y0{D0, f0 * D0, D0, f0 * D0};
    const auto y1 = grow(aini, aend, y0);

    const int n1 = 1 << level;
    const long long N = (long long)n1 * n1 * n1;
    Particles part;
    make_particles(part, 2 * N, [&](long long i, AMRParticle & p) {
        const long long j = i % N;
        p.species = int(i / N);
        const double q[3] = {double(j % n1) / n1, double((j / n1) % n1) / n1, double(j / (n1 * n1)) / n1};
        for (int d = 0; d < 3; d++)
            p.Pos[d] = q[d];
        const double s = std::sin(2 * M_PI * q[0]) / (2 * M_PI);
        p.Pos[0] = q[0] - D0 * b * s;
        p.Vel[0] = -aini * aini * cosmo.E(aini) * f0 * D0 * b * s;
    });
    AMRNBodySimulation sim(par, part, aini);
    for (int st = 1; st <= nsteps; st++)
        sim.step(aini * std::pow(aend / aini, double(st) / nsteps));

    // Amplitude A of x - q = -A b sin(2 pi q) / (2 pi) per species, and the rms deviation from the linear solution
    double num[2] = {0, 0}, den[2] = {0, 0}, e2[2] = {0, 0};
    const double Dlin[2] = {y1[0], y1[2]};
    for (size_t i = 0; i < part.get_npart(); i++) {
        const int sp = part[i].species;
        const long long j = part[i].id % N;
        const double q0 = double(j % n1) / n1;
        double dx = part[i].Pos[0] - q0;
        dx -= std::round(dx);
        const double s = -b * std::sin(2 * M_PI * q0) / (2 * M_PI);
        num[sp] += dx * s;
        den[sp] += s * s;
        e2[sp] += (dx - Dlin[sp] * s) * (dx - Dlin[sp] * s);
    }
    for (int sp = 0; sp < 2; sp++) {
        FML::SumOverTasks(&num[sp]);
        FML::SumOverTasks(&den[sp]);
        FML::SumOverTasks(&e2[sp]);
    }
    TwoFluidResult r;
    const double D1end = cosmo.D1(aend);
    for (int sp = 0; sp < 2; sp++) {
        r.amp[sp] = num[sp] / den[sp] / D1end;
        r.lin[sp] = Dlin[sp] / D1end;
        r.errx[sp] = std::sqrt(e2[sp] / den[sp]) / Dlin[sp];
    }
    if (FML::ThisTask == 0)
        std::cout << "  " << model
                  << (model == "LCDM"                      ? ""
                      : model == "CovariantDarkScattering" ? " (alpha " + std::to_string(cds_alpha) + ")"
                                                           : " (beta " + std::to_string(beta) + ")")
                  << ", " << n1
                  << "^3 of each species: D_c / D = " << r.amp[0] << " (linear two-fluid " << r.lin[0]
                  << "), D_b / D = " << r.amp[1] << " (" << r.lin[1] << "); rms deviation " << r.errx[0] << " (CDM), "
                  << r.errx[1] << " (baryons)\n";
    return r;
}

void test_two_fluid() {
    if (FML::ThisTask == 0)
        std::cout << "\nCDM and baryons as two sets of particles (plane wave, D b = 0.15 at a = 0.8)\n";
    auto l = run_twofluid_wave("LCDM", 0.0, 0.0, 5, 40);
    // The force of the 32^3 grid on the wave is ~1% low (as for one fluid), so the model is compared relative to LCDM
    check(std::abs(l.amp[0] - 1.0) < 0.02 and std::abs(l.amp[1] / l.amp[0] - 1.0) < 1e-6,
          "LCDM: both species follow the one-fluid Zel'dovich solution together");
    auto c = run_twofluid_wave("CoupledQuintessence", 0.5, 0.3, 5, 40);
    const double split_sim = c.amp[1] / c.amp[0], split_lin = c.lin[1] / c.lin[0];
    if (FML::ThisTask == 0)
        std::cout << "  D_b / D_c = " << split_sim << " (linear two-fluid " << split_lin << "); relative to LCDM: D_c "
                  << c.amp[0] / l.amp[0] << " (" << c.lin[0] << "), D_b " << c.amp[1] / l.amp[1] << " (" << c.lin[1] << ")\n";
    check(std::abs(c.amp[0] / l.amp[0] / c.lin[0] - 1.0) < 3e-3 and std::abs(c.amp[1] / l.amp[1] / c.lin[1] - 1.0) < 3e-3,
          "coupled quintessence: each species follows the linear two-fluid growth (relative to LCDM)");
    check(std::abs(split_sim / split_lin - 1.0) < 1e-3 and split_lin < 0.9,
          "coupled quintessence: the baryon-CDM split D_b / D_c agrees with linear theory");
    // Covariant dark scattering (alpha = 1, w = -0.98): the CDM drag, the baryons only gravity
    auto d = run_twofluid_wave("CovariantDarkScattering", 0.0, 0.0, 5, 40, -0.98, 1.0);
    const double dsplit_sim = d.amp[1] / d.amp[0], dsplit_lin = d.lin[1] / d.lin[0];
    if (FML::ThisTask == 0)
        std::cout << "  D_b / D_c = " << dsplit_sim << " (linear two-fluid " << dsplit_lin << "); relative to LCDM: D_c "
                  << d.amp[0] / l.amp[0] << " (" << d.lin[0] << "), D_b " << d.amp[1] / l.amp[1] << " (" << d.lin[1] << ")\n";
    check(std::abs(d.amp[0] / l.amp[0] / d.lin[0] - 1.0) < 3e-3 and std::abs(d.amp[1] / l.amp[1] / d.lin[1] - 1.0) < 3e-3,
          "covariant dark scattering: each species follows the linear two-fluid growth (relative to LCDM)");
    check(std::abs(dsplit_sim / dsplit_lin - 1.0) < 1e-3 and dsplit_lin > 1.03,
          "covariant dark scattering: the baryon-CDM split D_b / D_c agrees with linear theory");
}

int main(int argc, char ** argv) {
    const std::string which = argc > 1 ? argv[1] : "all";
    if (which == "deposit" or which == "all")
        test_deposit();
    if (which == "blob" or which == "all")
        test_blob();
    if (which == "zeldovich" or which == "all")
        test_zeldovich();
    if (which == "cq" or which == "all")
        test_coupled_quintessence();
    if (which == "twofluid" or which == "all")
        test_two_fluid();
    if (FML::ThisTask == 0)
        std::cout << "\n" << (nfail ? std::to_string(nfail) + " tests FAILED" : "All tests passed") << "\n";
    return nfail ? 1 : 0;
}
