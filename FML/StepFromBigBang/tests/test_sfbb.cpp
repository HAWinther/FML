// Tests of StepFromBigBang with the exact (NUFFT) particle force.
//
//   test_sfbb eds  N kcut sigma   : Einstein-de Sitter, D1 = a from 1e-7 to 1: errors of bullfrog / lpt3 with n steps
//                                   against 64 BullFrog steps, and the distance of one step from (dealiased) 3LPT
//   test_sfbb lcdm N sigma        : LCDM (LPTGrowthFactors) to z = 19: linear exactness at sigma -> 0 and the errors
//   test_sfbb mg   N sigma        : a scale-dependent toy model (f(R)-like mu(a, k)): linear exactness per k of the
//                                   scale-dependent steps (hybrid) and the hybrid errors
//
// kcut: the linear spectrum P ~ k^-2 is cut at kcut k_Ny. sigma: rms of the linear delta on the lattice at a_end.
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>
#include <FML/StepFromBigBang/Forces.h>
#include <FML/StepFromBigBang/StepFromBigBang.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace FML::STEPFROMBIGBANG;
constexpr int NDIM = 3;

struct Part {
    double pos[3];
    double vel[3];
    double scr[3];
    double q[3];
    long long id;
    double * get_pos() { return pos; }
    const double * get_pos() const { return pos; }
    double * get_vel() { return vel; }
    const double * get_vel() const { return vel; }
    double * get_sfbb_scratch() { return scr; }
    double * get_q() { return q; }
    const double * get_q() const { return q; }
    void set_id(long long i) { id = i; }
    long long get_id() const { return id; }
    constexpr int get_ndim() const { return 3; }
};
using Particles = FML::PARTICLE::MPIParticles<Part>;
using SFBB = StepFromBigBang<NDIM, Part>;

static int NP;
static double NUFFT_EPS = 1e-12;

// The linear delta at a_end: P ~ k^-2 for k < kcut k_Ny, rms sigma on the lattice
FML::GRID::FFTWGrid<NDIM> make_delta(double kcut, double sigma, int seed = 1) {
    FML::GRID::FFTWGrid<NDIM> delta(NP, 1, 1);
    FML::RANDOM::RandomGenerator rng;
    rng.set_seed(seed);
    const double kny = M_PI * NP;
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier<NDIM>(
        delta, &rng, [](double k) { return std::pow(std::max(k, 1.0), -2.0); }, false);
    FML::GRID::for_each_fourier_mode<NDIM>(delta, [&](size_t i, double k) {
        if (k >= kcut * kny or k == 0.0)
            delta.set_fourier_from_index(i, 0.0);
    });
    auto tmp = delta;
    tmp.fftw_c2r();
    double s2 = 0;
    for (auto && ri : tmp.get_real_range())
        s2 += std::pow(tmp.get_real_from_index(ri), 2);
    const double norm = sigma / std::sqrt(s2 / std::pow(NP, NDIM));
    FML::GRID::for_each_fourier_mode<NDIM>(delta, [&](size_t i, double) {
        delta.set_fourier_from_index(i, delta.get_fourier_from_index(i) * FML::GRID::FloatType(norm));
    });
    return delta;
}

// The force callback: F = kappa0(a) * G with G = -grad laplace^-1 (mu(a, k) delta), exact with the NUFFT
SFBB::ForceFunction make_force(std::function<double(double)> kappa0,
                               std::function<double(double, double)> mu_of_a_kBox = nullptr) {
    return [kappa0, mu_of_a_kBox](Particles & part, double a, const SFBB::Filter & filter,
                                  const std::vector<Target> & targets) {
        std::array<std::vector<FML::GRID::FloatType>, NDIM> F;
        std::function<double(double)> filt = nullptr;
        if (filter or mu_of_a_kBox)
            filt = [&](double kBox) {
                double r = mu_of_a_kBox ? mu_of_a_kBox(a, kBox) : 1.0;
                if (filter)
                    r *= filter(kBox);
                return r;
            };
        FML::NUFFT::compute_force_nufft<NDIM, Part>(
            part.get_particles_ptr(), part.get_npart(), part.get_npart_total(), NP, 1.0, F, NUFFT_EPS, 2.0, filt);
        const double k0 = kappa0(a);
        auto * p = part.get_particles_ptr();
        for (size_t i = 0; i < part.get_npart(); i++) {
            double f[NDIM];
            for (int d = 0; d < NDIM; d++)
                f[d] = -k0 * F[d][i];   // the NUFFT returns -G for norm 1
            add_to_targets<NDIM>(p[i], f, targets);
        }
    };
}

// Displacements and velocities by particle id
struct State {
    std::vector<std::array<double, 3>> d, v;
};
State state_of(Particles & part) {
    State s;
    s.d.resize(part.get_npart_total());
    s.v.resize(part.get_npart_total());
    for (auto & p : part) {
        const long long id = p.get_id();
        const long long iz = id % NP, iy = (id / NP) % NP, ix = id / (NP * NP);
        const double q[3] = {double(ix) / NP, double(iy) / NP, double(iz) / NP};
        for (int d = 0; d < NDIM; d++) {
            double dx = p.get_pos()[d] - q[d];
            dx -= std::round(dx);
            s.d[id][d] = dx;
            s.v[id][d] = p.get_vel()[d];
        }
    }
    return s;
}
double rms_diff(const std::vector<std::array<double, 3>> & a, const std::vector<std::array<double, 3>> & b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); i++)
        for (int d = 0; d < 3; d++)
            s += std::pow(a[i][d] - b[i][d], 2);
    return std::sqrt(s / a.size());
}

// Run a scheme from scratch and return the final state
State run_scheme(const GrowthModel & g, SFBB::ForceFunction force, Settings s, const FML::GRID::FFTWGrid<NDIM> & delta,
                 double a_ref) {
    s.verbose = false;
    SFBB sfbb(g, force, s);
    Particles part;
    sfbb.make_start_particles(part, NP, 1.5, delta, a_ref);
    sfbb.run(part);
    return state_of(part);
}

// The nLPT (EdS) displacement of delta (at D = 1) by id, from the library's IC (dealiased)
State lpt_state(const FML::GRID::FFTWGrid<NDIM> & delta, int order) {
    Particles part;
    std::vector<FML::GRID::FFTWGrid<NDIM>> phi;
    FML::NBODY::LPTInitialConditionsOptions opt;
    opt.dealiasing = FML::COSMOLOGY::LPT::Dealiasing::Padding;
    FML::NBODY::NBodyInitialConditions<NDIM, Part>(
        part, NP, 1.5, delta, phi, order, 1.0, 0.0, std::vector<double>{1.0, 2.0, 3.0, 3.0}, nullptr, nullptr, nullptr,
        nullptr, nullptr, opt);
    return state_of(part);
}

void report(const std::string & name, int forces, const State & x, const State & ref, const State & za) {
    std::cout << "  " << std::left << std::setw(16) << name << std::right << std::setw(4) << forces << "   pos "
              << std::scientific << std::setprecision(3) << rms_diff(x.d, ref.d) / rms_diff(ref.d, za.d) << "   vel "
              << rms_diff(x.v, ref.v) / rms_diff(ref.v, za.v) << std::defaultfloat << "\n";
}

int main(int argc, char ** argv) {
    FML::init_fml();
    const std::string mode = argv[1];
    NP = std::stoi(argv[2]);
    if (const char * e = std::getenv("NUFFT_EPS"))
        NUFFT_EPS = std::stod(e);

    if (mode == "dump") {
        // Data for the accuracy figure. test_sfbb dump N kcut sigma outdir job...  (EdS, D = 1 at the end)
        //   ref: 64 BullFrog and 32 lpt3 steps; ic: Zel'dovich, 2LPT, 3LPT (dealiased, no curl); bf: 1..16 BullFrog;
        //   l3: 1..8 lpt3; fv: one BullFrog step with each force; forces: the force of each method at the Zel'dovich
        //   configuration at D = 1/2 (the exact one first); forces_late: the same after 16 BullFrog steps (D = 1).
        //   Displacements x - q (or forces) as doubles by id.
        const double kcut = std::stod(argv[3]), sigma = std::stod(argv[4]);
        const std::string out = argv[5];
        auto g = growth_EdS();
        auto kappa0 = [](double a) { return 1.5 * std::sqrt(a); };
        auto delta = make_delta(kcut, sigma);
        Settings s;
        s.a_start = 1e-7;
        s.a_end = 1.0;
        s.verbose = false;
        auto save = [&](const std::string & name, const std::vector<std::array<double, 3>> & d) {
            FILE * fp = fopen((out + "/" + name + ".bin").c_str(), "wb");
            fwrite(d.data(), sizeof(double), 3 * d.size(), fp);
            fclose(fp);
            std::cout << "wrote " << name << "\n" << std::flush;
        };
        auto exists = [&](const std::string & name) {
            FILE * fp = fopen((out + "/" + name + ".bin").c_str(), "rb");
            if (fp)
                fclose(fp);
            return fp != nullptr;
        };
        auto run = [&](const std::string & name, const std::string & scheme, int n, SFBB::ForceFunction f) {
            if (exists(name))
                return;
            auto ss = s;
            ss.scheme = scheme;
            ss.nsteps = n;
            Particles part;
            generate_particles_from_big_bang<NDIM, Part>(part, NP, 1.5, delta, 1.0, g, f, ss);
            save(name, state_of(part).d);
        };
        auto exact = make_force_nufft<NDIM, Part>(NP, kappa0);
        std::vector<std::pair<std::string, SFBB::ForceFunction>> variants{
            {"sheet_nufft", make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "NUFFT", "NUFFT")},
            {"sheet_PCS_1x", make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true)},
            {"sheet_PCS_2x", make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true)},
            {"sheet_PCS_2x_il2", make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true, 2)},
            {"pm_CIC_1x", make_force_pm<NDIM, Part>(NP, kappa0, nullptr, "CIC", false, false)},
            {"pm_PCS_1x_dc", make_force_pm<NDIM, Part>(NP, kappa0, nullptr, "PCS", false, true)},
            {"pm_PCS_1x_dc_il2", make_force_pm<NDIM, Part>(NP, kappa0, nullptr, "PCS", false, true, 2)},
            {"pm_CIC_2x", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "CIC", false, false)},
            {"pm_PCS_2x_dc", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "PCS", false, true)},
            {"pm_PCS_2x_dc_il2", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "PCS", false, true, 2)}};
        for (int i = 6; i < argc; i++) {
            const std::string job = argv[i];
            if (job == "ref") {
                run("bf64", "bullfrog", 64, exact);
                run("l3_32", "lpt3", 32, exact);
            } else if (job == "ic") {
                if (not exists("za"))
                    save("za", lpt_state(delta, 1).d);
                if (not exists("lpt2"))
                    save("lpt2", lpt_state(delta, 2).d);
                if (not exists("lpt3"))
                    save("lpt3", lpt_state(delta, 3).d);
            } else if (job == "bf") {
                for (int n : {1, 2, 4, 8, 16})
                    run("bf" + std::to_string(n), "bullfrog", n, exact);
            } else if (job == "l3") {
                for (int n : {1, 2, 4, 8})
                    run("l3_" + std::to_string(n), "lpt3", n, exact);
            } else if (job == "fv") {
                for (auto & [name, f] : variants)
                    run("bf1_" + name, "bullfrog", 1, f);
            } else if (job == "forces" or job == "forces_late") {
                // forces: at the Zel'dovich configuration at D = 1/2 (near the lattice); forces_late: at the end of 16
                // BullFrog steps (D = 1, far from the lattice for sigma ~ 1)
                const bool late = job == "forces_late";
                const std::string prefix = late ? "lateforce_" : "force_";
                const double a = late ? 1.0 : 0.5;
                Particles part;
                if (late) {
                    auto ss = s;
                    ss.nsteps = 16;
                    generate_particles_from_big_bang<NDIM, Part>(part, NP, 1.5, delta, 1.0, g, exact, ss);
                } else {
                    auto ss = s;
                    ss.a_start = 0.5;
                    SFBB sfbb(g, exact, ss);
                    sfbb.make_start_particles(part, NP, 1.5, delta, 1.0);
                }
                auto get = [&](const std::string & name, SFBB::ForceFunction f) {
                    if (exists(prefix + name))
                        return;
                    for (auto & p : part)
                        for (int d = 0; d < 3; d++)
                            p.scr[d] = 0;
                    f(part, a, nullptr, {{Target::Scratch, 1.0}});
                    std::vector<std::array<double, 3>> F(part.get_npart_total());
                    for (auto & p : part)
                        F[p.get_id()] = {p.scr[0], p.scr[1], p.scr[2]};
                    save(prefix + name, F);
                };
                get("exact", exact);
                for (auto & [name, f] : variants)
                    get(name, f);
            }
        }
        return 0;
    }

    if (mode == "forcecmp") {
        auto g = growth_EdS();
        auto kappa0 = [](double a) { return 1.0 + 0 * a; };
        auto delta = make_delta(0.25, 0.3);
        Settings s;
        s.a_start = 0.5;
        s.a_end = 1.0;
        s.verbose = false;
        SFBB sfbb(g, make_force_nufft<NDIM, Part>(NP, kappa0), s);
        Particles part;
        sfbb.make_start_particles(part, NP, 1.5, delta, 1.0);   // Zel'dovich at D = 0.5
        auto get = [&](SFBB::ForceFunction f) {
            for (auto & p : part)
                for (int d = 0; d < 3; d++)
                    p.scr[d] = 0;
            f(part, 0.5, nullptr, {{Target::Scratch, 1.0}});
            std::vector<std::array<double, 3>> F;
            for (auto & p : part)
                F.push_back({p.scr[0], p.scr[1], p.scr[2]});
            return F;
        };
        auto Fn = get(make_force_nufft<NDIM, Part>(NP, kappa0));
        std::vector<std::pair<std::string, SFBB::ForceFunction>> forces{
            {"CIC 1x", make_force_pm<NDIM, Part>(NP, kappa0, nullptr, "CIC", false, false)},
            {"CIC 2x", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "CIC", false, false)},
            {"PCS 1x dc", make_force_pm<NDIM, Part>(NP, kappa0, nullptr, "PCS", false, true)},
            {"PCS 1x dc interlaced 2", make_force_pm<NDIM, Part>(NP, kappa0, nullptr, "PCS", false, true, 2)},
            {"PCS 2x il+dc", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "PCS", true, true)},
            {"sheet R=2 nufft 1x", make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "NUFFT", "NUFFT")},
            {"sheet R=4 nufft 1x", make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 4, "NUFFT", "NUFFT")},
            {"sheet R=2 PCS 1x dc", make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true)},
            {"sheet R=2 PCS 2x dc", make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true)},
            {"PCS 2x dc interlaced 2", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "PCS", false, true, 2)},
            {"PCS 2x dc interlaced 4", make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "PCS", false, true, 4)},
            {"sheet PCS 1x dc interlaced 2", make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true, 2)},
            {"sheet PCS 2x dc interlaced 2",
             make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true, 2)},
            {"sheet PCS 2x dc interlaced 4",
             make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true, 4)},
            {"sheet nufft interlaced 2 (check)",
             make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "NUFFT", "NUFFT", true, 2)}};
        for (auto & [name, force_fn] : forces) {
            auto Fp = get(force_fn);
            double ab = 0, bb = 0, aa = 0;
            for (size_t i = 0; i < Fn.size(); i++)
                for (int d = 0; d < 3; d++) {
                    ab += Fp[i][d] * Fn[i][d];
                    bb += Fn[i][d] * Fn[i][d];
                    aa += Fp[i][d] * Fp[i][d];
                }
            std::cout << name << ": <Fpm Fnufft>/<Fnufft^2> = " << ab / bb << "  |Fpm|/|Fnufft| = " << std::sqrt(aa / bb)
                      << "  rel. rms difference " << std::sqrt(rms_diff(Fp, Fn) * rms_diff(Fp, Fn) / (bb / Fn.size()))
                      << "\n";
        }
        return 0;
    }

    if (mode == "api") {
        // The one-call function and the ready-made forces (EdS, smooth field, sigma = 0.3, one BullFrog step)
        auto g = growth_EdS();
        auto kappa0 = [](double a) { return 1.5 * std::sqrt(a); };
        auto delta = make_delta(0.25, 0.3);
        Settings s;
        s.a_start = 1e-7;
        s.a_end = 1.0;
        s.verbose = false;
        const State za = lpt_state(delta, 1);
        auto one_call = [&](SFBB::ForceFunction f, int n, double shift = 0.0) {
            auto ss = s;
            ss.nsteps = n;
            Particles part;
            generate_particles_from_big_bang<NDIM, Part>(part, NP, 1.5, delta, 1.0, g, f, ss, shift);
            auto st = state_of(part);
            for (auto & d : st.d)   // undo the lattice shift
                for (auto & x : d)
                    x -= shift, x -= std::round(x);
            return st;
        };
        const State own = run_scheme(g, make_force(kappa0), [&] { auto ss = s; ss.nsteps = 1; return ss; }(), delta, 1.0);
        const State lib = one_call(make_force_nufft<NDIM, Part>(NP, kappa0), 1);
        std::cout << "one call + make_force_nufft against the test's own force and class calls: "
                  << rms_diff(lib.d, own.d) << " (box units)\n";
        const State ref = one_call(make_force_nufft<NDIM, Part>(NP, kappa0), 64);
        std::cout << "error / beyond-Zel'dovich against 64 BullFrog steps (NUFFT):\n";
        report("1 step, NUFFT", 1, lib, ref, za);
        for (auto [name, nm, asg, il, dc] : std::vector<std::tuple<std::string, int, std::string, bool, bool>>{
                 {"CIC 1x", NP, "CIC", false, false},
                 {"CIC 2x", 2 * NP, "CIC", false, false},
                 {"PCS 1x il+dc", NP, "PCS", true, true},
                 {"PCS 2x il+dc", 2 * NP, "PCS", true, true}})
        {
            report("1 step, " + name, 1, one_call(make_force_pm<NDIM, Part>(nm, kappa0, nullptr, asg, il, dc), 1), ref, za);
        }
        for (auto [name, nm, R, asg] : std::vector<std::tuple<std::string, int, int, std::string>>{
                 {"sheet R=2 nufft", NP, 2, "NUFFT"}, {"sheet R=2 PCS 1x", NP, 2, "PCS"}, {"sheet R=2 PCS 2x", 2 * NP, 2, "PCS"}}) {
            report("1 step, " + name, 1, one_call(make_force_sheet<NDIM, Part>(nm, NP, kappa0, nullptr, R, asg, asg), 1), ref, za);
            report("  + half-cell shift", 1,
                   one_call(make_force_sheet<NDIM, Part>(nm, NP, kappa0, nullptr, R, asg, asg), 1, 0.5 / nm), ref, za);
        }
        report("1 step, PCS 2x dc interlaced 2", 1,
               one_call(make_force_pm<NDIM, Part>(2 * NP, kappa0, nullptr, "PCS", false, true, 2), 1), ref, za);
        report("1 step, sheet PCS 2x dc interlaced 2", 1,
               one_call(make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true, 2), 1), ref, za);
        report("1 step, sheet PCS 2x dc interlaced 4", 1,
               one_call(make_force_sheet<NDIM, Part>(2 * NP, NP, kappa0, nullptr, 2, "PCS", "PCS", true, 4), 1), ref, za);
        // Several steps with the sheet force against the NUFFT reference
        for (int n : {4, 16})
            report("bullfrog " + std::to_string(n) + " sheet nufft", n,
                   one_call(make_force_sheet<NDIM, Part>(NP, NP, kappa0, nullptr, 2, "NUFFT", "NUFFT"), n), ref, za);
        return 0;
    }

    if (mode == "eds") {
        const double kcut = std::stod(argv[3]), sigma = std::stod(argv[4]);
        auto g = growth_EdS();
        auto force = make_force([](double a) { return 1.5 * std::sqrt(a); });
        auto delta = make_delta(kcut, sigma);
        Settings s;
        s.a_start = 1e-7;
        s.a_end = 1.0;
        auto run = [&](const std::string & scheme, int n) {
            auto ss = s;
            ss.scheme = scheme;
            ss.nsteps = n;
            return run_scheme(g, force, ss, delta, 1.0);
        };
        const State ref = run("bullfrog", 64), za = lpt_state(delta, 1), l3 = lpt_state(delta, 3);
        std::cout << "EdS, kcut = " << kcut << ", sigma = " << sigma
                  << ": error / beyond-Zel'dovich against 64 BullFrog steps\n";
        report("3LPT IC (no curl)", 0, l3, ref, za);
        for (int n : {1, 2, 4, 8, 16})
            report("bullfrog " + std::to_string(n), n, run("bullfrog", n), ref, za);
        for (int n : {1, 2, 4, 8})
            report("lpt3 " + std::to_string(n), 2 * n, run("lpt3", n), ref, za);
        auto ref32 = run("lpt3", 32);
        std::cout << "  reference check: 32 lpt3 steps against 64 BullFrog: pos "
                  << rms_diff(ref32.d, ref.d) / rms_diff(ref.d, za.d) << "\n";
        // Order check: one step against 3LPT, relative to the 3LPT displacement
        auto b1 = run("bullfrog", 1), t1 = run("lpt3", 1);
        std::vector<std::array<double, 3>> zero(l3.d.size(), {0, 0, 0});
        std::cout << "  one step - 3LPT (no curl), / |3LPT|: bullfrog " << rms_diff(b1.d, l3.d) / rms_diff(l3.d, zero)
                  << "  lpt3 " << rms_diff(t1.d, l3.d) / rms_diff(l3.d, zero) << "\n";
        return 0;
    }

    if (mode == "lcdm" or mode == "mg") {
        const double sigma = std::stod(argv[3]);
        // RADIATION=1: radiation in the background (Omega_r h^2 = 4.2e-5, h = 0.7), so the start is deep in the radiation
        // era where D2 / D1^2 is far from EdS; KCUT: the cut of the spectrum (default 1)
        const double Or = std::getenv("RADIATION") ? 4.2e-5 / 0.49 : 0.0;
        const double Om = 0.3, OL = 1.0 - Om - Or, box = 200.0;
        const double kcut_env = std::getenv("KCUT") ? std::stod(std::getenv("KCUT")) : 1.0;
        FML::COSMOLOGY::GROWTH::LPTGrowthEquations eq;
        eq.E = [=](double x) { return std::sqrt(Or * std::exp(-4 * x) + Om * std::exp(-3 * x) + OL); };
        eq.dlogEdx = [=](double x) {
            const double e2 = Or * std::exp(-4 * x) + Om * std::exp(-3 * x) + OL;
            return -(2 * Or * std::exp(-4 * x) + 1.5 * Om * std::exp(-3 * x)) / e2;
        };
        eq.matter_density = [=](double x) { return Om * std::exp(-3 * x); };
        // f(R)-like toy: mu = 1 + 1/3 k^2 / (k^2 + m(a)^2), m = 300 a^-0.5 (c k / H0): k = 0.45 h/Mpc at z = 19
        auto mu = [](double x, double k) {
            const double m = 300.0 * std::exp(-0.5 * x);
            return 1.0 + (k * k) / (3.0 * (k * k + m * m));
        };
        FML::COSMOLOGY::GROWTH::LPTGrowthSettings gs;
        gs.a_min = 1e-7;
        gs.a_max = 2.0;
        gs.n_a = 2000;
        gs.a_normalize = 1.0 / 20.0;
        gs.start_in_radiation_era = false;
        if (Or > 0 and FML::ThisTask == 0)
            std::cout << "With radiation (a_eq = " << Or / Om << ")\n";
        if (mode == "mg") {
            eq.mu = mu;
            gs.scale_dependent = true;
            gs.koverH0_min = 1.0;
            gs.koverH0_max = 3e4;
            gs.n_k = 120;
        }
        auto lpt = std::make_shared<FML::COSMOLOGY::GROWTH::LPTGrowthFactors>(eq, gs);
        auto g = growth_from_LPTGrowthFactors(lpt, eq);
        auto kappa0 = [=](double a) {
            const double x = std::log(a), e = eq.E(x);
            return a * a * e * 1.5 * eq.matter_density(x) / (e * e);
        };
        SFBB::ForceFunction force;
        if (mode == "mg")
            force = make_force(kappa0, [=](double a, double kBox) {
                return mu(std::log(a), kBox / box * FML::UTILS::SI::c_over_H0_mpch);
            });
        else
            force = make_force(kappa0);
        Settings s;
        s.a_start = 1e-6;
        s.a_end = 1.0 / 20.0;
        s.boxsize = box;
        auto delta = make_delta(kcut_env, sigma);
        auto run = [&](const std::string & scheme, int n, int nsd = 1, double a_switch = 0) {
            auto ss = s;
            ss.scheme = scheme;
            ss.nsteps = n;
            ss.nsteps_scaledependent = nsd;
            ss.a_switch = a_switch > 0 ? a_switch : s.a_end;
            return run_scheme(g, force, ss, delta, s.a_end);
        };
        // Linear: Zel'dovich at a_end with the growth of each k (delta is at a_end): x = q + Psi, v = vnorm f Psi
        Particles zp;
        {
            std::vector<FML::GRID::FFTWGrid<NDIM>> phi;
            const double ae = s.a_end;
            auto vnk = [&](double kBox) {
                const double k = g.scale_dependent ? kBox / box * FML::UTILS::SI::c_over_H0_mpch : 0.0;
                return g.vnorm(ae) * g.dD1(ae, k) / g.D1(ae, k);
            };
            FML::NBODY::NBodyInitialConditions<NDIM, Part>(
                zp, NP, 1.5, delta, phi, 1, box, 1.0 / ae - 1.0, std::vector<double>{vnk(1e-6)}, vnk);
        }
        const State za = state_of(zp);
        std::vector<std::array<double, 3>> zero(za.d.size(), {0, 0, 0});
        std::cout << (mode == "mg" ? "Scale-dependent toy (f(R)-like)" : "LCDM") << " to z = 19, sigma = " << sigma
                  << ", kcut = " << kcut_env << "; at the start D2/D1^2 / EdS = "
                  << g.r2(s.a_start, 0) / std::pow(g.D1(s.a_start, 0), 2) << ", at the end "
                  << g.r2(s.a_end, 0) / std::pow(g.D1(s.a_end, 0), 2) << "\n";
        if (mode == "lcdm" and std::getenv("CONV")) {
            std::map<std::string, State> st;
            for (int n : {32, 64, 128})
                st["bullfrog " + std::to_string(n)] = run("bullfrog", n);
            for (int n : {16, 32, 64})
                st["lpt3 " + std::to_string(n)] = run("lpt3", n);
            const auto & r = st["bullfrog 128"];
            for (auto & [name, x] : st)
                std::cout << "  " << name << " vs bullfrog 128: pos " << rms_diff(x.d, r.d) / rms_diff(r.d, za.d)
                          << "  vel " << rms_diff(x.v, r.v) / rms_diff(r.v, za.v) << "\n";
            return 0;
        }
        if (mode == "lcdm") {
            const State ref = run("bullfrog", 64);
            std::cout << "  distance from linear (Zel'dovich at the end) / |displacement|: ";
            for (auto & [sch, n] : std::vector<std::pair<std::string, int>>{{"bullfrog", 1}, {"lpt3", 1}})
                std::cout << sch << " " << rms_diff(run(sch, n).d, za.d) / rms_diff(za.d, zero) << "  ";
            std::cout << "\n  error / beyond-Zel'dovich against 64 BullFrog steps:\n";
            for (int n : {1, 2, 4, 8})
                report("bullfrog " + std::to_string(n), n, run("bullfrog", n), ref, za);
            for (int n : {1, 2, 4})
                report("lpt3 " + std::to_string(n), 2 * n, run("lpt3", n), ref, za);
            auto c = run("lpt3", 32);
            std::cout << "  reference check: 32 lpt3 steps against 64 BullFrog: pos "
                      << rms_diff(c.d, ref.d) / rms_diff(ref.d, za.d) << "\n";
        } else {
            // Pure scale-dependent FastPM to a_end: linear per k
            for (int nsd : {1, 2, 8}) {
                auto x = run("hybrid", 0, nsd, s.a_end);
                std::cout << "  scale-dependent FastPM, " << nsd << " step(s) to the end: |x - linear| / |linear| = "
                          << rms_diff(x.d, za.d) / rms_diff(za.d, zero)
                          << ", velocity: " << rms_diff(x.v, za.v) / rms_diff(za.v, zero) << "\n";
            }
            for (int nbf : {1, 2}) {
                auto x = run("hybrid", nbf, 1, 1.0 / 101.0);
                std::cout << "  hybrid (1 sd step to z = 100, " << nbf
                          << " BullFrog): |x - linear| / |linear| = " << rms_diff(x.d, za.d) / rms_diff(za.d, zero)
                          << "\n";
            }
        }
        return 0;
    }
    std::cerr << "Unknown mode\n";
    return 1;
}
