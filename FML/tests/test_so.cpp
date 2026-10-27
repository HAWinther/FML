//==========================================================
// Tests for the spherical overdensity properties of halos (FriendsOfFriends/SphericalOverdensity.h): singular
// isothermal spheres (mean enclosed density 3 N / (4 pi R_max r^2)) on a uniform background, some of them on the
// boundaries of the task domains, must give the analytic radius, mass and center for any number of tasks, and
// spheres larger than the search radius must be reported as truncated. Halos with a core as large as the halo and
// ~100 particles (as at the resolution of a PM simulation) must not collapse to a few particles.
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <FML/Global/Global.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

constexpr int NDIM = 3;

struct Particle {
    double pos[NDIM];
    double vel[NDIM];
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
};

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
double wrap(double x) { return x - std::floor(x); }
std::string num(double x) {
    std::ostringstream os;
    os << std::setprecision(3) << x;
    return os.str();
}

int main() {
    const int nbackground = 20000;
    const int nhalo = 4000;
    const double Rmax = 0.02;
    const double vel_halo = 0.3;
    const std::vector<std::array<double, 3>> centers{
        {0.5, 0.3, 0.7}, {0.25, 0.6, 0.2}, {0.75, 0.1, 0.9}, {0.0, 0.5, 0.5}, {0.62, 0.8, 0.4}, {0.999, 0.2, 0.3}};

    // The same particles on all tasks; keep the ones in the local domain
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<Particle> all;
    for (int i = 0; i < nbackground; i++)
        all.push_back({{uni(gen), uni(gen), uni(gen)}, {0.0, 0.0, 0.0}});
    for (size_t h = 0; h < centers.size(); h++) {
        for (int i = 0; i < nhalo; i++) {
            const double r = Rmax * uni(gen);
            const double mu = 2.0 * uni(gen) - 1.0, phi = 2.0 * M_PI * uni(gen);
            const double s = std::sqrt(1.0 - mu * mu);
            all.push_back({{wrap(centers[h][0] + r * s * std::cos(phi)), wrap(centers[h][1] + r * s * std::sin(phi)),
                            wrap(centers[h][2] + r * mu)},
                           {vel_halo * (h + 1), 0.0, 0.0}});
        }
    }
    std::vector<Particle> local;
    for (auto & p : all)
        if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
            local.push_back(p);
    const double total_mass = double(all.size());

    // The halos whose (guessed) center is on this task, guess offset from the true center
    std::vector<std::array<double, 3>> guess;
    std::vector<size_t> which;
    for (size_t h = 0; h < centers.size(); h++) {
        std::array<double, 3> g{wrap(centers[h][0] + 0.003), wrap(centers[h][1] - 0.002), centers[h][2]};
        if (g[0] >= FML::xmin_domain and g[0] < FML::xmax_domain) {
            guess.push_back(g);
            which.push_back(h);
        }
    }

    // Overdensities that give R = 0.25, 0.5 and 0.75 Rmax: Delta rho_mean - n_bg = 3 N / (4 pi Rmax R^2)
    const std::vector<double> Rwanted{0.25 * Rmax, 0.5 * Rmax, 0.75 * Rmax};
    std::vector<double> thresholds;
    for (auto R : Rwanted)
        thresholds.push_back((3.0 * nhalo / (4.0 * M_PI * Rmax * R * R) + nbackground) / total_mass);
    std::vector<double> initial_radius(guess.size(), Rmax);
    const double search_radius = 0.05;
    auto so = FML::FOF::SphericalOverdensity<Particle, NDIM>(
        local.data(), local.size(), total_mass, guess, initial_radius, thresholds, search_radius);

    double maxerr_R = 0.0, maxerr_M = 0.0, maxerr_pos = 0.0, maxerr_vel = 0.0;
    for (size_t i = 0; i < so.size(); i++) {
        const size_t h = which[i];
        for (size_t it = 0; it < thresholds.size(); it++) {
            const double R = Rwanted[it];
            const double M = nhalo * R / Rmax + nbackground * 4.0 * M_PI / 3.0 * R * R * R;
            maxerr_R = std::max(maxerr_R, std::fabs(so[i].radius[it] / R - 1.0));
            maxerr_M = std::max(maxerr_M, std::fabs(so[i].mass[it] / M - 1.0));
        }
        for (int idim = 0; idim < NDIM; idim++) {
            double d = so[i].pos[idim] - centers[h][idim];
            d -= std::round(d);
            maxerr_pos = std::max(maxerr_pos, std::fabs(d));
        }
        maxerr_vel = std::max(maxerr_vel, std::fabs(so[i].vel[0] / (vel_halo * (h + 1)) - 1.0));
    }
    FML::MaxOverTasks(&maxerr_R);
    FML::MaxOverTasks(&maxerr_M);
    FML::MaxOverTasks(&maxerr_pos);
    FML::MaxOverTasks(&maxerr_vel);
    size_t nfound = so.size();
    FML::SumOverTasks(&nfound);
    const std::string label = std::to_string(FML::NTasks) + " task(s): ";
    check(nfound == centers.size(), label + "all halos measured");
    check(FML::FOF::SO_number_of_truncated_halos == 0, label + "no halo truncated");
    // Poisson noise with N(<R) = 1000 - 3000: for an isothermal sphere dln R = dln N / 2 (1.6% for N = 1000) and
    // dln M = 3 dln R (M = Delta rho V(R)); the limits are 3 sigma
    check(maxerr_R < 0.05, label + "radius of a singular isothermal sphere (max rel. error " + num(maxerr_R) + ")");
    check(maxerr_M < 0.15, label + "mass (max rel. error " + num(maxerr_M) + ")");
    check(maxerr_pos < 0.1 * Rmax, label + "shrinking sphere center (max error " + num(maxerr_pos / Rmax) + " Rmax)");
    check(maxerr_vel < 0.2, label + "mean velocity of the halo particles (max rel. error " + num(maxerr_vel) + ")");

    // A threshold whose radius is larger than the search radius: truncated
    const double Rbig = 0.08;
    std::vector<double> low{(3.0 * nhalo / (4.0 * M_PI * Rmax * Rbig * Rbig) + nbackground) / total_mass};
    FML::FOF::SphericalOverdensity<Particle, NDIM>(
        local.data(), local.size(), total_mass, guess, initial_radius, low, search_radius);
    check(FML::FOF::SO_number_of_truncated_halos == centers.size(), label + "spheres larger than the search radius reported");

    // Resolution test: NFW-like halos with a core as large as the halo (as halos in a PM/COLA simulation where R200
    // is about the force resolution), rho ~ 1 / ((r + r_c) (r + r_s)^2) with r_s = r_c = 2 R200, truncated at 2 R200,
    // with ~100 particles inside R200. The mean density in the core is only a little above Delta, the center is not
    // on a particle and the innermost particles are Poisson noise, so a sphere grown from the center that stops at
    // the first particle below the threshold often has 1-2 particles. The outermost crossing gives the expected
    // M200 on average (the flat profile makes the radius of each halo noisy)
    {
        const double R200 = 0.02, rs = 2.0 * R200, rc = rs, rt = 2.0 * R200, Delta = 200.0;
        const int nbg = 10000;
        // Halos on the domain boundaries for 1, 2 and 4 tasks and in between; > 2 rt apart
        std::vector<std::array<double, 3>> cnfw;
        for (int ix = 0; ix < 4; ix++)
            for (int iy = 0; iy < 3; iy++)
                cnfw.push_back({0.25 * ix, 0.2 + 0.3 * iy, 0.15 + 0.2 * ((ix + iy) % 4)});
        // Tabulated enclosed mass of the profile (in units of the mass within rt) and its inverse
        const int ntab = 20000;
        std::vector<double> rtab(ntab + 1), mtab(ntab + 1, 0.0);
        for (int i = 0; i <= ntab; i++)
            rtab[i] = rt * i / double(ntab);
        auto shell = [&](double r) { return r * r / ((r + rc) * (r + rs) * (r + rs)); };
        for (int i = 1; i <= ntab; i++)
            mtab[i] = mtab[i - 1] + 0.5 * (shell(rtab[i - 1]) + shell(rtab[i])) * (rtab[i] - rtab[i - 1]);
        for (auto & m : mtab)
            m /= mtab.back();
        const double f200 = mtab[ntab / 2];
        // Halo particles per halo so that the expected mean density inside R200 is Delta times the mean density:
        // f200 Nh + nbg V200 = Delta V200 (nbg + nhalos Nh)
        const double V200 = 4.0 * M_PI / 3.0 * R200 * R200 * R200;
        const int Nh = int(std::lround((Delta - 1.0) * nbg * V200 / (f200 - Delta * V200 * cnfw.size())));
        std::vector<Particle> allnfw;
        std::mt19937 gen2(7);
        for (int i = 0; i < nbg; i++)
            allnfw.push_back({{uni(gen2), uni(gen2), uni(gen2)}, {0.0, 0.0, 0.0}});
        for (auto & c : cnfw)
            for (int i = 0; i < Nh; i++) {
                const double u = uni(gen2);
                const size_t k = std::upper_bound(mtab.begin(), mtab.end(), u) - mtab.begin();
                const size_t k1 = std::min(std::max(k, size_t(1)), size_t(ntab));
                const double r = rtab[k1 - 1] + (rtab[k1] - rtab[k1 - 1]) * (u - mtab[k1 - 1]) / (mtab[k1] - mtab[k1 - 1]);
                const double mu = 2.0 * uni(gen2) - 1.0, phi = 2.0 * M_PI * uni(gen2);
                const double sn = std::sqrt(1.0 - mu * mu);
                allnfw.push_back({{wrap(c[0] + r * sn * std::cos(phi)), wrap(c[1] + r * sn * std::sin(phi)), wrap(c[2] + r * mu)},
                                  {0.0, 0.0, 0.0}});
            }
        const double total = double(allnfw.size());
        const double M200 = Delta * total * V200;
        std::vector<Particle> localnfw;
        for (auto & p : allnfw)
            if (p.pos[0] >= FML::xmin_domain and p.pos[0] < FML::xmax_domain)
                localnfw.push_back(p);
        std::vector<std::array<double, 3>> gnfw;
        for (auto & c : cnfw)
            if (c[0] >= FML::xmin_domain and c[0] < FML::xmax_domain)
                gnfw.push_back({wrap(c[0] + 0.002), wrap(c[1] - 0.002), c[2]});
        std::vector<double> rinit(gnfw.size(), R200);
        auto sonfw = FML::FOF::SphericalOverdensity<Particle, NDIM>(
            localnfw.data(), localnfw.size(), total, gnfw, rinit, {Delta}, 0.1);
        double sumerr = 0.0, minfrac = 1e30;
        for (auto & h : sonfw) {
            const double err = h.mass[0] / M200 - 1.0;
            sumerr += err;
            minfrac = std::min(minfrac, double(h.npart[0]) / M200);
        }
        size_t nnfw = sonfw.size();
        FML::SumOverTasks(&sumerr);
        FML::SumOverTasks(&nnfw);
        FML::MinOverTasks(&minfrac);
        check(nnfw == cnfw.size(), label + "all cored NFW halos measured");
        check(minfrac > 0.25, label + "cored NFW halos with N200 = " + num(M200) + ": no halo collapses to a few particles (min N/N200 = " + num(minfrac) + ")");
        check(std::fabs(sumerr / double(cnfw.size())) < 0.1, label + "mean M200 of the cored NFW halos (mean rel. error " + num(sumerr / double(cnfw.size())) + ")");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All SO tests passed\n" : std::to_string(fails) + " SO test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
