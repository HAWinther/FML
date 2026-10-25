//==========================================================
// Tests for the spherical overdensity properties of halos (FriendsOfFriends/SphericalOverdensity.h): singular
// isothermal spheres (mean enclosed density 3 N / (4 pi R_max r^2)) on a uniform background, some of them on the
// boundaries of the task domains, must give the analytic radius, mass and center for any number of tasks, and
// spheres larger than the search radius must be reported as truncated.
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FriendsOfFriends/SphericalOverdensity.h>
#include <FML/Global/Global.h>

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

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All SO tests passed\n" : std::to_string(fails) + " SO test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
