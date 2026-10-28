//==========================================================
// Tests for FML::SURVEY (Survey/GalaxiesToBox.h):
//   - the comoving distance r(z) against independent values (scipy, flat LCDM, OmegaM = 0.315)
//   - GalaxiesToBox: positions in [0,1) and |x - observer| * boxsize = r(z) for every galaxy (1, 2, 3, 10 galaxies)
//   - GalaxiesRandomsToBox for a survey that is not in the first octant: all the points in [0,1), the same shift and
//     scale for galaxies and randoms, |x - observer| * boxsize = r(z)
//   - the largest redshift with several threads (the r(z) table must cover it)
//   - with MPI (each task with different galaxies) all the tasks get the same box
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/Survey/GalaxiesToBox.h>

#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << x;
    return os.str();
}

struct Galaxy {
    double RA, DEC, z;
    double get_RA() const { return RA; }
    double get_DEC() const { return DEC; }
    double get_z() const { return z; }
};
struct Particle {
    double pos[3];
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};

const std::function<double(double)> hubble_over_c = [](double z) {
    return 1.0 / 2997.92458 * std::sqrt(0.315 * (1 + z) * (1 + z) * (1 + z) + 0.685);
};

// r(z) in Mpc/h from scipy.integrate.quad
const std::vector<std::pair<double, double>> r_of_z_reference{{0.1, 292.59081038},
                                                              {0.5, 1315.23482867},
                                                              {1.0, 2292.45120629},
                                                              {2.0, 3580.33881452}};
double r_reference(double z) {
    for (auto & p : r_of_z_reference)
        if (p.first == z)
            return p.second;
    return -1.0;
}

// Largest relative error of |x - observer| * boxsize against r(z) and whether all the points are in [0,1)
std::pair<double, bool> check_box(const std::vector<Galaxy> & gal,
                                  std::vector<Particle> & xyz,
                                  const std::vector<double> & observer,
                                  double boxsize) {
    double maxerr = 0.0;
    bool inside = true;
    for (size_t i = 0; i < gal.size(); i++) {
        double d2 = 0.0;
        for (int k = 0; k < 3; k++) {
            d2 += std::pow(xyz[i].pos[k] - observer[k], 2);
            inside = inside and xyz[i].pos[k] >= 0.0 and xyz[i].pos[k] < 1.0;
        }
        maxerr = std::max(maxerr, std::fabs(std::sqrt(d2) * boxsize / r_reference(gal[i].z) - 1.0));
    }
    return {maxerr, inside};
}

int main() {
    std::mt19937_64 rng(42 + FML::ThisTask);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const std::vector<double> zvalues{0.1, 0.5, 1.0, 2.0};

    // GalaxiesToBox with 1 (no extent: not scaled), 2, 3 and 10 galaxies anywhere on the sky
    for (int n : {2, 3, 10}) {
        std::vector<Galaxy> gal(n);
        for (auto & g : gal)
            g = {360.0 * uniform(rng) - 180.0, 180.0 * uniform(rng) - 90.0, zvalues[size_t(4 * uniform(rng)) % 4]};
        std::vector<Particle> xyz;
        std::vector<double> observer;
        double boxsize;
        FML::SURVEY::GalaxiesToBox(gal.data(), gal.size(), xyz, hubble_over_c, boxsize, observer, true, true, false);
        auto [maxerr, inside] = check_box(gal, xyz, observer, boxsize);
        int ok = maxerr < 1e-6 and inside ? 0 : 1;
        FML::SumOverTasks(&ok);
        check(ok == 0,
              "GalaxiesToBox, " + std::to_string(n) + " galaxies: in [0,1), |x - observer| = r(z) (" + num(maxerr) +
                  ")");
    }

    // Galaxies and randoms in the southern sky around RA = 200 (not the first octant)
    {
        auto make = [&](size_t n) {
            std::vector<Galaxy> v(n);
            for (auto & g : v)
                g = {180.0 + 40.0 * uniform(rng), -60.0 + 30.0 * uniform(rng), zvalues[size_t(4 * uniform(rng)) % 4]};
            return v;
        };
        auto gal = make(500), ran = make(5000);
        std::vector<Particle> gxyz, rxyz;
        std::vector<double> observer;
        double boxsize;
        FML::SURVEY::GalaxiesRandomsToBox(gal.data(),
                                          gal.size(),
                                          ran.data(),
                                          ran.size(),
                                          gxyz,
                                          rxyz,
                                          hubble_over_c,
                                          boxsize,
                                          true,
                                          true,
                                          observer,
                                          false);
        auto [eg, ig] = check_box(gal, gxyz, observer, boxsize);
        auto [er, ir] = check_box(ran, rxyz, observer, boxsize);
        int ok = eg < 1e-6 and er < 1e-6 and ig and ir ? 0 : 1;
        FML::SumOverTasks(&ok);
        check(ok == 0,
              "GalaxiesRandomsToBox, southern survey: all in [0,1), |x - observer| = r(z) (" + num(std::max(eg, er)) +
                  ")");

        // All the tasks have the same box (each task has different galaxies)
        double bmin = boxsize, bmax = boxsize, omin = observer[0], omax = observer[0];
        FML::MinOverTasks(&bmin);
        FML::MaxOverTasks(&bmax);
        FML::MinOverTasks(&omin);
        FML::MaxOverTasks(&omax);
        check(bmin == bmax and omin == omax, "the same boxsize and observer on all the tasks");
    }

    // The largest redshift is in the r(z) table also with several threads (one galaxy at z = 2 among many at 0.1)
    {
        std::vector<Galaxy> gal(200000, Galaxy{10.0, 10.0, 0.1});
        gal[17] = {20.0, 20.0, 2.0};
        std::vector<Particle> xyz;
        std::vector<double> observer;
        double boxsize;
        FML::SURVEY::GalaxiesToBox(gal.data(), gal.size(), xyz, hubble_over_c, boxsize, observer, false, false, false);
        const double r =
            std::sqrt(std::pow(xyz[17].pos[0], 2) + std::pow(xyz[17].pos[1], 2) + std::pow(xyz[17].pos[2], 2));
        check(std::fabs(r / r_reference(2.0) - 1.0) < 1e-6, "r(z = 2) with many galaxies and threads: " + num(r));
    }

    // All galaxies at z = 0: no exception, positions at the origin
    {
        std::vector<Galaxy> gal(5, Galaxy{10.0, 10.0, 0.0});
        std::vector<Particle> xyz;
        std::vector<double> observer;
        double boxsize;
        FML::SURVEY::GalaxiesToBox(gal.data(), gal.size(), xyz, hubble_over_c, boxsize, observer, true, false, false);
        check(xyz[0].pos[0] == 0.0 and boxsize == 0.0, "all galaxies at z = 0 (no scaling)");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All survey tests passed\n" : std::to_string(fails) + " survey test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
