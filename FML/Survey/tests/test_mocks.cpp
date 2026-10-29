// Tests of SurveyMocks.h: distances and round trips, redshifts with velocities, footprint area, radial selection,
// randoms and reproducibility over MPI tasks. Run with mpirun -np 1 and 2.
#include <FML/Global/Global.h>
#include <FML/Survey/SurveyMocks.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace FML::SURVEY::MOCKS;

namespace {
    int failures = 0;
    void check(bool ok, const std::string & label) {
        if (FML::ThisTask == 0)
            std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << "\n";
        failures += !ok;
    }
    const double c_over_H0 = speed_of_light_kms / 100.0; // Mpc/h
    double hubble_eds(double z) { return std::pow(1 + z, 1.5) / c_over_H0; }
    double hubble_lcdm(double z) { return std::sqrt(0.3 * std::pow(1 + z, 3) + 0.7) / c_over_H0; }

    // A uniform box (the same on every task)
    std::vector<MockObject> uniform_box(size_t n, double L, double sigma_v, std::uint64_t seed) {
        std::mt19937_64 g(seed);
        std::uniform_real_distribution<double> u(0.0, L);
        std::normal_distribution<double> v(0.0, sigma_v);
        std::vector<MockObject> out(n);
        for (size_t i = 0; i < n; i++) {
            out[i].position = {u(g), u(g), u(g)};
            out[i].velocity = {v(g), v(g), v(g)};
            out[i].id = i;
        }
        return out;
    }
} // namespace

int main() {
    // 1. r(z) against EdS, the inverse and the coordinate round trip
    {
        detail::DistanceRedshift dr(hubble_eds, 3.0);
        double err = 0, err_inv = 0;
        for (double z : {0.01, 0.1, 0.5, 1.0, 2.0, 2.9}) {
            const double exact = 2 * c_over_H0 * (1 - 1 / std::sqrt(1 + z));
            err = std::max(err, std::fabs(dr.r_of_z(z) / exact - 1));
            err_inv = std::max(err_inv, std::fabs(dr.z_of_r(dr.r_of_z(z)) / z - 1));
        }
        if (FML::ThisTask == 0)
            std::cout << "  r(z) relative error " << err << "\n";
        check(err < 1e-8, "r(z) = 2 c/H0 (1 - (1+z)^-1/2) in EdS");
        check(err_inv < 1e-12, "z(r(z)) = z");

        MockOptions o;
        o.boxsize = 500;
        o.observer = {250, 250, 250};
        o.rotation = rotation_from_pole({1, 1, 0}, {0, 0, 1});
        o.z_min = 0.0;
        o.z_max = 0.1;
        o.hubble_over_c = hubble_lcdm;
        o.redshift_space = false;
        o.parent_nbar = 1e-3;
        auto objs = uniform_box(2000, 500, 0, 1);
        if (FML::ThisTask != 0)
            objs.clear();
        const auto cat = make_survey(objs, o);
        detail::DistanceRedshift dr2(hubble_lcdm, 0.2);
        double e = 0;
        for (const auto & g : cat.galaxies) {
            const double r = dr2.r_of_z(g.z_cos);
            const Vec3 x{r * std::cos(g.dec * deg) * std::cos(g.ra * deg),
                         r * std::cos(g.dec * deg) * std::sin(g.ra * deg),
                         r * std::sin(g.dec * deg)};
            for (int d = 0; d < 3; d++)
                e = std::max(e, std::fabs(x[d] - g.position_real[d]));
            // back to the box frame: x_box = observer + R^T x_s, wrapped, must be the source object
            for (int d = 0; d < 3; d++) {
                double xb = o.observer[d];
                for (int k = 0; k < 3; k++)
                    xb += o.rotation[k][d] * g.position_real[k];
                xb -= g.replica[d] * o.boxsize;
                e = std::max(e, std::fabs(xb - objs[g.id].position[d]));
            }
        }
        double emax = e;
        FML::MaxOverTasks(&emax);
        check(emax < 1e-6, "(RA, Dec, z) -> Cartesian -> box position round trip (" + std::to_string(emax) + " Mpc/h)");
    }

    // 2. Velocities: z_obs = z_cos for v = 0; a receding object is redder by (1 + v_r/c)
    {
        MockOptions o;
        o.boxsize = 1000;
        o.observer = {500, 500, 500};
        o.z_max = 0.15;
        o.hubble_over_c = hubble_lcdm;
        o.parent_nbar = 1e-3;
        std::vector<MockObject> objs(2);
        objs[0].position = {700, 500, 500};
        objs[1].position = {700, 500, 500};
        objs[1].velocity = {600, 50, -30};
        objs[1].id = 1;
        if (FML::ThisTask != 0)
            objs.clear();
        const auto cat = make_survey(objs, o);
        bool ok = true;
        for (const auto & g : cat.galaxies) {
            if (g.id == 0)
                ok &= g.z_obs == g.z_cos && g.v_radial == 0;
            else
                ok &= g.v_radial == 600 &&
                      std::fabs((1 + g.z_obs) / (1 + g.z_cos) - (1 + 600 / speed_of_light_kms)) < 1e-13 &&
                      g.z_obs > g.z_cos && std::sqrt(detail::dot(g.position, g.position)) > 200;
        }
        int n = int(cat.galaxies.size());
        FML::SumOverTasks(&n);
        check(ok && n == 2, "zero velocity: z_obs = z_cos; receding: 1 + z_obs = (1 + z_cos)(1 + v_r/c), farther");
    }

    // 3-5. Footprint area, radial selection n(z), randoms (uniform box, LCDM, RA-Dec box footprint)
    {
        const double L = 1000;
        MockOptions o;
        o.boxsize = L;
        o.observer = {500, 500, 500};
        o.z_min = 0.02;
        o.z_max = 0.15;
        o.hubble_over_c = hubble_lcdm;
        o.footprint = ra_dec_box(300, 60, -10, 50); // through RA = 0
        const double parent = 4e-4;                 // 400000 objects in 1 Gpc/h^3
        o.target_nbar = [](double z) { return 1e-4 * (1 + 10 * z); };
        o.random_factor = 5;
        auto all = uniform_box(400000, L, 300, 7);
        std::vector<MockObject> mine;
        for (size_t i = 0; i < all.size(); i++)
            if (int(i % FML::NTasks) == FML::ThisTask)
                mine.push_back(all[i]);
        const auto cat = make_survey(mine, o);
        const auto ran = make_randoms(o, cat.parent_nbar);
        check(std::fabs(cat.parent_nbar / parent - 1) < 1e-12, "parent density N/L^3 summed over the tasks");

        // footprint fraction from the randoms
        const double area = 120 * deg * (std::sin(50 * deg) - std::sin(-10 * deg)) / (4 * M_PI);
        double nr = double(ran.size()), nd = double(cat.galaxies.size());
        FML::SumOverTasks(&nr);
        FML::SumOverTasks(&nd);
        // expected numbers: int n(z) dV in the footprint
        detail::DistanceRedshift dr(hubble_lcdm, 0.2);
        const int nb = 13;
        std::vector<double> expected(nb, 0.0), got(nb, 0.0), got_r(nb, 0.0);
        for (int b = 0; b < nb; b++) {
            const double z0 = o.z_min + (o.z_max - o.z_min) * b / nb, z1 = o.z_min + (o.z_max - o.z_min) * (b + 1) / nb;
            for (int k = 0; k < 100; k++) {
                const double za = z0 + (z1 - z0) * k / 100, zb = z0 + (z1 - z0) * (k + 1) / 100;
                const double ra = dr.r_of_z(za), rb = dr.r_of_z(zb);
                expected[b] += o.target_nbar(0.5 * (za + zb)) * 4 * M_PI / 3 * (rb * rb * rb - ra * ra * ra) * area;
            }
        }
        for (const auto & g : cat.galaxies)
            got[std::min(nb - 1, int((g.z_obs - o.z_min) / (o.z_max - o.z_min) * nb))] += 1;
        for (const auto & g : ran)
            got_r[std::min(nb - 1, int((g.z_obs - o.z_min) / (o.z_max - o.z_min) * nb))] += 1;
        FML::SumArrayOverTasks(got.data(), nb);
        FML::SumArrayOverTasks(got_r.data(), nb);
        double ntot_exp = 0, pull = 0, pull_r = 0;
        for (int b = 0; b < nb; b++) {
            ntot_exp += expected[b];
            pull = std::max(pull, std::fabs(got[b] - expected[b]) / std::sqrt(expected[b]));
            const double er = expected[b] * o.random_factor;
            pull_r = std::max(pull_r, std::fabs(got_r[b] - er) / std::sqrt(er));
        }
        if (FML::ThisTask == 0)
            std::cout << "  data " << nd << " (expected " << ntot_exp << "), randoms " << nr << " (expected "
                      << ntot_exp * o.random_factor << "), max |pull| of n(z): data " << pull << ", randoms " << pull_r
                      << ", footprint fraction " << area << "\n";
        check(pull < 4.5, "data n(z) = target n(z) in the footprint (Poisson)");
        check(pull_r < 4.5, "randoms n(z) = random_factor x target in the footprint (Poisson)");
        check(std::fabs(nr / (o.random_factor * ntot_exp) - 1) < 5 / std::sqrt(nr), "randoms: footprint area");
        bool in_footprint = true, nbar_ok = true;
        for (const auto & g : cat.galaxies) {
            in_footprint &= o.footprint(g.ra, g.dec) == 1.0;
            nbar_ok &= std::fabs(g.nbar - o.target_nbar(g.z_obs)) < 1e-15;
        }
        check(in_footprint && nbar_ok, "data inside the footprint, nbar = target n(z_obs)");
        // feasibility
        MockOptions bad = o;
        bad.target_nbar = [](double) { return 1e-3; };
        bool threw = false;
        try {
            make_survey(mine, bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "a target density above the box density is an error");

        // 7. reproducibility: the distributed catalogue = the serial one (order independent checksum)
        auto checksum = [](const std::vector<MockGalaxy> & gs) {
            double s = 0;
            for (const auto & g : gs)
                s += std::fmod(double(g.id) * 0.6180339887 + g.replica[0] * 0.1 + g.replica[1] * 0.01, 1.0) + g.z_obs;
            return s;
        };
        double distributed = checksum(cat.galaxies);
        FML::SumOverTasks(&distributed);
        MockOptions serial = o;
        serial.parent_nbar = cat.parent_nbar; // local: no reduction
        const auto ref = make_survey(all, serial);
        const double reference = checksum(ref.galaxies);
        check(std::fabs(distributed - reference) < 1e-9 * std::fabs(reference) && double(ref.galaxies.size()) == nd,
              "the catalogue does not depend on the number of MPI tasks");
    }

    if (FML::ThisTask == 0)
        std::cout << (failures ? "Some survey mock tests FAILED\n" : "All survey mock tests passed\n");
    return failures ? 1 : 0;
}
