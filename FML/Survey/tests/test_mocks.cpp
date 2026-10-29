// Tests of SurveyMocks.h: distances and round trips, redshifts with velocities, footprint area, radial selection,
// randoms and reproducibility over MPI tasks. Run with mpirun -np 1 and 2.
#include <FML/Global/Global.h>
#include <FML/Survey/SurveyMocks.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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

    // Lightcone input: positions anywhere (Mpc/h, observer frame), the given observed redshift is used
    {
        MockOptions o;
        o.boxsize = 500;
        o.observer = {250, 250, 250};
        o.z_max = 0.2;
        o.hubble_over_c = hubble_lcdm;
        o.parent_nbar = 1e-3;
        std::vector<MockObject> objs(2);
        objs[0].position = {250 + 450, 250, 250 - 30}; // outside the box
        objs[0].velocity = {300, 0, 0};
        objs[1] = objs[0];
        objs[1].id = 1;
        objs[1].z_obs = 0.15;
        if (FML::ThisTask != 0)
            objs.clear();
        const auto cat = make_survey_lightcone(objs, o);
        detail::DistanceRedshift dr(hubble_lcdm, 0.3);
        bool ok = true;
        for (const auto & g : cat.galaxies) {
            ok &= std::fabs(g.z_cos - dr.z_of_r(std::sqrt(450.0 * 450.0 + 30.0 * 30.0))) < 1e-12;
            ok &= g.id == 0 ? g.z_obs > g.z_cos : g.z_obs == 0.15;
        }
        int n = int(cat.galaxies.size());
        FML::SumOverTasks(&n);
        bool threw = false;
        try {
            MockOptions bad = o;
            bad.parent_nbar = 0;
            make_survey_lightcone(objs, bad);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(ok && n == 2 && threw,
              "lightcone input: positions outside the box, the given z_obs; parent_nbar required");
    }

    // 8. Remapping: one to one onto the cuboid, volume preserving, uniform
    {
        bool ok = true;
        std::mt19937_64 g(11);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (auto M : {std::array<std::array<int, 3>, 3>{{{1, 1, 0}, {0, 1, 0}, {0, 0, 1}}},
                       std::array<std::array<int, 3>, 3>{{{1, 1, 0}, {1, 0, 1}, {1, 0, 0}}},
                       std::array<std::array<int, 3>, 3>{{{5, 2, 1}, {2, 1, 0}, {0, 0, 1}}}}) {
            const auto R = make_remapping(M);
            const double vol = R.lengths[0] * R.lengths[1] * R.lengths[2];
            ok &= std::fabs(vol - 1) < 1e-12;
            std::vector<double> counts(8, 0.0);
            double back = 0;
            const int n = 100000;
            for (int i = 0; i < n; i++) {
                const Vec3 x{u(g), u(g), u(g)};
                const Vec3 y = remap_position(R, x);
                int oct = 0;
                for (int a = 0; a < 3; a++) {
                    ok &= y[a] >= 0 && y[a] < R.lengths[a];
                    oct += (y[a] > 0.5 * R.lengths[a]) << a;
                }
                counts[oct] += 1;
                // back to the box: sum_a y_a e_a = x mod 1
                for (int d = 0; d < 3; d++) {
                    double xb = 0;
                    for (int a = 0; a < 3; a++)
                        xb += y[a] * R.axes[a][d];
                    xb -= std::floor(xb);
                    back = std::max(back, std::min(std::fabs(xb - x[d]), 1 - std::fabs(xb - x[d])));
                }
            }
            for (double c : counts)
                ok &= std::fabs(c - n / 8.0) < 5 * std::sqrt(n / 8.0);
            ok &= back < 1e-12;
            if (FML::ThisTask == 0)
                std::cout << "  remapping L = " << R.lengths[0] << " x " << R.lengths[1] << " x " << R.lengths[2]
                          << " (" << R.offsets.size() << " candidate cells), max |x_back - x| = " << back << "\n";
        }
        bool threw = false;
        try {
            make_remapping({{{2, 0, 0}, {0, 1, 0}, {0, 0, 1}}});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(ok && threw, "remapping: inside the cuboid, back to the box, uniform, volume 1; det != 1 rejected");

        // a deep pencil survey in a remapped cuboid (not periodic): no repetition, the right density
        const double L = 500;
        auto objs = uniform_box(200000, L, 0, 5);
        if (FML::ThisTask != 0)
            objs.clear();
        const auto R = make_remapping({{{5, 2, 1}, {2, 1, 0}, {0, 0, 1}}});
        const auto rem = remap_objects(objs, R, L);
        MockOptions o;
        o.boxsize = L;
        o.periodic = false;
        o.extent = {R.lengths[0] * L, R.lengths[1] * L, R.lengths[2] * L};
        o.observer = {0.0, 0.5 * o.extent[1], 0.5 * o.extent[2]};
        o.rotation = rotation_from_pole({1, 0, 0}, {0, 1, 0});
        o.z_min = 0;
        o.z_max = 0.6;
        o.hubble_over_c = hubble_lcdm;
        o.redshift_space = false;
        const double half = std::atan(0.5 * std::min(o.extent[1], o.extent[2]) /
                                      detail::DistanceRedshift(hubble_lcdm, 1.0).r_of_z(0.6)) /
                            deg;
        o.footprint = [half](double, double dec) { return dec > 90 - half ? 1.0 : 0.0; };
        const auto cat = make_survey(rem, o);
        int n = int(cat.galaxies.size());
        FML::SumOverTasks(&n);
        const double rmax = detail::DistanceRedshift(hubble_lcdm, 1.0).r_of_z(0.6);
        const double expected = cat.parent_nbar * 2 * M_PI * (1 - std::cos(half * deg)) * rmax * rmax * rmax / 3;
        bool unique = true;
        std::vector<std::uint64_t> ids;
        for (const auto & gal : cat.galaxies)
            ids.push_back(gal.id);
        std::sort(ids.begin(), ids.end());
        unique = std::adjacent_find(ids.begin(), ids.end()) == ids.end();
        if (FML::ThisTask == 0)
            std::cout << "  pencil to z = 0.6 (" << rmax << " Mpc/h, box " << L << "): " << n << " objects, expected "
                      << expected << ", cuboid " << o.extent[0] << " x " << o.extent[1] << " x " << o.extent[2] << "\n";
        check(unique && std::fabs(n - expected) < 5 * std::sqrt(expected) && !cat.repeated && cat.truncated,
              "remapped cuboid: a survey deeper than the box without repeated objects, the right density");
    }

    // 9. Healpix RING pixels (healpy 1.20 reference values) and a Healpix footprint
    {
        const double th[] = {2.5475802757199624,
                             2.125132487475809,
                             0.9241052335784122,
                             1.4057235974378848,
                             2.5179263078117513,
                             1.7049444329502041,
                             1.612705997937011,
                             2.3192713655718764,
                             1.0824634210636321,
                             2.4538106597779192,
                             1.7900934350922484,
                             1.5373097035102328,
                             0.0,
                             3.141592653589793,
                             0.8410686705679303,
                             2.300523984021863,
                             1.0};
        const double ph[] = {2.705715650726193,
                             3.6869641623340965,
                             4.635971544195998,
                             6.008404365323131,
                             1.785688576349746,
                             4.0749422825363135,
                             4.374454120901312,
                             1.8392187093018624,
                             0.009362470809190032,
                             6.116431295535327,
                             1.8749101801040522,
                             1.9728322346422047,
                             0.0,
                             6.0,
                             1.0,
                             2.0,
                             6.283185307178586};
        const std::int64_t ref4[] = {173, 145, 35, 87, 171, 114, 99, 156, 40, 179, 109, 77, 0, 191, 26, 157, 40};
        const std::int64_t ref1024[] = {11505196,
                                        9601379,
                                        2503630,
                                        5257037,
                                        11396447,
                                        7135840,
                                        6554403,
                                        10570062,
                                        3336198,
                                        11156462,
                                        7658694,
                                        6081798,
                                        0,
                                        12582911,
                                        2095755,
                                        10485015,
                                        2893824};
        bool ok = true;
        for (int i = 0; i < 17; i++)
            ok &= healpix_ang2pix_ring(4, th[i], ph[i]) == ref4[i] &&
                  healpix_ang2pix_ring(1024, th[i], ph[i]) == ref1024[i];
        check(ok, "Healpix ang2pix (RING) = healpy for nside = 4 and 1024");
        // the first half of the pixels of an nside = 8 map (half the sky)
        std::vector<double> map(12 * 64, 0.0);
        for (std::size_t p = 0; p < map.size() / 2; p++)
            map[p] = 1.0;
        MockOptions o;
        o.boxsize = 1000;
        o.z_max = 0.1;
        o.hubble_over_c = hubble_lcdm;
        o.footprint = healpix_footprint(map);
        o.random_factor = 1;
        const auto ran = make_randoms(o, 1e-3);
        double north = 0, all_r = double(ran.size());
        for (const auto & g : ran)
            north += map[std::size_t(healpix_ang2pix_ring(8, (90 - g.dec) * deg, g.ra * deg))] == 1.0;
        FML::SumOverTasks(&north);
        FML::SumOverTasks(&all_r);
        detail::DistanceRedshift dr(hubble_lcdm, 0.2);
        const double expected = 1e-3 * 4 * M_PI / 3 * std::pow(dr.r_of_z(0.1), 3) / 2;
        check(north == all_r && std::fabs(all_r - expected) < 5 * std::sqrt(expected),
              "Healpix footprint: randoms only in the map pixels, half the sky");
    }

    // MANGLE: an RA/Dec box (weight 0.8), a circle and a ring (a circle minus a smaller one, weight 0.5) against
    // the analytic membership of random points, for several lookup-grid cell sizes
    {
        auto vec = [](double ra, double dec) {
            return std::array<double, 3>{std::cos(dec * deg) * std::cos(ra * deg),
                                         std::cos(dec * deg) * std::sin(ra * deg),
                                         std::sin(dec * deg)};
        };
        auto cap = [&](double ra, double dec, double cm) {
            const auto v = vec(ra, dec);
            std::ostringstream ss;
            ss << std::setprecision(17) << " " << v[0] << " " << v[1] << " " << v[2] << " " << cm << "\n";
            return ss.str();
        };
        const std::string file = "test_mangle_" + std::to_string(FML::ThisTask) + ".ply";
        {
            std::ofstream fp(file);
            fp << "3 polygons\npixelization 0s\nsnapped\nbalkanized\n";
            fp << "polygon 0 ( 4 caps, 0.8 weight, 0 pixel, 0.2 str):\n"
               << cap(0, 90, 1 - std::sin(10 * deg)) << cap(0, 90, -(1 - std::sin(40 * deg))) << cap(210, 0, 1)
               << cap(110, 0, 1);
            fp << "polygon 1 ( 1 caps, 1 weight, 0.095 str):\n" << cap(300, -30, 1 - std::cos(10 * deg));
            fp << "polygon 2 ( 2 caps, 0.5 weight, 0 pixel, 0.19 str):\n"
               << cap(60, 0, 1 - std::cos(15 * deg)) << cap(60, 0, -(1 - std::cos(5 * deg)));
        }
        auto angle = [&](double ra1, double dec1, double ra2, double dec2) {
            const auto a = vec(ra1, dec1), b = vec(ra2, dec2);
            return std::acos(std::min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])) / deg;
        };
        auto exact = [&](double ra, double dec) {
            if (ra > 120 && ra < 200 && dec > 10 && dec < 40)
                return 0.8;
            if (angle(ra, dec, 300, -30) < 10)
                return 1.0;
            const double t = angle(ra, dec, 60, 0);
            return t < 15 && t > 5 ? 0.5 : 0.0;
        };
        bool ok = true;
        for (double cell : {1.0, 7.0, 90.0}) {
            const auto mask = read_mangle_polygons(file, cell);
            const auto fp = mangle_footprint(mask);
            std::mt19937 gen(5);
            std::uniform_real_distribution<double> u(0, 1);
            int inside = 0;
            for (int i = 0; i < 200000; i++) {
                const double ra = 360 * u(gen), dec = std::asin(2 * u(gen) - 1) / deg;
                ok &= fp(ra, dec) == exact(ra, dec);
                inside += exact(ra, dec) > 0;
            }
            ok &= inside > 1000 && mask.polygons.size() == 3 && mask.polygons[2].area_str == 0.19;
        }
        bool threw = false;
        {
            std::ofstream fp(file);
            fp << "polygon 0 ( 1 caps, 2 weight, 0 pixel, 0.1 str):\n" << cap(0, 0, 0.1);
        }
        try {
            read_mangle_polygons(file);
        } catch (const std::exception &) {
            threw = true;
        }
        std::remove(file.c_str());
        check(ok && threw, "MANGLE polygons = analytic box/circle/ring membership (3 grid sizes); weight > 1 rejected");
    }

    if (FML::ThisTask == 0)
        std::cout << (failures ? "Some survey mock tests FAILED\n" : "All survey mock tests passed\n");
    return failures ? 1 : 0;
}
