#ifndef EXAMPLEDATA_HEADER
#define EXAMPLEDATA_HEADER

//==========================================================================================
// Synthetic input data for the examples, so they run without any data files
//
//  clustered_box      A toy clustered particle box: halos (Gaussian clouds with a power-law
//                     distribution of particle numbers) and a uniform background with spherical
//                     voids cut out. Positions, peculiar velocities and the voids
//  redshift_space     Positions moved along an axis by their velocities (redshift space)
//  survey_catalogue   Galaxies (clustered) and randoms in a patch of the sky: DEC, RA, z
//  write_ramses_snapshot  A RAMSES particle snapshot (info, header and part files) of particles
//
// Every task makes the same data (the random numbers are portable: the same on every platform),
// so an example can create its MPIParticles with Input::Replicated. It is meant for trying out the
// methods (halos to find, voids to find, a clustered P(k) and correlation function), not for
// science: it is not a cosmological simulation.
//==========================================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace FML {
    namespace EXAMPLEDATA {

        /// Portable random numbers (splitmix64): the same sequence on every platform and compiler (the std::
        /// distributions are not)
        class Random {
          public:
            explicit Random(uint64_t seed) : state(seed) {}
            uint64_t next() {
                uint64_t z = (state += 0x9E3779B97F4A7C15ull);
                z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                return z ^ (z >> 31);
            }
            /// Uniform in [0, 1)
            double uniform() { return double(next() >> 11) / double(1ull << 53); }
            /// Uniform in [a, b)
            double uniform(double a, double b) { return a + (b - a) * uniform(); }
            /// Standard normal (Box-Muller)
            double normal() {
                const double u1 = 1.0 - uniform(), u2 = uniform();
                return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
            }

          private:
            uint64_t state;
        };

        /// A particle: position in [0, box)^3 and peculiar velocity in km/s
        struct ExampleParticle {
            std::array<double, 3> pos;
            std::array<double, 3> vel;
        };

        /// A void cut out of the background: center in [0, box)^3 and radius
        struct ExampleVoid {
            std::array<double, 3> pos;
            double radius;
        };

        struct ClusteredBox {
            double box;
            std::vector<ExampleParticle> particles;
            std::vector<ExampleVoid> voids;
            size_t nhalos{0};
            size_t npart_in_halos{0};
        };

        /// A toy clustered box of about npart particles in [0, box)^3 (same units as box, e.g. Mpc/h) with peculiar
        /// velocities (km/s). Half of the particles are in halos: Gaussian clouds of N = 20 - 2000 particles (dN/dlnN
        /// ~ 1/N) of size 2 (N/1000)^(1/3) Mpc/h x box/1024, a bulk velocity of 300 km/s and a velocity dispersion of
        /// 400 (N/1000)^(1/3) km/s per axis. The other half is a uniform background (200 km/s per axis) with nvoids
        /// spherical voids of radius 3 - 6 % of the box cut out (no halos there either)
        inline ClusteredBox clustered_box(size_t npart = 100000, double box = 1024.0, int nvoids = 30,
                                          uint64_t seed = 1234) {
            ClusteredBox result;
            result.box = box;
            Random rng(seed);

            // The voids
            for (int i = 0; i < nvoids; i++)
                result.voids.push_back(
                    {{rng.uniform(0, box), rng.uniform(0, box), rng.uniform(0, box)}, box * rng.uniform(0.03, 0.06)});
            auto in_a_void = [&](const std::array<double, 3> & x) {
                for (auto & v : result.voids) {
                    double r2 = 0.0;
                    for (int d = 0; d < 3; d++) {
                        double dx = std::fabs(x[d] - v.pos[d]);
                        dx = std::min(dx, box - dx);
                        r2 += dx * dx;
                    }
                    if (r2 < v.radius * v.radius)
                        return true;
                }
                return false;
            };
            auto wrap = [&](double x) {
                x = std::fmod(x, box);
                return x < 0 ? x + box : x;
            };

            // The halos (until half of the particles are in halos)
            const double scale = box / 1024.0;
            while (result.npart_in_halos < npart / 2) {
                // N between 20 and 2000 with dN/dlnN ~ 1/N: N = 20 / (1 - u (1 - 20/2000))
                const size_t n = size_t(20.0 / (1.0 - rng.uniform() * (1.0 - 20.0 / 2000.0)));
                std::array<double, 3> center;
                do {
                    center = {rng.uniform(0, box), rng.uniform(0, box), rng.uniform(0, box)};
                } while (in_a_void(center));
                const double size = 2.0 * scale * std::cbrt(n / 1000.0);
                const double sigma_v = 400.0 * std::cbrt(n / 1000.0);
                const std::array<double, 3> bulk{300.0 * rng.normal(), 300.0 * rng.normal(), 300.0 * rng.normal()};
                for (size_t j = 0; j < n; j++) {
                    ExampleParticle p;
                    for (int d = 0; d < 3; d++) {
                        p.pos[d] = wrap(center[d] + size * rng.normal());
                        p.vel[d] = bulk[d] + sigma_v * rng.normal();
                    }
                    result.particles.push_back(p);
                }
                result.nhalos++;
                result.npart_in_halos += n;
            }

            // The background
            while (result.particles.size() < npart) {
                ExampleParticle p;
                p.pos = {rng.uniform(0, box), rng.uniform(0, box), rng.uniform(0, box)};
                if (in_a_void(p.pos))
                    continue;
                for (int d = 0; d < 3; d++)
                    p.vel[d] = 200.0 * rng.normal();
                result.particles.push_back(p);
            }
            return result;
        }

        /// The particles in redshift space: moved along the axis los (0, 1, 2) by v / (100 a E(a)) (Mpc/h for a box
        /// in Mpc/h) for flat LCDM, periodic
        inline std::vector<ExampleParticle> redshift_space(const ClusteredBox & data, double redshift = 0.42,
                                                           double OmegaM = 0.3, int los = 2) {
            const double a = 1.0 / (1.0 + redshift);
            const double aH = 100.0 * a * std::sqrt(OmegaM / (a * a * a) + 1.0 - OmegaM);
            auto result = data.particles;
            for (auto & p : result) {
                double x = std::fmod(p.pos[los] + p.vel[los] / aH, data.box);
                p.pos[los] = x < 0 ? x + data.box : x;
            }
            return result;
        }

        /// A galaxy or random in a survey: DEC and RA in degrees and the redshift
        struct SurveyObject {
            double DEC;
            double RA;
            double z;
        };

        /// Galaxies and randoms in the patch RA in [ra_min, ra_max], DEC in [dec_min, dec_max] (degrees), redshift
        /// in [zmin, zmax], uniform in comoving volume for flat LCDM (OmegaM). The galaxies are half in groups
        /// (Gaussian clouds of 2 - 20 galaxies of size 3 Mpc/h), half unclustered, so the correlation function is
        /// positive on small scales. The randoms are unclustered
        struct SurveyCatalogue {
            std::vector<SurveyObject> galaxies;
            std::vector<SurveyObject> randoms;
        };
        inline SurveyCatalogue survey_catalogue(size_t ngalaxies = 20000,
                                                size_t nrandoms = 60000,
                                                double ra_min = 150.0,
                                                double ra_max = 200.0,
                                                double dec_min = 0.0,
                                                double dec_max = 30.0,
                                                double zmin = 0.2,
                                                double zmax = 0.5,
                                                double OmegaM = 0.315,
                                                uint64_t seed = 4321) {
            Random rng(seed);
            const double deg = M_PI / 180.0;

            // Comoving distance r(z) in Mpc/h (table) and its inverse
            const int nz = 2000;
            const double c_over_H0 = 2997.92458;
            std::vector<double> ztab(nz), rtab(nz);
            const double zend = zmax * 1.5 + 0.1;
            double r = 0.0;
            for (int i = 0; i < nz; i++) {
                ztab[i] = zend * i / (nz - 1);
                if (i > 0) {
                    auto invE = [&](double zz) {
                        return 1.0 / std::sqrt(OmegaM * std::pow(1 + zz, 3) + 1.0 - OmegaM);
                    };
                    r += c_over_H0 * 0.5 * (invE(ztab[i - 1]) + invE(ztab[i])) * (ztab[i] - ztab[i - 1]);
                }
                rtab[i] = r;
            }
            auto interp = [&](const std::vector<double> & x, const std::vector<double> & y, double xx) {
                size_t i = 1;
                while (i < x.size() - 1 and x[i] < xx)
                    i++;
                const double t = (xx - x[i - 1]) / (x[i] - x[i - 1]);
                return y[i - 1] + t * (y[i] - y[i - 1]);
            };
            const double rmin = interp(ztab, rtab, zmin), rmax = interp(ztab, rtab, zmax);
            const double sdec_min = std::sin(dec_min * deg), sdec_max = std::sin(dec_max * deg);

            // A point uniform in comoving volume in the patch, as (r, RA, DEC)
            auto random_point = [&]() {
                const double rr = std::cbrt(rng.uniform(rmin * rmin * rmin, rmax * rmax * rmax));
                return std::array<double, 3>{rr, rng.uniform(ra_min, ra_max), std::asin(rng.uniform(sdec_min, sdec_max)) / deg};
            };
            auto in_patch = [&](const std::array<double, 3> & s) {
                return s[0] >= rmin and s[0] <= rmax and s[1] >= ra_min and s[1] <= ra_max and s[2] >= dec_min and
                       s[2] <= dec_max;
            };
            auto to_object = [&](const std::array<double, 3> & s) {
                return SurveyObject{s[2], s[1], interp(rtab, ztab, s[0])};
            };

            SurveyCatalogue result;
            for (size_t i = 0; i < nrandoms; i++)
                result.randoms.push_back(to_object(random_point()));

            // Unclustered half
            for (size_t i = 0; i < ngalaxies / 2; i++)
                result.galaxies.push_back(to_object(random_point()));
            // Groups: Gaussian clouds in comoving space, kept if inside the patch
            while (result.galaxies.size() < ngalaxies) {
                const auto c = random_point();
                const double x = c[0] * std::cos(c[2] * deg) * std::cos(c[1] * deg);
                const double y = c[0] * std::cos(c[2] * deg) * std::sin(c[1] * deg);
                const double z = c[0] * std::sin(c[2] * deg);
                const int n = 2 + int(rng.uniform() * 19);
                for (int j = 0; j < n and result.galaxies.size() < ngalaxies; j++) {
                    const double xx = x + 3.0 * rng.normal(), yy = y + 3.0 * rng.normal(), zz = z + 3.0 * rng.normal();
                    const double rr = std::sqrt(xx * xx + yy * yy + zz * zz);
                    std::array<double, 3> s{rr, std::atan2(yy, xx) / deg, std::asin(zz / rr) / deg};
                    if (s[1] < 0)
                        s[1] += 360.0;
                    if (in_patch(s))
                        result.galaxies.push_back(to_object(s));
                }
            }
            return result;
        }

        /// Write the particles (positions in [0, box), velocities in km/s) as a RAMSES snapshot
        /// snapdir = <folder>/output_<NNNNN> (info_<NNNNN>.txt, header_<NNNNN>.txt and ncpu files
        /// part_<NNNNN>.out<icpu>; dark matter particles with the fields POS VEL MASS ID LEVEL FAMILY TAG, as RAMSES
        /// writes them). Call it on one task
        inline void write_ramses_snapshot(const std::string & folder,
                                          int outputnr,
                                          const std::vector<ExampleParticle> & particles,
                                          double box,
                                          double aexp = 1.0,
                                          int ncpu = 4,
                                          double OmegaM = 0.3,
                                          double h = 0.7) {
            char num[16];
            snprintf(num, sizeof(num), "%05d", outputnr);
            const std::string dir = folder + "/output_" + num;
            std::filesystem::create_directories(dir);

            // Code units: lengths in units of the box, unit_l = box (cm), unit_t such that v = x_code / t_code
            const double Mpc_cm = 3.08567758e24;
            const double unit_l = box * Mpc_cm * aexp / h;
            const double unit_t = aexp * aexp * Mpc_cm / (100.0 * h * 1e5);
            const double vel_to_code = 1e5 * unit_t / unit_l; // km/s -> code units
            const int levelmin = 7;
            std::ofstream info(dir + "/info_" + num + ".txt");
            info << "ncpu        =  " << std::setw(10) << ncpu << "\nndim        =           3\nlevelmin    =  "
                 << std::setw(10) << levelmin << "\nlevelmax    =  " << std::setw(10) << levelmin
                 << "\nngridmax    =      100000\nnstep_coarse=  " << std::setw(10) << 100 << "\n\n";
            info.precision(15);
            info << std::scientific << "boxlen      =  " << 1.0 << "\ntime        =  " << -1.0
                 << "\naexp        =  " << aexp << "\nH0          =  " << 100.0 * h << "\nomega_m     =  " << OmegaM
                 << "\nomega_l     =  " << 1.0 - OmegaM << "\nomega_k     =  " << 0.0 << "\nomega_b     =  " << 0.0
                 << "\nunit_l      =  " << unit_l << "\nunit_d      =  " << 1.0 << "\nunit_t      =  " << unit_t
                 << "\n";
            info.close();
            const size_t npart = particles.size();
            std::ofstream header(dir + "/header_" + num + ".txt");
            header << " Total number of particles\n " << npart << "\n Total number of dark matter particles\n "
                   << npart << "\n Total number of star particles\n 0\n Total number of sink particles\n 0\n"
                   << " Particle fields\npos vel mass iord level family tag \n";
            header.close();

            // Fortran unformatted records
            auto record = [](std::ofstream & fp, const auto & v) {
                const int n = int(v.size() * sizeof(v[0]));
                fp.write((const char *)&n, sizeof(int));
                if (n > 0)
                    fp.write((const char *)v.data(), n);
                fp.write((const char *)&n, sizeof(int));
            };
            size_t start = 0;
            for (int icpu = 0; icpu < ncpu; icpu++) {
                const size_t n = npart / ncpu + (size_t(icpu) < npart % ncpu ? 1 : 0);
                std::ofstream fp(dir + "/part_" + num + ".out" + std::string(5 - std::to_string(icpu + 1).size(), '0') +
                                     std::to_string(icpu + 1),
                                 std::ios::binary);
                record(fp, std::vector<int>{ncpu});
                record(fp, std::vector<int>{3});
                record(fp, std::vector<int>{int(n)});
                record(fp, std::vector<int>{1, 2, 3, 4});
                record(fp, std::vector<int>{0});
                record(fp, std::vector<double>{0.0});
                record(fp, std::vector<double>{0.0});
                record(fp, std::vector<int>{0});
                std::vector<double> buf(n);
                for (int d = 0; d < 3; d++) {
                    for (size_t j = 0; j < n; j++)
                        buf[j] = particles[start + j].pos[d] / box;
                    record(fp, buf);
                }
                for (int d = 0; d < 3; d++) {
                    for (size_t j = 0; j < n; j++)
                        buf[j] = particles[start + j].vel[d] * vel_to_code;
                    record(fp, buf);
                }
                std::vector<double> mass(n, 1.0 / double(npart));
                std::vector<int> id(n), level(n, levelmin);
                std::vector<signed char> family(n, 1), tag(n, 0);
                for (size_t j = 0; j < n; j++)
                    id[j] = int(start + j + 1);
                record(fp, mass);
                record(fp, id);
                record(fp, level);
                record(fp, family);
                record(fp, tag);
                start += n;
            }
        }
    } // namespace EXAMPLEDATA
} // namespace FML
#endif
