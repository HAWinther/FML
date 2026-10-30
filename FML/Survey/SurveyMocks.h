#ifndef FML_SURVEY_MOCKS_HEADER
#define FML_SURVEY_MOCKS_HEADER

// =============================================================================
// Survey mocks from periodic simulation output (as make_survey): snapshot projection of a periodic box around an
// observer, RA/Dec and redshifts with RSD, an angular footprint (callback, RA-Dec box, Healpix map), a radial
// selection and matching randoms; optionally a volume-preserving cuboid remapping of the box. MANGLE polygon
// footprints; ascii or (USE_HDF5) HDF5 output.
//
//   * Snapshot projection: one snapshot is tiled (replicas) around the observer; every object keeps the dynamics of
//     the snapshot while its redshift comes from its distance. This is not an evolving lightcone.
//   * Units: comoving lengths in Mpc/h, peculiar velocities in km/s, H(z)/c in h/Mpc (user supplied), RA/Dec in
//     degrees. The survey frame is x_s = R (x - observer) (rows of R = the survey axes; Dec = 90 along z_s).
//   * Redshifts: z_cos from inverting r(z) = int dz c/H; 1 + z_obs = (1 + z_cos)(1 + v_r/c) with v_r = v . r_hat
//     (receding: redder). The redshift-space position is r(z_obs) r_hat. With redshift_space = false z_obs = z_cos.
//   * Selection (per object, never applied twice):
//       selection_probability = target_nbar(z_obs) / parent_nbar x (apply_completeness ? completeness : 1),
//     the probability with which the object was kept (an error if it exceeds 1: the target is not feasible);
//       completeness = footprint(ra, dec), stored also when it is not used for the downsampling;
//       nbar = expected density of the final catalogue at the object (for FKP weights and I2 in SurveyPower).
//     Without a target n(z) all objects in the redshift range (and, with apply_completeness, in the footprint) are
//     kept and nbar = parent_nbar x completeness. parent_nbar is the mean density of the box (N/L^3 over all tasks,
//     or MockOptions::parent_nbar).
//   * Redshift-space distortions assume |v_r| < 0.03 c (objects are searched in the real-space shell that can reach
//     the redshift range with such velocities).
//   * Randoms: angles uniform on the sphere, distances from n(z) r^2 dr (n = target, or parent), the same footprint
//     and nbar; random_factor randoms per expected data object.
//   * Reproducible: every random decision uses a Philox stream keyed by (seed, object id, replica), so the
//     catalogue does not depend on the number of MPI tasks or threads. Randoms are keyed by their index.
//   * MPI: every task projects its own objects (no gathers; only the parent density is summed over tasks). Randoms
//     are split over the tasks by index.
//   * Tiling repeats structures when r_max > L/2 (MockCatalog::repeated is then true; each object keeps its source
//     id and replica index).
// =============================================================================

#include <FML/Global/Global.h>
#include <FML/ParticleTypes/ReflectOnParticleMethods.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#ifdef USE_HEALPIX
#include <healpix_map.h>
#include <healpix_map_fitsio.h>
#endif
#ifdef USE_HDF5
#include <hdf5.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::SURVEY::MOCKS {

    using Vec3 = std::array<double, 3>;
    constexpr double speed_of_light_kms = 299792.458;
    constexpr double deg = 3.14159265358979323846 / 180.0;

    /// An object of the periodic box: position in [0, L) (Mpc/h), peculiar velocity (km/s), a unique id
    /// (with subid, e.g. the ordinal of an HOD galaxy in its halo) and a weight carried to the output
    struct MockObject {
        Vec3 position{}, velocity{};
        std::uint64_t id = 0;
        std::uint32_t subid = 0;
        double weight = 1.0;
        double z_obs = -1; // >= 0: the observed redshift is given (lightcone output with relativistic effects)
    };

    struct MockGalaxy {
        double ra = 0, dec = 0, z_cos = 0, z_obs = 0;
        Vec3 position{};      // survey frame, observer at the origin, from z_obs (redshift space)
        Vec3 position_real{}; // survey frame, real space (from z_cos)
        double v_radial = 0;  // km/s
        double completeness = 1, selection_probability = 1, nbar = 0, weight = 1;
        std::uint64_t id = 0;
        std::uint32_t subid = 0;
        std::array<int, 3> replica{};
    };

    struct MockCatalog {
        std::vector<MockGalaxy> galaxies;
        double parent_nbar = 0;
        double max_selection_probability = 0;
        bool repeated = false;  // periodic and r_max > L/2: the same structures appear more than once
        bool truncated = false; // not periodic and the sphere r_max around the observer leaves the volume
    };

    /// The angular footprint: completeness in [0, 1] as a function of (RA, Dec) in degrees
    using Footprint = std::function<double(double, double)>;
    inline Footprint full_sky() {
        return [](double, double) { return 1.0; };
    }
    /// RA in [ra_min, ra_max] (wrapping through 360 if ra_min > ra_max) and Dec in [dec_min, dec_max]
    inline Footprint ra_dec_box(double ra_min, double ra_max, double dec_min, double dec_max) {
        return [=](double ra, double dec) {
            const bool in_ra = ra_min <= ra_max ? (ra >= ra_min && ra <= ra_max) : (ra >= ra_min || ra <= ra_max);
            return in_ra && dec >= dec_min && dec <= dec_max ? 1.0 : 0.0;
        };
    }

    struct MockOptions {
        double boxsize = 0;   // Mpc/h
        bool periodic = true; // false: positions in [0, extent) (e.g. a remapped cuboid, remap_objects), no tiling
        Vec3 extent{};        // the non-periodic volume (Mpc/h)
        Vec3 observer{};      // observer in the box or volume (Mpc/h)
        std::array<Vec3, 3> rotation{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; // rows: the survey axes (orthonormal)
        double z_min = 0, z_max = 0;                                     // redshift range (of z_obs)
        std::function<double(double)> hubble_over_c;                     // H(z)/c in h/Mpc
        bool redshift_space = true;
        Footprint footprint = full_sky();
        bool apply_completeness = true;            // downsample by the completeness
        std::function<double(double)> target_nbar; // n(z) in (h/Mpc)^3 of the final catalogue; empty: no downsampling
        double parent_nbar = 0;                    // density of the box; 0: N/L^3 summed over the tasks
        double random_factor = 10;                 // randoms per expected data object
        std::uint64_t seed = 1, random_seed = 2;
        std::uint32_t stream = 0;
    };

    namespace detail {
        inline double dot(const Vec3 & a, const Vec3 & b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

        /// r(z) on a fine uniform grid in z (Simpson in every cell) with linear interpolation both ways
        class DistanceRedshift {
            std::vector<double> z, r;

          public:
            DistanceRedshift(const std::function<double(double)> & hubble_over_c, double z_max, int n = 200000) {
                if (!hubble_over_c || !(z_max > 0) || n < 2)
                    throw std::invalid_argument("DistanceRedshift: need H(z)/c and z_max > 0");
                z.resize(n);
                r.resize(n);
                const double dz = z_max / (n - 1);
                for (int i = 0; i < n; i++) {
                    z[i] = i * dz;
                    if (i == 0)
                        continue;
                    const double h0 = hubble_over_c(z[i - 1]), hm = hubble_over_c(z[i] - 0.5 * dz),
                                 h1 = hubble_over_c(z[i]);
                    if (!(h0 > 0) || !(hm > 0) || !(h1 > 0))
                        throw std::invalid_argument("DistanceRedshift: H(z)/c must be positive");
                    r[i] = r[i - 1] + dz / 6.0 * (1.0 / h0 + 4.0 / hm + 1.0 / h1);
                }
            }
            double z_max() const { return z.back(); }
            double r_max() const { return r.back(); }
            double r_of_z(double zz) const { return interpolate(z, r, zz); }
            double z_of_r(double rr) const { return interpolate(r, z, rr); }
            static double interpolate(const std::vector<double> & x, const std::vector<double> & y, double v) {
                if (!(v >= x.front()) || !(v <= x.back()))
                    throw std::out_of_range("DistanceRedshift: outside the tabulated range");
                size_t i = size_t(std::upper_bound(x.begin(), x.end(), v) - x.begin());
                i = std::min(std::max(i, size_t(1)), x.size() - 1);
                const double t = (v - x[i - 1]) / (x[i] - x[i - 1]);
                return y[i - 1] + t * (y[i] - y[i - 1]);
            }
        };

        // RA in [0, 360), Dec in [-90, 90] of a direction in the survey frame
        inline void ra_dec(const Vec3 & n, double & ra, double & dec) {
            dec = std::asin(std::clamp(n[2], -1.0, 1.0)) / deg;
            ra = std::atan2(n[1], n[0]) / deg;
            if (ra < 0)
                ra += 360.0;
            if (ra >= 360.0)
                ra -= 360.0;
        }

        inline std::uint32_t replica_tag(const std::array<int, 3> & rep, std::uint32_t purpose) {
            const std::uint32_t code =
                std::uint32_t((rep[0] + 1024) * 2048u * 2048u + (rep[1] + 1024) * 2048u + (rep[2] + 1024));
            return purpose ^ (code * 2654435761u);
        }

        inline void check_options(const MockOptions & o) {
            const bool unbounded = !o.periodic && o.extent[0] == 0 && o.extent[1] == 0 && o.extent[2] == 0;
            if (!o.periodic && !unbounded && !(o.extent[0] > 0 && o.extent[1] > 0 && o.extent[2] > 0))
                throw std::invalid_argument("make_survey: a non-periodic volume needs its extent");
            if (unbounded && !(o.parent_nbar > 0))
                throw std::invalid_argument("make_survey: lightcone input (no volume) needs parent_nbar");
            if (!(o.boxsize > 0) || !(o.z_max > o.z_min) || !(o.z_min >= 0) || !o.hubble_over_c || !o.footprint ||
                !(o.random_factor >= 0))
                throw std::invalid_argument("make_survey: need boxsize > 0, 0 <= z_min < z_max, H(z)/c, a footprint");
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (std::fabs(dot(o.rotation[i], o.rotation[j]) - (i == j ? 1.0 : 0.0)) > 1e-10)
                        throw std::invalid_argument("make_survey: the rotation must be orthonormal");
        }
    } // namespace detail

    /// Rotation with the survey pole (Dec = 90) along the unit vector `pole` and RA = 0 towards the projection of
    /// `ra_zero` on the plane normal to it
    inline std::array<Vec3, 3> rotation_from_pole(Vec3 pole, Vec3 ra_zero) {
        auto normalize = [](Vec3 v) {
            const double n = std::sqrt(detail::dot(v, v));
            if (!(n > 0))
                throw std::invalid_argument("rotation_from_pole: zero vector");
            return Vec3{v[0] / n, v[1] / n, v[2] / n};
        };
        const Vec3 z = normalize(pole);
        const double p = detail::dot(ra_zero, z);
        const Vec3 x = normalize({ra_zero[0] - p * z[0], ra_zero[1] - p * z[1], ra_zero[2] - p * z[2]});
        const Vec3 y{z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]};
        return {x, y, z};
    }

    /// The mock catalogue of the local objects (collective: the parent density is summed over the tasks)
    inline MockCatalog make_survey(const std::vector<MockObject> & objects, const MockOptions & o) {
        detail::check_options(o);
        const double L = o.boxsize;
        // r(z) beyond z_max for the redshift-space distortions (|v_r|/c < 0.03 assumed)
        const detail::DistanceRedshift dr(o.hubble_over_c, 1.1 * (1.0 + o.z_max) - 1.0 + 0.05);
        const double r_min = dr.r_of_z(o.z_min), r_max = dr.r_of_z(o.z_max);
        // real-space shell that can reach [z_min, z_max] in redshift space
        const double vmax_over_c = 0.03;
        const double r_lo = o.redshift_space ? dr.r_of_z(std::max(0.0, (1 + o.z_min) / (1 + vmax_over_c) - 1)) : r_min;
        const double r_hi = o.redshift_space ? dr.r_of_z((1 + o.z_max) * (1 + vmax_over_c) - 1) : r_max;

        MockCatalog cat;
        const Vec3 extent = o.periodic ? Vec3{L, L, L} : o.extent;
        const bool unbounded = !o.periodic && extent[0] == 0 && extent[1] == 0 && extent[2] == 0;
        cat.repeated = o.periodic && r_max > 0.5 * L;
        for (int d = 0; d < 3; d++)
            cat.truncated = cat.truncated || (!o.periodic && !unbounded &&
                                              (o.observer[d] - r_max < 0 || o.observer[d] + r_max > extent[d]));
        if (o.parent_nbar > 0) {
            cat.parent_nbar = o.parent_nbar;
        } else {
            double ntot = double(objects.size());
            FML::SumOverTasks(&ntot);
            cat.parent_nbar = ntot / (extent[0] * extent[1] * extent[2]);
        }
        const int nrep = o.periodic ? int(std::ceil(r_hi / L)) + 1 : 0;

        for (const auto & obj : objects) {
            for (int d = 0; d < 3; d++)
                if (!((obj.position[d] >= 0 && obj.position[d] < extent[d]) ||
                      (unbounded && std::isfinite(obj.position[d]))) ||
                    !std::isfinite(obj.velocity[d]))
                    throw std::invalid_argument("make_survey: positions must be in [0, boxsize) (periodic) or [0, "
                                                "extent) (not periodic), velocities finite");
            for (int ix = -nrep; ix <= nrep; ix++)
                for (int iy = -nrep; iy <= nrep; iy++)
                    for (int iz = -nrep; iz <= nrep; iz++) {
                        const Vec3 rel{obj.position[0] + ix * L - o.observer[0],
                                       obj.position[1] + iy * L - o.observer[1],
                                       obj.position[2] + iz * L - o.observer[2]};
                        const double r = std::sqrt(detail::dot(rel, rel));
                        if (!(r >= r_lo && r <= r_hi) || !(r > 0))
                            continue;
                        const Vec3 xs{detail::dot(o.rotation[0], rel),
                                      detail::dot(o.rotation[1], rel),
                                      detail::dot(o.rotation[2], rel)};
                        const Vec3 vs{detail::dot(o.rotation[0], obj.velocity),
                                      detail::dot(o.rotation[1], obj.velocity),
                                      detail::dot(o.rotation[2], obj.velocity)};
                        const Vec3 n{xs[0] / r, xs[1] / r, xs[2] / r};
                        MockGalaxy g;
                        g.z_cos = dr.z_of_r(r);
                        g.v_radial = detail::dot(vs, n);
                        g.z_obs = !o.redshift_space ? g.z_cos :
                                  obj.z_obs >= 0    ? obj.z_obs :
                                                      g.z_cos + (1 + g.z_cos) * g.v_radial / speed_of_light_kms;
                        if (!(g.z_obs >= o.z_min && g.z_obs <= o.z_max))
                            continue;
                        detail::ra_dec(n, g.ra, g.dec);
                        g.completeness = o.footprint(g.ra, g.dec);
                        if (!(g.completeness >= 0 && g.completeness <= 1))
                            throw std::invalid_argument("make_survey: the footprint must return values in [0, 1]");
                        const double used_c = o.apply_completeness ? g.completeness : 1.0;
                        const double radial = o.target_nbar ? o.target_nbar(g.z_obs) / cat.parent_nbar : 1.0;
                        g.selection_probability = radial * used_c;
                        cat.max_selection_probability =
                            std::max(cat.max_selection_probability, g.selection_probability);
                        g.nbar = (o.target_nbar ? o.target_nbar(g.z_obs) : cat.parent_nbar) * used_c;
                        g.replica = {ix, iy, iz};
                        FML::RANDOM::COUNTER_BASED::Stream rng(o.seed ^ (std::uint64_t(obj.subid) << 40),
                                                               obj.id,
                                                               detail::replica_tag(g.replica, 0x53454C31u),
                                                               o.stream);
                        if (!(rng.uniform() < g.selection_probability))
                            continue;
                        const double rs = dr.r_of_z(g.z_obs);
                        for (int d = 0; d < 3; d++) {
                            g.position_real[d] = xs[d];
                            g.position[d] = rs * n[d];
                        }
                        g.weight = obj.weight;
                        g.id = obj.id;
                        g.subid = obj.subid;
                        cat.galaxies.push_back(g);
                    }
        }
        double pmax = cat.max_selection_probability;
        FML::MaxOverTasks(&pmax);
        cat.max_selection_probability = pmax;
        if (pmax > 1.0 + 1e-12)
            throw std::invalid_argument("make_survey: target_nbar exceeds the density of the box (selection "
                                        "probability " +
                                        std::to_string(pmax) + " > 1): the target is not feasible");
        return cat;
    }

    /// A survey from lightcone output: positions in Mpc/h in the frame of MockOptions::observer (not periodic, no
    /// tiling; each object has its own crossing time, so this is an evolving lightcone), velocities in km/s and,
    /// if known, the observed redshift (MockObject::z_obs, e.g. with the gravitational redshift). parent_nbar (the
    /// density of the simulation) must be given
    inline MockCatalog make_survey_lightcone(const std::vector<MockObject> & objects, MockOptions o) {
        o.periodic = false;
        o.extent = {0, 0, 0};
        return make_survey(objects, o);
    }

    /// Randoms with the footprint and radial selection of make_survey (no velocities: z_obs = z_cos). Each task
    /// makes its share of the candidate indices; the result does not depend on the number of tasks
    inline std::vector<MockGalaxy> make_randoms(const MockOptions & o, double parent_nbar) {
        detail::check_options(o);
        const detail::DistanceRedshift dr(o.hubble_over_c, o.z_max * 1.001 + 1e-6);
        // cumulative number of the full-sky selection in r: dN = n(z(r)) 4 pi r^2 dr
        const int nr = 20000;
        const double r_min = dr.r_of_z(o.z_min), r_max = dr.r_of_z(o.z_max);
        auto n_of_z = [&](double z) { return o.target_nbar ? o.target_nbar(z) : parent_nbar; };
        std::vector<double> r(nr), N(nr, 0.0);
        for (int i = 0; i < nr; i++) {
            r[i] = r_min + (r_max - r_min) * i / (nr - 1.0);
            if (i > 0) {
                const double rm = 0.5 * (r[i] + r[i - 1]);
                auto f = [&](double x) { return n_of_z(dr.z_of_r(x)) * 4.0 * M_PI * x * x; };
                N[i] = N[i - 1] + (r[i] - r[i - 1]) / 6.0 * (f(r[i - 1]) + 4.0 * f(rm) + f(r[i]));
            }
        }
        const double n_full_sky = N.back();
        if (!(n_full_sky > 0))
            throw std::invalid_argument("make_randoms: no objects expected in the redshift range");
        const std::uint64_t ncand = std::uint64_t(std::llround(o.random_factor * n_full_sky));
        const std::uint64_t first = ncand * std::uint64_t(FML::ThisTask) / std::uint64_t(FML::NTasks),
                            last = ncand * std::uint64_t(FML::ThisTask + 1) / std::uint64_t(FML::NTasks);
        std::vector<MockGalaxy> out;
        for (std::uint64_t i = first; i < last; i++) {
            FML::RANDOM::COUNTER_BASED::Stream rng(o.random_seed, i, 0x52414E44u, o.stream);
            const double mu = 2.0 * rng.uniform() - 1.0, phi = 2.0 * M_PI * rng.uniform();
            const double s = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            const Vec3 n{s * std::cos(phi), s * std::sin(phi), mu};
            MockGalaxy g;
            detail::ra_dec(n, g.ra, g.dec);
            g.completeness = o.footprint(g.ra, g.dec);
            const double used_c = o.apply_completeness ? g.completeness : 1.0;
            g.selection_probability = used_c;
            if (!(rng.uniform() < used_c))
                continue;
            const double rr = detail::DistanceRedshift::interpolate(N, r, rng.uniform() * n_full_sky);
            g.z_cos = g.z_obs = dr.z_of_r(rr);
            g.nbar = n_of_z(g.z_obs) * used_c;
            for (int d = 0; d < 3; d++)
                g.position[d] = g.position_real[d] = rr * n[d];
            g.id = i;
            out.push_back(g);
        }
        return out;
    }

    //=========================================================================
    // Volume-preserving cuboid remapping (Carlson & White 2010, BoxRemap). An integer matrix M with det = 1 maps
    // the periodic unit box one to one onto a cuboid with edges L1 = |u1|, L2 = |u2 - (u2.e1) e1|, L3 = 1/(L1 L2)
    // along the Gram-Schmidt axes e1, e2, e3 of its rows u1, u2, u3 (box units), so a deep or wide survey needs
    // no repetition. remap_objects returns the objects in the cuboid (Mpc/h, velocities in its axes); use them with
    // MockOptions::periodic = false and extent = Remapping::lengths x boxsize.
    //=========================================================================
    struct Remapping {
        std::array<Vec3, 3> axes{};              // e1, e2, e3
        Vec3 lengths{};                          // L1, L2, L3 (box units; L1 L2 L3 = 1)
        std::vector<std::array<int, 3>> offsets; // unit cells that intersect the cuboid
    };

    inline Remapping make_remapping(const std::array<std::array<int, 3>, 3> & M) {
        const long long det = (long long)M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
                              (long long)M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
                              (long long)M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        if (det != 1 && det != -1)
            throw std::invalid_argument("make_remapping: the matrix must be integer with determinant +-1");
        Remapping R;
        const Vec3 u1{double(M[0][0]), double(M[0][1]), double(M[0][2])};
        const Vec3 u2{double(M[1][0]), double(M[1][1]), double(M[1][2])};
        const double l1 = std::sqrt(detail::dot(u1, u1));
        const Vec3 e1{u1[0] / l1, u1[1] / l1, u1[2] / l1};
        const double p = detail::dot(u2, e1);
        const Vec3 w{u2[0] - p * e1[0], u2[1] - p * e1[1], u2[2] - p * e1[2]};
        const double l2 = std::sqrt(detail::dot(w, w));
        const Vec3 e2{w[0] / l2, w[1] / l2, w[2] / l2};
        const Vec3 e3{e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        R.axes = {e1, e2, e3};
        R.lengths = {l1, l2, 1.0 / (l1 * l2)};
        // unit cells n + [0,1)^3 whose projections overlap the cuboid on all three axes (a superset of the cells
        // that intersect it; the point test below decides)
        Vec3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
        for (int c = 0; c < 8; c++) {
            Vec3 x{};
            for (int k = 0; k < 3; k++)
                for (int d = 0; d < 3; d++)
                    x[d] += ((c >> k) & 1) * R.lengths[k] * R.axes[k][d];
            for (int d = 0; d < 3; d++) {
                lo[d] = std::min(lo[d], x[d]);
                hi[d] = std::max(hi[d], x[d]);
            }
        }
        for (int i = int(std::floor(lo[0])) - 1; i <= int(std::floor(hi[0])); i++)
            for (int j = int(std::floor(lo[1])) - 1; j <= int(std::floor(hi[1])); j++)
                for (int k = int(std::floor(lo[2])) - 1; k <= int(std::floor(hi[2])); k++) {
                    bool overlap = true;
                    for (int a = 0; a < 3 && overlap; a++) {
                        double pmin = 1e30, pmax = -1e30;
                        for (int c = 0; c < 8; c++) {
                            const Vec3 x{i + double(c & 1), j + double((c >> 1) & 1), k + double((c >> 2) & 1)};
                            const double q = detail::dot(x, R.axes[a]);
                            pmin = std::min(pmin, q);
                            pmax = std::max(pmax, q);
                        }
                        overlap = pmax > 0 && pmin < R.lengths[a];
                    }
                    if (overlap)
                        R.offsets.push_back({i, j, k});
                }
        return R;
    }

    /// The cuboid coordinates (box units, in [0, L1) x [0, L2) x [0, L3)) of a point x in [0, 1)^3
    inline Vec3 remap_position(const Remapping & R, const Vec3 & x) {
        for (double tol : {0.0, 1e-12, 1e-9}) {
            for (const auto & n : R.offsets) {
                const Vec3 xn{x[0] + n[0], x[1] + n[1], x[2] + n[2]};
                Vec3 y{};
                bool in = true;
                for (int a = 0; a < 3 && in; a++) {
                    y[a] = detail::dot(xn, R.axes[a]);
                    in = y[a] >= -tol && y[a] < R.lengths[a] + tol;
                }
                if (in) {
                    for (int a = 0; a < 3; a++)
                        y[a] = std::min(std::max(y[a], 0.0), std::nextafter(R.lengths[a], 0.0));
                    return y;
                }
            }
        }
        throw std::runtime_error("remap_position: no cell found (not a point of the unit box?)");
    }

    /// The objects in the remapped cuboid (positions in Mpc/h, velocities along the cuboid axes)
    inline std::vector<MockObject>
    remap_objects(const std::vector<MockObject> & objects, const Remapping & R, double boxsize) {
        std::vector<MockObject> out(objects);
        for (auto & o : out) {
            const Vec3 y =
                remap_position(R, {o.position[0] / boxsize, o.position[1] / boxsize, o.position[2] / boxsize});
            const Vec3 v = o.velocity;
            for (int a = 0; a < 3; a++) {
                o.position[a] = y[a] * boxsize;
                o.velocity[a] = detail::dot(v, R.axes[a]);
            }
        }
        return out;
    }

    //=========================================================================
    // Healpix footprints (RING ordering, no Healpix library needed; maps from an ascii file with one value per
    // line, or with USE_HEALPIX from FITS). Values are the completeness of the pixel.
    //=========================================================================
    inline std::int64_t healpix_ang2pix_ring(std::int64_t nside, double theta, double phi) {
        const double z = std::cos(theta), za = std::fabs(z);
        double tt = std::fmod(phi, 2 * M_PI);
        if (tt < 0)
            tt += 2 * M_PI;
        tt /= 0.5 * M_PI; // in [0, 4)
        const std::int64_t ncap = 2 * nside * (nside - 1), npix = 12 * nside * nside;
        auto imod = [](std::int64_t a, std::int64_t b) { return ((a % b) + b) % b; };
        if (za <= 2.0 / 3.0) {
            const double t1 = nside * (0.5 + tt), t2 = nside * z * 0.75;
            const std::int64_t jp = std::int64_t(t1 - t2), jm = std::int64_t(t1 + t2);
            const std::int64_t ir = nside + 1 + jp - jm, kshift = 1 - (ir & 1);
            const std::int64_t ip = imod((jp + jm - nside + kshift + 1) / 2, 4 * nside);
            return ncap + (ir - 1) * 4 * nside + ip;
        }
        const double tp = tt - std::floor(tt), tmp = nside * std::sqrt(3 * (1 - za));
        const std::int64_t jp = std::int64_t(tp * tmp), jm = std::int64_t((1.0 - tp) * tmp);
        const std::int64_t ir = jp + jm + 1, ip = imod(std::int64_t(tt * ir), 4 * ir);
        return z > 0 ? 2 * ir * (ir - 1) + ip : npix - 2 * ir * (ir + 1) + ip;
    }

    /// Footprint from a RING-ordered Healpix map of completeness values (npix = 12 nside^2)
    inline Footprint healpix_footprint(std::vector<double> map) {
        const std::int64_t nside = std::int64_t(std::llround(std::sqrt(map.size() / 12.0)));
        if (nside < 1 || std::size_t(12 * nside * nside) != map.size())
            throw std::invalid_argument("healpix_footprint: the map size is not 12 nside^2");
        for (double v : map)
            if (!(v >= 0 && v <= 1))
                throw std::invalid_argument("healpix_footprint: the values must be in [0, 1]");
        return [map = std::move(map), nside](double ra, double dec) {
            return map[std::size_t(healpix_ang2pix_ring(nside, (90.0 - dec) * deg, ra * deg))];
        };
    }

    /// A RING-ordered Healpix map from an ascii file (one value per line; '#' lines are skipped)
    inline std::vector<double> read_healpix_ascii(const std::string & filename) {
        std::ifstream fp(filename);
        if (!fp)
            throw std::runtime_error("read_healpix_ascii: cannot open " + filename);
        std::vector<double> map;
        std::string line;
        while (std::getline(fp, line))
            if (!line.empty() && line[0] != '#')
                map.push_back(std::stod(line));
        return map;
    }

#ifdef USE_HEALPIX
    /// A Healpix map from a FITS file (any ordering), returned in RING ordering
    inline std::vector<double> read_healpix_fits(const std::string & filename) {
        Healpix_Map<double> m;
        read_Healpix_map_from_fits(filename, m);
        if (m.Scheme() == NEST)
            m.swap_scheme();
        std::vector<double> map(std::size_t(m.Npix()));
        for (std::size_t i = 0; i < map.size(); i++)
            map[i] = m[int(i)];
        return map;
    }
#endif

    /// A MANGLE polygon (Swanson et al. 2008): the intersection of caps (x, y, z, cm), with (x, y, z) the unit
    /// axis and cm = 1 - cos(theta). A point r is in a cap if 1 - r.c < cm (cm >= 0) or 1 - r.c > -cm (cm < 0, the
    /// complement of the cap), and in the polygon if it is in all of its caps
    struct ManglePolygon {
        std::vector<std::array<double, 4>> caps;
        double weight = 1, area_str = 0;
        long long id = 0;
        bool contains(const Vec3 & r) const {
            for (const auto & c : caps) {
                const double d = 1 - (r[0] * c[0] + r[1] * c[1] + r[2] * c[2]);
                if (c[3] >= 0 ? !(d < c[3]) : !(d > -c[3]))
                    return false;
            }
            return true;
        }
    };

    /// A MANGLE mask. The completeness at (RA, Dec) is the weight of the first polygon that contains the point
    /// (balkanized masks have no overlaps), 0 outside all polygons. Polygons are looked up through an RA/Dec grid
    /// (cell_deg) filled from a bounding cap of every polygon (its smallest cap with cm > 0); the mangle pixel
    /// numbers are not used
    class MangleMask {
      public:
        std::vector<ManglePolygon> polygons;

        MangleMask() = default;
        explicit MangleMask(std::vector<ManglePolygon> polys, double cell_deg = 1.0) : polygons(std::move(polys)) {
            if (!(cell_deg > 0 && cell_deg <= 90))
                throw std::invalid_argument("MangleMask: cell_deg must be in (0, 90]");
            nra = std::max(1, int(std::lround(360.0 / cell_deg)));
            ndec = std::max(1, int(std::lround(180.0 / cell_deg)));
            grid.assign(std::size_t(nra) * ndec, {});
            for (int i = 0; i < int(polygons.size()); i++) {
                const auto & p = polygons[i];
                double cm = 2;
                std::array<double, 4> b{0, 0, 1, 2};
                for (const auto & c : p.caps)
                    if (c[3] > 0 && c[3] < cm) {
                        cm = c[3];
                        b = c;
                    }
                const double theta = std::acos(std::max(-1.0, 1 - cm)) / deg;
                const double dec0 = std::asin(std::max(-1.0, std::min(1.0, b[2]))) / deg;
                const double ra0 = std::atan2(b[1], b[0]) / deg;
                const double dlo = dec0 - theta, dhi = dec0 + theta;
                const int j0 = dec_cell(std::max(-90.0, dlo)) - 1, j1 = dec_cell(std::min(90.0, dhi)) + 1;
                const bool all_ra = dlo <= -90 || dhi >= 90;
                const double dra =
                    all_ra ? 180 : std::asin(std::min(1.0, std::sin(theta * deg) / std::cos(dec0 * deg))) / deg;
                const int ni = all_ra ? nra : std::min(nra, int(std::ceil(2 * dra / 360.0 * nra)) + 3);
                const int i0 = all_ra ? 0 : ra_cell(ra0 - dra) - 1;
                for (int j = std::max(0, j0); j <= std::min(ndec - 1, j1); j++)
                    for (int k = 0; k < ni; k++)
                        grid[std::size_t(j) * nra + std::size_t(((i0 + k) % nra + nra) % nra)].push_back(i);
            }
        }

        /// Index of the polygon that contains (RA, Dec) [deg], -1 if none
        int find(double ra, double dec) const {
            const Vec3 r{std::cos(dec * deg) * std::cos(ra * deg),
                         std::cos(dec * deg) * std::sin(ra * deg),
                         std::sin(dec * deg)};
            for (int i : grid[std::size_t(dec_cell(dec)) * nra + std::size_t(ra_cell(ra))])
                if (polygons[i].contains(r))
                    return i;
            return -1;
        }
        double operator()(double ra, double dec) const {
            const int i = find(ra, dec);
            return i < 0 ? 0.0 : polygons[i].weight;
        }

      private:
        int nra = 0, ndec = 0;
        std::vector<std::vector<int>> grid;
        int dec_cell(double dec) const { return std::min(ndec - 1, std::max(0, int((dec + 90) / 180 * ndec))); }
        int ra_cell(double ra) const {
            const double x = ra / 360 - std::floor(ra / 360);
            return std::min(nra - 1, int(x * nra));
        }
    };

    /// Read a MANGLE mask in the ascii polygon format (mangle's default output):
    ///   polygon <id> ( <n> caps, <w> weight, <p> pixel, <a> str):
    ///    <x> <y> <z> <cm>       (n lines)
    /// Header lines (N polygons, real, pixelization, snapped, balkanized) are skipped. Other formats (circle,
    /// vertices, edges, rectangle) can be converted with mangle's poly2poly. Weights must be in [0, 1]
    inline MangleMask read_mangle_polygons(const std::string & filename, double cell_deg = 1.0) {
        std::ifstream fp(filename);
        if (!fp)
            throw std::runtime_error("read_mangle_polygons: cannot open " + filename);
        std::vector<ManglePolygon> polys;
        std::string line;
        while (std::getline(fp, line)) {
            std::istringstream ss(line);
            std::string word;
            ss >> word;
            if (word == "circle" || word == "vertices" || word == "edges" || word == "rectangle")
                throw std::runtime_error("read_mangle_polygons: only the polygon format is read (convert " + filename +
                                         " with poly2poly)");
            if (word != "polygon")
                continue;
            ManglePolygon p;
            ss >> p.id;
            const auto open = line.find('(');
            if (open == std::string::npos)
                throw std::runtime_error("read_mangle_polygons: bad polygon line: " + line);
            std::string rest = line.substr(open + 1);
            for (char & ch : rest)
                if (ch == ',' || ch == ')' || ch == ':')
                    ch = ' ';
            std::istringstream rs(rest);
            double value;
            std::string key;
            long long ncaps = -1;
            while (rs >> value >> key) {
                if (key.rfind("cap", 0) == 0)
                    ncaps = (long long)(value);
                else if (key == "weight")
                    p.weight = value;
                else if (key == "str")
                    p.area_str = value;
            }
            if (ncaps < 0)
                throw std::runtime_error("read_mangle_polygons: no number of caps in: " + line);
            if (!(p.weight >= 0 && p.weight <= 1))
                throw std::runtime_error("read_mangle_polygons: weights must be in [0, 1] (polygon " +
                                         std::to_string(p.id) + ")");
            p.caps.resize(std::size_t(ncaps));
            for (auto & c : p.caps) {
                if (!std::getline(fp, line))
                    throw std::runtime_error("read_mangle_polygons: the file ends inside polygon " +
                                             std::to_string(p.id));
                std::istringstream cs(line);
                if (!(cs >> c[0] >> c[1] >> c[2] >> c[3]))
                    throw std::runtime_error("read_mangle_polygons: bad cap line: " + line);
            }
            polys.push_back(std::move(p));
        }
        if (polys.empty())
            throw std::runtime_error("read_mangle_polygons: no polygons in " + filename);
        return MangleMask(std::move(polys), cell_deg);
    }

    /// Footprint from a MANGLE mask (completeness = polygon weight, 0 outside)
    inline Footprint mangle_footprint(MangleMask mask) {
        auto m = std::make_shared<const MangleMask>(std::move(mask));
        return [m](double ra, double dec) { return (*m)(ra, dec); };
    }

    /// Convert to the catalogue points of SurveyMesh.h (or any type with position, weight and nbar)
    template <class Point>
    std::vector<Point> to_catalog_points(const std::vector<MockGalaxy> & galaxies, bool redshift_space = true) {
        std::vector<Point> out(galaxies.size());
        for (size_t i = 0; i < galaxies.size(); i++) {
            const auto & g = galaxies[i];
            for (int d = 0; d < 3; d++)
                out[i].position[d] = redshift_space ? g.position[d] : g.position_real[d];
            out[i].weight = g.weight;
            out[i].nbar = g.nbar;
        }
        return out;
    }

    /// Objects from FML particles (get_pos in units of the box, i.e. [0, 1), get_vel, get_id). velocity_to_kms
    /// converts the particle velocities to peculiar km/s (e.g. for SimulationCode code units)
    template <class T>
    std::vector<MockObject> objects_from_particles(const T * p,
                                                   size_t n,
                                                   double boxsize,
                                                   double velocity_to_kms,
                                                   bool positions_in_box_units = true) {
        std::vector<MockObject> out(n);
        for (size_t i = 0; i < n; i++) {
            const auto * x = FML::PARTICLE::GetPos(const_cast<T &>(p[i]));
            const auto * v = FML::PARTICLE::GetVel(const_cast<T &>(p[i]));
            for (int d = 0; d < 3; d++) {
                double xi = positions_in_box_units ? x[d] * boxsize : x[d];
                xi = std::fmod(xi, boxsize);
                out[i].position[d] = xi < 0 ? xi + boxsize : (xi >= boxsize ? 0.0 : xi);
                out[i].velocity[d] = v[d] * velocity_to_kms;
            }
            out[i].id = std::uint64_t(FML::PARTICLE::GetID(const_cast<T &>(p[i])));
        }
        return out;
    }

    /// Objects from HOD galaxies (FML/HOD/HOD.h: position in [0, L) Mpc/h, velocity in km/s, id = halo id and
    /// subid = ordinal, so central and satellites get independent random numbers)
    template <class Galaxy>
    std::vector<MockObject> objects_from_hod(const std::vector<Galaxy> & galaxies) {
        std::vector<MockObject> out(galaxies.size());
        for (size_t i = 0; i < galaxies.size(); i++) {
            out[i].position = {galaxies[i].position[0], galaxies[i].position[1], galaxies[i].position[2]};
            out[i].velocity = {galaxies[i].velocity[0], galaxies[i].velocity[1], galaxies[i].velocity[2]};
            out[i].id = galaxies[i].halo_id;
            out[i].subid = std::uint32_t(galaxies[i].ordinal);
        }
        return out;
    }

    /// Ascii catalogue: ra dec z_obs z_cos x y z (redshift space) v_r completeness selection_probability nbar weight
    /// id subid replica. With MPI every task writes <filename>.<task>
    inline void write_catalogue(const std::string & filename, const std::vector<MockGalaxy> & galaxies) {
        const std::string file = FML::NTasks > 1 ? filename + "." + std::to_string(FML::ThisTask) : filename;
        std::ofstream fp(file);
        if (!fp)
            throw std::runtime_error("write_catalogue: cannot open " + file);
        fp << "# ra dec z_obs z_cos x y z [Mpc/h, redshift space] v_r [km/s] completeness selection_probability "
              "nbar [(h/Mpc)^3] weight id subid replica_x replica_y replica_z\n";
        fp << std::setprecision(10);
        for (const auto & g : galaxies)
            fp << g.ra << " " << g.dec << " " << g.z_obs << " " << g.z_cos << " " << g.position[0] << " "
               << g.position[1] << " " << g.position[2] << " " << g.v_radial << " " << g.completeness << " "
               << g.selection_probability << " " << g.nbar << " " << g.weight << " " << g.id << " " << g.subid << " "
               << g.replica[0] << " " << g.replica[1] << " " << g.replica[2] << "\n";
    }

#ifdef USE_HDF5
    /// HDF5 catalogue with the columns of write_catalogue as datasets: ra, dec, z_obs, z_cos, position (n x 3,
    /// Mpc/h, redshift space), v_radial, completeness, selection_probability, nbar, weight, id, subid, replica
    /// (n x 3). With MPI every task writes <filename>.<task>
    inline void write_catalogue_hdf5(const std::string & filename, const std::vector<MockGalaxy> & galaxies) {
        const std::string file = FML::NTasks > 1 ? filename + "." + std::to_string(FML::ThisTask) : filename;
        const hid_t f = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (f < 0)
            throw std::runtime_error("write_catalogue_hdf5: cannot create " + file);
        const hsize_t n = galaxies.size();
        bool ok = true;
        auto write = [&](const char * name, hid_t filetype, hid_t memtype, const void * data, hsize_t ncols) {
            const hsize_t dims[2] = {n, ncols};
            const hid_t space = H5Screate_simple(ncols == 1 ? 1 : 2, dims, nullptr);
            const hid_t d = H5Dcreate2(f, name, filetype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            ok = ok && space >= 0 && d >= 0 &&
                 (n == 0 || H5Dwrite(d, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0);
            if (d >= 0)
                H5Dclose(d);
            if (space >= 0)
                H5Sclose(space);
        };
        auto column = [&](const char * name, double MockGalaxy::* field) {
            std::vector<double> v(n);
            for (hsize_t i = 0; i < n; i++)
                v[i] = galaxies[i].*field;
            write(name, H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, v.data(), 1);
        };
        column("ra", &MockGalaxy::ra);
        column("dec", &MockGalaxy::dec);
        column("z_obs", &MockGalaxy::z_obs);
        column("z_cos", &MockGalaxy::z_cos);
        column("v_radial", &MockGalaxy::v_radial);
        column("completeness", &MockGalaxy::completeness);
        column("selection_probability", &MockGalaxy::selection_probability);
        column("nbar", &MockGalaxy::nbar);
        column("weight", &MockGalaxy::weight);
        std::vector<double> pos(3 * n);
        std::vector<std::uint64_t> id(n);
        std::vector<std::uint32_t> subid(n);
        std::vector<int> rep(3 * n);
        for (hsize_t i = 0; i < n; i++) {
            for (int d = 0; d < 3; d++) {
                pos[3 * i + d] = galaxies[i].position[d];
                rep[3 * i + d] = galaxies[i].replica[d];
            }
            id[i] = galaxies[i].id;
            subid[i] = galaxies[i].subid;
        }
        write("position", H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE, pos.data(), 3);
        write("id", H5T_STD_U64LE, H5T_NATIVE_UINT64, id.data(), 1);
        write("subid", H5T_STD_U32LE, H5T_NATIVE_UINT32, subid.data(), 1);
        write("replica", H5T_STD_I32LE, H5T_NATIVE_INT, rep.data(), 3);
        H5Fclose(f);
        if (!ok)
            throw std::runtime_error("write_catalogue_hdf5: failed to write " + file);
    }
#endif

} // namespace FML::SURVEY::MOCKS

#endif
