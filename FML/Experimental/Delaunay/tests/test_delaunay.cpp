// Tests of the native Delaunay triangulation and periodic Voronoi tessellation (FML/Experimental/Delaunay):
//   - exact predicates: orient/insphere in 2D and 3D against exact 128-bit integer determinants on small integer
//     coordinates (many exact degeneracies), and the floating-point filter against the exact evaluation on nearly
//     degenerate double input
//   - Delaunay<D>: valid, positively oriented, locally Delaunay triangulations of random, clustered, lattice (fully
//     cospherical) and duplicated points; brute-force empty circumspheres for small sets
//   - PeriodicVoronoi<D>: volumes sum to the box, closed cells (sum of A n = 0), divergence-theorem volumes from the
//     face centroids, exact lattices (cubes/squares), translation invariance, tiny point sets (cells neighbouring
//     their own images), clustered sets (buffer growth), thread-count independence, invalid input
#include <FML/Experimental/Delaunay/PeriodicVoronoi.h>
#include <FML/tests/TestHelpers.h>

#include <random>

using FML::TESTS::check;
using FML::TESTS::num;
using namespace FML::DELAUNAY;

namespace {

    using i128 = __int128;
    int sgn(i128 x) { return x > 0 ? 1 : (x < 0 ? -1 : 0); }

    i128 det2(i128 a, i128 b, i128 c, i128 d) { return a * d - b * c; }
    i128 det3(const i128 m[3][3]) {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    i128 det4(const i128 m[4][4]) {
        i128 r = 0;
        for (int c = 0; c < 4; c++) {
            i128 s[3][3];
            for (int i = 1; i < 4; i++)
                for (int j = 0, jj = 0; j < 4; j++)
                    if (j != c)
                        s[i - 1][jj++] = m[i][j];
            r += (c % 2 ? -1 : 1) * m[0][c] * det3(s);
        }
        return r;
    }

    // Reference orientation and in-sphere signs (same conventions as orient/insphere) from integer coordinates
    template <int D>
    int orient_ref(const std::array<std::array<long, D>, D + 1> & p) {
        if constexpr (D == 2)
            return sgn(det2(p[1][0] - p[0][0], p[1][1] - p[0][1], p[2][0] - p[0][0], p[2][1] - p[0][1]));
        else {
            i128 m[3][3];
            for (int k = 0; k < 3; k++)
                for (int d = 0; d < 3; d++)
                    m[k][d] = p[k + 1][d] - p[0][d];
            return sgn(det3(m));
        }
    }
    // Lifted determinant with rows (p_i - q, |p_i - q|^2): its sign times (D == 2 ? 1 : -1) is insphere
    template <int D>
    int insphere_ref(const std::array<std::array<long, D>, D + 1> & p, const std::array<long, D> & q) {
        i128 m[D + 1][D + 1];
        for (int k = 0; k <= D; k++) {
            i128 l = 0;
            for (int d = 0; d < D; d++) {
                m[k][d] = p[k][d] - q[d];
                l += m[k][d] * m[k][d];
            }
            m[k][D] = l;
        }
        if constexpr (D == 2)
            return sgn(det3(m));
        else
            return -sgn(det4(m));
    }

    template <int D>
    void test_predicates_integer() {
        std::mt19937_64 rng(1 + D);
        std::uniform_int_distribution<long> U(-3, 3);
        int bad_orient = 0, bad_insphere = 0, degenerate = 0, cospherical = 0;
        const int n = 200000;
        for (int t = 0; t < n; t++) {
            std::array<std::array<long, D>, D + 1> p;
            std::array<long, D> q;
            for (auto & x : p)
                for (auto & c : x)
                    c = U(rng);
            for (auto & c : q)
                c = U(rng);
            std::array<std::array<double, D>, D + 1> pd;
            std::array<double, D> qd;
            for (int k = 0; k <= D; k++)
                for (int d = 0; d < D; d++)
                    pd[k][d] = double(p[k][d]);
            for (int d = 0; d < D; d++)
                qd[d] = double(q[d]);
            std::array<const double *, D + 1> ptr;
            for (int k = 0; k <= D; k++)
                ptr[k] = pd[k].data();
            const int o = orient_ref<D>(p);
            bad_orient += orient<D>(ptr) != o;
            if (o == 0) {
                degenerate++;
                continue;
            }
            if (o < 0) {
                std::swap(p[0], p[1]);
                std::swap(ptr[0], ptr[1]);
            }
            const int s = insphere_ref<D>(p, q);
            cospherical += s == 0;
            bad_insphere += insphere<D>(ptr, qd.data()) != s;
        }
        check(bad_orient == 0 && bad_insphere == 0,
              std::to_string(D) + "D orient/insphere = exact integer determinants (" + std::to_string(n) +
                  " random small-integer cases, " + std::to_string(degenerate) + " degenerate simplices, " +
                  std::to_string(cospherical) + " cospherical)");
    }

    // Nearly degenerate doubles: points on a plane/sphere perturbed by a few ulps; the filtered predicate must agree
    // with the exact evaluation
    void test_predicates_filter() {
        std::mt19937_64 rng(7);
        std::uniform_real_distribution<double> U(-1, 1);
        int bad = 0;
        const int n = 100000;
        for (int t = 0; t < n; t++) {
            double p[5][3];
            // Four points on the plane z = a x + b y + c (rounded), the fifth on a sphere through the first four
            const double a = U(rng), b = U(rng), c = U(rng);
            for (int k = 0; k < 4; k++) {
                p[k][0] = U(rng);
                p[k][1] = U(rng);
                p[k][2] = a * p[k][0] + b * p[k][1] + c;
                if (t % 3 == 0)
                    p[k][2] = std::nextafter(p[k][2], 2.0);
            }
            const double th = U(rng) * M_PI, ph = U(rng) * M_PI;
            for (int k = 0; k < 4; k++) {
                const double r = 0.7;
                const double tk = th + k, pk = ph + 2 * k;
                p[k][0] = r * std::sin(tk) * std::cos(pk);
                p[k][1] = r * std::sin(tk) * std::sin(pk);
                p[k][2] = r * std::cos(tk);
                if (t % 2 == 0)
                    for (int d = 0; d < 3; d++)
                        p[k][d] = a * U(rng) + p[k][d] * (1 + 1e-16 * U(rng));
            }
            p[4][0] = 0.7 * std::sin(th - 1) * std::cos(ph + 1);
            p[4][1] = 0.7 * std::sin(th - 1) * std::sin(ph + 1);
            p[4][2] = 0.7 * std::cos(th - 1);
            bad += orient3d_shewchuk(p[0], p[1], p[2], p[3]) != EXACT::orient3d_shewchuk(p[0], p[1], p[2], p[3]);
            bad += insphere_shewchuk(p[0], p[1], p[2], p[3], p[4]) !=
                   EXACT::insphere_shewchuk(p[0], p[1], p[2], p[3], p[4]);
            bad += orient2d_shewchuk(p[0], p[1], p[2]) != EXACT::orient2d_shewchuk(p[0], p[1], p[2]);
            bad += incircle_shewchuk(p[0], p[1], p[2], p[4]) != EXACT::incircle_shewchuk(p[0], p[1], p[2], p[4]);
        }
        check(bad == 0,
              "filtered predicates = exact evaluation on nearly degenerate doubles (" + std::to_string(n) + " cases)");
    }

    template <int D>
    std::vector<std::array<double, D>> random_points(size_t n, double L, unsigned seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> U(0, L);
        std::vector<std::array<double, D>> p(n);
        for (auto & x : p)
            for (auto & c : x)
                c = U(rng);
        return p;
    }

    template <int D>
    std::vector<std::array<double, D>> lattice(int n, double L, double offset) {
        std::vector<std::array<double, D>> p;
        std::array<int, D> c{};
        while (true) {
            std::array<double, D> x;
            for (int d = 0; d < D; d++)
                x[d] = (c[d] + offset) * L / n;
            p.push_back(x);
            int d = D - 1;
            while (d >= 0 && c[d] == n - 1)
                c[d--] = 0;
            if (d < 0)
                break;
            c[d]++;
        }
        return p;
    }

    template <int D>
    std::vector<std::array<double, D>> clustered(size_t n, double L, unsigned seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> U(0, L);
        std::normal_distribution<double> G(0, 0.01 * L);
        std::vector<std::array<double, D>> p(n);
        for (size_t i = 0; i < n; i++)
            for (int d = 0; d < D; d++) {
                // Half in a blob across the corner of the box, half uniform
                double x = (i % 2) ? U(rng) : 0.99 * L + G(rng);
                p[i][d] = x - L * std::floor(x / L);
            }
        return p;
    }

    template <int D>
    void test_triangulation(const std::string & name, const std::vector<std::array<double, D>> & p, bool brute) {
        std::array<double, D> lo, hi;
        lo.fill(-1.0);
        hi.fill(2.0);
        Delaunay<D> dt(lo, hi);
        dt.insert(p);
        std::string msg = dt.check(true);
        if (msg.empty() && brute) {
            // No vertex strictly inside the circumsphere of any simplex
            for (const auto & S : dt.get_simplices()) {
                if (!S.alive())
                    continue;
                std::array<const double *, D + 1> ptr;
                for (int k = 0; k <= D; k++)
                    ptr[k] = dt.vertex(S.v[k]).data();
                for (int v = 0; v < dt.nvertices(); v++)
                    if (insphere<D>(ptr, dt.vertex(v).data()) > 0)
                        msg = "vertex inside a circumsphere";
            }
        }
        check(msg.empty(),
              std::to_string(D) + "D Delaunay, " + name + ": valid and Delaunay" +
                  (brute ? " (brute-force empty spheres)" : "") + (msg.empty() ? "" : ": " + msg));
    }

    template <int D>
    struct VoronoiSummary {
        double sum_volume{0.0};
        double max_closure{0.0};
        double max_divergence{0.0};
        bool ok_lists{true};
    };

    template <int D>
    VoronoiSummary<D> summarize(const PeriodicVoronoi<D> & v) {
        VoronoiSummary<D> s;
        std::vector<size_t> seen(v.faces().size(), 0);
        for (size_t i = 0; i < v.ncells(); i++) {
            s.sum_volume += v.cells()[i].volume;
            std::array<double, D> an{};
            double asum = 0.0, vdiv = 0.0;
            const auto ri = v.position(i);
            auto range = v.cell_faces(i);
            for (auto it = range.first; it != range.second; ++it) {
                const int64_t c = *it;
                const size_t f = size_t(c >= 0 ? c : -c - 1);
                seen[f]++;
                const auto & F = v.faces()[f];
                const double sign = c >= 0 ? 1.0 : -1.0;
                s.ok_lists = s.ok_lists && (c >= 0 ? F.local_i == i : F.local_j == i);
                // Face centroid as seen from this cell
                std::array<double, D> fc = F.centroid;
                if (c < 0)
                    for (int d = 0; d < D; d++)
                        fc[d] -= F.shift[d] * v.boxsize();
                double h = 0.0;
                for (int d = 0; d < D; d++) {
                    an[d] += sign * F.area * F.normal[d];
                    h += sign * F.normal[d] * (fc[d] - ri[d]);
                }
                asum += F.area;
                vdiv += F.area * h / D;
            }
            double m = 0.0;
            for (int d = 0; d < D; d++)
                m += an[d] * an[d];
            s.max_closure = std::max(s.max_closure, std::sqrt(m) / asum);
            s.max_divergence = std::max(s.max_divergence, std::fabs(vdiv / v.cells()[i].volume - 1.0));
        }
        for (auto n : seen)
            s.ok_lists = s.ok_lists && n == 2;
        return s;
    }

    template <int D>
    void test_voronoi(const std::string & name, const std::vector<std::array<double, D>> & p, double L) {
        PeriodicVoronoi<D> v(p, L);
        const auto s = summarize<D>(v);
        const double LD = std::pow(L, D);
        const bool ok = std::fabs(s.sum_volume / LD - 1.0) < 1e-12 && s.max_closure < 1e-10 &&
                        s.max_divergence < 1e-9 && s.ok_lists;
        check(ok,
              std::to_string(D) + "D periodic Voronoi, " + name + " (N = " + std::to_string(p.size()) +
                  "): sum V / L^D - 1 = " + num(s.sum_volume / LD - 1.0, 2) + ", closure " + num(s.max_closure, 2) +
                  ", divergence volume " + num(s.max_divergence, 2) + ", " + std::to_string(v.statistics().nrounds) +
                  " rounds, " + std::to_string(v.statistics().nghosts) + " ghosts" +
                  (s.ok_lists ? "" : ", BAD face lists"));
    }

    template <int D>
    void test_lattice_cells() {
        const int n = D == 3 ? 12 : 40;
        const double L = 7.3;
        for (double offset : {0.0, 0.5, 0.37}) {
            const auto p = lattice<D>(n, L, offset);
            PeriodicVoronoi<D> v(p, L);
            const double V0 = std::pow(L / n, D), A0 = std::pow(L / n, D - 1);
            double dv = 0.0, da = 0.0;
            bool counts = true;
            for (size_t i = 0; i < v.ncells(); i++) {
                dv = std::max(dv, std::fabs(v.cells()[i].volume / V0 - 1.0));
                auto r = v.cell_faces(i);
                counts = counts && (r.second - r.first) == 2 * D;
            }
            for (const auto & f : v.faces())
                da = std::max(da, std::fabs(f.area / A0 - 1.0));
            check(dv < 1e-9 && da < 1e-9 && counts,
                  std::to_string(D) + "D lattice (offset " + num(offset, 2) + "): every cell a cube with " +
                      std::to_string(2 * D) + " faces (max |V/V0 - 1| " + num(dv, 2) + ", |A/A0 - 1| " + num(da, 2) +
                      ", " + std::to_string(v.statistics().nfaces_dropped) + " zero-area faces dropped)");
        }
    }

    template <int D>
    void test_translation() {
        const double L = 3.0;
        auto p = random_points<D>(3000, L, 21);
        PeriodicVoronoi<D> a(p, L);
        std::array<double, D> shift;
        for (int d = 0; d < D; d++)
            shift[d] = 0.4321 * L * (d + 1);
        for (auto & x : p)
            for (int d = 0; d < D; d++) {
                x[d] += shift[d];
                x[d] -= L * std::floor(x[d] / L);
            }
        PeriodicVoronoi<D> b(p, L);
        double dv = 0.0;
        for (size_t i = 0; i < a.ncells(); i++)
            dv = std::max(dv, std::fabs(a.cells()[i].volume / b.cells()[i].volume - 1.0));
        check(a.faces().size() == b.faces().size() && dv < 1e-8,
              std::to_string(D) + "D periodic Voronoi is translation invariant (same face count, max volume change " +
                  num(dv, 2) + ")");
    }

    template <int D>
    void test_threads() {
#ifdef USE_OMP
        const auto p = clustered<D>(20000, 1.0, 31);
        const int nmax = omp_get_max_threads();
        omp_set_num_threads(1);
        PeriodicVoronoi<D> a(p, 1.0);
        omp_set_num_threads(std::max(2, nmax));
        PeriodicVoronoi<D> b(p, 1.0);
        omp_set_num_threads(nmax);
        bool same = a.faces().size() == b.faces().size();
        for (size_t i = 0; same && i < a.ncells(); i++)
            same = a.cells()[i].volume == b.cells()[i].volume && a.cells()[i].centroid == b.cells()[i].centroid;
        for (size_t f = 0; same && f < a.faces().size(); f++)
            same = a.faces()[f].area == b.faces()[f].area && a.faces()[f].j == b.faces()[f].j;
        check(same,
              std::to_string(D) + "D periodic Voronoi identical with 1 and " + std::to_string(std::max(2, nmax)) +
                  " threads");
#endif
    }

    template <int D>
    void test_invalid() {
        std::vector<std::array<double, D>> p = random_points<D>(10, 1.0, 3);
        p.push_back(p[4]);
        check(FML::TESTS::throws([&] { PeriodicVoronoi<D> v(p, 1.0); }),
              std::to_string(D) + "D periodic Voronoi rejects duplicate points");
        check(FML::TESTS::throws([&] { PeriodicVoronoi<D> v(std::vector<std::array<double, D>>{}, 1.0); }) &&
                  FML::TESTS::throws([&] { PeriodicVoronoi<D> v(random_points<D>(5, 1.0, 1), -1.0); }),
              std::to_string(D) + "D periodic Voronoi rejects no points and a nonpositive box");
        // A duplicate inserted directly returns the existing vertex
        std::array<double, D> lo, hi;
        lo.fill(0.0);
        hi.fill(1.0);
        Delaunay<D> dt(lo, hi);
        const auto pts = random_points<D>(50, 1.0, 9);
        const auto ids = dt.insert(pts);
        const int again = dt.insert(pts[17], 12345);
        check(again == ids[17] && dt.statistics().nduplicates == 1 && dt.check().empty(),
              std::to_string(D) + "D Delaunay returns the existing vertex for a duplicate point");
    }

    template <int D>
    void test_dimension() {
        test_predicates_integer<D>();
        test_triangulation<D>("300 random points", random_points<D>(300, 1.0, 2), true);
        test_triangulation<D>("lattice (all cospherical)", lattice<D>(D == 3 ? 6 : 16, 1.0, 0.37), true);
        test_triangulation<D>("20000 random points", random_points<D>(20000, 1.0, 3), false);
        test_triangulation<D>("20000 clustered points", clustered<D>(20000, 1.0, 4), false);
        test_triangulation<D>("lattice", lattice<D>(D == 3 ? 20 : 120, 1.0, 0.37), false);
        test_voronoi<D>("random", random_points<D>(20000, 100.0, 5), 100.0);
        test_voronoi<D>("clustered at the corner", clustered<D>(20000, 1.0, 6), 1.0);
        for (size_t n : {1, 2, 3, 7})
            test_voronoi<D>("tiny", random_points<D>(n, 1.0, unsigned(10 + n)), 1.0);
        test_lattice_cells<D>();
        test_translation<D>();
        test_threads<D>();
        test_invalid<D>();
    }

} // namespace

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    test_predicates_filter();
    test_dimension<2>();
    test_dimension<3>();
    return FML::TESTS::summary("Delaunay");
}
