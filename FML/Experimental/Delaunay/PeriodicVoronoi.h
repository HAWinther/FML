#ifndef FML_DELAUNAY_PERIODICVORONOI_HEADER
#define FML_DELAUNAY_PERIODICVORONOI_HEADER

#include <FML/Experimental/Delaunay/Delaunay.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

namespace FML {
    namespace DELAUNAY {

        //==========================================================================================
        /// Voronoi tessellation of points in a periodic box [0, boxsize)^D (D = 2, 3), serial with OpenMP for the
        /// geometry.
        ///
        /// The points are triangulated together with periodic images in a buffer around the box. The buffer starts
        /// at initial_buffer mean interparticle spacings and grows (inserting the images of the new shell into the
        /// same triangulation) until every Delaunay simplex touching a point in the box has its circumsphere inside
        /// the triangulated region: then no image outside can change these simplices, and the cells are those of
        /// the infinite periodic point set.
        ///
        /// Coordinates are mapped to the unit box and rounded to multiples of 2^-quantization_bits (default 40; a
        /// relative change of ~1e-12). Periodic images are then exact translates (u + integer), so the exact
        /// predicates see the same configuration around a point and around its images, and the symbolic perturbation
        /// orders points by (index, image shift), which is translation invariant. Cells and faces are therefore
        /// consistent across the box boundary, also for degenerate input such as lattices.
        ///
        /// Output, in the user's units:
        ///   cells: volume (area in 2D) and centroid (in the coordinates of the point, so possibly slightly outside
        ///          the box)
        ///   faces: each face once, between cell i and the image of cell j at shift (j + shift * boxsize), with
        ///          area (length in 2D), centroid (near point i), unit normal from i to the image of j and the
        ///          distance between the two points. A cell can neighbour itself (j == i, nonzero shift) when there
        ///          are very few points.
        ///   cell_faces(i): the faces of cell i, as indices f >= 0 (i is the face's cell i) or -(f + 1) (cell i is
        ///          the face's cell j, so the normal points into the cell)
        /// Faces of zero area (from degenerate, e.g. cospherical, points) are dropped by default.
        //==========================================================================================
        template <int D>
        class PeriodicVoronoi {
          public:
            using Point = std::array<double, D>;
            using Shift = std::array<int, D>;

            struct Options {
                /// Initial buffer in mean interparticle spacings (it grows as needed)
                double initial_buffer = 2.0;
                /// Coordinates are rounded to multiples of boxsize * 2^-quantization_bits (at most 48)
                int quantization_bits = 40;
                /// Drop faces with area below zero_area_tolerance * distance^(D-1)
                bool drop_zero_area_faces = true;
                double zero_area_tolerance = 1e-14;
                bool verbose = false;
            };

            struct Cell {
                double volume{0.0};
                Point centroid{};
            };

            struct Face {
                size_t i{0};
                size_t j{0};
                Shift shift{};
                double area{0.0};
                Point centroid{};
                Point normal{};
                double distance{0.0};
            };

            struct Statistics {
                size_t npoints{0};
                size_t nghosts{0};
                double buffer{0.0}; // final buffer in units of the box
                int nbuffer_steps{0};
                size_t nfaces_dropped{0};
                double time_triangulation{0.0};
                double time_geometry{0.0};
            };

            PeriodicVoronoi() = default;
            PeriodicVoronoi(const std::vector<Point> & points, double boxsize, const Options & options = Options{}) {
                build(points, boxsize, options);
            }

            void build(const std::vector<Point> & points, double boxsize, const Options & options = Options{}) {
                if (!(boxsize > 0.0) || !std::isfinite(boxsize))
                    throw std::invalid_argument("[PeriodicVoronoi] boxsize must be positive");
                if (options.quantization_bits < 1 || options.quantization_bits > 48)
                    throw std::invalid_argument("[PeriodicVoronoi] quantization_bits must be in [1, 48]");
                if (points.empty())
                    throw std::invalid_argument("[PeriodicVoronoi] No points");
                opt = options;
                L = boxsize;
                stats = Statistics{};
                stats.npoints = points.size();
                const auto t0 = std::chrono::steady_clock::now();
                triangulate(points);
                const auto t1 = std::chrono::steady_clock::now();
                compute_geometry();
                const auto t2 = std::chrono::steady_clock::now();
                stats.time_triangulation = std::chrono::duration<double>(t1 - t0).count();
                stats.time_geometry = std::chrono::duration<double>(t2 - t1).count();
            }

            size_t ncells() const { return cells_.size(); }
            const std::vector<Cell> & cells() const { return cells_; }
            const std::vector<Face> & faces() const { return faces_; }
            /// The faces of cell i (see the class description for the sign convention)
            std::pair<const int64_t *, const int64_t *> cell_faces(size_t i) const {
                return {cell_face_list.data() + cell_face_offset[i], cell_face_list.data() + cell_face_offset[i + 1]};
            }
            const Statistics & statistics() const { return stats; }
            const Delaunay<D> & triangulation() const { return dt; }
            double boxsize() const { return L; }
            /// The (quantized) position of point i used for the tessellation, in the user's units
            Point position(size_t i) const {
                Point x = dt.vertex(vertex_of_point[i]);
                for (auto & c : x)
                    c *= L;
                return x;
            }

          private:
            static constexpr int NV = D + 1;
            static constexpr int MAXSHIFT = 2; // images up to +-2 boxes (buffer at most 2 boxes)
            static constexpr uint64_t NCODE = (D == 2) ? 25 : 125;

            Options opt{};
            double L{1.0};
            Delaunay<D> dt{};
            std::vector<int> vertex_of_point{};
            // Per vertex (index - D - 1): the point and the image shift
            std::vector<size_t> point_of_vertex{};
            std::vector<Shift> shift_of_vertex{};
            std::vector<Cell> cells_{};
            std::vector<Face> faces_{};
            std::vector<size_t> cell_face_offset{};
            std::vector<int64_t> cell_face_list{};
            Statistics stats{};

            static uint64_t shift_code(const Shift & s) {
                uint64_t c = 0;
                for (int d = 0; d < D; d++)
                    c = c * uint64_t(2 * MAXSHIFT + 1) + uint64_t(s[d] + MAXSHIFT);
                return c;
            }
            static bool shift_positive(const Shift & s) {
                for (int d = 0; d < D; d++)
                    if (s[d] != 0)
                        return s[d] > 0;
                return false;
            }
            size_t point_of(int v) const { return point_of_vertex[size_t(v - NV)]; }
            const Shift & shift_of(int v) const { return shift_of_vertex[size_t(v - NV)]; }

            void insert_images(const std::vector<Point> & u, double b_old, double b_new) {
                const int m = int(std::ceil(b_new));
                std::vector<Point> pos;
                std::vector<uint64_t> keys;
                std::vector<std::pair<size_t, Shift>> info;
                auto inside = [](const Point & x, double b) {
                    for (int d = 0; d < D; d++)
                        if (!(x[d] >= -b && x[d] < 1.0 + b))
                            return false;
                    return true;
                };
                for (size_t i = 0; i < u.size(); i++) {
                    Shift s;
                    s.fill(-m);
                    while (true) {
                        bool zero = true;
                        for (int d = 0; d < D; d++)
                            zero = zero && s[d] == 0;
                        if (!zero) {
                            Point x;
                            for (int d = 0; d < D; d++)
                                x[d] = u[i][d] + double(s[d]);
                            if (inside(x, b_new) && !inside(x, b_old)) {
                                pos.push_back(x);
                                keys.push_back(uint64_t(i) * NCODE + shift_code(s));
                                info.push_back({i, s});
                            }
                        }
                        int d = D - 1;
                        while (d >= 0 && s[d] == m)
                            s[d--] = -m;
                        if (d < 0)
                            break;
                        s[d]++;
                    }
                }
                const auto order = Delaunay<D>::hilbert_order(pos);
                for (size_t k : order) {
                    const size_t ndup = dt.statistics().nduplicates;
                    dt.insert(pos[k], keys[k]);
                    if (dt.statistics().nduplicates != ndup)
                        throw std::runtime_error("[PeriodicVoronoi] An image coincides with another point");
                    point_of_vertex.push_back(info[k].first);
                    shift_of_vertex.push_back(info[k].second);
                }
                stats.nghosts += pos.size();
            }

            void triangulate(const std::vector<Point> & points) {
                const size_t N = points.size();
                const double q = std::ldexp(1.0, opt.quantization_bits);
                std::vector<Point> u(N);
                for (size_t i = 0; i < N; i++)
                    for (int d = 0; d < D; d++) {
                        if (!std::isfinite(points[i][d]))
                            throw std::invalid_argument("[PeriodicVoronoi] Point not finite");
                        double x = points[i][d] / L;
                        x -= std::floor(x);
                        x = std::round(x * q) / q;
                        if (x >= 1.0)
                            x -= 1.0;
                        u[i][d] = x;
                    }
                Point lo, hi;
                lo.fill(-MAXSHIFT - 0.5);
                hi.fill(1.0 + MAXSHIFT + 0.5);
                dt.init(lo, hi);
                point_of_vertex.clear();
                shift_of_vertex.clear();
                vertex_of_point.assign(N, -1);

                // The points in the box
                const auto order = Delaunay<D>::hilbert_order(u);
                for (size_t i : order) {
                    const size_t ndup = dt.statistics().nduplicates;
                    const int v = dt.insert(u[i], uint64_t(i) * NCODE + shift_code(Shift{}));
                    if (dt.statistics().nduplicates != ndup)
                        throw std::invalid_argument("[PeriodicVoronoi] Duplicate points (after quantization): " +
                                                    std::to_string(i));
                    vertex_of_point[i] = v;
                    point_of_vertex.push_back(i);
                    shift_of_vertex.push_back(Shift{});
                }

                // Images in a growing buffer until every cell is complete
                double b_old = 0.0;
                double b = opt.initial_buffer * std::pow(double(N), -1.0 / D);
                b = std::min(b, double(MAXSHIFT));
                std::vector<size_t> pending(N);
                std::iota(pending.begin(), pending.end(), size_t(0));
                std::vector<int> star;
                std::vector<unsigned> visited;
                unsigned stamp = 0;
                while (true) {
                    insert_images(u, b_old, b);
                    stats.nbuffer_steps++;
                    double need_max = 0.0;
                    std::vector<size_t> still;
                    for (size_t i : pending) {
                        const double need = required_buffer(vertex_of_point[i], star, visited, stamp);
                        if (need >= b) {
                            still.push_back(i);
                            need_max = std::max(need_max, need);
                        }
                    }
                    if (opt.verbose)
                        std::cout << "[PeriodicVoronoi] buffer " << b << " (box units): " << still.size()
                                  << " incomplete cells, " << stats.nghosts << " images\n";
                    if (still.empty())
                        break;
                    if (b >= double(MAXSHIFT))
                        throw std::runtime_error(
                            "[PeriodicVoronoi] The buffer needed exceeds 2 boxes (too few points?)");
                    pending.swap(still);
                    b_old = b;
                    const double grow = std::isfinite(need_max) ? 1.05 * need_max : 2.0 * b;
                    b = std::min(double(MAXSHIFT), std::max(1.25 * b, grow));
                }
                stats.buffer = b;
            }

            /// The buffer (box units) needed for the cell of vertex v to be final (infinity if a super vertex is a
            /// neighbour)
            double
            required_buffer(int v, std::vector<int> & star, std::vector<unsigned> & visited, unsigned & stamp) const {
                dt.incident_simplices(v, star, visited, ++stamp);
                double need = 0.0;
                for (int s : star) {
                    const auto & S = dt.simplex(s);
                    for (int k = 0; k < NV; k++)
                        if (dt.is_super(S.v[k]))
                            return std::numeric_limits<double>::infinity();
                    Point c;
                    double r2;
                    dt.circumsphere(s, c, r2);
                    const double r = std::sqrt(r2) * (1.0 + 1e-9) + 1e-12;
                    for (int d = 0; d < D; d++)
                        need = std::max(need, std::max(-(c[d] - r), c[d] + r - 1.0));
                }
                return need;
            }

            void compute_geometry() {
                const size_t N = vertex_of_point.size();
                const auto & simplices = dt.get_simplices();
                std::vector<Point> cc(simplices.size());
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t s = 0; s < simplices.size(); s++) {
                    if (!simplices[s].alive())
                        continue;
                    bool super = false;
                    for (int k = 0; k < NV; k++)
                        super = super || dt.is_super(simplices[s].v[k]);
                    if (super)
                        continue;
                    double r2;
                    dt.circumsphere(int(s), cc[s], r2);
                }

                std::vector<std::vector<Face>> per_cell(N);
                std::vector<size_t> dropped(N, 0);
#ifdef USE_OMP
#pragma omp parallel
#endif
                {
                    std::vector<int> star, seen;
                    std::vector<unsigned> visited;
                    unsigned stamp = 0;
                    std::vector<Point> ring;
#ifdef USE_OMP
#pragma omp for schedule(dynamic, 256)
#endif
                    for (size_t i = 0; i < N; i++) {
                        const int vi = vertex_of_point[i];
                        dt.incident_simplices(vi, star, visited, ++stamp);
                        seen.clear();
                        for (int s : star) {
                            const auto & S = dt.simplex(s);
                            for (int k = 0; k < NV; k++) {
                                const int w = S.v[k];
                                if (w == vi || std::find(seen.begin(), seen.end(), w) != seen.end())
                                    continue;
                                seen.push_back(w);
                                if (dt.is_super(w))
                                    throw std::runtime_error("[PeriodicVoronoi] Internal error: incomplete cell");
                                const size_t j = point_of(w);
                                const Shift & sh = shift_of(w);
                                if (!(j > i || (j == i && shift_positive(sh))))
                                    continue;
                                Face f;
                                f.i = i;
                                f.j = j;
                                f.shift = sh;
                                if (!face_geometry(vi, w, s, cc, ring, f)) {
                                    dropped[i]++;
                                    continue;
                                }
                                per_cell[i].push_back(f);
                            }
                        }
                    }
                }

                // Faces in cell order, the face lists of the cells and the cell volumes/centroids
                faces_.clear();
                for (size_t i = 0; i < N; i++) {
                    faces_.insert(faces_.end(), per_cell[i].begin(), per_cell[i].end());
                    stats.nfaces_dropped += dropped[i];
                    std::vector<Face>().swap(per_cell[i]);
                }
                std::vector<size_t> count(N + 1, 0);
                for (const auto & f : faces_) {
                    count[f.i + 1]++;
                    count[f.j + 1]++;
                }
                cell_face_offset.assign(N + 1, 0);
                for (size_t i = 0; i < N; i++)
                    cell_face_offset[i + 1] = cell_face_offset[i] + count[i + 1];
                cell_face_list.assign(cell_face_offset[N], 0);
                std::vector<size_t> fill(cell_face_offset.begin(), cell_face_offset.end() - 1);
                for (size_t f = 0; f < faces_.size(); f++) {
                    cell_face_list[fill[faces_[f].i]++] = int64_t(f);
                    cell_face_list[fill[faces_[f].j]++] = -int64_t(f) - 1;
                }

                cells_.assign(N, Cell{});
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t i = 0; i < N; i++) {
                    const Point & ri = dt.vertex(vertex_of_point[i]);
                    double vol = 0.0;
                    Point cen{};
                    for (size_t k = cell_face_offset[i]; k < cell_face_offset[i + 1]; k++) {
                        const int64_t code = cell_face_list[k];
                        const Face & f = faces_[size_t(code >= 0 ? code : -code - 1)];
                        Point fc = f.centroid;
                        if (code < 0)
                            for (int d = 0; d < D; d++)
                                fc[d] -= f.shift[d];
                        const double cone = f.area * 0.5 * f.distance / D;
                        vol += cone;
                        for (int d = 0; d < D; d++)
                            cen[d] += cone * (ri[d] + D * fc[d]) / (D + 1);
                    }
                    cells_[i].volume = vol;
                    for (int d = 0; d < D; d++)
                        cells_[i].centroid[d] = vol > 0.0 ? cen[d] / vol : ri[d];
                }

                // To the user's units
                const double LD1 = std::pow(L, D - 1);
                for (auto & c : cells_) {
                    c.volume *= LD1 * L;
                    for (auto & x : c.centroid)
                        x *= L;
                }
                for (auto & f : faces_) {
                    f.area *= LD1;
                    f.distance *= L;
                    for (auto & x : f.centroid)
                        x *= L;
                }
            }

            /// The Voronoi face between vertices vi and w (an edge of simplex s), in internal units. Returns false if
            /// the face has zero area and is dropped.
            bool face_geometry(int vi, int w, int s, const std::vector<Point> & cc, std::vector<Point> & ring, Face & f)
                const {
                const Point & ri = dt.vertex(vi);
                const Point & rw = dt.vertex(w);
                double dist2 = 0.0;
                for (int d = 0; d < D; d++) {
                    f.normal[d] = rw[d] - ri[d];
                    dist2 += f.normal[d] * f.normal[d];
                }
                f.distance = std::sqrt(dist2);
                for (auto & x : f.normal)
                    x /= f.distance;

                if constexpr (D == 2) {
                    const auto & S = dt.simplex(s);
                    int third = -1;
                    for (int k = 0; k < NV; k++)
                        if (S.v[k] != vi && S.v[k] != w)
                            third = k;
                    const int t = S.n[third];
                    const Point & a = cc[s];
                    const Point & b = cc[t];
                    f.area = std::hypot(b[0] - a[0], b[1] - a[1]);
                    for (int d = 0; d < D; d++)
                        f.centroid[d] = 0.5 * (a[d] + b[d]);
                } else {
                    // The circumcenters of the simplices around the edge, in order
                    ring.clear();
                    const auto & S0 = dt.simplex(s);
                    int x = -1, y = -1;
                    for (int k = 0; k < NV; k++)
                        if (S0.v[k] != vi && S0.v[k] != w)
                            (x < 0 ? x : y) = k;
                    int shared = S0.v[y];
                    int cur = S0.n[x];
                    ring.push_back(cc[s]);
                    int guard = 0;
                    while (cur != s) {
                        if (cur < 0 || ++guard > 100000)
                            throw std::runtime_error("[PeriodicVoronoi] Internal error: open ring around an edge");
                        ring.push_back(cc[cur]);
                        const auto & C = dt.simplex(cur);
                        int ks = -1, z = -1;
                        for (int k = 0; k < NV; k++) {
                            const int v = C.v[k];
                            if (v == shared)
                                ks = k;
                            else if (v != vi && v != w)
                                z = v;
                        }
                        cur = C.n[ks];
                        shared = z;
                    }
                    // Fan triangulation; signed areas along the normal
                    double total = 0.0;
                    Point cen{};
                    const Point & c0 = ring[0];
                    for (size_t k = 1; k + 1 < ring.size(); k++) {
                        double e1[3], e2[3];
                        for (int d = 0; d < 3; d++) {
                            e1[d] = ring[k][d] - c0[d];
                            e2[d] = ring[k + 1][d] - c0[d];
                        }
                        const double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                                              e1[2] * e2[0] - e1[0] * e2[2],
                                              e1[0] * e2[1] - e1[1] * e2[0]};
                        const double a = 0.5 * (cr[0] * f.normal[0] + cr[1] * f.normal[1] + cr[2] * f.normal[2]);
                        total += a;
                        for (int d = 0; d < 3; d++)
                            cen[d] += a * (c0[d] + ring[k][d] + ring[k + 1][d]) / 3.0;
                    }
                    f.area = std::fabs(total);
                    for (int d = 0; d < 3; d++) {
                        if (total != 0.0) {
                            f.centroid[d] = cen[d] / total;
                        } else {
                            double m = 0.0;
                            for (const auto & r : ring)
                                m += r[d];
                            f.centroid[d] = m / double(ring.size());
                        }
                    }
                }
                if (opt.drop_zero_area_faces && f.area <= opt.zero_area_tolerance * std::pow(f.distance, D - 1))
                    return false;
                return true;
            }
        };

    } // namespace DELAUNAY
} // namespace FML

#endif
