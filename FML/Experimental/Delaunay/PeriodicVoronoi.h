#ifndef FML_DELAUNAY_PERIODICVORONOI_HEADER
#define FML_DELAUNAY_PERIODICVORONOI_HEADER

#include <FML/Experimental/Delaunay/Delaunay.h>
#include <FML/Global/ChunkedExchange.h>
#include <FML/Global/Global.h>

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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

namespace FML {
    namespace DELAUNAY {

        //==========================================================================================
        /// Voronoi tessellation of points in a periodic box [0, boxsize)^D (D = 2, 3), with MPI.
        ///
        /// Every task gives its own points (any distribution over the tasks: slabs, pencils, Peano-Hilbert ranges,
        /// or none at all) and gets the cells of exactly those points and the faces touching them: the cells of
        /// the Voronoi tessellation of all the points of all the tasks in the periodic box. Collective.
        ///
        /// Each task triangulates its points together with copies of other tasks' points and periodic images
        /// (ghosts). A first set of ghosts comes from a buffer of initial_buffer mean interparticle spacings around
        /// the bounding box of the task's points. Then every Delaunay simplex touching a local point is certified:
        /// its circumsphere is searched on every task (and periodic image) whose bounding box it intersects, the
        /// points found inside are inserted, and a simplex that was searched and survives is final. Rounds continue
        /// until no task has uncertified simplices. A simplex whose sphere only meets the task's own points needs no
        /// communication.
        ///
        /// Exactness and determinism. Coordinates are mapped to the unit box and rounded to multiples of
        /// 2^-quantization_bits (default 40; a relative change of ~1e-12), so a point, its periodic images and its
        /// copies on other tasks have exactly translated coordinates and all coordinate differences are exact. The
        /// symbolic perturbation orders points by (global id, image shift), which is translation invariant. The
        /// geometry (circumcentres from key-ordered vertices, faces from a canonically ordered ring, sums in a
        /// canonical order, all from exact coordinate differences) depends only on the configuration. Cells and
        /// faces are therefore bitwise identical for any number of tasks and any distribution of the points, and a
        /// face shared by two tasks is identical on both.
        ///
        /// Output on each task, in the user's units:
        ///   cells:  the local points in input order: global id, volume (area in 2D) and centroid (in the
        ///           coordinates of the point, so possibly slightly outside the box)
        ///   faces:  each face touching a local cell once, between point i and the image of point j at
        ///           x_j + shift * boxsize (global ids; j > i, or j == i with a lexicographically positive shift),
        ///           with area (length in 2D), centroid (near point i, in i's box coordinates), unit normal from i to
        ///           the image of j, distance between the two points, the tasks holding i and j and their indices
        ///           there, and local_i/local_j: the local cell index on this task or npos. Sorted by (i, j, shift).
        ///   cell_faces(c): the faces of local cell c: f >= 0 (cell c is the face's i) or -(f + 1) (cell c is the
        ///           face's j; the normal points into it)
        /// Global ids are the given ids, or by default the position in the concatenation of the tasks' points in
        /// rank order (so the input index in a serial run). Faces of zero area (degenerate, e.g. cospherical, points)
        /// are dropped by default.
        //==========================================================================================
        template <int D>
        class PeriodicVoronoi {
          public:
            using Point = std::array<double, D>;
            using Shift = std::array<int, D>;
            static constexpr size_t npos = std::numeric_limits<size_t>::max();

            struct Options {
                /// Initial buffer of ghosts around the bounding box of a task's points, in mean interparticle spacings
                double initial_buffer = 2.0;
                /// Coordinates are rounded to multiples of boxsize * 2^-quantization_bits (at most 48)
                int quantization_bits = 40;
                /// Drop faces with area below zero_area_tolerance * distance^(D-1)
                bool drop_zero_area_faces = true;
                double zero_area_tolerance = 1e-14;
                /// Bound on each MPI message
                size_t max_message_bytes = 8 * 1024 * 1024;
                int max_rounds = 100;
                /// update(): how the triangulation follows the points. Automatic: a sample of the moved vertices is
                /// tried in place and a cost estimate chooses between moving every vertex (in place where its
                /// simplices survive, else removed and inserted again) and triangulating from scratch (keeping the
                /// ghosts). Incremental / Retriangulate force one of them (the result is the same).
                enum class UpdateMethod { Automatic, Incremental, Retriangulate };
                UpdateMethod update_method = UpdateMethod::Automatic;
                /// update(): start again with build() when the ghosts have grown to this many times their number
                /// after the last build
                double max_ghost_growth = 3.0;
                bool verbose = false;
                /// Compute the cells and faces (false: only the triangulation, e.g. for point location)
                bool compute_geometry = true;
            };

            struct Cell {
                size_t id{0};
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
                int task_i{0};
                int task_j{0};
                size_t index_i{0};
                size_t index_j{0};
                size_t local_i{npos};
                size_t local_j{npos};
            };

            struct Statistics {
                size_t npoints{0}; // local points
                size_t npoints_total{0};
                size_t nghosts{0}; // ghosts inserted on this task
                int nrounds{0};
                size_t nqueries{0}; // sphere/box searches sent by this task (including to itself)
                size_t nfaces_dropped{0};
                double initial_buffer{0.0}; // in units of the box
                double time_triangulation{0.0};
                double time_geometry{0.0};
                // Updates (cumulative since the last build)
                int nupdates{0};
                size_t nmoved_in_place{0};
                size_t nreinserted{0};
                int nretriangulated{0};
                int nrebuilds{0}; // update() that called build() (too many ghosts)
            };

            PeriodicVoronoi() = default;
            PeriodicVoronoi(const std::vector<Point> & points,
                            double boxsize,
                            const Options & options = Options{},
                            const std::vector<size_t> * ids = nullptr) {
                build(points, boxsize, options, ids);
            }

            /// Collective
            void build(const std::vector<Point> & points,
                       double boxsize,
                       const Options & options = Options{},
                       const std::vector<size_t> * ids = nullptr) {
                if (!(boxsize > 0.0) || !std::isfinite(boxsize))
                    throw std::invalid_argument("[PeriodicVoronoi] boxsize must be positive");
                if (options.quantization_bits < 1 || options.quantization_bits > 48)
                    throw std::invalid_argument("[PeriodicVoronoi] quantization_bits must be in [1, 48]");
                opt = options;
                L = boxsize;
                stats = Statistics{};
                region = BBox{};
                const auto t0 = std::chrono::steady_clock::now();
                triangulate(points, ids);
                const auto t1 = std::chrono::steady_clock::now();
                if (opt.compute_geometry)
                    compute_geometry();
                const auto t2 = std::chrono::steady_clock::now();
                stats.time_triangulation = std::chrono::duration<double>(t1 - t0).count();
                stats.time_geometry = std::chrono::duration<double>(t2 - t1).count();
            }

            /// Collective. Move the points: points[i] is the new position of the i-th point given to build() (the same
            /// points on the same tasks in the same order). The result is identical (bitwise) to build() with the
            /// new positions, but cheaper: the ghosts are kept and moved with their points, vertices are moved in
            /// place where their simplices survive, the others removed and inserted again or (if more than
            /// incremental_fraction of them) the triangulation is rebuilt, then the simplices are certified again
            /// (new ghosts fetched where needed). Points may cross the box edges.
            void update(const std::vector<Point> & points) {
                int bad = points.size() != u.size() ? 1 : 0;
                FML::MaxOverTasks(&bad);
                if (bad)
                    throw std::invalid_argument("[PeriodicVoronoi::update] Not the same number of points as in build");
                int rebuild =
                    double(stats.nghosts) > opt.max_ghost_growth * double(std::max<size_t>(ghosts_after_build, 1)) ? 1 :
                                                                                                                     0;
                FML::MaxOverTasks(&rebuild);
                if (rebuild) {
                    const Statistics old = stats;
                    const std::vector<size_t> ids = gid_local;
                    const BBox reg = region;
                    build(points, L, opt, &ids);
                    region = reg;
                    certify(false, region.nonempty != 0);
                    stats.nupdates = old.nupdates + 1;
                    stats.nrebuilds = old.nrebuilds + 1;
                    return;
                }
                const auto t0 = std::chrono::steady_clock::now();
                move_vertices(points);
                exchange_boxes();
                certify(false);
                const auto t1 = std::chrono::steady_clock::now();
                if (opt.compute_geometry)
                    compute_geometry();
                const auto t2 = std::chrono::steady_clock::now();
                stats.time_triangulation = std::chrono::duration<double>(t1 - t0).count();
                stats.time_geometry = std::chrono::duration<double>(t2 - t1).count();
                stats.nupdates++;
            }

            size_t ncells() const { return cells_.size(); }
            const std::vector<Cell> & cells() const { return cells_; }
            const std::vector<Face> & faces() const { return faces_; }
            /// The faces of local cell c (see the class description for the sign convention)
            std::pair<const int64_t *, const int64_t *> cell_faces(size_t c) const {
                return {cell_face_list.data() + cell_face_offset[c], cell_face_list.data() + cell_face_offset[c + 1]};
            }
            const Statistics & statistics() const { return stats; }
            const Delaunay<D> & triangulation() const { return dt; }
            double boxsize() const { return L; }
            /// The vertex of the triangulation of local point c
            int local_vertex(size_t c) const { return vertex_of_local[c]; }
            /// The point of vertex v of the triangulation (v >= Delaunay<D>::nsuper()): its task, its index on that
            /// task and the image (periodic shift) of it the vertex is
            void vertex_owner(int v, int & task, size_t & index, Shift & image) const {
                const VertexInfo & I = vinfo(v);
                task = I.task;
                index = I.index;
                image = I.shift;
            }

            /// Collective. Make the triangulation exact also in the box [lo, hi] (user units, need not be inside the
            /// box [0, L): positions there are those of the triangulation, i.e. of the images of the points): every
            /// simplex meeting the box is a Delaunay simplex of all the points, so point location there
            /// (triangulation() locate) gives the true Delaunay simplex. Fetches the ghosts needed. A task with nothing
            /// to cover passes an empty box (lo > hi). The box is kept (as the union with earlier ones) for later
            /// updates
            void cover(const Point & lo, const Point & hi) {
                bool any = true;
                for (int d = 0; d < D; d++)
                    any = any && lo[d] <= hi[d];
                if (any) {
                    for (int d = 0; d < D; d++) {
                        const double a = lo[d] / L, b = hi[d] / L;
                        region.lo[d] = region.nonempty ? std::min(region.lo[d], a) : a;
                        region.hi[d] = region.nonempty ? std::max(region.hi[d], b) : b;
                    }
                    region.nonempty = 1;
                }
                certify(false, any);
            }

            /// The (quantized) position of local point c used for the tessellation, in the user's units
            Point position(size_t c) const {
                Point x = dt.vertex(vertex_of_local[c]);
                for (int d = 0; d < D; d++)
                    x[d] = (x[d] - double(wrap[c][d])) * L;
                return x;
            }

          private:
            static constexpr int NV = D + 1;
            static constexpr int MAXSHIFT = 2; // images up to +-2 boxes
            static constexpr uint64_t NCODE = (D == 2) ? 25 : 125;

            struct VertexInfo {
                size_t gid;
                size_t index; // index on the task holding the point
                int task;
                Shift shift; // image of the point's internal position
                Shift wrap;  // the point's internal position minus its position in the box
            };

            struct BBox {
                double lo[D];
                double hi[D];
                int nonempty;
            };

            // A search sent to a task: its points x with x + shift inside a box [a, b] or a ball |. - a| <= r
            struct Query {
                int requester;
                int ball;
                double a[D];
                double b[D];
                double r;
                Shift shift;
            };
            struct Reply {
                size_t gid;
                size_t index;
                int task;
                Shift shift;
                Shift wrap;
                double pos[D];
            };

            // Per simplex: the vertex keys in increasing order, the lowest-key vertex and the circumcentre relative to
            // it
            struct SimplexGeometry {
                std::array<uint64_t, NV> keys;
                int base;
                Point cc;
            };

            // Geometry of a face in internal units, relative to the canonical point i
            struct FaceInternal {
                Face f;
                Point centroid_rel;
                Point delta; // position of (the image of) j minus that of i, exact
            };

            Options opt{};
            double L{1.0};
            Delaunay<D> dt{};
            std::vector<Point> u{};    // internal positions of the local points: quantized box position + wrap
            std::vector<Shift> wrap{}; // per local point, in [-1, 1] per axis (0 after build)
            BBox mine{};
            BBox region{}; // extra box (internal units) where the triangulation must be exact (cover())
            std::vector<BBox> boxes{};
            std::unordered_set<uint64_t> present{};           // keys of the vertices
            std::vector<std::unordered_set<uint64_t>> sent{}; // keys sent to each task
            double b0{0.0};
            size_t ghosts_after_build{0};
            std::vector<size_t> gid_local{};
            std::vector<int> vertex_of_local{};
            std::vector<VertexInfo> info{}; // per vertex (index - NV)
            std::vector<Cell> cells_{};
            std::vector<Face> faces_{};
            std::vector<size_t> cell_face_offset{};
            std::vector<int64_t> cell_face_list{};
            Statistics stats{};

            // Grid of the local points for the searches
            Point grid_lo{}, grid_hi{};
            std::array<int, D> grid_n{};
            std::vector<size_t> grid_offset{}, grid_points{};

            static uint64_t shift_code(const Shift & s) {
                uint64_t c = 0;
                for (int d = 0; d < D; d++)
                    c = c * uint64_t(2 * MAXSHIFT + 1) + uint64_t(s[d] + MAXSHIFT);
                return c;
            }
            static uint64_t make_key(size_t gid, const Shift & s) { return uint64_t(gid) * NCODE + shift_code(s); }
            static bool shift_positive(const Shift & s) {
                for (int d = 0; d < D; d++)
                    if (s[d] != 0)
                        return s[d] > 0;
                return false;
            }
            const VertexInfo & vinfo(int v) const { return info[size_t(v - NV)]; }
            static int nthreads_now() {
#ifdef USE_OMP
                return omp_get_num_threads();
#else
                return 1;
#endif
            }
            /// Does a simplex have a vertex that is a local point (the local points are the vertices NV .. NV+N-1)
            bool has_local_vertex(const typename Delaunay<D>::Simplex & S) const {
                const int hi = NV + int(vertex_of_local.size());
                for (int k = 0; k < NV; k++)
                    if (S.v[k] >= NV && S.v[k] < hi)
                        return true;
                return false;
            }

            /// Must a simplex be certified: it has a local vertex or it meets the region to cover
            bool relevant(const typename Delaunay<D>::Simplex & S) const {
                if (has_local_vertex(S))
                    return true;
                if (!region.nonempty)
                    return false;
                std::array<Point, NV> q;
                for (int k = 0; k < NV; k++)
                    q[k] = dt.vertex(S.v[k]);
                return simplex_meets_box(q, region);
            }

            /// Does the simplex q meet the box B (separating axes: the box axes, the simplex facet normals and, in
            /// 3D, the cross products of the simplex edges with the box axes; touching counts as meeting)
            static bool simplex_meets_box(const std::array<Point, NV> & q, const BBox & B) {
                auto separated = [&](const Point & a) {
                    double smin = 0.0, smax = 0.0, scale = 0.0;
                    for (int k = 0; k < NV; k++) {
                        double t = 0.0;
                        for (int d = 0; d < D; d++) {
                            t += a[d] * q[k][d];
                            scale = std::max(scale, std::abs(a[d] * q[k][d]));
                        }
                        smin = k == 0 ? t : std::min(smin, t);
                        smax = k == 0 ? t : std::max(smax, t);
                    }
                    double bmin = 0.0, bmax = 0.0;
                    for (int d = 0; d < D; d++) {
                        bmin += a[d] * (a[d] > 0 ? B.lo[d] : B.hi[d]);
                        bmax += a[d] * (a[d] > 0 ? B.hi[d] : B.lo[d]);
                        scale = std::max({scale, std::abs(a[d] * B.lo[d]), std::abs(a[d] * B.hi[d])});
                    }
                    const double eps = 1e-12 * scale;
                    return smax < bmin - eps || smin > bmax + eps;
                };
                Point a{};
                for (int d = 0; d < D; d++) {
                    a.fill(0.0);
                    a[d] = 1.0;
                    if (separated(a))
                        return false;
                }
                if constexpr (D == 2) {
                    for (int i = 0; i < NV; i++) {
                        const Point & p0 = q[i];
                        const Point & p1 = q[(i + 1) % NV];
                        a = {p1[1] - p0[1], p0[0] - p1[0]};
                        if (separated(a))
                            return false;
                    }
                } else {
                    std::array<Point, 6> edges;
                    int ne = 0;
                    for (int i = 0; i < NV; i++)
                        for (int j = i + 1; j < NV; j++) {
                            for (int d = 0; d < D; d++)
                                edges[ne][d] = q[j][d] - q[i][d];
                            ne++;
                        }
                    auto cross = [](const Point & u, const Point & v) {
                        return Point{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
                    };
                    // Facet normals (facet opposite vertex k: edges from one of its vertices)
                    for (int k = 0; k < NV; k++) {
                        const int i0 = (k + 1) % NV, i1 = (k + 2) % NV, i2 = (k + 3) % NV;
                        Point u, v;
                        for (int d = 0; d < D; d++) {
                            u[d] = q[i1][d] - q[i0][d];
                            v[d] = q[i2][d] - q[i0][d];
                        }
                        if (separated(cross(u, v)))
                            return false;
                    }
                    for (int e = 0; e < ne; e++)
                        for (int d = 0; d < D; d++) {
                            Point axis{};
                            axis[d] = 1.0;
                            const Point c = cross(edges[e], axis);
                            if ((c[0] != 0.0 || c[1] != 0.0 || c[2] != 0.0) && separated(c))
                                return false;
                        }
                }
                return true;
            }

            //======================================================================
            // Searches on the local points
            //======================================================================

            void build_grid() {
                const size_t n = u.size();
                grid_offset.clear();
                grid_points.clear();
                if (n == 0)
                    return;
                grid_lo = u[0];
                grid_hi = u[0];
                for (const auto & x : u)
                    for (int d = 0; d < D; d++) {
                        grid_lo[d] = std::min(grid_lo[d], x[d]);
                        grid_hi[d] = std::max(grid_hi[d], x[d]);
                    }
                const int nper = std::max(1, int(std::pow(double(n) / 2.0, 1.0 / D)));
                size_t ncells = 1;
                for (int d = 0; d < D; d++) {
                    grid_n[d] = grid_hi[d] > grid_lo[d] ? nper : 1;
                    ncells *= size_t(grid_n[d]);
                }
                std::vector<size_t> cell(n);
                grid_offset.assign(ncells + 1, 0);
                for (size_t i = 0; i < n; i++) {
                    size_t c = 0;
                    for (int d = 0; d < D; d++)
                        c = c * size_t(grid_n[d]) + size_t(bin(d, u[i][d]));
                    cell[i] = c;
                    grid_offset[c + 1]++;
                }
                for (size_t c = 0; c < ncells; c++)
                    grid_offset[c + 1] += grid_offset[c];
                grid_points.resize(n);
                std::vector<size_t> fill(grid_offset.begin(), grid_offset.end() - 1);
                for (size_t i = 0; i < n; i++)
                    grid_points[fill[cell[i]]++] = i;
            }
            int bin(int d, double x) const {
                if (grid_n[d] == 1)
                    return 0;
                const double w = (x - grid_lo[d]) / (grid_hi[d] - grid_lo[d]) * grid_n[d];
                if (!(w > 0.0))
                    return 0;
                if (w >= grid_n[d] - 1)
                    return grid_n[d] - 1;
                return int(w);
            }

            /// The local points x with x + q.shift inside the box or ball of the query
            template <class F>
            void search(const Query & q, F && found) const {
                if (u.empty())
                    return;
                std::array<int, D> b0, b1;
                for (int d = 0; d < D; d++) {
                    const double lo = (q.ball ? q.a[d] - q.r : q.a[d]) - q.shift[d];
                    const double hi = (q.ball ? q.a[d] + q.r : q.b[d]) - q.shift[d];
                    if (hi < grid_lo[d] || lo > grid_hi[d])
                        return;
                    b0[d] = bin(d, lo);
                    b1[d] = bin(d, hi);
                }
                std::array<int, D> c = b0;
                while (true) {
                    size_t cell = 0;
                    for (int d = 0; d < D; d++)
                        cell = cell * size_t(grid_n[d]) + size_t(c[d]);
                    for (size_t k = grid_offset[cell]; k < grid_offset[cell + 1]; k++) {
                        const size_t i = grid_points[k];
                        Point y;
                        bool inside = true;
                        double r2 = 0.0;
                        for (int d = 0; d < D; d++) {
                            y[d] = u[i][d] + double(q.shift[d]);
                            if (q.ball)
                                r2 += (y[d] - q.a[d]) * (y[d] - q.a[d]);
                            else
                                inside = inside && y[d] >= q.a[d] && y[d] <= q.b[d];
                        }
                        if (q.ball)
                            inside = r2 <= q.r * q.r;
                        if (inside)
                            found(i, y);
                    }
                    int d = D - 1;
                    while (d >= 0 && c[d] == b1[d]) {
                        c[d] = b0[d];
                        d--;
                    }
                    if (d < 0)
                        break;
                    c[d]++;
                }
            }

            //======================================================================
            // Triangulation with certified ghosts
            //======================================================================

            /// Queries to every task (and periodic image) whose bounding box meets the box [a, b] or the ball (a, r)
            void add_queries(const std::vector<BBox> & boxes,
                             bool ball,
                             const Point & a,
                             const Point & b,
                             double r,
                             std::vector<std::vector<Query>> & out,
                             size_t & nq) const {
                Point lo, hi; // bounding box of the region
                for (int d = 0; d < D; d++) {
                    lo[d] = ball ? a[d] - r : a[d];
                    hi[d] = ball ? a[d] + r : b[d];
                }
                for (int t = 0; t < FML::NTasks; t++) {
                    const BBox & B = boxes[size_t(t)];
                    if (!B.nonempty)
                        continue;
                    // The shifts s with B + s overlapping the region's bounding box, axis by axis
                    Shift s0, s1;
                    bool any = true;
                    for (int d = 0; d < D; d++) {
                        s0[d] = std::max(-MAXSHIFT, int(std::ceil(lo[d] - B.hi[d])));
                        s1[d] = std::min(MAXSHIFT, int(std::floor(hi[d] - B.lo[d])));
                        any = any && s0[d] <= s1[d];
                    }
                    if (!any)
                        continue;
                    Shift s = s0;
                    while (true) {
                        bool self = t == FML::ThisTask;
                        for (int d = 0; d < D; d++)
                            self = self && s[d] == 0;
                        bool meets = !self;
                        if (meets && ball) {
                            double dist2 = 0.0;
                            for (int d = 0; d < D; d++) {
                                const double blo = B.lo[d] + s[d], bhi = B.hi[d] + s[d];
                                const double e = a[d] < blo ? blo - a[d] : (a[d] > bhi ? a[d] - bhi : 0.0);
                                dist2 += e * e;
                            }
                            meets = dist2 <= r * r;
                        }
                        if (meets) {
                            Query q;
                            q.requester = FML::ThisTask;
                            q.ball = ball ? 1 : 0;
                            for (int d = 0; d < D; d++) {
                                q.a[d] = a[d];
                                q.b[d] = b[d];
                            }
                            q.r = r;
                            q.shift = s;
                            out[size_t(t)].push_back(q);
                            nq++;
                        }
                        int d = D - 1;
                        while (d >= 0 && s[d] == s1[d]) {
                            s[d] = s0[d];
                            d--;
                        }
                        if (d < 0)
                            break;
                        s[d]++;
                    }
                }
            }

            /// Quantized positions in the unit box
            void quantize(const std::vector<Point> & points, std::vector<Point> & out) const {
                const double q = std::ldexp(1.0, opt.quantization_bits);
                int bad = 0;
                out.assign(points.size(), Point{});
                for (size_t i = 0; i < points.size(); i++)
                    for (int d = 0; d < D; d++) {
                        if (!std::isfinite(points[i][d])) {
                            bad = 1;
                            continue;
                        }
                        double x = points[i][d] / L;
                        x -= std::floor(x);
                        x = std::round(x * q) / q;
                        if (x >= 1.0)
                            x -= 1.0;
                        out[i][d] = x;
                    }
                FML::MaxOverTasks(&bad);
                if (bad)
                    throw std::invalid_argument("[PeriodicVoronoi] Points not finite");
            }

            /// The bounding boxes of the local points of all the tasks
            void exchange_boxes() {
                BBox b{};
                b.nonempty = !u.empty();
                for (int d = 0; d < D; d++) {
                    b.lo[d] = u.empty() ? 0.0 : u[0][d];
                    b.hi[d] = b.lo[d];
                }
                for (const auto & x : u)
                    for (int d = 0; d < D; d++) {
                        b.lo[d] = std::min(b.lo[d], x[d]);
                        b.hi[d] = std::max(b.hi[d], x[d]);
                    }
                mine = b;
                boxes = FML::GatherFromTasks(&b);
                build_grid();
            }

            void triangulate(const std::vector<Point> & points, const std::vector<size_t> * ids) {
                const size_t N = points.size();
                int bad = (ids && ids->size() != N) ? 1 : 0;
                FML::MaxOverTasks(&bad);
                if (bad)
                    throw std::invalid_argument("[PeriodicVoronoi] ids of the wrong size");
                quantize(points, u);
                wrap.assign(N, Shift{});

                // Global ids
                size_t n_local = N;
                const auto counts = FML::GatherFromTasks(&n_local);
                size_t offset = 0, ntotal = 0;
                for (int t = 0; t < FML::NTasks; t++) {
                    if (t < FML::ThisTask)
                        offset += counts[size_t(t)];
                    ntotal += counts[size_t(t)];
                }
                if (ntotal == 0)
                    throw std::invalid_argument("[PeriodicVoronoi] No points");
                gid_local.resize(N);
                for (size_t i = 0; i < N; i++)
                    gid_local[i] = ids ? (*ids)[i] : offset + i;
                stats.npoints = N;
                stats.npoints_total = ntotal;
                exchange_boxes();

                // The local points (vertices NV .. NV+N-1)
                Point lo, hi;
                lo.fill(-2.0 * MAXSHIFT - 0.5);
                hi.fill(1.0 + 2.0 * MAXSHIFT + 0.5);
                dt.init(lo, hi);
                info.clear();
                vertex_of_local.assign(N, -1);
                present.clear();
                present.reserve(2 * N);
                sent.assign(size_t(FML::NTasks), std::unordered_set<uint64_t>{});
                int duplicates = 0;
                for (size_t i : Delaunay<D>::hilbert_order(u)) {
                    const size_t ndup = dt.statistics().nduplicates;
                    const uint64_t key = make_key(gid_local[i], Shift{});
                    const int v = dt.insert(u[i], key);
                    if (dt.statistics().nduplicates != ndup || !present.insert(key).second) {
                        duplicates = 1;
                        continue;
                    }
                    vertex_of_local[i] = v;
                    info.push_back({gid_local[i], i, FML::ThisTask, Shift{}, Shift{}});
                }
                FML::MaxOverTasks(&duplicates);
                if (duplicates)
                    throw std::invalid_argument("[PeriodicVoronoi] Duplicate points (after quantization) or ids");
                b0 = std::min(double(MAXSHIFT), opt.initial_buffer * std::pow(double(ntotal), -1.0 / D));
                stats.initial_buffer = b0;
                certify(true);
                ghosts_after_build = stats.nghosts;
            }

            /// Ghost searches until every simplex with a local vertex (or meeting the region) is certified (first:
            /// start with the buffer around the points; first_region: around the region)
            void certify(bool first, bool first_region = false) {
                {
                    int fr = first_region ? 1 : 0;
                    FML::MaxOverTasks(&fr);
                    first_region = fr != 0;
                }
                const size_t N = u.size();
                double bsuper = 0.0;
                std::vector<uint64_t> certified; // birth + 1 of the certified simplex in a slot
                int duplicates = 0;
                for (int round = 0;; round++) {
                    if (round >= opt.max_rounds)
                        throw std::runtime_error("[PeriodicVoronoi] Too many rounds of ghost searches");
                    std::vector<std::vector<Query>> queries(size_t(FML::NTasks));
                    std::vector<std::pair<int, uint64_t>> searched;
                    size_t nq = 0;
                    int error = 0;
                    auto box_queries = [&](double margin, bool points, bool reg) {
                        for (const BBox * B : {points ? &mine : nullptr, reg ? &region : nullptr}) {
                            if (B == nullptr || !B->nonempty)
                                continue;
                            Point a, b;
                            for (int d = 0; d < D; d++) {
                                a[d] = B->lo[d] - margin;
                                b[d] = B->hi[d] + margin;
                            }
                            add_queries(boxes, false, a, b, 0.0, queries, nq);
                        }
                    };
                    if (round == 0 && (first || first_region)) {
                        box_queries(b0, first, true);
                    } else {
                        // Every live simplex with a local vertex (the local points are vertices NV .. NV+N-1)
                        const auto & simplices = dt.get_simplices();
                        certified.resize(simplices.size(), 0);
                        bool need_super = false;
                        for (size_t sl = 0; sl < simplices.size(); sl++) {
                            const auto & S = simplices[sl];
                            if (!S.alive() || certified[sl] == dt.birth(int(sl)) + 1 || !relevant(S))
                                continue;
                            bool super = false;
                            for (int k = 0; k < NV; k++)
                                super = super || dt.is_super(S.v[k]);
                            if (super) {
                                need_super = true;
                                continue;
                            }
                            const int s = int(sl);
                            Point c;
                            double r2;
                            dt.circumsphere(s, c, r2);
                            const double r = std::sqrt(r2) * (1.0 + 1e-6) + 1e-9;
                            const size_t before = nq;
                            add_queries(boxes, true, c, c, r, queries, nq);
                            if (nq == before)
                                certified[sl] = dt.birth(s) + 1;
                            else
                                searched.push_back({s, dt.birth(s)});
                        }
                        if (need_super) {
                            bsuper = bsuper > 0.0 ? 2.0 * bsuper : 2.0 * b0;
                            if (bsuper > double(MAXSHIFT))
                                error = 1;
                            else
                                box_queries(bsuper, true, true);
                        }
                    }
                    FML::MaxOverTasks(&error);
                    if (error)
                        throw std::runtime_error(
                            "[PeriodicVoronoi] The ghost buffer needed exceeds 2 boxes (too few points?)");
                    size_t nq_all = nq;
                    FML::SumOverTasks(&nq_all);
                    stats.nqueries += nq;
                    if (opt.verbose && FML::ThisTask == 0)
                        std::cout << "[PeriodicVoronoi] round " << round << ": " << nq_all << " searches\n";
                    if (nq_all == 0 && (round > 0 || !(first || first_region)))
                        break;

                    // Searches to the tasks, replies back
                    std::vector<Query> received;
                    FML::COMMUNICATION::exchange_records_chunked(
                        queries,
                        [&](const Query * r, size_t n) { received.insert(received.end(), r, r + n); },
                        opt.max_message_bytes);
                    std::vector<std::vector<Reply>> replies(size_t(FML::NTasks));
                    for (const auto & qr : received) {
                        auto & seen = sent[size_t(qr.requester)];
                        search(qr, [&](size_t i, const Point & y) {
                            if (!seen.insert(make_key(gid_local[i], qr.shift)).second)
                                return;
                            Reply rep;
                            rep.gid = gid_local[i];
                            rep.index = i;
                            rep.task = FML::ThisTask;
                            rep.shift = qr.shift;
                            rep.wrap = wrap[i];
                            for (int d = 0; d < D; d++)
                                rep.pos[d] = y[d];
                            replies[size_t(qr.requester)].push_back(rep);
                        });
                    }
                    std::vector<Reply> got;
                    FML::COMMUNICATION::exchange_records_chunked(
                        replies,
                        [&](const Reply * r, size_t n) { got.insert(got.end(), r, r + n); },
                        opt.max_message_bytes);

                    // Insert the new ghosts
                    std::vector<Point> pos;
                    std::vector<size_t> which;
                    for (size_t k = 0; k < got.size(); k++) {
                        if (!present.insert(make_key(got[k].gid, got[k].shift)).second)
                            continue;
                        Point x;
                        for (int d = 0; d < D; d++)
                            x[d] = got[k].pos[d];
                        pos.push_back(x);
                        which.push_back(k);
                    }
                    for (size_t k : Delaunay<D>::hilbert_order(pos)) {
                        const Reply & r = got[which[k]];
                        const size_t ndup = dt.statistics().nduplicates;
                        dt.insert(pos[k], make_key(r.gid, r.shift));
                        if (dt.statistics().nduplicates != ndup) {
                            duplicates = 1;
                            continue;
                        }
                        info.push_back({r.gid, r.index, r.task, r.shift, r.wrap});
                    }
                    stats.nghosts += pos.size();
                    FML::MaxOverTasks(&duplicates);
                    if (duplicates)
                        throw std::invalid_argument("[PeriodicVoronoi] Duplicate points (after quantization)");
                    // Searched simplices that survived are final
                    certified.resize(dt.get_simplices().size(), 0);
                    for (const auto & sb : searched) {
                        const auto & S = dt.simplex(sb.first);
                        if (S.alive() && dt.birth(sb.first) == sb.second)
                            certified[size_t(sb.first)] = sb.second + 1;
                    }
                    stats.nrounds = round + 1;
                }
                (void)N;
            }

            /// Move the vertices to the new positions: the local points continuously (internal positions are the
            /// box positions plus a wrap count, kept in [-1, 1] per axis, so a point crossing the box edge moves a
            /// little instead of jumping), the ghosts to the new positions of their points (pulled from their tasks)
            void move_vertices(const std::vector<Point> & points) {
                const size_t N = u.size();
                std::vector<Point> wrapped;
                quantize(points, wrapped);
                for (size_t i = 0; i < N; i++)
                    for (int d = 0; d < D; d++) {
                        const double old_box = u[i][d] - double(wrap[i][d]);
                        const double step = wrapped[i][d] - old_box;
                        int w = wrap[i][d] + (step > 0.5 ? -1 : (step < -0.5 ? 1 : 0));
                        if (w > 1 || w < -1)
                            w = 0;
                        wrap[i][d] = w;
                        u[i][d] = wrapped[i][d] + double(w);
                    }

                // New positions (and wraps) of the ghosts from their tasks
                struct Request {
                    int requester;
                    size_t index;
                };
                struct Answer {
                    int task;
                    size_t index;
                    double pos[D];
                    Shift wrap;
                };
                std::vector<std::vector<Request>> requests(size_t(FML::NTasks));
                std::vector<std::unordered_set<size_t>> asked(size_t(FML::NTasks));
                for (const auto & I : info)
                    if (I.task != FML::ThisTask && asked[size_t(I.task)].insert(I.index).second)
                        requests[size_t(I.task)].push_back({FML::ThisTask, I.index});
                std::vector<Request> todo;
                FML::COMMUNICATION::exchange_records_chunked(
                    requests,
                    [&](const Request * r, size_t n) { todo.insert(todo.end(), r, r + n); },
                    opt.max_message_bytes);
                std::vector<std::vector<Answer>> answers(size_t(FML::NTasks));
                for (const auto & r : todo) {
                    Answer a;
                    a.task = FML::ThisTask;
                    a.index = r.index;
                    for (int d = 0; d < D; d++)
                        a.pos[d] = u[r.index][d];
                    a.wrap = wrap[r.index];
                    answers[size_t(r.requester)].push_back(a);
                }
                std::unordered_map<uint64_t, std::pair<Point, Shift>> remote;
                FML::COMMUNICATION::exchange_records_chunked(
                    answers,
                    [&](const Answer * a, size_t n) {
                        for (size_t k = 0; k < n; k++) {
                            Point x;
                            for (int d = 0; d < D; d++)
                                x[d] = a[k].pos[d];
                            remote[(uint64_t(a[k].task) << 40) | uint64_t(a[k].index)] = {x, a[k].wrap};
                        }
                    },
                    opt.max_message_bytes);

                // The new position of every vertex
                std::vector<Point> target(info.size());
                for (size_t k = 0; k < info.size(); k++) {
                    auto & I = info[k];
                    Point base;
                    if (I.task == FML::ThisTask) {
                        base = u[I.index];
                        I.wrap = wrap[I.index];
                    } else {
                        const auto & rw = remote.at((uint64_t(I.task) << 40) | uint64_t(I.index));
                        base = rw.first;
                        I.wrap = rw.second;
                    }
                    for (int d = 0; d < D; d++)
                        target[k][d] = base[d] + double(I.shift[d]);
                }

                // Move in place where the simplices survive and remove and insert the others, or triangulate from
                // scratch. Estimated costs (measured, per vertex, in units of a fresh insertion): an in-place check
                // ~0.6 (3D) / 0.4 (2D), a removal and insertion ~6 (3D) / 6 (2D)
                std::vector<int> moved;
                for (size_t k = 0; k < info.size(); k++)
                    if (dt.vertex(NV + int(k)) != target[k])
                        moved.push_back(NV + int(k));
                using UM = typename Options::UpdateMethod;
                bool incremental = opt.update_method == UM::Incremental;
                std::vector<int> failed;
                std::vector<char> tried(moved.size(), 0);
                if (opt.update_method == UM::Automatic && !moved.empty()) {
                    const size_t nsample = std::min<size_t>(moved.size(), 256);
                    size_t nfail = 0;
                    for (size_t m = 0; m < nsample; m++) {
                        const size_t k = m * moved.size() / nsample;
                        tried[k] = 1;
                        if (!dt.move_in_place(moved[k], target[size_t(moved[k] - NV)])) {
                            failed.push_back(moved[k]);
                            nfail++;
                        }
                    }
                    const double check = D == 3 ? 0.6 : 0.4, reinsert = 6.0;
                    const double fail_fraction = double(nfail) / double(nsample);
                    incremental = double(moved.size()) * (check + fail_fraction * reinsert) < double(info.size());
                }
                if (incremental) {
                    for (size_t k = 0; k < moved.size(); k++)
                        if (!tried[k] && !dt.move_in_place(moved[k], target[size_t(moved[k] - NV)]))
                            failed.push_back(moved[k]);
                }
                int error = 0;
                try {
                    if (incremental) {
                        for (int v : failed)
                            dt.move(v, target[size_t(v - NV)]);
                        stats.nreinserted += failed.size();
                        stats.nmoved_in_place += moved.size() - failed.size();
                    } else {
                        for (int v : moved)
                            dt.set_position(v, target[size_t(v - NV)]);
                        dt.retriangulate();
                        stats.nretriangulated++;
                    }
                } catch (const std::invalid_argument &) {
                    error = 1;
                }
                FML::MaxOverTasks(&error);
                if (error)
                    throw std::invalid_argument("[PeriodicVoronoi::update] Duplicate points (after quantization)");
            }

            //======================================================================
            // Geometry (deterministic: depends only on the configuration)
            //======================================================================

            /// Vertices of a simplex ordered by key (a translation invariant order)
            std::array<int, NV> sorted_vertices(int s) const {
                std::array<int, NV> v = dt.simplex(s).v;
                std::sort(v.begin(), v.end(), [&](int a, int b) { return dt.key(a) < dt.key(b); });
                return v;
            }

            /// Circumcentre relative to the first (lowest-key) vertex, from exact coordinate differences
            Point circumcenter_rel(const std::array<int, NV> & v) const {
                const Point & p0 = dt.vertex(v[0]);
                double A[D][D], b[D];
                for (int k = 0; k < D; k++) {
                    const Point & pk = dt.vertex(v[k + 1]);
                    b[k] = 0.0;
                    for (int d = 0; d < D; d++) {
                        A[k][d] = pk[d] - p0[d];
                        b[k] += A[k][d] * A[k][d];
                    }
                    b[k] *= 0.5;
                }
                Point c;
                if constexpr (D == 2) {
                    const double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
                    c[0] = (b[0] * A[1][1] - b[1] * A[0][1]) / det;
                    c[1] = (A[0][0] * b[1] - A[1][0] * b[0]) / det;
                } else {
                    const double r12[3] = {A[1][1] * A[2][2] - A[1][2] * A[2][1],
                                           A[1][2] * A[2][0] - A[1][0] * A[2][2],
                                           A[1][0] * A[2][1] - A[1][1] * A[2][0]};
                    const double r20[3] = {A[2][1] * A[0][2] - A[2][2] * A[0][1],
                                           A[2][2] * A[0][0] - A[2][0] * A[0][2],
                                           A[2][0] * A[0][1] - A[2][1] * A[0][0]};
                    const double r01[3] = {A[0][1] * A[1][2] - A[0][2] * A[1][1],
                                           A[0][2] * A[1][0] - A[0][0] * A[1][2],
                                           A[0][0] * A[1][1] - A[0][1] * A[1][0]};
                    const double det = A[0][0] * r12[0] + A[0][1] * r12[1] + A[0][2] * r12[2];
                    for (int d = 0; d < 3; d++)
                        c[d] = (b[0] * r12[d] + b[1] * r20[d] + b[2] * r01[d]) / det;
                }
                return c;
            }

            void compute_geometry() {
                const size_t N = vertex_of_local.size();
                const auto & simplices = dt.get_simplices();
                // Circumcentres (relative to the lowest-key vertex) of the simplices touching local points
                std::vector<SimplexGeometry> sg(simplices.size());
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t s = 0; s < simplices.size(); s++) {
                    if (!simplices[s].alive() || !has_local_vertex(simplices[s]))
                        continue;
                    const auto v = sorted_vertices(int(s));
                    for (int k = 0; k < NV; k++)
                        sg[s].keys[k] = dt.key(v[k]);
                    sg[s].base = v[0];
                    sg[s].cc = circumcenter_rel(v);
                }

                std::vector<FaceInternal> all;
                size_t ndropped = 0;
                int error = 0;
#ifdef USE_OMP
#pragma omp parallel reduction(+ : ndropped)
#endif
                {
                    std::vector<FaceInternal> mine;
                    mine.reserve(8 * N / size_t(std::max(1, nthreads_now())) + 16);
                    std::vector<int> star, ring;
                    std::vector<Point> pts;
                    std::vector<unsigned> visited, vseen(size_t(dt.nvertices()), 0u);
                    unsigned stamp = 0;
#ifdef USE_OMP
#pragma omp for schedule(dynamic, 256)
#endif
                    // Over the local vertices in insertion (Peano-Hilbert) order, for memory locality
                    for (size_t k = 0; k < N; k++) {
                        const int vi = NV + int(k);
                        dt.incident_simplices(vi, star, visited, ++stamp);
                        for (int s : star) {
                            const auto & S = dt.simplex(s);
                            for (int k = 0; k < NV; k++) {
                                const int w = S.v[k];
                                if (w == vi || vseen[size_t(w)] == stamp)
                                    continue;
                                vseen[size_t(w)] = stamp;
                                if (dt.is_super(w)) {
#ifdef USE_OMP
#pragma omp atomic write
#endif
                                    error = 1;
                                    continue;
                                }
                                const VertexInfo & I = vinfo(vi);
                                const VertexInfo & W = vinfo(w);
                                const bool canon_vi = W.gid > I.gid || (W.gid == I.gid && shift_positive(W.shift));
                                if (!canon_vi && W.task == FML::ThisTask)
                                    continue; // computed for the local cell of W
                                mine.emplace_back();
                                FaceInternal & F = mine.back();
                                const int base = canon_vi ? vi : w;
                                const int other = canon_vi ? w : vi;
                                if (!face_geometry(base, other, s, sg, ring, pts, F)) {
                                    mine.pop_back();
                                    ndropped++;
                                    continue;
                                }
                                const VertexInfo & Bi = vinfo(base);
                                const VertexInfo & Bj = vinfo(other);
                                F.f.i = Bi.gid;
                                F.f.j = Bj.gid;
                                for (int d = 0; d < D; d++)
                                    F.f.shift[d] = (Bj.shift[d] + Bj.wrap[d]) - (Bi.shift[d] + Bi.wrap[d]);
                                F.f.task_i = Bi.task;
                                F.f.task_j = Bj.task;
                                F.f.index_i = Bi.index;
                                F.f.index_j = Bj.index;
                                F.f.local_i = Bi.task == FML::ThisTask ? Bi.index : npos;
                                F.f.local_j = Bj.task == FML::ThisTask ? Bj.index : npos;
                                // Centroid in i's box coordinates (the base vertex minus its image shift)
                                const Point & pb = dt.vertex(base);
                                for (int d = 0; d < D; d++)
                                    F.f.centroid[d] = (pb[d] - double(Bi.shift[d] + Bi.wrap[d])) + F.centroid_rel[d];
                            }
                        }
                    }
#ifdef USE_OMP
#pragma omp critical
#endif
                    {
                        if (all.empty())
                            all.swap(mine);
                        else
                            all.insert(all.end(), mine.begin(), mine.end());
                    }
                }
                FML::MaxOverTasks(&error);
                if (error)
                    throw std::runtime_error("[PeriodicVoronoi] Internal error: incomplete cell");

                // Faces sorted by (i, j, shift) (unique on a task), via an index sort
                stats.nfaces_dropped = ndropped;
                std::vector<size_t> perm(all.size()); // face f (sorted) is all[perm[f]]
                {
                    struct SortKey {
                        size_t i, j;
                        uint64_t code;
                        size_t index;
                    };
                    std::vector<SortKey> keys(all.size());
                    for (size_t k = 0; k < all.size(); k++)
                        keys[k] = {all[k].f.i, all[k].f.j, shift_code(all[k].f.shift), k};
                    std::sort(keys.begin(), keys.end(), [](const SortKey & a, const SortKey & b) {
                        if (a.i != b.i)
                            return a.i < b.i;
                        if (a.j != b.j)
                            return a.j < b.j;
                        return a.code < b.code;
                    });
                    for (size_t k = 0; k < keys.size(); k++)
                        perm[k] = keys[k].index;
                }
                std::vector<size_t> count(N + 1, 0);
                for (const auto & F : all) {
                    if (F.f.local_i != npos)
                        count[F.f.local_i + 1]++;
                    if (F.f.local_j != npos)
                        count[F.f.local_j + 1]++;
                }
                cell_face_offset.assign(N + 1, 0);
                for (size_t i = 0; i < N; i++)
                    cell_face_offset[i + 1] = cell_face_offset[i] + count[i + 1];
                cell_face_list.assign(cell_face_offset[N], 0);
                std::vector<size_t> fill(cell_face_offset.begin(), cell_face_offset.end() - 1);
                for (size_t f = 0; f < all.size(); f++) {
                    const Face & F = all[perm[f]].f;
                    if (F.local_i != npos)
                        cell_face_list[fill[F.local_i]++] = int64_t(f);
                    if (F.local_j != npos)
                        cell_face_list[fill[F.local_j]++] = -int64_t(f) - 1;
                }

                // Cells: volumes and centroids from the faces in the canonical (sorted) order
                cells_.assign(N, Cell{});
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t c = 0; c < N; c++) {
                    double vol = 0.0;
                    Point cen{};
                    for (size_t k = cell_face_offset[c]; k < cell_face_offset[c + 1]; k++) {
                        const int64_t code = cell_face_list[k];
                        const FaceInternal & F = all[perm[size_t(code >= 0 ? code : -code - 1)]];
                        const double cone = F.f.area * 0.5 * F.f.distance / D;
                        vol += cone;
                        for (int d = 0; d < D; d++) {
                            const double rel = code >= 0 ? F.centroid_rel[d] : F.centroid_rel[d] - F.delta[d];
                            cen[d] += cone * (D * rel) / (D + 1);
                        }
                    }
                    cells_[c].id = gid_local[c];
                    cells_[c].volume = vol;
                    for (int d = 0; d < D; d++)
                        cells_[c].centroid[d] = (u[c][d] - double(wrap[c][d])) + (vol > 0.0 ? cen[d] / vol : 0.0);
                }

                // To the user's units
                const double LD1 = std::pow(L, D - 1);
                for (auto & c : cells_) {
                    c.volume *= LD1 * L;
                    for (auto & x : c.centroid)
                        x *= L;
                }
                faces_.resize(all.size());
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
                for (size_t f = 0; f < all.size(); f++) {
                    faces_[f] = all[perm[f]].f;
                    faces_[f].area *= LD1;
                    faces_[f].distance *= L;
                    for (auto & x : faces_[f].centroid)
                        x *= L;
                }
            }

            /// The Voronoi face between vertices base (the canonical point i) and other (an edge of simplex s), in
            /// internal units relative to base. Returns false if the face has zero area and is dropped.
            bool face_geometry(int base,
                               int other,
                               int s,
                               const std::vector<SimplexGeometry> & sg,
                               std::vector<int> & ring,
                               std::vector<Point> & pts,
                               FaceInternal & F) const {
                const Point & rb = dt.vertex(base);
                const Point & ro = dt.vertex(other);
                double dist2 = 0.0;
                for (int d = 0; d < D; d++) {
                    F.delta[d] = ro[d] - rb[d];
                    dist2 += F.delta[d] * F.delta[d];
                }
                F.f.distance = std::sqrt(dist2);
                for (int d = 0; d < D; d++)
                    F.f.normal[d] = F.delta[d] / F.f.distance;

                // The simplices around the edge in cyclic order
                ring.clear();
                const auto & S0 = dt.simplex(s);
                if constexpr (D == 2) {
                    int third = -1;
                    for (int k = 0; k < NV; k++)
                        if (S0.v[k] != base && S0.v[k] != other)
                            third = k;
                    ring.push_back(s);
                    ring.push_back(S0.n[third]);
                } else {
                    int x = -1, y = -1;
                    for (int k = 0; k < NV; k++)
                        if (S0.v[k] != base && S0.v[k] != other)
                            (x < 0 ? x : y) = k;
                    int shared = S0.v[y];
                    int cur = S0.n[x];
                    ring.push_back(s);
                    int guard = 0;
                    while (cur != s) {
                        if (cur < 0 || ++guard > 100000)
                            throw std::runtime_error("[PeriodicVoronoi] Internal error: open ring around an edge");
                        ring.push_back(cur);
                        const auto & C = dt.simplex(cur);
                        int ks = -1, z = -1;
                        for (int k = 0; k < NV; k++) {
                            const int v = C.v[k];
                            if (v == shared)
                                ks = k;
                            else if (v != base && v != other)
                                z = v;
                        }
                        cur = C.n[ks];
                        shared = z;
                    }
                }
                // Canonical start (the smallest simplex) and direction (towards its smaller ring neighbour); the
                // circumcentres relative to base from exact differences
                const size_t n = ring.size();
                size_t m = 0;
                for (size_t k = 1; k < n; k++)
                    if (sg[size_t(ring[k])].keys < sg[size_t(ring[m])].keys)
                        m = k;
                const bool forward =
                    n < 3 || sg[size_t(ring[(m + 1) % n])].keys < sg[size_t(ring[(m + n - 1) % n])].keys;
                pts.resize(n);
                for (size_t k = 0; k < n; k++) {
                    const SimplexGeometry & G = sg[size_t(ring[forward ? (m + k) % n : (m + n - k) % n])];
                    const Point & p0 = dt.vertex(G.base);
                    for (int d = 0; d < D; d++)
                        pts[k][d] = (p0[d] - rb[d]) + G.cc[d];
                }

                if constexpr (D == 2) {
                    F.f.area = std::hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
                    for (int d = 0; d < D; d++)
                        F.centroid_rel[d] = 0.5 * (pts[0][d] + pts[1][d]);
                } else {
                    double total = 0.0;
                    Point cen{};
                    const Point & c0 = pts[0];
                    for (size_t k = 1; k + 1 < n; k++) {
                        double e1[3], e2[3];
                        for (int d = 0; d < 3; d++) {
                            e1[d] = pts[k][d] - c0[d];
                            e2[d] = pts[k + 1][d] - c0[d];
                        }
                        const double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1],
                                              e1[2] * e2[0] - e1[0] * e2[2],
                                              e1[0] * e2[1] - e1[1] * e2[0]};
                        const double a = 0.5 * (cr[0] * F.f.normal[0] + cr[1] * F.f.normal[1] + cr[2] * F.f.normal[2]);
                        total += a;
                        for (int d = 0; d < 3; d++)
                            cen[d] += a * (c0[d] + pts[k][d] + pts[k + 1][d]) / 3.0;
                    }
                    F.f.area = std::fabs(total);
                    for (int d = 0; d < 3; d++) {
                        if (total != 0.0) {
                            F.centroid_rel[d] = cen[d] / total;
                        } else {
                            double mean = 0.0;
                            for (const auto & p : pts)
                                mean += p[d];
                            F.centroid_rel[d] = mean / double(n);
                        }
                    }
                }
                if (opt.drop_zero_area_faces && F.f.area <= opt.zero_area_tolerance * (D == 3 ? dist2 : F.f.distance))
                    return false;
                return true;
            }
        };

    } // namespace DELAUNAY
} // namespace FML

#endif
