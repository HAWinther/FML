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
                bool verbose = false;
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
                const auto t0 = std::chrono::steady_clock::now();
                triangulate(points, ids);
                const auto t1 = std::chrono::steady_clock::now();
                compute_geometry();
                const auto t2 = std::chrono::steady_clock::now();
                stats.time_triangulation = std::chrono::duration<double>(t1 - t0).count();
                stats.time_geometry = std::chrono::duration<double>(t2 - t1).count();
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
            /// The (quantized) position of local point c used for the tessellation, in the user's units
            Point position(size_t c) const {
                Point x = dt.vertex(vertex_of_local[c]);
                for (auto & v : x)
                    v *= L;
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
                Shift shift;
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
            std::vector<Point> u{}; // local quantized positions in [0,1)
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

            void triangulate(const std::vector<Point> & points, const std::vector<size_t> * ids) {
                const size_t N = points.size();
                const double q = std::ldexp(1.0, opt.quantization_bits);
                int bad = 0;
                u.assign(N, Point{});
                for (size_t i = 0; i < N; i++)
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
                        u[i][d] = x;
                    }
                if (ids && ids->size() != N)
                    bad = 1;
                FML::MaxOverTasks(&bad);
                if (bad)
                    throw std::invalid_argument("[PeriodicVoronoi] Points not finite, or ids of the wrong size");

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

                // Bounding boxes of all the tasks
                BBox mine{};
                mine.nonempty = N > 0;
                for (int d = 0; d < D; d++) {
                    mine.lo[d] = N ? 1.0 : 0.0;
                    mine.hi[d] = 0.0;
                }
                for (const auto & x : u)
                    for (int d = 0; d < D; d++) {
                        mine.lo[d] = std::min(mine.lo[d], x[d]);
                        mine.hi[d] = std::max(mine.hi[d], x[d]);
                    }
                const auto boxes = FML::GatherFromTasks(&mine);
                build_grid();

                // The local points
                Point lo, hi;
                lo.fill(-MAXSHIFT - 0.5);
                hi.fill(1.0 + MAXSHIFT + 0.5);
                dt.init(lo, hi);
                info.clear();
                vertex_of_local.assign(N, -1);
                std::unordered_set<uint64_t> present;
                present.reserve(2 * N);
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
                    info.push_back({gid_local[i], i, FML::ThisTask, Shift{}});
                }
                FML::MaxOverTasks(&duplicates);
                if (duplicates)
                    throw std::invalid_argument("[PeriodicVoronoi] Duplicate points (after quantization) or ids");

                // Certification rounds
                const double b0 = std::min(double(MAXSHIFT), opt.initial_buffer * std::pow(double(ntotal), -1.0 / D));
                stats.initial_buffer = b0;
                double bsuper = 0.0;
                std::vector<uint64_t> certified; // birth + 1 of the certified simplex in a slot
                std::vector<std::unordered_set<uint64_t>> sent(size_t(FML::NTasks));
                for (int round = 0;; round++) {
                    if (round >= opt.max_rounds)
                        throw std::runtime_error("[PeriodicVoronoi] Too many rounds of ghost searches");
                    std::vector<std::vector<Query>> queries(size_t(FML::NTasks));
                    std::vector<std::pair<int, uint64_t>> searched;
                    size_t nq = 0;
                    int error = 0;
                    auto box_queries = [&](double margin) {
                        if (!mine.nonempty)
                            return;
                        Point a, b;
                        for (int d = 0; d < D; d++) {
                            a[d] = mine.lo[d] - margin;
                            b[d] = mine.hi[d] + margin;
                        }
                        add_queries(boxes, false, a, b, 0.0, queries, nq);
                    };
                    if (round == 0) {
                        box_queries(b0);
                    } else {
                        // Every live simplex with a local vertex (the local points are vertices NV .. NV+N-1)
                        const auto & simplices = dt.get_simplices();
                        certified.resize(simplices.size(), 0);
                        bool need_super = false;
                        for (size_t sl = 0; sl < simplices.size(); sl++) {
                            const auto & S = simplices[sl];
                            if (!S.alive() || certified[sl] == dt.birth(int(sl)) + 1 || !has_local_vertex(S))
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
                                box_queries(bsuper);
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
                    if (nq_all == 0 && round > 0)
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
                        info.push_back({r.gid, r.index, r.task, r.shift});
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
                                    F.f.shift[d] = Bj.shift[d] - Bi.shift[d];
                                F.f.task_i = Bi.task;
                                F.f.task_j = Bj.task;
                                F.f.index_i = Bi.index;
                                F.f.index_j = Bj.index;
                                F.f.local_i = Bi.task == FML::ThisTask ? Bi.index : npos;
                                F.f.local_j = Bj.task == FML::ThisTask ? Bj.index : npos;
                                // Centroid in i's box coordinates (the base vertex minus its image shift)
                                const Point & pb = dt.vertex(base);
                                for (int d = 0; d < D; d++)
                                    F.f.centroid[d] = (pb[d] - double(Bi.shift[d])) + F.centroid_rel[d];
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
                        cells_[c].centroid[d] = u[c][d] + (vol > 0.0 ? cen[d] / vol : 0.0);
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
