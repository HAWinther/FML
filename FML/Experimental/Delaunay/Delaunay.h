#ifndef FML_DELAUNAY_HEADER
#define FML_DELAUNAY_HEADER

#include <FML/Experimental/Delaunay/Predicates.h>
#include <FML/PeanoHilbertDomain/HilbertCurve.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML {

    //==========================================================================================
    /// Native Delaunay triangulation and Voronoi tessellation in 2D and 3D (no external dependencies).
    ///
    /// Delaunay<D>: incremental Bowyer-Watson insertion inside an enclosing super simplex, with exact predicates
    /// (Predicates.h) and a symbolic perturbation for cospherical points, so that any input (lattices included)
    /// gives a valid, unique triangulation. PeriodicVoronoi<D> (PeriodicVoronoi.h) builds the Voronoi cells and faces
    /// of points in a periodic box on top of it.
    //==========================================================================================

    namespace DELAUNAY {

        //==========================================================================================
        /// Delaunay triangulation of points in D = 2 or 3 dimensions.
        ///
        /// The first D+1 vertices are the corners of a large super simplex that contains every point; the
        /// triangulation is the Delaunay triangulation of the points and these corners, so simplices near the convex
        /// hull of the points have super vertices (PeriodicVoronoi surrounds the points by periodic images so that
        /// the cells it uses never touch them).
        ///
        /// Degenerate input (D+2 or more cospherical points) is resolved by the symbolic perturbation of Devillers &
        /// Teillaud (2011, "Perturbations for Delaunay and weighted Delaunay 3D triangulations"), the scheme CGAL uses:
        /// points are perturbed in the lifted direction in the order of a per-vertex key (by default the vertex
        /// index). The result is the Delaunay triangulation of the perturbed points, a valid triangulation of the
        /// input. Duplicate points are not inserted: insert returns the existing vertex.
        ///
        /// Simplex s has vertices v[0..D] (positively oriented) and neighbours n[k] across the facet opposite v[k]
        /// (-1 on the outer facets of the super simplex). Deleted simplices are recycled.
        //==========================================================================================
        template <int D>
        class Delaunay {
            static_assert(D == 2 || D == 3, "Delaunay: D must be 2 or 3");

          public:
            static constexpr int NV = D + 1;
            using Point = std::array<double, D>;

            struct Simplex {
                std::array<int, NV> v{};
                std::array<int, NV> n{};
                bool alive() const { return v[0] >= 0; }
            };

            /// Statistics of the insertions
            struct Statistics {
                size_t ninserted{0};
                size_t nduplicates{0};
                size_t nwalk_steps{0};
                size_t ncavity_simplices{0};
                size_t nperturbed{0};
                size_t nmoved_in_place{0};   // moves that kept the simplices
                size_t nmoved_reinserted{0}; // moves by removal and insertion
            };

            Delaunay() = default;

            /// Create the super simplex enclosing the box [lo, hi] (with a wide margin). Clears the triangulation.
            Delaunay(const Point & lo, const Point & hi) { init(lo, hi); }

            /// super_key_top: the super vertices get the keys super_key_top - (D+1) ... super_key_top - 1 (the
            /// default is the top of the uint64 range); point keys must differ from them. super_scale: the size of the
            /// super simplex in units of the box (default 30)
            void init(const Point & lo,
                      const Point & hi,
                      uint64_t super_key_top = std::numeric_limits<uint64_t>::max(),
                      double super_scale = 30.0) {

                vertices.clear();
                keys.clear();
                simplices.clear();
                births.clear();
                next_birth = 0;
                free_list.clear();
                vertex_simplex.clear();
                mark.clear();
                stats = Statistics{};
                double size = 0.0;
                for (int d = 0; d < D; d++) {
                    if (!(hi[d] > lo[d]) || !std::isfinite(lo[d]) || !std::isfinite(hi[d]))
                        throw std::invalid_argument("[Delaunay] Invalid bounding box");
                    size = std::max(size, hi[d] - lo[d]);
                }
                box_lo = lo;
                box_hi = hi;
                // The corner simplex {x_d >= a, sum_d (x_d - a) <= S} contains the cube [a, a + S/D]^D. Take a wide
                // margin so the super vertices are far from the points (exact predicates make this safe)
                const double a = *std::min_element(lo.begin(), lo.end()) - (super_scale / 3.0) * size;
                const double S = D * super_scale * size;
                for (int k = 0; k < NV; k++) {
                    Point p;
                    p.fill(a);
                    if (k > 0)
                        p[k - 1] = a + S;
                    add_vertex(p, super_key_top - uint64_t(NV - k));
                }
                Simplex s;
                for (int k = 0; k < NV; k++) {
                    s.v[k] = k;
                    s.n[k] = -1;
                }
                simplices.push_back(s);
                mark.push_back(0);
                births.push_back(next_birth++);
                if (orient_of(s.v) < 0)
                    std::swap(simplices[0].v[0], simplices[0].v[1]);
                for (int k = 0; k < NV; k++)
                    vertex_simplex[k] = 0;
                last_simplex = 0;
            }

            /// Insert a point with a perturbation key (the order of the symbolic perturbation; give distinct keys).
            /// Returns the vertex index (an existing vertex if the point is a duplicate). The point must be inside the
            /// bounding box given to init.
            int insert(const Point & x, uint64_t key) {
                for (int d = 0; d < D; d++)
                    if (!(x[d] >= box_lo[d] && x[d] <= box_hi[d]))
                        throw std::invalid_argument("[Delaunay] Point outside the bounding box (or not finite)");
                const int s0 = locate(x.data(), last_simplex);
                // A duplicate is a vertex of the simplex it is located in
                for (int k = 0; k < NV; k++) {
                    const int w = simplices[s0].v[k];
                    if (vertices[w] == x) {
                        stats.nduplicates++;
                        return w;
                    }
                }
                const int q = add_vertex(x, key);
                insert_vertex(q, s0);
                stats.ninserted++;
                return q;
            }

            /// Insert points in a spatially sorted (Peano-Hilbert) order. Keys default to the vertex index. Returns
            /// the vertex index of each point.
            std::vector<int> insert(const std::vector<Point> & points,
                                    const std::vector<uint64_t> * point_keys = nullptr) {
                if (point_keys && point_keys->size() != points.size())
                    throw std::invalid_argument("[Delaunay] Keys and points have different sizes");
                std::vector<int> result(points.size(), -1);
                for (size_t i : hilbert_order(points)) {
                    const uint64_t key = point_keys ? (*point_keys)[i] : uint64_t(vertices.size());
                    result[i] = insert(points[i], key);
                }
                return result;
            }

            /// The order of points along a Peano-Hilbert curve through their bounding box
            static std::vector<size_t> hilbert_order(const std::vector<Point> & points) {
                using HC = FML::GRID::HilbertCurve<D>;
                std::vector<size_t> order(points.size());
                std::iota(order.begin(), order.end(), size_t(0));
                if (points.size() < 2)
                    return order;
                Point lo = points[0], hi = points[0];
                for (const auto & p : points)
                    for (int d = 0; d < D; d++) {
                        lo[d] = std::min(lo[d], p[d]);
                        hi[d] = std::max(hi[d], p[d]);
                    }
                const double nmax = double((uint64_t(1) << HC::LMAX) - 1);
                std::vector<uint64_t> hkeys(points.size());
                for (size_t i = 0; i < points.size(); i++) {
                    typename HC::Coord X;
                    for (int d = 0; d < D; d++) {
                        const double w = hi[d] > lo[d] ? (points[i][d] - lo[d]) / (hi[d] - lo[d]) : 0.0;
                        X[d] = typename HC::coord_type(std::min(nmax, std::max(0.0, w * nmax)));
                    }
                    hkeys[i] = HC::key_from_coord(X);
                }
                std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return hkeys[a] < hkeys[b]; });
                return order;
            }

            /// Move vertex v (not a super vertex) to p, inside the bounding box. The result is the Delaunay
            /// triangulation of the moved points (the same as triangulating them from scratch): v is moved in place if
            /// its simplices stay positively oriented and Delaunay, otherwise it is removed and inserted again at p
            /// (keeping its index and key). Returns true if it was moved in place. p must not coincide with another
            /// vertex (std::invalid_argument; the triangulation is then left without v).
            bool move(int v, const Point & p) {
                if (move_in_place(v, p))
                    return true;
                incident_simplices(v, work_star, work_visited, ++work_stamp);
                remove_vertex(v, work_star);
                vertices[v] = p;
                const int s0 = locate(p.data(), last_simplex);
                for (int k = 0; k < NV; k++)
                    if (vertices[simplices[s0].v[k]] == p)
                        throw std::invalid_argument("[Delaunay::move] The point coincides with another vertex");
                insert_vertex(v, s0);
                stats.nmoved_reinserted++;
                return false;
            }

            /// Move vertex v to p if its simplices stay positively oriented and Delaunay (then the triangulation is
            /// that of the moved points); otherwise leave everything unchanged. Returns whether it was moved.
            bool move_in_place(int v, const Point & p) {
                if (v < NV || v >= int(vertices.size()))
                    throw std::invalid_argument("[Delaunay::move] Not a movable vertex");
                for (int d = 0; d < D; d++)
                    if (!(p[d] >= box_lo[d] && p[d] <= box_hi[d]))
                        throw std::invalid_argument("[Delaunay::move] Point outside the bounding box (or not finite)");
                if (vertices[v] == p) {
                    stats.nmoved_in_place++;
                    return true;
                }
                incident_simplices(v, work_star, work_visited, ++work_stamp);
                const Point old = vertices[v];
                vertices[v] = p;
                if (star_is_delaunay(v, work_star)) {
                    stats.nmoved_in_place++;
                    return true;
                }
                vertices[v] = old;
                return false;
            }

            /// Set the position of a vertex without changing the simplices: the triangulation is invalid until
            /// retriangulate() is called
            void set_position(int v, const Point & p) {
                if (v < NV || v >= int(vertices.size()))
                    throw std::invalid_argument("[Delaunay::set_position] Not a movable vertex");
                for (int d = 0; d < D; d++)
                    if (!(p[d] >= box_lo[d] && p[d] <= box_hi[d]))
                        throw std::invalid_argument("[Delaunay] Point outside the bounding box (or not finite)");
                vertices[v] = p;
            }

            /// Triangulate the current vertex positions from scratch (in Peano-Hilbert order), keeping the vertex
            /// indices and keys. Throws std::invalid_argument if two vertices coincide.
            void retriangulate() {
                simplices.clear();
                births.clear();
                next_birth = 0;
                free_list.clear();
                mark.clear();
                stamp = 0;
                Simplex s0;
                for (int k = 0; k < NV; k++) {
                    s0.v[k] = k;
                    s0.n[k] = -1;
                }
                simplices.push_back(s0);
                mark.push_back(0);
                births.push_back(next_birth++);
                if (orient_of(s0.v) < 0)
                    std::swap(simplices[0].v[0], simplices[0].v[1]);
                vertex_simplex.assign(vertices.size(), -1);
                for (int k = 0; k < NV; k++)
                    vertex_simplex[size_t(k)] = 0;
                last_simplex = 0;
                std::vector<Point> pts(vertices.begin() + NV, vertices.end());
                for (size_t i : hilbert_order(pts)) {
                    const int v = int(i) + NV;
                    const int s = locate(vertices[v].data(), last_simplex);
                    for (int k = 0; k < NV; k++)
                        if (vertices[simplices[s].v[k]] == vertices[v])
                            throw std::invalid_argument("[Delaunay::retriangulate] Two vertices coincide");
                    insert_vertex(v, s);
                }
            }

            /// The simplices whose circumspheres contain the point p with key `key` (with the perturbation): the
            /// simplices that inserting it would replace. Adds p as an unconnected vertex (for retriangulating holes)
            std::vector<int> conflict_region(const Point & p, uint64_t key) {
                const int s0 = locate(p.data(), last_simplex);
                for (int k = 0; k < NV; k++)
                    if (vertices[simplices[s0].v[k]] == p)
                        throw std::invalid_argument("[Delaunay::conflict_region] The point is a vertex");
                const int q = add_vertex(p, key);
                find_cavity(q, s0);
                return cavity;
            }

            // Accessors
            int nvertices() const { return int(vertices.size()); }
            static constexpr int nsuper() { return NV; }
            bool is_super(int v) const { return v < NV; }
            const Point & vertex(int v) const { return vertices[v]; }
            uint64_t key(int v) const { return keys[v]; }
            const std::vector<Simplex> & get_simplices() const { return simplices; }
            const Simplex & simplex(int s) const { return simplices[s]; }
            size_t nsimplices_alive() const { return simplices.size() - free_list.size(); }
            /// A live simplex having v as a vertex
            int incident_simplex(int v) const { return vertex_simplex[v]; }
            const Statistics & statistics() const { return stats; }
            /// A number unique to each simplex ever created (slots of deleted simplices are reused, so this tells
            /// whether the simplex in a slot is still the one seen earlier)
            uint64_t birth(int s) const { return births[size_t(s)]; }

            /// The simplices having v as a vertex (by walking across the facets that contain v)
            void incident_simplices(int v, std::vector<int> & out) const {
                out.clear();
                out.push_back(vertex_simplex[v]);
                for (size_t i = 0; i < out.size(); i++) {
                    const Simplex & s = simplices[out[i]];
                    for (int k = 0; k < NV; k++) {
                        if (s.v[k] == v)
                            continue;
                        const int t = s.n[k];
                        if (t >= 0 && std::find(out.begin(), out.end(), t) == out.end())
                            out.push_back(t);
                    }
                }
            }

            /// Faster version for repeated calls: visited is a work array (one per thread) that this resizes to the
            /// number of simplices; stamp must differ between calls (use a counter starting at 1)
            void
            incident_simplices(int v, std::vector<int> & out, std::vector<unsigned> & visited, unsigned stamp) const {
                if (visited.size() < simplices.size())
                    visited.resize(simplices.size(), 0u);
                out.clear();
                out.push_back(vertex_simplex[v]);
                visited[size_t(out[0])] = stamp;
                for (size_t i = 0; i < out.size(); i++) {
                    const Simplex & s = simplices[out[i]];
                    for (int k = 0; k < NV; k++) {
                        if (s.v[k] == v)
                            continue;
                        const int t = s.n[k];
                        if (t >= 0 && visited[size_t(t)] != stamp) {
                            visited[size_t(t)] = stamp;
                            out.push_back(t);
                        }
                    }
                }
            }

            /// Circumcenter and squared circumradius of a simplex (floating point, relative to its first vertex)
            void circumsphere(int s, Point & center, double & radius2) const {
                const Simplex & S = simplices[s];
                const Point & p0 = vertices[S.v[0]];
                double A[D][D], b[D];
                for (int k = 0; k < D; k++) {
                    const Point & pk = vertices[S.v[k + 1]];
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
                    const double c00 = A[1][1] * A[2][2] - A[1][2] * A[2][1];
                    const double c01 = A[1][2] * A[2][0] - A[1][0] * A[2][2];
                    const double c02 = A[1][0] * A[2][1] - A[1][1] * A[2][0];
                    const double det = A[0][0] * c00 + A[0][1] * c01 + A[0][2] * c02;
                    // Cramer's rule via the cross products of the rows
                    const double r12[3] = {A[1][1] * A[2][2] - A[1][2] * A[2][1],
                                           A[1][2] * A[2][0] - A[1][0] * A[2][2],
                                           A[1][0] * A[2][1] - A[1][1] * A[2][0]};
                    const double r20[3] = {A[2][1] * A[0][2] - A[2][2] * A[0][1],
                                           A[2][2] * A[0][0] - A[2][0] * A[0][2],
                                           A[2][0] * A[0][1] - A[2][1] * A[0][0]};
                    const double r01[3] = {A[0][1] * A[1][2] - A[0][2] * A[1][1],
                                           A[0][2] * A[1][0] - A[0][0] * A[1][2],
                                           A[0][0] * A[1][1] - A[0][1] * A[1][0]};
                    for (int d = 0; d < 3; d++)
                        c[d] = (b[0] * r12[d] + b[1] * r20[d] + b[2] * r01[d]) / det;
                }
                radius2 = 0.0;
                for (int d = 0; d < D; d++) {
                    radius2 += c[d] * c[d];
                    center[d] = p0[d] + c[d];
                }
            }

            /// Check the triangulation: positive orientation of every simplex, symmetric neighbours sharing a facet,
            /// and (if delaunay) the empty-circumsphere property across every facet with the perturbed predicate.
            /// Returns an empty string if valid, else a description of the first problem.
            std::string check(bool delaunay = true) const {
                for (size_t s = 0; s < simplices.size(); s++) {
                    const Simplex & S = simplices[s];
                    if (!S.alive())
                        continue;
                    if (orient_of(S.v) <= 0)
                        return "simplex " + std::to_string(s) + " is not positively oriented";
                    for (int k = 0; k < NV; k++) {
                        const int t = S.n[k];
                        if (t < 0)
                            continue;
                        const Simplex & T = simplices[t];
                        if (!T.alive())
                            return "simplex " + std::to_string(s) + " has a deleted neighbour";
                        const int j = index_of_neighbour(T, int(s));
                        if (j < 0)
                            return "asymmetric neighbours " + std::to_string(s) + " " + std::to_string(t);
                        for (int i = 0; i < NV; i++)
                            if (i != k && index_of_vertex(T, S.v[i]) < 0)
                                return "neighbours do not share a facet " + std::to_string(s) + " " + std::to_string(t);
                        if (index_of_vertex(T, S.v[k]) >= 0)
                            return "neighbours share all vertices " + std::to_string(s) + " " + std::to_string(t);
                        if (delaunay && insphere_perturbed(int(s), T.v[j]) > 0)
                            return "facet of simplex " + std::to_string(s) + " is not locally Delaunay";
                    }
                }
                return "";
            }

          private:
            std::vector<Point> vertices{};
            std::vector<uint64_t> keys{};
            std::vector<Simplex> simplices{};
            std::vector<uint64_t> births{};
            uint64_t next_birth{0};
            std::vector<int> free_list{};
            std::vector<int> vertex_simplex{};
            std::vector<int> mark{}; // per simplex: stamp of the last cavity test (+ in cavity, - not in cavity)
            int stamp{0};
            int last_simplex{0};
            Point box_lo{}, box_hi{};
            Statistics stats{};
            uint64_t rng_state{0x9E3779B97F4A7C15ull};

            // Work buffers (kept to avoid reallocations)
            std::vector<int> cavity{};
            struct BoundaryFacet {
                std::array<int, NV> v;
                int k;      // index of the facet (opposite v[k] in the cavity simplex)
                int nb;     // the simplex outside
                int nb_idx; // the index of the facet in nb
            };
            std::vector<BoundaryFacet> boundary{};
            struct Ridge {
                uint64_t key;
                int s;
                int j;
            };
            std::vector<Ridge> ridges{};
            std::vector<int> table{};
            std::vector<int> work_star{};
            std::vector<unsigned> work_visited{};
            unsigned work_stamp{0};

            /// After moving v: are its simplices positively oriented and every facet of them locally Delaunay? (A
            /// valid triangulation that is locally Delaunay everywhere is the Delaunay triangulation; the facets of
            /// the other simplices did not change.)
            bool star_is_delaunay(int v, const std::vector<int> & star) const {
                for (int s : star)
                    if (orient_of(simplices[s].v) <= 0)
                        return false;
                for (int s : star) {
                    const Simplex & S = simplices[s];
                    for (int k = 0; k < NV; k++) {
                        const int t = S.n[k];
                        if (t < 0)
                            continue;
                        // A facet containing v is shared with another simplex of the star: check it once
                        if (S.v[k] != v && t < s)
                            continue;
                        const Simplex & T = simplices[t];
                        if (insphere_perturbed(s, T.v[index_of_neighbour(T, s)]) > 0)
                            return false;
                    }
                }
                return true;
            }

            /// Remove vertex v (whose simplices are `star`): the hole is filled with the simplices of the Delaunay
            /// triangulation of its link vertices that conflict with v (Devillers & Teillaud 2011), which with the
            /// same keys and coordinates is the Delaunay triangulation without v. The link is triangulated inside a
            /// super simplex 1e4 times its size (fast); if that is not enough (the result is checked: positively
            /// oriented and locally Delaunay across every facet of the filled hole) again with 1e45 times (exact for
            /// any input whose degree-5 determinants do not overflow, but slower).
            void remove_vertex(int v, const std::vector<int> & star) {
                std::vector<int> link;
                std::vector<HoleFacet> hole;
                for (int s : star) {
                    const Simplex & S = simplices[s];
                    HoleFacet h;
                    int n = 0;
                    int kv = -1;
                    for (int k = 0; k < NV; k++) {
                        const int w = S.v[k];
                        if (w == v) {
                            kv = k;
                            continue;
                        }
                        h.key[size_t(n++)] = w;
                        if (std::find(link.begin(), link.end(), w) == link.end())
                            link.push_back(w);
                    }
                    std::sort(h.key.begin(), h.key.end());
                    h.outside = S.n[kv];
                    h.outside_idx = h.outside >= 0 ? index_of_neighbour(simplices[h.outside], s) : -1;
                    h.matched = false;
                    hole.push_back(h);
                }
                for (int s : star) {
                    simplices[s].v[0] = -1;
                    free_list.push_back(s);
                }
                std::vector<int> created;
                if (fill_hole(v, link, hole, 1e4, created))
                    return;
                // Undo and retry with a huge super simplex
                for (int c : created) {
                    simplices[c].v[0] = -1;
                    free_list.push_back(c);
                }
                for (auto & h : hole) {
                    h.matched = false;
                    if (h.outside >= 0)
                        simplices[h.outside].n[h.outside_idx] = -1;
                }
                if (!fill_hole(v, link, hole, 1e45, created))
                    throw std::runtime_error("[Delaunay::move] Could not retriangulate the hole of a removed vertex");
            }

            struct HoleFacet {
                std::array<int, D> key;
                int outside;
                int outside_idx;
                bool matched;
            };

            /// Fill the hole of removed vertex v from the triangulation of its link (super simplex `scale` times the
            /// link); false if the result is not valid and Delaunay (the hole is then filled but must be undone)
            bool fill_hole(int v,
                           const std::vector<int> & link,
                           std::vector<HoleFacet> & hole,
                           double scale,
                           std::vector<int> & created) {
                created.clear();
                Point lo = vertices[v], hi = vertices[v];
                for (int w : link)
                    for (int d = 0; d < D; d++) {
                        lo[d] = std::min(lo[d], vertices[w][d]);
                        hi[d] = std::max(hi[d], vertices[w][d]);
                    }
                double size = 0.0;
                for (int d = 0; d < D; d++)
                    size = std::max(size, hi[d] - lo[d]);
                for (int d = 0; d < D; d++) {
                    lo[d] -= 0.1 * size;
                    hi[d] += 0.1 * size;
                }
                // Its super vertices get keys below those of this triangulation's super vertices
                Delaunay<D> small;
                small.init(lo, hi, keys[0] - uint64_t(NV), scale);
                std::vector<int> to_big(size_t(NV), -1);
                for (int w : link) {
                    const int sid = small.insert(vertices[w], keys[w]);
                    if (sid != int(to_big.size()))
                        throw std::runtime_error("[Delaunay::move] Internal error: duplicate link vertex");
                    to_big.push_back(w);
                }
                const std::vector<int> cav = small.conflict_region(vertices[v], keys[v]);
                for (int c : cav)
                    for (int k = 0; k < NV; k++)
                        if (small.simplices[size_t(c)].v[k] < NV)
                            return false; // the super simplex is too close
                std::vector<int> new_id(small.simplices.size(), -1);
                for (int c : cav) {
                    new_id[size_t(c)] = new_simplex();
                    created.push_back(new_id[size_t(c)]);
                }
                bool ok = true;
                for (int c : cav) {
                    const Simplex & C = small.simplices[size_t(c)];
                    const int nid = new_id[size_t(c)];
                    Simplex & S = simplices[nid];
                    for (int k = 0; k < NV; k++)
                        S.v[k] = to_big[size_t(C.v[k])];
                    for (int k = 0; k < NV; k++) {
                        const int nb = C.n[k];
                        if (nb >= 0 && new_id[size_t(nb)] >= 0) {
                            S.n[k] = new_id[size_t(nb)];
                            continue;
                        }
                        std::array<int, D> key;
                        int n = 0;
                        for (int i = 0; i < NV; i++)
                            if (i != k)
                                key[size_t(n++)] = S.v[i];
                        std::sort(key.begin(), key.end());
                        bool found = false;
                        for (auto & h : hole) {
                            if (h.matched || h.key != key)
                                continue;
                            h.matched = true;
                            found = true;
                            S.n[k] = h.outside;
                            if (h.outside >= 0)
                                simplices[h.outside].n[h.outside_idx] = nid;
                            break;
                        }
                        if (!found) {
                            S.n[k] = -1;
                            ok = false;
                        }
                    }
                }
                for (const auto & h : hole)
                    ok = ok && h.matched;
                // Check the new simplices: positive orientation and locally Delaunay across all their facets
                for (int nid : created) {
                    if (!ok)
                        break;
                    const Simplex & S = simplices[nid];
                    if (orient_of(S.v) <= 0) {
                        ok = false;
                        break;
                    }
                    for (int k = 0; k < NV && ok; k++) {
                        const int t = S.n[k];
                        if (t < 0)
                            continue;
                        const Simplex & T = simplices[t];
                        const int j = index_of_neighbour(T, nid);
                        ok = j >= 0 && insphere_perturbed(nid, T.v[j]) < 0;
                    }
                }
                if (!ok)
                    return false;
                for (int nid : created) {
                    for (int k = 0; k < NV; k++)
                        vertex_simplex[simplices[nid].v[k]] = nid;
                    last_simplex = nid;
                }
                vertex_simplex[v] = -1;
                return true;
            }

            int add_vertex(const Point & x, uint64_t key) {
                vertices.push_back(x);
                keys.push_back(key);
                vertex_simplex.push_back(-1);
                return int(vertices.size()) - 1;
            }

            static int index_of_vertex(const Simplex & s, int v) {
                for (int k = 0; k < NV; k++)
                    if (s.v[k] == v)
                        return k;
                return -1;
            }
            static int index_of_neighbour(const Simplex & s, int t) {
                for (int k = 0; k < NV; k++)
                    if (s.n[k] == t)
                        return k;
                return -1;
            }

            int orient_of(const std::array<int, NV> & v) const {
                std::array<const double *, NV> p;
                for (int k = 0; k < NV; k++)
                    p[k] = vertices[v[k]].data();
                return orient<D>(p);
            }

            /// insphere with the symbolic perturbation: never 0 (q not a vertex of s)
            int insphere_perturbed(int s, int q) const {
                const Simplex & S = simplices[s];
                std::array<const double *, NV> p;
                for (int k = 0; k < NV; k++)
                    p[k] = vertices[S.v[k]].data();
                const int r = insphere<D>(p, vertices[q].data());
                if (r != 0)
                    return r;
                // Devillers & Teillaud: the leading nonzero term of the perturbed determinant, in decreasing order of
                // the perturbation of the D+2 points
                std::array<int, NV + 1> ids;
                for (int k = 0; k < NV; k++)
                    ids[k] = S.v[k];
                ids[NV] = q;
                std::sort(ids.begin(), ids.end(), [&](int a, int b) { return keys[a] < keys[b]; });
                for (int i = NV; i >= 0; i--) {
                    if (ids[i] == q)
                        return -1;
                    const int k = index_of_vertex(S, ids[i]);
                    std::array<const double *, NV> pk = p;
                    pk[k] = vertices[q].data();
                    const int o = orient<D>(pk);
                    if (o != 0)
                        return o;
                }
                return -1;
            }

            uint64_t next_random() {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                return rng_state;
            }

            /// Visibility walk (stochastic, so it terminates): a simplex whose closure contains x
            int locate(const double * x, int start) {
                int s = start;
                if (s < 0 || s >= int(simplices.size()) || !simplices[s].alive())
                    s = vertex_simplex[0];
                size_t steps = 0;
                while (true) {
                    const Simplex & S = simplices[s];
                    std::array<const double *, NV> p;
                    for (int k = 0; k < NV; k++)
                        p[k] = vertices[S.v[k]].data();
                    const int k0 = int(next_random() % NV);
                    int next = -1;
                    for (int t = 0; t < NV; t++) {
                        const int k = (k0 + t) % NV;
                        std::array<const double *, NV> pk = p;
                        pk[k] = x;
                        if (orient<D>(pk) < 0) {
                            next = S.n[k];
                            break;
                        }
                    }
                    if (next < 0)
                        break;
                    s = next;
                    steps++;
                }
                stats.nwalk_steps += steps;
                return s;
            }

            int new_simplex() {
                if (!free_list.empty()) {
                    const int s = free_list.back();
                    free_list.pop_back();
                    births[size_t(s)] = next_birth++;
                    return s;
                }
                simplices.emplace_back();
                mark.push_back(0);
                births.push_back(next_birth++);
                return int(simplices.size()) - 1;
            }

            /// The simplices whose circumspheres contain vertex q (with the perturbation), starting from s0 which
            /// contains it, in `cavity`, and the facets of their boundary in `boundary`
            void find_cavity(int q, int s0) {
                if (++stamp == std::numeric_limits<int>::max()) {
                    std::fill(mark.begin(), mark.end(), 0);
                    stamp = 1;
                }
                cavity.clear();
                boundary.clear();
                cavity.push_back(s0);
                mark[s0] = stamp;
                for (size_t i = 0; i < cavity.size(); i++) {
                    const int s = cavity[i];
                    for (int k = 0; k < NV; k++) {
                        const int t = simplices[s].n[k];
                        bool in = false;
                        if (t >= 0) {
                            if (mark[t] == stamp)
                                continue;
                            if (mark[t] != -stamp) {
                                in = insphere_perturbed(t, q) > 0;
                                mark[t] = in ? stamp : -stamp;
                                if (in) {
                                    cavity.push_back(t);
                                    continue;
                                }
                            }
                        }
                        BoundaryFacet f;
                        f.v = simplices[s].v;
                        f.k = k;
                        f.nb = t;
                        f.nb_idx = t >= 0 ? index_of_neighbour(simplices[t], s) : -1;
                        boundary.push_back(f);
                    }
                }
            }

            void insert_vertex(int q, int s0) {
                find_cavity(q, s0);
                stats.ncavity_simplices += cavity.size();
                for (int s : cavity) {
                    simplices[s].v[0] = -1;
                    free_list.push_back(s);
                }
                // New simplices: each boundary facet joined to q (q replaces the vertex opposite the facet, which
                // keeps the positive orientation since q is on the same side of the facet)
                ridges.clear();
                for (const auto & f : boundary) {
                    const int s = new_simplex();
                    Simplex & S = simplices[s];
                    S.v = f.v;
                    S.v[f.k] = q;
                    S.n.fill(-1);
                    S.n[f.k] = f.nb;
                    if (f.nb >= 0)
                        simplices[f.nb].n[f.nb_idx] = s;
                    for (int k = 0; k < NV; k++)
                        vertex_simplex[S.v[k]] = s;
                    // The other facets contain q and are shared with other new simplices: match them by the D-1
                    // vertices other than q
                    for (int j = 0; j < NV; j++) {
                        if (j == f.k)
                            continue;
                        uint64_t key;
                        if constexpr (D == 2) {
                            key = uint64_t(S.v[3 - j - f.k]);
                        } else {
                            int a = -1, b = -1;
                            for (int i = 0; i < NV; i++)
                                if (i != j && i != f.k)
                                    (a < 0 ? a : b) = S.v[i];
                            if (a > b)
                                std::swap(a, b);
                            key = (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
                        }
                        ridges.push_back({key, s, j});
                    }
                    last_simplex = s;
                }
                // Match the ridges in pairs with a small open-addressing hash table
                size_t cap = 16;
                while (cap < 2 * ridges.size())
                    cap *= 2;
                if (table.size() < cap)
                    table.resize(cap);
                std::fill(table.begin(), table.begin() + long(cap), -1);
                size_t matched = 0;
                for (size_t r = 0; r < ridges.size(); r++) {
                    size_t h = size_t((ridges[r].key * 0x9E3779B97F4A7C15ull) >> 32) & (cap - 1);
                    while (true) {
                        const int other = table[h];
                        if (other == -1) {
                            table[h] = int(r);
                            break;
                        }
                        if (other >= 0 && ridges[size_t(other)].key == ridges[r].key) {
                            const Ridge & a = ridges[size_t(other)];
                            const Ridge & b = ridges[r];
                            simplices[a.s].n[a.j] = b.s;
                            simplices[b.s].n[b.j] = a.s;
                            table[h] = -2; // matched (a key occurs exactly twice); probing continues past it
                            matched += 2;
                            break;
                        }
                        h = (h + 1) & (cap - 1);
                    }
                }
                if (matched != ridges.size())
                    throw std::runtime_error("[Delaunay] Internal error: unmatched facets in the cavity");
            }
        };

    } // namespace DELAUNAY
} // namespace FML

#endif
