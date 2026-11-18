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
            };

            Delaunay() = default;

            /// Create the super simplex enclosing the box [lo, hi] (with a wide margin). Clears the triangulation.
            Delaunay(const Point & lo, const Point & hi) { init(lo, hi); }

            void init(const Point & lo, const Point & hi) {
                vertices.clear();
                keys.clear();
                simplices.clear();
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
                const double a = *std::min_element(lo.begin(), lo.end()) - 10.0 * size;
                const double S = D * 30.0 * size;
                for (int k = 0; k < NV; k++) {
                    Point p;
                    p.fill(a);
                    if (k > 0)
                        p[k - 1] = a + S;
                    add_vertex(p, std::numeric_limits<uint64_t>::max() - uint64_t(NV - k));
                }
                Simplex s;
                for (int k = 0; k < NV; k++) {
                    s.v[k] = k;
                    s.n[k] = -1;
                }
                simplices.push_back(s);
                mark.push_back(0);
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
                    return s;
                }
                simplices.emplace_back();
                mark.push_back(0);
                return int(simplices.size()) - 1;
            }

            void insert_vertex(int q, int s0) {
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
