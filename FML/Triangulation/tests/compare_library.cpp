// The native PeriodicVoronoi as a replacement for the library's CGAL tessellation (Triangulation/PeriodicDelaunay.h):
// on the same particles (on the slabs of MPIParticles, any number of tasks) compare what the library's users take from
// MPIPeriodicDelaunay:
//   - the Voronoi volume of every particle (VoronoiVolume: ZOBOV, watershed binning, ORIGAMI)
//   - the neighbour graph (for_each_adjacent_vertex: watershed, ORIGAMI), with and without the zero-area faces
//   - (3D) the DTFE density at the particles, (D+1) m / volume of the incident simplices (DTFE::create)
// and the timings. Build with make compare_library [CGAL2D=true] CGAL_INCLUDE=... ; run
// mpirun -np N ./compare_library [npart]
#include <FML/Triangulation/PeriodicVoronoi.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/Triangulation/PeriodicDelaunay.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace FML::TRIANGULATION;
constexpr int ND = CGAL_NDIM;

struct Part {
    double pos[ND];
    long long id;
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return ND; }
};

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void wrap(double & x) {
    x -= std::floor(x);
    if (x >= 1.0)
        x = 0.0;
}

// The same particles on every task
static std::vector<Part> make_particles(const std::string & type, size_t n) {
    std::mt19937 gen(4242);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<Part> part;
    auto add = [&](const std::array<double, ND> & x) {
        Part p{};
        for (int d = 0; d < ND; d++) {
            p.pos[d] = x[d];
            wrap(p.pos[d]);
        }
        p.id = (long long)part.size();
        part.push_back(p);
    };
    if (type == "random+clumps") {
        for (size_t i = 0; i < n; i++) {
            std::array<double, ND> x;
            for (auto & v : x)
                v = U(gen);
            add(x);
        }
        // Clumps, also on the slab boundaries and box corners
        for (double c0 : {0.0, 0.25, 0.5})
            for (double c1 : {0.0, 0.5}) {
                std::normal_distribution<double> G(0.0, 0.02);
                for (size_t i = 0; i < n / 40; i++) {
                    std::array<double, ND> x;
                    x[0] = c0 + G(gen);
                    x[1] = c1 + G(gen);
                    if (ND == 3)
                        x[ND - 1] = 0.5 + G(gen);
                    add(x);
                }
            }
    } else if (type == "clustered") {
        // Nested clusters (a crude Soneira-Peebles): strongly inhomogeneous, many close pairs
        std::vector<std::array<double, ND>> centres(1);
        for (auto & v : centres[0])
            v = 0.5;
        double r = 0.25;
        for (int level = 0; level < 5; level++, r /= 3.5) {
            std::vector<std::array<double, ND>> next;
            for (auto & c : centres)
                for (int k = 0; k < 6; k++) {
                    std::array<double, ND> x;
                    for (int d = 0; d < ND; d++)
                        x[d] = c[d] + r * (2 * U(gen) - 1);
                    next.push_back(x);
                }
            centres = next;
        }
        std::normal_distribution<double> G(0.0, 1e-3);
        for (size_t i = 0; i < n; i++) {
            auto x = centres[size_t(U(gen) * centres.size()) % centres.size()];
            for (auto & v : x)
                v += G(gen);
            add(x);
        }
        for (size_t i = 0; i < n / 10; i++) { // a background so that there are no huge empty cells
            std::array<double, ND> x;
            for (auto & v : x)
                v = U(gen);
            add(x);
        }
    } else if (type == "lattice" || type == "lattice+jitter") {
        const int m = int(std::round(std::pow(double(n), 1.0 / ND)));
        std::normal_distribution<double> G(0.0, type == "lattice" ? 0.0 : 1e-6);
        const size_t total = size_t(std::pow(m, ND));
        for (size_t i = 0; i < total; i++) {
            std::array<double, ND> x;
            size_t k = i;
            for (int d = 0; d < ND; d++) {
                x[d] = (double(k % m) + 0.5) / m + G(gen);
                k /= m;
            }
            add(x);
        }
    }
    return part;
}

template <class T>
static std::vector<T> gather_all(const std::vector<T> & local) {
    std::vector<T> all;
#ifdef USE_MPI
    int n = int(local.size() * sizeof(T));
    std::vector<int> counts(FML::NTasks), displ(FML::NTasks, 0);
    MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
    for (int i = 1; i < FML::NTasks; i++)
        displ[i] = displ[i - 1] + counts[i - 1];
    all.resize((displ.back() + counts.back()) / sizeof(T));
    MPI_Allgatherv(local.data(), n, MPI_BYTE, all.data(), counts.data(), displ.data(), MPI_BYTE, FML::communicator);
#else
    all = local;
#endif
    return all;
}

struct Result {
    std::map<long long, double> volume;
    std::map<long long, double> dtfe;
    std::set<std::pair<long long, long long>> edges;
    double time{0.0};
    long long nbad{0};
};

using Edge = std::pair<long long, long long>;
static Edge edge(long long a, long long b) { return a < b ? Edge{a, b} : Edge{b, a}; }

static Result run_cgal(FML::PARTICLE::MPIParticles<Part> & mp, double buffer_fraction) {
    Result r;
    MPIPeriodicDelaunay<Part, VertexDataWatershed> D;
    FML::Barrier();
    const double t0 = now();
    D.create(mp.get_particles_ptr(), mp.get_npart(), buffer_fraction, 0.3,
             [](VertexDataWatershed * v, Part * p) { v->part_ptr = p; });
    std::vector<double> volumes;
    D.VoronoiVolume(volumes);
    FML::Barrier();
    r.time = now() - t0;
    r.nbad = D.num_bad_cells();
    FML::SumOverTasks(&r.nbad);

    const auto & vs = D.get_vertex_handles_regular();
    std::vector<std::pair<long long, double>> vol;
    std::vector<Edge> edges;
    std::vector<typename MPIPeriodicDelaunay<Part, VertexDataWatershed>::Vertex_handle> work;
    for (size_t i = 0; i < mp.get_npart(); i++) {
        const long long id = mp.get_particles_ptr()[i].id;
        vol.push_back({id, volumes[i]});
        D.for_each_adjacent_vertex(vs[i], work, [&](auto w) {
            auto * p = static_cast<Part *>(w->info().part_ptr);
            if (p)
                edges.push_back(edge(id, p->id));
            else
                edges.push_back(edge(id, -1)); // a guard point
        });
    }
#if CGAL_NDIM == 3
    // DTFE::create: 4 m / (sum of the incident tetrahedra) / total mass
    std::vector<std::pair<long long, double>> rho;
    const auto & dt = D.get_delaunay_triangulation();
    const double ntot = double(mp.get_npart_total());
    for (size_t i = 0; i < mp.get_npart(); i++) {
        std::vector<typename std::remove_reference_t<decltype(dt)>::Cell_handle> cells;
        dt.incident_cells(vs[i], std::back_inserter(cells));
        double volume = 0.0;
        for (auto & c : cells) {
            const auto ptet = dt.periodic_tetrahedron(c);
            std::array<std::array<double, 3>, 4> q;
            for (int k = 0; k < 4; k++) {
                auto pt = dt.point(ptet[k]);
                q[k] = {pt.x(), pt.y(), pt.z()};
            }
            double a[3], b[3], cc[3];
            for (int d = 0; d < 3; d++) {
                a[d] = q[1][d] - q[0][d];
                b[d] = q[2][d] - q[0][d];
                cc[d] = q[3][d] - q[0][d];
            }
            volume += std::abs(a[0] * (b[1] * cc[2] - b[2] * cc[1]) - a[1] * (b[0] * cc[2] - b[2] * cc[0]) +
                               a[2] * (b[0] * cc[1] - b[1] * cc[0])) /
                      6.0;
        }
        rho.push_back({mp.get_particles_ptr()[i].id, 4.0 / volume / ntot});
    }
    for (auto & [id, v] : gather_all(rho))
        r.dtfe[id] = v;
#endif
    for (auto & [id, v] : gather_all(vol))
        r.volume[id] = v;
    for (auto & e : gather_all(edges))
        r.edges.insert(e);
    return r;
}

static Result run_native(FML::PARTICLE::MPIParticles<Part> & mp, bool drop_zero, double & t_tri, double & t_geo) {
    using PV = FML::DELAUNAY::PeriodicVoronoi<ND>;
    Result r;
    std::vector<typename PV::Point> pts(mp.get_npart());
    std::vector<size_t> ids(mp.get_npart());
    for (size_t i = 0; i < mp.get_npart(); i++) {
        for (int d = 0; d < ND; d++)
            pts[i][d] = mp.get_particles_ptr()[i].pos[d];
        ids[i] = size_t(mp.get_particles_ptr()[i].id);
    }
    typename PV::Options opt;
    opt.drop_zero_area_faces = drop_zero;
    if (const char * q = std::getenv("QBITS"))
        opt.quantization_bits = std::atoi(q);
    FML::Barrier();
    const double t0 = now();
    PV V(pts, 1.0, opt, &ids);
    FML::Barrier();
    r.time = now() - t0;
    t_tri = V.statistics().time_triangulation;
    t_geo = V.statistics().time_geometry;

    std::vector<std::pair<long long, double>> vol;
    for (auto & c : V.cells())
        vol.push_back({(long long)c.id, c.volume});
    std::vector<Edge> edges;
    for (auto & f : V.faces())
        edges.push_back(edge((long long)f.i, (long long)f.j));

    if constexpr (ND == 3) {
        // DTFE density from the native triangulation: the vertex of local cell c has key id * 125 + code(0)
        const auto & dt = V.triangulation();
        constexpr uint64_t NCODE = 125, CODE0 = 62;
        std::map<size_t, int> vertex_of;
        for (int v = dt.nsuper(); v < dt.nvertices(); v++)
            if (dt.key(v) % NCODE == CODE0)
                vertex_of[size_t(dt.key(v) / NCODE)] = v;
        const double ntot = double(mp.get_npart_total());
        std::vector<int> star;
        std::vector<std::pair<long long, double>> rho;
        for (auto & c : V.cells()) {
            const int v = vertex_of.at(c.id);
            dt.incident_simplices(v, star);
            double volume = 0.0;
            for (int s : star) {
                const auto & S = dt.simplex(s);
                const auto & q0 = dt.vertex(S.v[0]);
                double m[3][3];
                for (int k = 0; k < 3; k++)
                    for (int d = 0; d < 3; d++)
                        m[k][d] = dt.vertex(S.v[k + 1])[d] - q0[d];
                volume += std::abs(m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                                   m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                                   m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])) /
                          6.0;
            }
            rho.push_back({(long long)c.id, 4.0 / volume / ntot});
        }
        for (auto & [id, x] : gather_all(rho))
            r.dtfe[id] = x;
    }
    for (auto & [id, v] : gather_all(vol))
        r.volume[id] = v;
    for (auto & e : gather_all(edges))
        r.edges.insert(e);
    return r;
}

static double max_rel(const std::map<long long, double> & a, const std::map<long long, double> & b, size_t & nmiss) {
    double m = 0.0;
    nmiss = 0;
    for (auto & [id, v] : a) {
        auto it = b.find(id);
        if (it == b.end()) {
            nmiss++;
            continue;
        }
        m = std::max(m, std::abs(v - it->second) / std::abs(it->second));
    }
    nmiss += b.size() > a.size() ? b.size() - a.size() : 0;
    return m;
}

int main(int argc, char ** argv) {
    const size_t n = argc > 1 ? std::stoul(argv[1]) : (ND == 3 ? 50000 : 40000);
    int nfail = 0;
    for (std::string type : {"random+clumps", "clustered", "lattice+jitter", "lattice"}) {
        const auto all = make_particles(type, n);
        std::vector<Part> mine;
        for (size_t i = FML::ThisTask; i < all.size(); i += FML::NTasks)
            mine.push_back(all[i]);
        FML::PARTICLE::MPIParticles<Part> mp;
        mp.create(mine, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed);

        const double buffer = FML::NTasks >= 4 ? 0.9 : 0.5;
        const Result C = run_cgal(mp, buffer);
        double tt, tg, tt2, tg2;
        const Result N = run_native(mp, false, tt, tg);
        const Result Nd = run_native(mp, true, tt2, tg2);

        if (FML::ThisTask != 0)
            continue;
        size_t miss_v, miss_r = 0;
        const double dv = max_rel(N.volume, C.volume, miss_v);
        const double dr = ND == 3 ? max_rel(N.dtfe, C.dtfe, miss_r) : 0.0;
        double vtot = 0.0;
        for (auto & [id, v] : N.volume)
            vtot += v;
        // Edges: CGAL also links to guard points (id -1) if a cell is unreliable
        size_t only_c = 0, only_n = 0, guard = 0;
        for (auto & e : C.edges) {
            if (e.first < 0)
                guard++;
            else if (!N.edges.count(e))
                only_c++;
        }
        for (auto & e : N.edges)
            if (!C.edges.count(e))
                only_n++;
        size_t dropped = 0;
        for (auto & e : N.edges)
            if (!Nd.edges.count(e))
                dropped++;
        const bool degenerate = type == "lattice";
        // Quantization moves the points by up to 2^-41 of the box, so the volume of a cell of size h changes by about
        // 2^-41 / h (1e-7 for the smallest cells here; 1e-9 with QBITS=48)
        const double vtol = 1e-6;
        const bool ok = miss_v == 0 && dv < vtol && vtot > 1 - 1e-10 && vtot < 1 + 1e-10 && C.nbad == 0 &&
                        (degenerate || (only_c == 0 && only_n == 0 && dr < 1e-8));
        nfail += ok ? 0 : 1;
        std::printf("== %dD %-15s N=%zu tasks=%d: %s\n", ND, type.c_str(), C.volume.size(), FML::NTasks,
                    ok ? "OK" : "DIFFERENT");
        std::printf("   volumes: max rel diff %.2e (missing %zu), sum %.15f; CGAL bad cells %lld\n", dv, miss_v, vtot,
                    C.nbad);
        std::printf("   edges: CGAL %zu, native %zu; only CGAL %zu, only native %zu, to guard points %zu; "
                    "zero-area faces dropped by default %zu\n",
                    C.edges.size(), N.edges.size(), only_c, only_n, guard, dropped);
        if (ND == 3)
            std::printf("   DTFE vertex density: max rel diff %.2e (missing %zu)%s\n", dr, miss_r,
                        degenerate ? " (degenerate: the triangulation is not unique)" : "");
        std::printf("   time: CGAL create+VoronoiVolume %.3f s, native %.3f s (triangulation %.3f, geometry %.3f)\n",
                    C.time, N.time, tt, tg);
    }
    if (FML::ThisTask == 0)
        std::printf("%s\n", nfail ? "SOME DIFFERENCES" : "ALL OK");
    return nfail ? 1 : 0;
}
