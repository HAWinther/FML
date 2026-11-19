// MPI tests of PeriodicVoronoi (FML/Experimental/Delaunay): the cells and faces must be bitwise identical for any
// distribution of the points over the tasks. The reference is the run with all points on task 0 (the other tasks
// empty); then the same points distributed in x slabs, x-y pencils, Peano-Hilbert ranges, at random, and all on the
// last task. Every task's cells and faces are gathered on task 0 and compared with the reference: every cell
// identical, every face present on the task(s) of its two points (twice if they are on different tasks) and
// identical there. Random, clustered and lattice (degenerate) points in 2D and 3D.
#include <FML/Experimental/Delaunay/PeriodicVoronoi.h>
#include <FML/tests/TestHelpers.h>

#include <map>
#include <random>
#include <tuple>

using FML::TESTS::check;
using namespace FML::DELAUNAY;

namespace {

    template <int D>
    struct CellRecord {
        size_t id;
        double volume;
        double centroid[D];
    };
    template <int D>
    struct FaceRecord {
        size_t i, j, index_i, index_j;
        int shift[D];
        int task_i, task_j, holder;
        double area, distance;
        double centroid[D], normal[D];
    };

    /// All the records of all the tasks on task 0 (empty elsewhere)
    template <class T>
    std::vector<T> gather_on_root(const std::vector<T> & local) {
#ifdef USE_MPI
        int n = int(local.size() * sizeof(T));
        std::vector<int> counts(size_t(FML::NTasks)), displs(size_t(FML::NTasks), 0);
        MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, FML::communicator);
        size_t total = 0;
        for (int t = 0; t < FML::NTasks; t++) {
            displs[size_t(t)] = int(total);
            total += size_t(counts[size_t(t)]);
        }
        std::vector<T> all(FML::ThisTask == 0 ? total / sizeof(T) : 0);
        MPI_Gatherv(local.data(),
                    n,
                    MPI_BYTE,
                    all.data(),
                    counts.data(),
                    displs.data(),
                    MPI_BYTE,
                    0,
                    FML::communicator);
        return all;
#else
        return local;
#endif
    }

    template <int D>
    std::vector<std::array<double, D>> make_points(const std::string & type, size_t n, double L, unsigned seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> U(0, L);
        std::normal_distribution<double> G(0, 0.02 * L);
        std::vector<std::array<double, D>> p;
        if (type == "lattice") {
            const int m = int(std::round(std::pow(double(n), 1.0 / D)));
            std::array<int, D> c{};
            while (true) {
                std::array<double, D> x;
                for (int d = 0; d < D; d++)
                    x[d] = (c[d] + 0.25) * L / m;
                p.push_back(x);
                int d = D - 1;
                while (d >= 0 && c[d] == m - 1)
                    c[d--] = 0;
                if (d < 0)
                    break;
                c[d]++;
            }
            return p;
        }
        p.resize(n);
        for (size_t i = 0; i < n; i++)
            for (int d = 0; d < D; d++) {
                double x = (type == "clustered" && i % 2 == 0) ? 0.97 * L + G(rng) : U(rng);
                p[i][d] = x - L * std::floor(x / L);
            }
        return p;
    }

    /// The task of each point for a distribution
    template <int D>
    std::vector<int> distribute(const std::string & how, const std::vector<std::array<double, D>> & p, double L) {
        const int P = FML::NTasks;
        std::vector<int> owner(p.size(), 0);
        if (how == "task0")
            return owner;
        if (how == "last") {
            std::fill(owner.begin(), owner.end(), P - 1);
            return owner;
        }
        if (how == "slab") {
            for (size_t i = 0; i < p.size(); i++)
                owner[i] = std::min(P - 1, int(p[i][0] / L * P));
        } else if (how == "pencil") {
            int px = 1;
            for (int k = 1; k * k <= P; k++)
                if (P % k == 0)
                    px = k;
            const int py = P / px;
            for (size_t i = 0; i < p.size(); i++)
                owner[i] = std::min(px - 1, int(p[i][0] / L * px)) * py + std::min(py - 1, int(p[i][1] / L * py));
        } else if (how == "hilbert") {
            std::vector<std::array<double, D>> u = p;
            const auto order = Delaunay<D>::hilbert_order(u);
            for (size_t k = 0; k < order.size(); k++)
                owner[order[k]] = int(k * size_t(P) / order.size());
        } else if (how == "random") {
            std::mt19937_64 rng(77);
            for (auto & o : owner)
                o = int(rng() % uint64_t(P));
        }
        return owner;
    }

    template <int D>
    void collect(const PeriodicVoronoi<D> & v, std::vector<CellRecord<D>> & cells, std::vector<FaceRecord<D>> & faces) {
        cells.clear();
        faces.clear();
        for (const auto & c : v.cells()) {
            CellRecord<D> r;
            r.id = c.id;
            r.volume = c.volume;
            for (int d = 0; d < D; d++)
                r.centroid[d] = c.centroid[d];
            cells.push_back(r);
        }
        for (const auto & f : v.faces()) {
            FaceRecord<D> r;
            r.i = f.i;
            r.j = f.j;
            r.index_i = f.index_i;
            r.index_j = f.index_j;
            r.task_i = f.task_i;
            r.task_j = f.task_j;
            r.holder = FML::ThisTask;
            r.area = f.area;
            r.distance = f.distance;
            for (int d = 0; d < D; d++) {
                r.shift[d] = f.shift[d];
                r.centroid[d] = f.centroid[d];
                r.normal[d] = f.normal[d];
            }
            faces.push_back(r);
        }
    }

    template <int D>
    bool same_geometry(const FaceRecord<D> & a, const FaceRecord<D> & b) {
        bool same = a.area == b.area && a.distance == b.distance;
        for (int d = 0; d < D; d++)
            same = same && a.centroid[d] == b.centroid[d] && a.normal[d] == b.normal[d];
        return same;
    }

    template <int D>
    using FaceKey = std::tuple<size_t, size_t, std::array<int, D>>;
    template <int D>
    FaceKey<D> key_of(const FaceRecord<D> & f) {
        std::array<int, D> s;
        for (int d = 0; d < D; d++)
            s[d] = f.shift[d];
        return FaceKey<D>{f.i, f.j, s};
    }

    /// The reference: all the points on task 0 (collective; the maps are filled on every task)
    template <int D>
    void reference(const std::vector<std::array<double, D>> & p,
                   double L,
                   std::map<size_t, CellRecord<D>> & ref_cells,
                   std::map<FaceKey<D>, FaceRecord<D>> & ref_faces) {
        std::vector<std::array<double, D>> local;
        std::vector<size_t> ids;
        if (FML::ThisTask == 0) {
            local = p;
            ids.resize(p.size());
            std::iota(ids.begin(), ids.end(), size_t(0));
        }
        PeriodicVoronoi<D> v(local, L, {}, &ids);
        std::vector<CellRecord<D>> c;
        std::vector<FaceRecord<D>> f;
        collect<D>(v, c, f);
        ref_cells.clear();
        ref_faces.clear();
        for (const auto & r : c)
            ref_cells[r.id] = r;
        for (const auto & r : f)
            ref_faces[key_of<D>(r)] = r;
    }

    /// Gather every task's cells and faces on task 0 and compare with the reference (collective; true on all
    /// tasks if identical)
    template <int D>
    bool compare(const PeriodicVoronoi<D> & v,
                 size_t npoints,
                 const std::vector<int> & owner,
                 const std::map<size_t, CellRecord<D>> & ref_cells,
                 const std::map<FaceKey<D>, FaceRecord<D>> & ref_faces,
                 size_t & nfaces,
                 size_t & nshared) {
        std::vector<CellRecord<D>> c;
        std::vector<FaceRecord<D>> f;
        collect<D>(v, c, f);
        const auto all_cells = gather_on_root(c);
        const auto all_faces = gather_on_root(f);
        int ok = 1;
        size_t ncells = 0;
        nfaces = nshared = 0;
        if (FML::ThisTask == 0) {
            // Cells: each once, identical
            std::vector<char> seen(npoints, 0);
            for (const auto & r : all_cells) {
                const auto it = ref_cells.find(r.id);
                bool same = it != ref_cells.end() && r.id < seen.size() && !seen[r.id] && r.volume == it->second.volume;
                for (int d = 0; d < D && same; d++)
                    same = r.centroid[d] == it->second.centroid[d];
                ok = ok && same;
                if (r.id < seen.size())
                    seen[r.id] = 1;
                ncells++;
            }
            ok = ok && ncells == npoints;
            // Faces: identical, held by the task(s) of their points, once per such task
            std::map<FaceKey<D>, std::vector<int>> holders;
            for (const auto & r : all_faces) {
                const auto it = ref_faces.find(key_of<D>(r));
                ok = ok && it != ref_faces.end() && same_geometry<D>(r, it->second) &&
                     (r.holder == owner[r.i] || r.holder == owner[r.j]) && r.task_i == owner[r.i] &&
                     r.task_j == owner[r.j];
                holders[key_of<D>(r)].push_back(r.holder);
            }
            for (const auto & kv : ref_faces) {
                const auto it = holders.find(kv.first);
                const int ti = owner[std::get<0>(kv.first)], tj = owner[std::get<1>(kv.first)];
                const size_t expected = ti == tj ? 1 : 2;
                ok = ok && it != holders.end() && it->second.size() == expected;
                nfaces++;
                nshared += ti != tj;
            }
        }
        FML::MaxOverTasks(&ok);
        return ok;
    }

    template <int D>
    void test_distributions(const std::string & type, size_t n) {
        const double L = 50.0;
        const auto p = make_points<D>(type, n, L, 3);
        std::map<size_t, CellRecord<D>> ref_cells;
        std::map<FaceKey<D>, FaceRecord<D>> ref_faces;
        reference<D>(p, L, ref_cells, ref_faces);
        for (std::string how : {"slab", "pencil", "hilbert", "random", "last"}) {
            const auto owner = distribute<D>(how, p, L);
            std::vector<std::array<double, D>> local;
            std::vector<size_t> ids;
            for (size_t i = 0; i < p.size(); i++)
                if (owner[i] == FML::ThisTask) {
                    local.push_back(p[i]);
                    ids.push_back(i);
                }
            PeriodicVoronoi<D> v(local, L, {}, &ids);
            size_t nfaces, nshared;
            const bool ok = compare<D>(v, p.size(), owner, ref_cells, ref_faces, nfaces, nshared);
            size_t ghosts = v.statistics().nghosts, nq = v.statistics().nqueries;
            FML::SumOverTasks(&ghosts);
            FML::SumOverTasks(&nq);
            check(ok,
                  std::to_string(D) + "D " + type + " (N = " + std::to_string(p.size()) + "), " + how + " on " +
                      std::to_string(FML::NTasks) + " tasks: cells and faces bitwise identical to all-on-task-0 (" +
                      std::to_string(nfaces) + " faces, " + std::to_string(nshared) + " shared between tasks; " +
                      std::to_string(v.statistics().nrounds) + " rounds, " + std::to_string(ghosts) + " ghosts, " +
                      std::to_string(nq) + " searches)");
        }
    }

    /// update(): the points move (and cross task and box boundaries but stay on their tasks); after each update the
    /// cells and faces are bitwise identical to a build of the new positions with all points on task 0
    template <int D>
    void test_update(const std::string & type, size_t n) {
        using PV = PeriodicVoronoi<D>;
        using UM = typename PV::Options::UpdateMethod;
        const double L = 50.0;
        const auto p0 = make_points<D>(type, n, L, 3);
        const double h = L * std::pow(double(p0.size()), -1.0 / D);
        for (std::string how : {"slab", "hilbert", "random"}) {
            for (UM method : {UM::Automatic, UM::Incremental}) {
                auto p = p0;
                const auto owner = distribute<D>(how, p, L);
                std::vector<size_t> ids;
                for (size_t i = 0; i < p.size(); i++)
                    if (owner[i] == FML::ThisTask)
                        ids.push_back(i);
                auto local_points = [&]() {
                    std::vector<std::array<double, D>> local;
                    for (size_t i : ids)
                        local.push_back(p[i]);
                    return local;
                };
                typename PV::Options opt;
                opt.update_method = method;
                PV v(local_points(), L, opt, &ids);
                std::mt19937_64 rng(99);
                std::normal_distribution<double> G(0.0, 0.3 * h);
                bool ok = true;
                size_t nfaces = 0, nshared = 0;
                for (int step = 0; step < 3; step++) {
                    for (auto & x : p)
                        for (int d = 0; d < D; d++) {
                            x[d] += G(rng) + (step == 1 ? 0.0 : 0.7 * h);
                            x[d] -= L * std::floor(x[d] / L);
                        }
                    v.update(local_points());
                    std::map<size_t, CellRecord<D>> ref_cells;
                    std::map<FaceKey<D>, FaceRecord<D>> ref_faces;
                    reference<D>(p, L, ref_cells, ref_faces);
                    ok = compare<D>(v, p.size(), owner, ref_cells, ref_faces, nfaces, nshared) && ok;
                }
                size_t inplace = v.statistics().nmoved_in_place, reins = v.statistics().nreinserted;
                int retri = v.statistics().nretriangulated;
                FML::SumOverTasks(&inplace);
                FML::SumOverTasks(&reins);
                FML::SumOverTasks(&retri);
                check(ok,
                      std::to_string(D) + "D " + type + " (N = " + std::to_string(p.size()) + "), " + how + " on " +
                          std::to_string(FML::NTasks) + " tasks, " +
                          (method == UM::Automatic ? "automatic" : "incremental") +
                          ": 3 updates bitwise identical to builds on task 0 (" + std::to_string(inplace) +
                          " in place, " + std::to_string(reins) + " reinserted, " + std::to_string(retri) +
                          " retriangulations)");
            }
        }
    }

    /// cover(): every task makes the triangulation exact over a box (its x-slab with margins, also outside the unit
    /// box) for points spread over the tasks; then (and after moving the points with update()) the
    /// simplex located at random points of the box is a true Delaunay simplex: no super vertex and no point (with
    /// its periodic images) strictly inside its circumsphere (exact test, brute force over all the points)
    template <int D>
    void test_cover(const std::string & type, const std::string & how, size_t n) {
        const double L = 1.0;
        auto p = make_points<D>(type, n, L, 31);
        const double q = std::ldexp(1.0, 40); // on the quantization grid, so the vertices are the points
        for (auto & x : p)
            for (auto & xi : x) {
                xi = std::round(xi * q) / q;
                if (xi >= L)
                    xi -= L;
            }
        const auto owner = distribute<D>(how, p, L);
        std::vector<std::array<double, D>> mine;
        std::vector<size_t> ids, mine_index;
        for (size_t i = 0; i < p.size(); i++)
            if (owner[i] == FML::ThisTask) {
                mine.push_back(p[i]);
                ids.push_back(i);
                mine_index.push_back(i);
            }
        PeriodicVoronoi<D> V(mine, L, typename PeriodicVoronoi<D>::Options{}, &ids);
        std::array<double, D> lo, hi;
        for (int d = 0; d < D; d++) {
            lo[d] = -0.05;
            hi[d] = 1.03;
        }
        lo[0] = double(FML::ThisTask) / FML::NTasks - 0.07;
        hi[0] = double(FML::ThisTask + 1) / FML::NTasks + 0.04;
        auto verify = [&](const std::vector<std::array<double, D>> & pts) {
            V.cover(lo, hi);
            const auto & dt = V.triangulation();
            std::mt19937_64 rng(1000 + FML::ThisTask);
            std::uniform_real_distribution<double> U(0.0, 1.0);
            uint64_t state = 1;
            int hint = -1;
            long long nbad = 0;
            for (int s = 0; s < 150; s++) {
                std::array<double, D> x;
                for (int d = 0; d < D; d++)
                    x[d] = lo[d] + (hi[d] - lo[d]) * U(rng);
                const int t = dt.locate(x, hint, state);
                hint = t;
                const auto & S = dt.simplex(t);
                std::array<const double *, D + 1> v;
                bool super = false;
                for (int k = 0; k <= D; k++) {
                    super = super || dt.is_super(S.v[k]);
                    v[k] = dt.vertex(S.v[k]).data();
                }
                if (super) {
                    nbad++;
                    continue;
                }
                const int o = orient<D>(v);
                for (const auto & y : pts) {
                    std::array<int, D> im{};
                    for (int d = 0; d < D; d++)
                        im[d] = -1;
                    while (true) {
                        std::array<double, D> z;
                        for (int d = 0; d < D; d++)
                            z[d] = y[d] + im[d] * L;
                        if (o * insphere<D>(v, z.data()) > 0)
                            nbad++;
                        int d = D - 1;
                        while (d >= 0 && im[d] == 1)
                            im[d--] = -1;
                        if (d < 0)
                            break;
                        im[d]++;
                    }
                }
            }
            FML::SumOverTasks(&nbad);
            return nbad;
        };
        const long long nbad = verify(p);
        // Move the points a little (update keeps the box)
        std::mt19937_64 rng(5);
        std::normal_distribution<double> G(0.0, 0.002);
        auto moved = p;
        for (auto & x : moved)
            for (auto & xi : x) {
                xi = std::round((xi + G(rng)) * q) / q;
                xi -= L * std::floor(xi / L);
                if (xi >= L)
                    xi -= L;
            }
        std::vector<std::array<double, D>> mine_moved;
        for (size_t i : mine_index)
            mine_moved.push_back(moved[i]);
        V.update(mine_moved);
        const long long nbad_moved = verify(moved);
        check(nbad == 0 && nbad_moved == 0,
              std::to_string(D) + "D " + type + ", " + how +
                  ": cover(): the simplices located in the box are Delaunay simplices "
                  "(also after update)");
    }

} // namespace

int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    test_distributions<2>("random", 20000);
    test_distributions<2>("clustered", 20000);
    test_distributions<2>("lattice", 4900);
    test_distributions<3>("random", 20000);
    test_distributions<3>("clustered", 20000);
    test_distributions<3>("lattice", 4096);
    test_update<2>("random", 6000);
    test_update<2>("lattice", 4900);
    test_update<3>("random", 6000);
    test_update<3>("clustered", 6000);
    test_cover<2>("random", "slab", 4000);
    test_cover<2>("clustered", "random", 4000);
    test_cover<3>("random", "slab", 3000);
    test_cover<3>("clustered", "hilbert", 3000);
    return FML::TESTS::summary("Delaunay MPI");
}
