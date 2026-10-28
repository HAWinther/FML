#include <FML/Morphology/VoxelMorphology.h>
#include <iostream>
#include <set>
using namespace FML::MORPHOLOGY;
using Cell = std::array<int, 3>;
int failures = 0;
void check(bool ok, const std::string& label) {
    int bad = !ok;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << label << '\n';
}
bool near(double a, double b) { return std::abs(a - b) < 1e-10 * std::max({1.0, std::abs(a), std::abs(b)}); }
BinarySlab slab(int n, const std::set<Cell>& cells) {
    BinarySlab s;
    s.nmesh = n;
    s.x_start = n * FML::ThisTask / FML::NTasks;
    s.local_nx = n * (FML::ThisTask + 1) / FML::NTasks - s.x_start;
    s.cells.resize(s.local_nx * n * n);
    for (auto c : cells)
        if (c[0] >= int(s.x_start) && c[0] < int(s.x_start + s.local_nx))
            s.cells[((c[0] - s.x_start) * n + c[1]) * n + c[2]] = 1;
    return s;
}
CubicalCounts reference(int n, const std::set<Cell>& cells, Boundary b) {
    std::set<Cell> vertices;
    std::set<std::array<int, 4>> edges, faces;
    auto wrap = [&](Cell c) {
        if (b == Boundary::Periodic)
            for (int& x : c)
                x %= n;
        return c;
    };
    auto entity = [&](int axis, Cell c) {
        c = wrap(c);
        return std::array<int, 4>{axis, c[0], c[1], c[2]};
    };
    for (auto c : cells) {
        for (int i = 0; i < 8; ++i)
            vertices.insert(wrap({c[0] + (i & 1), c[1] + ((i >> 1) & 1), c[2] + ((i >> 2) & 1)}));
        for (int a = 0; a < 3; ++a) {
            int u = (a + 1) % 3, v = (a + 2) % 3;
            for (int i = 0; i < 4; ++i) {
                auto d = c;
                d[u] += i & 1;
                d[v] += (i >> 1) & 1;
                edges.insert(entity(a, d));
            }
            for (int i = 0; i < 2; ++i) {
                auto d = c;
                d[a] += i;
                faces.insert(entity(a, d));
            }
        }
    }
    return {vertices.size(), edges.size(), faces.size(), cells.size()};
}
void compare(int n, const std::set<Cell>& cells, Boundary b, const std::string& label) {
    auto m = measure(slab(n, cells), double(n), b);
    auto r = reference(n, cells, b);
    check(m.counts.vertices == r.vertices && m.counts.edges == r.edges && m.counts.faces == r.faces &&
              m.counts.cubes == r.cubes,
          label);
}
void shape(const std::set<Cell>& cells, Boundary b, double volume, double area, double h, int chi,
           const std::string& label) {
    auto m = measure(slab(8, cells), 8, b);
    check(near(m.volume, volume) && near(m.surface_area, area) && near(m.integrated_mean_curvature, h * M_PI) &&
              m.euler_characteristic == chi,
          label);
    compare(8, cells, b, label + " explicit cell-complex reference");
}
struct Field {
    int n, nx, start;
    bool real = true;
    std::vector<double> values;
    explicit Field(int size)
        : n(size), nx(size * (FML::ThisTask + 1) / FML::NTasks - size * FML::ThisTask / FML::NTasks),
          start(size * FML::ThisTask / FML::NTasks),
          values(size_t(nx) * n * (n + 2), std::numeric_limits<double>::quiet_NaN()) {
        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < n; ++y)
                for (int z = 0; z < n; ++z)
                    values[(size_t(x) * n + y) * (n + 2) + z] = (x + start + y + z) % 3 - 1;
    }
    int get_nmesh() const { return n; }
    int get_local_nx() const { return nx; }
    int get_local_x_start() const { return start; }
    bool get_grid_status_real() { return real; }
    double get_real(Cell c) const { return values.at((size_t(c[0]) * n + c[1]) * (n + 2) + c[2]); }
};
int main() {
    shape({}, Boundary::Periodic, 0, 0, 0, 0, "empty periodic set");
    shape({{3, 3, 3}}, Boundary::Open, 1, 6, 3, 1, "isolated cube");
    shape({{3, 3, 3}, {4, 3, 3}}, Boundary::Open, 2, 10, 4, 1, "face sharing cubes");
    shape({{3, 3, 3}, {4, 4, 3}}, Boundary::Open, 2, 12, 5, 1, "edge touching closed cubes");
    shape({{3, 3, 3}, {4, 4, 4}}, Boundary::Open, 2, 12, 6, 1, "corner touching closed cubes");
    shape({{2, 2, 2}, {5, 5, 5}}, Boundary::Open, 2, 12, 6, 2, "disconnected cubes");
    std::set<Cell> full;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z)
                full.insert({x, y, z});
    shape(full, Boundary::Open, 512, 384, 24, 1, "open full box");
    shape(full, Boundary::Periodic, 512, 0, 0, 0, "periodic full torus");
    full.erase({3, 3, 3});
    shape(full, Boundary::Open, 511, 390, 21, 2, "open box with a cavity");
    shape(full, Boundary::Periodic, 511, 6, -3, 1, "periodic box with a cavity");
    shape({{0, 2, 2}, {7, 2, 2}}, Boundary::Periodic, 2, 10, 4, 1, "periodic seam connectivity");
    shape({{0, 2, 2}, {7, 2, 2}}, Boundary::Open, 2, 12, 6, 2, "open seam stays disconnected");
    std::set<Cell> ring;
    for (int x = 0; x < 8; ++x)
        ring.insert({x, 2, 2});
    shape(ring, Boundary::Periodic, 8, 32, 8, 0, "periodic solid ring");
    auto s = slab(8, {{3, 3, 3}});
    auto a = measure(s, 8, Boundary::Open), b = measure(s, 16, Boundary::Open);
    check(near(b.volume, 8 * a.volume) && near(b.surface_area, 4 * a.surface_area) &&
              near(b.integrated_mean_curvature, 2 * a.integrated_mean_curvature) &&
              b.euler_characteristic == a.euler_characteristic && near(a.crofton_density[1], (2.0 / 9) * 3 / 512),
          "physical units and explicit Crofton normalization");
    for (int n : {1, 2, 7}) {
        std::set<Cell> random;
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y)
                for (int z = 0; z < n; ++z)
                    if ((x * 71 + y * 23 + z * 11 + 5) % 17 < 6)
                        random.insert({x, y, z});
        for (auto bound : {Boundary::Open, Boundary::Periodic})
            compare(n, random, bound,
                    "independent set-union reference N=" + std::to_string(n) +
                        (bound == Boundary::Open ? " open" : " periodic"));
    }
    Field f(7);
    auto ge = excursions(f, {0, 1, -1}, 7), gt = excursions(f, {0}, 7, Boundary::Periodic, ThresholdRule::Greater);
    std::set<Cell> cge, cgt;
    for (int x = 0; x < 7; ++x)
        for (int y = 0; y < 7; ++y)
            for (int z = 0; z < 7; ++z) {
                int v = (x + y + z) % 3 - 1;
                if (v >= 0)
                    cge.insert({x, y, z});
                if (v > 0)
                    cgt.insert({x, y, z});
            }
    auto rge = reference(7, cge, Boundary::Periodic), rgt = reference(7, cgt, Boundary::Periodic);
    check(ge.size() == 3 && ge[1].threshold == 1 && ge[2].threshold == -1 && ge[0].measures.counts.cubes == rge.cubes &&
              ge[0].measures.counts.vertices == rge.vertices && gt[0].measures.counts.cubes == rgt.cubes,
          "excursions preserve threshold order, ties and physical cells while ignoring NaN padding");
    f.real = false;
    bool failed = false;
    try {
        excursions(f, {0}, 7);
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    check(failed, "Fourier fields rejected collectively");
    f.real = true;
    if (f.nx)
        f.values[0] = std::numeric_limits<double>::infinity();
    failed = false;
    try {
        excursions(f, {0}, 7);
    } catch (const std::invalid_argument&) {
        failed = true;
    }
    check(failed, "nonfinite physical cells rejected collectively");
    if (FML::NTasks > 1) {
        Field good(7);
        failed = false;
        try {
            excursions(good, {double(FML::ThisTask)}, 7);
        } catch (const std::invalid_argument&) {
            failed = true;
        }
        check(failed, "different threshold schemas rejected across MPI ranks");
    }
    if (!FML::ThisTask)
        std::cout << (failures ? "Morphology tests FAILED" : "All morphology tests passed") << '\n';
    return failures ? 1 : 0;
}
