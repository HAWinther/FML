// GhostedCells (per-cell data with ghost layers from the neighbouring tasks) on slabs and pencils: after
// communicate() every cell, ghosts and corners included, holds the value of the cell it wraps to
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GhostedCells.h>
#include <FML/Global/Global.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <string>

using FML::TESTS::check;

namespace {
    template <int N, class Grid>
    bool ghosts_ok(const Grid & layout, int nl, int nr) {
        const int n = layout.get_nmesh();
        FML::GRID::GhostedCells<long long, N> cells(layout, nl, nr, -1);
        auto global_index = [&](std::array<int, N> g) {
            long long i = 0;
            for (int a = 0; a < N; a++)
                i = i * n + g[a];
            return i;
        };
        // The local cells
        for (long long i = 0; i < cells.size(); i++)
            if (not cells.is_ghost(i))
                cells[i] = global_index(cells.global_coord(cells.coord(i)));
        cells.communicate();
        bool ok = true;
        long long nghost = 0;
        for (long long i = 0; i < cells.size(); i++) {
            ok = ok and cells[i] == global_index(cells.global_coord(cells.coord(i)));
            nghost += cells.is_ghost(i);
        }
        // Whole axes wrap: index(c) of a coordinate outside [0, Nmesh) is that of the wrapped one
        std::array<int, N> c{};
        for (int a = 0; a < N; a++)
            if (not cells.is_split(a))
                c[a] = -1;
        if (cells.size() > 0)
            ok = ok and cells.at(c) == global_index(cells.global_coord(c));
        long long expected = 0;
        if (cells.size() > 0) {
            long long all = 1, main = 1;
            for (int a = 0; a < N; a++) {
                const auto r = cells.get_local_range(a);
                all *= r[1] + (cells.is_split(a) ? nl + nr : 0);
                main *= r[1];
            }
            expected = all - main;
        }
        ok = ok and nghost == expected;
        int good = ok ? 1 : 0;
        FML::MinOverTasks(&good);
        return good == 1;
    }
} // namespace

int main() {
    const int n = FML::NTasks % 3 == 0 ? 12 : 16;
    {
        FML::GRID::FFTWGrid<3> slab(n, 1, 1);
        check(ghosts_ok<3>(slab, 1, 1), "slabs 3D: 1 ghost layer on each side");
        check(ghosts_ok<3>(slab, 1, 0), "slabs 3D: 1 ghost layer on the left");
        FML::GRID::FFTWGrid<2> slab2(n, 1, 1);
        check(ghosts_ok<2>(slab2, 1, 2), "slabs 2D: 1 left and 2 right ghost layers");
    }
    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> pen(n, 0, 0, layout);
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0);
        check(ghosts_ok<3>(pen, 1, 1), name + ": 1 ghost layer on each side, corners included");
        check(ghosts_ok<3>(pen, 2, 1), name + ": 2 left and 1 right ghost layers");
        check(ghosts_ok<3>(pen, 1, 0), name + ": 1 ghost layer on the left");
    }
    return FML::TESTS::summary("ghosted cells");
}
