// The voxel morphology (cubical counts, Minkowski functionals) of a field split in x and y (BinaryBox,
// pencils) equals that of the slabs, periodic and open; excursions of a PencilGrid equal those of an FFTWGrid
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/Morphology/VoxelMorphology.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/GridTestHelpers.h>
#include <FML/tests/TestHelpers.h>

#include <cmath>
#include <string>

using FML::TESTS::check;
namespace M = FML::MORPHOLOGY;

namespace {
    // A deterministic field (the same cell gets the same value on any decomposition)
    double value(int x, int y, int z) {
        return std::sin(0.9 * x + 0.3 * y * y) + std::cos(0.7 * y - 0.4 * z) + 0.5 * std::sin(1.3 * z + 0.2 * x * z);
    }
    bool same(const M::Measures & a, const M::Measures & b) {
        return a.counts.vertices == b.counts.vertices and a.counts.edges == b.counts.edges and
               a.counts.faces == b.counts.faces and a.counts.cubes == b.counts.cubes and
               a.euler_characteristic == b.euler_characteristic and a.volume == b.volume and
               a.surface_area == b.surface_area and a.integrated_mean_curvature == b.integrated_mean_curvature;
    }
} // namespace

int main() {
    const int n = FML::NTasks % 3 == 0 ? 12 : 16;
    const double threshold = 0.3;
    // Slabs
    M::BinarySlab slab;
    slab.nmesh = n;
    slab.x_start = size_t(n) * FML::ThisTask / FML::NTasks;
    slab.local_nx = size_t(n) * (FML::ThisTask + 1) / FML::NTasks - slab.x_start;
    slab.cells.resize(slab.local_nx * n * n);
    for (size_t ix = 0; ix < slab.local_nx; ix++)
        for (int y = 0; y < n; y++)
            for (int z = 0; z < n; z++)
                slab.cells[(ix * n + y) * n + z] = value(int(slab.x_start + ix), y, z) >= threshold;
    const auto ref_periodic = M::measure(slab, 100.0, M::Boundary::Periodic);
    const auto ref_open = M::measure(slab, 100.0, M::Boundary::Open);

    FML::GRID::FFTWGrid<3> fgrid(n);
    FML::TESTS::GRIDS::fill<3>(fgrid, [&](auto x) {
        return value(int(std::lround(x[0] * n)), int(std::lround(x[1] * n)), int(std::lround(x[2] * n)));
    });
    const std::vector<double> thresholds{-1.0, 0.0, 0.3, 1.2};
    const auto ref_exc = M::excursions(fgrid, thresholds, 100.0, M::Boundary::Periodic);
    const auto ref_exc_open = M::excursions(fgrid, thresholds, 100.0, M::Boundary::Open, M::ThresholdRule::Greater);

    for (int p0 = 1; p0 <= FML::NTasks; p0++) {
        if (FML::NTasks % p0)
            continue;
        const auto layout = FML::PencilDomain::uniform(p0, FML::NTasks / p0);
        FML::GRID::PencilGrid<3> pgrid(n, 0, 0, layout);
        const std::string name = "pencils " + std::to_string(p0) + "x" + std::to_string(FML::NTasks / p0);
        M::BinaryBox box;
        box.nmesh = n;
        box.x_start = size_t(pgrid.get_local_range(0)[0]);
        box.local_nx = size_t(pgrid.get_local_range(0)[1]);
        box.y_start = size_t(pgrid.get_local_range(1)[0]);
        box.local_ny = size_t(pgrid.get_local_range(1)[1]);
        box.cells.resize(box.local_nx * box.local_ny * n);
        for (size_t ix = 0; ix < box.local_nx; ix++)
            for (size_t iy = 0; iy < box.local_ny; iy++)
                for (int z = 0; z < n; z++)
                    box.cells[(ix * box.local_ny + iy) * n + z] =
                        value(int(box.x_start + ix), int(box.y_start + iy), z) >= threshold;
        check(same(M::measure(box, 100.0, M::Boundary::Periodic), ref_periodic),
              name + ": periodic counts and functionals of the slabs");
        check(same(M::measure(box, 100.0, M::Boundary::Open), ref_open), name + ": open counts and functionals of the slabs");

        FML::TESTS::GRIDS::fill<3>(pgrid, [&](auto x) {
            return value(int(std::lround(x[0] * n)), int(std::lround(x[1] * n)), int(std::lround(x[2] * n)));
        });
        const auto exc = M::excursions(pgrid, thresholds, 100.0, M::Boundary::Periodic);
        const auto exc_open = M::excursions(pgrid, thresholds, 100.0, M::Boundary::Open, M::ThresholdRule::Greater);
        bool ok = exc.size() == ref_exc.size() and exc_open.size() == ref_exc_open.size();
        for (size_t i = 0; ok and i < exc.size(); i++)
            ok = same(exc[i].measures, ref_exc[i].measures) and same(exc_open[i].measures, ref_exc_open[i].measures);
        check(ok, name + ": excursions of a PencilGrid (4 thresholds, periodic and open) = those of the FFTWGrid");
    }
    return FML::TESTS::summary("morphology on pencils");
}
