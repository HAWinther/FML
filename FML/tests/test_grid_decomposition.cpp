// The decomposition of every grid type (grid.decomposition(), FML::GRID::particle_domain_of): the FML::Domain of the
// cells each task holds, compared with the local range of the grid gathered from all tasks. FFTWGrid: its slabs (the
// uniform Slab: NTasks divides Nmesh), MPIGrid: its slabs (also for N not divisible by NTasks) or the domain it was
// made on, PencilGrid: its pencils (P0 x 1 pencils as a Slab)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>
#include <FML/MPIGrid/MPIGrid.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <string>
#include <vector>

using FML::TESTS::check;

namespace {
    // The x-ranges [start, start + n) / Nmesh of all tasks as a Slab
    FML::Domain gathered_slabs(long long start, long long n, int Nmesh) {
        double lo = double(start) / Nmesh, hi = double(start + n) / Nmesh;
        return FML::Domain::slab(FML::GatherFromTasks(&lo), FML::GatherFromTasks(&hi));
    }
} // namespace

int main() {
    const int N = FML::NTasks;

    // FFTWGrid: its slabs, the uniform Slab (the constructor requires NTasks to divide Nmesh)
    for (int Nmesh : {12 * N, 16 * N, 30 * N}) {
        FML::GRID::FFTWGrid<3> grid(Nmesh, 1, 1);
        const auto actual = gathered_slabs(grid.get_local_x_start(), grid.get_local_nx(), Nmesh);
        const auto d = grid.decomposition();
        check(d == actual and d.valid_slab_partition() and FML::GRID::particle_domain_of(grid) == d,
              "FFTWGrid<3> Nmesh = " + std::to_string(Nmesh) + ": decomposition() = its FFTW slabs");
        check(FML::same_partition(d, FML::Domain::uniform_slab(N)),
              "FFTWGrid<3> Nmesh = " + std::to_string(Nmesh) + ": the uniform Slab");
    }
    {
        FML::GRID::FFTWGrid<2> grid(12 * N);
        check(grid.decomposition() == gathered_slabs(grid.get_local_x_start(), grid.get_local_nx(), 12 * N),
              "FFTWGrid<2>: decomposition() = its FFTW slabs");
    }

    // MPIGrid: its own slabs (one more slice on the first N % NTasks tasks), or the domain it was made on
    for (int Ngrid : {16, 4 * N + 3}) {
        FML::GRID::MPIGrid<3, double> grid(Ngrid, true, 1, 1);
        const auto r = grid.get_local_range(0);
        check(grid.decomposition() == gathered_slabs(r[0], r[1], Ngrid),
              "MPIGrid Ngrid = " + std::to_string(Ngrid) + ": decomposition() = its slabs");
    }
    if (16 % N == 0) {
        const auto d = FML::Domain::uniform_slab(N);
        FML::GRID::MPIGrid<3, double> grid(16, true, d, 1, 1);
        check(grid.decomposition() == d, "MPIGrid on a domain: decomposition() = the domain");
    }

    // PencilGrid: its pencils; P0 x 1 pencils are a Slab; particle_domain_of is the same
    for (int p0 = 1; p0 <= N; p0++) {
        if (N % p0 != 0)
            continue;
        const auto pencils = FML::PencilDomain::uniform(p0, N / p0);
        FML::GRID::PencilGrid<3> grid(16 * N, 1, 1, pencils);
        const auto d = grid.decomposition();
        const std::string label = "PencilGrid " + std::to_string(p0) + " x " + std::to_string(N / p0);
        check(d == FML::Domain(pencils) and FML::GRID::particle_domain_of(grid) == d and
                  grid.get_domain().process_grid()[0] == p0,
              label + ": decomposition() = its pencils (the geometry is kept)");
        check(d.is_slab() == (N / p0 == 1), label + ": a Slab exactly when P1 = 1");
        if (N / p0 == 1)
            check(FML::same_partition(d, FML::GRID::FFTWGrid<3>(16 * N).decomposition()),
                  label + ": the same partition as an FFTWGrid of the same size");
    }
    return FML::TESTS::summary("grid decomposition");
}
