// The NUFFT on pencils. With the particles in the domain of a PencilGrid (and the fine grids made on its
// pencils) the NUFFT gives the same as with an FFTWGrid and the particles on the slabs:
//   - particles_to_fourier_grid_nufft (type 1): the same Fourier modes
//   - nufft_type2: the same values at the particles (PSWF and ES kernels)
//   - compute_force_nufft_with_kernel: the same forces
//   - copy_fourier_modes between grids of different sizes and types (FFTWGrid <-> PencilGrid): the same modes
// on every pencil layout of the tasks (3D) and on the PencilGrid<2> slabs (2D)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/FFTWGrid/GridInterface.h>
#include <FML/Global/Global.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/NUFFT/NUFFT.h>
#include <FML/PencilDomain/PencilGrid.h>
#include <FML/tests/TestHelpers.h>

#include <algorithm>
#include <complex>
#include <map>
#include <random>
#include <string>
#include <vector>

using FML::TESTS::check;
using FML::TESTS::num;
using FloatType = FML::GRID::FloatType;
using ComplexType = FML::GRID::ComplexType;

namespace {

    template <int N>
    struct Part {
        double pos[N];
        double id;
        double * get_pos() { return pos; }
        constexpr int get_ndim() const { return N; }
    };

    // The same particles on every task (uniform plus a clump), each task takes every NTasks-th
    template <int N>
    std::vector<Part<N>> my_particles(int npart) {
        std::mt19937 gen(4321);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::normal_distribution<double> G(0.0, 0.05);
        std::vector<Part<N>> mine;
        for (int i = 0; i < npart; i++) {
            Part<N> p{};
            for (int d = 0; d < N; d++) {
                p.pos[d] = i % 4 == 0 ? 0.3 + G(gen) : U(gen);
                p.pos[d] -= std::floor(p.pos[d]);
                if (p.pos[d] >= 1.0)
                    p.pos[d] = 0.0;
            }
            p.id = i;
            if (i % FML::NTasks == FML::ThisTask)
                mine.push_back(p);
        }
        return mine;
    }

    // All Fourier modes of a grid on every task: [gx][gy][gz] (the last axis [0, n/2])
    template <int N, template <int> class Grid>
    std::vector<ComplexType> global_modes(const Grid<N> & grid) {
        const int n = grid.get_nmesh();
        size_t ntot = n / 2 + 1;
        for (int d = 0; d < N - 1; d++)
            ntot *= n;
        std::vector<double> re(ntot, 0.0), im(ntot, 0.0);
        for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(grid.get_local_nrows_fourier()); irow++)
            for (auto && index : grid.get_fourier_range_of_rows(irow, irow + 1)) {
                const auto m = grid.get_fourier_integer_wavevector_from_index(index);
                size_t k = 0;
                for (int d = 0; d < N; d++)
                    k = k * (d == N - 1 ? n / 2 + 1 : n) + (m[d] < 0 ? m[d] + n : m[d]);
                re[k] = std::real(grid.get_fourier_from_index(index));
                im[k] = std::imag(grid.get_fourier_from_index(index));
            }
        FML::SumArrayOverTasks(re.data(), re.size());
        FML::SumArrayOverTasks(im.data(), im.size());
        std::vector<ComplexType> all(ntot);
        for (size_t k = 0; k < ntot; k++)
            all[k] = ComplexType(re[k], im[k]);
        return all;
    }

    double max_rel_diff(const std::vector<ComplexType> & a, const std::vector<ComplexType> & b) {
        double d = 0.0, m = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            d = std::max(d, double(std::abs(a[i] - b[i])));
            m = std::max(m, double(std::abs(b[i])));
        }
        return a.size() == b.size() ? d / m : 1e100;
    }

    // Values at the particles by id on every task
    template <int N>
    std::map<long, double> by_id(FML::PARTICLE::MPIParticles<Part<N>> & p, const std::vector<FloatType> & values) {
        std::vector<double> local;
        for (size_t i = 0; i < p.get_npart(); i++) {
            local.push_back(p[i].id);
            local.push_back(values[i]);
        }
        int n = int(local.size());
        std::vector<int> counts(FML::NTasks), displ(FML::NTasks, 0);
        MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
        for (int i = 1; i < FML::NTasks; i++)
            displ[i] = displ[i - 1] + counts[i - 1];
        std::vector<double> all(displ.back() + counts.back());
        MPI_Allgatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displ.data(), MPI_DOUBLE,
                       FML::communicator);
        std::map<long, double> m;
        for (size_t k = 0; k < all.size(); k += 2)
            m[long(all[k])] = all[k + 1];
        return m;
    }

    double max_rel_diff(const std::map<long, double> & a, const std::map<long, double> & b) {
        if (a.size() != b.size())
            return 1e100;
        double d = 0.0, m = 0.0;
        for (auto & [id, v] : b) {
            d = std::max(d, std::abs(a.at(id) - v));
            m = std::max(m, std::abs(v));
        }
        return d / m;
    }

    template <int N>
    void run(const std::vector<FML::PencilDomain> & layouts) {
        const int nmesh = 48, nsmall = 24, npart = N == 3 ? 20000 : 8000;
        const std::string dim = std::to_string(N) + "D";
        auto mine = my_particles<N>(npart);
        const double eps = 1e-8;

        // The reference: FFTWGrid, the particles on the slabs
        FML::PARTICLE::MPIParticles<Part<N>> pslab;
        { auto copy = mine; pslab.create(copy, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed); }
        FML::GRID::FFTWGrid<N> ref(nmesh);
        FML::NUFFT::particles_to_fourier_grid_nufft<N>(pslab.get_particles_ptr(), pslab.get_npart(), npart, ref, eps);
        const auto ref_modes = global_modes(ref);
        std::vector<FloatType> v;
        FML::NUFFT::nufft_type2<N>(ref, pslab.get_particles_ptr(), pslab.get_npart(), v,
                                   FML::NUFFT::PSWFKernel<N>(eps, 2.0));
        const auto ref_type2 = by_id(pslab, v);
        FML::NUFFT::nufft_type2<N>(ref, pslab.get_particles_ptr(), pslab.get_npart(), v, FML::NUFFT::ESKernel(1e-6));
        const auto ref_type2_es = by_id(pslab, v);
        std::array<std::vector<FloatType>, N> force;
        FML::NUFFT::compute_force_nufft_with_kernel<N>(pslab.get_particles_ptr(), pslab.get_npart(), npart, nmesh,
                                                       1.5, force, FML::NUFFT::PSWFKernel<N>(eps, 2.0));
        std::array<std::map<long, double>, N> ref_force;
        for (int d = 0; d < N; d++)
            ref_force[d] = by_id(pslab, force[d]);
        FML::GRID::FFTWGrid<N> ref_small(nsmall);
        FML::NUFFT::copy_fourier_modes<N>(ref, ref_small, [](const std::array<int, N> & m) { return 1.0 + m[0]; });
        const auto ref_small_modes = global_modes(ref_small);

        for (auto & layout : layouts) {
            const std::string name = dim + " pencils " + std::to_string(layout.process_grid()[0]) + "x" +
                                     std::to_string(layout.process_grid()[1]);
            FML::GRID::PencilGrid<N> grid(nmesh, 0, 0, layout);
            FML::PARTICLE::MPIParticles<Part<N>> p(FML::GRID::particle_domain_of(grid));
            { auto copy = mine; p.create(copy, 3 * mine.size() + 1000, FML::PARTICLE::Input::Distributed); }

            // Type 1
            FML::NUFFT::particles_to_fourier_grid_nufft<N>(p.get_particles_ptr(), p.get_npart(), npart, grid, eps);
            const double d1 = max_rel_diff(global_modes(grid), ref_modes);
            check(d1 < 1e-12, name + ": the density modes (type 1) of the slabs (rel. diff " + num(d1, 2) + ")");

            // The reference modes on the pencils (FFTWGrid -> PencilGrid of the same size) and type 2
            FML::NUFFT::copy_fourier_modes<N>(ref, grid, [](const std::array<int, N> &) { return 1.0; });
            std::vector<ComplexType> trimmed = ref_modes;
            {
                // (copy_fourier_modes keeps |m| < n/2: the Nyquist planes are zero)
                FML::GRID::FFTWGrid<N> tmp(nmesh);
                FML::NUFFT::copy_fourier_modes<N>(ref, tmp, [](const std::array<int, N> &) { return 1.0; });
                trimmed = global_modes(tmp);
            }
            const double dc = max_rel_diff(global_modes(grid), trimmed);
            check(dc == 0.0, name + ": copy_fourier_modes FFTWGrid -> PencilGrid (rel. diff " + num(dc, 2) + ")");
            FML::NUFFT::nufft_type2<N>(grid, p.get_particles_ptr(), p.get_npart(), v, FML::NUFFT::PSWFKernel<N>(eps, 2.0));
            const double d2 = max_rel_diff(by_id(p, v), ref_type2);
            FML::NUFFT::nufft_type2<N>(grid, p.get_particles_ptr(), p.get_npart(), v, FML::NUFFT::ESKernel(1e-6));
            const double d2es = max_rel_diff(by_id(p, v), ref_type2_es);
            check(d2 < 1e-12 and d2es < 1e-12,
                  name + ": the values at the particles (type 2, PSWF and ES; rel. diff " + num(d2, 2) + ", " +
                      num(d2es, 2) + ")");

            // The force
            FML::NUFFT::compute_force_nufft_with_kernel<N>(grid, p.get_particles_ptr(), p.get_npart(), npart, 1.5,
                                                           force, FML::NUFFT::PSWFKernel<N>(eps, 2.0));
            double df = 0.0;
            for (int d = 0; d < N; d++)
                df = std::max(df, max_rel_diff(by_id(p, force[d]), ref_force[d]));
            check(df < 1e-12, name + ": the forces at the particles (rel. diff " + num(df, 2) + ")");

            // Between sizes and types: PencilGrid -> PencilGrid, PencilGrid -> FFTWGrid, FFTWGrid -> PencilGrid
            FML::NUFFT::copy_fourier_modes<N>(ref, grid, [](const std::array<int, N> &) { return 1.0; });
            auto factor = [](const std::array<int, N> & m) { return 1.0 + m[0]; };
            FML::GRID::PencilGrid<N> small(nsmall, 0, 0, layout);
            FML::NUFFT::copy_fourier_modes<N>(grid, small, factor);
            const double ds1 = max_rel_diff(global_modes(small), ref_small_modes);
            FML::GRID::FFTWGrid<N> small_fftw(nsmall);
            FML::NUFFT::copy_fourier_modes<N>(grid, small_fftw, factor);
            const double ds2 = max_rel_diff(global_modes(small_fftw), ref_small_modes);
            FML::NUFFT::copy_fourier_modes<N>(ref, small, factor);
            const double ds3 = max_rel_diff(global_modes(small), ref_small_modes);
            // ... and up again: the small modes in a large grid
            FML::GRID::PencilGrid<N> large(nmesh, 0, 0, layout);
            FML::GRID::FFTWGrid<N> large_fftw(nmesh);
            FML::NUFFT::copy_fourier_modes<N>(small, large, [](const std::array<int, N> &) { return 1.0; });
            FML::NUFFT::copy_fourier_modes<N>(ref_small, large_fftw, [](const std::array<int, N> &) { return 1.0; });
            const double ds4 = max_rel_diff(global_modes(large), global_modes(large_fftw));
            check(ds1 == 0.0 and ds2 == 0.0 and ds3 == 0.0 and ds4 == 0.0,
                  name + ": copy_fourier_modes between sizes and grid types (" + num(ds1, 2) + ", " + num(ds2, 2) +
                      ", " + num(ds3, 2) + ", " + num(ds4, 2) + ")");
        }
    }

} // namespace

int main() {
    std::vector<FML::PencilDomain> layouts3;
    for (int p0 = 1; p0 <= FML::NTasks; p0++)
        if (FML::NTasks % p0 == 0)
            layouts3.push_back(FML::PencilDomain::uniform(p0, FML::NTasks / p0));
    run<3>(layouts3);
    run<2>({FML::PencilDomain::uniform(FML::NTasks, 1)});
    return FML::TESTS::summary("NUFFT on pencils");
}
