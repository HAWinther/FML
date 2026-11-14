#ifndef FML_TESTS_GRID_TEST_HELPERS_H
#define FML_TESTS_GRID_TEST_HELPERS_H
// Helpers for comparing grids decomposed differently (FFTWGrid slabs, PencilGrid pencils): every task
// gets the whole field (real space by global cell, Fourier space by integer wave-vector)
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <vector>

namespace FML::TESTS::GRIDS {

    using ComplexType = FML::GRID::ComplexType;

    /// Loop over the local real cells of a grid: f(index, global cell)
    template <class Grid, class F>
    void for_cells(const Grid & g, F && f) {
        for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(g.get_local_nrows()); irow++)
            for (auto && index : g.get_real_range_of_rows(irow, irow + 1)) {
                auto c = g.get_coord_from_index(index);
                for (int d = 0; d < int(c.size()); d++)
                    c[d] += int(g.get_local_range(d)[0]);
                f(index, c);
            }
    }

    /// Fill the real grid with f(position of the cell)
    template <int N, class Grid, class F>
    void fill(Grid & g, F && f) {
        const int n = g.get_nmesh();
        for_cells(g, [&](auto index, auto c) {
            std::array<double, N> x;
            for (int d = 0; d < N; d++)
                x[d] = c[d] / double(n);
            g.set_real_from_index(index, f(x));
        });
        g.set_grid_status_real(true);
    }

    /// The whole real field on every task (by global cell, row major)
    template <class Grid>
    std::vector<double> global_real(const Grid & g) {
        const int n = g.get_nmesh();
        const int ndim = g.get_ndim();
        size_t ntot = 1;
        for (int d = 0; d < ndim; d++)
            ntot *= n;
        std::vector<double> all(ntot, 0.0);
        for_cells(g, [&](auto index, auto c) {
            size_t k = 0;
            for (int d = 0; d < ndim; d++)
                k = k * n + c[d];
            all[k] = g.get_real_from_index(index);
        });
        FML::SumArrayOverTasks(all.data(), all.size());
        return all;
    }

    /// All the Fourier modes on every task: [kx][ky][kz] (the last axis [0, n/2])
    template <class Grid>
    std::vector<ComplexType> global_fourier(const Grid & g) {
        const int n = g.get_nmesh();
        const int ndim = g.get_ndim();
        size_t ntot = n / 2 + 1;
        for (int d = 0; d < ndim - 1; d++)
            ntot *= n;
        std::vector<double> re(ntot, 0.0), im(ntot, 0.0);
        for (FML::IndexIntType irow = 0; irow < FML::IndexIntType(g.get_local_nrows_fourier()); irow++)
            for (auto && index : g.get_fourier_range_of_rows(irow, irow + 1)) {
                const auto m = g.get_fourier_integer_wavevector_from_index(index);
                size_t k = 0;
                for (int d = 0; d < ndim; d++)
                    k = k * (d == ndim - 1 ? n / 2 + 1 : n) + (m[d] < 0 ? m[d] + n : m[d]);
                re[k] = std::real(g.get_fourier_from_index(index));
                im[k] = std::imag(g.get_fourier_from_index(index));
            }
        FML::SumArrayOverTasks(re.data(), re.size());
        FML::SumArrayOverTasks(im.data(), im.size());
        std::vector<ComplexType> all(ntot);
        for (size_t k = 0; k < ntot; k++)
            all[k] = ComplexType(re[k], im[k]);
        return all;
    }

    /// max |a - b| / max |b|
    template <class V>
    double max_rel_diff(const V & a, const V & b) {
        if (a.size() != b.size())
            return 1e100;
        double d = 0.0, m = 0.0;
        for (size_t i = 0; i < a.size(); i++) {
            d = std::max(d, double(std::abs(a[i] - b[i])));
            m = std::max(m, double(std::abs(b[i])));
        }
        return m > 0.0 ? d / m : d;
    }

} // namespace FML::TESTS::GRIDS
#endif
