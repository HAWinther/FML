#ifndef FML_SPH_CELLS_HEADER
#define FML_SPH_CELLS_HEADER
#include <FML/SPH/Kernel.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace FML::SPH::detail {
    template <int D>
    Vector<D> separation(const Vector<D> & a, const Vector<D> & b, double L) {
        Vector<D> x{};
        for (int d = 0; d < D; ++d) {
            x[d] = a[d] - b[d];
            if (x[d] > .5 * L)
                x[d] -= L;
            if (x[d] < -.5 * L)
                x[d] += L;
        }
        return x;
    }
    template <int D>
    double dot(const Vector<D> & a, const Vector<D> & b) {
        double v = 0;
        for (int d = 0; d < D; ++d)
            v += a[d] * b[d];
        return v;
    }
    /// Flat periodic cell index. Query visits each aliased cell once, without allocations.
    template <int D>
    class Cells {
        double L_;
        int n_ = 1;
        std::vector<size_t> offset_, indices_;
        size_t cell(const Vector<D> & x) const {
            size_t id = 0;
            for (int d = 0; d < D; ++d)
                id = id * n_ + std::min(n_ - 1, int(x[d] / L_ * n_));
            return id;
        }

      public:
        template <class Records, class GetPosition>
        void build(const Records & records, GetPosition position, double L, double radius, size_t budget) {
            L_ = L;
            // Bound the per-axis count before conversion; radius can be tiny.
            int cap = std::max(1, int(std::pow(double(budget), 1. / D)));
            while (std::pow(double(cap), D) > double(budget))
                --cap;
            n_ = std::max(1, int(std::min(double(cap), std::floor(L / radius))));
            size_t count = 1;
            for (int d = 0; d < D; ++d)
                count *= size_t(n_);
            offset_.assign(count + 1, 0);
            for (const auto & r : records)
                ++offset_[cell(position(r)) + 1];
            std::partial_sum(offset_.begin(), offset_.end(), offset_.begin());
            auto cursor = offset_;
            indices_.resize(records.size());
            for (size_t i = 0; i < records.size(); ++i)
                indices_[cursor[cell(position(records[i]))]++] = i;
        }
        template <class Function>
        void neighbors(const Vector<D> & x, Function && visit) const {
            std::array<std::array<int, 3>, D> axis{};
            std::array<int, D> sizes{};
            for (int d = 0; d < D; ++d) {
                int q = std::min(n_ - 1, int(x[d] / L_ * n_));
                for (int shift = -1; shift <= 1; ++shift) {
                    int c = (q + shift + n_) % n_;
                    bool seen = false;
                    for (int j = 0; j < sizes[d]; ++j)
                        seen |= axis[d][j] == c;
                    if (!seen)
                        axis[d][sizes[d]++] = c;
                }
            }
            size_t total = 1;
            for (int d = 0; d < D; ++d)
                total *= sizes[d];
            for (size_t v = 0; v < total; ++v) {
                size_t k = v, id = 0;
                for (int d = 0; d < D; ++d) {
                    id = id * n_ + axis[d][k % sizes[d]];
                    k /= sizes[d];
                }
                for (size_t j = offset_[id]; j < offset_[id + 1]; ++j)
                    visit(indices_[j]);
            }
        }
    };
} // namespace FML::SPH::detail
#endif
