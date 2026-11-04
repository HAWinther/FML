#ifndef FML_WAVE_COMPLEX_GRID_H
#define FML_WAVE_COMPLEX_GRID_H
#include <FML/FFTWGrid/FFTWGrid.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#ifndef USE_FFTW
#error "WaveDarkMatter requires USE_FFTW"
#endif
#if defined(SINGLE_PRECISION_FFTW)
#define FML_WAVE_FFT(name) fftwf_##name
#elif defined(LONG_DOUBLE_PRECISION_FFTW)
#define FML_WAVE_FFT(name) fftwl_##name
#else
#define FML_WAVE_FFT(name) fftw_##name
#endif
namespace FML::WAVEDM {
    using Real = FML::GRID::FloatType;
    using Complex = FML::GRID::ComplexType;
    namespace detail {
        inline void require(bool ok, const char * message) {
            int bad = !ok;
            FML::MaxOverTasks(&bad);
            if (bad)
                throw std::invalid_argument(message);
        }
        template <class T>
        inline void same(T value, const char * message) {
            T lo = value, hi = value;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, message);
        }
    } // namespace detail
    // Full N^3 complex transform, with untransposed FFTW slabs in BOTH domains.
    // Forward normalized by N^3, inverse unnormalized: coefficients are Fourier-series amplitudes.
    class ComplexGrid {
        int n_{};
        ptrdiff_t nx_{}, start_{};
        size_t owned_{}, allocated_{};
        Complex * data_{};
        FML::GRID::my_fftw_plan forward_{}, inverse_{};
        bool spectral_{};
        std::uint64_t transforms_{};
        void release() noexcept {
            if (forward_)
                FML_WAVE_FFT(destroy_plan)(forward_);
            if (inverse_)
                FML_WAVE_FFT(destroy_plan)(inverse_);
            if (data_)
                FML_WAVE_FFT(free)(data_);
            forward_ = nullptr;
            inverse_ = nullptr;
            data_ = nullptr;
        }

      public:
        explicit ComplexGrid(int n) : n_(n) {
            detail::same(n, "[WaveDM] Mesh differs between ranks");
            detail::require(n >= 2 &&
                                size_t(n) <=
                                    size_t(std::cbrt(double(std::numeric_limits<ptrdiff_t>::max() / sizeof(Complex)))),
                            "[WaveDM] Invalid or unaddressable mesh");
            ptrdiff_t allocated = ptrdiff_t(n) * n * n;
#ifdef USE_MPI
            allocated = FML_WAVE_FFT(mpi_local_size_3d)(n, n, n, FML::communicator, &nx_, &start_);
#else
            nx_ = n;
            start_ = 0;
#endif
            detail::require(allocated >= 0 && nx_ >= 0 && nx_ <= n && (nx_ == 0 || (start_ >= 0 && start_ + nx_ <= n)),
                            "[WaveDM] Invalid FFTW complex distribution");
            owned_ = size_t(nx_) * n * n;
            allocated_ = std::max(size_t(1), size_t(allocated));
            detail::require(allocated_ >= owned_ && allocated_ <= std::numeric_limits<size_t>::max() / sizeof(Complex),
                            "[WaveDM] Invalid FFTW allocation extent");
            data_ = reinterpret_cast<Complex *>(FML_WAVE_FFT(alloc_complex)(allocated_));
            try {
                detail::require(data_ != nullptr, "[WaveDM] Complex allocation failed");
                std::uninitialized_value_construct_n(data_, allocated_);
                auto * native = reinterpret_cast<FML::GRID::my_fftw_complex *>(data_);
                static_assert(sizeof(Complex) == sizeof(FML::GRID::my_fftw_complex));
#ifdef USE_MPI
                forward_ = FML_WAVE_FFT(
                    mpi_plan_dft_3d)(n, n, n, native, native, FML::communicator, FFTW_FORWARD, FFTW_ESTIMATE);
                inverse_ = FML_WAVE_FFT(
                    mpi_plan_dft_3d)(n, n, n, native, native, FML::communicator, FFTW_BACKWARD, FFTW_ESTIMATE);
#else
                forward_ = FML_WAVE_FFT(plan_dft_3d)(n, n, n, native, native, FFTW_FORWARD, FFTW_ESTIMATE);
                inverse_ = FML_WAVE_FFT(plan_dft_3d)(n, n, n, native, native, FFTW_BACKWARD, FFTW_ESTIMATE);
#endif
                detail::require(forward_ && inverse_, "[WaveDM] Complex FFT plan creation failed");
                std::fill_n(data_, allocated_, Complex{});
            } catch (...) {
                release();
                throw;
            }
        }
        ~ComplexGrid() { release(); }
        ComplexGrid(const ComplexGrid &) = delete;
        ComplexGrid & operator=(const ComplexGrid &) = delete;
        ComplexGrid(ComplexGrid && other) noexcept { swap(other); }
        ComplexGrid & operator=(ComplexGrid && other) noexcept {
            if (this != &other) {
                ComplexGrid tmp(std::move(other));
                swap(tmp);
            }
            return *this;
        }
        void swap(ComplexGrid & other) noexcept {
            using std::swap;
            swap(n_, other.n_);
            swap(nx_, other.nx_);
            swap(start_, other.start_);
            swap(owned_, other.owned_);
            swap(allocated_, other.allocated_);
            swap(data_, other.data_);
            swap(forward_, other.forward_);
            swap(inverse_, other.inverse_);
            swap(spectral_, other.spectral_);
            swap(transforms_, other.transforms_);
        }
        int mesh() const { return n_; }
        ptrdiff_t local_nx() const { return nx_; }
        ptrdiff_t local_start() const { return start_; }
        size_t size() const { return owned_; }
        size_t allocated_bytes() const { return allocated_ * sizeof(Complex); }
        std::uint64_t transforms() const { return transforms_; }
        bool spectral() const { return spectral_; }
        Complex * data() { return data_; }
        const Complex * data() const { return data_; }
        std::array<int, 3> coordinate(size_t i) const {
            return {int(start_ + ptrdiff_t(i / (size_t(n_) * n_))), int(i / n_ % n_), int(i % n_)};
        }
        std::array<int, 3> mode(size_t i) const {
            auto k = coordinate(i);
            for (auto & q : k)
                if (q > n_ / 2)
                    q -= n_;
            return k;
        }
        void copy_values(const ComplexGrid & other) {
            detail::require(n_ == other.n_ && nx_ == other.nx_ && start_ == other.start_,
                            "[WaveDM] Complex layouts differ");
            std::copy_n(other.data_, owned_, data_);
            spectral_ = other.spectral_;
        }
        void forward() {
            detail::require(!spectral_, "[WaveDM] Forward FFT requires position-space input");
            FML_WAVE_FFT(execute)(forward_);
            const Real inv = Real(1) / (Real(n_) * Real(n_) * Real(n_));
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < owned_; ++i)
                data_[i] *= inv;
            spectral_ = true;
            ++transforms_;
        }
        void inverse() {
            detail::require(spectral_, "[WaveDM] Inverse FFT requires Fourier input");
            FML_WAVE_FFT(execute)(inverse_);
            spectral_ = false;
            ++transforms_;
        }
    };
} // namespace FML::WAVEDM
#undef FML_WAVE_FFT
#endif
