#ifndef FML_PENCIL_GRID_H
#define FML_PENCIL_GRID_H

#include <FML/FFTWGrid/FFTWGrid.h>

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

// =====================================================================================================================
// PencilGrid: the real and Fourier data of a grid distributed in pencils, FFTs by MPI transposes, ghost layers. Its
// layout is a PencilDomain made from the mesh (Global/Domain.h: the cells and Fourier modes of each rank, the
// communicators and the FFT workspace, shared by the grids made with it; also the domain of the particles that go
// with the grid, particle_domain()).
// =====================================================================================================================


#ifndef USE_MPI
#error "PencilGrid requires USE_MPI=true"
#endif
#ifndef USE_FFTW
#error "PencilGrid requires USE_FFTW=true"
#endif

namespace FML::GRID {

    struct PencilRealCell {
        int i{0}, j{0}, k{0};
        size_t index{0};
    };
    struct PencilFourierCell {
        int a{0}, b{0}, ix{0};
        size_t index{0};
    };

    class PencilRealRange {
      public:
        class iterator {
          public:
            iterator(PencilRealCell cell, int j0, int j1, int nz, size_t row_skip, size_t plane_skip)
                : cell_(cell), j0_(j0), j1_(j1), nz_(nz), row_skip_(row_skip), plane_skip_(plane_skip) {}
            iterator(PencilRealCell cell, int nz, size_t row, size_t row_end, int ny,
                     int nyg, int nghost_x, int nghost_y, int nzpad)
                : cell_(cell), nz_(nz), linear_(true), row_(row), row_end_(row_end), ny_(ny),
                  nyg_(nyg), nghost_x_(nghost_x), nghost_y_(nghost_y), nzpad_(nzpad) {}
            const PencilRealCell & operator*() const { return cell_; }
            iterator & operator++();
            bool operator==(const iterator & rhs) const {
                return linear_ ? row_ == rhs.row_ && cell_.k == rhs.cell_.k : cell_.index == rhs.cell_.index;
            }
            bool operator!=(const iterator & rhs) const { return !(*this == rhs); }

          private:
            PencilRealCell cell_{};
            int j0_{0}, j1_{0}, nz_{0};
            size_t row_skip_{0}, plane_skip_{0};
            bool linear_{false};
            size_t row_{0}, row_end_{0};
            int ny_{0}, nyg_{0}, nghost_x_{0}, nghost_y_{0}, nzpad_{0};
        };

        PencilRealRange(int i0, int i1, int j0, int j1, int nz, int nzpad, int nyg, int nghost_x, int nghost_y);
        PencilRealRange(size_t row_begin, size_t row_end, int nx, int ny, int nz,
                        int nzpad, int nyg, int nghost_x, int nghost_y);
        iterator begin() const;
        iterator end() const;
        size_t size() const { return (linear_ ? row_end_ - row_begin_ : size_t(i1_ - i0_) * size_t(j1_ - j0_)) * size_t(nz_); }

      private:
        int i0_, i1_, j0_, j1_, nz_, nzpad_, nyg_, nghost_x_, nghost_y_;
        bool linear_{false};
        size_t row_begin_{0}, row_end_{0};
        int nx_{0}, ny_{0};
        iterator make(int i) const;
    };

    class PencilFourierRange {
      public:
        class iterator {
          public:
            iterator(PencilFourierCell cell, int nkz, int nx) : cell_(cell), nkz_(nkz), nx_(nx) {}
            iterator(PencilFourierCell cell, size_t row, size_t row_end, int nkz, int nx)
                : cell_(cell), nkz_(nkz), nx_(nx), linear_(true), row_(row), row_end_(row_end) {}
            const PencilFourierCell & operator*() const { return cell_; }
            iterator & operator++();
            bool operator==(const iterator & rhs) const { return linear_ ? row_ == rhs.row_ && cell_.ix == rhs.cell_.ix : cell_.index == rhs.cell_.index; }
            bool operator!=(const iterator & rhs) const { return !(*this == rhs); }

          private:
            PencilFourierCell cell_{};
            int nkz_{0}, nx_{0};
            bool linear_{false};
            size_t row_{0}, row_end_{0};
        };

        PencilFourierRange(int nky, int nkz, int nx) : nky_(nky), nkz_(nkz), nx_(nx) {}
        PencilFourierRange(size_t row_begin, size_t row_end, int nky, int nkz, int nx)
            : nky_(nky), nkz_(nkz), nx_(nx), linear_(true), row_begin_(row_begin), row_end_(row_end) {}
        iterator begin() const;
        iterator end() const;
        size_t size() const { return (linear_ ? row_end_ - row_begin_ : size_t(nky_) * size_t(nkz_)) * size_t(nx_); }

      private:
        int nky_, nkz_, nx_;
        bool linear_{false};
        size_t row_begin_{0}, row_end_{0};
    };

    /// In-place 1D, 2D, or 3D real/Fourier grid on a 2D Cartesian decomposition.
    /// 1D uses a distributed real slab and an all-gather/full-complex FFT; 2D/3D
    /// use the pencil transpose path.
    class PencilGrid {
      public:
        enum class Status { Real, Fourier };

        explicit PencilGrid(std::shared_ptr<const PencilDomain> decomposition, int nghost = 0,
                            unsigned fftw_flags = FFTW_MEASURE);
        ~PencilGrid();
        PencilGrid(const PencilGrid & other);
        PencilGrid(PencilGrid && other) noexcept;
        PencilGrid & operator=(PencilGrid other) noexcept { swap(other); return *this; }
        void swap(PencilGrid & other) noexcept;

        void fftw_r2c();
        void fftw_c2r();
        void communicate_boundaries();
        void add_ghosts_to_owners();

        PencilRealRange get_real_range() const { return PencilRealRange(0, nx_, 0, ny_, mesh_[2], nzpad_, nyg_, nghost_, nghost_y_); }
        PencilRealRange get_real_range_with_ghosts() const {
            return PencilRealRange(-nghost_, nx_ + nghost_, -nghost_y_, ny_ + nghost_y_, mesh_[2], nzpad_, nyg_, nghost_, nghost_y_);
        }
        PencilFourierRange get_fourier_range() const { return PencilFourierRange(nky_, nkz_, mesh_[0]); }
        PencilRealRange real_range() const { return get_real_range(); }
        PencilRealRange real_range_with_ghosts() const { return get_real_range_with_ghosts(); }
        PencilFourierRange fourier_range() const { return get_fourier_range(); }

        FloatType & operator[](const PencilRealCell & cell);
        const FloatType & operator[](const PencilRealCell & cell) const;
        ComplexType & operator[](const PencilFourierCell & cell);
        const ComplexType & operator[](const PencilFourierCell & cell) const;
        FloatType & real(int i, int j, int k);
        const FloatType & real(int i, int j, int k) const;
        ComplexType & fourier(int a, int b, int ix);
        const ComplexType & fourier(int a, int b, int ix) const;

        FloatType * get_real_grid();
        const FloatType * get_real_grid() const;
        ComplexType * get_fourier_grid();
        const ComplexType * get_fourier_grid() const;
        my_fftw_complex * get_fftw_grid();

        size_t get_index_real(const std::array<int, 3> & coord) const;
        size_t get_index_fourier(const std::array<int, 3> & coord) const;
        std::array<int, 3> get_coord_from_index(size_t index) const;
        std::array<int, 3> get_fourier_coord_from_index(size_t index) const;
        FloatType get_real(const std::array<int, 3> & coord) const;
        FloatType get_real_from_index(size_t index) const;
        void set_real(const std::array<int, 3> & coord, FloatType value);
        void set_real_from_index(size_t index, FloatType value);
        void add_real(const std::array<int, 3> & coord, FloatType value);
        ComplexType get_fourier(const std::array<int, 3> & coord) const;
        ComplexType get_fourier_from_index(size_t index) const;
        void set_fourier(const std::array<int, 3> & coord, ComplexType value);
        void set_fourier_from_index(size_t index, ComplexType value);

        std::array<int, 3> global_cell(const PencilRealCell & cell) const;
        std::array<double, 3> get_real_position(const std::array<int, 3> & coord) const;
        std::array<double, 3> position(const PencilRealCell & cell) const { return get_real_position({cell.i, cell.j, cell.k}); }
        std::array<int, 3> get_fourier_integer_wavevector(const std::array<int, 3> & coord) const;
        std::array<double, 3> get_fourier_wavevector(const std::array<int, 3> & coord) const;
        std::array<double, 3> get_fourier_wavevector_from_index(size_t index) const;
        double hermitian_weight(const PencilFourierCell & cell) const;
        void get_fourier_wavevector_and_norm_by_index(size_t index, std::array<double, 3> & wavevector,
                                                      double & magnitude) const;

        void fill_real_grid(FloatType value);
        void fill_real_grid(const std::function<FloatType(std::array<double, 3> &)> & function);
        void fill_fourier_grid(ComplexType value);
        void fill_fourier_grid(const std::function<ComplexType(std::array<double, 3> &)> & function);

        explicit operator bool() const { return main_ != nullptr; }
        const PencilDomain & decomposition() const { return *decomposition_; }
        std::shared_ptr<const PencilDomain> decomposition_ptr() const { return decomposition_; }
        int nghost() const { return nghost_; }
        int nghost(int axis) const { return axis == 0 ? nghost_ : (axis == 1 ? nghost_y_ : 0); }
        Status status() const { return status_; }
        void set_status(Status status) { status_ = status; }
        void set_grid_status_real(bool is_real) { status_ = is_real ? Status::Real : Status::Fourier; }
        bool get_grid_status_real() const { return status_ == Status::Real; }
        std::array<int, 3> get_mesh() const { return mesh_; }
        int get_nmesh(int axis) const { return mesh_.at(size_t(axis)); }
        int get_ndim() const { return decomposition_ ? decomposition_->ndim() : 3; }
        int get_local_nx() const { return nx_; }
        int get_local_ny() const { return ny_; }
        int get_local_x_start() const { return xs_; }
        int get_local_y_start() const { return ys_; }
        int get_local_nky() const { return nky_; }
        int get_local_nkz() const { return nkz_; }
        int get_local_ky_start() const { return kys_; }
        int get_local_kz_start() const { return kzs_; }
        size_t get_ntot_real() const { return size_t(nx_) * size_t(ny_) * size_t(nzpad_); }
        size_t get_ntot_fourier() const { return size_t(nky_) * size_t(nkz_) * size_t(mesh_[0]); }
        size_t get_ntot_fourier_alloc() const { return nmain_; }
        size_t allocated_storage_bytes() const { return nmain_ * sizeof(ComplexType); }
        PencilRealRange get_real_range_of_rows(size_t row_begin, size_t row_end) const;
        PencilFourierRange get_fourier_range_of_rows(size_t row_begin, size_t row_end) const;
        size_t get_local_nrows() const { return size_t(nx_) * size_t(nky_); }
        void info() const;
        void free();

      private:
        std::shared_ptr<const PencilDomain> decomposition_;
        int nghost_{0};
        int nghost_y_{0};
        unsigned fftw_flags_{FFTW_MEASURE};
        std::array<int, 3> mesh_{};
        int nzc_{0}, nzpad_{0};
        int xs_{0}, nx_{0}, ys_{0}, ny_{0};
        int kys_{0}, nky_{0}, kzs_{0}, nkz_{0};
        int nxg_{0}, nyg_{0};
        ComplexType * main_{nullptr};
        size_t nmain_{0};
        size_t nscratch_{0};
        my_fftw_plan plan_r2c_{nullptr}, plan_c2r_{nullptr}, plan_y_forward_{nullptr}, plan_y_backward_{nullptr};
        my_fftw_plan plan_x_forward_{nullptr}, plan_x_backward_{nullptr};
        std::vector<MPI_Datatype> type_a_, type_b_row_, type_b_column_, type_c_;
        std::array<MPI_Datatype, 10> ghost_types_{};
        Status status_{Status::Real};

        FloatType * real_data() { return reinterpret_cast<FloatType *>(main_); }
        const FloatType * real_data() const { return reinterpret_cast<const FloatType *>(main_); }
        size_t real_offset(int i, int j, int k) const;
        size_t fourier_offset(int a, int b, int ix) const;
        void check_real(int i, int j, int k) const;
        void check_fourier(int a, int b, int ix) const;
        void setup();
        void create_plans();
        void create_datatypes();
        void release();
        static int fold(int k, int n) { return k <= n / 2 ? k : k - n; }
        static int wrap(int i, int n) { i %= n; return i < 0 ? i + n : i; }
    };

} // namespace FML::GRID

#endif
