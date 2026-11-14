#ifndef FML_WAVE_SCHRODINGER_POISSON_H
#define FML_WAVE_SCHRODINGER_POISSON_H
#include <FML/WaveDarkMatter/TimeFactors.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <vector>
namespace FML::WAVEDM {
    struct Diagnostics {
        double mass{}, kinetic{}, potential{}, max_density{}, spectral_tail_fraction{};
        double energy() const { return kinetic + potential; }
    };
    struct MemoryReport {
        size_t complex_buffers{}, real_buffers{}, external_density{};
        size_t total() const { return complex_buffers + real_buffers + external_density; }
        // FFTW internal plan/work allocations are opaque and excluded; measure peak RSS separately.
    };
    /// The solver on a grid type: FML::GRID::CFFTWGrid<3> (slabs, the default) or FML::GRID::CPencilGrid<3> (pencils)
    template <class Grid = FML::GRID::CFFTWGrid<3>>
    class SolverT {
      public:
        using GridType = Grid;

      private:
        std::unique_ptr<Grid> wave_, candidate_, scratch_;
        Physics physics_;
        double box_, time_, a_;
        std::uint64_t steps_{};
        // k2_ in the Fourier layout, potential_ (and external_) in the position-space layout
        std::vector<double> k2_, potential_;
        std::vector<Real> external_;
        std::uint64_t transforms_{};
        int mesh() const { return wave_->get_nmesh(); }
        size_t nreal() const { return size_t(wave_->get_ntot_real()); }
        size_t nfourier() const { return size_t(wave_->get_ntot_fourier()); }
        void forward(Grid & g) {
            detail::require(g.get_grid_status_real(), "[WaveDM] Forward FFT requires position-space input");
            g.fftw_forward();
            ++transforms_;
        }
        void inverse(Grid & g) {
            detail::require(!g.get_grid_status_real(), "[WaveDM] Inverse FFT requires Fourier input");
            g.fftw_backward();
            ++transforms_;
        }
        double volume_element() const { return std::pow(box_ / mesh(), 3); }
        static bool finite(Complex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }
        void check_layout() const {
            auto same_layout = [](const Grid & a, const Grid & b) {
                return a.get_nmesh() == b.get_nmesh() && a.get_ntot_real() == b.get_ntot_real() &&
                       a.get_ntot_fourier() == b.get_ntot_fourier() && a.get_local_range(0) == b.get_local_range(0) &&
                       a.get_local_range(1) == b.get_local_range(1) && a.get_n_extra_slices_left() == 0 &&
                       a.get_n_extra_slices_right() == 0;
            };
            detail::require(same_layout(*wave_, *candidate_) && same_layout(*wave_, *scratch_) &&
                                nfourier() == k2_.size() && nreal() == potential_.size(),
                            "[WaveDM] Mutable wave grid was replaced with an incompatible layout");
        }
        void check_wave(const Grid & g) const {
            check_layout();
            bool good = g.get_grid_status_real();
            const Complex * data = g.get_real_grid();
            for (size_t i = 0; i < nreal(); ++i)
                good &= finite(data[i]) && std::isfinite(double(std::norm(data[i])));
            detail::require(good, "[WaveDM] Need a finite, representable position-space density/wavefunction");
        }
        void solve_potential(const Grid & wave) {
            if (physics_.G == 0) {
                std::fill(potential_.begin(), potential_.end(), 0);
                return;
            }
            detail::copy_values(*scratch_, wave);
            const size_t nr = nreal(), nf = nfourier();
            const Complex * psi = wave.get_real_grid();
            Complex * data = scratch_->get_real_grid();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < nr; ++i)
                data[i] = Complex(Real(std::norm(psi[i]) + (external_.empty() ? 0 : external_[i])), 0);
            forward(*scratch_);
            Complex * fourier = scratch_->get_fourier_grid();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < nf; ++i) {
                const double factor = k2_[i] == 0 ? 0 : -4 * std::acos(-1.) * physics_.G / k2_[i];
                fourier[i] *= Real(factor);
            }
            inverse(*scratch_);
            data = scratch_->get_real_grid();
            bool good = true;
            for (size_t i = 0; i < nr; ++i) {
                potential_[i] = double(data[i].real());
                good &= std::isfinite(potential_[i]);
            }
            detail::require(good, "[WaveDM] Nonfinite periodic potential");
        }
        void phase_kick(Grid & wave, double weight) {
            bool good = true;
            for (double u : potential_)
                good &= std::isfinite((u / physics_.eta) * weight);
            detail::require(good, "[WaveDM] Potential phase overflows");
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < nreal(); ++i) {
                const double angle = std::remainder(-(potential_[i] / physics_.eta) * weight, 2 * std::acos(-1.));
                wave.get_real_grid()[i] *= Complex(Real(std::cos(angle)), Real(std::sin(angle)));
            }
        }

        // psi(k) *= exp(-i alpha k^2) = prod_d exp(-i alpha k_d^2): the phase is a product of three 1D factors (a table
        // of N values per step instead of cos/sin of every cell). The Fourier rows run along one axis (slabs: kz,
        // pencils: kx): the factor of the two other axes is computed once per row
        void kinetic_drift(Grid & wave, double alpha) {
            const int n = wave.get_nmesh();
            const double twopi = 2 * std::acos(-1.);
            std::vector<Complex> factor(n);
            for (int j = 0; j < n; ++j) {
                const int m = j > n / 2 ? j - n : j;
                const double k = twopi * m / box_;
                const double angle = std::remainder(-alpha * k * k, twopi);
                factor[j] = Complex(Real(std::cos(angle)), Real(std::sin(angle)));
            }
            constexpr int axis = Grid::fourier_row_axis;
            constexpr int a = axis == 0 ? 1 : 0, b = axis == 2 ? 1 : 2;
            const auto nrows = wave.get_local_nrows_fourier();
            Complex * fourier = wave.get_fourier_grid();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (FML::IndexIntType row = 0; row < nrows; ++row) {
                const auto k = wave.get_fourier_integer_wavevector_from_index(row * n);
                const Complex fab = factor[(k[a] + n) % n] * factor[(k[b] + n) % n];
                Complex * data = fourier + row * n;
                for (int j = 0; j < n; ++j)
                    data[j] *= fab * factor[j];
            }
        }

      public:
        SolverT(int mesh, double boxsize, Physics physics = {}, double time = 0, double a = 1)
            : physics_(physics), box_(boxsize), time_(time), a_(a) {
            for (double x : {boxsize, physics.eta, physics.G, time, a})
                detail::same(x, "[WaveDM] Solver parameters differ between ranks");
            detail::require(boxsize > 0 && std::isfinite(boxsize) && std::isfinite(boxsize * boxsize * boxsize) &&
                                physics.eta > 0 && std::isfinite(physics.eta) && physics.G >= 0 &&
                                std::isfinite(physics.G) && std::isfinite(time) && a > 0 && std::isfinite(a),
                            "[WaveDM] Invalid units, coupling, box, time or scale factor");
            detail::same(mesh, "[WaveDM] Mesh differs between ranks");
            detail::require(mesh >= 2 && size_t(mesh) <= size_t(std::cbrt(double(std::numeric_limits<ptrdiff_t>::max() /
                                                                                   sizeof(Complex)))),
                            "[WaveDM] Invalid or unaddressable mesh");
            wave_ = std::make_unique<Grid>(detail::make_grid<Grid>(mesh));
            candidate_ = std::make_unique<Grid>(detail::make_grid<Grid>(mesh));
            scratch_ = std::make_unique<Grid>(detail::make_grid<Grid>(mesh));
            k2_.resize(nfourier());
            potential_.resize(nreal());
            bool valid = true;
            for (size_t i = 0; i < k2_.size(); ++i) {
                const auto mode = wave_->get_fourier_integer_wavevector_from_index(FML::IndexIntType(i));
                double k2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double k = 2 * std::acos(-1.) * mode[d] / box_;
                    k2 += k * k;
                }
                k2_[i] = k2;
                k2max_ = std::max(k2max_, k2);
                valid &= std::isfinite(k2);
            }
            FML::MaxOverTasks(&k2max_);
            detail::require(valid && std::isfinite(volume_element()) && volume_element() > 0,
                            "[WaveDM] Unrepresentable spatial scales");
        }
        SolverT(const SolverT &) = delete;
        SolverT & operator=(const SolverT &) = delete;
        // The wave in position space: the local cells are get_real_grid()[i] for i < get_ntot_real().
        // Mutable access may change the wave, so the cached potential of the wave is no longer valid
        Grid & wave() {
            potential_valid_ = false;
            return *wave_;
        }
        const Grid & wave() const { return *wave_; }
        /// The number of local cells (position space) and the global integer coordinates of local cell i
        size_t local_size() const { return nreal(); }
        std::array<int, 3> coordinate(size_t i) const { return detail::global_coordinate(*wave_, i); }
        Physics physics() const { return physics_; }
        double boxsize() const { return box_; }
        double time() const { return time_; }
        double scale_factor() const { return a_; }
        std::uint64_t steps() const { return steps_; }
        const std::vector<Real> & external_density() const { return external_; }
        // Source is absolute comoving mass density, not delta or a mass fraction. It is frozen for one advance.
        void set_external_density(const std::vector<Real> & rho) {
            // Empty vectors on empty FFT slabs still represent a present source; use a global-presence flag below.
            detail::require(rho.size() == nreal(), "[WaveDM] External density layout mismatch");
            bool valid = true;
            for (auto x : rho)
                valid &= x >= 0 && std::isfinite(x);
            detail::require(valid, "[WaveDM] External comoving density must be finite and nonnegative");
            external_ = rho;
            external_enabled_ = true;
            potential_valid_ = false;
        }
        void clear_external_density() {
            external_.clear();
            external_enabled_ = false;
            potential_valid_ = false;
        }
        bool has_external_density() const { return external_enabled_; }
        void set_restart_steps(std::uint64_t steps) {
            detail::same(steps, "[WaveDM] Restart counters differ between ranks");
            steps_ = steps;
        }
        std::array<double, 3> position(size_t i) const {
            const auto ix = coordinate(i);
            return {box_ * ix[0] / mesh(), box_ * ix[1] / mesh(), box_ * ix[2] / mesh()};
        }
        // Serial callback evaluation (outside OpenMP); input errors cannot leave half-initialized live data.
        void initialize(const std::function<Complex(std::array<double, 3>)> & field) {
            check_layout();
            detail::require(bool(field), "[WaveDM] Missing initial wavefunction");
            if (!candidate_->get_grid_status_real())
                inverse(*candidate_);
            bool valid = true;
            try {
                for (size_t i = 0; i < nreal(); ++i) {
                    auto z = field(position(i));
                    candidate_->get_real_grid()[i] = z;
                    valid &= finite(z) && std::isfinite(double(std::norm(z)));
                }
            } catch (const std::exception &) {
                valid = false;
            }
            detail::require(valid, "[WaveDM] Invalid initial wavefunction");
            detail::copy_values(*wave_, *candidate_);
            potential_valid_ = false;
        }
        void initialize_density_phase(const std::function<double(std::array<double, 3>)> & rho,
                                      const std::function<double(std::array<double, 3>)> & phase) {
            detail::require(bool(rho) && bool(phase), "[WaveDM] Missing density/phase callback");
            initialize([&](auto x) {
                const double r = rho(x), p = phase(x);
                if (r < 0 || !std::isfinite(r) || !std::isfinite(p))
                    throw std::invalid_argument("[WaveDM] Invalid density or phase");
                return Complex(Real(std::sqrt(r) * std::cos(p)), Real(std::sqrt(r) * std::sin(p)));
            });
        }
        // peculiar velocity u=grad(sigma); theta=a*sigma/eta for single-stream periodic initial data.
        void initialize_density_velocity_potential(const std::function<double(std::array<double, 3>)> & rho,
                                                   const std::function<double(std::array<double, 3>)> & sigma) {
            detail::require(bool(sigma), "[WaveDM] Missing peculiar-velocity potential");
            initialize_density_phase(rho, [&](auto x) { return (a_ / physics_.eta) * sigma(x); });
        }
        // Candidate-buffer transaction: validation/overflow failure leaves wave, time, a and step count unchanged.
        void advance(const TimeFactors & factors) {
            for (double x : {factors.dt, factors.kinetic, factors.potential, factors.a_begin, factors.a_end})
                detail::same(x, "[WaveDM] Step factors differ between ranks");
            detail::same(int(external_enabled_), "[WaveDM] External-source state differs between ranks");
            const double dt = factors.dt;
            detail::require(
                std::isfinite(dt) && std::isfinite(factors.kinetic) && std::isfinite(factors.potential) &&
                    factors.a_begin == a_ && factors.a_end > 0 && std::isfinite(factors.a_end) &&
                    std::isfinite(time_ + dt) &&
                    (dt == 0 ? (factors.kinetic == 0 && factors.potential == 0 && factors.a_begin == factors.a_end) :
                               ((factors.kinetic > 0) == (dt > 0) && (factors.potential > 0) == (dt > 0) &&
                                factors.kinetic != 0 && factors.potential != 0)) &&
                    steps_ != std::numeric_limits<std::uint64_t>::max(),
                "[WaveDM] Invalid step factors or exhausted counter");
            check_wave(*wave_);
            if (dt == 0)
                return;
            const double alpha = .5 * physics_.eta * factors.kinetic;
            detail::require(std::isfinite(alpha * k2max_), "[WaveDM] Kinetic phase overflows");
            // The potential of the wave at the start of the step is the one at the end of the previous step (unless
            // the wave or the source changed in between)
            const bool reuse_potential = potential_valid_;
            potential_valid_ = false;
            detail::copy_values(*candidate_, *wave_);
            if (not reuse_potential)
                solve_potential(*candidate_);
            phase_kick(*candidate_, .5 * factors.potential);
            forward(*candidate_);
            kinetic_drift(*candidate_, alpha);
            inverse(*candidate_);
            check_wave(*candidate_);
            solve_potential(*candidate_);
            phase_kick(*candidate_, .5 * factors.potential);
            check_wave(*candidate_);
            wave_.swap(candidate_);
            potential_valid_ = true;
            time_ += dt;
            a_ = factors.a_end;
            ++steps_;
        }
        void step(double dt) { advance(TimeFactors::fixed(dt, a_)); }
        void step_scale_factor(double next_a, const std::function<double(double)> & H, int panels = 4) {
            advance(expansion_factors(a_, next_a, H, panels));
        }
        // Zero-mean comoving potential U=a*Phi, in code velocity^2. Recomputed from the live wave/source.
        const std::vector<double> & potential() {
            check_wave(*wave_);
            if (not potential_valid_)
                solve_potential(*wave_);
            potential_valid_ = true;
            return potential_;
        }
        // Full-complex odd derivatives. Even-N Nyquist derivatives are zero, as for FML's real PM gradients.
        std::array<std::vector<Real>, 3> gravity_acceleration() {
            const auto & U = potential();
            const double max_derivative = ((2 * std::acos(-1.) * (mesh() / 2) / box_) / a_) / a_ / a_;
            detail::require(std::isfinite(max_derivative) && max_derivative <= std::numeric_limits<Real>::max(),
                            "[WaveDM] Comoving acceleration conversion overflows");
            detail::copy_values(*scratch_, *wave_);
            for (size_t i = 0; i < U.size(); ++i)
                scratch_->get_real_grid()[i] = Complex(Real(U[i]), 0);
            forward(*scratch_);
            detail::copy_values(*candidate_, *scratch_);
            std::array<std::vector<Real>, 3> result;
            for (int d = 0; d < 3; ++d) {
                detail::copy_values(*scratch_, *candidate_);
                for (size_t i = 0; i < nfourier(); ++i) {
                    const auto mode = scratch_->get_fourier_integer_wavevector_from_index(FML::IndexIntType(i));
                    double k = 2 * std::acos(-1.) * mode[d] / box_;
                    if (mesh() % 2 == 0 && std::abs(mode[d]) == mesh() / 2)
                        k = 0;
                    scratch_->get_fourier_grid()[i] *=
                        Complex(0, Real(((-k / a_) / a_) / a_)); // Comoving acceleration -grad(U)/a^3.
                }
                inverse(*scratch_);
                result[d].resize(U.size());
                bool valid = true;
                for (size_t i = 0; i < U.size(); ++i) {
                    result[d][i] = scratch_->get_real_grid()[i].real();
                    valid &= std::isfinite(result[d][i]);
                }
                detail::require(valid, "[WaveDM] Nonfinite comoving acceleration");
            }
            return result;
        }
        Diagnostics diagnostics(double tail_start = .8) {
            detail::same(tail_start, "[WaveDM] Tail cutoff differs between ranks");
            detail::require(tail_start > 0 && tail_start <= 1 && std::isfinite(tail_start),
                            "[WaveDM] Invalid spectral tail cutoff");
            check_wave(*wave_);
            if (not potential_valid_)
                solve_potential(*wave_);
            potential_valid_ = true;
            Diagnostics result;
            double sourceW = 0;
            for (size_t i = 0; i < nreal(); ++i) {
                const double rho = double(std::norm(wave_->get_real_grid()[i]));
                result.mass += rho;
                result.max_density = std::max(result.max_density, rho);
                sourceW += (rho + (external_.empty() ? 0 : double(external_[i]))) * potential_[i];
            }
            FML::SumOverTasks(&result.mass);
            FML::MaxOverTasks(&result.max_density);
            FML::SumOverTasks(&sourceW);
            result.mass *= volume_element();
            result.potential = .5 / a_ * volume_element() * sourceW;
            detail::copy_values(*scratch_, *wave_);
            forward(*scratch_);
            double kinetic = 0, tail = 0, norm = 0;
            for (size_t i = 0; i < nfourier(); ++i) {
                const double power = double(std::norm(scratch_->get_fourier_grid()[i]));
                kinetic += k2_[i] * power;
                norm += power;
                const auto k = scratch_->get_fourier_integer_wavevector_from_index(FML::IndexIntType(i));
                bool high = false;
                for (int d = 0; d < 3; ++d)
                    high |= std::abs(k[d]) >= tail_start * mesh() / 2.;
                if (high)
                    tail += power;
            }
            FML::SumOverTasks(&kinetic);
            FML::SumOverTasks(&tail);
            FML::SumOverTasks(&norm);
            result.kinetic = .5 * std::pow(physics_.eta / a_, 2) * std::pow(box_, 3) * kinetic;
            result.spectral_tail_fraction = norm > 0 ? tail / norm : 0;
            detail::require(std::isfinite(result.mass) && std::isfinite(result.kinetic) &&
                                std::isfinite(result.potential) && std::isfinite(result.spectral_tail_fraction),
                            "[WaveDM] Diagnostic sum overflows");
            return result;
        }
        // Conservative phase-based step suggestion, NOT an error estimate; includes the entire represented k cube.
        double suggested_timestep(double phase_limit = .5) {
            detail::same(phase_limit, "[WaveDM] Phase limit differs between ranks");
            detail::require(phase_limit > 0 && std::isfinite(phase_limit), "[WaveDM] Invalid phase limit");
            const auto & U = potential();
            double maxU = 0;
            for (double u : U)
                maxU = std::max(maxU, std::abs(u));
            FML::MaxOverTasks(&maxU);
            const double kmax2 = 3 * std::pow(2 * std::acos(-1.) * (mesh() / 2) / box_, 2);
            const double frequency = std::max(.5 * physics_.eta * kmax2 / a_ / a_, maxU / physics_.eta / a_);
            return phase_limit / frequency;
        }
        MemoryReport memory() const {
            return {wave_->allocated_storage_bytes() + candidate_->allocated_storage_bytes() +
                        scratch_->allocated_storage_bytes(),
                    (k2_.capacity() + potential_.capacity()) * sizeof(double),
                    external_.capacity() * sizeof(Real)};
        }
        std::uint64_t fft_count() const { return transforms_; }

      private:
        bool external_enabled_{false};
        // potential_ is that of the current wave (and source)
        bool potential_valid_{false};
        // The largest k^2 of the grid (for the kinetic phase overflow check)
        double k2max_{0.0};
    };

    /// The solver on FFTW-MPI slabs
    using Solver = SolverT<>;
} // namespace FML::WAVEDM
#endif
