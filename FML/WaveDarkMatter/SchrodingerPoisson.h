#ifndef FML_WAVE_SCHRODINGER_POISSON_H
#define FML_WAVE_SCHRODINGER_POISSON_H
#include <FML/WaveDarkMatter/TimeFactors.h>
#include <functional>
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
    class Solver {
        std::unique_ptr<ComplexGrid> wave_, candidate_, scratch_;
        Physics physics_;
        double box_, time_, a_;
        std::uint64_t steps_{};
        std::vector<double> k2_, potential_;
        std::vector<Real> external_;
        double volume_element() const { return std::pow(box_ / wave_->mesh(), 3); }
        static bool finite(Complex z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }
        void check_layout() const {
            detail::require(wave_->mesh() == candidate_->mesh() && wave_->mesh() == scratch_->mesh() &&
                                wave_->size() == k2_.size() && wave_->local_start() == candidate_->local_start() &&
                                wave_->local_start() == scratch_->local_start(),
                            "[WaveDM] Mutable wave grid was replaced with an incompatible layout");
        }
        void check_wave(const ComplexGrid & g) const {
            check_layout();
            bool good = !g.spectral();
            for (size_t i = 0; i < g.size(); ++i)
                good &= finite(g.data()[i]) && std::isfinite(double(std::norm(g.data()[i])));
            detail::require(good, "[WaveDM] Need a finite, representable position-space density/wavefunction");
        }
        void solve_potential(const ComplexGrid & wave) {
            if (physics_.G == 0) {
                std::fill(potential_.begin(), potential_.end(), 0);
                return;
            }
            scratch_->copy_values(wave);
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < wave.size(); ++i)
                scratch_->data()[i] =
                    Complex(Real(std::norm(wave.data()[i]) + (external_.empty() ? 0 : external_[i])), 0);
            scratch_->forward();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < wave.size(); ++i) {
                const double factor = k2_[i] == 0 ? 0 : -4 * std::acos(-1.) * physics_.G / k2_[i];
                scratch_->data()[i] *= Real(factor);
            }
            scratch_->inverse();
            bool good = true;
            for (size_t i = 0; i < wave.size(); ++i) {
                potential_[i] = double(scratch_->data()[i].real());
                good &= std::isfinite(potential_[i]);
            }
            detail::require(good, "[WaveDM] Nonfinite periodic potential");
        }
        void phase_kick(ComplexGrid & wave, double weight) {
            bool good = true;
            for (double u : potential_)
                good &= std::isfinite((u / physics_.eta) * weight);
            detail::require(good, "[WaveDM] Potential phase overflows");
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < wave.size(); ++i) {
                const double angle = std::remainder(-(potential_[i] / physics_.eta) * weight, 2 * std::acos(-1.));
                wave.data()[i] *= Complex(Real(std::cos(angle)), Real(std::sin(angle)));
            }
        }

      public:
        Solver(int mesh, double boxsize, Physics physics = {}, double time = 0, double a = 1)
            : physics_(physics), box_(boxsize), time_(time), a_(a) {
            for (double x : {boxsize, physics.eta, physics.G, time, a})
                detail::same(x, "[WaveDM] Solver parameters differ between ranks");
            detail::require(boxsize > 0 && std::isfinite(boxsize) && std::isfinite(boxsize * boxsize * boxsize) &&
                                physics.eta > 0 && std::isfinite(physics.eta) && physics.G >= 0 &&
                                std::isfinite(physics.G) && std::isfinite(time) && a > 0 && std::isfinite(a),
                            "[WaveDM] Invalid units, coupling, box, time or scale factor");
            wave_ = std::make_unique<ComplexGrid>(mesh);
            candidate_ = std::make_unique<ComplexGrid>(mesh);
            scratch_ = std::make_unique<ComplexGrid>(mesh);
            k2_.resize(wave_->size());
            potential_.resize(wave_->size());
            bool valid = true;
            for (size_t i = 0; i < k2_.size(); ++i) {
                const auto mode = wave_->mode(i);
                double k2 = 0;
                for (int d = 0; d < 3; ++d) {
                    const double k = 2 * std::acos(-1.) * mode[d] / box_;
                    k2 += k * k;
                }
                k2_[i] = k2;
                valid &= std::isfinite(k2);
            }
            detail::require(valid && std::isfinite(volume_element()) && volume_element() > 0,
                            "[WaveDM] Unrepresentable spatial scales");
        }
        Solver(const Solver &) = delete;
        Solver & operator=(const Solver &) = delete;
        ComplexGrid & wave() { return *wave_; }
        const ComplexGrid & wave() const { return *wave_; }
        Physics physics() const { return physics_; }
        double boxsize() const { return box_; }
        double time() const { return time_; }
        double scale_factor() const { return a_; }
        std::uint64_t steps() const { return steps_; }
        const std::vector<Real> & external_density() const { return external_; }
        // Source is absolute comoving mass density, not delta or a mass fraction. It is frozen for one advance.
        void set_external_density(const std::vector<Real> & rho) {
            // Empty vectors on empty FFT slabs still represent a present source; use a global-presence flag below.
            detail::require(rho.size() == wave_->size(), "[WaveDM] External density layout mismatch");
            bool valid = true;
            for (auto x : rho)
                valid &= x >= 0 && std::isfinite(x);
            detail::require(valid, "[WaveDM] External comoving density must be finite and nonnegative");
            external_ = rho;
            external_enabled_ = true;
        }
        void clear_external_density() {
            external_.clear();
            external_enabled_ = false;
        }
        bool has_external_density() const { return external_enabled_; }
        void set_restart_steps(std::uint64_t steps) {
            detail::same(steps, "[WaveDM] Restart counters differ between ranks");
            steps_ = steps;
        }
        std::array<double, 3> position(size_t i) const {
            const auto ix = wave_->coordinate(i);
            return {box_ * ix[0] / wave_->mesh(), box_ * ix[1] / wave_->mesh(), box_ * ix[2] / wave_->mesh()};
        }
        // Serial callback evaluation (outside OpenMP); input errors cannot leave half-initialized live data.
        void initialize(const std::function<Complex(std::array<double, 3>)> & field) {
            check_layout();
            detail::require(bool(field), "[WaveDM] Missing initial wavefunction");
            if (candidate_->spectral())
                candidate_->inverse();
            bool valid = true;
            try {
                for (size_t i = 0; i < wave_->size(); ++i) {
                    auto z = field(position(i));
                    candidate_->data()[i] = z;
                    valid &= finite(z) && std::isfinite(double(std::norm(z)));
                }
            } catch (const std::exception &) {
                valid = false;
            }
            detail::require(valid, "[WaveDM] Invalid initial wavefunction");
            wave_->copy_values(*candidate_);
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
            bool valid = true;
            for (double k2 : k2_)
                valid &= std::isfinite((.5 * physics_.eta * factors.kinetic) * k2);
            detail::require(valid, "[WaveDM] Kinetic phase overflows");
            candidate_->copy_values(*wave_);
            solve_potential(*candidate_);
            phase_kick(*candidate_, .5 * factors.potential);
            candidate_->forward();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (size_t i = 0; i < wave_->size(); ++i) {
                const double angle =
                    std::remainder(-(.5 * physics_.eta * factors.kinetic) * k2_[i], 2 * std::acos(-1.));
                candidate_->data()[i] *= Complex(Real(std::cos(angle)), Real(std::sin(angle)));
            }
            candidate_->inverse();
            check_wave(*candidate_);
            solve_potential(*candidate_);
            phase_kick(*candidate_, .5 * factors.potential);
            check_wave(*candidate_);
            wave_.swap(candidate_);
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
            solve_potential(*wave_);
            return potential_;
        }
        // Full-complex odd derivatives. Even-N Nyquist derivatives are zero, as for FML's real PM gradients.
        std::array<std::vector<Real>, 3> gravity_acceleration() {
            const auto & U = potential();
            const double max_derivative = ((2 * std::acos(-1.) * (wave_->mesh() / 2) / box_) / a_) / a_ / a_;
            detail::require(std::isfinite(max_derivative) && max_derivative <= std::numeric_limits<Real>::max(),
                            "[WaveDM] Comoving acceleration conversion overflows");
            scratch_->copy_values(*wave_);
            for (size_t i = 0; i < U.size(); ++i)
                scratch_->data()[i] = Complex(Real(U[i]), 0);
            scratch_->forward();
            candidate_->copy_values(*scratch_);
            std::array<std::vector<Real>, 3> result;
            for (int d = 0; d < 3; ++d) {
                scratch_->copy_values(*candidate_);
                for (size_t i = 0; i < U.size(); ++i) {
                    const auto mode = scratch_->mode(i);
                    double k = 2 * std::acos(-1.) * mode[d] / box_;
                    if (wave_->mesh() % 2 == 0 && std::abs(mode[d]) == wave_->mesh() / 2)
                        k = 0;
                    scratch_->data()[i] *=
                        Complex(0, Real(((-k / a_) / a_) / a_)); // Comoving acceleration -grad(U)/a^3.
                }
                scratch_->inverse();
                result[d].resize(U.size());
                bool valid = true;
                for (size_t i = 0; i < U.size(); ++i) {
                    result[d][i] = scratch_->data()[i].real();
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
            solve_potential(*wave_);
            Diagnostics result;
            double sourceW = 0;
            for (size_t i = 0; i < wave_->size(); ++i) {
                const double rho = double(std::norm(wave_->data()[i]));
                result.mass += rho;
                result.max_density = std::max(result.max_density, rho);
                sourceW += (rho + (external_.empty() ? 0 : double(external_[i]))) * potential_[i];
            }
            FML::SumOverTasks(&result.mass);
            FML::MaxOverTasks(&result.max_density);
            FML::SumOverTasks(&sourceW);
            result.mass *= volume_element();
            result.potential = .5 / a_ * volume_element() * sourceW;
            scratch_->copy_values(*wave_);
            scratch_->forward();
            double kinetic = 0, tail = 0, norm = 0;
            for (size_t i = 0; i < wave_->size(); ++i) {
                const double power = double(std::norm(scratch_->data()[i]));
                kinetic += k2_[i] * power;
                norm += power;
                const auto k = scratch_->mode(i);
                bool high = false;
                for (int d = 0; d < 3; ++d)
                    high |= std::abs(k[d]) >= tail_start * wave_->mesh() / 2.;
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
            const double kmax2 = 3 * std::pow(2 * std::acos(-1.) * (wave_->mesh() / 2) / box_, 2);
            const double frequency = std::max(.5 * physics_.eta * kmax2 / a_ / a_, maxU / physics_.eta / a_);
            return phase_limit / frequency;
        }
        MemoryReport memory() const {
            return {wave_->allocated_bytes() + candidate_->allocated_bytes() + scratch_->allocated_bytes(),
                    (k2_.capacity() + potential_.capacity()) * sizeof(double),
                    external_.capacity() * sizeof(Real)};
        }
        std::uint64_t fft_count() const {
            return wave_->transforms() + candidate_->transforms() + scratch_->transforms();
        }

      private:
        bool external_enabled_{false};
    };
} // namespace FML::WAVEDM
#endif
