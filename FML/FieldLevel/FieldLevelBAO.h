#ifndef FML_FIELD_LEVEL_BAO_HEADER
#define FML_FIELD_LEVEL_BAO_HEADER
// Prototype of field-level inference of the BAO scale (in the spirit of arXiv:2407.01524): the parameters are the
// white-noise initial field s (one unit Gaussian per cell) and the BAO scale beta = r_BAO / r_fid. The forward model is
// either linear (pred = b delta_lin) or 1LPT (Zel'dovich particles assigned with CIC, pred = b delta_1LPT), compared to
// the data mode by mode below a sharp cutoff kmax with a Gaussian likelihood of constant noise power P_eps. The
// posterior is sampled jointly in (s, beta) with HMC (FML/MCMC/HMC.h) using hand-written adjoint gradients and a
// Fourier-diagonal mass matrix.
//
// Serial (FFTWGrid without MPI; OpenMP over cells where it is free of races). See README.md for the conventions.
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MCMC/HMC.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace FML::FIELDLEVEL {
    using Complex = std::complex<double>;
    using Field = std::vector<double>; // compact real field, index (ix * N + iy) * N + iz

    /// No-wiggle transfer function of Eisenstein & Hu (1998), k in h/Mpc.
    inline double
    eisenstein_hu_nowiggle_transfer(double k, double Omega_m = .31, double Omega_b = .049, double h = .68) {
        const double theta = 2.7255 / 2.7, omh2 = Omega_m * h * h, obh2 = Omega_b * h * h, fb = Omega_b / Omega_m;
        const double s = 44.5 * std::log(9.83 / omh2) / std::sqrt(1 + 10 * std::pow(obh2, .75)); // Mpc
        const double alpha = 1 - .328 * std::log(431 * omh2) * fb + .38 * std::log(22.3 * omh2) * fb * fb;
        const double gamma = Omega_m * h * (alpha + (1 - alpha) / (1 + std::pow(.43 * k * h * s, 4)));
        const double q = k * theta * theta / gamma;
        const double L0 = std::log(2 * M_E + 1.8 * q), C0 = 14.2 + 731 / (1 + 62.5 * q);
        return L0 / (L0 + C0 * q * q);
    }

    /// Toy BAO spectrum P(k, beta) = P_smooth(k) [1 + A sin(k beta r_fid + phase) exp(-k / k_damp)]; k in h/Mpc.
    struct BAOSpectrum {
        std::function<double(double)> smooth; // P_smooth(k) [(Mpc/h)^3]
        double amplitude{.08};
        double r_fid{105.}; // Mpc/h
        double k_damp{.25}; // h/Mpc
        double phase{0.};

        /// Default smooth part: EH no-wiggle with n_s, normalised to P_smooth(0.1 h/Mpc) = p01.
        static BAOSpectrum eisenstein_hu(double p01 = 5000., double n_s = .965) {
            BAOSpectrum spectrum;
            const double norm = p01 / (std::pow(.1, n_s) * std::pow(eisenstein_hu_nowiggle_transfer(.1), 2));
            spectrum.smooth = [=](double k) {
                return norm * std::pow(k, n_s) * std::pow(eisenstein_hu_nowiggle_transfer(k), 2);
            };
            return spectrum;
        }
        double wiggle(double k, double beta) const {
            return amplitude * std::sin(k * beta * r_fid + phase) * std::exp(-k / k_damp);
        }
        double operator()(double k, double beta) const { return smooth(k) * (1 + wiggle(k, beta)); }
        /// d ln P / d beta
        double dlnP_dbeta(double k, double beta) const {
            const double dw = amplitude * k * r_fid * std::cos(k * beta * r_fid + phase) * std::exp(-k / k_damp);
            return dw / (1 + wiggle(k, beta));
        }
    };

    /// Linear: pred = b delta_lin.
    /// LPT1:   pred = b delta_CIC of Zel'dovich-displaced particles (Eulerian linear bias on the matter field).
    /// LPT:    1LPT or 2LPT particles (lpt_order) carrying Lagrangian bias weights
    ///         w = 1 + b1L delta + b2L (delta^2 - <delta^2>) + bG2L G2,  b1L = bias - 1, G2 = (d_i d_j phi)^2 -
    ///         delta^2, optionally displaced into redshift space along z: x_z += f (Psi1_z + 2 Psi2_z); pred = delta of
    ///         the weighted CIC density. On large scales pred -> (b + f mu^2) delta_lin.
    enum class Model { Linear, LPT1, LPT };

    /// Mass assignment of the LPT particles to the density grid.
    /// Smooth: a compact smooth kernel (an exponential-of-semicircle shape tapered by 1 - z^2, kernel_width nodes)
    ///         followed by deconvolution of its Fourier transform, so the predicted modes are the Fourier transform of
    ///         the particle distribution, (1/N_p) sum_p exp(-i k.x_p), to ~1e-6 for kmax well below the Nyquist
    ///         frequency of the density grid, with a continuous gradient.
    /// CIC:    cloud-in-cell without deconvolution (cheaper; the CIC window is part of the model). CIC is linear in a
    ///         displacement smaller than half a cell, so it misses the mode coupling of small displacements.
    enum class Assignment { Smooth, CIC };

    struct Settings {
        int nmesh{32};        // grid of the initial field (and of the particles for 1LPT)
        int nmesh_density{0}; // density grid of the LPT models; 0 means 2 nmesh
        Assignment assignment{Assignment::Smooth};
        int kernel_width{6};       // Smooth assignment: support in density-grid cells (even, 4..8)
        double boxsize{1000.};     // Mpc/h
        double kmax{.09};          // h/Mpc; cutoff of both the initial field and the likelihood, must be below Nyquist
        double bias{1.5};          // linear bias b
        double noise_power{1000.}; // P_eps [(Mpc/h)^3]
        Model model{Model::Linear};
        double beta_min{.8}, beta_max{1.2}; // uniform prior on beta
        // Model::LPT only
        int lpt_order{2};      // 1 or 2
        double b2{0};          // Lagrangian b2 (weight b2 (delta^2 - <delta^2>))
        double bG2{0};         // Lagrangian tidal bias (weight bG2 G2)
        double growth_rate{0}; // f; 0 = real space, otherwise redshift space along the z axis
        // Parametrisation of the sampled field x: s_k = (T(beta_ref) / T(beta))^c x_k, with s the unit white noise.
        // c = 0 samples s itself (non-centred), c = 1 samples delta / T(beta_ref) (centred). A negative value picks per
        // mode c = S / (S + N) = b^2 P / (b^2 P + P_eps), which decorrelates beta from x in both regimes.
        double centering{-1};
        double beta_ref{1.};
    };

    /// Forward models, likelihood and adjoint gradients.
    /// FFT conventions (FFTWGrid): r2c divides by N^3, c2r is the plain sum, so <|delta_k|^2> = P(k) / V.
    /// With the white noise s (prior 1/2 sum s^2) and delta_k = T(beta) s_k, the sampled field is x with
    /// s_k = r_k(beta) x_k (see Settings::centering), and the potential is
    ///   U(x, beta) = 1/2 sum_cells s^2 - ln det(ds/dx) + 1/2 sum_{all k, 0<|k|<kmax} |d_k - pred_k|^2 / sigma^2
    /// with sigma^2 = P_eps / V; the sum over the full (Hermitian) set of modes is the exact Gaussian likelihood of a
    /// real field. Every Field argument below is the sampled field x unless the name says white.
    class FieldLevelBAO {
      public:
        struct Mode {
            int n[3];                  // integer wavevector
            double k;                  // |k| in h/Mpc
            double weight;             // 1 or 2: multiplicity in the half-complex layout
            std::size_t fourier_index; // index in the half-complex array of the nmesh grid
            std::size_t density_index; // index in the half-complex array of the nmesh_density grid
            double centering;          // c of this mode
            double t_ref;              // T(k, beta_ref)
            Complex density_factor{1}; // applied to the density-grid mode (offset phase, kernel deconvolution)
        };

        FieldLevelBAO(const Settings & settings, BAOSpectrum spectrum)
            : s_(settings), spectrum_(std::move(spectrum)), grid_(settings.nmesh) {
            if (s_.nmesh_density == 0)
                s_.nmesh_density = 2 * s_.nmesh;
            const double kf = 2 * M_PI / s_.boxsize;
            if (s_.assignment == Assignment::Smooth &&
                !(s_.kernel_width >= 4 && s_.kernel_width <= 8 && s_.kernel_width % 2 == 0))
                throw std::invalid_argument("[FieldLevelBAO] kernel_width must be 4, 6 or 8");
            if (s_.model == Model::LPT && !(s_.lpt_order == 1 || s_.lpt_order == 2))
                throw std::invalid_argument("[FieldLevelBAO] lpt_order must be 1 or 2");
            if (!(s_.nmesh >= 4 && s_.centering <= 1 && s_.beta_ref >= s_.beta_min && s_.beta_ref <= s_.beta_max &&
                  s_.boxsize > 0 && s_.kmax > 0 && s_.noise_power > 0 && s_.bias != 0 && s_.beta_min < s_.beta_max &&
                  s_.beta_min > 0 && spectrum_.smooth))
                throw std::invalid_argument("[FieldLevelBAO] Invalid settings");
            if (s_.kmax >= kf * (s_.nmesh / 2) || s_.kmax >= kf * (s_.nmesh_density / 2))
                throw std::invalid_argument("[FieldLevelBAO] kmax must lie below the Nyquist frequency of both grids");
            if (s_.model != Model::Linear)
                density_ = FML::GRID::FFTWGrid<3>(s_.nmesh_density);
            const int N = s_.nmesh;
            ncells_ = std::size_t(N) * N * N;
            volume_ = std::pow(s_.boxsize, 3);
            sigma2_ = s_.noise_power / volume_;
            real_index_.resize(ncells_);
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz < N; ++iz)
                        real_index_[cell(ix, iy, iz)] = grid_.get_index_real({ix, iy, iz});
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz <= N / 2; ++iz) {
                        const int n[3] = {ix <= N / 2 ? ix : ix - N, iy <= N / 2 ? iy : iy - N, iz};
                        const double k = kf * std::sqrt(double(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]));
                        if (k == 0 || k >= s_.kmax)
                            continue;
                        Mode m{{n[0], n[1], n[2]},
                               k,
                               double(FML::GRID::fourier_mode_multiplicity(iz, N)),
                               std::size_t(grid_.get_index_fourier({ix, iy, iz})),
                               0,
                               0,
                               0};
                        const double S = std::pow(linear_amplitude(n), 2) * spectrum_(k, s_.beta_ref);
                        m.centering = s_.centering < 0 ? S / (S + s_.noise_power) : s_.centering;
                        if (s_.model != Model::Linear) {
                            const int Nd = s_.nmesh_density;
                            m.density_index = density_.get_index_fourier({(n[0] + Nd) % Nd, (n[1] + Nd) % Nd, iz});
                        }
                        m.t_ref = transfer(m, s_.beta_ref);
                        modes_.push_back(m);
                    }
            // Per-mode factor of the density modes: undo the lattice offset (CIC) or the kernel (Smooth)
            if (s_.model != Model::Linear) {
                const int Nd = s_.nmesh_density;
                if (std::size_t(density_.get_index_real({Nd - 1, Nd - 2, 3})) != density_row(Nd - 1, Nd - 2) + 3)
                    throw std::logic_error("[FieldLevelBAO] Unexpected FFTWGrid real-space layout");
                std::vector<double> kernel_ft(Nd / 2 + 1, 1.);
                if (s_.assignment == Assignment::Smooth) {
                    const double h = .5 * s_.kernel_width;
                    const int nq = 4000;
                    for (int n = 0; n <= Nd / 2; ++n) {
                        const double omega = 2 * M_PI * n / Nd;
                        double sum = 0; // Simpson on t in [0, h] of 2 phi(t / h) cos(omega t)
                        for (int j = 0; j <= nq; ++j) {
                            const double t = h * j / nq, wj = (j == 0 || j == nq) ? 1 : (j % 2 ? 4 : 2);
                            sum += wj * kernel(t / h) * std::cos(omega * t);
                        }
                        kernel_ft[n] = 2 * sum * h / nq / 3;
                    }
                }
                for (auto & m : modes_) {
                    const double phase = 2 * M_PI * lattice_offset() * double(m.n[0] + m.n[1] + m.n[2]);
                    m.density_factor =
                        Complex(std::cos(phase), std::sin(phase)) /
                        (kernel_ft[std::abs(m.n[0])] * kernel_ft[std::abs(m.n[1])] * kernel_ft[std::abs(m.n[2])]);
                }
            }
            // Every half-complex index of the nmesh grid with its integer wavevector (for operators on all modes)
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz <= N / 2; ++iz)
                        all_modes_.push_back({std::size_t(grid_.get_index_fourier({ix, iy, iz})),
                                              {ix <= N / 2 ? ix : ix - N, iy <= N / 2 ? iy : iy - N, iz}});
        }

        /// Large-scale amplitude of the prediction relative to delta_lin: b, or b + f mu^2 in redshift space
        double linear_amplitude(const int n[3]) const {
            if (s_.model != Model::LPT || s_.growth_rate == 0)
                return s_.bias;
            const double n2 = double(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            return s_.bias + s_.growth_rate * double(n[2] * n[2]) / n2;
        }

        const Settings & settings() const { return s_; }
        const std::vector<Mode> & modes() const { return modes_; }
        const BAOSpectrum & spectrum() const { return spectrum_; }
        std::size_t ncells() const { return ncells_; }
        double noise_variance() const { return sigma2_; }

        /// T(k, beta) such that delta_k = T s_k with s_k = r2c(s)
        double transfer(const Mode & m, double beta) const {
            return std::sqrt(spectrum_(m.k, beta) * double(ncells_) / volume_);
        }

        /// r = s_k / x_k and the effective transfer T r applied to x, for a mode
        double ratio(const Mode & m, double beta) const {
            return m.centering == 0 ? 1. : std::pow(m.t_ref / transfer(m, beta), m.centering);
        }
        double effective_transfer(const Mode & m, double beta) const { return transfer(m, beta) * ratio(m, beta); }

        /// Conversions between the white noise s and the sampled field x
        Field sampled_from_white(const Field & s, double beta) const {
            std::vector<double> inverse(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                inverse[i] = 1 / ratio(modes_[i], beta);
            return filter(s, inverse);
        }
        Field white_from_sampled(const Field & x, double beta) const {
            std::vector<double> r(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                r[i] = ratio(modes_[i], beta);
            return filter(x, r);
        }

        /// Total particle displacement (box units, one entry per cell of the nmesh grid) of the LPT models
        std::array<Field, 3> displacement(const Field & x, double beta) const {
            if (s_.model == Model::Linear)
                throw std::invalid_argument("[FieldLevelBAO] The linear model has no displacement");
            Workspace w;
            forward(x, beta, w);
            return w.psi;
        }

        /// Forward model: predicted modes (one per entry of modes())
        std::vector<Complex> predict(const Field & s, double beta) const {
            Workspace w;
            return forward(s, beta, w);
        }

        void set_data(std::vector<Complex> data) {
            if (data.size() != modes_.size())
                throw std::invalid_argument("[FieldLevelBAO] Data must have one entry per mode");
            data_ = std::move(data);
        }
        const std::vector<Complex> & data() const { return data_; }

        /// Mock data from the white noise s: forward model plus Gaussian noise of power P_eps (drawn as a real white
        /// noise field, so the modes are Hermitian; data set with set_data must be Hermitian on the iz = 0 plane too)
        std::vector<Complex> make_mock(const Field & white, double beta, std::mt19937_64 & rng) const {
            auto pred = predict(sampled_from_white(white, beta), beta);
            std::normal_distribution<double> gauss;
            const double sigma_cell = std::sqrt(sigma2_ * double(ncells_));
            Field noise(ncells_);
            for (auto & x : noise)
                x = sigma_cell * gauss(rng);
            auto & g = grid_;
            to_grid(noise, g);
            g.fftw_r2c();
            const auto * f = g.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                pred[i] += Complex(f[modes_[i].fourier_index]);
            return pred;
        }

        Field white_noise(std::mt19937_64 & rng) const {
            std::normal_distribution<double> gauss;
            Field s(ncells_);
            for (auto & x : s)
                x = gauss(rng);
            return s;
        }

        /// U = -ln posterior (up to a constant); +infinity outside the beta prior
        double potential(const Field & s, double beta) const {
            if (beta < s_.beta_min || beta > s_.beta_max)
                return HUGE_VAL;
            Workspace w;
            const auto pred = forward(s, beta, w);
            return prior_term(s, w, beta) + likelihood_term(pred);
        }

        /// U and its gradient with respect to s and beta (adjoint method; the cost is about twice a forward model)
        double potential_and_gradient(const Field & s, double beta, Field & grad_s, double & grad_beta) const {
            Workspace w;
            const auto pred = forward(s, beta, w);
            check_data();
            // Gbar = dU/dpred on the full Hermitian array (inner product sum_full Re(conj a b)) = -(d - pred) / sigma^2
            std::vector<Complex> gbar(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                gbar[i] = -(data_[i] - pred[i]) / sigma2_;
            // Q: adjoint of the initial modes s_k on the half-complex nmesh grid, such that
            //   grad_x = c2r(T_eff Q) + prior gradient,   dU/dbeta (likelihood) = N^3 sum_full Re(conj(Q) T_eff' x_k)
            std::vector<Complex> Q(modes_.size());
            const double N3 = double(ncells_);
            if (s_.model == Model::Linear) {
                for (std::size_t i = 0; i < modes_.size(); ++i)
                    Q[i] = s_.bias * gbar[i] / N3;
            } else {
                lpt_adjoint(gbar, w, Q);
            }
            grad_beta = 0;
            auto & g = grid_;
            g.fill_fourier_grid(FML::GRID::ComplexType(0));
            auto * f = g.get_fourier_grid();
            // Prior 1/2 sum s^2 = 1/2 sum x^2 + 1/2 N^3 sum_full (r^2 - 1) |x_k|^2, Jacobian -sum_full ln r, and
            // d ln r / d beta = -c/2 dlnP/dbeta, d ln T_eff / d beta = (1 - c)/2 dlnP/dbeta
            for (std::size_t i = 0; i < modes_.size(); ++i) {
                const auto & m = modes_[i];
                const double r = ratio(m, beta), Teff = transfer(m, beta) * r;
                const double half_dlnP = .5 * spectrum_.dlnP_dbeta(m.k, beta);
                f[m.fourier_index] = FML::GRID::ComplexType(Teff * Q[i] + (r * r - 1) * w.sk[i]);
                grad_beta +=
                    m.weight * N3 * std::real(std::conj(Q[i]) * w.sk[i]) * Teff * (1 - m.centering) * half_dlnP;
                grad_beta -= m.weight * N3 * r * r * std::norm(w.sk[i]) * m.centering * half_dlnP;
                grad_beta += m.weight * m.centering * half_dlnP;
            }
            g.set_grid_status_real(false);
            g.fftw_c2r();
            from_grid(g, grad_s);
            for (std::size_t c = 0; c < ncells_; ++c)
                grad_s[c] += s[c];
            return prior_term(s, w, beta) + likelihood_term(pred);
        }

        /// Linear model only: exact marginal (over s) log likelihood of beta, -1/2 sum_full (|d|^2/C + ln C)
        double linear_marginal_log_likelihood(double beta) const {
            check_data();
            double result = 0;
            for (std::size_t i = 0; i < modes_.size(); ++i) {
                const double C = s_.bias * s_.bias * spectrum_(modes_[i].k, beta) / volume_ + sigma2_;
                result -= .5 * modes_[i].weight * (std::norm(data_[i]) / C + std::log(C));
            }
            return result;
        }

        /// Fourier-diagonal HMC mass: 1 + A^2 P(k, beta) / P_eps inside kmax (A = linear_amplitude), 1 outside (the
        /// exact Hessian of the linear model's potential in s; a good preconditioner for the LPT models on large
        /// scales)
        std::vector<double> mass_spectrum(double beta) const {
            std::vector<double> mass(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                mass[i] =
                    1 + std::pow(linear_amplitude(modes_[i].n), 2) * spectrum_(modes_[i].k, beta) / s_.noise_power;
            return mass;
        }

        /// Apply a Fourier-diagonal filter to a real field: multiply mode i of modes() by per_mode[i], keep all others
        Field filter(const Field & x, const std::vector<double> & per_mode) const {
            auto & g = grid_;
            to_grid(x, g);
            g.fftw_r2c();
            auto * f = g.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                f[modes_[i].fourier_index] *= FML::GRID::FloatType(per_mode[i]);
            g.fftw_c2r();
            Field y(ncells_);
            from_grid(g, y);
            return y;
        }

      private:
        struct Workspace {
            std::vector<Complex> sk;  // r2c(s) on the modes
            std::array<Field, 3> psi; // total displacement (box units), per particle
            Field delta;              // Lagrangian linear density (Model::LPT)
            std::array<Field, 6> tij; // c2r(k_i k_j / k^2 delta_k): xx, yy, zz, xy, xz, yz (Model::LPT)
            Field weight;             // bias weights (Model::LPT; empty means 1)
            double prefactor{1};      // pred = prefactor * delta_CIC
        };
        struct FullMode {
            std::size_t index;
            int n[3];
        };
        std::vector<FullMode> all_modes_;
        Settings s_;
        BAOSpectrum spectrum_;
        mutable FML::GRID::FFTWGrid<3> grid_;
        mutable FML::GRID::FFTWGrid<3> density_;
        std::vector<Mode> modes_;
        std::vector<FML::IndexIntType> real_index_;
        std::vector<Complex> data_;
        std::size_t ncells_{0};
        double volume_{0}, sigma2_{0};

        std::size_t cell(int ix, int iy, int iz) const { return (std::size_t(ix) * s_.nmesh + iy) * s_.nmesh + iz; }
        void to_grid(const Field & x, FML::GRID::FFTWGrid<3> & g) const {
            auto * r = g.get_real_grid();
            for (std::size_t c = 0; c < ncells_; ++c)
                r[real_index_[c]] = FML::GRID::FloatType(x[c]);
            g.set_grid_status_real(true);
        }
        void from_grid(FML::GRID::FFTWGrid<3> & g, Field & x) const {
            x.resize(ncells_);
            const auto * r = g.get_real_grid();
            for (std::size_t c = 0; c < ncells_; ++c)
                x[c] = r[real_index_[c]];
        }
        void check_data() const {
            if (data_.size() != modes_.size())
                throw std::logic_error("[FieldLevelBAO] No data set");
        }
        double prior_term(const Field & x, const Workspace & w, double beta) const {
            double sum = 0;
            for (double v : x)
                sum += v * v;
            sum *= .5;
            for (std::size_t i = 0; i < modes_.size(); ++i) {
                const double r = ratio(modes_[i], beta);
                sum += modes_[i].weight * (.5 * double(ncells_) * (r * r - 1) * std::norm(w.sk[i]) - std::log(r));
            }
            return sum;
        }
        double likelihood_term(const std::vector<Complex> & pred) const {
            check_data();
            double chi2 = 0;
            for (std::size_t i = 0; i < modes_.size(); ++i)
                chi2 += modes_[i].weight * std::norm(data_[i] - pred[i]);
            return .5 * chi2 / sigma2_;
        }

        std::vector<Complex> forward(const Field & s, double beta, Workspace & w) const {
            if (s.size() != ncells_)
                throw std::invalid_argument("[FieldLevelBAO] Field has the wrong size");
            auto & g = grid_;
            to_grid(s, g);
            g.fftw_r2c();
            const auto * f = g.get_fourier_grid();
            w.sk.resize(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                w.sk[i] = Complex(f[modes_[i].fourier_index]);
            std::vector<Complex> pred(modes_.size());
            if (s_.model == Model::Linear) {
                for (std::size_t i = 0; i < modes_.size(); ++i)
                    pred[i] = s_.bias * effective_transfer(modes_[i], beta) * w.sk[i];
                return pred;
            }
            const int N = s_.nmesh;
            std::vector<Complex> dk(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                dk[i] = effective_transfer(modes_[i], beta) * w.sk[i];
            // Psi1_k = i k / k^2 delta_k (box units, k = 2 pi n)
            for (int c = 0; c < 3; ++c)
                w.psi[c] = from_modes(dk, [&](const Mode & m) { return Complex(0, kc(m.n, c) / k2(m.n)); });
            w.weight.clear();
            w.prefactor = s_.bias;
            if (s_.model == Model::LPT) {
                w.prefactor = 1;
                const bool second = s_.lpt_order == 2, tidal = second || s_.bG2 != 0;
                w.delta = from_modes(dk, [](const Mode &) { return Complex(1); });
                if (tidal)
                    for (int a = 0; a < 6; ++a)
                        w.tij[a] = from_modes(dk, [&](const Mode & m) {
                            return Complex(kc(m.n, pair_[a][0]) * kc(m.n, pair_[a][1]) / k2(m.n));
                        });
                // S2 = 1/2 (delta^2 - t_ij t_ij) = -G2 / 2;  Psi2 = (3/7) D[S2] with D_c = c2r(i k_c / k^2 r2c(.))
                Field G2;
                if (tidal) {
                    G2.resize(ncells_);
                    for (std::size_t p = 0; p < ncells_; ++p) {
                        double tt = 0;
                        for (int a = 0; a < 6; ++a)
                            tt += (a < 3 ? 1 : 2) * w.tij[a][p] * w.tij[a][p];
                        G2[p] = tt - w.delta[p] * w.delta[p];
                    }
                }
                if (second) {
                    Field S2(ncells_);
                    for (std::size_t p = 0; p < ncells_; ++p)
                        S2[p] = -.5 * G2[p];
                    for (int c = 0; c < 3; ++c) {
                        const auto psi2 = gradient_full(S2, c, +1);
                        const double factor = 3. / 7 * (1 + (c == 2 ? 2 * s_.growth_rate : 0));
                        for (std::size_t p = 0; p < ncells_; ++p)
                            w.psi[c][p] = w.psi[c][p] * (1 + (c == 2 ? s_.growth_rate : 0)) + factor * psi2[p];
                    }
                } else if (s_.growth_rate != 0) {
                    for (auto & v : w.psi[2])
                        v *= 1 + s_.growth_rate;
                }
                const double b1L = s_.bias - 1;
                double mean_d2 = 0;
                for (double v : w.delta)
                    mean_d2 += v * v;
                mean_d2 /= double(ncells_);
                w.weight.resize(ncells_);
                for (std::size_t p = 0; p < ncells_; ++p)
                    w.weight[p] = 1 + b1L * w.delta[p] + s_.b2 * (w.delta[p] * w.delta[p] - mean_d2) +
                                  (tidal ? s_.bG2 * G2[p] : 0.);
            }
            const int Nd = s_.nmesh_density;
            auto & d = density_;
            d.fill_real_grid(FML::GRID::FloatType(0));
            auto * rho = d.get_real_grid();
            const double fill = double(std::size_t(Nd) * Nd * Nd) / double(ncells_);
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz < N; ++iz) {
                        const auto p = cell(ix, iy, iz);
                        const double mass = fill * (w.weight.empty() ? 1. : w.weight[p]);
                        Taps t[3];
                        particle_taps(ix, iy, iz, w, p, t);
                        for (int a = 0; a < t[0].n; ++a)
                            for (int b = 0; b < t[1].n; ++b) {
                                const double wab = mass * t[0].w[a] * t[1].w[b];
                                auto * row = rho + density_row(t[0].node[a], t[1].node[b]);
                                for (int e = 0; e < t[2].n; ++e)
                                    row[t[2].node[e]] += FML::GRID::FloatType(wab * t[2].w[e]);
                            }
                    }
            d.fftw_r2c(); // the mean (k = 0) is never used, so delta = rho - 1 is implicit
            const auto * fd = d.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                pred[i] = w.prefactor * modes_[i].density_factor * Complex(fd[modes_[i].density_index]);
            return pred;
        }

        static constexpr int pair_[6][2] = {{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}};
        static double kc(const int n[3], int c) { return 2 * M_PI * n[c]; }
        static double k2(const int n[3]) { return 4 * M_PI * M_PI * double(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); }

        // c2r of coefficients on the modes (times a per-mode kernel); all other modes zero
        template <class Kernel>
        Field from_modes(const std::vector<Complex> & coeff, Kernel kernel) const {
            auto & g = grid_;
            g.fill_fourier_grid(FML::GRID::ComplexType(0));
            auto * f = g.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                f[modes_[i].fourier_index] = FML::GRID::ComplexType(kernel(modes_[i]) * coeff[i]);
            g.set_grid_status_real(false);
            g.fftw_c2r();
            Field y;
            from_grid(g, y);
            return y;
        }
        // Q += conj(kernel) r2c(field) on the modes
        template <class Kernel>
        void add_adjoint(const Field & field, Kernel kernel, std::vector<Complex> & Q) const {
            auto & g = grid_;
            to_grid(field, g);
            g.fftw_r2c();
            const auto * f = g.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                Q[i] += std::conj(kernel(modes_[i])) * Complex(f[modes_[i].fourier_index]);
        }
        // c2r(sign i k_c / k^2 r2c(field)) over all modes; k = 0 and modes on a Nyquist plane are zeroed so the
        // operator is real and its adjoint is the same with sign -> -sign
        Field gradient_full(const Field & field, int c, int sign) const {
            auto & g = grid_;
            to_grid(field, g);
            g.fftw_r2c();
            auto * f = g.get_fourier_grid();
            const int half = s_.nmesh / 2;
            for (const auto & m : all_modes_) {
                const bool zero = (m.n[0] == 0 && m.n[1] == 0 && m.n[2] == 0) || std::abs(m.n[0]) == half ||
                                  std::abs(m.n[1]) == half || m.n[2] == half;
                f[m.index] =
                    zero ? FML::GRID::ComplexType(0) :
                           f[m.index] * FML::GRID::ComplexType(0, FML::GRID::FloatType(sign * kc(m.n, c) / k2(m.n)));
            }
            g.fftw_c2r();
            Field y;
            from_grid(g, y);
            return y;
        }

        // CIC: particles start half a density cell off the density-grid nodes. On a node, CIC responds to a
        // displacement through |Psi| (mass leaves the node to whichever side it moves), which is not differentiable at
        // Psi = 0 and generates harmonics at linear order in the amplitude (~40% of the signal at k dx_d ~ 0.4 in a
        // plane-wave test). The translation of the field by the offset s is undone with exp(i k.s) on the modes.
        double lattice_offset() const { return s_.assignment == Assignment::CIC ? .5 / s_.nmesh_density : 0.; }

        // Smooth kernel on z in [-1, 1]: (1 - z^2) exp(beta (sqrt(1 - z^2) - 1)), beta = 2.3 width; continuous with a
        // continuous derivative at the support edge
        double kernel(double z) const {
            const double r = 1 - z * z;
            return r <= 0 ? 0. : r * std::exp(2.3 * s_.kernel_width * (std::sqrt(r) - 1));
        }
        double kernel_derivative(double z) const {
            const double r = 1 - z * z;
            if (r <= 0)
                return 0.;
            const double beta = 2.3 * s_.kernel_width, sr = std::sqrt(r);
            return -z * std::exp(beta * (sr - 1)) * (2 + beta * sr);
        }

        // Assignment weights along one axis: nodes, weights and their derivatives with respect to the position u in
        // density-grid cells
        struct Taps {
            int n{0};
            int node[9];
            double w[9], dw[9];
        };
        // Offset of the row (i, j, 0) in the padded real density grid (checked against FFTWGrid in the constructor)
        std::size_t density_row(int i, int j) const {
            const int Nd = s_.nmesh_density;
            return (std::size_t(i) * Nd + j) * std::size_t(2 * (Nd / 2 + 1));
        }
        void axis_taps(double u, Taps & t) const {
            const int Nd = s_.nmesh_density;
            u -= Nd * std::floor(u / Nd);
            if (s_.assignment == Assignment::CIC) {
                const double fl = std::floor(u);
                const int i0 = int(fl) % Nd;
                const double f = u - fl;
                t.n = 2;
                t.node[0] = i0, t.node[1] = (i0 + 1) % Nd;
                t.w[0] = 1 - f, t.w[1] = f;
                t.dw[0] = -1, t.dw[1] = 1;
                return;
            }
            const double h = .5 * s_.kernel_width;
            const int start = int(std::floor(u - h)) + 1;
            t.n = s_.kernel_width + 1;
            for (int j = 0; j < t.n; ++j) {
                const int node = start + j;
                const double z = (node - u) / h;
                t.node[j] = ((node % Nd) + Nd) % Nd;
                t.w[j] = kernel(z);
                t.dw[j] = -kernel_derivative(z) / h;
            }
        }
        void particle_taps(int ix, int iy, int iz, const Workspace & w, std::size_t p, Taps t[3]) const {
            const int N = s_.nmesh, Nd = s_.nmesh_density;
            const int idx[3] = {ix, iy, iz};
            for (int c = 0; c < 3; ++c)
                axis_taps((double(idx[c]) / N + lattice_offset() + w.psi[c][p]) * Nd, t[c]);
        }

        // Adjoint of the LPT models: from Gbar on the modes to Q on the modes
        void lpt_adjoint(const std::vector<Complex> & gbar, const Workspace & w, std::vector<Complex> & Q) const {
            const int N = s_.nmesh, Nd = s_.nmesh_density;
            const double Nd3 = double(std::size_t(Nd) * Nd * Nd);
            // delta_d-bar = A^T (prefactor Gbar) = c2r(prefactor Gbar) / Nd^3 on the density grid
            auto & d = density_;
            d.fill_fourier_grid(FML::GRID::ComplexType(0));
            auto * fd = d.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                fd[modes_[i].density_index] =
                    FML::GRID::ComplexType(w.prefactor * std::conj(modes_[i].density_factor) * gbar[i] / Nd3);
            d.set_grid_status_real(false);
            d.fftw_c2r();
            const auto * dbar = d.get_real_grid();
            // Per particle: xbar = fill w sum_y dbar_y dW(x - y)/dx (box units), wbar = fill sum_y dbar_y W(x - y)
            const double fill = Nd3 / double(ncells_);
            const bool weighted = !w.weight.empty();
            std::array<Field, 3> xbar;
            for (auto & x : xbar)
                x.assign(ncells_, 0);
            Field wbar(weighted ? ncells_ : 0);
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz < N; ++iz) {
                        const auto p = cell(ix, iy, iz);
                        Taps t[3];
                        particle_taps(ix, iy, iz, w, p, t);
                        double gx[3] = {0, 0, 0}, interp = 0;
                        for (int a = 0; a < t[0].n; ++a)
                            for (int b = 0; b < t[1].n; ++b) {
                                const auto * row = dbar + density_row(t[0].node[a], t[1].node[b]);
                                double s0 = 0, s2 = 0; // sum_e v w_e and sum_e v dw_e
                                for (int e = 0; e < t[2].n; ++e) {
                                    const double v = row[t[2].node[e]];
                                    s0 += v * t[2].w[e];
                                    s2 += v * t[2].dw[e];
                                }
                                gx[0] += t[0].dw[a] * t[1].w[b] * s0;
                                gx[1] += t[0].w[a] * t[1].dw[b] * s0;
                                gx[2] += t[0].w[a] * t[1].w[b] * s2;
                                interp += t[0].w[a] * t[1].w[b] * s0;
                            }
                        const double mass = weighted ? w.weight[p] : 1.;
                        for (int c = 0; c < 3; ++c)
                            xbar[c][p] = fill * mass * Nd * gx[c];
                        if (weighted)
                            wbar[p] = fill * interp;
                    }
            std::fill(Q.begin(), Q.end(), Complex(0));
            const auto psi1_kernel = [&](int c) {
                return [this, c](const Mode & m) { return Complex(0, kc(m.n, c) / k2(m.n)); };
            };
            if (s_.model != Model::LPT) {
                for (int c = 0; c < 3; ++c)
                    add_adjoint(xbar[c], psi1_kernel(c), Q);
                return;
            }
            // x = q + Psi1 + Psi2 + f zhat (Psi1_z + 2 Psi2_z), Psi2 = (3/7) D[S2], S2 = -G2 / 2
            const double f = s_.growth_rate;
            const bool second = s_.lpt_order == 2, tidal = second || s_.bG2 != 0;
            Field deltabar(ncells_, 0.);
            std::array<Field, 6> tbar;
            if (tidal)
                for (auto & t : tbar)
                    t.assign(ncells_, 0.);
            // G2-bar collects the weight and the 2LPT contributions; G2 = sum t_ij^2 - delta^2
            Field G2bar(tidal ? ncells_ : 0, 0.);
            if (second) {
                Field S2bar(ncells_, 0.);
                for (int c = 0; c < 3; ++c) {
                    Field psi2bar = xbar[c];
                    const double factor = 3. / 7 * (1 + (c == 2 ? 2 * f : 0));
                    const auto back = gradient_full(psi2bar, c, -1);
                    for (std::size_t p = 0; p < ncells_; ++p)
                        S2bar[p] += factor * back[p];
                }
                for (std::size_t p = 0; p < ncells_; ++p)
                    G2bar[p] -= .5 * S2bar[p];
            }
            // Weights w = 1 + b1L delta + b2 (delta^2 - mean delta^2) + bG2 G2
            const double b1L = s_.bias - 1;
            double mean_wbar = 0;
            for (double v : wbar)
                mean_wbar += v;
            mean_wbar /= double(ncells_);
            for (std::size_t p = 0; p < ncells_; ++p) {
                deltabar[p] += wbar[p] * (b1L + 2 * s_.b2 * w.delta[p]) - 2 * s_.b2 * w.delta[p] * mean_wbar;
                if (tidal)
                    G2bar[p] += s_.bG2 * wbar[p];
            }
            if (tidal)
                for (std::size_t p = 0; p < ncells_; ++p) {
                    deltabar[p] -= 2 * G2bar[p] * w.delta[p];
                    for (int a = 0; a < 6; ++a)
                        tbar[a][p] += (a < 3 ? 2 : 4) * G2bar[p] * w.tij[a][p];
                }
            // Psi1-bar: x_z carries (1 + f) Psi1_z
            for (int c = 0; c < 3; ++c) {
                if (c == 2 && f != 0)
                    for (auto & v : xbar[c])
                        v *= 1 + f;
                add_adjoint(xbar[c], psi1_kernel(c), Q);
            }
            add_adjoint(deltabar, [](const Mode &) { return Complex(1); }, Q);
            if (tidal)
                for (int a = 0; a < 6; ++a)
                    add_adjoint(
                        tbar[a],
                        [&](const Mode & m) { return Complex(kc(m.n, pair_[a][0]) * kc(m.n, pair_[a][1]) / k2(m.n)); },
                        Q);
        }
    };

    /// Sample (x, beta) jointly with FML::MCMC::run_hmc. The field block has the Fourier-diagonal mass
    /// mass_spectrum(beta0) applied with FFTs; beta has a diagonal mass 1 / beta_scale^2 (adapted during warmup) and
    /// reflects at the prior edges. The recorded rows are {beta}; result.last holds the final x followed by beta.
    inline FML::MCMC::HMCResult
    sample(const FieldLevelBAO & model, Field x, double beta, FML::MCMC::HMCOptions options, double beta_scale = .02) {
        const auto & st = model.settings();
        if (!(beta > st.beta_min && beta < st.beta_max))
            throw std::invalid_argument("[FieldLevel::sample] Initial beta outside the prior");
        const std::size_t n = model.ncells();
        if (x.size() != n)
            throw std::invalid_argument("[FieldLevel::sample] Field has the wrong size");
        const auto mass = model.mass_spectrum(beta);
        std::vector<double> sqrt_mass(mass.size()), inv_mass(mass.size());
        for (std::size_t i = 0; i < mass.size(); ++i) {
            sqrt_mass[i] = std::sqrt(mass[i]);
            inv_mass[i] = 1 / mass[i];
        }
        FML::MCMC::HMCMass hmc_mass;
        hmc_mass.n_custom = n;
        hmc_mass.apply_sqrt = [&](std::vector<double> & p) { p = model.filter(p, sqrt_mass); };
        hmc_mass.apply_inverse = [&](std::vector<double> & p) { p = model.filter(p, inv_mass); };
        hmc_mass.diagonal = {1 / (beta_scale * beta_scale)};
        options.lower = {st.beta_min};
        options.upper = {st.beta_max};
        Field field(n), grad_field;
        const auto potential = [&](const std::vector<double> & state, std::vector<double> & grad) {
            std::copy(state.begin(), state.begin() + std::ptrdiff_t(n), field.begin());
            double grad_beta = 0;
            const double U = model.potential_and_gradient(field, state[n], grad_field, grad_beta);
            std::copy(grad_field.begin(), grad_field.end(), grad.begin());
            grad[n] = grad_beta;
            return U;
        };
        x.push_back(beta);
        return FML::MCMC::run_hmc(potential, std::move(x), std::move(hmc_mass), options);
    }
} // namespace FML::FIELDLEVEL
#endif
