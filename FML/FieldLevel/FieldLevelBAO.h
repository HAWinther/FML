#ifndef FML_FIELD_LEVEL_BAO_HEADER
#define FML_FIELD_LEVEL_BAO_HEADER
// Prototype of field-level inference of the BAO scale (in the spirit of arXiv:2407.01524): the parameters are the
// white-noise initial field s (one unit Gaussian per cell) and the BAO scale beta = r_BAO / r_fid. The forward model is
// either linear (pred = b delta_lin) or 1LPT (Zel'dovich particles assigned with CIC, pred = b delta_1LPT), compared to
// the data mode by mode below a sharp cutoff kmax with a Gaussian likelihood of constant noise power P_eps. The posterior
// is sampled jointly in (s, beta) with HMC using hand-written adjoint gradients and a Fourier-diagonal mass matrix.
//
// Serial (FFTWGrid without MPI; OpenMP over cells where it is free of races). See README.md for the conventions.
#include <FML/FFTWGrid/FFTWGrid.h>
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
    inline double eisenstein_hu_nowiggle_transfer(double k, double Omega_m = .31, double Omega_b = .049, double h = .68) {
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
        double r_fid{105.};   // Mpc/h
        double k_damp{.25};   // h/Mpc
        double phase{0.};

        /// Default smooth part: EH no-wiggle with n_s, normalised to P_smooth(0.1 h/Mpc) = p01.
        static BAOSpectrum eisenstein_hu(double p01 = 5000., double n_s = .965) {
            BAOSpectrum spectrum;
            const double norm = p01 / (std::pow(.1, n_s) * std::pow(eisenstein_hu_nowiggle_transfer(.1), 2));
            spectrum.smooth = [=](double k) { return norm * std::pow(k, n_s) * std::pow(eisenstein_hu_nowiggle_transfer(k), 2); };
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

    enum class Model { Linear, LPT1 };

    struct Settings {
        int nmesh{32};             // grid of the initial field (and of the particles for 1LPT)
        int nmesh_density{0};      // CIC grid of the 1LPT density; 0 means 2 nmesh
        double boxsize{1000.};     // Mpc/h
        double kmax{.09};          // h/Mpc; cutoff of both the initial field and the likelihood, must be below Nyquist
        double bias{1.5};          // linear bias b
        double noise_power{1000.}; // P_eps [(Mpc/h)^3]
        Model model{Model::Linear};
        double beta_min{.8}, beta_max{1.2}; // uniform prior on beta
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
            int n[3];       // integer wavevector
            double k;       // |k| in h/Mpc
            double weight;  // 1 or 2: multiplicity in the half-complex layout
            std::size_t fourier_index; // index in the half-complex array of the nmesh grid
            std::size_t density_index; // index in the half-complex array of the nmesh_density grid
            double centering;          // c of this mode
            double t_ref;              // T(k, beta_ref)
        };

        FieldLevelBAO(const Settings & settings, BAOSpectrum spectrum)
            : s_(settings), spectrum_(std::move(spectrum)), grid_(settings.nmesh) {
            if (s_.nmesh_density == 0)
                s_.nmesh_density = 2 * s_.nmesh;
            const double kf = 2 * M_PI / s_.boxsize;
            if (!(s_.nmesh >= 4 && s_.centering <= 1 && s_.beta_ref >= s_.beta_min && s_.beta_ref <= s_.beta_max && s_.boxsize > 0 && s_.kmax > 0 && s_.noise_power > 0 && s_.bias != 0 &&
                  s_.beta_min < s_.beta_max && s_.beta_min > 0 && spectrum_.smooth))
                throw std::invalid_argument("[FieldLevelBAO] Invalid settings");
            if (s_.kmax >= kf * (s_.nmesh / 2) || s_.kmax >= kf * (s_.nmesh_density / 2))
                throw std::invalid_argument("[FieldLevelBAO] kmax must lie below the Nyquist frequency of both grids");
            if (s_.model == Model::LPT1)
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
                        Mode m{{n[0], n[1], n[2]}, k, double(FML::GRID::fourier_mode_multiplicity(iz, N)),
                               std::size_t(grid_.get_index_fourier({ix, iy, iz})), 0, 0, 0};
                        const double S = s_.bias * s_.bias * spectrum_(k, s_.beta_ref);
                        m.centering = s_.centering < 0 ? S / (S + s_.noise_power) : s_.centering;
                        if (s_.model == Model::LPT1) {
                            const int Nd = s_.nmesh_density;
                            m.density_index = density_.get_index_fourier({(n[0] + Nd) % Nd, (n[1] + Nd) % Nd, iz});
                        }
                        m.t_ref = transfer(m, s_.beta_ref);
                        modes_.push_back(m);
                    }
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
                grad_beta += m.weight * N3 * std::real(std::conj(Q[i]) * w.sk[i]) * Teff * (1 - m.centering) * half_dlnP;
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

        /// Fourier-diagonal HMC mass: 1 + b^2 P(k, beta) / P_eps inside kmax, 1 outside (the exact Hessian of the
        /// linear model's potential in s; a good preconditioner for 1LPT on large scales)
        std::vector<double> mass_spectrum(double beta) const {
            std::vector<double> mass(modes_.size());
            for (std::size_t i = 0; i < modes_.size(); ++i)
                mass[i] = 1 + s_.bias * s_.bias * spectrum_(modes_[i].k, beta) / s_.noise_power;
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
            std::vector<Complex> sk;      // r2c(s) on the modes
            std::array<Field, 3> psi;     // 1LPT displacement (box units), per particle
        };
        Settings s_;
        BAOSpectrum spectrum_;
        mutable FML::GRID::FFTWGrid<3> grid_;
        mutable FML::GRID::FFTWGrid<3> density_;
        std::vector<Mode> modes_;
        std::vector<FML::IndexIntType> real_index_;
        std::vector<Complex> data_;
        std::size_t ncells_{0};
        double volume_{0}, sigma2_{0};

        std::size_t cell(int ix, int iy, int iz) const {
            return (std::size_t(ix) * s_.nmesh + iy) * s_.nmesh + iz;
        }
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
            // 1LPT: Psi_k = i k / k^2 delta_k (box units, k = 2 pi n), particles at x = q + Psi
            const int N = s_.nmesh;
            for (int c = 0; c < 3; ++c) {
                g.fill_fourier_grid(FML::GRID::ComplexType(0));
                auto * fc = g.get_fourier_grid();
                for (std::size_t i = 0; i < modes_.size(); ++i) {
                    const auto & m = modes_[i];
                    const double kc = 2 * M_PI * m.n[c];
                    const double k2 = 4 * M_PI * M_PI * double(m.n[0] * m.n[0] + m.n[1] * m.n[1] + m.n[2] * m.n[2]);
                    fc[m.fourier_index] = FML::GRID::ComplexType(Complex(0, kc / k2) * effective_transfer(m, beta) * w.sk[i]);
                }
                g.set_grid_status_real(false);
                g.fftw_c2r();
                from_grid(g, w.psi[c]);
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
                        const double q[3] = {double(ix) / N, double(iy) / N, double(iz) / N};
                        int i0[3];
                        double t[3];
                        cic_cell(q, w, p, Nd, i0, t);
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b)
                                for (int e = 0; e < 2; ++e) {
                                    const double wgt =
                                        (a ? t[0] : 1 - t[0]) * (b ? t[1] : 1 - t[1]) * (e ? t[2] : 1 - t[2]);
                                    rho[d.get_index_real({(i0[0] + a) % Nd, (i0[1] + b) % Nd, (i0[2] + e) % Nd})] +=
                                        FML::GRID::FloatType(fill * wgt);
                                }
                    }
            d.fftw_r2c(); // the mean (k = 0) is never used, so delta = rho - 1 is implicit
            const auto * fd = d.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                pred[i] = s_.bias * Complex(fd[modes_[i].density_index]);
            return pred;
        }

        // Lower CIC cell and fractional offsets of particle p (lattice position q plus displacement)
        void cic_cell(const double q[3], const Workspace & w, std::size_t p, int Nd, int i0[3], double t[3]) const {
            for (int c = 0; c < 3; ++c) {
                double u = (q[c] + w.psi[c][p]) * Nd;
                u -= Nd * std::floor(u / Nd);
                const double fl = std::floor(u);
                i0[c] = int(fl) % Nd;
                t[c] = u - fl;
            }
        }

        // Adjoint of the 1LPT model: from Gbar on the modes to Q on the modes
        void lpt_adjoint(const std::vector<Complex> & gbar, const Workspace & w, std::vector<Complex> & Q) const {
            const int N = s_.nmesh, Nd = s_.nmesh_density;
            const double Nd3 = double(std::size_t(Nd) * Nd * Nd);
            // delta_d-bar = A^T (b Gbar) = c2r(b Gbar) / Nd^3 on the density grid
            auto & d = density_;
            d.fill_fourier_grid(FML::GRID::ComplexType(0));
            auto * fd = d.get_fourier_grid();
            for (std::size_t i = 0; i < modes_.size(); ++i)
                fd[modes_[i].density_index] = FML::GRID::ComplexType(s_.bias * gbar[i] / Nd3);
            d.set_grid_status_real(false);
            d.fftw_c2r();
            const auto * dbar = d.get_real_grid();
            // Particle adjoint xbar_p = fill * sum_y dbar_y dW(x_p - y)/dx (box units); Psi-bar = xbar
            const double fill = Nd3 / double(ncells_);
            std::array<Field, 3> psibar;
            for (auto & x : psibar)
                x.assign(ncells_, 0);
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
            for (int ix = 0; ix < N; ++ix)
                for (int iy = 0; iy < N; ++iy)
                    for (int iz = 0; iz < N; ++iz) {
                        const auto p = cell(ix, iy, iz);
                        const double q[3] = {double(ix) / N, double(iy) / N, double(iz) / N};
                        int i0[3];
                        double t[3];
                        cic_cell(q, w, p, Nd, i0, t);
                        double gx[3] = {0, 0, 0};
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b)
                                for (int e = 0; e < 2; ++e) {
                                    const double v = dbar[d.get_index_real(
                                        {(i0[0] + a) % Nd, (i0[1] + b) % Nd, (i0[2] + e) % Nd})];
                                    const double wx = a ? t[0] : 1 - t[0], wy = b ? t[1] : 1 - t[1],
                                                 wz = e ? t[2] : 1 - t[2];
                                    const double dx = a ? 1 : -1, dy = b ? 1 : -1, dz = e ? 1 : -1;
                                    gx[0] += v * dx * wy * wz;
                                    gx[1] += v * wx * dy * wz;
                                    gx[2] += v * wx * wy * dz;
                                }
                        for (int c = 0; c < 3; ++c)
                            psibar[c][p] = fill * Nd * gx[c];
                    }
            // Q = sum_c conj(i k_c / k^2) r2c(psibar_c)
            std::fill(Q.begin(), Q.end(), Complex(0));
            auto & g = grid_;
            for (int c = 0; c < 3; ++c) {
                to_grid(psibar[c], g);
                g.fftw_r2c();
                const auto * f = g.get_fourier_grid();
                for (std::size_t i = 0; i < modes_.size(); ++i) {
                    const auto & m = modes_[i];
                    const double kc = 2 * M_PI * m.n[c];
                    const double k2 = 4 * M_PI * M_PI * double(m.n[0] * m.n[0] + m.n[1] * m.n[1] + m.n[2] * m.n[2]);
                    Q[i] += Complex(0, -kc / k2) * Complex(f[m.fourier_index]);
                }
            }
        }
    };

    /// HMC over (s, beta) with a Fourier-diagonal mass for s, a scalar mass for beta and reflection at the prior edges.
    struct HMCSettings {
        int n_warmup{100};
        int n_samples{300};
        int n_leapfrog{10};         // jittered uniformly in [n_leapfrog / 2, n_leapfrog]
        double step_size{.5};       // initial; adapted during warmup towards target_acceptance
        double target_acceptance{.7};
        double beta_scale{.02};     // initial guess of the posterior std of beta (sets its mass); adapted in warmup
        std::uint64_t seed{1};
        bool verbose{false};
    };

    struct Chain {
        std::vector<double> beta, potential;
        double acceptance{0}, step_size{0}, beta_scale{0};
        Field last_s;
    };

    inline Chain run_hmc(const FieldLevelBAO & model, Field s, double beta, const HMCSettings & hs) {
        std::mt19937_64 rng(hs.seed);
        std::normal_distribution<double> gauss;
        std::uniform_real_distribution<double> uniform;
        const auto & st = model.settings();
        if (!(beta > st.beta_min && beta < st.beta_max))
            throw std::invalid_argument("[run_hmc] Initial beta outside the prior");
        // The mass is fixed at the starting beta (it only preconditions, it need not be exact)
        const auto mass = model.mass_spectrum(beta);
        std::vector<double> sqrt_mass(mass.size()), inv_mass(mass.size());
        for (std::size_t i = 0; i < mass.size(); ++i) {
            sqrt_mass[i] = std::sqrt(mass[i]);
            inv_mass[i] = 1 / mass[i];
        }
        double eps = hs.step_size, beta_scale = hs.beta_scale;
        Field grad(s.size());
        double grad_beta = 0;
        double U = model.potential_and_gradient(s, beta, grad, grad_beta);
        Chain chain;
        std::vector<double> warm_beta;
        int accepted = 0;
        for (int it = 0; it < hs.n_warmup + hs.n_samples; ++it) {
            const bool warm = it < hs.n_warmup;
            const double m_beta = 1 / (beta_scale * beta_scale);
            Field xi(s.size());
            for (auto & x : xi)
                x = gauss(rng);
            Field p = model.filter(xi, sqrt_mass);
            double p_beta = std::sqrt(m_beta) * gauss(rng);
            const auto kinetic = [&](const Field & mom, double mom_beta) {
                const auto v = model.filter(mom, inv_mass);
                double K = 0;
                for (std::size_t c = 0; c < v.size(); ++c)
                    K += mom[c] * v[c];
                return .5 * K + .5 * mom_beta * mom_beta / m_beta;
            };
            const double H0 = U + kinetic(p, p_beta);
            Field s_new = s, grad_new = grad;
            double beta_new = beta, grad_beta_new = grad_beta, U_new = U;
            const int nlf = hs.n_leapfrog / 2 + int(uniform(rng) * (hs.n_leapfrog - hs.n_leapfrog / 2 + 1));
            for (int l = 0; l < std::max(1, nlf); ++l) {
                for (std::size_t c = 0; c < p.size(); ++c)
                    p[c] -= .5 * eps * grad_new[c];
                p_beta -= .5 * eps * grad_beta_new;
                const auto v = model.filter(p, inv_mass);
                for (std::size_t c = 0; c < p.size(); ++c)
                    s_new[c] += eps * v[c];
                beta_new += eps * p_beta / m_beta;
                // Reflect at the prior boundaries (volume preserving and reversible)
                for (int r = 0; r < 100 && (beta_new < st.beta_min || beta_new > st.beta_max); ++r) {
                    beta_new = beta_new < st.beta_min ? 2 * st.beta_min - beta_new : 2 * st.beta_max - beta_new;
                    p_beta = -p_beta;
                }
                U_new = model.potential_and_gradient(s_new, beta_new, grad_new, grad_beta_new);
                for (std::size_t c = 0; c < p.size(); ++c)
                    p[c] -= .5 * eps * grad_new[c];
                p_beta -= .5 * eps * grad_beta_new;
            }
            const double H1 = U_new + kinetic(p, p_beta);
            const double accept_prob = std::isfinite(H1) ? std::min(1., std::exp(H0 - H1)) : 0.;
            if (uniform(rng) < accept_prob) {
                s.swap(s_new);
                grad.swap(grad_new);
                beta = beta_new;
                grad_beta = grad_beta_new;
                U = U_new;
                if (!warm)
                    ++accepted;
            }
            if (warm) {
                eps *= std::exp(.1 * (accept_prob - hs.target_acceptance));
                warm_beta.push_back(beta);
                // Re-estimate the beta scale from the second half of the warmup so far
                if (it >= 20 && it % 10 == 0) {
                    const std::size_t start = warm_beta.size() / 2;
                    double mean = 0, var = 0;
                    for (auto i = start; i < warm_beta.size(); ++i)
                        mean += warm_beta[i];
                    mean /= double(warm_beta.size() - start);
                    for (auto i = start; i < warm_beta.size(); ++i)
                        var += (warm_beta[i] - mean) * (warm_beta[i] - mean);
                    var /= double(warm_beta.size() - start - 1);
                    if (var > 0)
                        beta_scale = std::clamp(std::sqrt(var), .3 * beta_scale, 3 * beta_scale);
                }
            } else {
                chain.beta.push_back(beta);
                chain.potential.push_back(U);
            }
            if (hs.verbose && (it % 20 == 0))
                std::cout << "[run_hmc] it " << it << " beta " << beta << " U " << U << " eps " << eps
                          << " beta_scale " << beta_scale << " acc " << accept_prob << "\n";
        }
        chain.acceptance = hs.n_samples ? double(accepted) / hs.n_samples : 0;
        chain.step_size = eps;
        chain.beta_scale = beta_scale;
        chain.last_s = std::move(s);
        return chain;
    }
} // namespace FML::FIELDLEVEL
#endif
