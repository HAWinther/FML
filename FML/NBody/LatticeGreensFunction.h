#ifndef FML_LATTICE_GREENS_FUNCTION_H
#define FML_LATTICE_GREENS_FUNCTION_H
#include <FML/NBody/ForceOptions.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace FML::NBODY {

    // Defined in NBody.h (the kernels of the force): the correction is computed with the same kernels
    inline double gradient_kernel_fourier(double k_j, int Nmesh, int KERNEL);
    template <int NDIM>
    double
    greens_function_laplace_operator_fourier(double kmag2, std::array<double, NDIM> & kvec, int Nmesh, int KERNEL);

    /// @brief The lattice-calibrated Green's function of a PM force (3D).
    ///
    /// While the particles are close to a cubic lattice (the initial conditions), a PM force has a fixed linear
    /// response T(k) = F_PM(k) / F_exact(k) to a displacement mode k of the particles. T depends on the
    /// assignment window, the Green's function and gradient kernels, the deconvolution, the grid factor
    /// m = Nmesh / Npart_1D (an integer), the offset of the lattice from the grid nodes and the interlacing shifts.
    /// It is 1 for an exact force; for CIC with Nmesh = Npart_1D it is ~0.6 at half the particle Nyquist frequency.
    /// This class computes T in closed form and returns the factor C(q) that multiplies the Green's function of
    /// every force mode q so that the linear force is exact for every particle mode k = fold(q):
    ///
    ///     C = (1 - E) / (T - E),
    ///
    /// with E the response from the force field that the undisplaced lattice already has (only for m >= 3).
    /// Every force mode that folds onto the same particle mode k gets the same factor.
    ///
    /// The closed form. The particles sit at x_l = l m + s (units of the force cell h, offset s) and are displaced
    /// by A exp(i k.x_l). Their images K_j = k + 2 pi j / m (j any integer) land on the grid mode
    /// q_r = k + 2 pi r / m (r = j mod m, folded into the grid zone). The window W(K) = prod_d w(K_d) with
    /// w(x) = sinc(x/2)^p and the offset phase factorize, so all the image sums are products of 1D sums
    ///     a_r(k) = sum_{j = r mod m} w(K_j) e^{-i phi_j},  b_r(k) = sum K_j w(K_j) e^{-i phi_j},
    ///     c_r(k) = sum w(K_j) e^{+i phi_j},  phi_j = 2 pi j s / m,
    /// and the force on the mode, along k, is
    ///     F(k) = sum_r i khat.D(q_r) G(q_r) [-i sum_d khat_d b_{r_d} prod_{i != d} a_{r_i}] prod_i c_{r_i},
    /// T = -Re F (F_exact = -1 in these units). With interlacing F and E are averaged over the shifts. E is a
    /// quadratic form in khat (a 3x3 matrix). The derivation and the tests (agreement with the measured response to
    /// |Delta T| < 0.003, and the linear and 20-step tests) are in the linear step test report.
    ///
    /// The correction is only right while the particles are close to the lattice: use it at early times. On a force
    /// grid finer than the particle grid the largest corrections (T small, near the particle Nyquist frequency)
    /// amplify the lattice images as the particles move, so they are tapered (taper_kmin, taper_kmax in units of
    /// the particle Nyquist frequency) and capped (cap). The particle modes beyond taper_kmax get no correction.
    ///
    /// Cost and memory: the 1D sums are 3 m Npart_1D complex numbers per interlacing shift and dimension. For m = 1
    /// there is one term per mode and the factor is computed on the fly. For m >= 2 all the m^3 force modes that
    /// fold onto a particle mode share it, so prepare() tabulates it for the particle modes that the local force
    /// slab folds onto and their mirrors (k_z >= 0, using T(-k) = T(k); no communication): between 1/(2 m^2) (one
    /// task) and 1/m^2 of the local force grid in single precision.
    class LatticeGreensFunction {
      public:
        struct Settings {
            int Npart_1D{0};
            int Nmesh{0};
            int assignment_order{2};                         // p: 2 = CIC, 3 = TSC, 4 = PCS, 5 = PQS
            std::array<double, 3> lattice_offset{};          // offset of the lattice from the nodes (force cells)
            std::vector<std::array<double, 3>> shifts{{}};   // interlacing shifts of the particles (force cells)
            int laplace_kernel{CONTINUOUS_GREENS_FUNCTION};
            int gradient_kernel{CONTINUOUS_GRADIENT};
            int deconvolve_window_power{0};                  // 0: no deconvolution; 2: G divided by W^2
            double taper_kmin{1.0};                          // full correction below (units of the particle k_Ny)
            double taper_kmax{1.0};                          // no correction above
            double cap{3.0};                                 // the factor is clipped to [1/cap, cap] (may be inf)
            int n_images{20000};                             // the image sums run over |l| <= n_images
            bool use_table{true};                            // tabulate the factor for m >= 2 (else on the fly)
        };

        explicit LatticeGreensFunction(const Settings & settings);

        /// The factor that multiplies the Green's function of the force mode with integer wavevector iq
        /// (FFTW convention, components in (-Nmesh/2, Nmesh/2])
        double correction(const std::array<int, 3> & iq) const {
            const auto n = fold(iq);
            if (n[0] == 0 and n[1] == 0 and n[2] == 0)
                return 1.0;
            if (not table.empty()) {
                // The table has n_z >= 0: T(-k) = T(k) (the response to the conjugate mode is the conjugate)
                auto t = n;
                if (t[2] < 0)
                    for (auto & v : t)
                        v = v == -Np / 2 ? v : -v;
                const int ux = table_row[t[0] + Np / 2];
                if (ux >= 0)
                    return table[(size_t(ux) * Np + size_t(t[1] + Np / 2)) * size_t(Np / 2 + 1) + size_t(std::abs(t[2]))];
            }
            return correction_of_particle_mode(n);
        }

        /// Tabulate the factor for the particle modes of the force slab [x_start, x_start + nx) and their mirrors
        /// (for m >= 2; nothing for m = 1). Call before the (threaded) loop over the modes; it only rebuilds when the
        /// slab changes
        void prepare(long x_start, long nx) const;

        /// The response T(k) and the lattice-field term E(khat) for the particle mode n (integers in
        /// [-Npart_1D/2, Npart_1D/2)), before the taper and the cap
        std::pair<double, double> response(const std::array<int, 3> & n) const;

        /// The factor for the particle mode n (taper and cap applied)
        double correction_of_particle_mode(const std::array<int, 3> & n) const;

        int grid_factor() const { return m; }
        size_t table_bytes() const { return table.size() * sizeof(float); }
        const Settings & get_settings() const { return s; }

      private:
        Settings s;
        int Np{0};
        int Nm{0};
        int m{1};
        // The 1D sums [shift][dim][r][n + Np/2] (a, b, c) and the same at k = 0 (for E) [shift][dim][r]
        std::vector<std::array<std::vector<std::complex<double>>, 3>> A, B, C;
        std::vector<std::array<std::vector<std::complex<double>>, 3>> A0, B0, C0;
        // The lattice-field term: E(khat) = -Re sum_{de} khat_d khat_e M_de (averaged over the shifts)
        std::array<std::array<std::complex<double>, 3>, 3> M{};
        bool has_E{false};
        // The table of the factor for m >= 2: rows of the local slab (particle x-mode -> row, -1 if not local)
        mutable std::vector<float> table;
        mutable std::vector<int> table_row;
        mutable long table_x_start{-1}, table_nx{-1};

        std::array<int, 3> fold(const std::array<int, 3> & iq) const {
            std::array<int, 3> n;
            for (int d = 0; d < 3; d++) {
                int v = ((iq[d] + Np / 2) % Np + Np) % Np - Np / 2;
                n[d] = v;
            }
            return n;
        }
        size_t idx(int r, int n) const { return size_t(r) * Np + size_t(n + Np / 2); }
        // The kernels of the code in units of the force cell (q in units of 1/h, iq the integer grid wavevector)
        double green(const std::array<int, 3> & iq) const;
        double gradient(int iq) const { return gradient_kernel_fourier(2.0 * M_PI * iq, Nm, s.gradient_kernel) / Nm; }
        double window(const std::array<int, 3> & iq) const;
        int wrap_mesh(int i) const { return ((i + Nm / 2) % Nm + Nm) % Nm - Nm / 2; }
        static void sums1d(double x, int p, double phase0, double phase_per_l, int L, std::complex<double> & a,
                           std::complex<double> & b, std::complex<double> & c);
    };

    //=================================================================================================================

    /// Z_q(theta, x) = sum_{l in Z} e^{i l theta} / (x + l)^q for non-integer x. In closed form for q = 1 (the
    /// principal value) and q = 2: with S = pi cot(pi x) for theta = 0 and S = pi e^{i (pi - theta) x} / sin(pi x)
    /// for 0 < theta < 2 pi, Z_1 = S and Z_2 = -dS/dx. For q >= 3 the sum over |l| <= L (error ~ L^{1-q})
    inline std::complex<double> lattice_greens_Z(int q, double theta, double x, int L) {
        theta = std::fmod(theta, 2.0 * M_PI);
        if (theta < 0.0)
            theta += 2.0 * M_PI;
        if (theta > 2.0 * M_PI - 1e-13)
            theta = 0.0;
        const double sn = std::sin(M_PI * x);
        const double cot = std::cos(M_PI * x) / sn;
        if (q == 1 or q == 2) {
            const bool zero = theta < 1e-13;
            const double alpha = M_PI - theta;
            const std::complex<double> S =
                zero ? std::complex<double>(M_PI * cot, 0.0) : M_PI * std::polar(1.0, alpha * x) / sn;
            if (q == 1)
                return S;
            // dS/dx: theta = 0: -pi^2 / sin^2; otherwise S (i alpha - pi cot)
            const std::complex<double> dS =
                zero ? std::complex<double>(-M_PI * M_PI / (sn * sn), 0.0)
                     : S * std::complex<double>(-M_PI * cot, alpha);
            return -dS;
        }
        std::complex<double> sum = 0.0;
        const std::complex<double> rot = std::polar(1.0, theta);
        std::complex<double> e = std::polar(1.0, -double(L) * theta);
        for (int l = -L; l <= L; l++, e *= rot) {
            if ((l + L) % 1024 == 0)
                e = std::polar(1.0, double(l) * theta);
            double y = x + l, yq = y;
            for (int i = 1; i < q; i++)
                yq *= y;
            sum += e / yq;
        }
        return sum;
    }

    /// The image sums over the residue class j = r + m l: K = 2 pi (x + l), x = (n / Np + r) / m,
    /// w = sinc(K/2)^p = sin(pi x)^p (-1)^{l p} / (pi (x + l))^p and the phase phi = phase0 + l phase_per_l:
    ///     a = sum w e^{-i phi},  b = sum K w e^{-i phi},  c = sum w e^{+i phi}
    /// in terms of Z_q (closed form for the slowly converging q = 1, 2 of CIC)
    inline void LatticeGreensFunction::sums1d(double x,
                                              int p,
                                              double phase0,
                                              double phase_per_l,
                                              int L,
                                              std::complex<double> & a,
                                              std::complex<double> & b,
                                              std::complex<double> & c) {
        const double xr = x - std::round(x);
        if (std::abs(xr) < 1e-14) {
            // sin(pi x) = 0: only the image with K = 0 contributes (w = 1)
            const long l0 = -std::lround(x);
            const double ph = phase0 + double(l0) * phase_per_l;
            a = std::polar(1.0, -ph);
            c = std::polar(1.0, +ph);
            b = 0.0;
            return;
        }
        const double sp = std::pow(std::sin(M_PI * x), p);
        const double sign_phase = (p % 2 == 1) ? M_PI : 0.0; // (-1)^{l p} = e^{i l pi} for odd p
        const double pip = std::pow(M_PI, p);
        const std::complex<double> em = std::polar(1.0, -phase0), ep = std::polar(1.0, phase0);
        a = sp / pip * em * lattice_greens_Z(p, sign_phase - phase_per_l, x, L);
        c = sp / pip * ep * lattice_greens_Z(p, sign_phase + phase_per_l, x, L);
        b = 2.0 * sp / (pip / M_PI) * em * lattice_greens_Z(p - 1, sign_phase - phase_per_l, x, L);
    }

    inline LatticeGreensFunction::LatticeGreensFunction(const Settings & settings) : s(settings) {
        Np = s.Npart_1D;
        Nm = s.Nmesh;
        if (Np <= 0 or Nm <= 0 or Nm % Np != 0)
            throw std::invalid_argument("[LatticeGreensFunction] Nmesh must be an integer multiple of Npart_1D");
        if (Np % 2 != 0)
            throw std::invalid_argument("[LatticeGreensFunction] Npart_1D must be even");
        if (s.assignment_order < 2 or s.assignment_order > 5)
            throw std::invalid_argument("[LatticeGreensFunction] The assignment must be CIC, TSC, PCS or PQS");
        if (s.shifts.empty())
            s.shifts.push_back({});
        if (not(s.taper_kmin > 0.0 and s.taper_kmin <= s.taper_kmax) or not(s.cap >= 1.0))
            throw std::invalid_argument("[LatticeGreensFunction] Invalid taper or cap");
        m = Nm / Np;
        const int p = s.assignment_order;
        const int nshift = int(s.shifts.size());
        A.resize(nshift), B.resize(nshift), C.resize(nshift);
        A0.resize(nshift), B0.resize(nshift), C0.resize(nshift);
        for (int is = 0; is < nshift; is++) {
            for (int d = 0; d < 3; d++) {
                // The offset of the shifted lattice from the nodes in force cells: phi_j = 2 pi j so / m with
                // j = r + m l, so phase0 = 2 pi r so / m and the phase per l is 2 pi so
                const double so = s.lattice_offset[d] + s.shifts[is][d];
                // The same for every dimension unless the offsets differ
                bool same = false;
                for (int e = 0; e < d; e++) {
                    if (s.lattice_offset[e] + s.shifts[is][e] == so) {
                        A[is][d] = A[is][e], B[is][d] = B[is][e], C[is][d] = C[is][e];
                        A0[is][d] = A0[is][e], B0[is][d] = B0[is][e], C0[is][d] = C0[is][e];
                        same = true;
                        break;
                    }
                }
                if (same)
                    continue;
                auto & a = A[is][d];
                auto & b = B[is][d];
                auto & c = C[is][d];
                a.assign(size_t(m) * Np, 0.0), b.assign(size_t(m) * Np, 0.0), c.assign(size_t(m) * Np, 0.0);
                A0[is][d].assign(m, 0.0), B0[is][d].assign(m, 0.0), C0[is][d].assign(m, 0.0);
#ifdef USE_OMP
#pragma omp parallel for collapse(2)
#endif
                for (int r = 0; r < m; r++)
                    for (int n = -Np / 2; n < Np / 2; n++) {
                        const double x = (double(n) / Np + r) / m;
                        sums1d(x, p, 2.0 * M_PI * r * so / m, 2.0 * M_PI * so, s.n_images, a[idx(r, n)], b[idx(r, n)],
                               c[idx(r, n)]);
                    }
                for (int r = 0; r < m; r++)
                    sums1d(double(r) / m, p, 2.0 * M_PI * r * so / m, 2.0 * M_PI * so, s.n_images, A0[is][d][r],
                           B0[is][d][r], C0[is][d][r]);
            }
        }

        // The lattice-field term (m >= 3; for m = 1, 2 the lattice harmonics sit where the gradient vanishes)
        for (auto & row : M)
            row.fill(0.0);
        has_E = m >= 3;
        if (has_E) {
            const std::complex<double> I(0, 1);
            for (int is = 0; is < nshift; is++) {
                for (int rx = 0; rx < m; rx++)
                    for (int ry = 0; ry < m; ry++)
                        for (int rz = 0; rz < m; rz++) {
                            if (rx == 0 and ry == 0 and rz == 0)
                                continue;
                            const std::array<int, 3> r{rx, ry, rz};
                            std::array<int, 3> iq;
                            for (int d = 0; d < 3; d++)
                                iq[d] = wrap_mesh(r[d] * Np);
                            double G = green(iq);
                            if (s.deconvolve_window_power > 0)
                                G /= std::pow(window(iq), s.deconvolve_window_power);
                            std::complex<double> dens0 = 1.0;
                            std::array<std::complex<double>, 3> Bc, Cc;
                            for (int d = 0; d < 3; d++) {
                                dens0 *= A0[is][d][r[d]];
                                Bc[d] = std::conj(B0[is][d][r[d]]);
                                Cc[d] = C0[is][d][r[d]];
                            }
                            for (int d = 0; d < 3; d++) {
                                const double Dd = gradient(iq[d]);
                                for (int e = 0; e < 3; e++) {
                                    std::complex<double> grad = Bc[e];
                                    for (int i = 0; i < 3; i++)
                                        if (i != e)
                                            grad *= Cc[i];
                                    M[d][e] += I * Dd * G * dens0 * I * grad / double(nshift);
                                }
                            }
                        }
            }
        }
    }

    inline double LatticeGreensFunction::green(const std::array<int, 3> & iq) const {
        std::array<double, 3> kvec;
        double k2 = 0.0;
        for (int d = 0; d < 3; d++) {
            kvec[d] = 2.0 * M_PI * iq[d];
            k2 += kvec[d] * kvec[d];
        }
        if (k2 == 0.0)
            return 0.0;
        return greens_function_laplace_operator_fourier<3>(k2, kvec, Nm, s.laplace_kernel) * double(Nm) * double(Nm);
    }

    inline double LatticeGreensFunction::window(const std::array<int, 3> & iq) const {
        double w = 1.0;
        for (int d = 0; d < 3; d++) {
            const double x = M_PI * double(iq[d]) / double(Nm); // q/2 in units of 1/h
            w *= x == 0.0 ? 1.0 : std::sin(x) / x;
        }
        return std::pow(w, s.assignment_order);
    }

    inline std::pair<double, double> LatticeGreensFunction::response(const std::array<int, 3> & n) const {
        const std::complex<double> I(0, 1);
        std::array<double, 3> khat;
        double norm = 0.0;
        for (int d = 0; d < 3; d++) {
            khat[d] = double(n[d]);
            norm += khat[d] * khat[d];
        }
        if (norm == 0.0)
            return {1.0, 0.0};
        norm = std::sqrt(norm);
        for (int d = 0; d < 3; d++)
            khat[d] /= norm;

        const int nshift = int(s.shifts.size());
        std::complex<double> F = 0.0;
        for (int rx = 0; rx < m; rx++)
            for (int ry = 0; ry < m; ry++)
                for (int rz = 0; rz < m; rz++) {
                    const std::array<int, 3> r{rx, ry, rz};
                    std::array<int, 3> iq;
                    for (int d = 0; d < 3; d++)
                        iq[d] = wrap_mesh(n[d] + r[d] * Np);
                    double G = green(iq);
                    if (G == 0.0)
                        continue;
                    if (s.deconvolve_window_power > 0)
                        G /= std::pow(window(iq), s.deconvolve_window_power);
                    double Dk = 0.0;
                    for (int d = 0; d < 3; d++)
                        Dk += khat[d] * gradient(iq[d]);
                    if (Dk == 0.0)
                        continue;
                    std::complex<double> sum = 0.0;
                    for (int is = 0; is < nshift; is++) {
                        std::array<std::complex<double>, 3> a, b, c;
                        for (int d = 0; d < 3; d++) {
                            const size_t j = idx(r[d], n[d]);
                            a[d] = A[is][d][j], b[d] = B[is][d][j], c[d] = C[is][d][j];
                        }
                        const std::complex<double> dens =
                            -I * (khat[0] * b[0] * a[1] * a[2] + khat[1] * a[0] * b[1] * a[2] + khat[2] * a[0] * a[1] * b[2]);
                        sum += dens * c[0] * c[1] * c[2];
                    }
                    F += I * Dk * G * sum / double(nshift);
                }
        double E = 0.0;
        if (has_E) {
            std::complex<double> e = 0.0;
            for (int d = 0; d < 3; d++)
                for (int f = 0; f < 3; f++)
                    e += khat[d] * khat[f] * M[d][f];
            E = -e.real();
        }
        return {-F.real() + E, E};
    }

    inline double LatticeGreensFunction::correction_of_particle_mode(const std::array<int, 3> & n) const {
        const double kk = std::sqrt(double(n[0]) * n[0] + double(n[1]) * n[1] + double(n[2]) * n[2]) / (0.5 * Np);
        if (kk >= s.taper_kmax)
            return 1.0;
        const auto [T, E] = response(n);
        const double denom = T - E;
        // No response (e.g. a gradient that vanishes for a Nyquist component): nothing to correct
        double R = denom > 0.0 ? (1.0 - E) / denom : 1.0;
        R = std::clamp(R, 1.0 / s.cap, s.cap);
        double w = 1.0;
        if (kk > s.taper_kmin)
            w = 0.5 * (1.0 + std::cos(M_PI * (kk - s.taper_kmin) / (s.taper_kmax - s.taper_kmin)));
        return 1.0 + (R - 1.0) * w;
    }

    inline void LatticeGreensFunction::prepare(long x_start, long nx) const {
        if (m == 1 or not s.use_table)
            return;
        if (x_start == table_x_start and nx == table_nx)
            return;
        // The particle x-modes of the slab
        table_row.assign(Np, -1);
        std::vector<int> modes;
        for (long i = x_start; i < x_start + nx; i++) {
            const int iqx = int(i) <= Nm / 2 ? int(i) : int(i) - Nm;
            const int n = fold({iqx, 0, 0})[0];
            // the row and its mirror (the modes with n_z < 0 are looked up at -n)
            for (int nn : {n, n == -Np / 2 ? n : -n}) {
                if (table_row[nn + Np / 2] < 0) {
                    table_row[nn + Np / 2] = int(modes.size());
                    modes.push_back(nn);
                }
            }
        }
        const size_t nz = size_t(Np / 2 + 1);
        table.assign(modes.size() * size_t(Np) * nz, 1.0f);
#ifdef USE_OMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (size_t ux = 0; ux < modes.size(); ux++)
            for (int ny = -Np / 2; ny < Np / 2; ny++)
                for (size_t nzi = 0; nzi < nz; nzi++) {
                    const std::array<int, 3> n{modes[ux], ny, int(nzi) == Np / 2 ? -Np / 2 : int(nzi)};
                    table[(ux * Np + size_t(ny + Np / 2)) * nz + nzi] =
                        (n[0] == 0 and n[1] == 0 and n[2] == 0) ? 1.0f : float(correction_of_particle_mode(n));
                }
        table_x_start = x_start;
        table_nx = nx;
    }

} // namespace FML::NBODY
#endif
