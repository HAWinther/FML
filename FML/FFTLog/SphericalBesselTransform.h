#ifndef FML_FFTLOG_SPHERICALBESSELTRANSFORM_H
#define FML_FFTLOG_SPHERICALBESSELTRANSFORM_H

#include <cmath>
#include <complex>
#include <fftw3.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace FML {
    namespace SOLVERS {
        namespace FFTLog {

            //=========================================================================================================
            ///
            /// Spherical Bessel transforms of functions sampled on one fixed logarithmic grid, for the Bessel orders
            /// l = 0, ..., L-1 (the kernels are computed once):
            ///
            ///   g_l(y) = C int_0^inf dx x^2 j_l(x y) f(x),   C = 4 pi (fourier = false) or 1/(2 pi^2) (fourier = true)
            ///
            /// i.e. with fourier = true a power spectrum to a correlation function and with fourier = false the
            /// inverse (times (2 pi)^3). The mcfit-style FFTLog (Yin Li, https://github.com/eelregit/mcfit) of
            /// velocileptors (Chen, Vlah & White; Utils/spherical_bessel_transform*.py), reproduced exactly: the input
            /// is zero padded to 2^(ceil(log2 N) + 1) points, the tilt is max(0, 3/2 - l) and the output grid y is
            /// chosen to reduce ringing (low_ring). real_nyquist makes the Nyquist coefficient of the kernel real (the
            /// FFTW version of velocileptors, used by its LPT modules; the numpy version, used for its q-functions,
            /// does not). The output grid is y_i ~ 1/x_(N-1-i), returned by y(l).
            ///
            /// Not thread safe: use one object per thread. The constructor and destructor create and destroy FFTW plans
            /// (serialize with other FFTW planning).
            ///
            //=========================================================================================================
            class SphericalBesselTransform {
              public:
                using DVector = std::vector<double>;
                using Complex = std::complex<double>;

                SphericalBesselTransform() = default;
                SphericalBesselTransform(const DVector & x,
                                         int L,
                                         bool fourier,
                                         bool low_ring = true,
                                         bool real_nyquist = false)
                    : x(x), L(L) {
                    Nx = int(x.size());
                    if (Nx < 2 or L < 1 or not(x.front() > 0.0) or not(x.back() > x.front()))
                        throw std::invalid_argument("[SphericalBesselTransform] need a log grid with >= 2 points, L >= 1");
                    const double sqrtpi = fourier ? std::sqrt(M_PI) / (2.0 * M_PI * M_PI) : std::sqrt(M_PI);
                    Delta = std::log(x.back() / x.front()) / (Nx - 1);
                    N = 1 << (int(std::ceil(std::log2(double(Nx)))) + 1);
                    const int Npad = N - Nx;
                    ileft = Npad - Npad / 2;
                    xpow.resize(L);
                    ys.resize(L);
                    us.resize(L);
                    tilt.resize(L);
                    auto UK = [&](double nu, Complex z) {
                        return sqrtpi * std::exp(std::log(2.0) * (z - 2.0) + lngamma(0.5 * (nu + z)) -
                                                 lngamma(0.5 * (3.0 + nu - z)));
                    };
                    for (int l = 0; l < L; l++) {
                        const double q = std::max(0.0, 1.5 - l);
                        tilt[l] = q;
                        double lnxy = 0.0;
                        if (low_ring)
                            lnxy = Delta / M_PI * std::arg(UK(l, Complex(q, M_PI / Delta)));
                        ys[l].resize(Nx);
                        for (int i = 0; i < Nx; i++)
                            ys[l][i] = std::exp(lnxy - Delta) * x[i] / (x.front() * x.back());
                        us[l].resize(N / 2 + 1);
                        for (int m = 0; m <= N / 2; m++) {
                            us[l][m] = UK(l, Complex(q, 2.0 * M_PI / N / Delta * m));
                            if (low_ring)
                                us[l][m] *= std::exp(Complex(0.0, -2.0 * M_PI * lnxy / N / Delta * m));
                        }
                        if (real_nyquist)
                            us[l][N / 2] = us[l][N / 2].real();
                        xpow[l].resize(Nx);
                        for (int i = 0; i < Nx; i++)
                            xpow[l][i] = std::pow(x[i], 3.0 - q);
                    }
                    in = fftw_alloc_real(N);
                    out = fftw_alloc_complex(N / 2 + 1);
                    forward = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
                    backward = fftw_plan_dft_c2r_1d(N, out, in, FFTW_ESTIMATE);
                }
                ~SphericalBesselTransform() { release(); }
                SphericalBesselTransform(const SphericalBesselTransform &) = delete;
                SphericalBesselTransform & operator=(const SphericalBesselTransform &) = delete;
                SphericalBesselTransform(SphericalBesselTransform && o) noexcept { *this = std::move(o); }
                SphericalBesselTransform & operator=(SphericalBesselTransform && o) noexcept {
                    if (this != &o) {
                        release();
                        x = std::move(o.x);
                        ys = std::move(o.ys);
                        xpow = std::move(o.xpow);
                        us = std::move(o.us);
                        tilt = std::move(o.tilt);
                        L = o.L;
                        Nx = o.Nx;
                        N = o.N;
                        ileft = o.ileft;
                        Delta = o.Delta;
                        in = o.in;
                        out = o.out;
                        forward = o.forward;
                        backward = o.backward;
                        o.in = nullptr;
                        o.out = nullptr;
                        o.forward = o.backward = nullptr;
                    }
                    return *this;
                }

                /// The output grid of order l
                const DVector & y(int l) const { return ys.at(l); }
                const DVector & input_grid() const { return x; }
                int size() const { return Nx; }

                /// g_l(y_i) of f(x_i) (f.size() == x.size())
                void transform(int l, const double * f, double * g) {
                    if (l < 0 or l >= L)
                        throw std::invalid_argument("[SphericalBesselTransform] l out of range");
                    for (int i = 0; i < N; i++)
                        in[i] = 0.0;
                    for (int i = 0; i < Nx; i++)
                        in[ileft + i] = xpow[l][i] * f[i];
                    fftw_execute(forward);
                    const auto & u = us[l];
                    for (int m = 0; m <= N / 2; m++) {
                        const Complex v = std::conj(Complex(out[m][0], out[m][1]) * u[m]);
                        out[m][0] = v.real();
                        out[m][1] = v.imag();
                    }
                    fftw_execute(backward);
                    const double q = tilt[l];
                    for (int i = 0; i < Nx; i++)
                        g[i] = in[ileft + i] / N * std::pow(ys[l][i], -q);
                }
                DVector transform(int l, const DVector & f) {
                    if (int(f.size()) != Nx)
                        throw std::invalid_argument("[SphericalBesselTransform] f has the wrong size");
                    DVector g(Nx);
                    transform(l, f.data(), g.data());
                    return g;
                }

                /// ln Gamma(z) for Re(z) > 0 (Stirling series after shifting to Re(z) >= 15; the branch is irrelevant
                /// here as only exp of sums is used)
                static Complex lngamma(Complex z) {
                    Complex shift = 0.0;
                    while (z.real() < 15.0) {
                        shift += std::log(z);
                        z += 1.0;
                    }
                    const Complex iz = 1.0 / z, iz2 = iz * iz;
                    const Complex series =
                        iz * (1.0 / 12.0 +
                              iz2 * (-1.0 / 360.0 +
                                     iz2 * (1.0 / 1260.0 +
                                            iz2 * (-1.0 / 1680.0 + iz2 * (1.0 / 1188.0 + iz2 * (-691.0 / 360360.0))))));
                    return (z - 0.5) * std::log(z) - z + 0.5 * std::log(2.0 * M_PI) + series - shift;
                }

              private:
                DVector x;
                std::vector<DVector> ys, xpow;
                std::vector<std::vector<Complex>> us;
                DVector tilt;
                int L{0}, Nx{0}, N{0}, ileft{0};
                double Delta{0.0};
                double * in{nullptr};
                fftw_complex * out{nullptr};
                fftw_plan forward{nullptr}, backward{nullptr};
                void release() {
                    if (forward)
                        fftw_destroy_plan(forward);
                    if (backward)
                        fftw_destroy_plan(backward);
                    if (in)
                        fftw_free(in);
                    if (out)
                        fftw_free(out);
                    forward = backward = nullptr;
                    in = nullptr;
                    out = nullptr;
                }
            };

        } // namespace FFTLog
    } // namespace SOLVERS
} // namespace FML

#endif
