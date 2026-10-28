//==========================================================
// Tests for LPT/DisplacementFields.h and the random field generators it is used with:
//   - 3LPT potentials (phi_1LPT, phi_2LPT, phi_3LPT_a, phi_3LPT_b and the curl term A) for a band-limited
//     density field made of a few plane waves. The reference is computed exactly in the test with a sparse
//     plane-wave algebra (products are convolutions of the mode lists, derivatives and inverse Laplacians act
//     mode by mode), so it is independent of the FFTs and of the library code. As the 3LPT sources have at most
//     3x the bandwidth of delta (< Nmesh/2) there is no aliasing and the agreement is to round-off.
//   - compute_3LPT_displacement_field (3D against the reference, 2D against the 3LPT potentials)
//   - the spherical collapse potential (phi_SC = -source/k^2) and ALPT (finite, R=0 limit = 1LPT+2LPT,
//     R=inf limit = phi_SC)
//   - a non-gaussian field where P(k)/P_primordial(k) is 0/0 at k=0 is finite
//   - the growth factors for a w = -0.8 background against an ODE solved here and the EdS limit at a = 1
//   - RandomGenerator::set_seed resets the normal distribution and white noise slices differ
//   - GaussianRandomField.h and DisplacementFields.h can be included in several translation units
//     (test_lpt_odr.cpp)
// Works with any number of MPI tasks dividing Nmesh = 16 (1, 2, 4, ...)
// Build: test_lpt.o test_lpt_odr.o Global.o Spline.o ODESolver.o (needs USE_FFTW and USE_GSL)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/RandomFields/NonLocalGaussianRandomField.h>
#include <FML/RandomGenerator/RandomGenerator.h>

#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <map>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

int lpt_odr_check(); // defined in test_lpt_odr.cpp

using cplx = std::complex<double>;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << std::setprecision(4) << x;
    return os.str();
}

//==========================================================
// Sparse plane-wave fields f(x) = sum_n c_n exp(2 pi i n.x) (box = 1)
//==========================================================
template <int N>
using Modes = std::map<std::array<int, N>, cplx>;

template <int N>
Modes<N> add(const Modes<N> & a, const Modes<N> & b, double fa = 1.0, double fb = 1.0) {
    Modes<N> r;
    for (auto & [n, c] : a)
        r[n] += fa * c;
    for (auto & [n, c] : b)
        r[n] += fb * c;
    return r;
}
template <int N>
Modes<N> mul(const Modes<N> & a, const Modes<N> & b) {
    Modes<N> r;
    for (auto & [na, ca] : a)
        for (auto & [nb, cb] : b) {
            std::array<int, N> n;
            for (int i = 0; i < N; i++)
                n[i] = na[i] + nb[i];
            r[n] += ca * cb;
        }
    return r;
}
template <int N>
double k2_of(const std::array<int, N> & n) {
    double k2 = 0.0;
    for (int i = 0; i < N; i++)
        k2 += std::pow(2.0 * M_PI * n[i], 2);
    return k2;
}
// d_i d_j f
template <int N>
Modes<N> hess(const Modes<N> & a, int i, int j) {
    Modes<N> r;
    for (auto & [n, c] : a)
        r[n] = -c * (2.0 * M_PI * n[i]) * (2.0 * M_PI * n[j]);
    return r;
}
// The potential p with Laplacian(p) = f (the k = 0 mode set to zero)
template <int N>
Modes<N> solve(const Modes<N> & a) {
    Modes<N> r;
    for (auto & [n, c] : a) {
        double k2 = k2_of<N>(n);
        r[n] = k2 > 0.0 ? -c / k2 : 0.0;
    }
    return r;
}
// d_i f
template <int N>
Modes<N> grad(const Modes<N> & a, int i) {
    Modes<N> r;
    for (auto & [n, c] : a)
        r[n] = cplx(0, 2.0 * M_PI * n[i]) * c;
    return r;
}
template <int N>
double eval(const Modes<N> & a, const std::array<double, N> & x) {
    double v = 0.0;
    for (auto & [n, c] : a) {
        double arg = 0.0;
        for (int i = 0; i < N; i++)
            arg += 2.0 * M_PI * n[i] * x[i];
        v += (c * std::exp(cplx(0, arg))).real();
    }
    return v;
}
template <int N>
double maxabs(const Modes<N> & a) {
    double m = 0.0;
    for (auto & [n, c] : a)
        m = std::max(m, std::abs(c));
    return m;
}
// A sum of cosines a cos(2 pi n.x + theta)
template <int N>
Modes<N> plane_waves(const std::vector<std::array<int, N>> & ns,
                     const std::vector<double> & amps,
                     const std::vector<double> & phases) {
    Modes<N> r;
    for (size_t m = 0; m < ns.size(); m++) {
        std::array<int, N> mn;
        for (int i = 0; i < N; i++)
            mn[i] = -ns[m][i];
        r[ns[m]] += 0.5 * amps[m] * std::exp(cplx(0, phases[m]));
        r[mn] += 0.5 * amps[m] * std::exp(cplx(0, -phases[m]));
    }
    return r;
}

//==========================================================
// Grid helpers
//==========================================================
template <int N>
std::array<int, N> integer_wavevector(const FML::GRID::FFTWGrid<N> & g, long long int fourier_index) {
    std::array<double, N> kvec;
    double kmag2;
    g.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);
    std::array<int, N> n;
    for (int i = 0; i < N; i++)
        n[i] = int(std::lround(kvec[i] / (2.0 * M_PI)));
    return n;
}

// max |grid(k) - ref(k)| over all modes (global)
template <int N>
double max_diff_fourier(const FML::GRID::FFTWGrid<N> & g, const Modes<N> & ref) {
    double d = 0.0;
    for (int islice = 0; islice < g.get_local_nx(); islice++)
        for (auto && fi : g.get_fourier_range(islice, islice + 1)) {
            auto n = integer_wavevector<N>(g, fi);
            auto it = ref.find(n);
            cplx r = it == ref.end() ? cplx(0.0) : it->second;
            cplx v = g.get_fourier_from_index(fi);
            double diff = std::abs(v - r);
            if (not(diff == diff))
                diff = 1e100; // NaN
            d = std::max(d, diff);
        }
    FML::MaxOverTasks(&d);
    return d;
}

// max |grid(x) - ref(x)| over all cells (global)
template <int N>
double max_diff_real(const FML::GRID::FFTWGrid<N> & g, const Modes<N> & ref) {
    double d = 0.0;
    const int Nmesh = g.get_nmesh();
    for (int islice = 0; islice < g.get_local_nx(); islice++)
        for (auto && ri : g.get_real_range(islice, islice + 1)) {
            auto c = g.get_coord_from_index(ri);
            std::array<double, N> x;
            x[0] = double(c[0] + g.get_local_x_start()) / Nmesh;
            for (int i = 1; i < N; i++)
                x[i] = double(c[i]) / Nmesh;
            double diff = std::fabs(g.get_real_from_index(ri) - eval<N>(ref, x));
            if (not(diff == diff))
                diff = 1e100;
            d = std::max(d, diff);
        }
    FML::MaxOverTasks(&d);
    return d;
}

template <int N>
FML::GRID::FFTWGrid<N> make_delta_fourier(const Modes<N> & delta, int Nmesh) {
    FML::GRID::FFTWGrid<N> g(Nmesh, 1, 1);
    for (int islice = 0; islice < g.get_local_nx(); islice++)
        for (auto && ri : g.get_real_range(islice, islice + 1)) {
            auto c = g.get_coord_from_index(ri);
            std::array<double, N> x;
            x[0] = double(c[0] + g.get_local_x_start()) / Nmesh;
            for (int i = 1; i < N; i++)
                x[i] = double(c[i]) / Nmesh;
            g.set_real_from_index(ri, eval<N>(delta, x));
        }
    g.fftw_r2c();
    return g;
}

template <int N>
bool all_finite_fourier(const FML::GRID::FFTWGrid<N> & g) {
    int ok = 1;
    for (int islice = 0; islice < g.get_local_nx(); islice++)
        for (auto && fi : g.get_fourier_range(islice, islice + 1)) {
            auto v = g.get_fourier_from_index(fi);
            if (not std::isfinite(double(v.real())) or not std::isfinite(double(v.imag())))
                ok = 0;
        }
    FML::MinOverTasks(&ok);
    return ok == 1;
}

//==========================================================
// The independent 3LPT reference (EdS growing mode at D = 1, displacement Psi = grad phi + curl A)
//==========================================================
struct Reference3LPT {
    Modes<3> p1, p2, p3a, p3b;
    std::array<Modes<3>, 3> A;
};
Reference3LPT reference_3LPT(const Modes<3> & delta) {
    Reference3LPT r;
    // Psi1 = grad p1 with Laplacian(p1) = -delta
    r.p1 = solve<3>(delta);
    for (auto & [n, c] : r.p1)
        c = -c;
    std::array<std::array<Modes<3>, 3>, 3> G1, G2;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            G1[i][j] = hess<3>(r.p1, i, j);
    // mu2 = 1/2 [ (tr G1)^2 - G1_ij G1_ji ],  chi with Laplacian(chi) = mu2,  phi_2LPT = -3/7 chi
    Modes<3> tr1 = add<3>(add<3>(G1[0][0], G1[1][1]), G1[2][2]);
    Modes<3> mu2 = mul<3>(tr1, tr1);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mu2 = add<3>(mu2, mul<3>(G1[i][j], G1[j][i]), 1.0, -1.0);
    for (auto & [n, c] : mu2)
        c *= 0.5;
    Modes<3> chi = solve<3>(mu2);
    r.p2 = add<3>(chi, {}, -3.0 / 7.0);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            G2[i][j] = hess<3>(chi, i, j);
    // phi_3LPT_a = -1/3 solve(det G1)
    Modes<3> det;
    const int perm[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};
    const double sgn[6] = {1, 1, 1, -1, -1, -1};
    for (int p = 0; p < 6; p++)
        det = add<3>(det, mul<3>(mul<3>(G1[0][perm[p][0]], G1[1][perm[p][1]]), G1[2][perm[p][2]]), 1.0, sgn[p]);
    r.p3a = add<3>(solve<3>(det), {}, -1.0 / 3.0);
    // phi_3LPT_b = 10/21 solve( 1/2 [ tr G1 tr G2 - G1_ij G2_ji ] )
    Modes<3> tr2 = add<3>(add<3>(G2[0][0], G2[1][1]), G2[2][2]);
    Modes<3> mu2m = mul<3>(tr1, tr2);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            mu2m = add<3>(mu2m, mul<3>(G1[i][j], G2[j][i]), 1.0, -1.0);
    r.p3b = add<3>(solve<3>(mu2m), {}, 0.5 * 10.0 / 21.0);
    // curl Psi3c = -1/7 eps_ijk G1_lj G2_lk and Psi3c = curl A with -Laplacian(A) = curl Psi3c
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3, k = (i + 2) % 3;
        Modes<3> curl;
        for (int l = 0; l < 3; l++) {
            curl = add<3>(curl, mul<3>(G1[l][j], G2[l][k]));
            curl = add<3>(curl, mul<3>(G1[l][k], G2[l][j]), 1.0, -1.0);
        }
        r.A[i] = add<3>(solve<3>(curl), {}, 1.0 / 7.0);
    }
    return r;
}

//==========================================================
// The tests
//==========================================================
const int Nmesh = 16;

void test_3LPT(bool alternate = false,
               FML::COSMOLOGY::LPT::Dealiasing dealiasing = FML::COSMOLOGY::LPT::Dealiasing::None) {
    using Grid = FML::GRID::FFTWGrid<3>;
    // Bandwidth 2 per axis so the 3LPT sources (bandwidth 6) are exactly representable on 16^3
    Modes<3> delta = plane_waves<3>({{1, 0, 0}, {0, 1, 1}, {1, -1, 2}, {2, 1, 0}, {0, 2, -1}},
                                    {0.3, 0.25, 0.2, 0.15, 0.1},
                                    {0.1, 1.0, 2.0, -0.7, 2.5});
    if (alternate)
        delta = plane_waves<3>({{2, 0, 1}, {0, 2, -1}, {1, 1, 0}, {1, -1, 2}},
                               {0.17, -0.23, 0.31, 0.09},
                               {1.2, -0.5, 2.1, 0.7});
    auto ref = reference_3LPT(delta);

    Grid delta_fourier = make_delta_fourier<3>(delta, Nmesh);

    // 1LPT and 2LPT from the separate functions
    Grid phi1, phi2;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta_fourier, phi1);
    FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_fourier, phi2, dealiasing);
    double e1 = max_diff_fourier<3>(phi1, ref.p1) / maxabs<3>(ref.p1);
    double e2 = max_diff_fourier<3>(phi2, ref.p2) / maxabs<3>(ref.p2);
    check(e1 < 1e-12, "compute_1LPT_potential_fourier matches the plane-wave reference (rel err " + num(e1) + ")");
    check(e2 < 1e-12, "compute_2LPT_potential_fourier matches the plane-wave reference (rel err " + num(e2) + ")");

    // 3LPT potentials
    Grid p1, p2, p3a, p3b;
    std::array<Grid, 3> A;
    FML::COSMOLOGY::LPT::compute_3LPT_potential_fourier<3>(delta_fourier, p1, p2, p3a, p3b, A, false, dealiasing);
    double r1 = max_diff_fourier<3>(p1, ref.p1) / maxabs<3>(ref.p1);
    double r2 = max_diff_fourier<3>(p2, ref.p2) / maxabs<3>(ref.p2);
    double r3a = max_diff_fourier<3>(p3a, ref.p3a) / maxabs<3>(ref.p3a);
    double r3b = max_diff_fourier<3>(p3b, ref.p3b) / maxabs<3>(ref.p3b);
    double rA = 0.0, Amax = 0.0;
    for (int i = 0; i < 3; i++)
        Amax = std::max(Amax, maxabs<3>(ref.A[i]));
    for (int i = 0; i < 3; i++)
        rA = std::max(rA, max_diff_fourier<3>(A[i], ref.A[i]) / Amax);
    check(r1 < 1e-12 and r2 < 1e-12,
          "3LPT: phi_1LPT and phi_2LPT match the reference (rel err " + num(r1) + ", " + num(r2) + ")");
    check(r3a < 1e-10, "3LPT: phi_3LPT_a matches the reference (rel err " + num(r3a) + ")");
    check(r3b < 1e-10, "3LPT: phi_3LPT_b matches the reference (rel err " + num(r3b) + ")");
    check(rA < 1e-10, "3LPT: curl term A_3LPT matches the reference (rel err " + num(rA) + ")");

    // Without the curl term the scalar potentials are the same
    {
        Grid q1, q2, q3a, q3b;
        std::array<Grid, 3> B;
        FML::COSMOLOGY::LPT::compute_3LPT_potential_fourier<3>(delta_fourier, q1, q2, q3a, q3b, B, true, dealiasing);
        double s3b = max_diff_fourier<3>(q3b, ref.p3b) / maxabs<3>(ref.p3b);
        check(s3b < 1e-10, "3LPT: phi_3LPT_b with ignore_curl_term (rel err " + num(s3b) + ")");
    }

    // The total displacement field and its time derivative
    const double dlogDdt = 0.7;
    std::array<Grid, 3> Psi, dPsidt;
    FML::COSMOLOGY::LPT::compute_3LPT_displacement_field<3>(delta_fourier, Psi, dPsidt, dlogDdt, false, dealiasing);
    double ePsi = 0.0, edPsi = 0.0;
    for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3, k = (i + 2) % 3;
        Modes<3> p3 = add<3>(ref.p3a, ref.p3b);
        Modes<3> curlA = add<3>(grad<3>(ref.A[k], j), grad<3>(ref.A[j], k), 1.0, -1.0);
        Modes<3> psi = add<3>(add<3>(add<3>(grad<3>(ref.p1, i), grad<3>(ref.p2, i)), grad<3>(p3, i)), curlA);
        Modes<3> dpsi =
            add<3>(add<3>(add<3>(grad<3>(ref.p1, i), grad<3>(ref.p2, i), 1.0, 2.0), grad<3>(p3, i), 1.0, 3.0),
                   curlA,
                   dlogDdt,
                   3.0 * dlogDdt);
        ePsi = std::max(ePsi, max_diff_real<3>(Psi[i], psi) / maxabs<3>(psi));
        edPsi = std::max(edPsi, max_diff_real<3>(dPsidt[i], dpsi) / maxabs<3>(dpsi));
    }
    check(ePsi < 1e-10, "compute_3LPT_displacement_field: Psi matches the reference (rel err " + num(ePsi) + ")");
    check(edPsi < 1e-10, "compute_3LPT_displacement_field: dPsidt matches the reference (rel err " + num(edPsi) + ")");
}

// High-frequency products deliberately cross the original mesh Nyquist limit.
// The reference convolves integer wavevectors without wrapping or intermediate
// truncation; only the final observable is restricted to the output mesh.
void test_dealiased_LPT() {
    using namespace FML::COSMOLOGY::LPT;
    using Grid = FML::GRID::FFTWGrid<3>;
    const auto delta = plane_waves<3>({{7, 1, 2}, {-6, 2, 1}, {1, 7, -5}, {3, -4, 6}},
                                      {0.17, -0.23, 0.31, 0.19},
                                      {1.2, -0.5, 2.1, 0.7});
    auto input = make_delta_fourier<3>(delta, Nmesh);
    auto reference = reference_3LPT(delta);
    auto crop_reference = [](Modes<3> modes) {
        for (auto it = modes.begin(); it != modes.end();) {
            bool keep = true;
            for (auto wave : it->first)
                keep = keep && 2 * std::abs(wave) < Nmesh;
            if (keep)
                ++it;
            else
                it = modes.erase(it);
        }
        return modes;
    };
    // Real-space sampling/FFT leakage is amplified by inverse Laplacians at low k.
    const double tolerance = std::max(5e-12, 500 * double(std::numeric_limits<FML::GRID::FloatType>::epsilon()));
    auto compare = [&](const Grid & field, const Modes<3> & modes, const std::string & label) {
        const auto cropped = crop_reference(modes);
        const double error = max_diff_fourier<3>(field, cropped) / std::max(1e-30, maxabs<3>(cropped));
        double source_error = 0.0, source_scale = 0.0;
        for (auto index : field.get_fourier_range()) {
            const auto wave = integer_wavevector<3>(field, index);
            const auto it = cropped.find(wave);
            const cplx expected = it == cropped.end() ? cplx(0) : it->second;
            const double k2 = k2_of<3>(wave);
            source_error = std::max(source_error, k2 * std::abs(cplx(field.get_fourier_from_index(index)) - expected));
            source_scale = std::max(source_scale, k2 * std::abs(expected));
        }
        FML::MaxOverTasks(&source_error);
        FML::MaxOverTasks(&source_scale);
        source_error /= std::max(1e-30, source_scale);
        // Inverse Laplacians can amplify float source round-off by O(Nmesh^2)
        // relative to these deliberately high-frequency reference potentials.
        const double potential_tolerance =
            std::max(5e-12, 32 * Nmesh * Nmesh * double(std::numeric_limits<FML::GRID::FloatType>::epsilon()));
        check(error < potential_tolerance && source_error < tolerance,
              "de-aliased " + label + " matches unwrapped convolution (potential rel err " + num(error) +
                  ", source rel err " + num(source_error) + ")");
    };
    Grid p1, p2, pa, pb, quadratic, legacy;
    std::array<Grid, 3> A;
    compute_2LPT_potential_fourier(input, quadratic, Dealiasing::Padding);
    compare(quadratic, reference.p2, "standalone 2LPT");
    compute_2LPT_potential_fourier(input, legacy);
    check(max_diff_fourier<3>(legacy, crop_reference(reference.p2)) > 1e-7,
          "high-frequency fixture exposes legacy quadratic aliasing");
    for (bool ignore_curl : {false, true}) {
        compute_3LPT_potential_fourier<3>(input, p1, p2, pa, pb, A, ignore_curl, Dealiasing::Padding);
        compare(p1, reference.p1, "1LPT");
        compare(p2, reference.p2, "2LPT in 3LPT");
        compare(pa, reference.p3a, "3LPT-a");
        compare(pb, reference.p3b, "3LPT-b (including unresolved intermediate 2LPT modes)");
        if (!ignore_curl)
            for (int d = 0; d < 3; d++)
                compare(A[d], reference.A[d], "curl " + std::to_string(d));
    }
    // Nyquist planes have no unambiguous signed wavevector on an even grid.
    // Explicitly adding them must not contaminate any de-aliased output.
    auto contaminated = input;
    for (auto index : contaminated.get_fourier_range()) {
        auto wave = integer_wavevector<3>(contaminated, index);
        if (std::abs(wave[0]) == Nmesh / 2 || std::abs(wave[1]) == Nmesh / 2 || wave[2] == Nmesh / 2)
            contaminated.set_fourier_from_index(index, 0.03);
    }
    compute_2LPT_potential_fourier(contaminated, quadratic, Dealiasing::Padding);
    compare(quadratic, reference.p2, "2LPT excludes input/output Nyquist planes");
    compute_3LPT_potential_fourier<3>(input, p1, p2, pa, pb, A, false);
    for (const auto & entry : std::vector<std::pair<const Grid *, const Modes<3> *>>{{&p2, &reference.p2},
                                                                                     {&pa, &reference.p3a},
                                                                                     {&pb, &reference.p3b},
                                                                                     {&A[0], &reference.A[0]}}) {
        const auto cropped = crop_reference(*entry.second);
        const double error = max_diff_fourier<3>(*entry.first, cropped) / maxabs<3>(cropped);
        check(error > 1e-3, "high-frequency fixture exposes legacy nonlinear aliasing (rel err " + num(error) + ")");
    }
    auto padded = FML::GRID::resize_fourier_grid(input, detail::padded_lpt_mesh(Nmesh, 3));
    auto restored = FML::GRID::resize_fourier_grid(padded, Nmesh, 1, 1);
    compare(restored, delta, "Fourier padding/cropping normalization");
    check(max_diff_fourier<3>(input, delta) < tolerance, "de-aliasing preserves the input grid");
    check(restored.get_n_extra_slices_left() == 1 && restored.get_n_extra_slices_right() == 1,
          "Fourier cropping preserves requested boundary allocation");
    std::array<Grid, 3> psi, velocity;
    constexpr double rate = 0.63;
    compute_3LPT_displacement_field<3>(input, psi, velocity, rate, false, Dealiasing::Padding);
    auto total = add<3>(add<3>(add<3>(reference.p1, reference.p2), reference.p3a), reference.p3b);
    auto derivative =
        add<3>(add<3>(add<3>(reference.p1, reference.p2, 1, 2), reference.p3a, 1, 3), reference.p3b, 1, 3);
    for (int d = 0; d < 3; d++) {
        int j = (d + 1) % 3, k = (d + 2) % 3;
        auto curl = add<3>(grad<3>(reference.A[k], j), grad<3>(reference.A[j], k), 1, -1);
        auto expected = crop_reference(add<3>(grad<3>(total, d), curl));
        auto expected_velocity = crop_reference(add<3>(grad<3>(derivative, d), curl, rate, 3 * rate));
        check(max_diff_real<3>(psi[d], expected) < tolerance, "de-aliased displacement component " + std::to_string(d));
        check(max_diff_real<3>(velocity[d], expected_velocity) < tolerance,
              "de-aliased velocity component " + std::to_string(d));
    }
    // Exercise 2D and odd input meshes as well as cross-rank slab remapping.
    std::vector<int> meshes{16};
    if (FML::NTasks == 1)
        meshes.push_back(15);
    for (int mesh : meshes) {
        auto modes = plane_waves<2>({{6, 1}, {-5, 3}}, {0.2, 0.3}, {0.2, 0.7});
        auto small = make_delta_fourier<2>(modes, mesh);
        auto large = FML::GRID::resize_fourier_grid(small, detail::padded_lpt_mesh(mesh, 3));
        auto back = FML::GRID::resize_fourier_grid(large, mesh);
        check(max_diff_fourier<2>(back, modes) < tolerance,
              "2D Fourier resize round trip (mesh " + std::to_string(mesh) + ")");
        FML::GRID::FFTWGrid<2> q1, q2, qa, qb;
        std::array<FML::GRID::FFTWGrid<2>, 2> B;
        compute_3LPT_potential_fourier<2>(small, q1, q2, qa, qb, B, false, Dealiasing::Padding);
        Modes<3> embedded;
        for (auto & [wave, value] : modes)
            embedded[{wave[0], wave[1], 0}] = value;
        auto exact = reference_3LPT(embedded);
        // Preserve the library's existing 2D definitions: 3a uses a 2x2
        // determinant; 3b uses H1_xx H2_yy - H1_xy H2_xy.
        exact.p3a = add<3>(exact.p2, {}, -7.0 / 9.0);
        const auto chi = add<3>(exact.p2, {}, -7.0 / 3.0);
        auto mixed = add<3>(mul<3>(hess<3>(exact.p1, 0, 0), hess<3>(chi, 1, 1)),
                            mul<3>(hess<3>(exact.p1, 0, 1), hess<3>(chi, 0, 1)),
                            1,
                            -1);
        exact.p3b = add<3>(solve<3>(mixed), {}, 10.0 / 21.0);
        auto project = [mesh](const Modes<3> & field) {
            Modes<2> result;
            for (auto & [wave, value] : field)
                if (2 * std::abs(wave[0]) < mesh && 2 * std::abs(wave[1]) < mesh)
                    result[{wave[0], wave[1]}] = value;
            return result;
        };
        for (const auto & entry :
             std::vector<std::pair<const FML::GRID::FFTWGrid<2> *, Modes<2>>>{{&q2, project(exact.p2)},
                                                                              {&qa, project(exact.p3a)},
                                                                              {&qb, project(exact.p3b)},
                                                                              {&B[0], project(exact.A[2])}}) {
            double error = max_diff_fourier<2>(*entry.first, entry.second);
            check(error < tolerance,
                  "2D padded nonlinear potential matches unwrapped convolution (mesh " + std::to_string(mesh) +
                      ", abs err " + num(error) + ")");
        }
    }
}

void test_3LPT_2D() {
    using Grid = FML::GRID::FFTWGrid<2>;
    Modes<2> delta = plane_waves<2>({{1, 0}, {1, 2}, {0, 1}, {2, -1}}, {0.3, 0.2, 0.25, 0.1}, {0.3, 1.1, -0.4, 2.0});
    Grid delta_fourier = make_delta_fourier<2>(delta, Nmesh);
    Grid p1, p2, p3a, p3b;
    std::array<Grid, 2> A;
    FML::COSMOLOGY::LPT::compute_3LPT_potential_fourier<2>(delta_fourier, p1, p2, p3a, p3b, A, false);

    // Psi = grad(phi_1 + phi_2 + phi_3a + phi_3b) + (d_y A, -d_x A) computed with the 1LPT helpers
    Grid sum = p1, sumd = p1, Az = A[0];
    for (int islice = 0; islice < sum.get_local_nx(); islice++)
        for (auto && fi : sum.get_fourier_range(islice, islice + 1)) {
            auto v3 = p3a.get_fourier_from_index(fi) + p3b.get_fourier_from_index(fi);
            sum.set_fourier_from_index(fi, p1.get_fourier_from_index(fi) + p2.get_fourier_from_index(fi) + v3);
            sumd.set_fourier_from_index(fi,
                                        p1.get_fourier_from_index(fi) +
                                            FML::GRID::FloatType(2.0) * p2.get_fourier_from_index(fi) +
                                            FML::GRID::FloatType(3.0) * v3);
        }
    std::array<Grid, 2> gpsi, gdpsi, gA;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<2>(sum, gpsi);
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<2>(sumd, gdpsi);
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<2>(Az, gA);

    const double dlogDdt = 1.3;
    std::array<Grid, 2> Psi, dPsidt;
    FML::COSMOLOGY::LPT::compute_3LPT_displacement_field<2>(delta_fourier, Psi, dPsidt, dlogDdt, false);
    double e = 0.0, ed = 0.0, scale = 0.0;
    for (int islice = 0; islice < Psi[0].get_local_nx(); islice++)
        for (auto && ri : Psi[0].get_real_range(islice, islice + 1)) {
            double curl[2] = {gA[1].get_real_from_index(ri), -gA[0].get_real_from_index(ri)};
            for (int i = 0; i < 2; i++) {
                double ref = gpsi[i].get_real_from_index(ri) + curl[i];
                double refd = dlogDdt * (gdpsi[i].get_real_from_index(ri) + 3.0 * curl[i]);
                scale = std::max(scale, std::fabs(ref));
                e = std::max(e, std::fabs(Psi[i].get_real_from_index(ri) - ref));
                ed = std::max(ed, std::fabs(dPsidt[i].get_real_from_index(ri) - refd));
            }
        }
    FML::MaxOverTasks(&e);
    FML::MaxOverTasks(&ed);
    FML::MaxOverTasks(&scale);
    check(e / scale < 1e-12 and ed / scale < 1e-12 and scale > 0.0,
          "compute_3LPT_displacement_field<2>: Psi and dPsidt agree with the 2D potentials (rel err " + num(e / scale) +
              ", " + num(ed / scale) + ")");
}

void test_ALPT() {
    using Grid = FML::GRID::FFTWGrid<3>;
    Modes<3> delta = plane_waves<3>({{1, 0, 0}, {0, 1, 1}, {1, -1, 2}}, {0.6, 0.5, 0.4}, {0.1, 1.0, 2.0});
    Grid delta_fourier = make_delta_fourier<3>(delta, Nmesh);

    // Spherical collapse: Laplacian(phi_SC) = 3(sqrt(max(1 - 2/3 D delta, 0)) - 1)
    const double DoverDini = 1.5;
    Grid sc;
    FML::COSMOLOGY::LPT::compute_spherical_collapse_potential(delta_fourier, sc, DoverDini);
    Grid source(Nmesh, 1, 1);
    for (int islice = 0; islice < source.get_local_nx(); islice++)
        for (auto && ri : source.get_real_range(islice, islice + 1)) {
            auto c = source.get_coord_from_index(ri);
            std::array<double, 3> x{double(c[0] + source.get_local_x_start()) / Nmesh,
                                    double(c[1]) / Nmesh,
                                    double(c[2]) / Nmesh};
            double v = 1.0 - 2.0 / 3.0 * DoverDini * eval<3>(delta, x);
            source.set_real_from_index(ri, 3.0 * (std::sqrt(std::max(v, 0.0)) - 1.0));
        }
    source.fftw_r2c();
    double esc = 0.0, scmax = 0.0;
    for (int islice = 0; islice < sc.get_local_nx(); islice++)
        for (auto && fi : sc.get_fourier_range(islice, islice + 1)) {
            std::array<double, 3> kvec;
            double k2;
            sc.get_fourier_wavevector_and_norm2_by_index(fi, kvec, k2);
            cplx ref = k2 > 0.0 ? cplx(-source.get_fourier_from_index(fi) / FML::GRID::FloatType(k2)) : cplx(0.0);
            cplx v = sc.get_fourier_from_index(fi);
            double d = std::abs(v - ref);
            esc = std::max(esc, d == d ? d : 1e100);
            scmax = std::max(scmax, std::abs(ref));
        }
    FML::MaxOverTasks(&esc);
    FML::MaxOverTasks(&scmax);
    check(esc / scmax < 1e-12,
          "spherical collapse potential = -source/k^2 with k=0 mode zero (rel err " + num(esc / scmax) + ")");

    // To linear order phi_SC = D phi_1LPT
    {
        Modes<3> tiny = plane_waves<3>({{1, 0, 0}, {0, 1, 1}}, {1e-6, 1e-6}, {0.1, 1.0});
        Grid dt = make_delta_fourier<3>(tiny, Nmesh);
        Grid sct;
        FML::COSMOLOGY::LPT::compute_spherical_collapse_potential(dt, sct, 1.0);
        Modes<3> p1 = solve<3>(tiny);
        for (auto & [n, c] : p1)
            c = -c;
        double e = max_diff_fourier<3>(sct, p1) / maxabs<3>(p1);
        check(e < 1e-5, "spherical collapse potential = phi_1LPT to linear order (rel err " + num(e) + ")");
    }

    // ALPT: finite, and the two limits of the smoothing
    Grid phi1, phi2, alpt0, alptinf, sc1;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta_fourier, phi1);
    FML::COSMOLOGY::LPT::compute_2LPT_potential_fourier(delta_fourier, phi2);
    FML::COSMOLOGY::LPT::compute_ALPT_potential_fourier(delta_fourier, phi2, alpt0, 0.0, "sharpk");
    FML::COSMOLOGY::LPT::compute_ALPT_potential_fourier(delta_fourier, phi2, alptinf, 1e6, "sharpk");
    FML::COSMOLOGY::LPT::compute_spherical_collapse_potential(delta_fourier, sc1, 1.0);
    check(all_finite_fourier<3>(alpt0) and all_finite_fourier<3>(alptinf), "ALPT potential is finite (no NaN)");
    double e0 = 0.0, einf = 0.0, s0 = 0.0, sinf = 0.0;
    for (int islice = 0; islice < alpt0.get_local_nx(); islice++)
        for (auto && fi : alpt0.get_fourier_range(islice, islice + 1)) {
            cplx ref0 = phi1.get_fourier_from_index(fi) + phi2.get_fourier_from_index(fi);
            cplx refinf = sc1.get_fourier_from_index(fi);
            double d0 = std::abs(cplx(alpt0.get_fourier_from_index(fi)) - ref0);
            double dinf = std::abs(cplx(alptinf.get_fourier_from_index(fi)) - refinf);
            e0 = std::max(e0, d0 == d0 ? d0 : 1e100);
            einf = std::max(einf, dinf == dinf ? dinf : 1e100);
            s0 = std::max(s0, std::abs(ref0));
            sinf = std::max(sinf, std::abs(refinf));
        }
    FML::MaxOverTasks(&e0);
    FML::MaxOverTasks(&einf);
    FML::MaxOverTasks(&s0);
    FML::MaxOverTasks(&sinf);
    check(e0 / s0 < 1e-12, "ALPT with no smoothing (R=0) = phi_1LPT + phi_2LPT (rel err " + num(e0 / s0) + ")");
    check(einf / sinf < 1e-12, "ALPT with total smoothing = phi_SC (rel err " + num(einf / sinf) + ")");
}

void test_nongaussian_dc_mode() {
    using Grid = FML::GRID::FFTWGrid<3>;
    // P_primordial(k)/V and P(k)/V are both 0 at k = 0 so their ratio is 0/0 = NaN there
    auto Pprim = [](double kB) { return kB == 0.0 ? 0.0 : 1e-9 * std::pow(kB, -3.0); };
    auto Pdelta = [](double kB) { return kB == 0.0 ? 0.0 : 1e-3 * std::pow(kB, -1.5); };
    auto ratio = [&](double kB) { return Pdelta(kB) / Pprim(kB); };

    Grid delta(Nmesh, 1, 1), phi(Nmesh, 1, 1);
    FML::RANDOM::GSLRandomGenerator rng1(42), rng2(42);
    FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier_cosmology<3>(delta,
                                                                                           &rng1,
                                                                                           ratio,
                                                                                           Pprim,
                                                                                           false,
                                                                                           100.0,
                                                                                           "local");
    FML::RANDOM::NONGAUSSIAN::generate_nonlocal_gaussian_random_field_fourier<3>(phi,
                                                                                 &rng2,
                                                                                 Pprim,
                                                                                 false,
                                                                                 100.0,
                                                                                 "local");
    check(all_finite_fourier<3>(delta), "non-gaussian field with P/P_primordial = 0/0 at k=0 is finite");
    // The other modes are sqrt(ratio) * phi and the DC mode is zero
    double e = 0.0, s = 0.0;
    for (int islice = 0; islice < delta.get_local_nx(); islice++)
        for (auto && fi : delta.get_fourier_range(islice, islice + 1)) {
            std::array<double, 3> kvec;
            double kmag;
            delta.get_fourier_wavevector_and_norm_by_index(fi, kvec, kmag);
            cplx ref = kmag == 0.0 ? cplx(0.0) : cplx(phi.get_fourier_from_index(fi)) * std::sqrt(ratio(kmag));
            double d = std::abs(cplx(delta.get_fourier_from_index(fi)) - ref);
            e = std::max(e, d == d ? d : 1e100);
            s = std::max(s, std::abs(ref));
        }
    FML::MaxOverTasks(&e);
    FML::MaxOverTasks(&s);
    check(e / s < 1e-12,
          "non-gaussian field: delta = sqrt(P/P_primordial) phi and delta(k=0) = 0 (rel err " + num(e / s) + ")");
    delta.fftw_c2r();
    int ok = 1;
    for (int islice = 0; islice < delta.get_local_nx(); islice++)
        for (auto && ri : delta.get_real_range(islice, islice + 1))
            if (not std::isfinite(double(delta.get_real_from_index(ri))))
                ok = 0;
    FML::MinOverTasks(&ok);
    check(ok == 1, "non-gaussian field is finite in real space");
}

void test_growth_factors() {
    // w = -0.8 background (not LCDM): D'' + (2 + dlogH/dloga) D' = 1.5 OmegaM / (a^3 H^2) D (x = log a)
    const double Om = 0.3, w = -0.8, zini = 49.0;
    auto H = [=](double a) { return std::sqrt(Om / (a * a * a) + (1 - Om) * std::pow(a, -3.0 * (1.0 + w))); };
    auto dlogHdx = [=](double a) {
        double H2 = Om / (a * a * a) + (1 - Om) * std::pow(a, -3.0 * (1.0 + w));
        double dH2dx = -3.0 * Om / (a * a * a) - 3.0 * (1.0 + w) * (1 - Om) * std::pow(a, -3.0 * (1.0 + w));
        return 0.5 * dH2dx / H2;
    };
    FML::INTERPOLATION::SPLINE::Spline D1, D2, D3a, D3b;
    FML::COSMOLOGY::LPT::compute_LPT_growth_factors_LCDM(Om, zini, H, D1, D2, D3a, D3b);

    // RK4 from a = 1e-3 (growing mode D = a, D' = D) to a = 1 with an analytic dlogH/dx
    auto rhs = [&](double x, double y0, double y1, double & d0, double & d1) {
        double a = std::exp(x), h = H(a);
        d0 = y1;
        d1 = 1.5 * Om / (a * a * a * h * h) * y0 - (2.0 + dlogHdx(a)) * y1;
    };
    auto integrate_to = [&](double xend) {
        double x = std::log(1e-3), y0 = 1.0, y1 = 1.0;
        const int n = 20000;
        const double dx = (xend - x) / n;
        for (int i = 0; i < n; i++) {
            double k1a, k1b, k2a, k2b, k3a, k3b, k4a, k4b;
            rhs(x, y0, y1, k1a, k1b);
            rhs(x + dx / 2, y0 + dx / 2 * k1a, y1 + dx / 2 * k1b, k2a, k2b);
            rhs(x + dx / 2, y0 + dx / 2 * k2a, y1 + dx / 2 * k2b, k3a, k3b);
            rhs(x + dx, y0 + dx * k3a, y1 + dx * k3b, k4a, k4b);
            y0 += dx / 6 * (k1a + 2 * k2a + 2 * k3a + k4a);
            y1 += dx / 6 * (k1b + 2 * k2b + 2 * k3b + k4b);
            x += dx;
        }
        return y0;
    };
    const double Dref = integrate_to(0.0) / integrate_to(std::log(1.0 / (1.0 + zini)));
    const double Dfml = D1(0.0);
    check(std::fabs(Dfml / Dref - 1.0) < 1e-5,
          "growth factor D1(a=1) for w=-0.8: " + num(Dfml) + " vs ODE " + num(Dref) + " (rel err " +
              num(Dfml / Dref - 1.0) + ")");

    // EdS: D1 = a/aini and D2 = -3/7 D1^2, D3a = -1/3 D1^3, D3b = 5/21 D1^3 exactly (also at a = 1)
    auto HEdS = [](double a) { return std::pow(a, -1.5); };
    FML::COSMOLOGY::LPT::compute_LPT_growth_factors_LCDM(1.0, zini, HEdS, D1, D2, D3a, D3b);
    auto growth = FML::COSMOLOGY::growth_history_from_log_a_splines(D1, D2);
    const double expected_linear = D1(std::log(0.5));
    const double expected_second = D2(std::log(0.5));
    check(growth.linear(0.5) == expected_linear && growth.second_order(0.5) == expected_second &&
              std::abs(growth.linear_rate(0.5) - 1.0) < 1e-5 && std::abs(growth.second_order_rate(0.5) - 2.0) < 1e-5,
          "shared growth history preserves LPT normalization and signed second-order rates");
    double d1 = D1(0.0);
    double e = std::fabs(d1 / (1.0 + zini) - 1.0);
    e = std::max(e, std::fabs(D2(0.0) / (d1 * d1) + 3.0 / 7.0) / (3.0 / 7.0));
    e = std::max(e, std::fabs(D3a(0.0) / (d1 * d1 * d1) + 1.0 / 3.0) / (1.0 / 3.0));
    e = std::max(e, std::fabs(D3b(0.0) / (d1 * d1 * d1) - 5.0 / 21.0) / (5.0 / 21.0));
    check(e < 1e-5,
          "EdS growth factors at a=1: D1 = a/aini, D2/D1^2 = -3/7, D3a/D1^3 = -1/3, D3b/D1^3 = 5/21 "
          "(max rel err " +
              num(e) + ")");
    D1.free();
    D2.free();
    check(growth.linear(0.5) == expected_linear && growth.second_order(0.5) == expected_second,
          "shared growth history owns spline data after source splines are released");
}

void test_random() {
    // set_seed must reset the cached number of std::normal_distribution
    FML::RANDOM::RandomGenerator a(1), b(5), c(std::vector<unsigned int>(624, 5));
    a.generate_normal();
    a.set_seed(5u);
    double x1 = a.generate_normal(), x2 = a.generate_normal();
    double y1 = b.generate_normal(), y2 = b.generate_normal();
    check(x1 == y1 and x2 == y2, "RandomGenerator::set_seed(unsigned) resets the normal distribution");
    FML::RANDOM::RandomGenerator d(2);
    d.generate_normal(); // the second number of the pair is now cached
    d.set_seed(std::vector<unsigned int>(624, 5));
    double z1 = d.generate_normal();
    check(z1 == c.generate_normal(), "RandomGenerator::set_seed(vector) resets the normal distribution");

    // White noise: each slice has its own seed so the slices must not start with the same number
    FML::GRID::FFTWGrid<3> g(Nmesh);
    FML::RANDOM::RandomGenerator rng(1234);
    rng.generate_normal(); // a normal number drawn earlier (leaves a cached value in the distribution)
    FML::RANDOM::GAUSSIAN::generate_white_noise_field_real<3>(g, &rng);
    std::vector<double> first(Nmesh, 0.0);
    for (int ix = 0; ix < g.get_local_nx(); ix++)
        first[ix + g.get_local_x_start()] = g.get_real_from_index(g.get_index_real({ix, 0, 0}));
    FML::SumArrayOverTasks(first.data(), Nmesh);
    int ndup = 0;
    for (int i = 0; i < Nmesh; i++)
        for (int j = i + 1; j < Nmesh; j++)
            if (first[i] == first[j])
                ndup++;
    check(ndup == 0,
          "white noise: the slices do not start with the same value (" + std::to_string(ndup) + " equal pairs)");
}

int main() {
    test_3LPT();
    test_3LPT(true); // Different tensor cross terms and signs for streamed Hessian accumulation
    test_3LPT(false, FML::COSMOLOGY::LPT::Dealiasing::Padding);
    test_dealiased_LPT();
    test_3LPT_2D();
    test_ALPT();
    test_nongaussian_dc_mode();
    test_growth_factors();
    test_random();
    check(lpt_odr_check() == 1,
          "GaussianRandomField.h and DisplacementFields.h can be included in several translation units");

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All LPT tests passed\n" : std::to_string(fails) + " LPT test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
