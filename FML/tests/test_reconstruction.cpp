//==========================================================
// Tests for the RSD reconstruction (LPT/Reconstruction.h), the Hessian (TidalTensor/Hessian.h) and the Fourier
// smoothing and grid PDF (Smoothing/SmoothingFourier.h) against analytic results:
//   - smoothing filters of a single Fourier mode (gaussian, tophat, sharpk in 3D, tophat = 2 J1(kR)/kR in 2D)
//   - the Hessian of a sum of Fourier modes (3D and 2D), of its potential, the mixed derivatives of a Nyquist mode
//   - eigenvalues and eigenvectors of the Hessian (M v = lambda v, with OpenMP threads) against GSL per cell
//   - compute_grid_PDF: reused output vectors, the largest value, a constant grid
//   - RSD removal of Zeldovich-displaced particles: the removed shift matches the RSD shift that was put in and the
//     result converges with the number of iterations (same result for any number of MPI tasks)
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/Reconstruction.h>
#include <FML/Smoothing/SmoothingFourier.h>
#include <FML/TidalTensor/Hessian.h>

#include <gsl/gsl_eigen.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

template <int N>
using Grid = FML::GRID::FFTWGrid<N>;
const double tp = 2.0 * M_PI;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os << x;
    return os.str();
}

// Largest |g - f(x)| over the grid (all tasks)
template <int N>
double maxdiff(Grid<N> & g, std::function<double(std::array<double, N> &)> f) {
    double md = 0.0;
    for (int islice = 0; islice < g.get_local_nx(); islice++)
        for (auto && ri : g.get_real_range(islice, islice + 1)) {
            auto x = g.get_real_position(g.get_coord_from_index(ri));
            md = std::max(md, std::fabs(g.get_real_from_index(ri) - f(x)));
        }
    FML::MaxOverTasks(&md);
    return md;
}

//==========================================================
// RSD reconstruction of Zeldovich-displaced particles
//==========================================================
struct ReconParticle {
    double pos[3];
    double xreal[3]; // real-space position
    double s0[3];    // redshift-space position
    double * get_pos() { return pos; }
    constexpr int get_ndim() const { return 3; }
};
uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
double periodic(double d) { return d - std::round(d); }

// Returns {rms of the RSD shift put in, slope of the removed shift against it, rms of the residual} in cells
std::array<double, 3> reconstruction_test(int Nmesh, int niter, double beta, int los) {
    // Gaussian random field (task independent) with a smooth cut off, displacements with rms 0.5 cells
    Grid<3> delta(Nmesh, 1, 1);
    delta.fill_real_grid([&](std::array<double, 3> & x) {
        uint64_t i0 = llround(x[0] * Nmesh) % Nmesh, i1 = llround(x[1] * Nmesh) % Nmesh,
                 i2 = llround(x[2] * Nmesh) % Nmesh;
        uint64_t a = splitmix(i0 * Nmesh * Nmesh + i1 * Nmesh + i2 + 12345), b = splitmix(a);
        double u1 = (a >> 11) * (1.0 / 9007199254740992.0) + 1e-300, u2 = (b >> 11) * (1.0 / 9007199254740992.0);
        return std::sqrt(-2 * std::log(u1)) * std::cos(tp * u2);
    });
    delta.fftw_r2c();
    const double kc = tp * Nmesh / 8.0;
    FML::GRID::multiply_fourier_by_function_of_k<3>(delta, [&](double k) {
        return k == 0 ? 0.0 : std::exp(-0.5 * k * k / (kc * kc));
    });
    Grid<3> phi;
    FML::COSMOLOGY::LPT::compute_1LPT_potential_fourier(delta, phi);
    std::array<Grid<3>, 3> Psi;
    FML::COSMOLOGY::LPT::from_LPT_potential_to_displacement_vector<3>(phi, Psi);
    double s2 = 0;
    for (int d = 0; d < 3; d++)
        for (auto && ri : Psi[d].get_real_range())
            s2 += std::pow(Psi[d].get_real_from_index(ri), 2);
    FML::SumOverTasks(&s2);
    const double scale = 0.5 / Nmesh / std::sqrt(s2 / (3.0 * std::pow(Nmesh, 3)));
    for (int d = 0; d < 3; d++) {
        for (auto && ri : Psi[d].get_real_range())
            Psi[d].set_real_from_index(ri, Psi[d].get_real_from_index(ri) * scale);
        Psi[d].communicate_boundaries();
    }

    // Particles: lattice + Psi (real space), + f Psi_los (redshift space, f = beta with bias 1)
    FML::PARTICLE::MPIParticles<ReconParticle> part;
    part.create_particle_grid(Nmesh, 3.0, FML::xmin_domain, FML::xmax_domain);
    std::array<std::vector<FML::GRID::FloatType>, 3> Pq;
    FML::INTERPOLATION::interpolate_grid_vector_to_particle_positions<3, ReconParticle>(Psi,
                                                                                        part.get_particles_ptr(),
                                                                                        part.get_npart(),
                                                                                        Pq,
                                                                                        "CIC");
    auto * p = part.get_particles_ptr();
    for (size_t i = 0; i < part.get_npart(); i++) {
        for (int d = 0; d < 3; d++) {
            double x = p[i].pos[d] + Pq[d][i];
            x -= std::floor(x);
            p[i].xreal[d] = x >= 1.0 ? 0.0 : x;
            double s = p[i].xreal[d] + (d == los ? beta * Pq[d][i] : 0.0);
            s -= std::floor(s);
            p[i].pos[d] = p[i].s0[d] = s >= 1.0 ? 0.0 : s;
        }
    }
    part.communicate_particles();

    std::vector<double> los_direction(3, 0.0);
    los_direction[los] = 1.0;
    FML::COSMOLOGY::LPT::RSDReconstructionFourierMethod<3, ReconParticle>(part,
                                                                          "CIC",
                                                                          los_direction,
                                                                          Nmesh,
                                                                          niter,
                                                                          beta,
                                                                          {"gaussian", 0.3 / Nmesh},
                                                                          false);

    double S[5] = {0, 0, 0, 0, 0};
    p = part.get_particles_ptr();
    for (size_t i = 0; i < part.get_npart(); i++) {
        const double in = periodic(p[i].s0[los] - p[i].xreal[los]);
        const double removed = periodic(p[i].s0[los] - p[i].pos[los]);
        S[0] += in * in;
        S[1] += in * removed;
        S[2] += std::pow(periodic(p[i].pos[los] - p[i].xreal[los]), 2);
        S[3] += 1.0;
        for (int d = 0; d < 3; d++)
            if (d != los)
                S[4] += std::fabs(periodic(p[i].pos[d] - p[i].s0[d]));
    }
    for (auto & s : S)
        FML::SumOverTasks(&s);
    return {std::sqrt(S[0] / S[3]) * Nmesh, S[1] / S[0], S[4] > 0 ? -1.0 : std::sqrt(S[2] / S[3]) * Nmesh};
}

template <int N>
void test_tensor_empty_storage() {
    Grid<N> source(16, 1, 2);
    source.fill_fourier_grid({17, -9});
    source.fill_real_grid([](std::array<double, N> & x) { return 1 + std::cos(tp * x[0]); });
    auto snapshot = source;
    auto blank = source.clone_empty();
    bool good = blank.get_nmesh() == 16 && blank.get_n_extra_slices_left() == 1 &&
                blank.get_n_extra_slices_right() == 2 && blank.get_grid_status_real();
    for (auto i : blank.get_real_range())
        good &= blank.get_real_from_index(i) == 0;
    const auto left = source.get_n_extra_slices_left() * source.get_ntot_real_slice_alloc() / 2;
    for (ptrdiff_t i = -left; i < source.get_ntot_fourier_alloc() - left; ++i) {
        good &= source.get_fourier_from_index(i) == snapshot.get_fourier_from_index(i);
        if (i < 0 || i >= source.get_ntot_fourier())
            good &= blank.get_fourier_from_index(i) == source.get_fourier_from_index(i);
    }
    auto * a = source.get_real_grid();
    auto * b = blank.get_real_grid();
    const ptrdiff_t stride = 2 * (16 / 2 + 1);
    for (ptrdiff_t row = 0; row < 2 * source.get_ntot_fourier(); row += stride)
        for (int j = 16; j < stride; ++j)
            good &= a[row + j] == b[row + j];
    source.set_grid_status_real(false);
    auto fourier_blank = source.clone_empty();
    good &= !fourier_blank.get_grid_status_real();
    for (auto i : fourier_blank.get_fourier_range())
        good &= fourier_blank.get_fourier_from_index(i) == FML::GRID::ComplexType(0);
    source.set_grid_status_real(true);
    std::vector<Grid<N>> hessian, eval, evec;
    FML::HESSIAN::ComputeHessianWithFT(source, hessian);
    double error = maxdiff<N>(hessian[0], [](std::array<double, N> & x) { return -tp * tp * std::cos(tp * x[0]); });
    for (size_t j = 1; j < hessian.size(); ++j)
        error = std::max(error, maxdiff<N>(hessian[j], [](std::array<double, N> &) { return 0.; }));
    FML::HESSIAN::SymmetricTensorEigensystem(hessian, eval, evec, true);
    for (auto * outputs : {&eval, &evec})
        for (auto & g : *outputs) {
            good &= g.get_n_extra_slices_left() == 1 && g.get_n_extra_slices_right() == 2 && g.get_grid_status_real();
            for (ptrdiff_t i = -left; i < g.get_ntot_fourier_alloc() - left; ++i)
                if (i < 0 || i >= g.get_ntot_fourier())
                    good &= g.get_fourier_from_index(i) == hessian[0].get_fourier_from_index(i);
        }
    int bad = !good;
    FML::MaxOverTasks(&bad);
    check(!bad && error < 1e-9,
          std::to_string(N) + "D empty tensor allocation preserves input, halos/padding and analytic derivatives");
}

int main() {
    test_tensor_empty_storage<2>();
    test_tensor_empty_storage<3>();
    const int Nmesh = 32;

    // Smoothing of a single mode in 3D
    {
        std::array<int, 3> m = {3, -1, 2};
        const double k = tp * std::sqrt(14.0), R = 0.04, x = k * R;
        auto mode = [&](std::array<double, 3> & y) { return std::cos(tp * (m[0] * y[0] + m[1] * y[1] + m[2] * y[2])); };
        for (std::string filter : {"gaussian", "tophat", "sharpk"}) {
            Grid<3> g(Nmesh);
            g.fill_real_grid(mode);
            g.fftw_r2c();
            FML::GRID::smoothing_filter_fourier_space(g, R, filter);
            g.fftw_c2r();
            const double W = filter == "gaussian" ? std::exp(-x * x / 2) :
                             filter == "tophat"   ? 3 * (std::sin(x) - x * std::cos(x)) / (x * x * x) :
                                                    (x < 1 ? 1.0 : 0.0);
            check(maxdiff<3>(g, [&](std::array<double, 3> & y) { return W * mode(y); }) < 1e-12,
                  "3D " + filter + " filter of a single mode");
        }
    }

    // 2D tophat: 2 J1(kR) / kR (value from scipy)
    {
        Grid<2> g(Nmesh);
        auto mode = [&](std::array<double, 2> & y) { return std::cos(tp * (3 * y[0] + 2 * y[1])); };
        g.fill_real_grid(mode);
        g.fftw_r2c();
        FML::GRID::smoothing_filter_fourier_space(g, 0.1, "tophat");
        g.fftw_c2r();
        const double W = 0.4818942589026656; // kR = 2pi sqrt(13) 0.1
        const double md = maxdiff<2>(g, [&](std::array<double, 2> & y) { return W * mode(y); });
        check(md < 1e-7, "2D tophat filter = 2 J1(kR) / kR (error " + num(md) + ")");
    }

    // Hessian of a sum of modes and of its potential
    auto phi = [](std::array<double, 3> & x) {
        return std::cos(tp * (x[0] + 2 * x[1])) + std::sin(tp * (3 * x[2] - x[1])) +
               0.5 * std::cos(tp * x[0]) * std::cos(tp * x[2]);
    };
    auto H = [](int i, int j, std::array<double, 3> & x) {
        const double a = tp * (x[0] + 2 * x[1]), b = tp * (3 * x[2] - x[1]);
        const int ka[3] = {1, 2, 0}, kb[3] = {0, -1, 3};
        double v = -tp * tp * ka[i] * ka[j] * std::cos(a) - tp * tp * kb[i] * kb[j] * std::sin(b);
        const double cx = std::cos(tp * x[0]), sx = std::sin(tp * x[0]), cz = std::cos(tp * x[2]),
                     sz = std::sin(tp * x[2]);
        if ((i == 0 and j == 0) or (i == 2 and j == 2))
            v += -0.5 * tp * tp * cx * cz;
        if ((i == 0 and j == 2) or (i == 2 and j == 0))
            v += 0.5 * tp * tp * sx * sz;
        return v;
    };
    {
        Grid<3> f(Nmesh);
        f.fill_real_grid(phi);
        std::vector<Grid<3>> hess;
        FML::HESSIAN::ComputeHessianWithFT(f, hess, 1.0, false);
        double md = 0.0;
        for (int i = 0, n = 0; i < 3; i++)
            for (int j = i; j < 3; j++, n++)
                md = std::max(md, maxdiff<3>(hess[n], [&](std::array<double, 3> & x) { return H(i, j, x); }));
        check(hess.size() == 6 and md < 1e-9, "3D Hessian = analytic (" + num(md) + ")");

        Grid<3> lap(Nmesh);
        lap.fill_real_grid([&](std::array<double, 3> & x) { return H(0, 0, x) + H(1, 1, x) + H(2, 2, x); });
        std::vector<Grid<3>> hp;
        FML::HESSIAN::ComputeHessianWithFT(lap, hp, 1.0, true);
        md = 0.0;
        for (int i = 0, n = 0; i < 3; i++)
            for (int j = i; j < 3; j++, n++)
                md = std::max(md, maxdiff<3>(hp[n], [&](std::array<double, 3> & x) { return H(i, j, x); }));
        check(md < 1e-9, "3D Hessian of the potential = analytic (" + num(md) + ")");

        // Eigensystem against GSL per cell (and M v = lambda v), with threads
        std::vector<Grid<3>> eval, evec, eval2, evec2;
        FML::HESSIAN::SymmetricTensorEigensystem(hess, eval, evec, true);
        FML::HESSIAN::SymmetricTensorEigensystem(hess, eval2, evec2, false);
        double mde = 0.0, mdv = 0.0;
        gsl_matrix * A = gsl_matrix_alloc(3, 3);
        gsl_vector * ev = gsl_vector_alloc(3);
        gsl_eigen_symm_workspace * w = gsl_eigen_symm_alloc(3);
        for (int islice = 0; islice < f.get_local_nx(); islice++)
            for (auto && ri : f.get_real_range(islice, islice + 1)) {
                auto x = f.get_real_position(f.get_coord_from_index(ri));
                double M[3][3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        gsl_matrix_set(A, i, j, M[i][j] = H(i, j, x));
                gsl_eigen_symm(A, ev, w);
                std::vector<double> e = {ev->data[0], ev->data[1], ev->data[2]};
                std::sort(e.begin(), e.end(), std::greater<double>());
                for (int k = 0; k < 3; k++) {
                    mde = std::max({mde,
                                    std::fabs(eval[k].get_real_from_index(ri) - e[k]),
                                    std::fabs(eval2[k].get_real_from_index(ri) - e[k])});
                    // Component i of eigenvector k is evec[i * 3 + k]
                    for (int i = 0; i < 3; i++) {
                        double s = 0.0;
                        for (int j = 0; j < 3; j++)
                            s += M[i][j] * evec[j * 3 + k].get_real_from_index(ri);
                        mdv = std::max(mdv, std::fabs(s - e[k] * evec[i * 3 + k].get_real_from_index(ri)));
                    }
                }
            }
        gsl_matrix_free(A);
        gsl_vector_free(ev);
        gsl_eigen_symm_free(w);
        FML::MaxOverTasks(&mde);
        FML::MaxOverTasks(&mdv);
        check(mde < 1e-8 and mdv < 1e-8,
              "eigenvalues (" + num(mde) + ") and eigenvectors evec[i * N + k] (" + num(mdv) + ")");
    }
    {
        // 2D: 3 components
        Grid<2> f(Nmesh);
        f.fill_real_grid([](std::array<double, 2> & x) { return std::cos(tp * x[0]) * std::cos(tp * 2 * x[1]); });
        std::vector<Grid<2>> hess;
        FML::HESSIAN::ComputeHessianWithFT(f, hess, 1.0, false);
        const double mxy = hess.size() == 3 ?
                               maxdiff<2>(hess[1],
                                          [](std::array<double, 2> & x) {
                                              return 2.0 * tp * tp * std::sin(tp * x[0]) * std::sin(tp * 2 * x[1]);
                                          }) :
                               1.0;
        const double myy = hess.size() == 3 ?
                               maxdiff<2>(hess[2],
                                          [](std::array<double, 2> & x) {
                                              return -4.0 * tp * tp * std::cos(tp * x[0]) * std::cos(tp * 2 * x[1]);
                                          }) :
                               1.0;
        check(hess.size() == 3 and mxy < 1e-9 and myy < 1e-9, "2D Hessian: 3 components = analytic");
    }
    {
        // A Nyquist mode in x: f_xz = 0 on the grid
        Grid<3> f(Nmesh);
        auto nyq = [&](std::array<double, 3> & x) {
            const int ix = int(std::round(x[0] * Nmesh));
            return (ix % 2 ? -1.0 : 1.0) * std::cos(tp * x[2]);
        };
        f.fill_real_grid(nyq);
        std::vector<Grid<3>> hess;
        FML::HESSIAN::ComputeHessianWithFT(f, hess, 1.0, false);
        const double kny = M_PI * Nmesh;
        const double mxz = maxdiff<3>(hess[2], [](std::array<double, 3> &) { return 0.0; });
        const double mxx = maxdiff<3>(hess[0], [&](std::array<double, 3> & x) { return -kny * kny * nyq(x); });
        check(mxz < 1e-9 and mxx < 1e-9 * kny * kny, "Nyquist mode: f_xz = 0 (" + num(mxz) + "), f_xx = -k_ny^2 f");
    }

    // compute_grid_PDF
    {
        Grid<3> f(Nmesh);
        f.fill_real_grid(phi);
        Grid<3> f2(Nmesh);
        f2.fill_real_grid([&](std::array<double, 3> & y) { return std::pow(phi(y), 3); });
        std::vector<double> x, pdf, xr, pdfref;
        FML::GRID::compute_grid_PDF(f, 10, x, pdf);
        FML::GRID::compute_grid_PDF(f2, 10, xr, pdfref);
        FML::GRID::compute_grid_PDF(f2, 10, x, pdf);
        double md = 0.0;
        for (size_t i = 0; i < pdf.size(); i++)
            md = std::max(md, std::fabs(pdf[i] - pdfref[i]));
        check(md == 0.0, "PDF with reused output vectors = PDF with new vectors");

        // Half the cells 0 and half 1: two bins with the same probability
        Grid<3> g(Nmesh);
        g.fill_real_grid([&](std::array<double, 3> & y) { return double(int(std::round(y[1] * Nmesh)) % 2); });
        FML::GRID::compute_grid_PDF(g, 2, x, pdf);
        check(std::fabs(pdf[0] - 1.0) < 1e-12 and std::fabs(pdf[1] - 1.0) < 1e-12,
              "PDF includes the largest value (" + num(pdf[0]) + ", " + num(pdf[1]) + ")");

        Grid<3> c(Nmesh);
        c.fill_real_grid(2.5);
        FML::GRID::compute_grid_PDF(c, 5, x, pdf);
        double integral = 0.0;
        bool finite = true;
        for (size_t i = 0; i < pdf.size(); i++) {
            integral += pdf[i] * (x[1] - x[0]);
            finite = finite and std::isfinite(pdf[i]) and std::isfinite(x[i]);
        }
        check(finite and std::fabs(integral - 1.0) < 1e-12, "PDF of a constant grid");
    }

    // RSD reconstruction (beta = 0.5): the removed shift matches the RSD shift put in (a bit less: the CIC assignment
    // and interpolation smooth the field) and does not grow with more iterations (it did before: 2-4x too large)
    {
        auto r3 = reconstruction_test(Nmesh, 3, 0.5, 2);
        auto r6 = reconstruction_test(Nmesh, 6, 0.5, 2);
        auto rx = reconstruction_test(Nmesh, 3, 0.5, 0);
        check(r3[2] >= 0.0 and rx[2] >= 0.0, "RSD reconstruction only moves the particles along the line of sight");
        check(r3[1] > 0.8 and r3[1] < 1.1 and r3[2] < 0.35 * r3[0],
              "RSD reconstruction, 3 iterations: removed / put in = " + num(r3[1]) + ", rms residual " + num(r3[2]) +
                  " of " + num(r3[0]) + " cells");
        check(r6[1] > 0.8 and r6[1] < 1.1 and r6[2] < 0.35 * r6[0],
              "RSD reconstruction, 6 iterations: removed / put in = " + num(r6[1]) + ", rms residual " + num(r6[2]));
        check(rx[1] > 0.8 and rx[1] < 1.1 and rx[2] < 0.35 * rx[0],
              "RSD reconstruction, line of sight along x (across the MPI slabs): " + num(rx[1]));
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All reconstruction/Hessian/smoothing tests passed\n" :
                                   std::to_string(fails) + " reconstruction/Hessian/smoothing test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
