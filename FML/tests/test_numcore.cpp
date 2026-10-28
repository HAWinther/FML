//==========================================================
// Tests for the numerical core: FML::INTERPOLATION::SPLINE (Spline), FML::SOLVERS::ODESOLVER (ODESolver)
// and FML::MATH (Math). References from scipy (spherical_jn, airy) and analytic solutions.
//   - find_root_bisection: reversed range for splines, roots exactly at an end-point, NaN end-points,
//     no endless loop when epsilon is below the floating point precision, unchanged results otherwise
//   - j_ell_array at zeros of j_0 (x = n pi), for ell close to lmax and for small x; j_ell for x > 14000
//   - linspace with 0 and 1 points, Airy_Ai outside the spline range, WKB at the turning point
//   - splines: values/derivatives, 2D splines with decreasing axes, copies keep the warning setting,
//     evaluation from nested OpenMP regions
//   - ODESolver: accuracy, exceptions from the ODE function, finite-difference jacobian
// Bad input that must give an error (throw without MPI, MPI_Abort with MPI) is tested with
// the extra modes (run alone, must fail with the given message):
//   ./test_numcore spline_npts      "[GSLSpline::create] spline"  "needs at least 3 points"
//   ./test_numcore spline2d_deriv   "[GSLSpline2D::eval_deriv] (2,1)"
//   ./test_numcore spline2d_monotone "[GSLSpline2D::create] y-array for spline"
//   ./test_numcore ode_nojacobian   "requires the jacobian"
// Run with: make run (in this folder) or make test (from the top folder)
//==========================================================
#include <FML/Global/Global.h>
#include <FML/Math/Math.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif

using DVector = std::vector<double>;
using Spline = FML::INTERPOLATION::SPLINE::Spline;
using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
using FML::SOLVERS::ODESOLVER::ODEFunction;
using FML::SOLVERS::ODESOLVER::ODEFunctionJacobian;
using FML::SOLVERS::ODESOLVER::ODESolver;

int fails = 0;
void check(bool ok, std::string what) {
    if (FML::ThisTask == 0)
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << what << "\n";
    if (not ok)
        fails++;
}
std::string num(double x) {
    std::ostringstream os;
    os.precision(10);
    os << x;
    return os.str();
}
bool close(double a, double b, double rtol) { return std::fabs(a - b) <= rtol * std::fabs(b); }

// Returns true if f throws std::exception whose message contains msg
template <class F>
bool throws_with(F && f, std::string msg) {
    try {
        f();
    } catch (std::exception & e) {
        return std::string(e.what()).find(msg) != std::string::npos;
    }
    return false;
}

Spline2D make_spline2d(bool decreasing_x, bool decreasing_y) {
    const int nx = 21, ny = 17;
    DVector x(nx), y(ny), z(nx * ny);
    for (int i = 0; i < nx; i++)
        x[i] = decreasing_x ? 2.0 - 0.1 * i : 0.1 * i;
    for (int j = 0; j < ny; j++)
        y[j] = decreasing_y ? 1.6 - 0.1 * j : 0.1 * j;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            z[i + nx * j] = std::sin(x[i]) * std::exp(0.5 * y[j]) + x[i] * x[i] * y[j];
    return Spline2D(x, y, z, "f(x,y)");
}

// Modes for input that must give an error (aborts with MPI so they are run as separate processes)
int bad_input(std::string mode) {
    if (mode == "spline_npts") {
        Spline s(DVector{0.0, 1.0}, DVector{0.0, 1.0}, "two points");
        std::cout << "No error: " << s(0.5) << "\n";
    } else if (mode == "spline2d_deriv") {
        auto s = make_spline2d(false, false);
        const double v = s.eval_deriv(0.5, 0.5, 2, 1);
        std::cout << "No error: " << v << "\n";
    } else if (mode == "spline2d_monotone") {
        Spline2D s(DVector{0.0, 1.0, 2.0, 3.0}, DVector{0.0, 1.0, 1.0, 3.0}, DVector(16, 1.0), "nonmonotone");
        std::cout << "No error: " << s(0.5, 0.5) << "\n";
    } else if (mode == "ode_nojacobian") {
        ODEFunction f = [](double, const double * y, double * dydx) {
            dydx[0] = -y[0];
            return GSL_SUCCESS;
        };
        DVector x{0.0, 1.0}, y{1.0};
        ODESolver ode;
        ode.solve(f, x, y, gsl_odeiv2_step_bsimp);
        std::cout << "No error: " << ode.get_final_data()[0] << "\n";
    } else {
        std::cout << "Unknown mode " << mode << "\n";
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 1)
        return bad_input(argv[1]);

    //==========================================================
    // Root finding
    //==========================================================
    {
        DVector x = FML::MATH::linspace(0.0, 10.0, 101), y(101);
        for (int i = 0; i < 101; i++)
            y[i] = x[i] * x[i];
        Spline s(x, y, "x^2");
        const double r1 = FML::MATH::find_root_bisection(s, 4.0, {0.0, 5.0});
        double r2 = -1.0;
        try {
            r2 = FML::MATH::find_root_bisection(s, 4.0, {5.0, 0.0});
        } catch (std::exception & e) {
        }
        check(std::fabs(r1 - 2.0) < 1e-6 and r1 == r2, "spline root with reversed range: " + num(r2));
        double r3 = -1.0, r4 = -1.0;
        try {
            r3 = FML::MATH::find_root_bisection(s, 100.0);
            r4 = FML::MATH::find_root_bisection(s, 0.0);
        } catch (std::exception & e) {
        }
        check(r3 == 10.0 and r4 == 0.0, "spline roots exactly at the end-points: " + num(r3) + " " + num(r4));

        // Unchanged result for a normal case (lower end of the final interval)
        auto f = [](double x) { return x - 1.0; };
        const double r5 = FML::MATH::find_root_bisection(f, {0.0, 3.0});
        check(r5 < 1.0 and 1.0 - r5 < 3e-7 and r5 == FML::MATH::find_root_bisection(f, {3.0, 0.0}),
              "bisection x-1 on (0,3): " + num(r5));
        // f = 0 belongs to the side of xlow (SphericalCollapse relies on this): converges to the end of the plateau
        const double r8 = FML::MATH::find_root_bisection([](double x) { return x < 0.4 ? 0.0 : 1.0; }, {0.0, 1.0});
        check(std::fabs(r8 - 0.4) < 1e-6, "bisection with f = 0 on part of the range: " + num(r8));

        // NaN at an end-point must be an error, not a wrong root
        auto fnan = [](double x) { return std::sqrt(x - 1.0) - 0.5; };
        check(throws_with([&]() { FML::MATH::find_root_bisection(fnan, {0.0, 3.0}); }, "NaN"),
              "bisection with f(xlow) = NaN throws");
        check(throws_with([&]() { FML::MATH::find_root_bisection([](double x) { return x * x + 1.0; }, {0.0, 3.0}); },
                          "same sign"),
              "bisection without a root throws");
    }

    //==========================================================
    // Spherical bessel functions (references from scipy.special.spherical_jn)
    //==========================================================
    {
        auto a = FML::MATH::j_ell_array(10, 2.0 * M_PI);
        check(close(a[4], 1.859324505594785e-01, 1e-10), "j_ell_array j_4(2 pi) (zero of j_0): " + num(a[4]));
        auto b = FML::MATH::j_ell_array(50, 10.0 * M_PI);
        check(close(b[28], 4.786217200196330e-02, 1e-10), "j_ell_array j_28(10 pi) (zero of j_0): " + num(b[28]));
        auto c = FML::MATH::j_ell_array(10, 6.862);
        check(close(c[10], 5.772085051241143e-03, 1e-9), "j_ell_array j_10(6.862) with lmax = 10: " + num(c[10]));
        auto d = FML::MATH::j_ell_array(100, 84.012);
        check(close(d[100], 7.933616953491621e-06, 1e-8), "j_ell_array j_100(84.012) with lmax = 100: " + num(d[100]));
        auto e = FML::MATH::j_ell_array(5, 1.187);
        check(close(e[5], 2.146883323046425e-04, 1e-10), "j_ell_array j_5(1.187) with lmax = 5: " + num(e[5]));
        const double j10 = FML::MATH::j_ell(10, 20000.0);
        const double j2 = FML::MATH::j_ell(2, 14001.0);
        check(close(j10, -2.921094489548591e-05, 1e-8) and close(j2, -6.293323508535110e-05, 1e-8),
              "j_ell for x > 14000: " + num(j10) + " " + num(j2));
    }

    //==========================================================
    // linspace, Airy, WKB
    //==========================================================
    {
        auto l1 = FML::MATH::linspace(1.0, 2.0, 1);
        auto l0 = FML::MATH::linspace(1.0, 2.0, 0);
        auto l3 = FML::MATH::linspace(1.0, 2.0, 3);
        check(l1.size() == 1 and l1[0] == 1.0 and l0.empty() and l3[0] == 1.0 and l3[1] == 1.5 and l3[2] == 2.0,
              "linspace with 1, 0 and 3 points (as numpy)");

        const double a1 = FML::MATH::Airy_Ai(5.0001);
        const double a2 = FML::MATH::Airy_Ai(-50.0001);
        const double a3 = FML::MATH::Airy_Ai(7.0);
        check(close(a1, 1.083196894554103e-04, 1e-8) and close(a2, -1.619782821175962e-01, 1e-8) and
                  close(a3, 7.492128863997157e-07, 1e-8),
              "Airy_Ai outside the spline range: " + num(a1) + " " + num(a2) + " " + num(a3));
        check(close(FML::MATH::Airy_Ai(0.0), 1.0 / std::pow(3., 2. / 3.) / std::tgamma(2. / 3.), 1e-6), "Airy_Ai(0)");

        const double w = FML::MATH::HyperSphericalBesselWKB(10, 1.0, std::sqrt(110.0), 0.0);
        check(std::isfinite(w) and close(w, 7.862800300679687e-02, 1e-2), "WKB j_10 at the turning point: " + num(w));

        // ell = 0 is exact: sin(nu chi) / (nu sin_K(chi)) (the WKB formula divides by sqrt(ell(ell+1)))
        const double w0 = FML::MATH::HyperSphericalBesselWKB(0, 1.0, 5.0, 0.0);
        const double w0open = FML::MATH::HyperSphericalBesselWKB(0, 3.0, 0.7, -1.0);
        check(close(w0, std::sin(5.0) / 5.0, 1e-14) and close(w0open, std::sin(2.1) / (3.0 * std::sinh(0.7)), 1e-14),
              "WKB ell = 0 (flat and open): " + num(w0) + " " + num(w0open));
    }

    //==========================================================
    // Splines
    //==========================================================
    {
        const int n = 201;
        DVector x(n), y(n);
        for (int i = 0; i < n; i++) {
            x[i] = 0.05 * i;
            y[i] = std::sin(x[i]);
        }
        Spline s(x, y, "sin");
        check(std::fabs(s(3.3) - std::sin(3.3)) < 1e-6 and std::fabs(s.deriv_x(3.3) - std::cos(3.3)) < 1e-5 and
                  std::fabs(s.deriv_xx(3.3) + std::sin(3.3)) < 1e-3,
              "1D spline value and derivatives");
        check(s(12.0) == s(10.0) and s.deriv_x(12.0) == s.deriv_x(10.0), "1D spline out of range: clamped x");

        // A copy keeps the out-of-bounds warning setting
        s.set_out_of_bounds_warning(true);
        Spline s2(s), s3;
        s3 = s;
        std::stringstream buffer;
        auto * old = std::cout.rdbuf(buffer.rdbuf());
        s2(20.0);
        s3(20.0);
        std::cout.rdbuf(old);
        check(buffer.str().find("out of bounds") != std::string::npos and
                  buffer.str().find("out of bounds") != buffer.str().rfind("out of bounds"),
              "copies keep the out-of-bounds warning setting");

        // Integral (and a > b), moving a spline, and the uniform-grid spline
        check(std::fabs(s.integral(0.0, 3.0) - (1.0 - std::cos(3.0))) < 1e-6 and
                  std::fabs(s.integral(3.0, 0.0) + s.integral(0.0, 3.0)) < 1e-15,
              "1D spline integral");
        Spline sm(std::move(s2));
        Spline sm2;
        sm2 = std::move(sm);
        check(sm2 and !sm and !s2 and std::fabs(sm2(3.3) - s(3.3)) == 0.0, "moved splines");
        FML::INTERPOLATION::SPLINE::UniformSpline u(0.0, 0.05, y);
        double uerr = 0.0;
        for (double xx = 0.0; xx <= 10.0; xx += 0.0123)
            uerr = std::max(uerr, std::fabs(u(xx) - std::sin(xx)));
        check(uerr < 1e-6 and u(-1.0) == y.front() and u(11.0) == y.back(),
              "uniform-grid spline: max error " + num(uerr));

        // 2D with decreasing axes gives the same as increasing axes
        auto f = make_spline2d(false, false);
        auto g = make_spline2d(true, false);
        auto h = make_spline2d(true, true);
        double maxdiff = 0.0;
        for (double xx : {0.13, 0.77, 1.5})
            for (double yy : {0.21, 0.9, 1.43}) {
                maxdiff = std::max(maxdiff, std::fabs(f(xx, yy) - g(xx, yy)));
                maxdiff = std::max(maxdiff, std::fabs(f.deriv_xy(xx, yy) - h.deriv_xy(xx, yy)));
            }
        const double exact = std::sin(0.77) * std::exp(0.45) + 0.77 * 0.77 * 0.9;
        check(maxdiff < 1e-12 and std::fabs(f(0.77, 0.9) - exact) < 1e-5,
              "2D spline with decreasing axes: max difference " + num(maxdiff));

#ifndef USE_MPI
        // Without MPI errors are exceptions
        check(throws_with([]() { Spline t(DVector{0.0}, DVector{0.0}); }, "needs at least"),
              "1D spline with 1 point throws");
        check(throws_with([&]() { f.eval_deriv(0.5, 0.5, 2, 1); }, "eval_deriv"), "2D eval_deriv(2,1) throws");
        check(throws_with([&]() { f.eval_deriv(0.5, 0.5, 4, -1); }, "eval_deriv"), "2D eval_deriv(4,-1) throws");
        check(throws_with([]() { Spline2D t(DVector{0, 1}, DVector{0, 1}, DVector(4, 0.0)); }, "needs at least"),
              "2D bicubic spline with 2x2 points throws");
#endif

#ifdef USE_OMP
        // Evaluation from a nested (inactive) parallel region inside a parallel region must give the same results
        const int npts = 200000;
        DVector q(npts), ref(npts), res(npts);
        for (int i = 0; i < npts; i++) {
            q[i] = std::fmod(i * 0.6180339887, 1.0) * 10.0;
            ref[i] = s(q[i]);
        }
        const int nthreads = omp_get_max_threads();
#pragma omp parallel
        {
            const int id = omp_get_thread_num();
            const int nt = omp_get_num_threads();
            const int i0 = id * (npts / nt), i1 = id == nt - 1 ? npts : i0 + npts / nt;
#pragma omp parallel for
            for (int i = i0; i < i1; i++)
                res[i] = s(q[i]);
        }
        check(res == ref, "spline evaluation in nested OpenMP regions (" + std::to_string(nthreads) + " threads)");
#endif
    }

    //==========================================================
    // ODESolver
    //==========================================================
    {
        ODEFunction f = [](double, const double * y, double * dydx) {
            dydx[0] = y[1];
            dydx[1] = -y[0];
            return GSL_SUCCESS;
        };
        DVector x{0.0, 0.3, 1.7, 10.0}, yini{0.0, 1.0};
        ODESolver ode;
        ode.solve(f, x, yini);
        auto y = ode.get_data();
        auto dy = ode.get_derivative_data();
        double maxerr = 0.0;
        for (size_t i = 0; i < x.size(); i++)
            maxerr = std::max({maxerr, std::fabs(y[i][0] - std::sin(x[i])), std::fabs(dy[i][0] - std::cos(x[i]))});
        check(maxerr < 1e-6 and ode.get_data_transpose()[0][2] == y[2][0],
              "ODESolver y'' = -y: max error " + num(maxerr));

        // An implicit stepper with the jacobian
        ODEFunction g = [](double t, const double * yy, double * dydx) {
            dydx[0] = -1000.0 * (yy[0] - std::cos(t));
            return GSL_SUCCESS;
        };
        ODEFunctionJacobian J = [](double t, const double *, double * dfdy, double * dfdt) {
            dfdy[0] = -1000.0;
            dfdt[0] = -1000.0 * std::sin(t);
            return GSL_SUCCESS;
        };
        DVector x2{0.0, 1.0}, y2{0.0};
        ODESolver ode2;
        ode2.solve(g, x2, y2, gsl_odeiv2_step_bsimp, J);
        const double a = 1000.0, A = a * a / (a * a + 1), B = a / (a * a + 1);
        const double exact = A * std::cos(1.0) + B * std::sin(1.0) - A * std::exp(-a);
        check(std::fabs(ode2.get_final_data()[0] - exact) < 1e-6, "ODESolver stiff problem with bsimp + jacobian");
#ifndef USE_MPI
        check(throws_with(
                  [&]() {
                      ODESolver o;
                      o.solve(g, x2, y2, gsl_odeiv2_step_msbdf);
                  },
                  "requires the jacobian"),
              "ODESolver implicit stepper without jacobian throws");
#endif

        // An implicit stepper with a finite-difference jacobian (no jacobian given)
        ODESolver ode3;
        ode3.set_finite_difference_jacobian(true);
        double fderr = 0.0;
        for (auto st : {gsl_odeiv2_step_msbdf, gsl_odeiv2_step_bsimp, gsl_odeiv2_step_rk4imp}) {
            ode3.solve(g, x2, y2, st);
            fderr = std::max(fderr, std::fabs(ode3.get_final_data()[0] - exact));
        }
        check(fderr < 1e-6, "ODESolver implicit steppers with a finite-difference jacobian: error " + num(fderr));
        // ... also integrating backwards in time (exp(x) from x = 0 to -5): the same as with the exact jacobian
        ODEFunction grow = [](double, const double * yy, double * dydx) {
            dydx[0] = yy[0];
            return GSL_SUCCESS;
        };
        ODEFunctionJacobian Jg = [](double, const double *, double * dfdy, double * dfdt) {
            dfdy[0] = 1.0;
            dfdt[0] = 0.0;
            return GSL_SUCCESS;
        };
        ODESolver ode4(1e-3, 1e-10, 1e-10), ode5(1e-3, 1e-10, 1e-10);
        ode4.set_finite_difference_jacobian(true);
        ode4.solve(grow, DVector{0.0, -2.5, -5.0}, DVector{1.0}, gsl_odeiv2_step_msbdf);
        ode5.solve(grow, DVector{0.0, -2.5, -5.0}, DVector{1.0}, gsl_odeiv2_step_msbdf, Jg);
        const double berr = std::fabs(ode4.get_final_data()[0] / std::exp(-5.0) - 1.0);
        check(berr < 1e-6 and std::fabs(ode4.get_final_data()[0] / ode5.get_final_data()[0] - 1.0) < 1e-9,
              "ODESolver backwards with a finite-difference jacobian: error " + num(berr));

        // An exception from the ODE function is passed on and the solver can be used again
        ODEFunction bad = [](double t, const double *, double * dydx) {
            if (t > 0.5)
                throw std::runtime_error("user error");
            dydx[0] = 1.0;
            return GSL_SUCCESS;
        };
        check(throws_with([&]() { ode2.solve(bad, x2, y2); }, "user error"), "exception from the ODE function");
        ode2.solve(g, x2, y2, gsl_odeiv2_step_bsimp, J);
        check(std::fabs(ode2.get_final_data()[0] - exact) < 1e-6, "ODESolver reused after an exception");
    }

    // Root finding with epsilon below the floating point precision must terminate (used to loop forever,
    // so this is the last test)
    {
        const double r6 = FML::MATH::find_root_bisection([](double x) { return x - 1.5; }, {1.0, 2.0}, 1e-17);
        const double r7 =
            FML::MATH::find_root_bisection([](double x) { return x - (1e10 + 0.3); }, {1e10, 1e10 + 1.0}, 1e-7);
        check(std::fabs(r6 - 1.5) < 1e-15 and std::fabs(r7 - (1e10 + 0.3)) < 1e-5,
              "bisection terminates when epsilon is below the floating point precision");
    }

    if (FML::ThisTask == 0)
        std::cout << (fails == 0 ? "All numcore tests passed\n" : std::to_string(fails) + " numcore test(s) FAILED\n");
    return fails == 0 ? 0 : 1;
}
