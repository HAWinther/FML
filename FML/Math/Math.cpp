#include "Math.h"

namespace FML {
    namespace MATH {

        //=======================================================================
        // Algorithms
        //=======================================================================

        //=======================================================================
        //
        // Simple root finding method: Newtons method. Returns an optional: either it finds a root or it doesn't
        // The stopping criterion is |y(x)| < epsilon
        // Example use:
        // auto x = find_root(f, dydx, 0.0, 1e-8);
        // if(x.has_value()){
        //   std::cout << "The root is: "<< x.value << "\n";
        // } else {
        //   // No root found
        //   // Try again with different starting value
        //   // ...
        // }
        //
        //=======================================================================

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L))
        std::optional<double> find_root_newton(const std::function<double(double)> & function,
                                               const std::function<double(double)> & derivative,
                                               double x0,
                                               double epsilon) {
            const int niter_max = 100;
            double x = x0;
            int iter = 0;
            while (iter++ < niter_max) {
                double dydx = derivative(x);
                double y = function(x);
                if (std::abs(y) < epsilon)
                    return x;
                if (dydx == 0.0)
                    return {};
                x = x - y / dydx;
            }
            return {};
        }
#endif

        //=======================================================================
        // Find a zero of a given function on a given interval using bisection
        // If no root is found it will throw an error
        //=======================================================================
        double find_root_bisection(const std::function<double(double)> & function,
                                   std::pair<double, double> xrange,
                                   double epsilon) {
            double x_low = std::min(xrange.first, xrange.second);
            double x_high = std::max(xrange.first, xrange.second);
            double y_low = function(x_low);
            double y_high = function(x_high);
            const int sign = y_high > y_low ? 1 : -1;

            // Accuracy we want
            const double delta_x = (x_high - x_low) * epsilon;

            // Sanity check
            if (y_low * y_high > 0.0) {
                throw std::runtime_error("Error binary_search_for_value. f(xlow) and f(xhigh) "
                                         "have the same sign!");
            }

            // Do the binary search
            int count = 0;
            while (x_high - x_low > delta_x) {
                double x_mid = (x_high + x_low) / 2.0;
                double y_mid = function(x_mid) * sign;
                if (y_mid > 0.0) {
                    x_high = x_mid;
                } else {
                    x_low = x_mid;
                }
                count++;
            }

            return x_low;
        }

        //=======================================================================
        // Find a zero of y - y_value = 0 with the function given as a spline
        //=======================================================================
#ifdef USE_GSL
        double find_root_bisection(const Spline & y, double y_value, std::pair<double, double> xrange, double epsilon) {
            const int nsearch = 20;

            // The search range. If not provided use the full range from the spline
            if (xrange.first == 0.0 && xrange.second == 0.0) {
                xrange = y.get_xrange();
            }

            // Arrange search range so that x_low < x_high)
            double x_low = std::min(xrange.first, xrange.second);
            double x_high = std::max(xrange.first, xrange.second);

            // We measure function values relative to y_value
            double y_high = y(x_high) - y_value;
            double y_low = y(x_low) - y_value;

            // If the endpoints are both larger than zero do
            // a coarse grid search to see if we can find a good value
            // If not then throw an error
            if (y_low * y_high >= 0.0) {
                // Make a search grid
                auto x_array = linspace(x_low, x_high, nsearch);

                // Are the points positive or negative?
                int sign = y_low > 0.0 ? 1 : -1;

                for (int i = 0; i < nsearch; i++) {
                    double ycur = y(x_array[i]) - y_value;
                    if (sign * ycur < 0.0) {
                        y_low = ycur;
                        x_low = x_array[i];
                        break;
                    }
                    if (i == nsearch - 1) {
                        throw std::runtime_error("Error binary_search_for_value. Could not find a good interval to "
                                                 "start search");
                    }
                }
            }
            xrange.first = x_low;

            auto function = [&](double x) -> double { return (y(x) - y_value); };
            return find_root_bisection(function, xrange, epsilon);
        }
#endif

        //=======================================================================
        // Hyperspherical bessel function Phi_ell^nu(chi) for a flat and open Universe K <= 0.0
        // There are issues in a closed Universe as the square roots can be negative
        //=======================================================================

        DVector Hyperspherical_j_ell_array(int lmax, const double nu, const double chi, double K) {
            // In this method chi is sqrt|K|(eta0-eta) and K is 0,-1,+1
            const double chinu = chi * nu;
            const double chinu2 = chinu * chinu;

            // Geometry factors (cotc is equivalent of sinc, i.e. cot(x)*x)
            double cotcKchi = 1.0;
            double sincKchi = 1.0;
            if (K == 0.0) {
                cotcKchi = 1.0;
                sincKchi = 1.0;
            } else if (K < 0.0) {
                cotcKchi = chi > 1e-8 ? chi / tanh(chi) : 1.0;
                sincKchi = chi > 1e-8 ? sinh(chi) / chi : 1.0;
            } else {
                cotcKchi = chi > 1e-8 ? chi / tan(chi) : 1.0;
                sincKchi = chi > 1e-8 ? sin(chi) / chi : 1.0;
            }

            // Start the recursion at a large enough lmax such that j_ell/j_ell-1 ~ 0
            // and bring it down to lmax where we start to store the values
            int lstart = std::max(lmax, int(lmax < 10 ? 5 * chinu : (lmax < 100 ? 1.6 * chinu : 1.2 * chinu)));
            double h = 0.0;
            double sqrtnu2chi2old = sqrt(chinu2 - chi * chi * K * (lstart + 1) * (lstart + 1));
            for (int k = lstart; k >= lmax + 1; k--) {
                double sqrtnu2chi2 = sqrt(chinu2 - chi * chi * K * k * k);
                h = sqrtnu2chi2old / ((2 * k + 1) * cotcKchi - sqrtnu2chi2 * h);
                sqrtnu2chi2old = sqrtnu2chi2;
            }

            // Recursion relation for j_(n+1) / jn
            DVector res(lmax + 1, 0.0);
            for (int k = lmax; k >= 1; k--) {
                double sqrtnu2chi2 = sqrt(chinu2 - chi * chi * K * k * k);
                h = sqrtnu2chi2old / ((2 * k + 1) * cotcKchi - sqrtnu2chi2 * h);
                res[k] = h;
                sqrtnu2chi2old = sqrtnu2chi2;
            }

            // Transform ratios into j_ell
            res[0] = chinu == 0.0 ? (chi == 0.0 ? 1.0 : 1.0 / sincKchi) : sin(chinu) / (chinu * sincKchi);
            for (int ell = 1; ell <= lmax; ell++) {
                res[ell] *= res[ell - 1];
            }

            return res;
        }

        // For a given x compute j_ell(x) for all ell = 0, ..., lmax using a recursion function
        DVector j_ell_array(int lmax, const double x) {
            // Start the recursion at a large enough lmax such that j_ell/j_ell-1 ~ 0
            // and bring it down to lmax where we start to store the values
            int lstart = std::max(lmax, int(lmax < 10 ? 5 * x : (lmax < 100 ? 1.6 * x : 1.2 * x)));
            double h = 0.0;
            for (int k = lstart; k >= lmax + 1; k--) {
                h = x / (2 * k + 1 - x * h);
            }

            // Recursion relation for j_(n+1) / jn
            DVector res(lmax + 1, 0.0);
            for (int k = lmax; k >= 1; k--) {
                h = x / (2 * k + 1 - x * h);
                res[k] = h;
            }

            // Transform ratios into j_ell
            res[0] = x == 0.0 ? 1.0 : sin(x) / x;
            for (int ell = 1; ell <= lmax; ell++) {
                res[ell] *= res[ell - 1];
            }

            return res;
        }

#ifdef USE_GSL
        // GSL implementation
        double j_ell_gsl(const int ell, const double arg) {
            // For x=0 we have 0/0
            if (arg == 0.0)
                return ell == 0 ? 1.0 : 0.0;

            // The implementation fails for the largest arguments so simply put to zero
            // to avoid any issues with this
            const double arg_max_gsl = 9000.0;
            if (ell < 500 && arg > arg_max_gsl)
                return 0.0;

            return gsl_sf_bessel_jl(ell, arg);
        }
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L))
        // Standard library implementation. Only for C++17 onwards
        double j_ell_cxx(const int ell, const double arg) {
            // For x=0 we have 0/0
            if (arg == 0.0)
                return ell == 0 ? 1.0 : 0.0;

            // The implementation fails for the largest arguments so simply put to zero
            // to avoid any issues with this
            const double arg_max_libcxx = 14000.0;
            if (arg > arg_max_libcxx)
                return 0.0;
            return std::sph_bessel(ell, arg); // NOLINT
        }
#endif

        // Function to get the spherical Bessel function j_n(x)
        double j_ell(const int ell, const double arg) {
            // In this regime the function is ~1e-6 times the maximum value so put it to zero
            // to avoid issues with the library functions failing to compute it
            if (ell >= 10 && arg < (1.0 - 2.6 / sqrt(ell)) * ell)
                return 0.0;

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (__cplusplus >= 201703L))
            return j_ell_cxx(ell, arg);
#else
#ifdef USE_GSL
            return j_ell_gsl(ell, arg);
#endif
            throw std::runtime_error("No j_ell implementation availiable");
#endif
        }

        //=======================================================================
        // Useful methods
        //=======================================================================

        // Lineary spaced array ala Python Numpy linspace
        DVector linspace(double xmin, double xmax, int num) {
            DVector res(num);
            double delta_x = (xmax - xmin) / double(num - 1);
            for (int i = 0; i < num - 1; i++) {
                res[i] = xmin + delta_x * i;
            }
            // Just to make sure its exactly xmax at the endpoint
            res[num - 1] = xmax;
            return res;
        }

#ifdef USE_GSL
        // Airy function implemented as a singleton
        // Spline constructed first time we call the function
        class AiryFunction {
            static AiryFunction * instance;

            // Spline of the function around x = 0
            Spline AiryFunction_spline{"Uninitialized AiryFunction spline"};

            // Range for the spline
            // Outside of the range we use the asymptotic formula
            const double z_min = -50.0;
            const double z_max = 5.0;
            const int npts_forward = 100;
            const int npts_backward = 2000;

            AiryFunction() {
                // Create a spline of Ai(z) from z_min to z_max
                init();
            }

            void init() {

                // Integrate Ai(z) ODE from z = 0 -> z_max
                ODEFunction deriv_forward = [&](double x, const double * y, double * dydx) {
                    dydx[0] = y[1];
                    dydx[1] = x * y[0];
                    return GSL_SUCCESS;
                };

                // Integrate Ai(-z) ODE from z = 0 -> -z_min
                ODEFunction deriv_backward = [&](double x, const double * y, double * dydx) {
                    dydx[0] = y[1];
                    dydx[1] = -x * y[0];
                    return GSL_SUCCESS;
                };

                // Make points to store the solution at
                auto z_forward = linspace(0.0, z_max, npts_forward);
                auto z_backward = linspace(0.0, -z_min, npts_backward);

                // Solve forward equation
                double Yini_forward = 1.0 / pow(3., 2. / 3.) / tgamma(2. / 3.);
                double dYini_forward = -1.0 / pow(3., 1. / 3.) / tgamma(1. / 3.);
                DVector Y_ini_forward{Yini_forward, dYini_forward};
                ODESolver ode_forward(1e-3, 1e-8, 1e-8);
                ode_forward.solve(deriv_forward, z_forward, Y_ini_forward, gsl_odeiv2_step_rk2);
                auto Y_forward = ode_forward.get_data_by_component(0);

                // Solve backward equation
                double Yini_backward = 1.0 / pow(3., 2. / 3.) / tgamma(2. / 3.);
                double dYini_backward = 1.0 / pow(3., 1. / 3.) / tgamma(1. / 3.);
                DVector Y_ini_backward{Yini_backward, dYini_backward};
                ODESolver ode_backward(1e-3, 1e-8, 1e-8);
                ode_backward.solve(deriv_backward, z_backward, Y_ini_backward, gsl_odeiv2_step_rk2);
                auto Y_backward = ode_backward.get_data_by_component(0);

                // Merge the two regimes
                DVector z(npts_forward + npts_backward - 1);
                DVector Y(npts_forward + npts_backward - 1);
                size_t nsplit = npts_backward - 1;
                for (size_t i = 0; i < z.size(); i++) {
                    if (i < nsplit) {
                        z[i] = -z_backward[nsplit - i];
                        Y[i] = Y_backward[nsplit - i];
                    } else {
                        z[i] = z_forward[i - nsplit];
                        Y[i] = Y_forward[i - nsplit];
                    }
                }
                AiryFunction_spline.create(z, Y, "AiryFunction spline");
            }

          public:
            static AiryFunction * getInstance() {
                if (!instance)
                    instance = new AiryFunction;
                return instance;
            }

            double Ai(double z) {
                // Asymptotic values
                if (z < z_min)
                    return sin(2.0 / 3.0 * pow(-z, 1.5) + M_PI / 4.) / sqrt(M_PI * sqrt(-z));
                if (z > z_max)
                    return exp(-2.0 / 3.0 * pow(z, 1.5)) / sqrt(4.0 * M_PI * sqrt(z));
                // Return spline
                return AiryFunction_spline(z);
            }
        };

        AiryFunction * AiryFunction::instance = 0;

        // The Airy function Ai(z)
        // accelerator for the spline is generated). We do this below
        double Airy_Ai(double z) {
            static AiryFunction * a = a->getInstance();
            return a->Ai(z);
        }

        // If we call this in a multiprocessor setting we need to be sure that
        // the Airy spline must be made by the main thread
#ifdef AUTO_INIT_AIRY
        double init_airy_function_at_startup = Airy_Ai(0.0);
#endif
#endif

        //===========================================================================
        // How to compute j_ell(x) / j_(ell-1)(x) using the method below
        //
        //  int  ell = 10;
        //  double x = 10.0;
        //  std::function<double(int)> a = [=](int i){
        //    return -1.0;
        //  };
        //  std::function<double(int)> b = [=](int i){
        //    if(i == 0) return 0.0;
        //    return (2.0*(ell+i-1)+1)/x;
        //  };
        //  auto res = GeneralizedLentzMethod(a, b, 1e-7, 100);
        //  std::cout << -res.first << " " << j_ell(ell, x) / j_ell(ell-1, x) << "\n";
        //===========================================================================

        /// Evaluate continued fraction: (b0 + a1/(b1 + a2 /( ... ))). For example useful for computing ratios of
        /// spherical bessel functions.
        ///
        /// @param[in] a The function a[i]
        /// @param[in] b The function b[i]
        /// @param[in] epsilon Convergence criterion
        /// @param[in] maxsteps The maximum steps before we deem it not to converge
        ///
        /// \return The result and if it has converged or not
        ///
        std::pair<double, bool> GeneralizedLentzMethod(std::function<double(int)> & a,
                                                       std::function<double(int)> & b,
                                                       double epsilon,
                                                       int maxsteps) {

            const double tiny = 1e-30;

            double b0 = b(0);
            if (fabs(b0) < tiny)
                b0 = tiny;
            double f = b0;
            double C = b0;
            double D = 0.0;
            int j = 1;
            for (;;) {
                const double aa = a(j);
                const double bb = b(j);
                D = bb + aa * D;
                if (fabs(D) < tiny)
                    D = tiny;
                C = bb + aa / C;
                if (fabs(C) < tiny)
                    C = tiny;
                D = 1 / D;
                const double delta = C * D;
                f *= delta;
                j++;

                // Check for convergence
                if (fabs(delta - 1) < epsilon)
                    break;

                // Did not converge
                if (j > maxsteps) {
                    std::cout << "Did not converge\n";
                    return {f, false};
                }
            }
            return {f, true};
        }

        // C_n^(alpha)(x)
        double Gegenbauer(int n, double x, double alpha) {
            double C0 = 1.0;
            if (n == 0)
                return C0;
            double C1 = 2.0 * alpha * x;
            if (n == 1)
                return C1;
            double Cn = 0.0;
            for (int i = 2; i <= n; i++) {
                Cn = 1.0 / i * (2 * x * (i + alpha - 1) * C1 - (i + 2 * alpha - 2) * C0);
                C0 = C1;
                C1 = Cn;
            }
            return Cn;
        }

        // C_n^(alpha)(x) for n = 0,1,2,...,nmax
        DVector GegenbauerArray(int nmax, double x, double alpha) {
            if (nmax == 0)
                return {1.0};
            DVector Cn(nmax + 1, 1.0);
            Cn[1] = 2.0 * alpha * x;
            for (int i = 2; i <= nmax; i++) {
                Cn[i] = 1.0 / i * (2 * x * (i + alpha - 1) * Cn[i - 1] - (i + 2 * alpha - 2) * Cn[i - 2]);
            }
            return Cn;
        }

#ifdef USE_GSL
        double HyperSphericalBesselWKB(int ell, double nu, double chi, double K) {
            if (chi == 0.0)
                return ell == 0 ? 1.0 : 0.0;

            //================================================================
            // WKB approximation for the spherical bessel function
            // Depends on the Airy function
            // This is accurate enough for most purposes and as fast as possible
            // Here chi = sqrt(|OmegaK|) H0 (eta0 - eta(x))
            // nu = k / [sqrt(|OmegaK|) H0]
            // K is the sign of the curvature i.e. 0,-1,+1
            // For the flat case we can call this with (ell, 1.0, k*eta, 0.0)
            //================================================================

            // The geometry factor
            double sincKchi = 1.0;
            if (K == 0.0) {
                sincKchi = 1.0;
            } else if (K < 0.0) {
                sincKchi = chi > 1e-8 ? sinh(chi) / chi : 1.0;
            } else {
                sincKchi = chi > 1e-8 ? sin(chi) / chi : 1.0;
            }

            // For small curvatures then alpha -> infty
            const double sqrtellellp1 = sqrt(ell * (ell + 1));
            const double alpha = nu / sqrtellellp1, a2 = alpha * alpha;
            const double w = alpha * chi * sincKchi, w2 = w * w;
            const double sign = w < 1.0 ? 1.0 : -1.0;
            double S;
            const double sqrtw2m1 = sqrt(sign * (1 - w2));
            if (sign < 0.0) {
                if (K == 0.0) {
                    S = (atan(1.0 / sqrtw2m1) + sqrtw2m1 - M_PI / 2.);
                } else if (K < 0.0) {
                    const double sqrtw2pa2 = sqrt(w2 + a2);
                    S = (alpha * log((sqrtw2m1 + sqrtw2pa2) / sqrt(1 + a2)) + atan(sqrtw2pa2 / sqrtw2m1) / alpha -
                         M_PI / 2.);
                } else {
                    const double v = sqrt(a2 - w2) / sqrtw2m1 / alpha;
                    S = (atan(v) + alpha * atan(1.0 / (v * alpha)) - M_PI / 2.);
                }
            } else {
                if (K == 0.0) {
                    S = atanh(sqrtw2m1) - sqrtw2m1;
                } else if (K < 0.0) {
                    const double u = sqrtw2m1 * alpha / sqrt(a2 + w2);
                    S = (atanh(u) - alpha * atan(u / alpha));
                } else {
                    S = (atanh(sqrtw2m1 * alpha / sqrt(a2 - w2)) -
                         alpha * log((sqrt(a2 - w2) + sqrtw2m1) / sqrt(a2 - 1)));
                }
            }

            const double Zpow1over6 = pow(1.5 * fabs(S) * sqrtellellp1, 1. / 6.);
            double result = sqrt(M_PI / (w * sqrtw2m1)) * Zpow1over6 / sqrtellellp1;
            result *= Airy_Ai(sign * Zpow1over6 * Zpow1over6 * Zpow1over6 * Zpow1over6);
            return result;
        }
#endif

    } // namespace MATH
} // namespace FML
