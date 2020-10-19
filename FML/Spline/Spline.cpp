#include "Spline.h"

namespace FML {
    namespace INTERPOLATION {
        namespace SPLINE {

#ifdef USE_OMP
            const int nmax_threads = omp_get_max_threads();
#endif

            // How to handle an error
            void GSLSpline::throw_error(std::string errormessage) const {
#ifdef USE_MPI
                std::cout << errormessage << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
                abort();
#else
                throw std::runtime_error(errormessage);
#endif
            }

            // How to handle an error
            void GSLSpline2D::throw_error(std::string errormessage) const {
#ifdef USE_MPI
                std::cout << errormessage << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
                abort();
#else
                throw std::runtime_error(errormessage);
#endif
            }

            //====================================================
            // Constructors
            //====================================================

            GSLSpline::GSLSpline(std::string name) : GSLSpline() { this->name = name; }

            GSLSpline::GSLSpline(double * x,
                                 double * y,
                                 int nx,
                                 std::string splinename,
                                 const gsl_interp_type * interpoltype) {
                create(x, y, nx, splinename, interpoltype);
            }

            GSLSpline::GSLSpline(const DVector & x,
                                 const DVector & y,
                                 std::string splinename,
                                 const gsl_interp_type * interpoltype) {
                create(x, y, splinename, interpoltype);
            }

            //====================================================
            // Assignment constructor
            //====================================================
            GSLSpline & GSLSpline::operator=(const GSLSpline & rhs) {
                if (this != &rhs) {
                    // We just create the spline from scratch instead of copying all the data
                    if (rhs.spline != nullptr) {
                        create(rhs.spline->x, rhs.spline->y, rhs.size_x, rhs.name, rhs.interpoltype_used);
                    } else {
                        free();
                    }
                }
                return *this;
            }

            //====================================================
            // Copy constructor
            //====================================================
            GSLSpline::GSLSpline(const GSLSpline & rhs) {
                // We just create the spline from scratch instead of copying all the data
                if (rhs.spline != nullptr) {
                    create(rhs.spline->x, rhs.spline->y, rhs.size_x, rhs.name, rhs.interpoltype_used);
                } else {
                    free();
                }
            }

            GSLSpline::operator bool() const { return (spline != nullptr); }
            
            //====================================================
            // Create a GSL spline
            //====================================================
            void GSLSpline::create(const double * x,
                                   const double * y,
                                   int nx,
                                   std::string splinename,
                                   const gsl_interp_type * interpoltype) {
                // Clean up if we already have a spline allocated
                if (spline != nullptr) {
                    free();
                }

                // Check if array is increasing
                int sign_one = x[1] > x[0] ? 1.0 : -1.0;
                for (int i = 1; i < nx; i++) {
                    if ((x[i] - x[i - 1]) * sign_one <= 0) {
                        std::string errormessage =
                            "[GSLSpline::create] x-array for spline " + name + " is not monotone\n";
                        throw_error(errormessage);
                    }
                }

                // If array is reversed then reverse it since GSL
                // requires a monotonely increasing array
                DVector xx, yy;
                if (sign_one < 0) {
                    xx = DVector(nx);
                    yy = DVector(nx);
                    for (int i = 0; i < nx; i++) {
                        xx[i] = x[nx - 1 - i];
                        yy[i] = y[nx - 1 - i];
                    }
                    x = xx.data();
                    y = yy.data();
                }

                // Set class variables
                xmin = x[0];
                xmax = x[nx - 1];
                dx_min = (x[1] - x[0]) / 2.0;
                dx_max = (x[nx - 1] - x[nx - 2]) / 2.0;
                size_x = nx;
                name = splinename;
                interpoltype_used = interpoltype;

                // Make the spline
                spline = gsl_spline_alloc(interpoltype, nx);
                gsl_spline_init(spline, x, y, nx);

                // Make accelerators (one per thread if OpenMP)
                // If nthreads = 1 we are likely trying to create a spline 
                // inside a OMP region. In that case allocate as many as we have
                // threads availiable on the system
#ifdef USE_OMP
                int nthreads = 1;
#pragma omp parallel
                {
                    int id = omp_get_thread_num();
                    if (id == 0)
                        nthreads = omp_get_num_threads();
                }
                if(nthreads == 1)
                  nthreads = nmax_threads;

                // If we make a spline inside a nested openmp then
                // the stuff above does not work
                // nthreads = std::max(16, nthreads);
                xaccs = std::vector<gsl_interp_accel *>(nthreads);
                for (auto & xa : xaccs) {
                    xa = gsl_interp_accel_alloc();
                }
#else
                xacc = gsl_interp_accel_alloc();
#endif
            }

            void GSLSpline::create(const DVector & x,
                                   const DVector & y,
                                   std::string splinename,
                                   const gsl_interp_type * interpoltype) {
                if (x.size() != y.size()) {
                    std::string errormessage =
                        "[GSLSpline::create] x and y array must have the same number of elements for spline " + name +
                        "\n";
                    throw_error(errormessage);
                }
                create(x.data(), y.data(), int(x.size()), splinename, interpoltype);
            }

            //====================================================
            // Evaluate the function
            // NB: using closest points if out-of-bounds!
            //====================================================

            double GSLSpline::operator()(double x) const { return eval(x); }

            double GSLSpline::eval(double x) const {
                if (spline == nullptr) {
                    std::string errormessage = "[GSLSpline::eval] Spline " + name + " has not been created!\n";
                    throw_error(errormessage);
                }

                // If out of bounds show a warning and set x to boundary value
                out_of_bounds_check(x);
                x = std::max(x, xmin);
                x = std::min(x, xmax);

                // Return f, f' or f'' depending on value of deriv
#ifdef USE_OMP
                gsl_interp_accel * xacc_thread = xaccs[omp_get_thread_num()];
#else
                gsl_interp_accel * xacc_thread = xacc;
#endif
                return gsl_spline_eval(spline, x, xacc_thread);
            }

            double GSLSpline::eval_deriv(double x, int deriv) const {
                if (spline == nullptr) {
                    std::string errormessage = "[GSLSpline::eval_deriv] Spline " + name + " has not been created!\n";
                    throw_error(errormessage);
                }
                if (deriv < 0 || deriv > 2) {
                    std::string errormessage = "[GSLSpline::eval_deriv] Got deriv = " + std::to_string(deriv) +
                                               " for spline " + name + " Expected 0 = f, 1 = f' or 2 = f''\n";
                    throw_error(errormessage);
                }

                // If out of bounds show a warning and set x to boundary value
                out_of_bounds_check(x);
                x = std::max(x, xmin);
                x = std::min(x, xmax);

                // Return f, f' or f'' depending on value of deriv
#ifdef USE_OMP
                gsl_interp_accel * xacc_thread = xaccs[omp_get_thread_num()];
#else
                gsl_interp_accel * xacc_thread = xacc;
#endif

                double dydx = 0.0;
                if (deriv == 0) {
                    dydx = gsl_spline_eval(spline, x, xacc_thread);
                } else if (deriv == 1) {
                    dydx = gsl_spline_eval_deriv(spline, x, xacc_thread);
                } else if (deriv == 2) {
                    dydx = gsl_spline_eval_deriv2(spline, x, xacc_thread);
                }
                return dydx;
            }

            //====================================================
            // Free up memory
            //====================================================
            void GSLSpline::free() {
                if (spline != nullptr) {
                    // Reset class variables
                    xmin = xmax = 0.0;
                    size_x = 0;
                    dx_min = dx_max = 0.0;

                    // Free the spline
                    gsl_spline_free(spline);
                    spline = nullptr;

                    // Free accelerators
#ifdef USE_OMP
                    for (auto & xa : xaccs) {
                        gsl_interp_accel_free(xa);
                    }
                    std::vector<gsl_interp_accel *>().swap(xaccs);
#else
                    gsl_interp_accel_free(xacc);
                    xacc = nullptr;
#endif
                }
            }

            //====================================================
            // Rest of the class methods
            //====================================================
            void GSLSpline::out_of_bounds_check(double x) const {
                if (out_of_bounds_warning) {
                    if (x < xmin - dx_min || x > xmax + dx_max) {
                        std::cout << "Warning GSLSpline[" << name << "] ";
                        std::cout << "x = " << x << " is out of bounds (" << xmin << "," << xmax << ")\n";
                    }
                }
            }
            double GSLSpline::deriv_x(double x) const { return eval_deriv(x, 1); }
            double GSLSpline::deriv_xx(double x) const { return eval_deriv(x, 2); }
            std::pair<double, double> GSLSpline::get_xrange() const { return {xmin, xmax}; }
            std::string GSLSpline::get_name() const { return name; }
            void GSLSpline::set_out_of_bounds_warning(bool v) { out_of_bounds_warning = v; }

            //====================================================
            // Destructor
            //====================================================
            GSLSpline::~GSLSpline() { free(); }

            //====================================================
            // Constructors
            //====================================================

            GSLSpline2D::GSLSpline2D(std::string name) : GSLSpline2D() { this->name = name; }

            GSLSpline2D::GSLSpline2D(const double * x,
                                     const double * y,
                                     const double * z,
                                     int nx,
                                     int ny,
                                     std::string splinename,
                                     const gsl_interp2d_type * interpoltype) {
                create(x, y, z, nx, ny, splinename, interpoltype);
            }

            GSLSpline2D::GSLSpline2D(const DVector & x,
                                     const DVector & y,
                                     const DVector & z,
                                     std::string splinename,
                                     const gsl_interp2d_type * interpoltype) {
                create(x, y, z, splinename, interpoltype);
            }

            GSLSpline2D::GSLSpline2D(const DVector & x,
                                     const DVector & y,
                                     const DVector2D & z,
                                     std::string splinename,
                                     const gsl_interp2d_type * interpoltype) {
                create(x, y, z, splinename, interpoltype);
            }

            //====================================================
            // Assignment constructor
            //====================================================
            GSLSpline2D & GSLSpline2D::operator=(const GSLSpline2D & rhs) {
                if (this != &rhs) {
                    // We just create the spline from scratch instead of copying all the data
                    if (rhs.spline != nullptr) {
                        create(rhs.spline->xarr,
                               rhs.spline->yarr,
                               rhs.spline->zarr,
                               rhs.size_x,
                               rhs.size_y,
                               rhs.name,
                               rhs.interpoltype_used);
                    } else {
                        free();
                    }
                }
                return *this;
            }

            //====================================================
            // Copy constructor
            //====================================================
            GSLSpline2D::GSLSpline2D(const GSLSpline2D & rhs) {
                // We just create the spline from scratch instead of copying all the data
                if (rhs.spline != nullptr) {
                    create(rhs.spline->xarr,
                           rhs.spline->yarr,
                           rhs.spline->zarr,
                           rhs.size_x,
                           rhs.size_y,
                           rhs.name,
                           rhs.interpoltype_used);
                } else {
                    free();
                }
            }

            // For more easy evaluation
            double GSLSpline2D::operator()(double x, double y) const { return eval(x, y); }

            GSLSpline2D::operator bool() const { return (spline != nullptr); }

            //====================================================
            // Create a GSL 2D spline
            //====================================================
            void GSLSpline2D::create(const double * x,
                                     const double * y,
                                     const double * z,
                                     int nx,
                                     int ny,
                                     std::string splinename,
                                     const gsl_interp2d_type * interpoltype) {
                // Clean up if we already have a spline
                if (spline != nullptr) {
                    free();
                }

                // Set class variables
                xmin = x[0];
                xmax = x[nx - 1];
                ymin = y[0];
                ymax = y[ny - 1];
                dx_min = (x[1] - x[0]) / 2.0;
                dx_max = (x[nx - 1] - x[nx - 2]) / 2.0;
                dy_min = (y[1] - y[0]) / 2.0;
                dy_max = (y[ny - 1] - y[ny - 2]) / 2.0;
                size_x = nx;
                size_y = ny;
                name = splinename;
                interpoltype_used = interpoltype;

                // Create spline
                spline = gsl_spline2d_alloc(interpoltype, nx, ny);
                gsl_spline2d_init(spline, x, y, z, nx, ny);

                // Make accelerators (one per thread if OpenMP)
                // If nthreads = 1 we are likely trying to create a spline 
                // inside a OMP region. In that case allocate as many as we have
                // threads availiable on the system
#ifdef USE_OMP
                int nthreads = 1;
#pragma omp parallel
                {
                    int id = omp_get_thread_num();
                    if (id == 0)
                        nthreads = omp_get_num_threads();
                }
                if(nthreads == 1)
                  nthreads = nmax_threads;

                xaccs = std::vector<gsl_interp_accel *>(nthreads);
                yaccs = std::vector<gsl_interp_accel *>(nthreads);
                for (auto & xa : xaccs) {
                    xa = gsl_interp_accel_alloc();
                }
                for (auto & ya : yaccs) {
                    ya = gsl_interp_accel_alloc();
                }
#else
                xacc = gsl_interp_accel_alloc();
                yacc = gsl_interp_accel_alloc();
#endif
            }

            void GSLSpline2D::create(const DVector & x,
                                     const DVector & y,
                                     const DVector & z,
                                     std::string splinename,
                                     const gsl_interp2d_type * interpoltype) {
                if (x.size() * y.size() != z.size()) {
                    std::string errormessage = "[GSLSpline2D::create] z array for spline " + name +
                                               " have wrong number of elements nx*ny != nz\n";
                    throw_error(errormessage);
                }
                create(x.data(), y.data(), z.data(), int(x.size()), int(y.size()), splinename, interpoltype);
            }

            void GSLSpline2D::create(const DVector & x,
                                     const DVector & y,
                                     const DVector2D & z,
                                     std::string splinename,
                                     const gsl_interp2d_type * interpoltype) {
                int nz_x = int(z.size());
                int nz_y = nz_x == 0 ? 0 : int(z[0].size());
                int nxy = int(x.size() * y.size());
                if (nz_x * nz_y != nxy) {
                    std::string errormessage =
                        "[GSLSpline2D::create] z array for spline " + name + " have wrong dimensions\n";
                    throw_error(errormessage);
                }
                for (int i = 0; i < nz_x; i++) {
                    int n = int(z[i].size());
                    if (n != nz_y) {
                        std::string errormessage =
                            "[GSLSpline2D::create] z array for spline " + name + " have wrong dimensions\n";
                        throw_error(errormessage);
                    }
                }
                DVector f(nz_x * nz_y);
                for (int iy = 0; iy < nz_y; iy++) {
                    for (int ix = 0; ix < nz_x; ix++) {
                        f[ix + nz_x * iy] = z[ix][iy];
                    }
                }
                create(x.data(), y.data(), f.data(), int(x.size()), int(y.size()), splinename, interpoltype);
            }

            //====================================================
            // Lookup a value from a GSL 2D spline
            // Use closest points for out-of-bounds
            //====================================================
            double GSLSpline2D::eval(double x, double y) const {
                if (spline == nullptr) {
                    std::string errormessage = "[GSLSpline2D::eval] Spline " + name + " has not been created!\n";
                    throw_error(errormessage);
                }
                // If out of bounds show a warning and set x,y to boundary value
                out_of_bounds_check(x, y);
                x = std::max(xmin, x);
                x = std::min(xmax, x);
                y = std::max(ymin, y);
                y = std::min(ymax, y);

#ifdef USE_OMP
                gsl_interp_accel * xacc_thread = xaccs[omp_get_thread_num()];
                gsl_interp_accel * yacc_thread = yaccs[omp_get_thread_num()];
#else
                gsl_interp_accel * xacc_thread = xacc;
                gsl_interp_accel * yacc_thread = yacc;
#endif
                return gsl_spline2d_eval(spline, x, y, xacc_thread, yacc_thread);
            }

            double GSLSpline2D::eval_deriv(double x, double y, int derivx, int derivy) const {
                // Map (dx,dy) => n = derivx + 3*derivy
                // which gives 0 = f, 1 = f_x, 2 = f_xx, 3 = f_y, 4 = f_xy and (f_xyy), 6 =
                // f_yy
                const int n = derivx + 3 * derivy;

                if (spline == nullptr) {
                    std::string errormessage = "[GSLSpline2D::eval_deriv] Spline " + name + " has not been created!\n";
                    throw_error(errormessage);
                }
                if (n < 0 || n >= int(derivfunc.size())) {
                    std::string errormessage =
                        "[GSLSpline2D::eval_deriv] (" + std::to_string(derivx) + "," + std::to_string(derivy) + ") " +
                        "Expected (0,0) = f, (1,0) = f_x, (0,1), f_y, (2,0) = f_xx, (0,2) = f_yy or (1,1) = f_xy!\n";
                    throw_error(errormessage);
                }

                // If out of bounds show a warning and set x,y to boundary value
                out_of_bounds_check(x, y);
                x = std::max(xmin, x);
                x = std::min(xmax, x);
                y = std::max(ymin, y);
                y = std::min(ymax, y);

#ifdef USE_OMP
                gsl_interp_accel * xacc_thread = xaccs[omp_get_thread_num()];
                gsl_interp_accel * yacc_thread = yaccs[omp_get_thread_num()];
#else
                gsl_interp_accel * xacc_thread = xacc;
                gsl_interp_accel * yacc_thread = yacc;
#endif

                return derivfunc[n](spline, x, y, xacc_thread, yacc_thread);
            }

            //====================================================
            // Free up memory
            //====================================================
            void GSLSpline2D::free() {
                if (spline != nullptr) {
                    // Reset class variables
                    xmin = xmax = 0.0;
                    ymin = ymax = 0.0;
                    size_x = size_y = 0;
                    dx_min = dx_max = 0.0;
                    dy_min = dy_max = 0.0;

                    // Free the spline data
                    gsl_spline2d_free(spline);
                    spline = nullptr;

                    // Free accelerators
#ifdef USE_OMP
                    for (auto & xa : xaccs) {
                        gsl_interp_accel_free(xa);
                    }
                    std::vector<gsl_interp_accel *>().swap(xaccs);
                    for (auto & ya : yaccs) {
                        gsl_interp_accel_free(ya);
                    }
                    std::vector<gsl_interp_accel *>().swap(yaccs);
#else
                    gsl_interp_accel_free(xacc);
                    gsl_interp_accel_free(yacc);
                    xacc = nullptr;
                    yacc = nullptr;
#endif
                }
            }

            //====================================================
            // Destructor
            //====================================================
            GSLSpline2D::~GSLSpline2D() { free(); }

            //====================================================
            // Rest of the class methods
            //====================================================

            void GSLSpline2D::out_of_bounds_check(double x, double y) const {
                if (out_of_bounds_warning) {
                    if (x < xmin - dx_min || x > xmax + dx_max) {
                        std::cout << "Warning GSLSpline2D[" << name << "] ";
                        std::cout << "x = " << x << " is out of bounds (" << xmin << "," << xmax << ")\n";
                    }
                    if (y < ymin - dy_min || y > ymax + dy_max) {
                        std::cout << "Warning GSLSpline2D[" << name << "] ";
                        std::cout << "y = " << y << " is out of bounds (" << ymin << "," << ymax << ")\n";
                    }
                }
            }
            double GSLSpline2D::deriv_x(double x, double y) const { return eval_deriv(x, y, 1, 0); }
            double GSLSpline2D::deriv_xx(double x, double y) const { return eval_deriv(x, y, 2, 0); }
            double GSLSpline2D::deriv_y(double x, double y) const { return eval_deriv(x, y, 0, 1); }
            double GSLSpline2D::deriv_yy(double x, double y) const { return eval_deriv(x, y, 0, 2); }
            double GSLSpline2D::deriv_xy(double x, double y) const { return eval_deriv(x, y, 1, 1); }
            std::pair<double, double> GSLSpline2D::get_xrange() const { return {xmin, xmax}; }
            std::pair<double, double> GSLSpline2D::get_yrange() const { return {ymin, ymax}; }
            std::string GSLSpline2D::get_name() const { return name; }
            void GSLSpline2D::set_out_of_bounds_warning(bool v) { out_of_bounds_warning = v; }
        } // namespace SPLINE
    }     // namespace INTERPOLATION
} // namespace FML
