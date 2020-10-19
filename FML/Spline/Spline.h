#ifndef SPLINE_HEADER
#define SPLINE_HEADER
#include <cassert>
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <iostream>
#include <vector>
#ifdef USE_OMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
    namespace INTERPOLATION {

        /// This namespace deals with creating and using splines
        namespace SPLINE {

            // Type aliases
            class GSLSpline;
            class GSLSpline2D;
            using Spline = GSLSpline;
            using Spline2D = GSLSpline2D;
            using DVector = std::vector<double>;
            using DVector2D = std::vector<DVector>;

#ifndef SPLINE_FIDUCIAL_INTERPOL_TYPE
#define SPLINE_FIDUCIAL_INTERPOL_TYPE gsl_interp_cspline
#endif
#ifndef SPLINE_FIDUCIAL_INTERPOL_TYPE_2D
#define SPLINE_FIDUCIAL_INTERPOL_TYPE_2D gsl_interp2d_bicubic
#endif
#ifndef SPLINE_FIDUCIAL_SPLINE_WARNING
#define SPLINE_FIDUCIAL_SPLINE_WARNING false
#endif

            //====================================================
            ///
            /// This is a wrapper class for easy use of GSL splines
            /// It is (OpenMP) thread safe as long as splines are
            /// not created by a subthread and later tried to be used
            /// by more than that thread!
            ///
            /// The fiducial splines are cubic splines for 1D and
            /// bicubic for 2D assuming natural (y''=0) boundary
            /// conditions at the end-points
            ///
            /// The fiducial choice allows to get the function value
            /// and up to second order derivatives. NB: if you change
            /// the fiducial choice to a lower order spline,
            /// e.g to gsl_interp_linear, then the second derivative
            /// deriv_xx etc. will not work
            ///
            /// If evaluating out of bounds it will return the closest
            /// value. To get a warning when this happens call
            /// set_out_of_bounds_warning(true) before using the spline
            /// or set the define SPLINE_FIDUCIAL_SPLINE_WARNING to be true
            ///
            /// Errors are handled via the throw_error function
            ///
            /// Compile time defines:
            ///
            /// USE_MPI                           : Use MPI (only effect is in on how errors are handled)
            ///
            /// USE_OMP                           : Use OpenMP
            ///
            /// SPLINE_FIDUCIAL_INTERPOL_TYPE     : Cubic spline is the fiducial choice
            ///
            /// SPLINE_FIDUCIAL_INTERPOL_TYPE_2D  : Bicubic is the fiducial choice
            ///
            /// SPLINE_FIDUCIAL_SPLINE_WARNING    : Show warnings if we evaluate out of bounds
            ///
            //====================================================

            class GSLSpline {
              private:
                // GSL spline
                gsl_spline * spline{nullptr};

                // If we use threads then we must have a unique accelerator per thread
#ifdef USE_OMP
                std::vector<gsl_interp_accel *> xaccs{};
#else
                gsl_interp_accel * xacc{nullptr};
#endif

                const gsl_interp_type * interpoltype_used = SPLINE_FIDUCIAL_INTERPOL_TYPE;

                // Info about the spline
                int size_x{};
                double xmin{};
                double xmax{};
                double dx_min{};
                double dx_max{};
                std::string name{"NoName"};

                // Print warnings if out of bounds if wanted
                bool out_of_bounds_warning = SPLINE_FIDUCIAL_SPLINE_WARNING;
                void out_of_bounds_check(double x) const;

                // Handle errors
                void throw_error(std::string errormessage) const;

              public:
                GSLSpline() = default;

                /// Construct giving the spline a name (useful for error/warning messages)
                GSLSpline(std::string name);

                /// Construct a spline from pointers to x and y. Both must have nx elements.
                GSLSpline(double * x,
                          double * y,
                          int nx,
                          std::string splinename = "NoName",
                          const gsl_interp_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE);
                /// Construct a spline from vectors x and y. Both must have the same size.
                GSLSpline(const DVector & x,
                          const DVector & y,
                          std::string splinename = "NoName",
                          const gsl_interp_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE);
                GSLSpline(const GSLSpline & rhs);
                GSLSpline & operator=(const GSLSpline & rhs);
                ~GSLSpline();

                /// Is the spline created or not?
                explicit operator bool() const;

                /// Create a spline from pointers to x and y. Both must have nx elements.
                void create(const double * x,
                            const double * y,
                            int nx,
                            std::string splinename = "NoName",
                            const gsl_interp_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE);
                /// Create a spline from vectors x and y. Both must have the same size.
                void create(const DVector & x,
                            const DVector & y,
                            std::string splinename = "NoName",
                            const gsl_interp_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE);

                // Methods for spline lookup of function and its derivatives
                /// Overload of the () operator for easy evaluation of the spline
                double operator()(double x) const;
                /// Get the value of the spline (if out of bounds we use the closest value)
                double eval(double x) const;
                double eval_deriv(double x, int deriv) const;
                /// Get the value of the first derivative of the spline
                double deriv_x(double x) const;
                /// Get the value of the second derivative of the spline
                double deriv_xx(double x) const;

                // Some useful info
                /// Get the range the spline was created on
                std::pair<double, double> get_xrange() const;
                /// Get the name of the spline
                std::string get_name() const;
                /// Turn on/off warnings if we try to evaluate out of bounds (we use closest value in that case)
                void set_out_of_bounds_warning(bool v);

                /// Get the raw GSL data for x used to create the spline
                DVector get_x_data() { return DVector(spline->x, spline->x + spline->size); }
                /// Get the raw GSL data for y=f(x) used to create the spline
                DVector get_y_data() { return DVector(spline->y, spline->y + spline->size); }

                /// Free up memory associated with the spline
                void free();
            };

            /// Create 2D splines
            class GSLSpline2D {
              private:
                // GSL spline
                gsl_spline2d * spline{nullptr};

                // If we use threads then we must have a unique accelerator per thread
#ifdef USE_OMP
                std::vector<gsl_interp_accel *> xaccs{};
                std::vector<gsl_interp_accel *> yaccs{};
#else
                gsl_interp_accel * xacc{nullptr};
                gsl_interp_accel * yacc{nullptr};
#endif
                const gsl_interp2d_type * interpoltype_used = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D;

                // Info about the spline
                int size_x{};
                int size_y{};
                double xmin{};
                double xmax{};
                double ymin{};
                double ymax{};
                double dx_min{};
                double dx_max{};
                double dy_min{};
                double dy_max{};
                std::string name{"NoName"};

                // Print warnings if out of bounds if wanted
                bool out_of_bounds_warning = SPLINE_FIDUCIAL_SPLINE_WARNING;
                void out_of_bounds_check(double x, double y) const;

                // A list of all the (up to second order) derivative functions in GSL,
                // We map Dx^nx Dy^ny f => nx + 3*ny in the list for use in eval_deriv
                typedef double (
                    *evalfunc)(const gsl_spline2d *, double, double, gsl_interp_accel *, gsl_interp_accel *);

                std::vector<evalfunc> derivfunc{gsl_spline2d_eval,
                                                gsl_spline2d_eval_deriv_x,
                                                gsl_spline2d_eval_deriv_xx,
                                                gsl_spline2d_eval_deriv_y,
                                                gsl_spline2d_eval_deriv_xy,
                                                nullptr,
                                                gsl_spline2d_eval_deriv_yy};

                // Handle errors
                void throw_error(std::string errormessage) const;

              public:
                GSLSpline2D() = default;
                GSLSpline2D(std::string name);
                GSLSpline2D(const double * x,
                            const double * y,
                            const double * z,
                            int nx,
                            int ny,
                            std::string splinename = "NoName",
                            const gsl_interp2d_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D);
                GSLSpline2D(const DVector & x,
                            const DVector & y,
                            const DVector & z,
                            std::string splinename = "NoName",
                            const gsl_interp2d_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D);
                GSLSpline2D(const DVector & x,
                            const DVector & y,
                            const DVector2D & z,
                            std::string splinename = "NoName",
                            const gsl_interp2d_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D);
                GSLSpline2D(const GSLSpline2D & rhs);
                GSLSpline2D & operator=(const GSLSpline2D & rhs);
                ~GSLSpline2D();

                /// Is the spline created or not?
                explicit operator bool() const;

                /// Create a spline from pointers to x, y and z. z must have nx * ny elements.
                void create(const double * x,
                            const double * y,
                            const double * z,
                            int nx,
                            int ny,
                            std::string splinename = "NoName",
                            const gsl_interp2d_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D);
                /// Create a spline from vectors to x, y and z. Size of z must be nx * ny.
                void create(const DVector & x,
                            const DVector & y,
                            const DVector & z,
                            std::string splinename = "NoName",
                            const gsl_interp2d_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D);
                /// Create a spline from vectors to x, y and z. Size of z must be (nx, ny).
                void create(const DVector & x,
                            const DVector & y,
                            const DVector2D & z,
                            std::string splinename = "NoName",
                            const gsl_interp2d_type * interpoltype = SPLINE_FIDUCIAL_INTERPOL_TYPE_2D);

                // Methods for spline lookup of function and its derivatives
                /// Overload of the () operator for easy evaluation of the spline
                double operator()(double x, double y) const;
                /// Get the value of the spline (if out of bounds we use the closest value)
                double eval(double x, double y) const;
                /// General method to fetch derivatives. derivx is the number of x-derivatives and derivy is the number
                /// of y-derivatives (how many are availiable depends on the spline method).
                double eval_deriv(double x, double y, int derivx, int derivy) const;
                /// Get the value of the x-derivative of the spline
                double deriv_x(double x, double y) const;
                /// Get the value of the second x-derivative of the spline
                double deriv_xx(double x, double y) const;
                /// Get the value of the x,y-derivative of the spline
                double deriv_xy(double x, double y) const;
                /// Get the value of the y-derivative of the spline
                double deriv_y(double x, double y) const;
                /// Get the value of the second y-derivative of the spline
                double deriv_yy(double x, double y) const;

                // Some useful info
                /// Get the x-range the spline was created on
                std::pair<double, double> get_xrange() const;
                /// Get the y-range the spline was created on
                std::pair<double, double> get_yrange() const;
                /// Get the name of the spline
                std::string get_name() const;
                /// Turn on/off warnings if we try to evaluate out of bounds (we use closest value in that case)
                void set_out_of_bounds_warning(bool v);

                /// Get the raw GSL data for x used to create the spline
                DVector get_x_data() { return DVector(spline->xarr, spline->xarr + size_x); }
                /// Get the raw GSL data for y used to create the spline
                DVector get_y_data() { return DVector(spline->yarr, spline->yarr + size_y); }
                /// Get the raw GSL data for z=f(x,y) used to create the spline
                DVector get_z_data() { return DVector(spline->zarr, spline->zarr + size_x * size_y); }

                /// Free up memory associated with the spline
                void free();
            };
        } // namespace SPLINE
    }     // namespace INTERPOLATION
} // namespace FML
#endif
