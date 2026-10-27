#ifndef FML_COSMOLOGY_COMMON_H
#define FML_COSMOLOGY_COMMON_H
// Solver conventions: SI units internally; time coordinate x = ln(a).
// Compatibility names for shared FML components. Numerical implementations live in their owning modules.
#include <FML/FileUtils/DataPath.h>
#include <FML/Math/NumericalMethods.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/Spline/Spline.h>
#include <FML/Timing/Timer.h>
#include <FML/Units/Units.h>

namespace FML::COSMOLOGY {
using Spline1D = FML::INTERPOLATION::SPLINE::GSLSpline;
using Spline2D = FML::INTERPOLATION::SPLINE::GSLSpline2D;
using UniformSpline1D = FML::INTERPOLATION::SPLINE::UniformSpline;
using ODESolver = FML::SOLVERS::ODESOLVER::ODESolver;
using ODEFunction = FML::SOLVERS::ODESOLVER::ODEFunction;
using ODEStepper = const gsl_odeiv2_step_type *;
using FML::INTERPOLATION::SPLINE::spline2d_type;
namespace Constants = FML::UTILS::SI;

namespace Utils {
    using FML::FILEUTILS::fml_file;
    using FML::MATH::integrate;
    using FML::MATH::trapz;
    using FML::MATH::find_root;

    // Preserve cosmology's strict point count and physical-endpoint logspace convention.
    inline std::vector<double> linspace(double xmin, double xmax, int n) {
        if (n < 2) throw std::invalid_argument("linspace: need n >= 2");
        return FML::MATH::linspace(xmin, xmax, n);
    }
    inline std::vector<double> logspace(double xmin, double xmax, int n) {
        return FML::MATH::geomspace(xmin, xmax, n);
    }
    inline double j_ell(int ell, double x) { return FML::MATH::spherical_bessel_j(ell, x); }
    inline void j_ell_array(int lmax, double x, double * values) {
        FML::MATH::spherical_bessel_j_array(lmax, x, values);
    }

    // Solver output policy remains local to cosmology.
    inline bool verbose = true;
    class Timer : public FML::UTILS::Timer {
      public:
        using FML::UTILS::Timer::Timer;
        void print() const { FML::UTILS::Timer::print(verbose); }
    };
}
}
#endif
