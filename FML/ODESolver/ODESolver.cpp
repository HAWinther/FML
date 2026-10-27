#include "ODESolver.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace FML {
    namespace SOLVERS {
        namespace ODESOLVER {

            void ODESolver::throw_error(std::string errormessage) const {
#if defined(USE_MPI) && !defined(ODESOLVER_THROW_ON_ERROR)
                std::cout << errormessage << std::flush;
                MPI_Abort(MPI_COMM_WORLD, 1);
                abort();
#else
                throw std::runtime_error(errormessage);
#endif
            }

            //==================================================================
            // Wrappers for conversions between std::function and the raw
            // function pointer required by GSL
            //==================================================================

            ODEFunctionJacobian no_jacobian = []([[maybe_unused]] double t,
                                                 [[maybe_unused]] const double * y,
                                                 [[maybe_unused]] double * dfdy,
                                                 [[maybe_unused]] double * dfdt) { return GSL_SUCCESS; };

            ODEFunctionJacobian * no_jacobian_ptr = &no_jacobian;

            struct ODEParamWrapper {
              private:
                const ODEFunction & deriv;
                const ODEFunctionJacobian & jacobian;

              public:
                ODEParamWrapper(const ODEFunction & _deriv, const ODEFunctionJacobian & _jacobian)
                    : deriv(_deriv), jacobian(_jacobian) {}
                auto get_deriv() -> const ODEFunction & { return deriv; }
                auto get_jacobian() -> const ODEFunctionJacobian & { return jacobian; }
            };

            int ode_wrapper(double x, const double * y, double * dydx, void * param) {
                auto * p = static_cast<struct ODEParamWrapper *>(param);
                return (p->get_deriv())(x, y, dydx);
            }

            int jacobian_wrapper(double t, const double * y, double * dfdy, double * dfdt, void * param) {
                auto * p = static_cast<struct ODEParamWrapper *>(param);
                return (p->get_jacobian())(t, y, dfdy, dfdt);
            }

            // Jacobian by forward finite differences of the ODE function: dfdy[i*n + j] = df_i/dy_j with the
            // step h_j = 1e-8 max(|y_j|, 1e-8), and dfdt with a step 1e-8 max(|t|, 1) backwards along the
            // direction of integration (so the function is never evaluated beyond the end point)
            struct FiniteDifferenceJacobian {
                const ODEFunction & deriv;
                size_t n;
                double direction;
                DVector ytmp, f0, f1;
                FiniteDifferenceJacobian(const ODEFunction & d, size_t _n, double dir)
                    : deriv(d), n(_n), direction(dir), ytmp(_n), f0(_n), f1(_n) {}
                int operator()(double t, const double * y, double * dfdy, double * dfdt) {
                    int status = deriv(t, y, f0.data());
                    if (status != GSL_SUCCESS)
                        return status;
                    ytmp.assign(y, y + n);
                    for (size_t j = 0; j < n; j++) {
                        const double h = 1e-8 * std::max(std::fabs(y[j]), 1e-8);
                        ytmp[j] = y[j] + h;
                        status = deriv(t, ytmp.data(), f1.data());
                        if (status != GSL_SUCCESS)
                            return status;
                        for (size_t i = 0; i < n; i++)
                            dfdy[i * n + j] = (f1[i] - f0[i]) / h;
                        ytmp[j] = y[j];
                    }
                    const double ht = -direction * 1e-8 * std::max(std::fabs(t), 1.0);
                    status = deriv(t + ht, y, f1.data());
                    if (status != GSL_SUCCESS)
                        return status;
                    for (size_t i = 0; i < n; i++)
                        dfdt[i] = (f1[i] - f0[i]) / ht;
                    return GSL_SUCCESS;
                }
            };

            //==================================================================
            // Constructors
            //==================================================================

            ODESolver::ODESolver(double hstart, double abserr, double relerr)
                : hstart(hstart), abserr(abserr), relerr(relerr) {}

            void ODESolver::solve(const ODEFunction & ode_equation,
                                  const DVector & xarr,
                                  const DVector & yinitial,
                                  const gsl_odeiv2_step_type * stepper,
                                  const ODEFunctionJacobian & jacobian) {
                // If no jacobian is provided pass nullptr to GSL (the dummy no_jacobian does not set dfdy, dfdt
                // so implicit steppers would use uninitialized memory). We then throw if the stepper needs it,
                // unless the jacobian is to be computed by finite differences
                const bool has_jacobian = (&jacobian != no_jacobian_ptr);
                if (!has_jacobian && finite_difference_jacobian && xarr.size() >= 2 && !yinitial.empty()) {
                    FiniteDifferenceJacobian fd(ode_equation, yinitial.size(), xarr[1] > xarr[0] ? 1.0 : -1.0);
                    const ODEFunctionJacobian fdjac = [&fd](double t, const double * y, double * dfdy, double * dfdt) {
                        return fd(t, y, dfdy, dfdt);
                    };
                    struct ODEParamWrapper equations(ode_equation, fdjac);
                    solve(ode_wrapper, &equations, xarr, yinitial, stepper, jacobian_wrapper);
                    return;
                }
                struct ODEParamWrapper equations(ode_equation, jacobian);
                solve(ode_wrapper, &equations, xarr, yinitial, stepper, has_jacobian ? jacobian_wrapper : nullptr);
            }

            void ODESolver::set_finite_difference_jacobian(bool onoff) { finite_difference_jacobian = onoff; }

            //==================================================================
            // Class methods
            //==================================================================

            void ODESolver::solve(ODEFunctionPointer ode_equation,
                                  void * parameters,
                                  const DVector & xarr,
                                  const DVector & yinitial,
                                  const gsl_odeiv2_step_type * stepper,
                                  ODEFunctionPointerJacobian jacobian) {
                // Store the number of equations and size of x-array
                nequations = int(yinitial.size());
                num_x_points = int(xarr.size());
                if (num_x_points < 2) {
                    std::string errormessage = "[ODESolver::solve] The xarray needs to have at least size 2\n";
                    throw_error(errormessage);
                }
                if (nequations < 1) {
                    std::string errormessage = "[ODESolver::solve] The yinitial is empty\n";
                    throw_error(errormessage);
                }

                // Implicit steppers need the jacobian (GSL would call a null function pointer)
                if (jacobian == nullptr) {
                    for (auto st : {gsl_odeiv2_step_rk1imp,
                                    gsl_odeiv2_step_rk2imp,
                                    gsl_odeiv2_step_rk4imp,
                                    gsl_odeiv2_step_bsimp,
                                    gsl_odeiv2_step_msbdf}) {
                        if (stepper == st) {
                            std::string errormessage = "[ODESolver::solve] The stepper " + std::string(stepper->name) +
                                                       " requires the jacobian, but none was provided\n";
                            throw_error(errormessage);
                        }
                    }
                }

                // Are we integrating forward or backward?
                double sign = xarr[1] > xarr[0] ? 1.0 : -1.0;

                // Set up the ODE system
                gsl_odeiv2_system ode_system = {ode_equation, jacobian, size_t(yinitial.size()), parameters};
                // The driver is freed also if an exception is thrown (e.g. by the users ODE function)
                std::unique_ptr<gsl_odeiv2_driver, decltype(&gsl_odeiv2_driver_free)> ode_driver_ptr(
                    gsl_odeiv2_driver_alloc_y_new(&ode_system, stepper, std::abs(hstart) * sign, abserr, relerr),
                    &gsl_odeiv2_driver_free);
                gsl_odeiv2_driver * ode_driver = ode_driver_ptr.get();

                // Initialize with the initial condition
                double x = xarr[0];
                DVector y(yinitial);
                DVector dydx(nequations);
                ode_equation(x, y.data(), dydx.data(), parameters);

                // Allocate memory for the the results: data[i][j] = y_j(x_i)
                data = std::vector<DVector>(xarr.size(), yinitial);
                derivative_data = std::vector<DVector>(xarr.size(), dydx);

                if (verbose) {
                    std::cout << "ODESolver step " << std::setw(5) << 0 << " / " << num_x_points - 1 << " x: ["
                              << std::setw(10) << x << "] ";
                    std::cout << "y: [";
                    for (auto & ycomp : y) {
                        std::cout << " " << std::setw(10) << ycomp << " ";
                    }
                    std::cout << "]" << std::endl;
                }

                // Solve it step by step...
                for (int i = 1; i < num_x_points; i++) {
                    const double xnew = xarr[i];

                    if (verbose) {
                        std::cout << "ODESolver step " << std::setw(5) << i << " / " << num_x_points - 1 << " x: ["
                                  << std::setw(10) << xnew << "] ";
                    }

                    // Integrate from x -> xnew
                    const int status = gsl_odeiv2_driver_apply(ode_driver, &x, xnew, y.data());
                    if (status != GSL_SUCCESS) {
                        ode_driver_ptr.reset();
                        std::string errormessage =
                            "[ODESolver::solve] GSL gsl_odeiv2_driver_apply returned non-successful at x = " +
                            std::to_string(x) + " (target " + std::to_string(xnew) + "): " + gsl_strerror(status) + "\n";
                        throw_error(errormessage);
                    }

                    if (verbose) {
                        std::cout << "y: [";
                        for (auto & ycomp : y) {
                            std::cout << " " << std::setw(10) << ycomp << " ";
                        }
                        std::cout << "]" << std::endl;
                    }

                    // Store the derivative
                    ode_equation(x, y.data(), dydx.data(), parameters);

                    // Copy over result
                    data[i] = y;
                    derivative_data[i] = dydx;
                }
            }

            void ODESolver::set_verbose(bool onoff) { verbose = onoff; }

            DVector2D ODESolver::get_data() const { return data; }

            DVector2D ODESolver::get_data_transpose() const {
                auto data_transpose = std::vector<DVector>(nequations, DVector(num_x_points));
                for (int ix = 0; ix < num_x_points; ix++) {
                    for (int icomponent = 0; icomponent < nequations; icomponent++) {
                        data_transpose[icomponent][ix] = data[ix][icomponent];
                    }
                }
                return data_transpose;
            }

            DVector ODESolver::get_final_data() const {
                DVector res(nequations);
                for (int icomponent = 0; icomponent < nequations; icomponent++) {
                    res[icomponent] = data[num_x_points - 1][icomponent];
                }
                return res;
            }

            double ODESolver::get_final_data_by_component(int icomponent) const {
                return data[num_x_points - 1][icomponent];
            }

            DVector ODESolver::get_data_by_component(int icomponent) const {
                DVector res(num_x_points);
                for (int ix = 0; ix < num_x_points; ix++) {
                    res[ix] = data[ix][icomponent];
                }
                return res;
            }

            DVector ODESolver::get_data_by_xindex(int ix) const {
                DVector res(nequations);
                for (int icomponent = 0; icomponent < nequations; icomponent++) {
                    res[icomponent] = data[ix][icomponent];
                }
                return res;
            }

            DVector2D ODESolver::get_derivative_data() const { return derivative_data; }

            DVector ODESolver::get_derivative_data_by_component(int icomponent) const {
                DVector res(num_x_points);
                for (int ix = 0; ix < num_x_points; ix++) {
                    res[ix] = derivative_data[ix][icomponent];
                }
                return res;
            }

            void ODESolver::set_accuracy(const double h, const double a, const double r) {
                hstart = h;
                abserr = a;
                relerr = r;
            }
        } // namespace ODESOLVER
    }     // namespace SOLVERS
} // namespace FML
