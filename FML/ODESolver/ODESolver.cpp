#include "ODESolver.h"

namespace FML {
    namespace SOLVERS {
        namespace ODESOLVER {

            void ODESolver::throw_error(std::string errormessage) const {
#ifdef USE_MPI
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

            //==================================================================
            // Constructors
            //==================================================================

            ODESolver::ODESolver(double hstart, double abserr, double relerr)
                : hstart(hstart), abserr(abserr), relerr(relerr) {}

            void ODESolver::solve(ODEFunction & ode_equation,
                                  DVector & xarr,
                                  DVector & yinitial,
                                  const gsl_odeiv2_step_type * stepper,
                                  ODEFunctionJacobian & jacobian) {
                struct ODEParamWrapper equations(ode_equation, jacobian);
                solve(ode_wrapper, &equations, xarr, yinitial, stepper, jacobian_wrapper);
            }

            //==================================================================
            // Class methods
            //==================================================================

            void ODESolver::solve(ODEFunctionPointer ode_equation,
                                  void * parameters,
                                  DVector & xarr,
                                  DVector & yinitial,
                                  const gsl_odeiv2_step_type * stepper,
                                  ODEFunctionPointerJacobian jacobian) {
                // Store the number of equations and size of x-array
                nequations = int(yinitial.size());
                num_x_points = int(xarr.size());
                if (num_x_points < 2) {
                    std::string errormessage = "[ODESolver::solve] The xarray needs to have atleast size 2\n";
                    throw_error(errormessage);
                }
                if (nequations < 1) {
                    std::string errormessage = "[ODESolver::solve] The yinitial is empty\n";
                    throw_error(errormessage);
                }

                // Are we integrating forward or backward?
                double sign = xarr[1] > xarr[0] ? 1.0 : -1.0;

                // Set up the ODE system
                gsl_odeiv2_system ode_system = {ode_equation, jacobian, size_t(yinitial.size()), parameters};
                gsl_odeiv2_driver * ode_driver =
                    gsl_odeiv2_driver_alloc_y_new(&ode_system, stepper, std::abs(hstart) * sign, abserr, relerr);

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
                        std::string errormessage =
                            "[ODESolver::solve] GSL gsl_odeiv2_driver_apply returned non-successful\n";
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
                gsl_odeiv2_driver_free(ode_driver);
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
