#ifndef COSMOLOGY_SCALARFIELD_HEADER
#define COSMOLOGY_SCALARFIELD_HEADER

#include <FML/Global/Global.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

//=============================================================================
/// Base class for cosmologies with a minimally coupled scalar field phi (quintessence,
/// early dark energy) on top of baryons, CDM, radiation, neutrinos, curvature and
/// (optionally) a cosmological constant. The field only changes the background H(a)
/// (its perturbations are neglected, which is a good approximation on sub-horizon scales
/// for a light field).
///
/// Units: phi in units of the reduced Planck mass Mpl, the potential as
/// U(phi) = V(phi) / (3 H0^2 Mpl^2) (so U is a density parameter) and ' = d/dlna:
///
///   E^2 = (rho_other + U) / (1 - phi'^2 / 6)        (rho in units of the critical density today)
///   phi'' + (3 + dlnH/dlna) phi' + 3 dU/dphi / E^2 = 0
///   rho_phi = E^2 phi'^2 / 6 + U,   p_phi = E^2 phi'^2 / 6 - U
///
/// The field is integrated from a_start (deep in the radiation era, where it is frozen) to
/// a_end. For potentials with a minimum, once the field oscillates around it many times per
/// e-fold (half period < 0.025 in ln a) we replace it by a fluid with the cycle-averaged
/// equation of state w_oscillating() (w = (n-1)/(n+1) for V ~ |phi - phi_min|^(2n)).
/// Derived classes give the potential and determine its parameters in init() (e.g. by
/// shooting for the dark energy density today) using solve_field().
//=============================================================================
class CosmologyScalarField : public Cosmology {
  public:
    CosmologyScalarField() : Cosmology(1e-9, 10.0, 2500) {}

    double HoverH0_of_a(double a) const override { return std::sqrt(E2_of_a(a)); }

    double dlogHdloga_of_a(double a) const override {
        const double E2 = E2_of_a(a);
        return (drho_other_dloga(a) + drho_phi_dloga(a)) / (2.0 * E2);
    }

    /// The density parameter of all the dark energy (field + cosmological constant)
    double get_OmegaLambda(double a = 1.0) const override {
        if (not solved)
            return OmegaLambda;
        return (rho_phi(a) + OmegaLambda_bare) / E2_of_a(a);
    }

    /// The field density parameter
    double get_OmegaPhi(double a) const { return rho_phi(a) / E2_of_a(a); }

    /// The equation of state of the field
    double w_phi(double a) const {
        if (not solved)
            return -1.0;
        // p = -rho - (drho/dlna)/3
        return -1.0 - drho_phi_dloga(a) / (3.0 * rho_phi(a));
    }

    /// The field (in units of Mpl). Constant (phi_min) after the switch to the fluid
    double phi_of_a(double a) const { return phi_spline(std::log(a)); }

    void output_header(std::ofstream & fp) const override {
        Cosmology::output_header(fp);
        fp << " ";
        output_element(fp, "OmegaPhi");
        fp << " ";
        output_element(fp, "w_phi");
        fp << " ";
        output_element(fp, "phi/Mpl");
    }

    void output_row(std::ofstream & fp, double a) const override {
        Cosmology::output_row(fp, a);
        fp << " ";
        output_element(fp, get_OmegaPhi(a));
        fp << " ";
        output_element(fp, w_phi(a));
        fp << " ";
        output_element(fp, phi_of_a(a));
    }

  protected:
    /// The potential U(phi) = V(phi) / (3 H0^2 Mpl^2) and its derivative (phi in units of Mpl)
    virtual double U(double phi) const = 0;
    virtual double dUdphi(double phi) const = 0;

    /// For potentials with a minimum: where it is and the cycle-averaged w of oscillations around it
    virtual bool has_minimum() const { return false; }
    virtual double phi_min() const { return 0.0; }
    virtual double w_oscillating() const { return 0.0; }

    /// A cosmological constant in addition to the field
    double OmegaLambda_bare{0.0};

    /// The range we integrate the field over
    const double a_start{1e-10};
    const double a_end{10.0};
    const int npts_per_efold{1000};

    /// Everything except the field
    double rho_other(double a) const {
        return OmegaK / (a * a) + (OmegaCDM + Omegab) / (a * a * a) + OmegaR / (a * a * a * a) +
               this->get_rhoNu_exact(a) + OmegaLambda_bare;
    }
    double drho_other_dloga(double a) const {
        return -2.0 * OmegaK / (a * a) - 3.0 * (OmegaCDM + Omegab) / (a * a * a) - 4.0 * OmegaR / (a * a * a * a) +
               this->get_drhoNudloga_exact(a);
    }

    double rho_phi(double a) const { return std::exp(log_rho_phi_spline(std::log(a))); }
    double drho_phi_dloga(double a) const {
        if (a < a_start or a > a_end)
            return 0.0;
        return rho_phi(a) * log_rho_phi_spline.deriv_x(std::log(a));
    }
    double E2_of_a(double a) const { return rho_other(a) + rho_phi(a); }

    /// Information about the last solve
    struct FieldSolution {
        bool switched_to_fluid{false};
        double a_fluid{0.0};
        double peak_fraction{0.0}; // max of rho_phi / rho_total
        double a_peak{0.0};        // ...and where it is
    };
    FieldSolution last_solution;
    bool solved{false};

    //========================================================================
    /// Integrate the field from a_start (phi = phi_ini, phi' = dphi_ini) to a_stop (default a_end)
    /// and make the splines. Throws if the field makes E^2 <= 0
    //========================================================================
    void solve_field(double phi_ini, double dphi_ini, double a_stop = -1.0) {
        if (a_stop <= 0.0)
            a_stop = a_end;
        const double x_start = std::log(a_start);
        const double x_stop = std::log(a_stop);
        const int npts = int((x_stop - x_start) * npts_per_efold) + 1;
        const double dx = (x_stop - x_start) / double(npts - 1);

        auto E2_field = [&](double a, double phi, double dphi) {
            const double kin = 1.0 - dphi * dphi / 6.0;
            return (rho_other(a) + U(phi)) / kin;
        };

        FML::SOLVERS::ODESOLVER::ODEFunction deriv = [&](double x, const double * y, double * dydx) {
            const double a = std::exp(x);
            const double phi = y[0];
            const double dphi = y[1];
            const double E2 = E2_field(a, phi, dphi);
            const double dlogH = (drho_other_dloga(a) - E2 * dphi * dphi) / (2.0 * E2);
            dydx[0] = dphi;
            dydx[1] = -(3.0 + dlogH) * dphi - 3.0 * dUdphi(phi) / E2;
            return GSL_SUCCESS;
        };

        DVector x_arr(npts), phi_arr(npts), logrho_arr(npts);
        for (int i = 0; i < npts; i++)
            x_arr[i] = x_start + dx * i;

        last_solution = FieldSolution{};
        std::vector<double> y{phi_ini, dphi_ini};
        const int chunk = npts_per_efold / 4;
        double last_crossing = -1e100;
        int ncross = 0;
        bool fluid = false;
        double x_fluid = 0.0, logrho_fluid = 0.0;
        const double w_osc = w_oscillating();

        auto store = [&](int i, double phi, double dphi) {
            const double a = std::exp(x_arr[i]);
            const double E2 = E2_field(a, phi, dphi);
            if (not(E2 > 0.0) or not std::isfinite(E2))
                throw std::runtime_error(name + ": the field gives E^2 <= 0 (or nan) at a = " + std::to_string(a));
            const double rho = E2 * dphi * dphi / 6.0 + U(phi);
            if (not(rho > 0.0))
                throw std::runtime_error(name + ": the field energy density is <= 0 at a = " + std::to_string(a) +
                                         " (the potential must be positive)");
            phi_arr[i] = phi;
            logrho_arr[i] = std::log(rho);
        };
        store(0, y[0], y[1]);

        int i0 = 0;
        while (i0 < npts - 1) {
            const int i1 = std::min(i0 + chunk, npts - 1);
            if (fluid) {
                for (int i = i0 + 1; i <= i1; i++) {
                    phi_arr[i] = phi_min();
                    logrho_arr[i] = logrho_fluid - 3.0 * (1.0 + w_osc) * (x_arr[i] - x_fluid);
                }
                i0 = i1;
                continue;
            }
            DVector xs(x_arr.begin() + i0, x_arr.begin() + i1 + 1);
            FML::SOLVERS::ODESOLVER::ODESolver ode(1e-4 * dx, 1e-12, 1e-10);
            ode.solve(deriv, xs, y);
            auto sol = ode.get_data();
            for (int j = 1; j < int(xs.size()); j++) {
                const int i = i0 + j;
                store(i, sol[j][0], sol[j][1]);
                // Oscillations around the minimum: switch to a fluid when the field oscillates fast
                if (has_minimum() and not fluid) {
                    const double s0 = sol[j - 1][0] - phi_min(), s1 = sol[j][0] - phi_min();
                    if (s0 * s1 < 0.0) {
                        const double xc = xs[j - 1] + (xs[j] - xs[j - 1]) * s0 / (s0 - s1);
                        ncross++;
                        if (ncross >= 3 and xc - last_crossing < 0.025) {
                            fluid = true;
                            x_fluid = x_arr[i];
                            logrho_fluid = logrho_arr[i];
                            last_solution.switched_to_fluid = true;
                            last_solution.a_fluid = std::exp(x_fluid);
                            // Fill the rest of this chunk as a fluid
                            for (int k = i + 1; k <= i1; k++) {
                                phi_arr[k] = phi_min();
                                logrho_arr[k] = logrho_fluid - 3.0 * (1.0 + w_osc) * (x_arr[k] - x_fluid);
                            }
                            break;
                        }
                        last_crossing = xc;
                    }
                }
            }
            y = {sol.back()[0], sol.back()[1]};
            i0 = i1;
        }

        // Beyond a_stop (when we only solve part of the range) the splines return the last value
        log_rho_phi_spline.create(x_arr, logrho_arr, name + " log(rho_phi)(log(a))");
        phi_spline.create(x_arr, phi_arr, name + " phi(log(a))");
        solved = true;

        // The maximum of rho_phi / rho_total (refined with a parabola through the three points around it)
        int imax = 0;
        std::vector<double> logfrac(npts);
        for (int i = 0; i < npts; i++) {
            const double a = std::exp(x_arr[i]);
            const double rho = std::exp(logrho_arr[i]);
            logfrac[i] = std::log(rho / (rho_other(a) + rho));
            if (logfrac[i] > logfrac[imax])
                imax = i;
        }
        double xpeak = x_arr[imax], lpeak = logfrac[imax];
        if (imax > 0 and imax < npts - 1) {
            const double ym = logfrac[imax - 1], y0 = logfrac[imax], yp = logfrac[imax + 1];
            const double denom = ym - 2.0 * y0 + yp;
            if (denom < 0.0) {
                const double t = 0.5 * (ym - yp) / denom;
                xpeak = x_arr[imax] + t * dx;
                lpeak = y0 - 0.25 * (ym - yp) * t;
            }
        }
        last_solution.a_peak = std::exp(xpeak);
        last_solution.peak_fraction = std::exp(lpeak);
    }

    /// Root finding (Illinois / regula falsi on a bracket) for f(x) = 0 with f increasing.
    /// Expands the bracket [lo, hi] until it contains the root
    static double find_root_increasing(const std::function<double(double)> & f,
                                       double lo,
                                       double hi,
                                       double tol_f,
                                       const std::string & what) {
        double flo = f(lo), fhi = f(hi);
        for (int i = 0; i < 60 and flo > 0.0; i++) {
            hi = lo;
            fhi = flo;
            lo -= (hi - lo) + 1.0;
            flo = f(lo);
        }
        for (int i = 0; i < 60 and fhi < 0.0; i++) {
            lo = hi;
            flo = fhi;
            hi += (hi - lo) + 1.0;
            fhi = f(hi);
        }
        if (flo > 0.0 or fhi < 0.0)
            throw std::runtime_error(what + ": could not bracket the solution");
        int side = 0;
        double x = lo;
        for (int iter = 0; iter < 200; iter++) {
            x = (lo * fhi - hi * flo) / (fhi - flo);
            const double fx = f(x);
            if (std::fabs(fx) < tol_f or std::fabs(hi - lo) < 1e-14 * (1.0 + std::fabs(x)))
                return x;
            if (fx < 0.0) {
                lo = x;
                flo = fx;
                if (side == -1)
                    fhi *= 0.5;
                side = -1;
            } else {
                hi = x;
                fhi = fx;
                if (side == 1)
                    flo *= 0.5;
                side = 1;
            }
        }
        throw std::runtime_error(what + ": root finding did not converge");
        return x;
    }

    Spline log_rho_phi_spline{"[log(rho_phi) not yet created]"};
    Spline phi_spline{"[phi not yet created]"};
};

#endif
