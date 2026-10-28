#include <FML/Cosmology/Models/EarlyDarkEnergy.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    EarlyDarkEnergySpecies::EarlyDarkEnergySpecies(double f_EDE,
                                                   double log10_zc,
                                                   double theta_i_in,
                                                   double n_in,
                                                   const std::vector<std::shared_ptr<const Species>> & others_in,
                                                   double OmegaK_in)
        : ScalarFieldSpecies(others_in, OmegaK_in), fEDE(f_EDE), x_c(-std::log1p(std::pow(10.0, log10_zc))),
          theta_i(theta_i_in), n(n_in) {
        if (!(fEDE > 0.0 && fEDE < 1.0))
            throw std::invalid_argument("early dark energy: need 0 < f_EDE < 1");
        if (!(theta_i > 0.0 && theta_i < M_PI))
            throw std::invalid_argument("early dark energy: need 0 < theta_i < pi");
        // for n <= 2 the oscillating field (w = (n-1)/(n+1) <= 1/3) does not dilute faster than radiation, so its
        // fraction does not peak when it starts to oscillate and (f_EDE, z_c) do not define the model
        if (!(n > 2.0))
            throw std::invalid_argument("early dark energy: need n > 2 (f_EDE, z_c parameterization)");
        if (!(x_c > -18.0 && x_c < -1.0))
            throw std::invalid_argument("early dark energy: need 0.5 < log10(z_c) < 7.5");

        double rho_o0 = OmegaK;
        for (const auto & s : others)
            rho_o0 += s->rho(0.0);
        OmegaL_ = 1.0 - rho_o0;

        // Newton iteration on (ln mu, ln f) for (ln f_max, x_peak) = (ln f_EDE, x_c), on the solution up to x_c + 3
        const double u_i = 1.0 - std::cos(theta_i), s_i = std::sin(theta_i);
        for (int closure = 0; closure < 5; closure++) {
            auto residual = [&](double lnmu, double lnf, double & g1, double & g2) {
                mu_ = std::exp(lnmu);
                f_ = std::exp(lnf);
                solve_to(x_c + 3.0);
                double fm, xp;
                peak(fm, xp);
                g1 = std::log(fm / fEDE);
                g2 = xp - x_c;
            };
            // Initial guess: the frozen field has rho = mu u_i^n, and it starts rolling when the force term of
            // theta_xx = -3 mu n u^(n-1) sin(theta)/(f^2 E^2) is ~ 3 theta_i
            const double E2c = rho_others(x_c);
            double lnmu = std::log(fEDE * E2c / std::pow(u_i, n));
            double lnf = 0.5 * std::log(std::exp(lnmu) * n * std::pow(u_i, n - 1.0) * s_i / (theta_i * E2c));
            double g1, g2;
            residual(lnmu, lnf, g1, g2);
            for (int it = 0;; it++) {
                if (std::fabs(g1) < 1e-10 && std::fabs(g2) < 1e-9)
                    break;
                if (it > 40)
                    throw std::runtime_error("early dark energy: no solution for f_EDE and z_c (shooting failed)");
                const double h = 1e-5;
                double a1, a2, b1, b2;
                residual(lnmu + h, lnf, a1, a2);
                residual(lnmu, lnf + h, b1, b2);
                const double J11 = (a1 - g1) / h, J21 = (a2 - g2) / h, J12 = (b1 - g1) / h, J22 = (b2 - g2) / h;
                const double det = J11 * J22 - J12 * J21;
                double d1 = -(J22 * g1 - J12 * g2) / det, d2 = -(-J21 * g1 + J11 * g2) / det;
                // damped step
                const double scale = std::min(1.0, 1.0 / std::max(std::fabs(d1), std::fabs(d2)));
                double t = scale, n1 = 0, n2 = 0;
                const double norm = std::hypot(g1, g2);
                for (int ls = 0; ls < 30; ls++, t *= 0.5) {
                    residual(lnmu + t * d1, lnf + t * d2, n1, n2);
                    if (std::isfinite(n1) && std::isfinite(n2) && std::hypot(n1, n2) < norm)
                        break;
                }
                lnmu += t * d1;
                lnf += t * d2;
                g1 = n1;
                g2 = n2;
            }
            mu_ = std::exp(lnmu);
            f_ = std::exp(lnf);
            // full solution and closure: Omega_Lambda = 1 - rho_others(0) - rho_EDE(0)
            solve_to(10.0);
            const double OmegaL_new = 1.0 - rho_o0 - rho(0.0);
            const bool done = std::fabs(OmegaL_new - OmegaL_) < 1e-15;
            OmegaL_ = OmegaL_new;
            if (done)
                break;
        }
        solve_to(10.0);
    }

    void EarlyDarkEnergySpecies::solve_to(double x_end) {
        solve(f_ * theta_i, int(std::lround((x_end - x_ini) / dx)) + 1, x_end, 1e-14, 1e-11);
    }

    void EarlyDarkEnergySpecies::peak(double & f_max, double & x_peak) const {
        const int N = int(std::lround((x_fin - x_ini) / dx));
        int imax = 0;
        double best = -1.0;
        std::vector<double> F(N + 1);
        for (int i = 0; i <= N; i++) {
            F[i] = fraction(std::min(x_ini + i * dx, x_fin));
            if (F[i] > best)
                best = F[i], imax = i;
        }
        if (imax == 0 || imax == N) { // no interior maximum
            f_max = best;
            x_peak = x_ini + imax * dx;
            return;
        }
        // parabola through the three points around the maximum
        const double fm = F[imax - 1], f0 = F[imax], fp = F[imax + 1];
        const double den = fm - 2.0 * f0 + fp;
        const double t = 0.5 * (fm - fp) / den;
        x_peak = x_ini + (imax + t) * dx;
        f_max = f0 - 0.25 * (fm - fp) * t;
    }

    double EarlyDarkEnergySpecies::v(double phi) const { return mu_ * std::pow(1.0 - std::cos(phi / f_), n); }
    double EarlyDarkEnergySpecies::v_phi(double phi) const {
        const double th = phi / f_;
        return mu_ * n * std::pow(1.0 - std::cos(th), n - 1.0) * std::sin(th) / f_;
    }
    double EarlyDarkEnergySpecies::v_phiphi(double phi) const {
        const double th = phi / f_, u = 1.0 - std::cos(th), s = std::sin(th);
        double r = std::pow(u, n - 1.0) * std::cos(th);
        if (n != 1.0)
            r += (n - 1.0) * std::pow(u, n - 2.0) * s * s;
        return mu_ * n * r / (f_ * f_);
    }

    std::string EarlyDarkEnergySpecies::info() const {
        double fm, xp;
        peak(fm, xp);
        std::ostringstream os;
        os << "early dark energy (V = m^2 f^2 (1 - cos(phi/f))^n, n = " << n << ", theta_i = " << theta_i
           << "; f_EDE = " << fm << " at z_c = " << std::exp(-xp) - 1.0 << "; f = " << f_
           << " M_pl, m = " << mass_over_H0() << " H0, Omega_Lambda = " << OmegaL_ << ")";
        return os.str();
    }

} // namespace FML::COSMOLOGY
