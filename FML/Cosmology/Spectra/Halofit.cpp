#include <FML/Cosmology/Spectra/Halofit.h>
#include <FML/Cosmology/Common.h>

#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using namespace Constants;

    namespace {
        // sigma^2 and its first two derivatives with respect to ln R (trapezoid in ln k)
        void sigma2(const std::vector<double> & lnk,
                    const std::vector<double> & D2,
                    double R,
                    double & s,
                    double & s1,
                    double & s2) {
            s = s1 = s2 = 0.0;
            for (size_t i = 0; i < lnk.size(); i++) {
                const double w = (i == 0 || i + 1 == lnk.size() ? 0.5 : 1.0) * (lnk[1] - lnk[0]);
                const double y2 = std::exp(2.0 * lnk[i]) * R * R;
                if (y2 > 200.0)
                    break;
                const double f = w * D2[i] * std::exp(-y2);
                s += f;
                s1 += -2.0 * y2 * f;
                s2 += (-4.0 * y2 + 4.0 * y2 * y2) * f;
            }
        }
    } // namespace

    bool Halofit::nonlinear(const std::vector<double> & lnk,
                            const std::vector<double> & D2L,
                            double Om,
                            double Ode,
                            double w,
                            std::vector<double> & D2NL,
                            double & k_sigma,
                            double & n_eff,
                            double & C) {
        D2NL = D2L;
        double s, s1, s2;
        const double R_min = 1.0 / std::exp(lnk.back()) * 3.0; // the table must resolve the filter
        sigma2(lnk, D2L, R_min, s, s1, s2);
        if (s < 1.0)
            return false; // no nonlinear scale
        const double R = Utils::find_root(
            [&](double lnR) {
                double a, b, c;
                sigma2(lnk, D2L, std::exp(lnR), a, b, c);
                return std::log(a);
            },
            std::log(R_min),
            std::log(100.0),
            1e-10,
            1e-10);
        const double Rs = std::exp(R);
        sigma2(lnk, D2L, Rs, s, s1, s2);
        k_sigma = 1.0 / Rs;
        n_eff = -3.0 - s1 / s;
        C = -(s2 / s - (s1 / s) * (s1 / s));

        const double n = n_eff, n2 = n * n, n3 = n2 * n, n4 = n3 * n, wde = Ode * (1.0 + w);
        const double a =
            std::pow(10.0, 1.5222 + 2.8553 * n + 2.3706 * n2 + 0.9903 * n3 + 0.2250 * n4 - 0.6038 * C + 0.1749 * wde);
        const double b = std::pow(10.0, -0.5642 + 0.5864 * n + 0.5716 * n2 - 1.5474 * C + 0.2279 * wde);
        const double c = std::pow(10.0, 0.3698 + 2.0404 * n + 0.8161 * n2 + 0.5869 * C);
        const double gamma = 0.1971 - 0.0843 * n + 0.8460 * C;
        const double alpha = std::fabs(6.0835 + 1.3373 * n - 0.1959 * n2 - 5.5274 * C);
        const double beta = 2.0379 - 0.7354 * n + 0.3157 * n2 + 1.2490 * n3 + 0.3980 * n4 - 0.1682 * C;
        const double mu = 0.0;
        const double nu = std::pow(10.0, 5.2105 + 3.6902 * n);
        const double f1 = std::pow(Om, -0.0307), f2 = std::pow(Om, -0.0585), f3 = std::pow(Om, 0.0743);
        for (size_t i = 0; i < lnk.size(); i++) {
            const double y = std::exp(lnk[i]) / k_sigma, dl = D2L[i];
            const double Q = dl * std::pow(1.0 + dl, beta) / (1.0 + alpha * dl) * std::exp(-y / 4.0 - y * y / 8.0);
            const double Hp =
                a * std::pow(y, 3.0 * f1) / (1.0 + b * std::pow(y, f2) + std::pow(c * f3 * y, 3.0 - gamma));
            const double H = Hp / (1.0 + mu / y + nu / (y * y));
            D2NL[i] = Q + H;
        }
        return true;
    }

    Halofit::Halofit(const BackgroundCosmology & bg,
                     const std::function<double(double, double)> & P_lin,
                     double z_max,
                     int n_z) {
        // ln k grid [1/Mpc]: 1e-4 .. 1e3, dln k = 0.02 (the sigma integrals need the far tail at high z)
        std::vector<double> lnk;
        for (double l = std::log(1e-4); l <= std::log(1e3) + 1e-9; l += 0.02)
            lnk.push_back(l);
        lnk_min = lnk.front();
        lnk_max = lnk.back();
        x_min = -std::log1p(z_max);
        const auto xs = Utils::linspace(x_min, 0.0, n_z);
        std::vector<std::vector<double>> lr(xs.size(), std::vector<double>(lnk.size(), 0.0));
        for (size_t ix = 0; ix < xs.size(); ix++) {
            const double x = xs[ix];
            std::vector<double> D2(lnk.size());
            for (size_t i = 0; i < lnk.size(); i++) {
                const double k = std::exp(lnk[i]) / Mpc;
                D2[i] = k * k * k * P_lin(x, k) / (2.0 * M_PI * M_PI);
            }
            const double E2 = bg.E2_of_x(x);
            const double Om = (bg.rhoB(x) + bg.rhoCDM(x) + bg.rhoNcdm(x)) / E2;
            double Ode = bg.rhoDE(x) / E2, w = -1.0;
            if (Ode > 0.0)
                w = -1.0 - bg.drhoDEdx(x) / (3.0 * bg.rhoDE(x));
            else
                Ode = std::max(0.0, 1.0 - Om - (bg.rhoGamma(x) + bg.rhoNu(x)) / E2);
            std::vector<double> D2NL;
            double ks, ne, C;
            if (nonlinear(lnk, D2, Om, Ode, w, D2NL, ks, ne, C))
                for (size_t i = 0; i < lnk.size(); i++)
                    lr[ix][i] = std::log(D2NL[i] / D2[i]);
        }
        log_ratio.create(xs, lnk, lr, "halofit ln(P_NL/P_L)");
    }

    double Halofit::ratio(double x, double k) const {
        if (x < x_min)
            return 1.0;
        const double l = std::log(k * Mpc);
        if (l <= lnk_min)
            return 1.0;
        return std::exp(log_ratio(std::min(x, 0.0), std::min(l, lnk_max)));
    }

} // namespace FML::COSMOLOGY
