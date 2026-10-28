#ifndef FML_SPECIES_IC_H
#define FML_SPECIES_IC_H
#include <FML/InitialConditions/TransferProvider.h>
#include <FML/Spline/Spline.h>
#include <FML/Global/Global.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
namespace FML::INITIALCONDITIONS {
    //=============================================================================
    /// The ratios T_i / T_cb and the growth rates f_i of CDM and baryons at zini for the initial conditions with
    /// separate CDM and baryon particles: delta_i(k) = delta_cb(k) R_i(k,zini) with R_i = T_i / T_cb from the
    /// transfer functions (w_c R_c + w_b R_b = 1). The 1LPT velocity uses f_i = f_cb + dlog(R_i)/dlna where f_cb is
    /// the growth rate of the gravity model (so the total CDM+baryon velocity is exactly the growing mode, as for one
    /// species, and the relative velocity comes from the transfer functions). Splines of log(k) (k in h/Mpc).
    //=============================================================================
    struct TwoSpeciesICSplines {
        FML::INTERPOLATION::SPLINE::Spline ratio_cdm, ratio_baryon, f_cdm, f_baryon;
    };
    inline TwoSpeciesICSplines two_species_ic_splines(const TransferProvider & provider,
                                                      const std::function<double(double, double)> & growth_rate_cb,
                                                      double zini,
                                                      double weight_cdm,
                                                      double weight_baryon) {
        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        provider.require_ic_density_gauge();
        const double aini = 1.0 / (1.0 + zini);
        if (aini < provider.range().amin * (1.0 - 1e-10))
            throw std::runtime_error("particle_baryons: the transfer functions only go up to z = " +
                                     std::to_string(1.0 / provider.range().amin - 1.0) + " (need zini)");

        const double kmin = provider.range().kmin_hmpc;
        const double kmax = provider.range().kmax_hmpc;
        const int nk = 512;
        DVector logk(nk), ratio_cdm(nk), ratio_baryon(nk), f_cdm(nk), f_baryon(nk);
        double max_err = 0.0;
        // dlog(R_i)/dlna by finite differences over +-dx in log(a) (one-sided at the end of the table)
        const double dx = 0.05;
        const double xini = std::log(aini);
        const double xmin_table = std::log(provider.range().amin);
        const double xlo = std::max(xini - dx, xmin_table);
        const double xhi = provider.range_policy() == RangePolicy::Reject ?
                               std::min(xini + dx, std::log(provider.range().amax)) :
                               xini + dx;
        if (!(xhi > xlo))
            throw std::out_of_range("[TwoSpeciesIC] No time interval for the ratio derivative");
        auto ratios = [&](double k, double a) {
            const double Tcb = provider.density_over_k_mpc2(Species::CDMBaryon, k, a);
            return std::pair<double, double>(provider.density_over_k_mpc2(Species::CDM, k, a) / Tcb,
                                             provider.density_over_k_mpc2(Species::Baryon, k, a) / Tcb);
        };
        for (int i = 0; i < nk; i++) {
            logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
            const double k = std::exp(logk[i]);
            const auto r = ratios(k, aini), rlo = ratios(k, std::exp(xlo)), rhi = ratios(k, std::exp(xhi));
            ratio_cdm[i] = r.first;
            ratio_baryon[i] = r.second;
            const double fcb = growth_rate_cb(k, aini);
            f_cdm[i] = fcb + std::log(rhi.first / rlo.first) / (xhi - xlo);
            f_baryon[i] = fcb + std::log(rhi.second / rlo.second) / (xhi - xlo);
            max_err = std::max(max_err, std::fabs(weight_cdm * ratio_cdm[i] + weight_baryon * ratio_baryon[i] - 1.0));
        }
        TwoSpeciesICSplines s;
        s.ratio_cdm.create(logk, ratio_cdm, "T_cdm/T_cb(k,zini)");
        s.ratio_baryon.create(logk, ratio_baryon, "T_b/T_cb(k,zini)");
        s.f_cdm.create(logk, f_cdm, "f_cdm(k,zini)");
        s.f_baryon.create(logk, f_baryon, "f_b(k,zini)");
        if (FML::ThisTask == 0) {
            std::cout << "# Two species IC at z = " << zini << ": T_b/T_c = ";
            for (double k : {0.01, 0.1, 1.0}) {
                if (provider.range_policy() == RangePolicy::Reject && (k < kmin || k > kmax))
                    continue;
                std::cout << s.ratio_baryon(std::log(k)) / s.ratio_cdm(std::log(k)) << " (k = " << k << ") ";
            }
            std::cout << "\n# f_cdm, f_b at k = 0.1: " << s.f_cdm(std::log(0.1)) << " " << s.f_baryon(std::log(0.1))
                      << ". Max |w_c T_c/T_cb + w_b T_b/T_cb - 1| = " << max_err << "\n";
        }
        if (max_err > 1e-2)
            throw std::runtime_error("particle_baryons: OmegaCDM and Omegab do not match the transfer functions "
                                     "(w_c T_c + w_b T_b != T_cb)");
        return s;
    }

    //=============================================================================
    /// The mass contrast of particle_baryon_relative_mode = mass: eps_i(k) = (T_i - T_cb)(k,z=0) / T_cb(k,zini), the
    /// linear baryon-CDM relative mode today per unit delta_cb(zini) (Hahn, Rampf & Uhlemann 2021). Splines of log(k)
    /// (k in h/Mpc) for CDM and baryons (w_c eps_c + w_b eps_b = 0)
    //=============================================================================
    inline std::pair<FML::INTERPOLATION::SPLINE::Spline, FML::INTERPOLATION::SPLINE::Spline>
    two_species_mass_contrast_today(const TransferProvider & provider, double zini) {
        using DVector = FML::INTERPOLATION::SPLINE::DVector;
        provider.require_ic_density_gauge();
        const double aini = 1.0 / (1.0 + zini);
        const double kmin = provider.range().kmin_hmpc;
        const double kmax = provider.range().kmax_hmpc;
        const int nk = 512;
        DVector logk(nk), eps_cdm(nk), eps_baryon(nk);
        for (int i = 0; i < nk; i++) {
            logk[i] = std::log(kmin) + std::log(kmax / kmin) * i / double(nk - 1);
            const double k = std::exp(logk[i]);
            const double Tcb_ini = provider.density_over_k_mpc2(Species::CDMBaryon, k, aini);
            const double Tcb_today = provider.density_over_k_mpc2(Species::CDMBaryon, k, 1.0);
            eps_cdm[i] = (provider.density_over_k_mpc2(Species::CDM, k, 1.0) - Tcb_today) / Tcb_ini;
            eps_baryon[i] = (provider.density_over_k_mpc2(Species::Baryon, k, 1.0) - Tcb_today) / Tcb_ini;
        }
        FML::INTERPOLATION::SPLINE::Spline sc(logk, eps_cdm, "(T_cdm-T_cb)(k,z=0)/T_cb(k,zini)");
        FML::INTERPOLATION::SPLINE::Spline sb(logk, eps_baryon, "(T_b-T_cb)(k,z=0)/T_cb(k,zini)");
        return {sc, sb};
    }

} // namespace FML::INITIALCONDITIONS
#endif
