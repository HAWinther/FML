#ifndef FML_HALOMODEL_HMCODE2016_H
#define FML_HALOMODEL_HMCODE2016_H

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>
#include <functional>
#include <vector>

namespace FML::COSMOLOGY::HALOMODEL {

    //=================================================================================================================
    /// HMcode-2016 (Mead et al. 2015, 2016; as CAMB's halofit_version = mead2016) with the interface of HMcode2020:
    /// a table of P_NL/P_L in (x, k) for the Einstein-Boltzmann solver (nonlinear = hmcode2016) and a one-redshift
    /// spectrum. The calculation is HaloModel with Prescription::HMcode2016 (Halomodel.h), which also takes other
    /// expansion histories and mu(a) through a spherical-collapse model.
    /// Backgrounds: GR with smooth dark energy (lcdm, w0wa), as HMcode2020.
    //=================================================================================================================
    class HMcode2016 {
      public:
        using LinearPower = std::function<double(double, double)>;
        struct Options {
            double z_max = 10.0;
            int n_z = 60;
            int n_k = 512;
            double k_min = 1e-5, k_max = 100.0; // output table [h/Mpc]
            int accuracy_boost = 1;
        };

        // Both callbacks take (x=ln a, k[1/m]) and return P[k] in m^3, including extrapolation.
        // P_cb is the CDM+baryon spectrum relative to the mean CDM+baryon density, not total matter.
        HMcode2016(const BackgroundCosmology & bg, LinearPower P_m, LinearPower P_cb);
        HMcode2016(const BackgroundCosmology & bg, LinearPower P_m, LinearPower P_cb, Options options);
        double ratio(double x, double k) const; // P_NL,total / P_L,total; SI k

        // One redshift. k in h/Mpc, outputs in (Mpc/h)^3.
        struct Spectrum {
            std::vector<double> linear, one_halo, two_halo, nonlinear;
            double delta_c{}, Delta_v{}, sigma8_cb{}, n_eff{};
        };
        static Spectrum spectrum(const BackgroundCosmology & bg,
                                 const LinearPower & P_m,
                                 const LinearPower & P_cb,
                                 double x,
                                 const std::vector<double> & k);

      private:
        double x_min{}, lnk_min{}, lnk_max{}, h{};
        Spline2D log_ratio;
    };

} // namespace FML::COSMOLOGY::HALOMODEL
#endif
