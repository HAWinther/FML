#ifndef FML_COSMOLOGY_SPECTRA_HMCODE_H
#define FML_COSMOLOGY_SPECTRA_HMCODE_H

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Common.h>
#include <functional>
#include <vector>

namespace FML::COSMOLOGY {

    // Gravity-only HMcode-2020 (Mead et al., arXiv:2009.01858), including massive neutrinos.
    // This is separate from HALOMODEL::HaloModel's existing HMcode-2016 implementation.
    // Background: GR with smooth dark energy and conserved, pressureless late-time matter.
    // Baryonic feedback and modified-gravity reactions are not part of this prescription.
    class HMcode {
      public:
        using LinearPower = std::function<double(double, double)>;
        struct Options {
            double z_max = 10.0;
            int n_z = 60;
            int n_mass = 256;
            int n_k = 512;
            double k_min = 1e-5, k_max = 100.0; // output table [h/Mpc]
            double dlnk_integral = 0.015;
        };

        // Both callbacks take (x=ln a, k[1/m]) and return P[k] in m^3, including extrapolation.
        // P_cb is the CDM+baryon spectrum relative to the mean CDM+baryon density, not total matter.
        HMcode(const BackgroundCosmology & bg, LinearPower P_m, LinearPower P_cb);
        HMcode(const BackgroundCosmology & bg, LinearPower P_m, LinearPower P_cb, Options options);
        double ratio(double x, double k) const; // P_NL,total / P_L,total; SI k

        // Evaluate one redshift without constructing a time table. k in h/Mpc, outputs in (Mpc/h)^3.
        // Useful for external linear spectra, calibration checks and halo-term diagnostics.
        struct Spectrum {
            std::vector<double> linear, one_halo, two_halo, nonlinear;
            double delta_c{}, Delta_v{}, sigma8_cb{}, n_eff{};
        };
        static Spectrum spectrum(const BackgroundCosmology & bg,
                                 const LinearPower & P_m,
                                 const LinearPower & P_cb,
                                 double x,
                                 const std::vector<double> & k,
                                 int n_mass = 256,
                                 double dlnk_integral = 0.015);

      private:
        double x_min{}, lnk_min{}, lnk_max{}, h{};
        Spline2D log_ratio;
    };

} // namespace FML::COSMOLOGY
#endif
