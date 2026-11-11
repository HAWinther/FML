#ifndef FML_HALOMODEL_HALOMODELBACKGROUND_H
#define FML_HALOMODEL_HALOMODELBACKGROUND_H

//=====================================================================================================================
// The halo-model cosmology (HaloModelCosmology in Halomodel.h) from an FML BackgroundCosmology and linear spectra in
// the units of the Einstein-Boltzmann solver. Header only: link the Cosmology/Background objects (Cosmology.mk)
// when using it.
//=====================================================================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/HaloModel/Halomodel.h>
#include <FML/Math/Math.h>
#include <FML/Units/Units.h>

#include <stdexcept>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            struct BackgroundOptions {
                /// Include photons and massless neutrinos in E(x) for the growth and spherical collapse. HMcode
                /// (and HMcode2020 here) leaves them out (the growth then starts as D = a)
                bool include_radiation{false};
                /// mu(x) = G_eff / G for the growth and spherical collapse (empty: 1, i.e. GR)
                Func1 mu{};
            };

            /// The halo-model cosmology from a background. P_m and P_cb take (x = ln a, k [1/m]) and return P [m^3]
            /// (as PowerSpectrum::get_matter_power_spectrum and _cb in the solver); P_cb is needed with massive
            /// neutrinos. Omega_m and f_nu include the massive neutrinos, which count as matter in E(x) and Omega_m(x).
            inline HaloModelCosmology cosmology_from_background(const BackgroundCosmology & bg,
                                                                std::function<double(double, double)> P_m,
                                                                std::function<double(double, double)> P_cb = {},
                                                                BackgroundOptions opt = {}) {
                if (not P_m)
                    throw std::invalid_argument("[cosmology_from_background] P_m is required");
                const double Om = bg.rhoB(0) + bg.rhoCDM(0) + bg.rhoNcdm(0);
                const double fnu = bg.rhoNcdm(0) / Om;
                if (fnu > 0.0 and not P_cb)
                    throw std::invalid_argument("[cosmology_from_background] massive neutrinos: P_cb is required");

                // E(x), Omega_m(x), dln E/dx and w(x) tabulated and splined (the model owns copies), without
                // radiation unless asked for. 4000 points in x = ln a from 1e-7 to 2: interpolation errors < 1e-9
                const bool rad = opt.include_radiation;
                const auto x_array = FML::MATH::linspace(std::log(1e-7), std::log(2.0), 4000);
                DVector logE, OmegaM, dlogE, w;
                for (const double x : x_array) {
                    const double E2 = bg.E2_of_x(x) - (rad ? 0.0 : bg.rhoGamma(x) + bg.rhoNu(x));
                    const double dE2 = bg.dE2dx_of_x(x) - (rad ? 0.0 : bg.drhoGammadx(x) + bg.drhoNudx(x));
                    const double rho_de = bg.rhoDE(x);
                    logE.push_back(0.5 * std::log(E2));
                    OmegaM.push_back((bg.rhoB(x) + bg.rhoCDM(x) + bg.rhoNcdm(x)) / E2);
                    dlogE.push_back(0.5 * dE2 / E2);
                    w.push_back(rho_de > 0.0 ? -1.0 - bg.drhoDEdx(x) / (3.0 * rho_de) : -1.0);
                }
                auto spline = [&](const DVector & y, const std::string & name) {
                    auto s = std::make_shared<Spline>(x_array, y, name);
                    return Func1([s](double x) { return (*s)(x); });
                };
                const Func1 logE_x = spline(logE, "ln E(x)");
                const Func1 E = [logE_x](double x) { return std::exp(logE_x(x)); };
                const Func1 mu = opt.mu ? opt.mu : Func1([](double) { return 1.0; });
                HaloModelCosmology c;
                c.collapse =
                    SphericalCollapseModel(E, spline(OmegaM, "Omega_m(x)"), spline(dlogE, "dlnE/dx(x)"), mu, spline(w, "w(x)"));
                c.Omega_m = Om;
                c.f_nu = fnu;

                // To h/Mpc and (Mpc/h)^3
                const double length = FML::UTILS::SI::Mpc / bg.get_params().h, volume = length * length * length;
                c.P_m = [P_m, length, volume](double x, double k) { return P_m(x, k / length) / volume; };
                if (P_cb)
                    c.P_cb = [P_cb, length, volume](double x, double k) { return P_cb(x, k / length) / volume; };
                return c;
            }

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML

#endif
