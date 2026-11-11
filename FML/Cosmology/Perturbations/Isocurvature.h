#ifndef FML_COSMOLOGY_PERTURBATIONS_ISOCURVATURE_H
#define FML_COSMOLOGY_PERTURBATIONS_ISOCURVATURE_H

// =============================================================================
// Isocurvature modes (as CLASS): CDM (cdi), baryon (bi), neutrino density (nid) and neutrino velocity (niv)
// isocurvature, and the density isocurvature of an ultralight axion field (axi, model axion), each with its own
// primordial spectrum and correlation with the adiabatic mode:
//   P_RR(k)  = A_s (k/k_p)^{n_s - 1}                      adiabatic
//   P_II(k)  = f_I^2 A_s (k/k_p)^{n_I - 1}                isocurvature I (entropy S_I; CLASS's f_I, n_I)
//   P_RI(k)  = c_I sqrt(P_RR P_II)                        correlation (CLASS's c_ad_I, n_ad_I = 0)
//   P_IJ     = 0 for two isocurvature modes
// (CosmologyParameters::iso_f, iso_n, iso_c; modes with f_I = 0 are off). The spectra are quadratic in the
// initial conditions, so with the Cholesky factor L of the constant correlation matrix C (C_00 = 1, C_0I = c_I,
// C_IJ = delta_IJ) the matrix P(k) = sum_j u_j(k) u_j(k)^T with u_j = (L_0j, L_Ij f_I (k/k_p)^{(n_I - n_s)/2})
// sqrt(P_RR): independent components j, each one Perturbations solution with the k-dependent mixture of initial
// conditions u_j (PerturbationOptions::ic_adiabatic, iso_amplitude, iso_tilt) and the adiabatic primordial spectrum;
// PowerSpectrum sums the components (add_component). One component per active mode plus the adiabatic one.
// Requires sum_I c_I^2 < 1 (C positive definite).
//
// Axion isocurvature (axionCAMB's initial_condition 6): the frozen field's fluctuation from inflation, dphi = H_inf /
// (2 pi), gives the entropy S = delta rho_ax / rho_ax = 2 dphi / phi_i with
//   P_SS = H_inf^2 / (pi^2 phi_i^2) (k/k_pivot)^(n - 1),   n = 1 - r / 8,   r = 2 H_inf^2 / (pi^2 A_s M_pl^2)
// (phi_i, M_pl the reduced Planck mass), uncorrelated with the adiabatic mode. with_axion_isocurvature(p, bg, log10
// H_inf [GeV]) sets f_axi = H_inf / (pi phi_i sqrt(A_s)), n_axi and c_ad_axi = 0 from the axion of the background.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    namespace Isocurvature {
        extern const char * const mode_names[CosmologyParameters::n_iso_modes]; // cdi, bi, nid, niv, axi
        bool active(const CosmologyParameters & p);
        // Perturbation options of the independent components (the first contains the adiabatic mode)
        std::vector<PerturbationOptions> components(const CosmologyParameters & p, const PerturbationOptions & base);
        // The parameters with the axion isocurvature of an inflationary Hubble rate 10^log10_Hinf_GeV GeV (above);
        // throws if the background has no ultralight axion or f_axi is set already
        CosmologyParameters with_axion_isocurvature(const CosmologyParameters & p,
                                                    const BackgroundCosmology & bg,
                                                    double log10_Hinf_GeV);
    } // namespace Isocurvature

} // namespace FML::COSMOLOGY

#endif
