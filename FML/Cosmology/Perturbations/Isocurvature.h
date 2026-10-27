#ifndef FML_COSMOLOGY_PERTURBATIONS_ISOCURVATURE_H
#define FML_COSMOLOGY_PERTURBATIONS_ISOCURVATURE_H

// =============================================================================
// Isocurvature modes (as CLASS): CDM (cdi), baryon (bi), neutrino density (nid) and neutrino velocity (niv)
// isocurvature, each with its own primordial spectrum and correlation with the adiabatic mode:
//   P_RR(k)  = A_s (k/k_p)^{n_s - 1}                      adiabatic
//   P_II(k)  = f_I^2 A_s (k/k_p)^{n_I - 1}                isocurvature I (entropy S_I; CLASS's f_I, n_I)
//   P_RI(k)  = c_I sqrt(P_RR P_II)                        correlation (CLASS's c_ad_I, n_ad_I = 0)
//   P_IJ     = 0 for two isocurvature modes
// (CosmologyParameters::iso_f, iso_n, iso_c; modes with f_I = 0 are off). The spectra are quadratic in the
// initial conditions, so with the Cholesky factor L of the constant correlation matrix C (C_00 = 1, C_0I = c_I,
// C_IJ = delta_IJ) the matrix P(k) = sum_j u_j(k) u_j(k)^T with u_j = (L_0j, L_Ij f_I (k/k_p)^{(n_I - n_s)/2}) sqrt(P_RR):
// independent components j, each one Perturbations solution with the k-dependent mixture of initial conditions
// u_j (PerturbationOptions::ic_adiabatic, iso_amplitude, iso_tilt) and the adiabatic primordial spectrum;
// PowerSpectrum sums the components (add_component). One component per active mode plus the adiabatic one.
// Requires sum_I c_I^2 < 1 (C positive definite).
// =============================================================================

#include <FML/Cosmology/Background/CosmologyParameters.h>
#include <FML/Cosmology/Perturbations/Perturbations.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY {

namespace Isocurvature {
  extern const char* const mode_names[4];   // cdi, bi, nid, niv
  bool active(const CosmologyParameters& p);
  // Perturbation options of the independent components (the first contains the adiabatic mode)
  std::vector<PerturbationOptions> components(const CosmologyParameters& p, const PerturbationOptions& base);
}

} // namespace FML::COSMOLOGY

#endif
