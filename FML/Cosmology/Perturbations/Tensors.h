#ifndef FML_COSMOLOGY_PERTURBATIONS_TENSORS_H
#define FML_COSMOLOGY_PERTURBATIONS_TENSORS_H

// =============================================================================
// Tensor modes (gravitational waves) and their CMB spectra TT, EE, BB, TE, for a flat universe.
//
// Variables and equations follow CLASS (Lesgourgues & Tram; the "optimal" hierarchy of Pitrou et al.,
// arXiv:1305.3261, with s_l = 1 in flat space). Conformal time tau, ' = d/dtau, kappa' = a n_e sigma_T:
//   h'' + (2 + alpha_M) Hp h' + (1 + alpha_T) k^2 h = -4 sqrt6 a^2 sum_X (8 pi G/3) rho_X
//                                                    (F_0^X/15 + 2 F_2^X/21 + F_4^X/35) / M*^2
// (X = photons and free-streaming neutrinos; M*^2, alpha_M, alpha_T from Gravity::tensor, GR: 1, 0, 0),
//   photon temperature  F_0' = -k F_1 - kappa' (F_0 + sqrt6 P) + sqrt6 h'
//                       F_l' = k/(2l+1) [l F_{l-1} - (l+1) F_{l+1}] - kappa' F_l
//   photon polarization G_0' = -k G_1 - kappa' (G_0 - sqrt6 P),  G_l' as F_l
//   P = -(F_0/10 + F_2/7 + 3 F_4/70 - 3 G_0/5 + 6 G_2/7 - 3 G_4/70) / sqrt6
//   neutrinos           F_0' = -k F_1 + sqrt6 h',  F_l' = k/(2l+1) [l F_{l-1} - (l+1) F_{l+1}]
// truncated at l_max with F_L' = k F_{L-1} - (L+1) F_L / tau. Massive neutrinos are included as massless
// in the anisotropic stress with energy density 3 p_ncdm (CLASS's tensor_method = massless approximation).
// Tight coupling (kappa' >> k, Hp): P = -h'/(3 kappa'), photon stress -(32/15) a^2 rho_gamma h'/kappa'; at the
// switch F_0 = 4 sqrt6 h'/(3 kappa'), G_0 = -sqrt6 h'/(3 kappa').
// Initial conditions (superhorizon, radiation era): h = 1/sqrt6 [1 - (k tau)^2 / (6 + 8/5 R_fs)], h' from it,
// neutrino F_0 = sqrt6 (h - 1/sqrt6) (R_fs: free-streaming fraction of the radiation). With
// P_t(k) = r A_s (k/k_pivot)^n_t this is the standard normalization of r.
//
// Line of sight (x = ln a, g = visibility function per unit x):
//   Delta_T,l(k) = int dx [-(dh/dx) e^{-tau} + g P] sqrt(3/8 (l+2)!/(l-2)!) j_l(z)/z^2
//   Delta_E,l(k) = int dx sqrt6 g P (1/4) [-2 j_l + (l+1)(l+2) j_l/z^2 - 2 j_{l+1}/z]
//   Delta_B,l(k) = int dx sqrt6 g P (1/2) [(l+2) j_l/z - j_{l+1}]                    (z = k (eta0 - eta))
// (the flat limits of CLASS's tensor radial functions), and C_l^XY = 4 pi int dk/k P_t(k) Delta_X Delta_Y.
//
// Numerics: modes on a k-grid that resolves the late-time oscillations of h in k (spacing <= 2 pi/(eta0 n)
// below k_fine, then <= dk_max); sources stored on an x-grid and splined in (x, k); LOS on a fine k-grid.
// =============================================================================

#include <FML/Cosmology/Background/BackgroundCosmology.h>
#include <FML/Cosmology/Recombination/RecombinationHistory.h>
#include <FML/Cosmology/Common.h>

#include <string>
#include <vector>

namespace FML::COSMOLOGY {

    struct TensorOptions {
        // Mode k-grid [1/m]: log-spaced (n_k_log points per decade) with the spacing capped at
        // 2 pi/(eta0 n_k_per_oscillation) for k < k_fine and at dk_max above
        double k_min = 5e-5 / Constants::Mpc;
        double k_max = 0.15 / Constants::Mpc;
        double n_k_log = 25.0;
        double n_k_per_oscillation = 2.0;
        double k_fine = 0.02 / Constants::Mpc;
        double dk_max = 0.0015 / Constants::Mpc;
        // Stored x-grid
        double x_start = -18.0;
        int n_x_early = 100, n_x_recomb = 600, n_x_late = 400; // [x_start, -8], [-8, -6], [-6, 0]
        int lmax = 8;                                          // photon temperature and polarization hierarchies
        int lmax_nu = 16;                                      // neutrino hierarchy
        bool neutrinos = true;                                 // neutrino anisotropic stress (if Neff > 0)
        // Tight coupling ends when kappa'/Hp < tc_tau_min or k/kappa' > tc_k_over_tau, at the latest at z = tc_z_max
        double tc_tau_min = 100.0, tc_k_over_tau = 0.01, tc_z_max = 1700.0;
        double ode_abserr = 1e-8, ode_relerr = 1e-7; // C_ell converged to ~1e-4
        // Spectra
        int ell_max = 600;                    // tensor C_ell up to here (zero above; for r = 0.1 D_600^BB ~ 2e-4 muK^2)
        double los_n_k_per_oscillation = 3.0; // LOS k-grid: points per 2 pi/eta0 (and dlnk <= los_dlnk_max)
        double los_dlnk_max = 0.03;
        double los_n_eta_per_oscillation = 5.0; // LOS x-grid: uniform in eta at k_max, plus x around recombination
        double los_dx_recomb = 0.003;
    };

    class Tensors {
      public:
        Tensors(const BackgroundCosmology & cosmo,
                const RecombinationHistory & rec,
                TensorOptions options = TensorOptions());

        void solve(); // modes, sources, line of sight, C_ell

        // C_ell (dimensionless, divide by nothing; times T_CMB^2 for muK^2) for integer ell, 0 above ell_max
        double cell_TT(int ell) const { return at(tt, ell); }
        double cell_EE(int ell) const { return at(ee, ell); }
        double cell_BB(int ell) const { return at(bb, ell); }
        double cell_TE(int ell) const { return at(te, ell); }
        double get_Dell_TT(int ell) const; // ell(ell+1) C_ell/2pi T_CMB^2 [muK^2]
        double get_Dell_EE(int ell) const;
        double get_Dell_BB(int ell) const;
        double get_Dell_TE(int ell) const;
        int ell_max() const { return options.ell_max; }
        double primordial_power_spectrum(double k) const; // P_t(k) = r A_s (k/k_pivot)^n_t

        // Modes (for tests and output): h, dh/dx and the sources at (x, k), k [1/m] within the mode grid
        double get_h(double x, double k) const { return h_spline(x, k); }
        double get_dhdx(double x, double k) const { return hx_spline(x, k); }
        double get_source_T(double x, double k) const { return ST_spline(x, k); }
        double get_source_P(double x, double k) const { return SP_spline(x, k); }
        const std::vector<double> & get_k_array() const { return k_array; }
        // Evolve one mode and return (x, h, dh/dx) on the stored x-grid (independent of solve())
        std::vector<std::vector<double>> mode(double k) const;

        void output_cells(const std::string & filename) const; // ell, D_ell TT, EE, BB, TE [muK^2]
        void info() const;

      private:
        const BackgroundCosmology & cosmo;
        const RecombinationHistory & rec;
        TensorOptions options;
        std::vector<double> k_array, x_array;
        Spline2D h_spline, hx_spline, ST_spline, SP_spline;
        std::vector<double> tt, ee, bb, te; // index = ell
        bool solved = false;
        double nu_fraction = 0.0; // free-streaming fraction of the radiation (initial conditions)

        static double at(const std::vector<double> & v, int ell) {
            return ell >= 0 && ell < int(v.size()) ? v[ell] : 0.0;
        }
        void make_grids();
        // Evolves one mode; fills h, h_x, S_T, S_P on x_array
        void evolve(double k,
                    std::vector<double> & h,
                    std::vector<double> & hx,
                    std::vector<double> & ST,
                    std::vector<double> & SP) const;
        void line_of_sight();
    };

} // namespace FML::COSMOLOGY

#endif
