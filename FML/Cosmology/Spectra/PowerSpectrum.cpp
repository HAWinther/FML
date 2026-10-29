#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Spectra/Halofit.h>
#include <FML/Cosmology/Perturbations/Tensors.h>
#include <FML/Cosmology/Spectra/HypersphericalBessel.h>
#include <FML/Cosmology/Spectra/Lensing.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using namespace Constants;

    PowerSpectrum::PowerSpectrum(const BackgroundCosmology & cosmo_in,
                                 const RecombinationHistory & rec_in,
                                 const Perturbations & pert_in,
                                 PowerSpectrumOptions options_in)
        : cosmo(cosmo_in), rec(rec_in), pert(pert_in), options(options_in), pol(pert_in.has_polarization()) {
        comps.push_back(&pert);
    }

    void PowerSpectrum::add_component(const Perturbations & p) {
        if (p.get_k_array() != pert.get_k_array())
            throw std::invalid_argument("PowerSpectrum::add_component: different k-grid");
        if (p.has_polarization() != pol)
            throw std::invalid_argument("PowerSpectrum::add_component: polarization differs");
        comps.push_back(&p);
    }

    double PowerSpectrum::primordial_power_spectrum(double k) const {
        const auto & p = cosmo.get_params();
        return p.A_s * std::pow(k * Mpc / p.k_pivot, p.n_s - 1.0);
    }

    // =============================================================================
    // Grids
    // =============================================================================

    void PowerSpectrum::make_ell_list() {
        ells = options.ells;
        if (ells.empty()) {
            // Every ell up to 30: the spline of log D_ell between sparser nodes misses the reionization feature
            // of EE (2.5% at ell = 17 with nodes at 10, 12, 15, 20, 25; Planck's low-ell likelihoods use each
            // ell = 2..29)
            for (int l = 2; l <= 30; l++)
                ells.push_back(l);
            for (int l : {35, 40, 50, 60, 70, 80, 90})
                ells.push_back(l);
            // With lensing the unlensed spectra are needed beyond ell_max
            const int l_top = options.ell_max +
                              (options.lensing ? (options.lensing_delta_ell >= 0 ? options.lensing_delta_ell :
                                                                                   std::max(500, options.ell_max / 3)) :
                                                 0);
            if (options.ell_step < 1)
                throw std::invalid_argument("PowerSpectrum: ell_step must be >= 1");
            for (int l = 100; l < l_top; l += options.ell_step)
                ells.push_back(l);
            ells.push_back(l_top);
        }
        std::sort(ells.begin(), ells.end());
        ells.erase(std::unique(ells.begin(), ells.end()), ells.end());
        if (ells.front() < 2)
            throw std::invalid_argument("PowerSpectrum: ells must be >= 2");
    }

    void PowerSpectrum::make_k_grid() {
        const auto & kp = pert.get_k_array();
        const double k_min = options.k_min > 0.0 ? options.k_min : kp.front();
        const double k_max = options.k_max > 0.0 ? options.k_max : kp.back();
        if (k_min < kp.front() || k_max > kp.back())
            throw std::invalid_argument("PowerSpectrum: LOS k-range outside the perturbation k-range");
        // Theta_ell(k) is ~0 for k r < ell (r: transverse distance to last scattering), so the largest ell needs
        // k_max r_* > ell
        const double chi_ls = cosmo.sinK(cosmo.eta0() - cosmo.eta_of_x(rec.x_visibility_peak()));
        if (k_max * chi_ls < 1.05 * ells.back())
            throw std::invalid_argument("PowerSpectrum: the k-range (k_max = " + std::to_string(k_max * Mpc) +
                                        "/Mpc) does not reach ell = " + std::to_string(ells.back()) +
                                        " (needs k_max > " + std::to_string(1.05 * ells.back() / chi_ls * Mpc) +
                                        "/Mpc, better 1.2x that): increase pert_k_max or lower ell_max / "
                                        "lensing_delta_ell");
        const double dk_lin = 2.0 * M_PI / (cosmo.eta0() * options.n_k_per_oscillation);
        k_los.clear();
        for (double k = k_min; k < k_max; k += std::min(dk_lin, k * options.dlnk_max))
            k_los.push_back(k);
        k_los.push_back(k_max);

        // Closed universe: only k = sqrt(K (nu^2 - 1)) with integer nu >= 3 exist. Step through integer
        // nu with an integer stride that follows the spacing above. The stride must be uniform: Theta_ell^2
        // oscillates with a period of only ~2-3 strides in nu, and the trapezoidal rule loses its high
        // accuracy for oscillating integrands on an irregular grid (rounding the k-grid above to the
        // nearest integer nu gives a 4% error in C_ell).
        const double K = cosmo.get_K();
        if (K > 0.0) {
            const double nu_lo = std::ceil(std::sqrt(kp.front() * kp.front() / K + 1.0) - 1e-9);
            const double nu_hi = std::floor(std::sqrt(k_max * k_max / K + 1.0) + 1e-9);
            const double nu_start = std::max(nu_lo, std::round(std::sqrt(k_min * k_min / K + 1.0)));
            k_los.clear();
            double nu = nu_start;
            while (true) {
                const double k = std::sqrt(K * (nu * nu - 1.0));
                k_los.push_back(k);
                if (nu >= nu_hi)
                    break;
                const double dk = std::min(dk_lin, k * options.dlnk_max);
                const double dnu = dk * k / (K * nu); // dnu/dk = k / (K nu)
                nu = std::min(nu_hi, nu + std::max(1.0, std::floor(dnu + 1e-9)));
            }
        }
    }

    void PowerSpectrum::make_x_grid() {
        const double eta0 = cosmo.eta0();
        const double eta_start = cosmo.eta_of_x(options.x_start_los);

        // Inverse of eta(x)
        const auto xb = Utils::linspace(options.x_start_los - 0.1, 0.0, 20000);
        std::vector<double> etab(xb.size());
        for (size_t i = 0; i < xb.size(); i++)
            etab[i] = cosmo.eta_of_x(xb[i]);
        Spline1D x_of_eta(etab, xb, "x_of_eta");

        // Uniform in eta: n points per Bessel oscillation at k_max
        const double k_max = k_los.back();
        const int n_eta =
            std::max(100, static_cast<int>(options.n_eta_per_oscillation * k_max * (eta0 - eta_start) / (2.0 * M_PI)));
        x_los.clear();
        for (double eta : Utils::linspace(eta_start, eta0, n_eta))
            x_los.push_back(x_of_eta(eta));
        x_los.front() = options.x_start_los;
        x_los.back() = 0.0;

        // Fine uniform grid in x around recombination
        for (double x = options.x_recomb_min; x <= options.x_recomb_max; x += options.dx_recomb)
            x_los.push_back(x);

        std::sort(x_los.begin(), x_los.end());
        std::vector<double> unique;
        for (double x : x_los)
            if (unique.empty() || x - unique.back() > 1e-10)
                unique.push_back(x);
        x_los = unique;
    }

    void PowerSpectrum::make_bessel_tables() {
        const double z_max = 1.001 * k_los.back() * (cosmo.eta0() - cosmo.eta_of_x(options.x_start_los));
        const double dz = 2.0 * M_PI / options.bessel_points_per_2pi;
        const size_t nz = static_cast<size_t>(z_max / dz) + 4;
        bessel.assign(ells.size(), UniformSpline1D());
        bessel_z2.assign(pol ? ells.size() : 0, UniformSpline1D());
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t il = 0; il < ells.size(); il++) {
            std::vector<double> j(nz);
            for (size_t i = 0; i < nz; i++)
                j[i] = Utils::j_ell(ells[il], i * dz);
            bessel[il].create(0.0, dz, j);
            if (pol) {
                // j_ell(z)/z^2 -> 1/15 (ell = 2) or 0 (ell > 2) as z -> 0
                std::vector<double> jz(nz);
                jz[0] = ells[il] == 2 ? 1.0 / 15.0 : 0.0;
                for (size_t i = 1; i < nz; i++) {
                    const double z = i * dz;
                    jz[i] = (z < 0.1 && ells[il] == 2) ? (1.0 - z * z / 14.0) / 15.0 : j[i] / (z * z);
                }
                bessel_z2[il].create(0.0, dz, jz);
            }
        }
    }

    // Tables of j_ell(z) and j_ell'(z) = j_{ell-1}(z) - (ell+1)/z j_ell(z) on [0, z_max] for the flat
    // approximation of the hyperspherical Bessel functions (curved universes, large nu)
    void PowerSpectrum::make_bessel_tables_flat_approximation(double z_max) {
        const double dz = 2.0 * M_PI / options.bessel_points_per_2pi;
        const size_t nz = static_cast<size_t>(z_max / dz) + 4;
        bessel.assign(ells.size(), UniformSpline1D());
        bessel_d.assign(ells.size(), UniformSpline1D());
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t il = 0; il < ells.size(); il++) {
            const int l = ells[il];
            std::vector<double> j(nz), jd(nz);
            for (size_t i = 0; i < nz; i++) {
                const double z = i * dz;
                j[i] = Utils::j_ell(l, z);
                jd[i] = z > 0.0 ? Utils::j_ell(l - 1, z) - (l + 1.0) / z * j[i] : (l == 1 ? 1.0 / 3.0 : 0.0);
            }
            bessel[il].create(0.0, dz, j);
            bessel_d[il].create(0.0, dz, jd);
        }
    }

    double PowerSpectrum::lensing_source(double x, double k, double w, double deta) const {
        if (deta <= 0.0)
            return 0.0;
        const double chi_s = cosmo.eta0() - eta_star;
        if (deta >= chi_s)
            return 0.0; // before the source plane
        const double kernel = cosmo.sinK(chi_s - deta) / (cosmo.sinK(chi_s) * cosmo.sinK(deta));
        return -w * c / cosmo.Hp_of_x(x) * kernel * (cur->get_Psi(x, k) - cur->get_Phi(x, k));
    }

    // =============================================================================
    // Line of sight integration
    // =============================================================================

    void PowerSpectrum::line_of_sight_integration() {
        const size_t nx = x_los.size(), nk = k_los.size(), nl = ells.size();
        const double eta0 = cosmo.eta0();

        // Trapezoidal weights in x, and eta0 - eta on the grid
        std::vector<double> w(nx), deta(nx);
        for (size_t i = 0; i < nx; i++) {
            const double left = i > 0 ? x_los[i] - x_los[i - 1] : 0.0;
            const double right = i + 1 < nx ? x_los[i + 1] - x_los[i] : 0.0;
            w[i] = 0.5 * (left + right);
            deta[i] = eta0 - cosmo.eta_of_x(x_los[i]);
        }
        deta.back() = 0.0;

        // Lowest z where j_ell is not negligible (same criterion as Utils::j_ell)
        std::vector<double> z_low(nl);
        for (size_t il = 0; il < nl; il++) {
            const double nu = ells[il] + 0.5;
            z_low[il] = std::max(0.0, nu - 10.0 * std::cbrt(nu));
        }

        // sqrt((ell+2)!/(ell-2)!) for the E-mode transfer function
        std::vector<double> E_factor(nl);
        for (size_t il = 0; il < nl; il++) {
            const double l = ells[il];
            E_factor[il] = std::sqrt((l + 2.0) * (l + 1.0) * l * (l - 1.0));
        }

        Theta.assign(nl, std::vector<double>(nk, 0.0));
        ThetaE.assign(pol ? nl : 0, std::vector<double>(nk, 0.0));
        const bool lens = options.lensing;
        ThetaPhi.assign(lens ? nl : 0, std::vector<double>(nk, 0.0));
#pragma omp parallel for schedule(dynamic, 4)
        for (size_t ik = 0; ik < nk; ik++) {
            const double k = k_los[ik];
            std::vector<double> wS(nx), wSE(nx, 0.0), z(nx), wP(lens ? nx : 0, 0.0);
            for (size_t i = 0; i < nx; i++) {
                z[i] = k * deta[i];
                if (lens)
                    wP[i] = lensing_source(x_los[i], k, w[i], deta[i]);
                wS[i] = w[i] * cur->get_Source_T_term(options.source_term, x_los[i], k);
                // S_E j_ell(z) = (3 g Pi / 4) j_ell(z)/z^2; the j_ell/z^2 factor comes from its own table
                if (pol)
                    wSE[i] = w[i] * cur->get_Source_E_numerator(x_los[i], k);
            }
            for (size_t il = 0; il < nl; il++) {
                const auto & j = bessel[il];
                double sum = 0.0, sumE = 0.0;
                // z decreases along the grid; stop once j_ell is negligible
                if (pol) {
                    const auto & jz2 = bessel_z2[il];
                    for (size_t i = 0; i < nx && z[i] >= z_low[il]; i++) {
                        sum += wS[i] * j(z[i]);
                        sumE += wSE[i] * jz2(z[i]);
                    }
                    ThetaE[il][ik] = E_factor[il] * sumE;
                } else {
                    for (size_t i = 0; i < nx && z[i] >= z_low[il]; i++)
                        sum += wS[i] * j(z[i]);
                }
                Theta[il][ik] = sum;
                if (lens) {
                    double sumP = 0.0;
                    for (size_t i = 0; i < nx && z[i] >= z_low[il]; i++)
                        sumP += wP[i] * j(z[i]);
                    ThetaPhi[il][ik] = sumP;
                }
            }
        }
    }

    // Line of sight integration with the full radial functions: hyperspherical Bessel functions in a
    // curved universe (or j_ell when flat and options.use_radial_functions). See the header for the
    // formulas. For each k and each x the functions Phi_0..Phi_lcap are computed by recurrence, where
    // lcap is just beyond the turning point (Phi is exponentially small above it).
    void PowerSpectrum::line_of_sight_integration_radial() {
        if (options.source_term != Perturbations::SourceTerm::All)
            throw std::invalid_argument("PowerSpectrum: source components are only available with the flat projection");
        const size_t nx = x_los.size(), nk = k_los.size(), nl = ells.size();
        const double eta0 = cosmo.eta0();
        const double K = cosmo.get_K();
        const int sign = K > 0.0 ? 1 : (K < 0.0 ? -1 : 0);
        const int lmax_list = ells.back();

        std::vector<double> w(nx), deta(nx), g(nx);
        for (size_t i = 0; i < nx; i++) {
            const double left = i > 0 ? x_los[i] - x_los[i - 1] : 0.0;
            const double right = i + 1 < nx ? x_los[i + 1] - x_los[i] : 0.0;
            w[i] = 0.5 * (left + right);
            deta[i] = eta0 - cosmo.eta_of_x(x_los[i]);
            g[i] = rec.g_tilde_of_x(x_los[i]);
        }
        deta.back() = 0.0;

        std::vector<double> E_factor(nl);
        for (size_t il = 0; il < nl; il++) {
            const double l = ells[il];
            E_factor[il] = std::sqrt((l + 2.0) * (l + 1.0) * l * (l - 1.0));
        }

        // Flat approximation for large nu: tabulate j_ell, j_ell' up to the largest rescaled argument
        // chi sqrt(l(l+1))/chi_tp <= nu chi * (u / asinh u) with u = sqrt(l(l+1))/nu (open; <= nu chi closed)
        const double nu_flat = options.flat_approximation_nu;
        const double nu_max = sign == 0 ? 0.0 : std::sqrt(k_los.back() * k_los.back() + K) / std::sqrt(std::fabs(K));
        const bool use_flat_approx = sign != 0 && nu_max >= nu_flat;
        if (use_flat_approx) {
            const double u = std::sqrt(lmax_list * (lmax_list + 1.0)) / nu_flat;
            const double ratio = sign < 0 ? u / std::asinh(u) : 1.0;
            const double qmax = std::sqrt(k_los.back() * k_los.back() + K);
            make_bessel_tables_flat_approximation(1.01 * qmax * deta.front() * ratio);
        }
        std::vector<double> z_low(nl); // below this argument j_ell is negligible (as in Utils::j_ell)
        for (size_t il = 0; il < nl; il++)
            z_low[il] = std::max(0.0, ells[il] + 0.5 - 10.0 * std::cbrt(ells[il] + 0.5));

        Theta.assign(nl, std::vector<double>(nk, 0.0));
        ThetaE.assign(pol ? nl : 0, std::vector<double>(nk, 0.0));
        const bool lens = options.lensing;
        ThetaPhi.assign(lens ? nl : 0, std::vector<double>(nk, 0.0));
        std::string error;
        size_t n_flat_approx = 0;
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : n_flat_approx)
        for (size_t ik = 0; ik < nk; ik++) {
            try {
                const double k = k_los[ik];
                // Unit of length lambda: chi = (eta0 - eta)/lambda, nu = q lambda. Flat: lambda = 1/k, nu = 1.
                const double lambda = sign == 0 ? 1.0 / k : 1.0 / std::sqrt(std::fabs(K));
                const double q = std::sqrt(k * k + K);
                const double nu = sign == 0 ? 1.0 : q * lambda;
                const HypersphericalBessel hb(sign, sign == 1 ? std::round(nu) : nu, lmax_list + 1);
                const double kl = k * lambda, s2 = std::sqrt(1.0 - 3.0 * K / (k * k));
                const double nu2s = hb.nu() * hb.nu() - sign;

                std::vector<double> T(nl, 0.0), E(pol ? nl : 0, 0.0), Pp(lens ? nl : 0, 0.0), phi, dphi;

                if (use_flat_approx && hb.nu() >= nu_flat) {
                    // Rescaled flat Bessel functions (CLASS's flat approximation). Per ell: argument scale
                    // ra = sqrt(l(l+1))/chi_tp and amplitude A(chi) = amp [1 + c1 t + c2 t^2], t = atan(l/nu)(chi -
                    // chi_tp), bounded by chi/sin_K(chi) (max for open, min for closed), amp = (1 - sign
                    // l(l+1)/nu^2)^(-1/12).
                    n_flat_approx++;
                    const double nuv = hb.nu();
                    std::vector<double> S0(nx), S1(nx), S2(nx), SEw(nx), chi(nx), sK(nx), SP(lens ? nx : 0, 0.0);
                    for (size_t i = 0; i + 1 < nx; i++) {
                        chi[i] = deta[i] / lambda;
                        const double x = x_los[i];
                        if (lens)
                            SP[i] = lensing_source(x, k, w[i], deta[i]);
                        const double Pi = cur->get_Pi(x, k);
                        S0[i] = w[i] * cur->get_Source_T0(x, k);
                        S1[i] = -w[i] * g[i] * cur->get_v_b(x, k);
                        S2[i] = w[i] * 0.5 * g[i] * Pi;
                        sK[i] = hb.sinK(chi[i]);
                        const double kr = kl * sK[i];
                        SEw[i] = pol ? w[i] * 0.75 * g[i] * Pi / (s2 * kr * kr) : 0.0;
                    }
                    for (size_t il = 0; il < nl; il++) {
                        const double l = ells[il], lp = std::sqrt(l * (l + 1.0));
                        if (sign == 1 && lp >= nuv)
                            continue; // Phi_ell = 0 for ell >= nu (closed)
                        const double chi_tp = sign < 0 ? std::asinh(lp / nuv) : std::asin(lp / nuv);
                        const double ra = lp / chi_tp;
                        const double amp = std::pow(1.0 - sign * l * (l + 1.0) / (nuv * nuv), -1.0 / 12.0);
                        const double at = std::atan(l / nuv);
                        const auto & j = bessel[il];
                        const auto & jd = bessel_d[il];
                        double sum = 0.0, sumE = 0.0, sumP = 0.0;
                        for (size_t i = 0; i + 1 < nx; i++) {
                            const double z = chi[i] * ra;
                            if (z < z_low[il])
                                break; // chi decreases along the grid
                            const double t = at * (chi[i] - chi_tp);
                            const double bound = chi[i] / sK[i];
                            const double A = sign < 0 ? std::max(amp * (1.0 - 0.38 * t + 0.40 * t * t), bound) :
                                                        std::min(amp * (1.0 + 0.34 * t + 2.00 * t * t), bound);
                            const double J = j(z), Jd = jd(z);
                            const double Jdd = -2.0 / z * Jd - (1.0 - l * (l + 1.0) / (z * z)) * J;
                            const double P = A * J, dP = A * ra * Jd, d2P = A * ra * ra * Jdd;
                            sum += S0[i] * P + S1[i] * dP / kl + S2[i] * (3.0 * d2P / (kl * kl) + P) / (2.0 * s2);
                            if (pol)
                                sumE += SEw[i] * P;
                            if (lens)
                                sumP += SP[i] * P;
                        }
                        Theta[il][ik] = sum;
                        if (lens)
                            ThetaPhi[il][ik] = sumP;
                        if (pol)
                            ThetaE[il][ik] = E_factor[il] * sumE;
                    }
                    continue;
                }

                // The x-grid resolves the oscillations of the radial functions at k_max (uniform in eta outside
                // the recombination window). A mode with smaller q needs proportionally fewer points: outside the
                // window keep only every m-th point, m = floor(q_max / q), but never more than dx_max apart in x
                // (the sources themselves vary on that scale, e.g. reionization and the late ISW effect).
                // The trapezoidal weights are recomputed for the kept points.
                const double q_max = std::sqrt(k_los.back() * k_los.back() + K);
                const size_t m = std::max<size_t>(1, static_cast<size_t>(q_max / q));
                const double dx_max = options.dx_max_subsampled;
                std::vector<size_t> idx;
                for (size_t i = 0; i < nx; i++) {
                    const bool in_window = x_los[i] >= options.x_recomb_min && x_los[i] <= options.x_recomb_max;
                    const bool keep = idx.empty() || i + 1 == nx || in_window || i - idx.back() >= m ||
                                      x_los[i + 1] - x_los[idx.back()] > dx_max;
                    if (keep)
                        idx.push_back(i);
                }
                std::vector<double> wm(idx.size());
                for (size_t j = 0; j < idx.size(); j++) {
                    const double left = j > 0 ? x_los[idx[j]] - x_los[idx[j - 1]] : 0.0;
                    const double right = j + 1 < idx.size() ? x_los[idx[j + 1]] - x_los[idx[j]] : 0.0;
                    wm[j] = 0.5 * (left + right);
                }

                for (size_t j = 0; j + 1 < idx.size(); j++) {
                    const size_t i = idx[j];
                    const double chi = deta[i] / lambda;
                    if (chi <= 0.0)
                        continue;
                    const double ltp = hb.turning_point(chi);
                    const int lcap = std::min(lmax_list, int(ltp + 10.0 * std::cbrt(ltp + 1.0) + 10.0));
                    if (lcap < ells.front())
                        continue;
                    const double x = x_los[i];
                    const double S0 = wm[j] * cur->get_Source_T0(x, k);
                    const double SPj = lens ? lensing_source(x, k, wm[j], deta[i]) : 0.0;
                    const double S1 = -wm[j] * g[i] * cur->get_v_b(x, k);
                    const double Pi = cur->get_Pi(x, k);
                    const double S2 = wm[j] * 0.5 * g[i] * Pi;
                    const double sK = hb.sinK(chi), cK = hb.cotK(chi);
                    const double kr = kl * sK; // k times the transverse distance
                    const double SE = pol ? wm[j] * 0.75 * g[i] * Pi / (s2 * kr * kr) : 0.0;
                    hb.column_with_derivative(chi, lcap, phi, dphi);
                    for (size_t il = 0; il < nl && ells[il] <= lcap; il++) {
                        const int l = ells[il];
                        const double P = phi[l], dP = dphi[l];
                        const double d2P = -2.0 * cK * dP - (nu2s - l * (l + 1.0) / (sK * sK)) * P;
                        T[il] += S0 * P + S1 * dP / kl + S2 * (3.0 * d2P / (kl * kl) + P) / (2.0 * s2);
                        if (pol)
                            E[il] += SE * P;
                        if (lens)
                            Pp[il] += SPj * P;
                    }
                }
                for (size_t il = 0; il < nl; il++) {
                    Theta[il][ik] = T[il];
                    if (lens)
                        ThetaPhi[il][ik] = Pp[il];
                    if (pol)
                        ThetaE[il][ik] = E_factor[il] * E[il];
                }
            } catch (const std::exception & e) {
#pragma omp critical
                error = e.what();
            }
        }
        if (!error.empty())
            throw std::runtime_error("PowerSpectrum::line_of_sight_integration_radial: " + error);
        if (sign != 0 && Utils::verbose)
            std::cout << "[PowerSpectrum] curved projection: " << n_flat_approx << " of " << nk
                      << " modes with the flat approximation (nu >= " << nu_flat << ")\n";
    }

    void PowerSpectrum::compute_cells() {
        const size_t nl = ells.size(), nk = k_los.size();
        std::vector<double> P(nk);
        for (size_t ik = 0; ik < nk; ik++)
            P[ik] = primordial_power_spectrum(k_los[ik]);

        // Closed universe: sum over integer nu done as a trapezoid on the nu-nodes plus half of the first
        // term (Euler-Maclaurin), i.e. integrand(k_0) * (dk/dnu)_0 / 2 with dk/dnu = sqrt(K) q/k
        const double K = cosmo.get_K();
        const double first_weight = K > 0.0 ? 0.5 * std::sqrt(K) * std::sqrt(k_los[0] * k_los[0] + K) / k_los[0] : 0.0;
        auto cell1 = [&](const std::vector<double> & X, const std::vector<double> & Y) {
            std::vector<double> integrand(nk);
            for (size_t ik = 0; ik < nk; ik++)
                integrand[ik] = P[ik] * X[ik] * Y[ik] / k_los[ik];
            return 4.0 * M_PI * (Utils::trapz(k_los, integrand) + first_weight * integrand[0]);
        };
        // sum over the independent components; T: 0 = Theta, 1 = ThetaE, 2 = ThetaPhi
        auto cell = [&](int X, int Y, size_t il) {
            double s = 0.0;
            for (size_t c = 0; c < comps.size(); c++) {
                const auto & A = (X == 0 ? Theta_c : X == 1 ? ThetaE_c : ThetaPhi_c)[c][il];
                const auto & B = (Y == 0 ? Theta_c : Y == 1 ? ThetaE_c : ThetaPhi_c)[c][il];
                s += cell1(A, B);
            }
            return s;
        };

        cells.assign(nl, 0.0);
        cells_EE.assign(pol ? nl : 0, 0.0);
        cells_TE.assign(pol ? nl : 0, 0.0);
        std::vector<double> ell_d(nl), log_D(nl), log_DEE(nl), DTE(nl);
        for (size_t il = 0; il < nl; il++) {
            const double l = ells[il], f = l * (l + 1.0) / (2.0 * M_PI);
            ell_d[il] = l;
            cells[il] = cell(0, 0, il);
            if (!(cells[il] > 0.0))
                throw std::runtime_error("PowerSpectrum: C_ell^TT = " + std::to_string(cells[il]) + " at ell = " +
                                         std::to_string(ells[il]) + " (the k-range is too small for this ell?)");
            log_D[il] = std::log(f * cells[il]);
            if (pol) {
                cells_EE[il] = cell(1, 1, il);
                cells_TE[il] = cell(0, 1, il);
                if (!(cells_EE[il] > 0.0))
                    throw std::runtime_error("PowerSpectrum: C_ell^EE = " + std::to_string(cells_EE[il]) +
                                             " at ell = " + std::to_string(ells[il]) +
                                             " (the k-range is too small for this ell?)");
                log_DEE[il] = std::log(f * cells_EE[il]);
                DTE[il] = f * cells_TE[il];
            }
        }
        log_Dell_spline.create(ell_d, log_D, "log_Dell");
        if (pol) {
            log_Dell_EE_spline.create(ell_d, log_DEE, "log_Dell_EE");
            Dell_TE_spline.create(ell_d, DTE, "Dell_TE");
        }
        if (options.lensing) {
            std::vector<double> log_pp(nl);
            for (size_t il = 0; il < nl; il++)
                log_pp[il] =
                    std::log(ells[il] < options.lensing_limber_ell ? cell(2, 2, il) : cell_phiphi_limber(ells[il]));
            log_Cell_pp_spline.create(ell_d, log_pp, "log_Cell_pp");
        }
    }

    // Psi - Phi at (x, k); beyond the perturbation k-range extrapolated as A ln(beta k)/k^2 matched in value
    // and logarithmic slope at k_max (the asymptotic shape of the CDM transfer function)
    double PowerSpectrum::weyl_potential(double x, double k, int comp) const {
        const Perturbations & pc = *comps.at(comp);
        const double kmax = pc.get_k_array().back();
        auto W = [&](double kk) { return pc.get_Psi(x, kk) - pc.get_Phi(x, kk); };
        if (k <= kmax)
            return W(k);
        const double k1 = kmax / 1.1, w0 = W(kmax), w1 = W(k1);
        if (w0 == 0.0 || w0 * w1 <= 0.0)
            return w0 * (kmax / k) * (kmax / k);
        const double n = std::log(w0 / w1) / std::log(kmax / k1); // d ln W / d ln k
        // n = 1/ln(beta k) - 2  ->  ln(beta k_max) = 1/(n + 2); fall back to k^-2 if not in the CDM regime
        const double lb = (n > -2.0 && n < -1.0) ? 1.0 / (n + 2.0) : 0.0;
        const double shape = lb > 0.0 ? (lb + std::log(k / kmax)) / lb : 1.0;
        return w0 * (kmax / k) * (kmax / k) * shape;
    }

    // Limber approximation: C_ell^phiphi = int dchi [r(chi_*-chi)/(r(chi_*) r(chi))]^2 / r(chi)^2
    //                                      * (2 pi^2 / k^3) P_R(k) (Psi - Phi)^2,   k = (ell + 1/2)/r(chi)
    double PowerSpectrum::cell_phiphi_limber(int ell) const {
        const double x_s = rec.x_visibility_peak();
        const double chi_s = cosmo.eta0() - eta_star;
        const int n = 1500;
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            const double x = x_s + (0.0 - x_s) * (i + 0.5) / n; // midpoint rule in x
            const double dx = -x_s / n;
            const double chi = cosmo.eta0() - cosmo.eta_of_x(x);
            if (chi <= 0.0 || chi >= chi_s)
                continue;
            const double r = cosmo.sinK(chi);
            const double kernel = cosmo.sinK(chi_s - chi) / (cosmo.sinK(chi_s) * r);
            const double k = (ell + 0.5) / r;
            double W2 = 0.0;
            for (size_t cc = 0; cc < comps.size(); cc++)
                W2 += std::pow(weyl_potential(x, k, int(cc)), 2);
            if (halofit)
                W2 *= halofit->ratio(x, k);
            const double Pk = 2.0 * M_PI * M_PI / (k * k * k) * primordial_power_spectrum(k) * W2;
            sum += dx * c / cosmo.Hp_of_x(x) * kernel * kernel / (r * r) * Pk;
        }
        return sum;
    }

    // Lensed spectra: unlensed C_ell and C_ell^phiphi at every ell up to the top of the list, then the
    // correlation-function method of Lensing.h up to ell_max
    void PowerSpectrum::compute_lensing() {
        const int L = ells.back();
        ell_max_lensed = options.ell_max;
        if (L <= ell_max_lensed)
            throw std::runtime_error("PowerSpectrum: lensing needs ells beyond ell_max");
        std::vector<double> tt(L + 1, 0.0), ee, te, bb, pp(L + 1, 0.0);
        if (pol) {
            ee.assign(L + 1, 0.0);
            te.assign(L + 1, 0.0);
        }
        if (pol && tensors)
            bb.assign(L + 1, 0.0);
        for (int l = 2; l <= L; l++) {
            const double f = 2.0 * M_PI / (l * (l + 1.0));
            tt[l] = std::exp(log_Dell_spline(l)) * f;
            if (pol) {
                ee[l] = std::exp(log_Dell_EE_spline(l)) * f;
                te[l] = Dell_TE_spline(l) * f;
            }
            if (tensors) { // scalar + tensor, lensed together
                tt[l] += tensors->cell_TT(l);
                if (pol) {
                    ee[l] += tensors->cell_EE(l);
                    te[l] += tensors->cell_TE(l);
                    bb[l] = tensors->cell_BB(l);
                }
            }
            pp[l] = std::exp(log_Cell_pp_spline(l));
        }
        const auto r = Lensing::lens(tt, ee, te, bb, pp, ell_max_lensed);
        lensed_tt = r.tt;
        lensed_ee = r.ee;
        lensed_te = r.te;
        lensed_bb = r.bb;
    }

    void PowerSpectrum::solve() {
        Utils::Timer timer("PowerSpectrum::solve");
        make_ell_list();
        make_k_grid();
        make_x_grid();
        eta_star = cosmo.eta_of_x(rec.x_visibility_peak());
        const bool radial = options.use_radial_functions || cosmo.get_K() != 0.0;
        Utils::Timer tb;
        if (!radial)
            make_bessel_tables();
        const double t_bessel = tb.elapsed();
        Utils::Timer tl;
        Theta_c.clear();
        ThetaE_c.clear();
        ThetaPhi_c.clear();
        for (const Perturbations * p : comps) {
            cur = p;
            if (radial)
                line_of_sight_integration_radial();
            else
                line_of_sight_integration();
            Theta_c.push_back(Theta);
            ThetaE_c.push_back(ThetaE);
            ThetaPhi_c.push_back(ThetaPhi);
        }
        cur = comps.front();
        Theta = Theta_c.front(); // transfer functions of the first component
        ThetaE = ThetaE_c.front();
        ThetaPhi.clear();
        const double t_los = tl.elapsed();
        solve_matter();
        compute_cells();
        Utils::Timer tlens;
        if (options.lensing)
            compute_lensing();
        const double t_lens = tlens.elapsed();
        solved = true;
        if (Utils::verbose)
            std::cout << "[PowerSpectrum] Bessel tables " << t_bessel << " s, LOS integration " << t_los << " s"
                      << (options.lensing ? ", lensing " + std::to_string(t_lens) + " s" : "") << "\n";
        timer.print();
    }

    void PowerSpectrum::solve_matter() {
        if (matter_solved)
            return;
        if (options.nonlinear == PowerSpectrumOptions::Nonlinear::Halofit) {
            halofit = std::make_unique<Halofit>(cosmo, [this](double x, double k) {
                return get_matter_power_spectrum_extrapolated(x, k);
            });
            static std::atomic<bool> warned{false}; // once per process (parameter fits make many spectra)
            if (cosmo.get_params().has_massive_nu() && !warned.exchange(true))
                std::cerr << "Warning: Halofit (Takahashi 2012) has no massive-neutrino correction (Bird et al. 2012): "
                             "with m_nu > 0 the nonlinear P(k) is up to ~2% below CLASS's halofit, which includes it\n";
        }
        matter_solved = true;
    }

    // =============================================================================
    // Getters
    // =============================================================================

    PowerSpectrum::~PowerSpectrum() = default;

    double PowerSpectrum::get_matter_power_spectrum_nonlinear(double x, double k) const {
        const double P = get_matter_power_spectrum_extrapolated(x, k);
        return halofit ? P * halofit->ratio(x, k) : P;
    }

    void PowerSpectrum::require_solved(const char * f) const {
        if (!solved)
            throw std::runtime_error(std::string("PowerSpectrum::") + f + ": call solve() first");
    }

    double PowerSpectrum::tensor_dell(int which, double ell) const {
        if (!tensors)
            return 0.0;
        auto d = [&](int l) {
            switch (which) {
                case 0:
                    return tensors->get_Dell_TT(l);
                case 1:
                    return tensors->get_Dell_EE(l);
                case 2:
                    return tensors->get_Dell_BB(l);
                default:
                    return tensors->get_Dell_TE(l);
            }
        };
        const int l0 = int(std::floor(ell));
        const double f = ell - l0;
        return f == 0.0 ? d(l0) : (1.0 - f) * d(l0) + f * d(l0 + 1);
    }

    double PowerSpectrum::get_Dell_TT_scalar(double ell) const {
        require_solved("get_Dell_TT");
        const double T_muK = cosmo.get_params().TCMB * 1e6;
        return std::exp(log_Dell_spline(ell)) * T_muK * T_muK;
    }
    double PowerSpectrum::get_Dell_TT(double ell) const { return get_Dell_TT_scalar(ell) + tensor_dell(0, ell); }

    double PowerSpectrum::get_Cell_TT(double ell) const {
        require_solved("get_Cell_TT");
        const double T_muK = cosmo.get_params().TCMB * 1e6;
        return get_Dell_TT(ell) / (T_muK * T_muK) * 2.0 * M_PI / (ell * (ell + 1.0));
    }

    double PowerSpectrum::get_Dell_EE_scalar(double ell) const {
        require_solved("get_Dell_EE");
        if (!pol)
            throw std::runtime_error("PowerSpectrum::get_Dell_EE: no polarization");
        const double T_muK = cosmo.get_params().TCMB * 1e6;
        return std::exp(log_Dell_EE_spline(ell)) * T_muK * T_muK;
    }
    double PowerSpectrum::get_Dell_EE(double ell) const { return get_Dell_EE_scalar(ell) + tensor_dell(1, ell); }

    double PowerSpectrum::get_Dell_TE_scalar(double ell) const {
        require_solved("get_Dell_TE");
        if (!pol)
            throw std::runtime_error("PowerSpectrum::get_Dell_TE: no polarization");
        const double T_muK = cosmo.get_params().TCMB * 1e6;
        return Dell_TE_spline(ell) * T_muK * T_muK;
    }
    double PowerSpectrum::get_Dell_TE(double ell) const { return get_Dell_TE_scalar(ell) + tensor_dell(3, ell); }

    double PowerSpectrum::get_Dell_BB(double ell) const {
        require_solved("get_Dell_BB");
        return tensor_dell(2, ell);
    }

    double PowerSpectrum::get_Cell_phiphi(double ell) const {
        require_solved("get_Cell_phiphi");
        if (!options.lensing)
            throw std::runtime_error("PowerSpectrum::get_Cell_phiphi: lensing is off");
        return std::exp(log_Cell_pp_spline(ell));
    }

    namespace {
        void check_lensed(bool lensing, int ell, int lmax, bool pol_needed, bool pol) {
            if (!lensing)
                throw std::runtime_error("PowerSpectrum: lensed spectra require options.lensing = true");
            if (ell < 2 || ell > lmax)
                throw std::out_of_range("PowerSpectrum: lensed ell outside [2, ell_max]");
            if (pol_needed && !pol)
                throw std::runtime_error("PowerSpectrum: no polarization");
        }
    } // namespace

    double PowerSpectrum::get_Dell_TT_lensed(int ell) const {
        require_solved("get_Dell_TT_lensed");
        check_lensed(options.lensing, ell, ell_max_lensed, false, pol);
        const double T = cosmo.get_params().TCMB * 1e6;
        return ell * (ell + 1.0) / (2.0 * M_PI) * lensed_tt[ell] * T * T;
    }
    double PowerSpectrum::get_Dell_EE_lensed(int ell) const {
        require_solved("get_Dell_EE_lensed");
        check_lensed(options.lensing, ell, ell_max_lensed, true, pol);
        const double T = cosmo.get_params().TCMB * 1e6;
        return ell * (ell + 1.0) / (2.0 * M_PI) * lensed_ee[ell] * T * T;
    }
    double PowerSpectrum::get_Dell_TE_lensed(int ell) const {
        require_solved("get_Dell_TE_lensed");
        check_lensed(options.lensing, ell, ell_max_lensed, true, pol);
        const double T = cosmo.get_params().TCMB * 1e6;
        return ell * (ell + 1.0) / (2.0 * M_PI) * lensed_te[ell] * T * T;
    }
    double PowerSpectrum::get_Dell_BB_lensed(int ell) const {
        require_solved("get_Dell_BB_lensed");
        check_lensed(options.lensing, ell, ell_max_lensed, true, pol);
        const double T = cosmo.get_params().TCMB * 1e6;
        return ell * (ell + 1.0) / (2.0 * M_PI) * lensed_bb[ell] * T * T;
    }

    size_t PowerSpectrum::ell_index(int ell) const {
        const auto it = std::find(ells.begin(), ells.end(), ell);
        if (it == ells.end())
            throw std::invalid_argument("PowerSpectrum: ell = " + std::to_string(ell) + " not in the list");
        return it - ells.begin();
    }

    // Linear interpolation on the (fine) LOS k-grid
    double PowerSpectrum::interp_k(const std::vector<double> & f, double k) const {
        const auto ik = std::upper_bound(k_los.begin(), k_los.end(), k) - k_los.begin();
        if (ik == 0)
            return f.front();
        if (static_cast<size_t>(ik) >= k_los.size())
            return f.back();
        const double t = (k - k_los[ik - 1]) / (k_los[ik] - k_los[ik - 1]);
        return (1.0 - t) * f[ik - 1] + t * f[ik];
    }

    double PowerSpectrum::get_Theta_ell(int ell, double k) const {
        require_solved("get_Theta_ell");
        return interp_k(Theta[ell_index(ell)], k);
    }

    double PowerSpectrum::get_ThetaE_ell(int ell, double k) const {
        require_solved("get_ThetaE_ell");
        if (!pol)
            throw std::runtime_error("PowerSpectrum::get_ThetaE_ell: no polarization");
        return interp_k(ThetaE[ell_index(ell)], k);
    }

    // Comoving-gauge (synchronous for CDM) density contrast of the total matter, CDM + baryons + massive
    // neutrinos, as in CLASS's P_m:
    //   Delta_M = sum_i rho_i [delta_i + (d ln rho_i/dx) Hp v_i / (ck)] / sum_i rho_i
    // with d ln rho_i/dx = -3 (1 + w_i) for a non-interacting species (for the CDM it is taken from the
    // background, so that it also holds when the CDM exchanges energy, e.g. in coupled quintessence).
    // (The Poisson equation c^2 (k^2 - 3K) Phi = (3/2) Omega_M Hp^2 Delta_M gives the same up to the radiation
    // terms, which are 2e-3 at k = 1e-4 h/Mpc today.)
    double PowerSpectrum::get_Delta_M(double x, double k, int comp) const {
        const Perturbations & pert = *comps.at(comp);
        const double Hp = cosmo.Hp_of_x(x);
        if (!pert.has_massive_nu())
            return get_Delta_cb(x, k, comp);
        const double f = 3.0 * Hp / (c * k);
        const double rc = cosmo.rhoCDM(x), rb = cosmo.rhoB(x), rn = cosmo.rhoNcdm(x), pn = cosmo.pNcdm(x);
        const double fc = -cosmo.drhoCDMdx(x) / rc * Hp / (c * k);
        const double num = rc * (pert.get_delta_cdm(x, k) - fc * pert.get_v_cdm(x, k)) +
                           rb * (pert.get_delta_b(x, k) - f * pert.get_v_b(x, k)) + rn * pert.get_delta_ncdm(x, k) -
                           f * (rn + pn) * pert.get_v_ncdm(x, k);
        return num / (rc + rb + rn);
    }

    // CDM + baryons only (the "cb" spectrum), comoving gauge
    double PowerSpectrum::get_Delta_cb(double x, double k, int comp) const {
        const Perturbations & pert = *comps.at(comp);
        const double f = 3.0 * cosmo.Hp_of_x(x) / (c * k);
        const double rc = cosmo.rhoCDM(x), rb = cosmo.rhoB(x);
        const double fc = -cosmo.drhoCDMdx(x) / rc * cosmo.Hp_of_x(x) / (c * k);
        return (rc * (pert.get_delta_cdm(x, k) - fc * pert.get_v_cdm(x, k)) +
                rb * (pert.get_delta_b(x, k) - f * pert.get_v_b(x, k))) /
               (rc + rb);
    }

    double PowerSpectrum::get_matter_power_spectrum(double x, double k) const {
        double D2 = 0.0;
        for (size_t cc = 0; cc < comps.size(); cc++)
            D2 += std::pow(get_Delta_M(x, k, int(cc)), 2);
        return 2.0 * M_PI * M_PI / (k * k * k) * primordial_power_spectrum(k) * D2;
    }

    double PowerSpectrum::get_matter_power_spectrum_cb(double x, double k) const {
        double D2 = 0.0;
        for (size_t cc = 0; cc < comps.size(); cc++)
            D2 += std::pow(get_Delta_cb(x, k, int(cc)), 2);
        return 2.0 * M_PI * M_PI / (k * k * k) * primordial_power_spectrum(k) * D2;
    }

    // =============================================================================
    // Output
    // =============================================================================

    void PowerSpectrum::info() const {
        require_solved("info");
        std::cout << "\nPowerSpectrum:\n"
                  << "  ell: " << ells.size() << " values in [" << ells.front() << ", " << ells.back() << "]\n"
                  << "  LOS k-grid: " << k_los.size() << " points in [" << k_los.front() * Mpc << ", "
                  << k_los.back() * Mpc << "] /Mpc\n"
                  << "  LOS x-grid: " << x_los.size() << " points in [" << x_los.front() << ", " << x_los.back()
                  << "]\n";
        // First acoustic peak
        double l_peak = 0.0, D_peak = 0.0;
        for (double l = 100; l <= 400; l += 1.0) {
            const double D = get_Dell_TT(l);
            if (D > D_peak) {
                D_peak = D;
                l_peak = l;
            }
        }
        std::cout << "  First peak: ell = " << l_peak << ", D_ell = " << D_peak << " muK^2\n"
                  << "  D_ell at ell = 10: " << get_Dell_TT(10) << " muK^2\n";
        if (pol) {
            double lE = 0.0, DE = 0.0;
            for (double l = 200; l <= std::min(1500, ells.back()); l += 1.0)
                if (get_Dell_EE(l) > DE) {
                    DE = get_Dell_EE(l);
                    lE = l;
                }
            std::cout << "  EE maximum: ell = " << lE << ", D_ell = " << DE << " muK^2\n";
        }
        if (options.lensing) {
            double lp = 0.0, Dp = 0.0;
            for (int l = 2; l <= ell_max_lensed; l++) {
                const double D = std::pow(l * (l + 1.0), 2) * get_Cell_phiphi(l) / (2.0 * M_PI);
                if (D > Dp) {
                    Dp = D;
                    lp = l;
                }
            }
            std::cout << "  Lensing: [l(l+1)]^2 C_l^phiphi/2pi peaks at ell = " << lp << " (" << Dp << ")\n";
            const int l2 = std::min(2000, ell_max_lensed);
            std::cout << "  Lensed/unlensed TT at ell = " << l2 << ": " << get_Dell_TT_lensed(l2) / get_Dell_TT(l2)
                      << "\n";
            if (pol)
                std::cout << "  Lensed BB at ell = " << std::min(1000, ell_max_lensed) << ": "
                          << get_Dell_BB_lensed(std::min(1000, ell_max_lensed)) << " muK^2\n";
        }
        std::cout << "\n";
    }

    void PowerSpectrum::output_cells(const std::string & filename) const {
        require_solved("output_cells");
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_cells: cannot open " + filename);
        fp << "# ell  D_ell = ell(ell+1)C_ell/2pi TCMB^2 [muK^2] for TT, EE, TE (no lensing; EE = TE = 0 without "
              "polarization)\n";
        fp << std::setprecision(10);
        for (int l = 2; l <= ells.back(); l++)
            fp << l << " " << get_Dell_TT(l) << " " << (pol ? get_Dell_EE(l) : 0.0) << " "
               << (pol ? get_Dell_TE(l) : 0.0) << "\n";
    }

    void PowerSpectrum::output_cells_lensed(const std::string & filename) const {
        require_solved("output_cells_lensed");
        if (!options.lensing)
            throw std::runtime_error("PowerSpectrum::output_cells_lensed: lensing is off");
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_cells_lensed: cannot open " + filename);
        fp << "# ell  lensed D_ell = ell(ell+1)C_ell/2pi TCMB^2 [muK^2] for TT, EE, TE, BB;  [ell(ell+1)]^2 "
              "C_ell^phiphi/2pi\n";
        fp << std::setprecision(10);
        for (int l = 2; l <= ell_max_lensed; l++) {
            const double ll = l * (l + 1.0);
            fp << l << " " << get_Dell_TT_lensed(l) << " " << (pol ? get_Dell_EE_lensed(l) : 0.0) << " "
               << (pol ? get_Dell_TE_lensed(l) : 0.0) << " " << (pol ? get_Dell_BB_lensed(l) : 0.0) << " "
               << ll * ll * get_Cell_phiphi(l) / (2.0 * M_PI) << "\n";
        }
    }

    void PowerSpectrum::output_matter_power(const std::string & filename, double x) const {
        if (!matter_solved)
            throw std::runtime_error("PowerSpectrum::output_matter_power: call solve() or solve_matter() first");
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_matter_power: cannot open " + filename);
        const double h = cosmo.get_params().h;
        const double Mpc_h = Mpc / h;
        fp << "# x = " << x
           << ". k [h/Mpc]  P_m(k) [(Mpc/h)^3] (CDM + baryons + massive neutrinos)  P_cb(k) (CDM + baryons)"
           << (halofit ? "  P_m nonlinear (Halofit)" : "") << "\n";
        fp << std::scientific << std::setprecision(10);
        const auto & kp = pert.get_k_array();
        const double V = Mpc_h * Mpc_h * Mpc_h;
        for (double k : Utils::logspace(kp.front(), kp.back(), 500)) {
            fp << k * Mpc_h << " " << get_matter_power_spectrum(x, k) / V << " "
               << get_matter_power_spectrum_cb(x, k) / V;
            if (halofit)
                fp << " " << get_matter_power_spectrum_nonlinear(x, k) / V;
            fp << "\n";
        }
    }

    void PowerSpectrum::output_transfer(const std::string & filename, const std::vector<int> & ells_out) const {
        require_solved("output_transfer");
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_transfer: cannot open " + filename);
        fp << "# k [1/Mpc]";
        for (int l : ells_out)
            fp << "  Theta_" << l;
        fp << "\n" << std::scientific << std::setprecision(8);
        for (double k : k_los) {
            fp << k * Mpc;
            for (int l : ells_out)
                fp << " " << get_Theta_ell(l, k);
            fp << "\n";
        }
    }

} // namespace FML::COSMOLOGY
