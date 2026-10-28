// =============================================================================
// PowerSpectrum: two-point functions (CMB angular correlation, matter xi(r) and sigma_R, angular power
// spectrum of the projected matter density and its correlation function). See PowerSpectrum.h.
// =============================================================================

#include <FML/Cosmology/Spectra/PowerSpectrum.h>
#include <FML/Cosmology/Common.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace FML::COSMOLOGY {

    using namespace Constants;

    namespace {
        // Eisenstein & Hu (1998) no-wiggle transfer function (eqs. 29-31), k in 1/Mpc. Used only for the shape
        // of Delta_M beyond the computed k-range (accurate to ~1.5% up to k = 20/Mpc when anchored at 0.3/Mpc).
        double transfer_EH_nowiggle(const CosmologyParameters & p, double k) {
            const double h = p.h, om = p.OmegaM() * h * h, ob = p.OmegaB * h * h, fb = ob / om, Th = p.TCMB / 2.7;
            const double s = 44.5 * std::log(9.83 / om) / std::sqrt(1.0 + 10.0 * std::pow(ob, 0.75));
            const double aG = 1.0 - 0.328 * std::log(431.0 * om) * fb + 0.38 * std::log(22.3 * om) * fb * fb;
            const double G = p.OmegaM() * h * (aG + (1.0 - aG) / (1.0 + std::pow(0.43 * k * s, 4)));
            const double q = k / h * Th * Th / G;
            const double L = std::log(2.0 * M_E + 1.8 * q), C = 14.2 + 731.0 / (1.0 + 62.5 * q);
            return L / (L + C * q * q);
        }
    } // namespace

    double PowerSpectrum::get_Delta_M_extrapolated(double x, double k, int comp) const {
        const auto & kp = pert.get_k_array();
        const double kmin = kp.front(), kmax = kp.back();
        if (k < kmin)
            return get_Delta_M(x, kmin, comp) * (k / kmin) * (k / kmin); // superhorizon: Delta ~ k^2
        if (k <= kmax)
            return get_Delta_M(x, k, comp);
        // Delta_M ~ k^2 T(k), with the shape of T from Eisenstein & Hu, anchored at k_max
        const auto & p = cosmo.get_params();
        return get_Delta_M(x, kmax, comp) * (k / kmax) * (k / kmax) * transfer_EH_nowiggle(p, k * Mpc) /
               transfer_EH_nowiggle(p, kmax * Mpc);
    }

    double PowerSpectrum::get_matter_power_spectrum_extrapolated(double x, double k) const {
        double D2 = 0.0;
        for (size_t cc = 0; cc < comps.size(); cc++)
            D2 += std::pow(get_Delta_M_extrapolated(x, k, int(cc)), 2);
        return 2.0 * M_PI * M_PI / (k * k * k) * primordial_power_spectrum(k) * D2;
    }

    namespace {
        // P(k) tabulated for the 3D integrals, 1e-5 to 100 /Mpc: dln k = 0.002, and dk <= 5e-4/Mpc below 1/Mpc so
        // that the linear interpolation in Filon's method resolves the BAO wiggles (period 2 pi/r_d ~ 0.04/Mpc)
        // to ~5e-4 (~8000 points)
        void matter_table(const PowerSpectrum & ps, double x, std::vector<double> & k, std::vector<double> & P) {
            k.clear();
            for (double kk = 1e-5 / Mpc; kk < 100.0 / Mpc;
                 kk += (kk < 1.0 / Mpc ? std::min(0.002 * kk, 5e-4 / Mpc) : 0.002 * kk))
                k.push_back(kk);
            k.push_back(100.0 / Mpc);
            P.resize(k.size());
            for (size_t i = 0; i < k.size(); i++)
                P[i] = ps.get_matter_power_spectrum_extrapolated(x, k[i]);
        }
    } // namespace

    std::vector<double> PowerSpectrum::get_xi_matter(double x, const std::vector<double> & r, double R_smooth) const {
        std::vector<double> k, P;
        matter_table(*this, x, k, P);
        return TwoPoint::xi_of_r(k, P, r, R_smooth);
    }

    double PowerSpectrum::get_sigma_R(double x, double R) const {
        std::vector<double> k, P;
        matter_table(*this, x, k, P);
        return TwoPoint::sigma_R(k, P, R);
    }

    double PowerSpectrum::get_sigma8(double x) const { return get_sigma_R(x, 8.0 * Mpc / cosmo.get_params().h); }

    // =============================================================================
    // Projected matter density
    // =============================================================================

    std::vector<double> PowerSpectrum::get_angular_cl_matter(const TwoPoint::RedshiftWindow & win,
                                                             const std::vector<int> & ell_list,
                                                             Projection method,
                                                             int ell_limber) const {
        if (!win.n || !(win.z_max > win.z_min) || win.z_min < 0.0)
            throw std::invalid_argument("PowerSpectrum::get_angular_cl_matter: invalid redshift window");
        const bool flat = cosmo.get_K() == 0.0;
        if (method == Projection::Exact && !flat)
            throw std::invalid_argument(
                "PowerSpectrum::get_angular_cl_matter: the exact projection needs a flat universe");
        const double eta0 = cosmo.eta0();
        auto chi_of_z = [&](double z) { return eta0 - cosmo.eta_of_x(-std::log1p(z)); };

        // z(chi) on the support of the window
        const auto zs = Utils::linspace(win.z_min, win.z_max, 2000);
        std::vector<double> chis(zs.size());
        for (size_t i = 0; i < zs.size(); i++)
            chis[i] = chi_of_z(zs[i]);
        Spline1D z_of_chi(chis, zs);
        const double chi_min = chis.front(), chi_max = chis.back();

        // Which ell_list are done exactly
        std::vector<int> l_exact;
        for (int l : ell_list)
            if (method == Projection::Exact || (method == Projection::Auto && flat && l < ell_limber))
                l_exact.push_back(l);
        const int L_ex = l_exact.empty() ? 0 : *std::max_element(l_exact.begin(), l_exact.end());

        // Kernel W(chi) = n(z) H(z)/c on a chi-grid, normalised to int W dchi = 1. Moments for the k-range.
        auto make_kernel =
            [&](int n_chi, std::vector<double> & chi, std::vector<double> & W, std::vector<double> & xg) {
                chi = Utils::linspace(chi_min, chi_max, n_chi);
                W.resize(n_chi);
                xg.resize(n_chi);
                for (int i = 0; i < n_chi; i++) {
                    const double z = z_of_chi(chi[i]);
                    xg[i] = -std::log1p(z);
                    W[i] = std::max(win.n(z), 0.0) * cosmo.H_of_x(xg[i]) / c;
                }
                const double norm = Utils::trapz(chi, W);
                if (!(norm > 0.0))
                    throw std::runtime_error("PowerSpectrum::get_angular_cl_matter: window has no weight");
                for (double & w : W)
                    w /= norm;
            };
        std::vector<double> chi, W, xg;
        make_kernel(1000, chi, W, xg);
        double m1 = 0.0, m2 = 0.0;
        for (size_t i = 0; i + 1 < chi.size(); i++) {
            const double d = chi[i + 1] - chi[i];
            m1 += 0.5 * d * (W[i] * chi[i] + W[i + 1] * chi[i + 1]);
            m2 += 0.5 * d * (W[i] * chi[i] * chi[i] + W[i + 1] * chi[i + 1] * chi[i + 1]);
        }
        const double sig_chi = std::sqrt(std::max(m2 - m1 * m1, 0.0));
        const double b2 = win.bias * win.bias;

        std::vector<double> result(ell_list.size(), 0.0);

        // Exact: Delta_ell(k) = int dchi W b Delta_M(k, chi) j_ell(k chi); C_ell = 4 pi int dk/k P_R Delta_ell^2
        if (!l_exact.empty()) {
            const double chi_lo = std::max(chi_min, m1 - 3.0 * sig_chi);
            const double k_hi = std::min(1.5 * (L_ex + 1.0) / chi_lo + 8.0 / std::max(sig_chi, 1.0 * Mpc), 10.0 / Mpc);
            const double k_lo = std::max(1e-5 / Mpc, 0.05 / chi_max);
            std::vector<double> kg;
            for (double k = k_lo; k < k_hi; k += std::min(0.02 * k, 2.0 * M_PI / (10.0 * chi_max)))
                kg.push_back(k);
            kg.push_back(k_hi);
            // chi-grid resolving j_ell(k chi) at k_hi (10 points per oscillation) and the window
            const int n_chi = std::max(400, static_cast<int>((chi_max - chi_min) * k_hi * 10.0 / (2.0 * M_PI)) + 1);
            std::vector<double> chE, WE, xE;
            make_kernel(n_chi, chE, WE, xE);
            std::vector<double> wt(n_chi); // trapezoid weights times W b
            for (int i = 0; i < n_chi; i++) {
                const double d = (i > 0 ? chE[i] - chE[i - 1] : 0.0) + (i + 1 < n_chi ? chE[i + 1] - chE[i] : 0.0);
                wt[i] = 0.5 * d * WE[i] * win.bias;
            }
            const size_t nk = kg.size(), nl = l_exact.size(), nc = comps.size();
            std::vector<std::vector<double>> D2(nl, std::vector<double>(nk, 0.0)); // sum over components of Delta_ell^2
#pragma omp parallel for schedule(dynamic, 4)
            for (size_t ik = 0; ik < nk; ik++) {
                const double k = kg[ik];
                std::vector<double> jl(L_ex + 1);
                std::vector<std::vector<double>> acc(nc, std::vector<double>(L_ex + 1, 0.0));
                for (int i = 0; i < n_chi; i++) {
                    if (WE[i] <= 0.0)
                        continue;
                    Utils::j_ell_array(L_ex, k * chE[i], jl.data());
                    for (size_t cc = 0; cc < nc; cc++) {
                        const double f = wt[i] * get_Delta_M_extrapolated(xE[i], k, int(cc));
                        for (int l = 0; l <= L_ex; l++)
                            acc[cc][l] += f * jl[l];
                    }
                }
                for (size_t il = 0; il < nl; il++)
                    for (size_t cc = 0; cc < nc; cc++)
                        D2[il][ik] += acc[cc][l_exact[il]] * acc[cc][l_exact[il]];
            }
            for (size_t il = 0; il < nl; il++) {
                double s = 0.0;
                for (size_t ik = 0; ik + 1 < nk; ik++) {
                    auto g = [&](size_t j) { return primordial_power_spectrum(kg[j]) * D2[il][j]; };
                    s += 0.5 * std::log(kg[ik + 1] / kg[ik]) * (g(ik) + g(ik + 1));
                }
                for (size_t i = 0; i < ell_list.size(); i++)
                    if (ell_list[i] == l_exact[il])
                        result[i] = 4.0 * M_PI * s;
            }
        }

        // Limber: C_ell = int dchi W^2 b^2 / r^2 P((ell + 1/2)/r, chi)
        for (size_t i = 0; i < ell_list.size(); i++) {
            if (std::find(l_exact.begin(), l_exact.end(), ell_list[i]) != l_exact.end())
                continue;
            double s = 0.0;
            std::vector<double> g(chi.size());
            for (size_t j = 0; j < chi.size(); j++) {
                const double r = cosmo.sinK(chi[j]);
                g[j] = (W[j] > 0.0 && r > 0.0) ?
                           W[j] * W[j] * b2 / (r * r) *
                               get_matter_power_spectrum_extrapolated(xg[j], (ell_list[i] + 0.5) / r) :
                           0.0;
            }
            s = Utils::trapz(chi, g);
            result[i] = s;
        }
        return result;
    }

    std::vector<double> PowerSpectrum::get_angular_correlation_matter(const TwoPoint::RedshiftWindow & window,
                                                                      const std::vector<double> & theta,
                                                                      int ell_max) const {
        const int L1 = std::min(ell_max, 2000);
        std::vector<int> ell_list;
        for (int l = 0; l <= L1; l++)
            ell_list.push_back(l);
        std::vector<int> lhigh;
        if (ell_max > L1) {
            for (double l = L1 + 1; l < ell_max; l *= 1.03)
                lhigh.push_back(static_cast<int>(l));
            lhigh.push_back(ell_max);
            lhigh.erase(std::unique(lhigh.begin(), lhigh.end()), lhigh.end());
            ell_list.insert(ell_list.end(), lhigh.begin(), lhigh.end());
        }
        const auto c = get_angular_cl_matter(window, ell_list);
        std::vector<double> cl(ell_max + 1, 0.0);
        for (int l = 0; l <= L1; l++)
            cl[l] = c[l];
        if (!lhigh.empty()) { // interpolate log C_ell in log ell between L1 and the high nodes
            std::vector<double> lx = {std::log(double(L1))}, ly = {std::log(c[L1])};
            for (size_t i = 0; i < lhigh.size(); i++) {
                lx.push_back(std::log(double(lhigh[i])));
                ly.push_back(std::log(c[L1 + 1 + i]));
            }
            Spline1D s(lx, ly);
            for (int l = L1 + 1; l <= ell_max; l++)
                cl[l] = std::exp(s(std::log(double(l))));
        }
        const int l_taper = static_cast<int>(0.8 * ell_max); // cosine taper over the top 20%
        for (int l = l_taper; l <= ell_max; l++) {
            const double t = double(l - l_taper) / std::max(ell_max - l_taper, 1);
            cl[l] *= 0.5 * (1.0 + std::cos(M_PI * t));
        }
        return TwoPoint::angular_correlation(cl, theta, 0.0, 2);
    }

    // =============================================================================
    // CMB correlation functions
    // =============================================================================

    TwoPoint::CMBCorrelation
    PowerSpectrum::get_cmb_correlation(const std::vector<double> & theta, bool lensed, double fwhm) const {
        require_solved("get_cmb_correlation");
        if (lensed && !options.lensing)
            throw std::runtime_error("PowerSpectrum::get_cmb_correlation: lensed requires options.lensing = true");
        const int L = options.ell_max;
        std::vector<double> tt(L + 1, 0.0), ee, te, bb;
        if (pol) {
            ee.assign(L + 1, 0.0);
            te.assign(L + 1, 0.0);
        }
        if (lensed && pol)
            bb.assign(L + 1, 0.0);
        for (int l = 2; l <= L; l++) {
            const double f = 2.0 * M_PI / (l * (l + 1.0)); // D_ell [muK^2] -> C_ell [muK^2]
            tt[l] = f * (lensed ? get_Dell_TT_lensed(l) : get_Dell_TT(l));
            if (pol) {
                ee[l] = f * (lensed ? get_Dell_EE_lensed(l) : get_Dell_EE(l));
                te[l] = f * (lensed ? get_Dell_TE_lensed(l) : get_Dell_TE(l));
                if (lensed)
                    bb[l] = f * get_Dell_BB_lensed(l);
            }
        }
        return TwoPoint::cmb_correlation(tt, ee, te, bb, theta, fwhm);
    }

    // =============================================================================
    // Output
    // =============================================================================

    void PowerSpectrum::output_cmb_correlation(const std::string & filename, bool lensed, double fwhm) const {
        std::vector<double> th;
        for (int i = 0; i < 1000; i++)
            th.push_back(0.01 * i * M_PI / 180.0);
        for (int i = 0; i <= 1700; i++)
            th.push_back((10.0 + 0.1 * i) * M_PI / 180.0);
        const int n = static_cast<int>(th.size());
        const auto C = get_cmb_correlation(th, lensed, fwhm);
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_cmb_correlation: cannot open " + filename);
        fp << "# " << (lensed ? "lensed" : "unlensed")
           << " CMB correlation functions [muK^2], beam fwhm = " << fwhm * 180.0 / M_PI * 60.0 << " arcmin, ell = 2.."
           << options.ell_max << "\n"
           << "# theta[deg]  C_TT  xi_+  xi_-  xi_X(TQ)\n";
        fp << std::scientific << std::setprecision(10);
        for (int i = 0; i < n; i++)
            fp << th[i] * 180.0 / M_PI << " " << C.TT[i] << " " << C.plus[i] << " " << C.minus[i] << " " << C.TQ[i]
               << "\n";
    }

    void PowerSpectrum::output_matter_correlation(const std::string & filename, double x, double R_smooth) const {
        const double Mpc_h = Mpc / cosmo.get_params().h;
        std::vector<double> r;
        for (double rr = 1.0; rr <= 200.0 + 1e-9; rr += 0.5)
            r.push_back(rr * Mpc_h);
        const auto xi = get_xi_matter(x, r, R_smooth);
        std::ofstream fp(filename);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_matter_correlation: cannot open " + filename);
        fp << "# x = " << x << ", Gaussian smoothing R = " << R_smooth / Mpc_h << " Mpc/h, sigma_8 = " << get_sigma8(x)
           << " (linear)\n# r[Mpc/h]  xi(r)  r^2 xi [(Mpc/h)^2]\n";
        fp << std::scientific << std::setprecision(10);
        for (size_t i = 0; i < r.size(); i++) {
            const double rr = r[i] / Mpc_h;
            fp << rr << " " << xi[i] << " " << rr * rr * xi[i] << "\n";
        }
    }

    void PowerSpectrum::output_angular_matter(const std::string & file_cl,
                                              const std::string & file_w,
                                              const TwoPoint::RedshiftWindow & window,
                                              int ell_max) const {
        std::vector<int> ell_list;
        for (int l = 0; l <= ell_max; l++)
            ell_list.push_back(l);
        const auto cl = get_angular_cl_matter(window, ell_list);
        std::ofstream fp(file_cl);
        if (!fp)
            throw std::runtime_error("PowerSpectrum::output_angular_matter: cannot open " + file_cl);
        fp << "# Projected density, bias = " << window.bias << ", z in [" << window.z_min << ", " << window.z_max
           << "]\n# ell  C_ell  ell(ell+1)C_ell/2pi\n";
        fp << std::scientific << std::setprecision(10);
        for (int l = 2; l <= ell_max; l++)
            fp << l << " " << cl[l] << " " << l * (l + 1.0) * cl[l] / (2.0 * M_PI) << "\n";

        std::vector<double> th_deg = Utils::logspace(0.01, 30.0, 300), th(th_deg.size());
        for (size_t i = 0; i < th.size(); i++)
            th[i] = th_deg[i] * M_PI / 180.0;
        const auto w = get_angular_correlation_matter(window, th);
        std::ofstream fw(file_w);
        if (!fw)
            throw std::runtime_error("PowerSpectrum::output_angular_matter: cannot open " + file_w);
        fw << "# Angular correlation function of the projected density (ell = 2..30000, tapered)\n# theta[deg]  "
              "w(theta)\n";
        fw << std::scientific << std::setprecision(10);
        for (size_t i = 0; i < th.size(); i++)
            fw << th_deg[i] << " " << w[i] << "\n";
    }

} // namespace FML::COSMOLOGY
