#include <FML/HaloModel/Halomodel.h>
#include <FML/FileUtils/FileUtils.h>
#include <FML/Math/Math.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            namespace {
                using FML::COSMOLOGY::SPHERICALCOLLAPSE::compute_growthfactor;

                // Extend a logY(logx) spline assuming Y ~ x^n close to the endpoints
                void powerlaw_extend_logY_of_logx_spline(Spline & logY_of_logx_spline,
                                                         double xmin_new,
                                                         double xmax_new,
                                                         int npts_new,
                                                         const std::string & label) {
                    const auto logx_array = logY_of_logx_spline.get_x_data();
                    const double logxmin = logx_array.front();
                    const double logxmax = logx_array.back();
                    // Slopes just inside the range
                    const double dlogx = 0.01 * (logxmax - logxmin);
                    const double slope_min = logY_of_logx_spline.deriv_x(logxmin + dlogx);
                    const double slope_max = logY_of_logx_spline.deriv_x(logxmax - dlogx);
                    auto logY_extrapolated = [&](double logx) {
                        if (logx < logxmin)
                            return logY_of_logx_spline(logxmin) + (logx - logxmin) * slope_min;
                        if (logx > logxmax)
                            return logY_of_logx_spline(logxmax) + (logx - logxmax) * slope_max;
                        return logY_of_logx_spline(logx);
                    };
                    const auto logx_array_new = FML::MATH::linspace(std::log(xmin_new), std::log(xmax_new), npts_new);
                    DVector logY_array_new;
                    for (const auto logx : logx_array_new)
                        logY_array_new.push_back(logY_extrapolated(logx));
                    logY_of_logx_spline = Spline(logx_array_new, logY_array_new, label);
                }

                // The k-grid of sigma(R) and the R-grid of its spline, and the k-grid of sigma_v (h/Mpc, Mpc/h)
                constexpr double sigma_kmin = 1e-8, sigma_kmax = 1e8;
                constexpr int sigma_nk = 500;
                constexpr double sigma_Rmin = 1e-8, sigma_Rmax = 1e8;
                constexpr int sigma_nR = 500;
                constexpr double sigmav_kmin = 1e-8, sigmav_kmax = 1e4;
                constexpr int sigmav_nk = 1000;

                // n_eff = -3 - dln sigma^2 / dln R where nu = 1 (the end of [1e-4, 1e4] Mpc/h if nu = 1 is outside)
                std::pair<double, double> compute_neff(const Spline & logsigma_of_logR_spline,
                                                       const Spline & lognu_of_logR_spline) {
                    const double logRmin = std::log(1e-4);
                    const double logRmax = std::log(1e4);
                    auto condition = [&](double logR) { return lognu_of_logR_spline(logR); };
                    double logR;
                    if (condition(logRmin) > 0.0)
                        logR = logRmin;
                    else if (condition(logRmax) < 0.0)
                        logR = logRmax;
                    else
                        logR = FML::MATH::find_root_bisection(condition, {logRmin, logRmax}, 1e-10);
                    const double neff = -3.0 - 2.0 * logsigma_of_logR_spline.deriv_x(logR);
                    return {std::exp(logR), neff};
                }

                // The one- and two-halo integrals over nu (Delta^2 = k^3 P / 2 pi^2 of each term), normalised with
                // int f dnu and int b f dnu (both 1 for a normalised mass function)
                void compute_one_and_two_halo_terms(const Spline & logR_of_lognu_spline,
                                                    const Spline2D & y_of_logR_and_logk_spline,
                                                    const DVector & logk_array,
                                                    DVector & Delta_onehalo,
                                                    DVector & Delta_twohalo,
                                                    const Func1 & Delta_lin_of_k,
                                                    const Func1 & pdf_of_nu,
                                                    const Func1 & bias_of_nu,
                                                    double eta,
                                                    int accuracy_boost,
                                                    bool verbose) {
                    // (200 points gave up to 1% errors at k = 20-100 h/Mpc)
                    const double numin = 1e-10, numax = 10;
                    const int npts_nu = 600 * accuracy_boost;
                    const auto lognu_array = FML::MATH::linspace(std::log(numin), std::log(numax), npts_nu);
                    const double dlognu = lognu_array[1] - lognu_array[0];

                    DVector f_nu(npts_nu), b_nu(npts_nu), logR(npts_nu);
                    double normone = 0.0, normtwo = 0.0;
                    for (int i = 0; i < npts_nu; i++) {
                        const double nu = std::exp(lognu_array[i]);
                        f_nu[i] = pdf_of_nu(nu);
                        b_nu[i] = bias_of_nu(nu);
                        logR[i] = logR_of_lognu_spline(lognu_array[i]);
                        normone += f_nu[i] * nu;
                        normtwo += b_nu[i] * f_nu[i] * nu;
                    }
                    normone *= dlognu;
                    normtwo *= dlognu;
                    if (verbose and (std::abs(normone - 1.0) > 0.001 or std::abs(normtwo - 1.0) > 0.001))
                        std::cout << "Note: int f(nu) dnu = " << normone << " and int b(nu) f(nu) dnu = " << normtwo
                                  << " (not normalised); the halo terms are divided by them\n";

                    Delta_onehalo.clear();
                    Delta_twohalo.clear();
                    for (const auto logk : logk_array) {
                        const double k = std::exp(logk);
                        double onehalo = 0.0, twohalo = 0.0;
                        for (int i = 0; i < npts_nu; i++) {
                            const double nu = std::exp(lognu_array[i]);
                            const double y = y_of_logR_and_logk_spline(logR[i], logk + lognu_array[i] * eta);
                            const double Moverrhomean = 4.0 * M_PI / 3.0 * std::exp(3.0 * logR[i]);
                            onehalo += Moverrhomean * f_nu[i] * (y * y) * nu;
                            twohalo += b_nu[i] * f_nu[i] * y * nu;
                        }
                        onehalo *= dlognu * (k * k * k) / (2.0 * M_PI * M_PI) / normone;
                        twohalo *= dlognu / normtwo;
                        Delta_onehalo.push_back(onehalo);
                        Delta_twohalo.push_back(twohalo * twohalo * Delta_lin_of_k(k));
                    }
                }
            } // namespace

            //=========================================================================================================
            // Input
            //=========================================================================================================

            void read_pofk_file(const std::string & filename_pofk,
                                Spline & logDelta_of_logk_spline,
                                double normalization,
                                int col_k,
                                int col_pofk) {
                auto data = FML::FILEUTILS::loadtxt(filename_pofk);
                DVector logk_array, logDelta_array;
                for (size_t i = 0; i < data.size(); i++) {
                    if (int(data[i].size()) <= std::max(col_k, col_pofk))
                        throw std::runtime_error("[read_pofk_file] Row " + std::to_string(i) + " of " + filename_pofk +
                                                 " has too few columns");
                    const double k = data[i][col_k];
                    const double pofk = data[i][col_pofk];
                    if (not(k > 0.0 and pofk > 0.0 and std::isfinite(k) and std::isfinite(pofk)))
                        throw std::runtime_error("[read_pofk_file] Row " + std::to_string(i) + " of " + filename_pofk +
                                                 " has k <= 0 or P(k) <= 0 (we spline log P(k))");
                    if (i > 0 and not(std::log(k) > logk_array.back()))
                        throw std::runtime_error("[read_pofk_file] k must be increasing in " + filename_pofk);
                    logk_array.push_back(std::log(k));
                    logDelta_array.push_back(std::log(pofk * (k * k * k) / (2.0 * M_PI * M_PI) * normalization));
                }
                logDelta_of_logk_spline = Spline(logk_array, logDelta_array, "logDelta(logk)");
                powerlaw_extend_logY_of_logx_spline(
                    logDelta_of_logk_spline, 1e-12, 1e12, 1000, "logDelta(logk) extrapolated");
            }

            HaloModelCosmology cosmology_from_pofk_spline(const SphericalCollapseModel & collapse,
                                                          const Spline & logDelta_of_logk,
                                                          double x_pofk) {
                auto D = std::make_shared<Spline>("D(x)");
                Spline f("f(x)");
                compute_growthfactor(collapse, *D, f);
                const double D_input = (*D)(x_pofk);
                auto spline = std::make_shared<Spline>(logDelta_of_logk);
                HaloModelCosmology c;
                c.collapse = collapse;
                c.Omega_m = collapse.OmegaM;
                c.f_nu = 0.0;
                c.P_m = [D, D_input, spline](double x, double k) {
                    const double growth = (*D)(x) / D_input;
                    return std::exp((*spline)(std::log(k))) * 2.0 * M_PI * M_PI / (k * k * k) * growth * growth;
                };
                return c;
            }

            HaloModelCosmology cosmology_from_pofk_file(const SphericalCollapseModel & collapse,
                                                        const std::string & filename,
                                                        double x_pofk,
                                                        double normalization,
                                                        int col_k,
                                                        int col_pofk) {
                Spline logDelta;
                read_pofk_file(filename, logDelta, normalization, col_k, col_pofk);
                return cosmology_from_pofk_spline(collapse, logDelta, x_pofk);
            }

            //=========================================================================================================
            // Profile transform
            //=========================================================================================================

            DVector fourier_transform_profile(const std::function<double(double)> & rho_r3,
                                              const DVector & k_array,
                                              double Rvir,
                                              int accuracy_boost) {
                // Lower limit: rho(rmin) < 1e-4 rho(Rvir/10)
                const double rho_start = rho_r3(Rvir / 10.0);
                int power_min = 1;
                while (power_min < 5) {
                    ++power_min;
                    if (rho_r3(Rvir / std::pow(10, power_min)) < 1e-4 * rho_start)
                        break;
                }
                // Upper limit
                int power_max = 0;
                while (power_max < 7) {
                    ++power_max;
                    if (rho_r3(Rvir * std::pow(2, power_max)) == 0.0)
                        break;
                }
                // Trapezoid rule in log r on [rmin, Rvir] and (if the profile is not zero outside) [Rvir, rmax]
                // separately, as a truncated profile jumps to zero at Rvir
                const double rmin = Rvir / std::pow(10, power_min), rmax = Rvir * std::pow(2, power_max);
                const bool zero_outside_Rvir = rho_r3(Rvir * (1.0 + 1e-8)) == 0.0;
                auto make_r_array = [](double r1, double r2, int n) {
                    auto r_array = FML::MATH::linspace(std::log(r1), std::log(r2), n);
                    for (auto & r : r_array)
                        r = std::exp(r);
                    r_array.front() = r1;
                    r_array.back() = r2;
                    return r_array;
                };
                const int npts_r_inside = power_min * 50 * accuracy_boost + 1;
                const int npts_r_outside =
                    zero_outside_Rvir ? 0 : int(power_max * std::log10(2.0) * 50 * accuracy_boost) + 2;
                const auto r_inside = make_r_array(rmin, Rvir, npts_r_inside);
                const double dlogr_inside = std::log(Rvir / rmin) / (npts_r_inside - 1);
                const auto r_outside = zero_outside_Rvir ? DVector{} : make_r_array(Rvir, rmax, npts_r_outside);
                const double dlogr_outside = zero_outside_Rvir ? 0.0 : std::log(rmax / Rvir) / (npts_r_outside - 1);

                DVector rho_inside(r_inside.size()), rho_outside(r_outside.size());
                for (size_t i = 0; i < r_inside.size(); i++)
                    rho_inside[i] = rho_r3(r_inside[i]);
                for (size_t i = 0; i < r_outside.size(); i++)
                    rho_outside[i] = rho_r3(r_outside[i]);
                // The value just inside Rvir (the profile can jump to zero at Rvir)
                rho_inside.back() = rho_r3(Rvir * (1.0 - 1e-12));

                auto trapezoid = [](const DVector & r_array, const DVector & rho_array, double dlogr, double k) {
                    double integral = 0.0;
                    for (size_t i = 0; i < r_array.size(); i++) {
                        const double kr = k * r_array[i];
                        const double j0 = kr < 1e-6 ? 1.0 - kr * kr / 6.0 : std::sin(kr) / kr;
                        const double weight = (i == 0 or i + 1 == r_array.size()) ? 0.5 : 1.0;
                        integral += weight * rho_array[i] * j0;
                    }
                    return integral * dlogr;
                };

                DVector y_array;
                for (const auto k : k_array) {
                    double integral = trapezoid(r_inside, rho_inside, dlogr_inside, k);
                    if (not zero_outside_Rvir)
                        integral += trapezoid(r_outside, rho_outside, dlogr_outside, k);
                    y_array.push_back(integral);
                }
                const double norm = y_array[0];
                for (auto & y : y_array)
                    y /= norm;
                return y_array;
            }

            //=========================================================================================================
            // Construction
            //=========================================================================================================

            HaloModel::HaloModel(HaloModelCosmology cosmology_, HaloModelOptions options_)
                : cosmology(std::move(cosmology_)), options(std::move(options_)) {
                init();
            }

            void HaloModel::set_options(HaloModelOptions options_) {
                options = std::move(options_);
                current = HaloModelState{};
                init();
            }

            void HaloModel::init() {
                if (not cosmology.P_m)
                    throw std::invalid_argument("[HaloModel] the cosmology has no linear power spectrum P_m");
                if (not(cosmology.Omega_m > 0.0) or not(cosmology.f_nu >= 0.0 and cosmology.f_nu < 1.0))
                    throw std::invalid_argument("[HaloModel] need Omega_m > 0 and 0 <= f_nu < 1");
                if (cosmology.f_nu > 0.0 and not cosmology.P_cb)
                    throw std::invalid_argument("[HaloModel] f_nu > 0 needs the cold-matter power spectrum P_cb");
                if (not(options.kmin > 0.0 and options.kmax > options.kmin and options.npts_k >= 2 and
                        options.accuracy_boost >= 1))
                    throw std::invalid_argument("[HaloModel] invalid k-range or accuracy options");
                const bool hmcode = options.prescription == Prescription::HMcode2016;

                if (options.verbose) {
                    std::cout << "\n#=====================================================\n";
                    std::cout << "# Initializing the halo model\n";
                    std::cout << "#=====================================================\n";
                    std::cout << "# Omega_m      " << cosmology.Omega_m << "\n";
                    std::cout << "# f_nu         " << cosmology.f_nu << "\n";
                    std::cout << "# Prescription " << (hmcode ? "HMcode-2016" : "Standard") << "\n";
                }

                if (not hmcode) {
                    timer.StartTiming("SphericalCollapse");
                    SPHERICALCOLLAPSE::SphericalCollapse sc(cosmology.collapse,
                                                            options.verbose,
                                                            options.npts_spherical_collapse,
                                                            options.xini_spherical_collapse);
                    sc.run_at_all_redshifts(options.zmax_spherical_collapse,
                                            deltac_of_x_spline,
                                            DeltaVir_of_x_spline,
                                            xta_of_x_spline,
                                            xnl_of_x_spline,
                                            xvir_of_x_spline,
                                            delta_ini_of_x_spline,
                                            growthfactor_of_x_spline,
                                            growthrate_of_x_spline);
                    timer.EndTiming("SphericalCollapse");
                } else {
                    compute_growthfactor(cosmology.collapse, growthfactor_of_x_spline, growthrate_of_x_spline);

                    // The Dolag et al. correction of the concentration for the expansion history: the GR growth
                    // (mu = 1) of the model against LCDM with the same Omega_m, normalised today, at z = 10
                    const auto & c = cosmology.collapse;
                    const double Om = c.OmegaM;
                    const Func1 one = [](double) { return 1.0; };
                    SphericalCollapseModel gr(c.Eofx, c.OmegaMofx, c.logEprimeofx, one, c.wofx);
                    const Func1 E = [Om](double x) { return std::sqrt(Om * std::exp(-3 * x) + 1 - Om); };
                    const Func1 Omx = [Om, E](double x) { return Om * std::exp(-3 * x) / (E(x) * E(x)); };
                    const Func1 dlogE = [Omx](double x) { return -1.5 * Omx(x); };
                    const Func1 w = [](double) { return -1.0; };
                    SphericalCollapseModel lcdm(E, Omx, dlogE, one, w);
                    Spline Dgr, Dlcdm, f;
                    compute_growthfactor(gr, Dgr, f);
                    compute_growthfactor(lcdm, Dlcdm, f);
                    const double x10 = -std::log(11.0);
                    dolag_2016 = HMcode2016Fits::dolag_factor((Dgr(x10) / Dgr(0.0)) / (Dlcdm(x10) / Dlcdm(0.0)));
                }
            }

            //=========================================================================================================
            // Helpers
            //=========================================================================================================

            double HaloModel::Delta2_m(double x, double k) const {
                return cosmology.P_m(x, k) * (k * k * k) / (2.0 * M_PI * M_PI);
            }
            double HaloModel::Delta2_cb(double x, double k) const {
                return (cosmology.P_cb ? cosmology.P_cb(x, k) : cosmology.P_m(x, k)) * (k * k * k) / (2.0 * M_PI * M_PI);
            }

            void HaloModel::check_computed(const std::string & where) const {
                if (not computed())
                    throw std::runtime_error("[HaloModel::" + where + "] call compute_at_redshift first");
            }
            void HaloModel::check_k(double k, const std::string & where) const {
                check_computed(where);
                // Outside [kmin, kmax] the splines are constant in Delta(k) (P ~ k^-3): wrong
                if (k < options.kmin * (1.0 - 1e-10) or k > options.kmax * (1.0 + 1e-10))
                    throw std::runtime_error("[HaloModel::" + where + "] k = " + std::to_string(k) +
                                             " is outside the computed range [kmin, kmax] = [" +
                                             std::to_string(options.kmin) + ", " + std::to_string(options.kmax) + "]");
            }

            HaloModelState HaloModel::hmcode2016_state_at(double x) const {
                const LinearVariance var_m(
                    [&](double k) { return Delta2_m(x, k); }, sigma_kmin, sigma_kmax, sigma_nk, options.window);
                const LinearVariance var_v(
                    [&](double k) { return Delta2_m(x, k); }, sigmav_kmin, sigmav_kmax, sigmav_nk, options.window);
                const LinearVariance var_cb(
                    [&](double k) { return Delta2_cb(x, k); }, sigma_kmin, sigma_kmax, sigma_nk, options.window);
                HaloModelState s;
                s.x = x;
                s.z = std::exp(-x) - 1.0;
                s.sigma8_m = var_m.sigma(8.0);
                s.sigma8 = var_cb.sigma(8.0);
                s.sigmav = var_v.sigma_v(0.0);
                s.sigmav100 = var_v.sigma_v(100.0);
                const double Om = cosmology.collapse.OmegaMofx(x);
                s.deltac = HMcode2016Fits::delta_c(s.sigma8_m, Om, cosmology.f_nu) * options.deltac_multiplier;
                s.DeltaVir = HMcode2016Fits::Delta_v(Om, cosmology.f_nu) * options.DeltaVir_multiplier;
                s.rnl = var_cb.radius_of_sigma(s.deltac, 1e-4, 1e4);
                if (options.window) {
                    const double e = 1e-3;
                    s.neff = -3.0 - (std::log(var_cb.sigma2(s.rnl * std::exp(e))) -
                                     std::log(var_cb.sigma2(s.rnl * std::exp(-e)))) / (2.0 * e);
                } else {
                    s.neff = -3.0 - 2.0 * var_cb.dlnsigma_dlnR(s.rnl);
                }
                s.kstar = HMcode2016Fits::kstar(s.sigmav);
                s.f_damp = HMcode2016Fits::f_damp(s.sigmav100);
                s.eta = HMcode2016Fits::eta(s.sigma8_m);
                s.alpha = HMcode2016Fits::alpha(s.neff);
                s.concentration_A = HMcode2016Fits::concentration_A * dolag_2016 * options.concentration_multiplier;
                s.rho_mean = critical_density() * cosmology.Omega_m;
                return s;
            }

            //=========================================================================================================
            // Compute everything at redshift z
            //=========================================================================================================

            void HaloModel::compute_at_redshift(double z) {
                if (not cosmology.P_m)
                    throw std::runtime_error("[HaloModel::compute_at_redshift] no cosmology (default-constructed)");
                const bool hmcode = options.prescription == Prescription::HMcode2016;
                if (not hmcode and z > options.zmax_spherical_collapse * (1.0 + 1e-10))
                    throw std::runtime_error("[HaloModel::compute_at_redshift] z = " + std::to_string(z) +
                                             " is larger than the largest redshift of the spherical collapse splines "
                                             "(zmax_spherical_collapse = " +
                                             std::to_string(options.zmax_spherical_collapse) +
                                             "): increase it in the options");
                // Invalid until done (an exception below leaves the model uncomputed)
                current = HaloModelState{};
                HaloModelState s;
                s.z = z;
                s.x = std::log(1.0 / (1.0 + z));
                const double x = s.x;
                s.rho_mean = critical_density() * cosmology.Omega_m;
                const bool verbose = options.verbose;

                if (verbose) {
                    std::cout << "\n#=====================================================\n";
                    std::cout << "# Compute at z = " << z << "\n";
                    std::cout << "#=====================================================\n";
                }

                //=====================================================
                // Variances: sigma(R) of the cold matter (splined), sigma8 and sigma_v of the total matter
                //=====================================================
                timer.StartTiming("sigma");
                const LinearVariance var_cb(
                    [&](double k) { return Delta2_cb(x, k); }, sigma_kmin, sigma_kmax, sigma_nk, options.window);
                const LinearVariance var_v(
                    [&](double k) { return Delta2_m(x, k); }, sigmav_kmin, sigmav_kmax, sigmav_nk, options.window);
                {
                    const auto logR_array = FML::MATH::linspace(std::log(sigma_Rmin), std::log(sigma_Rmax), sigma_nR);
                    DVector logsigma_array;
                    for (const auto logR : logR_array)
                        logsigma_array.push_back(0.5 * std::log(var_cb.sigma2(std::exp(logR))));
                    logsigma_of_logR_spline = Spline(logR_array, logsigma_array, "ln sigma(ln R)");
                }
                s.sigma8 = std::exp(logsigma_of_logR_spline(std::log(8.0)));
                if (cosmology.P_cb) {
                    const LinearVariance var_m(
                        [&](double k) { return Delta2_m(x, k); }, sigma_kmin, sigma_kmax, sigma_nk, options.window);
                    s.sigma8_m = var_m.sigma(8.0);
                } else {
                    s.sigma8_m = s.sigma8;
                }
                s.sigmav = var_v.sigma_v(0.0);
                s.sigmav100 = var_v.sigma_v(100.0);
                timer.EndTiming("sigma");

                //=====================================================
                // deltac and DeltaVir
                //=====================================================
                if (hmcode) {
                    const double Om = cosmology.collapse.OmegaMofx(x);
                    s.deltac = HMcode2016Fits::delta_c(s.sigma8_m, Om, cosmology.f_nu);
                    s.DeltaVir = HMcode2016Fits::Delta_v(Om, cosmology.f_nu);
                } else {
                    s.deltac = deltac_of_x_spline(x);
                    s.DeltaVir = DeltaVir_of_x_spline(x);
                }
                s.deltac *= options.deltac_multiplier;
                s.DeltaVir *= options.DeltaVir_multiplier;

                //=====================================================
                // nu(R) and the nonlinear scale nu(R) = 1
                //=====================================================
                {
                    const auto range = logsigma_of_logR_spline.get_xrange();
                    const auto logR_array = FML::MATH::linspace(range.first, range.second, 500);
                    DVector lognu_array;
                    for (const auto logR : logR_array)
                        lognu_array.push_back(std::log(s.deltac) - logsigma_of_logR_spline(logR));
                    lognu_of_logR_spline = Spline(logR_array, lognu_array, "ln nu(ln R)");
                    logR_of_lognu_spline = Spline(lognu_array, logR_array, "ln R(ln nu)");
                }
                const auto neff_data = compute_neff(logsigma_of_logR_spline, lognu_of_logR_spline);
                s.rnl = neff_data.first;
                s.neff = neff_data.second;

                //=====================================================
                // HMcode parameters
                //=====================================================
                if (hmcode) {
                    s.kstar = HMcode2016Fits::kstar(s.sigmav);
                    s.f_damp = HMcode2016Fits::f_damp(s.sigmav100);
                    s.eta = HMcode2016Fits::eta(s.sigma8_m);
                    s.alpha = HMcode2016Fits::alpha(s.neff);
                    s.concentration_A = HMcode2016Fits::concentration_A * dolag_2016;
                } else {
                    s.kstar = options.kstar;
                    s.f_damp = options.f_damp;
                    s.eta = options.eta;
                    s.alpha = options.alpha;
                    s.concentration_A = options.bullock_A;
                }
                s.concentration_A *= options.concentration_multiplier;
                // The accessors below use the state
                current = s;

                if (verbose) {
                    std::cout << "# deltac   = " << s.deltac << "\n";
                    std::cout << "# DeltaVir = " << s.DeltaVir << "\n";
                    std::cout << "# sigma8   = " << s.sigma8 << "\n";
                    std::cout << "# kstar    = " << s.kstar << "\n";
                    std::cout << "# f        = " << s.f_damp << "\n";
                    std::cout << "# eta      = " << s.eta << "\n";
                    std::cout << "# alpha    = " << s.alpha << "\n";
                    std::cout << "# A        = " << s.concentration_A << "\n";
                }

                try {
                    //=====================================================
                    // Bullock et al. formation redshifts: D(zf)/D(z) sigma(f M) = deltac
                    //=====================================================
                    timer.StartTiming("formation");
                    {
                        const double xmin_growth = growthfactor_of_x_spline.get_xrange().first;
                        const double D_x = growthfactor_of_x_spline(x);
                        auto growth_ratio = [&](double xf) {
                            if (xf < xmin_growth)
                                return growthfactor_of_x_spline(xmin_growth) / D_x * std::exp(xf - xmin_growth);
                            return growthfactor_of_x_spline(xf) / D_x;
                        };
                        const auto logR_array = FML::MATH::linspace(std::log(1e-8), std::log(1e8), 10000);
                        DVector zf_array, logM_array;
                        for (const auto logR : logR_array) {
                            const double M = mass_of_radius(std::exp(logR), s.rho_mean);
                            const double Rf = radius_of_mass(options.bullock_mass_fraction * M, s.rho_mean);
                            const double sigma_f = std::exp(logsigma_of_logR_spline(std::log(Rf)));
                            auto condition = [&](double xf) { return growth_ratio(xf) * sigma_f - s.deltac; };
                            double xf = x;
                            try {
                                xf = FML::MATH::find_root_bisection(condition, {std::log(1e-10), x}, 1e-8);
                            } catch (...) {
                            }
                            zf_array.push_back(std::exp(-xf) - 1.0);
                            logM_array.push_back(std::log(M));
                        }
                        formationredshift_of_logM_spline = Spline(logM_array, zf_array, "zf(ln M)");
                    }
                    timer.EndTiming("formation");

                    //=====================================================
                    // Fourier transform of the profiles y(R, k) (R the Lagrangian radius)
                    //=====================================================
                    timer.StartTiming("profile");
                    {
                        const int boost = options.accuracy_boost;
                        // (75 points for 16 decades gave up to 3.5% errors in P(k) at k > 10 h/Mpc)
                        const auto logk_array = FML::MATH::linspace(std::log(1e-8), std::log(1e8), 400 * boost);
                        const auto logR_array = FML::MATH::linspace(std::log(1e-8), std::log(1e8), 400 * boost);
                        DVector k_array(logk_array.size());
                        for (size_t j = 0; j < k_array.size(); j++)
                            k_array[j] = std::exp(logk_array[j]);
                        std::vector<DVector> y(logR_array.size(), DVector(k_array.size()));
                        const HaloModel & hm = *this;
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                        for (size_t i = 0; i < logR_array.size(); i++) {
                            const double R = std::exp(logR_array[i]);
                            const double M = mass_of_radius(R, s.rho_mean);
                            const double Rvir = R / std::cbrt(s.DeltaVir);
                            if (options.density_profile) {
                                y[i] = fourier_transform_profile(
                                    [&](double r) { return options.density_profile(r, M, hm); }, k_array, Rvir, boost);
                            } else {
                                const double c = concentration(M);
                                for (size_t j = 0; j < k_array.size(); j++)
                                    y[i][j] = nfw_fourier(k_array[j] * Rvir, c);
                                const double norm = y[i][0];
                                for (auto & yy : y[i])
                                    yy /= norm;
                            }
                        }
                        y_of_logR_and_logk_spline = Spline2D(logR_array, logk_array, y, "y(ln R, ln k)");
                    }
                    timer.EndTiming("profile");

                    //=====================================================
                    // Mass function dn/dln M and n(> M)
                    //=====================================================
                    timer.StartTiming("massfunction");
                    {
                        const int npts_M = 500;
                        const auto logM_array = FML::MATH::linspace(std::log(1e5), std::log(1e18), npts_M);
                        const double dlogM = logM_array[1] - logM_array[0];
                        DVector dndlogM_array(npts_M), n_array(npts_M, 0.0);
                        for (int i = npts_M - 1; i >= 0; i--) {
                            const double logR = std::log(radius_of_mass(std::exp(logM_array[i]), s.rho_mean));
                            const double nu = std::exp(lognu_of_logR_spline(logR));
                            const double dnudlogR = nu * lognu_of_logR_spline.deriv_x(logR);
                            dndlogM_array[i] = multiplicity(nu) * std::abs(dnudlogR) / (4.0 * M_PI * std::exp(3 * logR));
                            n_array[i] = dndlogM_array[i] * dlogM + (i < npts_M - 1 ? n_array[i + 1] : 0.0);
                        }
                        dndlogM_of_logM_spline = Spline(logM_array, dndlogM_array, "dn/dlnM(ln M)");
                        n_of_logM_spline = Spline(logM_array, n_array, "n(ln M)");
                    }
                    timer.EndTiming("massfunction");

                    //=====================================================
                    // P(k): one- and two-halo terms
                    //=====================================================
                    timer.StartTiming("pofk");
                    const auto logk_array =
                        FML::MATH::linspace(std::log(options.kmin), std::log(options.kmax), options.npts_k);
                    DVector D1h, D2h;
                    compute_one_and_two_halo_terms(
                        logR_of_lognu_spline,
                        y_of_logR_and_logk_spline,
                        logk_array,
                        D1h,
                        D2h,
                        [&](double k) { return Delta2_m(x, k); },
                        [&](double nu) { return multiplicity(nu); },
                        [&](double nu) { return bias(nu); },
                        s.eta,
                        options.accuracy_boost,
                        verbose);

                    // The modifications: damping of the one-halo term at low k and of the two-halo term (HMcode-2016:
                    // the linear P(k) is the two-halo term), the transition and the neutrino factor
                    const double one_halo_nu = hmcode ? 1.0 : std::pow(1.0 - cosmology.f_nu, 2);
                    DVector logD1h, logD2h, logD, logDlin;
                    for (size_t i = 0; i < logk_array.size(); i++) {
                        const double k = std::exp(logk_array[i]);
                        const double Delta_lin = Delta2_m(x, k);
                        double Delta_onehalo = D1h[i] * one_halo_nu;
                        double Delta_twohalo = hmcode ? Delta_lin : D2h[i];
                        if (s.f_damp != 0.0)
                            Delta_twohalo *= std::max(
                                1e-12, 1.0 - s.f_damp * std::pow(std::tanh(k * s.sigmav / std::sqrt(s.f_damp)), 2));
                        if (s.kstar > 0.0) {
                            const double k_over_kstar = std::min(k / s.kstar, 20.0);
                            Delta_onehalo *= k_over_kstar < 1e-5 ? k_over_kstar * k_over_kstar :
                                                                   1.0 - std::exp(-k_over_kstar * k_over_kstar);
                        }
                        // A minimum value as we spline the logarithms
                        Delta_onehalo = std::max(Delta_onehalo, 1e-30);
                        Delta_twohalo = std::max(Delta_twohalo, 1e-30);
                        const double Delta_full = s.alpha != 1.0 ?
                                                      hmcode_transition(Delta_onehalo, Delta_twohalo, s.alpha) :
                                                      Delta_onehalo + Delta_twohalo;
                        logD1h.push_back(std::log(Delta_onehalo));
                        logD2h.push_back(std::log(Delta_twohalo));
                        logD.push_back(std::log(Delta_full));
                        logDlin.push_back(std::log(Delta_lin));
                    }
                    logDelta_onehalo_of_logk_spline = Spline(logk_array, logD1h, "ln Delta_1h(ln k)");
                    logDelta_twohalo_of_logk_spline = Spline(logk_array, logD2h, "ln Delta_2h(ln k)");
                    logDelta_full_of_logk_spline = Spline(logk_array, logD, "ln Delta(ln k)");
                    logDelta_lin_of_logk_spline = Spline(logk_array, logDlin, "ln Delta_lin(ln k)");
                    timer.EndTiming("pofk");
                } catch (...) {
                    current = HaloModelState{};
                    throw;
                }
            }

            //=========================================================================================================
            // Accessors
            //=========================================================================================================

            double HaloModel::pofk(double k) const {
                check_k(k, "pofk");
                return 2.0 * M_PI * M_PI / (k * k * k) * std::exp(logDelta_full_of_logk_spline(std::log(k)));
            }
            double HaloModel::pofk_linear(double k) const {
                check_computed("pofk_linear");
                return cosmology.P_m(current.x, k);
            }
            double HaloModel::pofk_onehalo(double k) const {
                check_k(k, "pofk_onehalo");
                return 2.0 * M_PI * M_PI / (k * k * k) * std::exp(logDelta_onehalo_of_logk_spline(std::log(k)));
            }
            double HaloModel::pofk_twohalo(double k) const {
                check_k(k, "pofk_twohalo");
                return 2.0 * M_PI * M_PI / (k * k * k) * std::exp(logDelta_twohalo_of_logk_spline(std::log(k)));
            }
            double HaloModel::dndlogM(double M) const {
                check_computed("dndlogM");
                return dndlogM_of_logM_spline(std::log(M));
            }
            double HaloModel::n_of_M(double M) const {
                check_computed("n_of_M");
                return n_of_logM_spline(std::log(M));
            }
            double HaloModel::sigma(double R) const {
                check_computed("sigma");
                return std::exp(logsigma_of_logR_spline(std::log(R)));
            }
            double HaloModel::R_of_M(double M) const {
                check_computed("R_of_M");
                return radius_of_mass(M, current.rho_mean);
            }
            double HaloModel::Rvir_of_M(double M) const { return R_of_M(M) / std::cbrt(current.DeltaVir); }
            double HaloModel::nu_of_M(double M) const {
                check_computed("nu_of_M");
                return std::exp(lognu_of_logR_spline(std::log(R_of_M(M))));
            }
            double HaloModel::multiplicity(double nu) const {
                if (options.multiplicity)
                    return options.multiplicity(nu, *this);
                return options.sheth_tormen.f(nu);
            }
            double HaloModel::bias(double nu) const {
                if (options.bias)
                    return options.bias(nu, *this);
                return options.sheth_tormen.bias(nu, current.deltac);
            }
            double HaloModel::bias_of_M(double M) const { return bias(nu_of_M(M)); }
            double HaloModel::formation_redshift(double M) const {
                check_computed("formation_redshift");
                return formationredshift_of_logM_spline(std::log(M));
            }
            double HaloModel::concentration(double M) const {
                check_computed("concentration");
                if (options.concentration)
                    return options.concentration(M, *this);
                if (options.prescription == Prescription::HMcode2016 or
                    options.concentration_model == ConcentrationModel::Bullock2001) {
                    const double zf = formation_redshift(M);
                    return current.concentration_A * std::max(1.0, (1.0 + zf) / (1.0 + current.z));
                }
                return options.powerlaw_c0 * options.concentration_multiplier *
                       std::pow(M / options.powerlaw_M0, -options.powerlaw_beta) * std::exp(current.x);
            }
            double HaloModel::profile_fourier(double k, double M) const {
                check_computed("profile_fourier");
                return y_of_logR_and_logk_spline(std::log(R_of_M(M)), std::log(k));
            }

            double HaloModel::deltac_of_z(double z) const {
                const double x = std::log(1.0 / (1.0 + z));
                if (options.prescription == Prescription::HMcode2016)
                    return hmcode2016_state_at(x).deltac;
                return deltac_of_x_spline(x) * options.deltac_multiplier;
            }
            double HaloModel::DeltaVir_of_z(double z) const {
                const double x = std::log(1.0 / (1.0 + z));
                if (options.prescription == Prescription::HMcode2016)
                    return HMcode2016Fits::Delta_v(cosmology.collapse.OmegaMofx(x), cosmology.f_nu) *
                           options.DeltaVir_multiplier;
                return DeltaVir_of_x_spline(x) * options.DeltaVir_multiplier;
            }

            //=========================================================================================================
            // Output
            //=========================================================================================================

            void HaloModel::output_pofk(const std::string & filename, double kmin, double kmax, int npts_k) const {
                check_k(kmin, "output_pofk");
                check_k(kmax, "output_pofk");
                const auto logk_array = FML::MATH::linspace(std::log(kmin), std::log(kmax), npts_k);
                std::ofstream fp(filename);
                fp << "#  k (h/Mpc)     P_NL(k)     P_Lin(k)     P_1h(k)   P_2h(k)  (Mpc/h)^3  [z = " << current.z
                   << "]\n";
                for (const auto logk : logk_array) {
                    const double k = std::exp(logk);
                    fp << std::setw(10) << k << " " << std::setw(10) << pofk(k) << " " << std::setw(10)
                       << pofk_linear(k) << " " << std::setw(10) << pofk_onehalo(k) << " " << std::setw(10)
                       << pofk_twohalo(k) << "\n";
                }
            }

            void HaloModel::output_nofM(const std::string & filename, double Mmin, double Mmax, int npts_M) const {
                check_computed("output_nofM");
                const auto logM_array = FML::MATH::linspace(std::log(Mmin), std::log(Mmax), npts_M);
                std::ofstream fp(filename);
                fp << "#  M (Msun/h)    dn/dlnM (h/Mpc)^3      n(>M) (h/Mpc)^3      R (Mpc/h)    Rvir (Mpc/h)   nu   "
                      "c    b  [z = "
                   << current.z << "]\n";
                for (const auto logM : logM_array) {
                    const double M = std::exp(logM);
                    fp << std::setw(10) << M << "  " << std::setw(10) << dndlogM(M) << "  " << std::setw(10)
                       << n_of_M(M) << "  " << std::setw(10) << R_of_M(M) << "  " << std::setw(10) << Rvir_of_M(M)
                       << "  " << std::setw(10) << nu_of_M(M) << "  " << std::setw(10) << concentration(M) << "  "
                       << std::setw(10) << bias_of_M(M) << "\n";
                }
            }

            void HaloModel::output_deltac(const std::string & filename, double zmin, double zmax, int npts_z) const {
                const auto x_array =
                    FML::MATH::linspace(std::log(1.0 / (1.0 + zmax)), std::log(1.0 / (1.0 + zmin)), npts_z);
                std::ofstream fp(filename);
                if (options.prescription == Prescription::HMcode2016) {
                    fp << "#  z   deltac(z)   DeltaVir(z)   kstar(z)   f(z)   eta(z)   alpha(z)   A(z)   sigma8(z)  "
                          "[HMcode-2016]\n";
                    for (const auto x : x_array) {
                        const auto s = hmcode2016_state_at(x);
                        fp << std::setw(10) << s.z << "  " << std::setw(10) << s.deltac << "  " << std::setw(10)
                           << s.DeltaVir << "  " << std::setw(10) << s.kstar << "  " << std::setw(10) << s.f_damp
                           << "  " << std::setw(10) << s.eta << "  " << std::setw(10) << s.alpha << "  "
                           << std::setw(10) << s.concentration_A << "  " << std::setw(10) << s.sigma8_m << "\n";
                    }
                } else {
                    fp << "#  z     deltac(z)      DeltaVir(z)\n";
                    for (const auto x : x_array) {
                        const double z = std::exp(-x) - 1;
                        fp << std::setw(10) << z << "  " << std::setw(10) << deltac_of_z(z) << "  " << std::setw(10)
                           << DeltaVir_of_z(z) << "\n";
                    }
                }
            }

            void HaloModel::info() const {
                const bool hmcode = options.prescription == Prescription::HMcode2016;
                std::cout << "\n#=====================================================\n";
                std::cout << "# Halo model info ( z = " << (computed() ? std::to_string(current.z) : "not yet set")
                          << " )\n";
                std::cout << "#=====================================================\n";
                std::cout << "# Prescription      : " << (hmcode ? "HMcode-2016" : "Standard") << "\n";
                std::cout << "# Omega_m           : " << cosmology.Omega_m << "\n";
                std::cout << "# f_nu              : " << cosmology.f_nu << "\n";
                std::cout << "# deltac multiplier : " << options.deltac_multiplier << "\n";
                std::cout << "# DeltaVir multipl. : " << options.DeltaVir_multiplier << "\n";
                std::cout << "# c(M) multiplier   : " << options.concentration_multiplier << "\n";
                std::cout << "# Mass function     : "
                          << (options.multiplicity ? std::string("custom") :
                                                     "Sheth-Tormen (a = " + std::to_string(options.sheth_tormen.a) +
                                                         ", p = " + std::to_string(options.sheth_tormen.p) + ")")
                          << "\n";
                std::cout << "# Profile           : " << (options.density_profile ? "custom" : "NFW") << "\n";
                std::cout << "# c(M)              : "
                          << (options.concentration ? "custom" :
                              (hmcode or options.concentration_model == ConcentrationModel::Bullock2001) ?
                                                      "Bullock et al. 2001" :
                                                      "power law")
                          << "\n";
                if (not computed()) {
                    std::cout << "# (compute_at_redshift has not been run)\n";
                    return;
                }
                const auto & s = current;
                std::cout << "# deltac            : " << s.deltac << "\n";
                std::cout << "# DeltaVir          : " << s.DeltaVir << "\n";
                if (not hmcode and xta_of_x_spline and xvir_of_x_spline and xnl_of_x_spline) {
                    std::cout << "# znl               : " << std::exp(-xnl_of_x_spline(s.x)) - 1.0 << "\n";
                    std::cout << "# zturnaround       : " << std::exp(-xta_of_x_spline(s.x)) - 1.0 << "\n";
                    std::cout << "# zvir              : " << std::exp(-xvir_of_x_spline(s.x)) - 1.0 << "\n";
                }
                std::cout << "# sigma8 (cold)     : " << s.sigma8 << "\n";
                std::cout << "# sigma8 (total)    : " << s.sigma8_m << "\n";
                std::cout << "# sigmav            : " << s.sigmav << " (Mpc/h)\n";
                std::cout << "# sigmav100         : " << s.sigmav100 << " (Mpc/h)\n";
                std::cout << "# rnl               : " << s.rnl << " (Mpc/h)\n";
                std::cout << "# neff              : " << s.neff << "\n";
                std::cout << "# kstar             : " << s.kstar << " (h/Mpc)\n";
                std::cout << "# f_damp            : " << s.f_damp << "\n";
                std::cout << "# eta               : " << s.eta << "\n";
                std::cout << "# alpha             : " << s.alpha << "\n";
                std::cout << "# A (concentration) : " << s.concentration_A << "\n";
                if (options.verbose) {
                    cosmology.collapse.info();
                    timer.PrintAllTimings();
                }
            }

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML
