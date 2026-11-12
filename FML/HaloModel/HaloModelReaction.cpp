#include "HaloModelReaction.h"

#include <FML/HaloModel/HaloModelTools.h>
#include <FML/SphericalCollapse/SphericalCollapse.h>

#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>

namespace FML {
    namespace COSMOLOGY {
        namespace HALOMODEL {

            using namespace SPHERICALCOLLAPSE;

            namespace {
                // A spherical-collapse model with the background E(x) and the clustering matter Omega_c
                SphericalCollapseModel collapse_model(HaloModelReaction::Func1 E,
                                                      HaloModelReaction::Func1 dlogE,
                                                      double Omega_c) {
                    Func1 Om = [E, Omega_c](double x) { return Omega_c * std::exp(-3.0 * x) / std::pow(E(x), 2); };
                    Func1 mu = [](double) { return 1.0; };
                    Func1 w = [](double) { return -1.0; };
                    SphericalCollapseModel m(E, Om, dlogE, mu, w);
                    m.OmegaM = Omega_c;
                    return m;
                }
                // D(a) normalised to a at early times
                double growth(const SphericalCollapseModel & m, double xini, double x) {
                    Spline D, f;
                    compute_growthfactor(m, D, f, xini, std::max(x, 0.0), 2000);
                    return D(x);
                }
            } // namespace

            HaloModelReaction::HaloModelReaction(const Cosmology & cosmology,
                                                 double a_,
                                                 const LinearSpectra & spectra,
                                                 FifthForceOfRadius fifth_force,
                                                 const Options & options)
                : cosmo(cosmology), a(a_), spec(spectra), opt(options), mg(bool(fifth_force)) {
                if (not(cosmo.OmegaM > 0.0 and cosmo.OmegaM < 1.0 and cosmo.OmegaNu >= 0.0 and
                        cosmo.OmegaNu < cosmo.OmegaM))
                    throw std::invalid_argument("HaloModelReaction: need 0 <= OmegaNu < OmegaM < 1");
                if (not(a > 0.0 and a <= 1.0))
                    throw std::invalid_argument("HaloModelReaction: need 0 < a <= 1");
                if (not spec.P or not spec.Pcb_lcdm)
                    throw std::invalid_argument("HaloModelReaction: give the target P and the LCDM cold spectrum");
                if (not spec.Pcb)
                    spec.Pcb = spec.P;
                if (opt.n_mass < 4 or opt.n_mass_pseudo < 4 or opt.n_mass_integral < 10 or
                    not(opt.log10M_max > opt.log10M_min))
                    throw std::invalid_argument("HaloModelReaction: invalid mass grid options");
                const double Om = cosmo.OmegaM;
                if (not cosmo.Eofx) {
                    cosmo.Eofx = [Om](double x) { return std::sqrt(Om * std::exp(-3.0 * x) + 1.0 - Om); };
                    cosmo.logEprimeofx = [Om](double x) {
                        const double m = Om * std::exp(-3.0 * x);
                        return -1.5 * m / (m + 1.0 - Om);
                    };
                } else if (not cosmo.logEprimeofx) {
                    throw std::invalid_argument("HaloModelReaction: give dlogE/dx with E(x)");
                }
                build(real, true, fifth_force);
                build(pseudo, false, {});
                if (mg)
                    E_ = std::pow(1.0 - cosmo.OmegaNu / Om, 2) * one_halo(opt.k_E) / one_halo_pseudo(opt.k_E);
            }

            void HaloModelReaction::build(Population & pop, bool is_real, const FifthForceOfRadius & fifth_force) {
                const double Om = cosmo.OmegaM, Ocb = Om - cosmo.OmegaNu;
                const double x = std::log(a);
                const auto & o = opt.collapse;
                // LCDM background (OmegaM total in H)
                const Func1 E_lcdm = [Om](double xx) { return std::sqrt(Om * std::exp(-3.0 * xx) + 1.0 - Om); };
                const Func1 dlogE_lcdm = [Om](double xx) {
                    const double m = Om * std::exp(-3.0 * xx);
                    return -1.5 * m / (m + 1.0 - Om);
                };
                // Real: target background, Omega_cb clusters; pseudo: LCDM, OmegaM clusters
                const double Oc = is_real ? Ocb : Om;
                const auto model = is_real ? collapse_model(cosmo.Eofx, cosmo.logEprimeofx, Ocb)
                                           : collapse_model(E_lcdm, dlogE_lcdm, Om);
                // delta_c is extrapolated with the growth of the LCDM background (source Oc)
                const auto lcdm = collapse_model(E_lcdm, dlogE_lcdm, Oc);
                const double growth_ratio = growth(lcdm, o.xini, x) / std::exp(o.xini);
                if (is_real)
                    gde = growth(lcdm, o.xini, 0.0) / growth(model, o.xini, 0.0);

                // sigma(R) of P_cb^LCDM (real) or P (pseudo)
                const Func1 & P = is_real ? spec.Pcb_lcdm : spec.P;
                const LinearVariance var(
                    [&P](double k) { return k * k * k * P(k) / (2.0 * M_PI * M_PI); }, opt.kmin_sigma, opt.kmax_sigma,
                    opt.n_sigma);
                pop.rho = critical_density() * Oc;
                pop.g = is_real ? gde : 1.0;

                // The most probable environment (real; GR collapses do not depend on it)
                double env = 0.0;
                if (is_real and mg) {
                    const double s2 = var.sigma2(8.0);
                    const double ds2dR = 2.0 * s2 * var.dlnsigma_dlnR(8.0) / 8.0;
                    env = environment_most_probable(s2, ds2dR, Ocb, 1.0 / a - 1.0);
                    zeta = env;
                }

                const int n = is_real ? opt.n_mass : opt.n_mass_pseudo;
                pop.halos.resize(n);
                std::exception_ptr exception = nullptr;
                // GR collapses do not depend on the mass: do one
                const bool mass_dependent = is_real and mg;
                ScreenedCollapseResult gr;
                if (not mass_dependent)
                    gr = screened_spherical_collapse(model, fifth_force_none(), x, 0.0, o);
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
                for (int i = 0; i < n; i++) {
                    try {
                        Halo & h = pop.halos[i];
                        h.log10M = opt.log10M_min + (opt.log10M_max - opt.log10M_min) * i / (n - 1.0);
                        h.R_th = radius_of_mass(std::pow(10.0, h.log10M), pop.rho);
                        const auto r = mass_dependent ?
                                           screened_spherical_collapse(model, fifth_force(h.R_th), x, env, o) :
                                           gr;
                        h.deltac = r.delta_ini * growth_ratio;
                        h.sigma = var.sigma(h.R_th);
                        h.a_vir = std::exp(r.xvir);
                        h.delta_vir = r.delta_vir;
                        h.DeltaVir = r.DeltaVir;
                    } catch (...) {
#ifdef USE_OMP
#pragma omp critical
#endif
                        if (not exception)
                            exception = std::current_exception();
                    }
                }
                if (exception)
                    std::rethrow_exception(exception);

                std::vector<double> lm, dc, sg, dv;
                for (const auto & h : pop.halos) {
                    lm.push_back(h.log10M);
                    dc.push_back(h.deltac);
                    sg.push_back(h.sigma);
                    dv.push_back(h.DeltaVir);
                }
                pop.deltac = Spline(lm, dc, "deltac(log10M)");
                pop.sigma = Spline(lm, sg, "sigma(log10M)");
                pop.DeltaVir = Spline(lm, dv, "DeltaVir(log10M)");

                // Quadrature nodes for the one-halo integral (Simpson in log10 M over ReACT's range)
                int nq = opt.n_mass_integral;
                if (nq % 2 == 0)
                    nq++;
                const double lo = 1.01 * opt.log10M_min, hi = 0.99 * opt.log10M_max, d = (hi - lo) / (nq - 1);
                const double A = opt.st_A * std::sqrt(2.0 * opt.st_a / M_PI);
                pop.lm.resize(nq);
                pop.weight.resize(nq);
                pop.rvir.resize(nq);
                pop.conc.resize(nq);
                for (int i = 0; i < nq; i++) {
                    const double l = lo + d * i, M = std::pow(10.0, l);
                    const double s = (i == 0 or i == nq - 1) ? 1.0 : (i % 2 ? 4.0 : 2.0);
                    const double deltac = pop.deltac(l), sigma = pop.sigma(l);
                    const double nu = deltac / sigma;
                    // d nu / d ln M
                    const double dnu = (pop.deltac.deriv_x(l) / sigma - deltac * pop.sigma.deriv_x(l) / (sigma * sigma)) /
                                       std::log(10.0);
                    const double fnu = A * (1.0 + std::pow(opt.st_a * nu * nu, -opt.st_p)) *
                                       std::exp(-opt.st_a * nu * nu / 2.0);
                    // n(M) = rho / M f(nu) dnu/dlnM, times (M / rho)^2 and dlnM = ln(10) dlog10M
                    const double nM = pop.rho / M * fnu * std::abs(dnu);
                    pop.lm[i] = l;
                    pop.weight[i] = s * d / 3.0 * std::log(10.0) * nM * std::pow(M / pop.rho, 2);
                    pop.rvir[i] = radius_of_mass(M, pop.rho * pop.DeltaVir(l));
                    // M*: sigma(M*) = delta_c(M) (on the sigma spline, monotonic in M)
                    double a0 = opt.log10M_min, b0 = opt.log10M_max;
                    double lstar;
                    if (pop.sigma(a0) < deltac)
                        lstar = a0;
                    else if (pop.sigma(b0) > deltac)
                        lstar = b0;
                    else {
                        while (b0 - a0 > 1e-10) {
                            const double m = 0.5 * (a0 + b0);
                            (pop.sigma(m) > deltac ? a0 : b0) = m;
                        }
                        lstar = 0.5 * (a0 + b0);
                    }
                    pop.conc[i] = pop.g * opt.c0 * std::pow(10.0, -opt.alpha * (l - lstar)) * a;
                }
            }

            double HaloModelReaction::one_halo_term(double k, const Population & pop) const {
                double s = 0.0;
                for (size_t i = 0; i < pop.lm.size(); i++) {
                    const double u = nfw_fourier(k * pop.rvir[i], pop.conc[i]);
                    s += pop.weight[i] * u * u;
                }
                return s;
            }

            double HaloModelReaction::Pnu(double k) const {
                const double Om = cosmo.OmegaM, On = cosmo.OmegaNu;
                if (On == 0.0)
                    return 0.0;
                const double t = (Om * std::sqrt(spec.P(k)) - (Om - On) * std::sqrt(spec.Pcb(k))) / On;
                return t * t;
            }

            bool HaloModelReaction::set_one_loop(double P_spt_real_k0, double P_spt_pseudo_k0) {
                kstar_set = true;
                if (not mg)
                    return true;
                const double k0 = opt.k0;
                const double fv = cosmo.OmegaNu / cosmo.OmegaM, fvt2 = (1 - fv) * (1 - fv);
                E_ = fvt2 * one_halo(opt.k_E) / one_halo_pseudo(opt.k_E);
                const double p1h = one_halo(k0), p1hp = one_halo_pseudo(k0);
                const double plm = spec.P(k0), plcb = spec.Pcb(k0), plnu = Pnu(k0);
                // The reaction at k0 with the one-loop spectra as the two-halo terms
                const double pspt = P_spt_real_k0 + p1h;
                const double rspt = (fvt2 * pspt + 2.0 * fv * (1 - fv) * std::sqrt(pspt * plnu) + fv * fv * plnu) /
                                    (P_spt_pseudo_k0 + p1hp);
                // Solve R_halo-model(k0) = rspt for exp(-k0/k*) (a quadratic in sqrt for neutrinos)
                const double a1 = (1.0 - E_) * plcb * fvt2;
                const double t1 = fv * fv * plnu + (plm + p1hp) * rspt - fvt2 * (E_ * plcb + p1h);
                const double t2 = 2.0 * std::sqrt(fv * fv * (plm + p1hp) * plnu * rspt);
                const double arg = E_ > 1.0 ? (t1 + t2) / a1 : (t1 - t2) / a1;
                auto fallback = [&]() {
                    kstar_ = 1.0;
                    E_ = 1.0;
                    return false;
                };
                if (not(arg >= 0.001 and arg <= 1.0))
                    return fallback();
                kstar_ = -k0 / std::log(arg);
                // Check that it is a root
                const double ph = ((1.0 - E_) * std::exp(-k0 / kstar_) + E_) * plcb + p1h;
                const double hm = (fvt2 * ph + 2.0 * fv * (1 - fv) * std::sqrt(ph * plnu) + fv * fv * plnu) / (plm + p1hp);
                if (std::abs(hm - rspt) > 1e-4)
                    return fallback();
                return true;
            }

            double HaloModelReaction::reaction(double k) const {
                if (k < opt.kmin_reaction)
                    return 1.0;
                if (mg and not kstar_set)
                    throw std::runtime_error("HaloModelReaction: set k* with set_one_loop for modified gravity");
                const double fv = cosmo.OmegaNu / cosmo.OmegaM;
                const double plm = spec.P(k), plcb = spec.Pcb(k), plnu = Pnu(k);
                const double damp = mg ? (1.0 - E_) * std::exp(-k / kstar_) + E_ : 1.0;
                const double ph = damp * plcb + one_halo(k);
                return ((1 - fv) * (1 - fv) * ph + 2.0 * fv * (1 - fv) * std::sqrt(ph * plnu) + fv * fv * plnu) /
                       (plm + one_halo_pseudo(k));
            }

            ReactionTable::ReactionTable(const std::vector<double> & z,
                                         const std::vector<double> & k,
                                         const std::vector<std::vector<double>> & R) {
                if (z.size() < 4 or k.size() < 4 or R.size() != z.size())
                    throw std::invalid_argument("ReactionTable: need >= 4 redshifts and k and R[z][k]");
                for (size_t i = 0; i < z.size(); i++) {
                    if (not(z[i] >= 0.0) or (i > 0 and not(z[i] > z[i - 1])) or R[i].size() != k.size())
                        throw std::invalid_argument("ReactionTable: z must be increasing and >= 0, R[z] of size k");
                    for (double r : R[i])
                        if (not(r > 0.0) or not std::isfinite(r))
                            throw std::invalid_argument("ReactionTable: R must be positive and finite");
                }
                std::vector<double> xs, lk;
                std::vector<std::vector<double>> lr;
                for (size_t i = 0; i < k.size(); i++) {
                    if (not(k[i] > 0.0) or (i > 0 and not(k[i] > k[i - 1])))
                        throw std::invalid_argument("ReactionTable: k must be positive and increasing");
                    lk.push_back(std::log(k[i]));
                }
                for (size_t i = z.size(); i-- > 0;) {
                    xs.push_back(-std::log1p(z[i]));
                    std::vector<double> row;
                    for (double r : R[i])
                        row.push_back(std::log(r));
                    lr.push_back(std::move(row));
                }
                xmin = xs.front();
                xmax = xs.back();
                lkmin = lk.front();
                lkmax = lk.back();
                lnR = FML::INTERPOLATION::SPLINE::Spline2D(xs, lk, lr, "ln R(log a, log k)");
            }

            double ReactionTable::operator()(double x, double k) const {
                if (not std::isfinite(x) or not(k > 0.0) or not std::isfinite(k))
                    throw std::invalid_argument("ReactionTable: need finite x and positive finite k");
                if (x < xmin or x > xmax)
                    throw std::out_of_range("ReactionTable: x outside the computed redshift range");
                const double lk = std::log(k);
                if (lk < lkmin)
                    return 1.0;
                return std::exp(lnR(x, std::min(lk, lkmax)));
            }

        } // namespace HALOMODEL
    } // namespace COSMOLOGY
} // namespace FML
