#include <FML/HaloModel/HMcode2016.h>
#include <FML/HaloModel/HaloModelBackground.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace FML::COSMOLOGY::HALOMODEL {
    namespace {
        void check_background(const BackgroundCosmology & bg) {
            const auto & model = bg.model_name();
            if ((model != "lcdm" && model != "w0wa") || bg.gravity_name() != "GR")
                throw std::invalid_argument("HMcode-2016 supports lcdm and w0wa backgrounds only (other models: "
                                            "HaloModel with a spherical-collapse model)");
        }
        HaloModel make(const BackgroundCosmology & bg,
                       const HMcode2016::LinearPower & Pm,
                       const HMcode2016::LinearPower & Pcb,
                       double kmin,
                       double kmax,
                       int nk,
                       int accuracy_boost) {
            check_background(bg);
            if (!Pm || !Pcb)
                throw std::invalid_argument("HMcode-2016: P_m and P_cb are required");
            HaloModelOptions o;
            o.prescription = Prescription::HMcode2016;
            o.kmin = kmin;
            o.kmax = kmax;
            o.npts_k = nk;
            o.accuracy_boost = accuracy_boost;
            return HaloModel(cosmology_from_background(bg, Pm, Pcb), o);
        }
    } // namespace

    HMcode2016::HMcode2016(const BackgroundCosmology & bg, LinearPower Pm, LinearPower Pcb)
        : HMcode2016(bg, std::move(Pm), std::move(Pcb), Options{}) {}

    HMcode2016::HMcode2016(const BackgroundCosmology & bg, LinearPower Pm, LinearPower Pcb, Options opt) {
        if (!(opt.z_max > 0 && opt.z_max < 1e4) || opt.n_z < 4 || opt.n_k < 4 ||
            !(opt.k_min > 0 && opt.k_max > opt.k_min))
            throw std::invalid_argument("HMcode-2016: invalid time or wavenumber table options");
        h = bg.get_params().h;
        x_min = -std::log1p(opt.z_max);
        lnk_min = std::log(opt.k_min);
        lnk_max = std::log(opt.k_max);
        auto hm = make(bg, Pm, Pcb, opt.k_min, opt.k_max, opt.n_k, opt.accuracy_boost);
        const auto xs = Utils::linspace(x_min, 0.0, opt.n_z);
        const auto lk = Utils::linspace(lnk_min, lnk_max, opt.n_k);
        std::vector<std::vector<double>> ratios;
        for (double x : xs) {
            hm.compute_at_redshift(std::max(std::expm1(-x), 0.0));
            std::vector<double> r;
            for (double l : lk) {
                const double k = std::exp(std::clamp(l, lnk_min, lnk_max));
                r.push_back(std::log(hm.pofk(k) / hm.pofk_linear(k)));
            }
            ratios.push_back(std::move(r));
        }
        log_ratio.create(xs, lk, ratios, "HMcode-2016 ln(P_NL/P_L)");
    }

    HMcode2016::Spectrum HMcode2016::spectrum(const BackgroundCosmology & bg,
                                             const LinearPower & Pm,
                                             const LinearPower & Pcb,
                                             double x,
                                             const std::vector<double> & k) {
        if (!std::isfinite(x) || x > 0.0 || k.empty())
            throw std::invalid_argument("HMcode-2016: require x <= 0 and at least one k");
        for (double kk : k)
            if (!(kk > 0) || !std::isfinite(kk))
                throw std::invalid_argument("HMcode-2016: k must be finite and positive");
        const auto [kmin, kmax] = std::minmax_element(k.begin(), k.end());
        const double lo = *kmin * 0.99, hi = *kmax * 1.01;
        auto hm = make(bg, Pm, Pcb, lo, hi, std::max(64, int(100 * std::log(hi / lo))), 1);
        hm.compute_at_redshift(std::max(std::expm1(-x), 0.0));
        Spectrum out;
        for (double kk : k) {
            out.linear.push_back(hm.pofk_linear(kk));
            out.one_halo.push_back(hm.pofk_onehalo(kk));
            out.two_halo.push_back(hm.pofk_twohalo(kk));
            out.nonlinear.push_back(hm.pofk(kk));
        }
        const auto & s = hm.state();
        out.delta_c = s.deltac;
        out.Delta_v = s.DeltaVir;
        out.sigma8_cb = s.sigma8;
        out.n_eff = s.neff;
        return out;
    }

    double HMcode2016::ratio(double x, double k) const {
        if (!(k > 0) || !std::isfinite(k) || !std::isfinite(x))
            throw std::invalid_argument("HMcode2016::ratio requires finite x and positive finite k");
        if (x < x_min || std::log(k * Constants::Mpc / h) < lnk_min)
            return 1.0;
        return std::exp(log_ratio(std::min(x, 0.0), std::min(std::log(k * Constants::Mpc / h), lnk_max)));
    }
} // namespace FML::COSMOLOGY::HALOMODEL
