#include <FML/ReACT/ReACT.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#ifdef USE_REACT
// ABI of ACTio-ReACTio a36dda02db6b537a4ddcc7bf044c446c10a58373, pyreact/react_wrapper.cpp.
// Keep upstream headers/types and globals out of FML's public interface.
extern "C" {
    int compute_reaction_nu_ext(int *, double *, int *, double *, int *, double *,
                               int *, double *, int *, double *, int *, double *, bool *,
                               double *, double *, double *, double *, double *, double *, double *,
                               int *, double *, int *, int *,
                               int *, int *, double *, int *, int *, double *, int *, int *, double *,
                               double *, bool *, int *);
    extern char error_message[];
}
#endif

namespace FML::COSMOLOGY {
    namespace {
        void require(bool ok, const char * message) {
            if (!ok)
                throw std::invalid_argument(std::string("ReACT: ") + message);
        }
        void increasing(const std::vector<double> & values, const char * message) {
            for (size_t i = 0; i < values.size(); ++i)
                require(std::isfinite(values[i]) && (i == 0 || values[i] > values[i - 1]), message);
        }
        void grid(const ReACT::Grid & p, size_t nz, size_t nk) {
            require(p.size() == nz, "spectrum has wrong number of redshifts");
            for (const auto & row : p) {
                require(row.size() == nk, "spectrum has wrong number of wavenumbers");
                for (double v : row)
                    require(std::isfinite(v) && v > 0, "spectra must be finite and strictly positive");
            }
        }
    }

    bool ReACT::available() {
#ifdef USE_REACT
        return true;
#else
        return false;
#endif
    }

    void ReACT::validate(const Input & in, const Options & opt) {
        const auto & c = in.cosmology;
        require(std::isfinite(c.h) && c.h > 0 && std::isfinite(c.n_s), "invalid h or n_s");
        require(std::isfinite(c.Omega_m) && c.Omega_m > 0 && c.Omega_m < 1 &&
                std::isfinite(c.Omega_b) && c.Omega_b >= 0 && c.Omega_b < c.Omega_m &&
                std::isfinite(c.Omega_nu) && c.Omega_nu >= 0 && c.Omega_nu < c.Omega_m - c.Omega_b,
                "require 0 <= Omega_b, Omega_nu and Omega_b + Omega_nu < Omega_m < 1");
        require(c.A_s >= 1.5e-9 && c.A_s <= 2.5e-9 && c.k_pivot >= 1e-3 && c.k_pivot <= 0.1,
                "upstream target-spectrum interface requires 1.5e-9 <= A_s <= 2.5e-9 and 0.001 <= k_pivot <= 0.1 /Mpc");
        require(in.k.size() >= 32 && in.z.size() >= 4, "need at least 32 k points and 4 redshifts");
        require(in.k.size() <= size_t(std::numeric_limits<int>::max()) / in.z.size(), "input grid is too large");
        increasing(in.k, "k must be finite and strictly increasing");
        increasing(in.z, "z must be finite and strictly increasing");
        require(in.k.front() > 0 && in.k.front() <= 1e-4 && in.k.back() >= 50,
                "linear input must cover k <= 1e-4 through k >= 50 h/Mpc for the internal integrals");
        require(in.z.front() >= 0 && in.z.back() <= 2.5, "redshifts must lie in [0, 2.5]");
        require(opt.mass_loop >= 10 && opt.mass_loop <= 1000, "mass_loop must lie in [10, 1000]");
        const int model = int(opt.model);
        require(model >= 1 && model <= 5, "unsupported model");
        for (double p : opt.parameters)
            require(std::isfinite(p), "model parameters must be finite");
        if (opt.model == Model::HuSawicki || opt.model == Model::NormalDGP)
            require(opt.parameters[0] > 0, "fR0 or Omega_rc must be positive (use GR for the exact GR limit)");
        const int used = opt.model == Model::GR ? 0 : opt.model == Model::CPL ? 2 : 1;
        for (int i = used; i < 20; ++i)
            require(opt.parameters[i] == 0, "unused model parameters must be zero");
        grid(in.total, in.z.size(), in.k.size());
        grid(in.cold, in.z.size(), in.k.size());
        grid(in.lcdm_cold, in.z.size(), in.k.size());
    }

    ReACT::Result ReACT::compute(const Input & in, const Options & options) {
        validate(in, options);
#ifndef USE_REACT
        throw std::runtime_error("ReACT backend is not enabled; build with USE_REACT=true and REACT_LIB=<backend lib directory> (see FML/ReACT/README.md)");
#else
        // Upstream has process-global collapse, growth, PT and error tables. Serialize every backend call,
        // including error retrieval. Completed results and FML interpolation tables have no backend state.
        static std::mutex backend_mutex;
        const std::lock_guard<std::mutex> lock(backend_mutex);
        auto flatten = [](const Grid & g) {
            std::vector<double> v;
            for (const auto & row : g)
                v.insert(v.end(), row.begin(), row.end());
            return v;
        };
        auto total = flatten(in.total), cold = flatten(in.cold), lcdm = flatten(in.lcdm_cold);
        auto k = in.k, z = in.z;
        std::reverse(z.begin(), z.end()); // upstream z argument decreases; input spectra remain increasing-z
        auto c = in.cosmology;
        auto pars = options.parameters;
        int nk = int(k.size()), nz = int(z.size()), np = nk * nz, maxp = 20;
        int mass_loop = options.mass_loop, model = int(options.model), verbose = options.verbose ? 1 : 0;
        bool is_transfer = false, compute_pseudo = true;
        std::vector<double> react(np), linear(np), pseudo(np);
        double sigma8 = 0;
        error_message[0] = '\0';
        const int status = compute_reaction_nu_ext(&np, total.data(), &nk, k.data(), &nz, z.data(),
            &np, cold.data(), &np, lcdm.data(), &nk, k.data(), &is_transfer,
            &c.h, &c.n_s, &c.Omega_m, &c.Omega_b, &c.Omega_nu, &c.A_s, &c.k_pivot,
            &maxp, pars.data(), &mass_loop, &model,
            &nk, &nz, react.data(), &nk, &nz, linear.data(), &nk, &nz, pseudo.data(),
            &sigma8, &compute_pseudo, &verbose);
        // Some upstream paths set the error buffer but do not propagate an error status.
        if (status != 0 || error_message[0] != '\0')
            throw std::runtime_error(std::string("ReACT backend failed: ") +
                                     std::string(error_message, std::find(error_message, error_message + 512, '\0')));
        Result out;
        out.k = in.k;
        out.z = in.z;
        out.sigma8_at_lowest_z = sigma8;
        out.reaction = out.linear = out.pseudo = out.nonlinear = Grid(nz, std::vector<double>(nk));
        for (int iz = 0; iz < nz; ++iz)
            for (int ik = 0; ik < nk; ++ik) {
                const size_t j = size_t(ik) * nz + (nz - 1 - iz);
                for (double v : {react[j], linear[j], pseudo[j]})
                    if (!(v > 0) || !std::isfinite(v))
                        throw std::runtime_error("ReACT backend returned non-positive or non-finite power/reaction");
                out.reaction[iz][ik] = react[j];
                out.linear[iz][ik] = linear[j];
                out.pseudo[iz][ik] = pseudo[j];
                out.nonlinear[iz][ik] = react[j] * pseudo[j];
            }
        return out;
#endif
    }

    ReACT::ReACT(const Result & result) {
        require(result.k.size() >= 4 && result.z.size() >= 4, "reaction table needs at least 4 k and z points");
        increasing(result.k, "k must be finite and strictly increasing");
        increasing(result.z, "z must be finite and strictly increasing");
        require(result.k.front() > 0 && result.z.front() >= 0 && result.z.back() <= 2.5, "invalid reaction table range");
        grid(result.reaction, result.z.size(), result.k.size());
        std::vector<double> xs, lk;
        Grid lr;
        for (double kk : result.k)
            lk.push_back(std::log(kk));
        for (size_t i = result.z.size(); i-- > 0;) {
            xs.push_back(-std::log1p(result.z[i]));
            std::vector<double> row;
            for (double r : result.reaction[i])
                row.push_back(std::log(r));
            lr.push_back(std::move(row));
        }
        xmin = xs.front();
        xmax = xs.back();
        lkmin = lk.front();
        lkmax = lk.back();
        log_reaction.create(xs, lk, lr, "ReACT ln R");
    }

    double ReACT::reaction(double x, double k) const {
        require(std::isfinite(x) && std::isfinite(k) && k > 0, "need finite x and positive finite k");
        if (x < xmin || x > xmax)
            throw std::out_of_range("ReACT reaction requested outside its computed redshift range");
        if (std::log(k) < lkmin)
            return 1.0;
        return std::exp(log_reaction(x, std::min(std::log(k), lkmax)));
    }
} // namespace FML::COSMOLOGY
