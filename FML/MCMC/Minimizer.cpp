#include <FML/MCMC/Minimizer.h>
#include <FML/MCMC/Parallel.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace FML::MCMC {

    namespace {
        constexpr double INF = std::numeric_limits<double>::infinity();
        using Vec = std::vector<double>;
        using Objective = std::function<double(const Vec &)>;

        struct NMResult {
            Vec u;
            double f = INF;
            long evals = 0;
        };

        // Nelder-Mead from u0 with an initial simplex of size step along the axes; stops when the spread of f over the
        // vertices is < ftol (all finite), the simplex has collapsed, or after max_evals evaluations. Adaptive
        // coefficients (Gao & Han 2012) for the dimension n.
        NMResult nelder_mead(const Objective & f, const Vec & u0, double f0, double step, double ftol, long max_evals) {
            const size_t n = u0.size();
            NMResult res;
            if (n == 0) {
                res.u = u0;
                res.f = std::isfinite(f0) ? f0 : f(u0);
                res.evals = std::isfinite(f0) ? 0 : 1;
                return res;
            }
            const double dn = double(n);
            const double alpha = 1.0, beta = 1.0 + 2.0 / dn, gamma = 0.75 - 0.5 / dn,
                         delta = n > 1 ? 1.0 - 1.0 / dn : 0.5; // (1 - 1/n = 0 would collapse a 1D simplex)
            std::vector<Vec> s(n + 1, u0);
            Vec fs(n + 1);
            long ev = 0;
            auto F = [&](const Vec & u) {
                ev++;
                return f(u);
            };
            fs[0] = std::isfinite(f0) ? f0 : F(u0);
            for (size_t i = 0; i < n; i++) {
                s[i + 1][i] += step;
                fs[i + 1] = F(s[i + 1]);
                if (!std::isfinite(fs[i + 1])) { // outside the prior: try the other side
                    s[i + 1][i] = u0[i] - step;
                    fs[i + 1] = F(s[i + 1]);
                }
            }
            std::vector<size_t> idx(n + 1);
            if (std::all_of(fs.begin(), fs.end(), [](double v) { return !std::isfinite(v); })) { // nowhere valid
                res.u = u0;
                res.evals = ev;
                return res;
            }
            while (true) {
                std::iota(idx.begin(), idx.end(), 0);
                std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return fs[a] < fs[b]; });
                const size_t ib = idx[0], iw = idx[n], isw = idx[n - 1];
                double size = 0.0;
                for (size_t k = 1; k <= n; k++)
                    for (size_t j = 0; j < n; j++)
                        size = std::max(size, std::fabs(s[idx[k]][j] - s[ib][j]));
                if ((std::isfinite(fs[iw]) && fs[iw] - fs[ib] < ftol) || size < 1e-9 || ev >= max_evals)
                    break;
                Vec c(n, 0.0); // centroid of all but the worst
                for (size_t k = 0; k < n; k++)
                    for (size_t j = 0; j < n; j++)
                        c[j] += s[idx[k]][j] / dn;
                auto along = [&](double t) {
                    Vec y(n);
                    for (size_t j = 0; j < n; j++)
                        y[j] = c[j] + t * (c[j] - s[iw][j]);
                    return y;
                };
                const Vec r = along(alpha);
                const double fr = F(r);
                if (fr < fs[ib]) {
                    const Vec e = along(alpha * beta);
                    const double fe = F(e);
                    if (fe < fr) {
                        s[iw] = e;
                        fs[iw] = fe;
                    } else {
                        s[iw] = r;
                        fs[iw] = fr;
                    }
                    continue;
                }
                if (fr < fs[isw]) {
                    s[iw] = r;
                    fs[iw] = fr;
                    continue;
                }
                const bool outside = fr < fs[iw];
                const Vec k = along(outside ? alpha * gamma : -gamma);
                const double fk = F(k);
                if (fk < (outside ? fr : fs[iw])) {
                    s[iw] = k;
                    fs[iw] = fk;
                    continue;
                }
                for (size_t m = 1; m <= n; m++) { // shrink towards the best
                    const size_t i = idx[m];
                    for (size_t j = 0; j < n; j++)
                        s[i][j] = s[ib][j] + delta * (s[i][j] - s[ib][j]);
                    fs[i] = F(s[i]);
                }
            }
            const size_t ib = size_t(std::min_element(fs.begin(), fs.end()) - fs.begin());
            res.u = s[ib];
            res.f = fs[ib];
            res.evals = ev;
            return res;
        }

        // Nelder-Mead with restarts around the best point (step halved each time) until a restart gains < ftol
        NMResult nm_restarts(const Objective & f,
                             const Vec & u0,
                             double f0,
                             double step,
                             double ftol,
                             long max_evals,
                             int max_restarts,
                             bool & converged) {
            NMResult best;
            best.u = u0;
            best.f = f0;
            converged = false;
            for (int r = 0; r <= max_restarts; r++) {
                const long left = max_evals - best.evals;
                if (left <= 0)
                    break;
                NMResult t = nelder_mead(f, best.u, best.f, step, ftol, left);
                const double gain = best.f - t.f;
                t.evals += best.evals;
                if (t.f <= best.f || !std::isfinite(best.f))
                    best = t;
                else
                    best.evals = t.evals;
                if (r > 0 && std::isfinite(best.f) && gain < ftol) {
                    converged = true;
                    break;
                }
                step = std::max(0.5 * step, 0.05);
            }
            return best;
        }
    } // namespace

    Minimizer::Minimizer(const Posterior & posterior, MinimizerOptions options)
        : post(posterior), opt(std::move(options)) {
        const size_t d = post.dim();
        if (opt.starts < 1 || opt.max_evals < 1 || !(opt.tol > 0.0) || opt.max_restarts < 0)
            throw std::invalid_argument("Minimizer: invalid options");
        cov = opt.covmat.empty() ? post.diagonal_covariance() : opt.covmat;
        bool ok = cov.size() == d;
        for (const auto & r : cov)
            ok = ok && r.size() == d;
        std::vector<std::vector<double>> L;
        if (!ok || !cholesky(cov, L))
            throw std::invalid_argument("Minimizer: the covariance must be a positive-definite d x d matrix");
    }

    MinimizerResult Minimizer::minimize(const std::vector<double> & x0, const std::vector<bool> & fixed_in) const {
        const size_t d = post.dim(), ns = post.n_slow();
        std::vector<bool> fixed = fixed_in;
        fixed.resize(d, false);
        // free parameters: outer (slow) and inner (fast) sets; without slow free parameters everything is outer
        std::vector<size_t> fi;
        for (size_t i = 0; i < d; i++)
            if (!fixed[i])
                fi.push_back(i);
        if (fi.empty())
            throw std::invalid_argument("Minimizer: no free parameters");
        size_t nS = 0;
        while (nS < fi.size() && fi[nS] < ns)
            nS++;
        if (nS == 0)
            nS = fi.size();
        const size_t nf = fi.size(), nF = nf - nS;
        // whitening: Cholesky factor of the covariance of the free parameters (slow first)
        std::vector<std::vector<double>> C(nf, Vec(nf)), L;
        for (size_t a = 0; a < nf; a++)
            for (size_t b = 0; b < nf; b++)
                C[a][b] = cov[fi[a]][fi[b]];
        if (!cholesky(C, L)) {
            L.assign(nf, Vec(nf, 0.0));
            for (size_t a = 0; a < nf; a++)
                L[a][a] = post.params()[fi[a]].proposal;
        }
        auto to_x = [&](const Vec & u) { // u: all nf whitened coordinates
            Vec x = x0;
            for (size_t a = 0; a < nf; a++) {
                double v = 0.0;
                for (size_t b = 0; b <= a; b++)
                    v += L[a][b] * u[b];
                x[fi[a]] += v;
            }
            return x;
        };
        MinimizerResult best;
        auto record = [&](const Vec & x, const Posterior::Point & p) {
            if (p.valid() && p.logpost > best.logpost) {
                best.logpost = p.logpost;
                best.loglike = p.loglike;
                best.x = x;
                best.derived = p.derived;
            }
        };
        auto f_full = [&](const Vec & u) {
            const Vec x = to_x(u);
            const auto p = post.evaluate(x);
            record(x, p);
            return p.valid() ? -2.0 * p.logpost : INF;
        };
        long slow_evals = 0;
        const bool progress = opt.verbose && opt.progress_every > 0 && Parallel::rank() == 0;
        auto count = [&]() {
            slow_evals++;
            if (progress && slow_evals % opt.progress_every == 0) {
#pragma omp critical(minimizer_output)
                std::cout << "    " << slow_evals << " evaluations, best -2 ln P = " << std::setprecision(10)
                          << -2.0 * best.logpost << "\n"
                          << std::flush;
            }
        };
        Vec uF(nF, 0.0); // warm start of the inner minimization
        bool first_inner = true;
        Objective outer;
        if (nF == 0) {
            outer = [&](const Vec & u) {
                count();
                return f_full(u);
            };
        } else {
            outer = [&](const Vec & uS) {
                count();
                Vec u(nf);
                std::copy(uS.begin(), uS.end(), u.begin());
                auto inner = [&](const Vec & v) {
                    std::copy(v.begin(), v.end(), u.begin() + long(nS));
                    return f_full(u);
                };
                const double f0 = inner(uF); // the previous optimum of the fast parameters
                if (!std::isfinite(f0))
                    return INF; // slow point outside the prior or the likelihood fails
                bool conv = false;
                const NMResult r =
                    nm_restarts(inner, uF, f0, first_inner ? 1.0 : 0.3, 0.1 * opt.tol, 400 * long(nF + 1), 1, conv);
                first_inner = false;
                if (std::isfinite(r.f))
                    uF = r.u;
                return r.f;
            };
        }
        bool converged = false;
        const Vec u0(nS, 0.0);
        const NMResult r = nm_restarts(outer, u0, INF, 1.0, opt.tol, opt.max_evals, opt.max_restarts, converged);
        if (!std::isfinite(r.f))
            throw std::runtime_error("Minimizer: no valid point found (the start is outside the prior "
                                     "or the likelihood fails there)");
        best.evals = slow_evals;
        best.converged = converged;
        return best;
    }

    template <class F>
    std::vector<MinimizerResult> Minimizer::parallel_tasks(int n, F task) const {
        const int rank = Parallel::rank(), np = Parallel::size();
        std::vector<int> mine;
        for (int k = rank; k < n; k += np)
            mine.push_back(k);
        std::vector<MinimizerResult> loc(mine.size());
        std::vector<std::string> err(mine.size());
        int nt = opt.threads > 0 ? opt.threads : int(mine.size());
#ifdef _OPENMP
        nt = std::max(1, std::min(nt, omp_get_max_threads()));
#else
        nt = 1;
#endif
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (int m = 0; m < int(mine.size()); m++) {
            try {
                loc[m] = task(mine[m]);
            } catch (const std::exception & e) {
                err[m] = e.what();
            }
        }
        // Every process must throw (a throw on some processes only would leave the others in the allgather)
        std::string error;
        for (const auto & e : err)
            if (!e.empty())
                error = e;
        Vec failed(1, error.empty() ? 0.0 : 1.0);
        Parallel::allreduce_sum(failed);
        if (failed[0] > 0.0)
            throw std::runtime_error(error.empty() ? "Minimizer: a task failed on another process" : error);
        // gather: [k, logpost, loglike, evals, converged, x (d), derived (nd)] per task
        const size_t d = post.dim(), nd = post.derived_names().size();
        Vec local;
        for (size_t m = 0; m < mine.size(); m++) {
            const auto & r = loc[m];
            local.insert(local.end(),
                         {double(mine[m]), r.logpost, r.loglike, double(r.evals), r.converged ? 1.0 : 0.0});
            Vec x = r.x, dv = r.derived;
            x.resize(d, 0.0);
            dv.resize(nd, 0.0);
            local.insert(local.end(), x.begin(), x.end());
            local.insert(local.end(), dv.begin(), dv.end());
        }
        const Vec g = Parallel::allgather(local);
        std::vector<MinimizerResult> out(n);
        const size_t w = 5 + d + nd;
        for (size_t k = 0; k + w <= g.size(); k += w) {
            MinimizerResult & r = out.at(size_t(g[k]));
            r.logpost = g[k + 1];
            r.loglike = g[k + 2];
            r.evals = long(g[k + 3]);
            r.converged = g[k + 4] != 0.0;
            r.x.assign(g.begin() + long(k + 5), g.begin() + long(k + 5 + d));
            r.derived.assign(g.begin() + long(k + 5 + d), g.begin() + long(k + w));
        }
        return out;
    }

    MinimizerResult Minimizer::run() {
        const size_t d = post.dim();
        std::vector<std::vector<double>> L;
        cholesky(cov, L);
        const Vec x0 = post.start();
        const bool talk = opt.verbose && Parallel::rank() == 0;
        if (talk)
            std::cout << "Minimizer: " << opt.starts << " start(s), " << d << " parameters (" << post.n_slow()
                      << " slow), tol " << opt.tol << " in -2 ln P\n"
                      << std::flush;
        all = parallel_tasks(opt.starts, [&](int k) {
            Vec xs = x0;
            if (k > 0) { // scattered start within the prior
                std::seed_seq seq{opt.seed, std::uint64_t(k), std::uint64_t(0x51ed27e1ULL)};
                std::mt19937_64 rng(seq);
                std::normal_distribution<double> N(0.0, 1.0);
                for (int tries = 0; tries < 1000; tries++) {
                    Vec z(d), y = x0;
                    for (auto & v : z)
                        v = N(rng);
                    for (size_t i = 0; i < d; i++)
                        for (size_t j = 0; j <= i; j++)
                            y[i] += opt.start_scatter * L[i][j] * z[j];
                    if (std::isfinite(post.log_prior(y))) {
                        xs = y;
                        break;
                    }
                }
            }
            const MinimizerResult r = minimize(xs);
            if (opt.verbose && Parallel::rank() == 0) {
#pragma omp critical(minimizer_output)
                std::cout << "  start " << k + 1 << ": -2 ln P = " << std::setprecision(10) << -2.0 * r.logpost
                          << ", -2 ln L = " << -2.0 * r.loglike << " (" << r.evals << " evaluations"
                          << (r.converged ? "" : ", not converged") << ")\n"
                          << std::flush;
            }
            return r;
        });
        if (talk && Parallel::size() > 1) { // the starts of the other processes were not printed
            std::cout << "  -2 ln P of the starts:";
            for (const auto & r : all)
                std::cout << " " << std::setprecision(10) << -2.0 * r.logpost
                          << (r.converged ? "" : " (not converged)");
            std::cout << "\n" << std::flush;
        }
        size_t ib = 0;
        for (size_t k = 1; k < all.size(); k++)
            if (all[k].logpost > all[ib].logpost)
                ib = k;
        return all[ib];
    }

    std::vector<MinimizerResult>
    Minimizer::profile(size_t i, const std::vector<double> & values, const std::vector<double> & from) const {
        if (i >= post.dim())
            throw std::invalid_argument("Minimizer::profile: no such parameter");
        std::vector<bool> fixed(post.dim(), false);
        fixed[i] = true;
        const bool talk = opt.verbose && Parallel::rank() == 0;
        return parallel_tasks(int(values.size()), [&](int k) {
            Vec x = from;
            x[i] = values[size_t(k)];
            const MinimizerResult r = minimize(x, fixed);
            if (talk) {
#pragma omp critical(minimizer_output)
                std::cout << "  " << post.params()[i].name << " = " << std::setprecision(6) << values[size_t(k)]
                          << ": -2 ln P = " << std::setprecision(10) << -2.0 * r.logpost << " (" << r.evals
                          << " evaluations" << (r.converged ? "" : ", not converged") << ")\n"
                          << std::flush;
            }
            return r;
        });
    }

    std::vector<double> Minimizer::minus_logpost(const std::vector<Vec> & pts) const {
        const bool talk = opt.verbose && Parallel::rank() == 0 && Parallel::size() == 1 && pts.size() > 20;
        Vec f(pts.size() + 1, 0.0); // the last entry counts the failed points (all processes throw together)
        for (size_t k = size_t(Parallel::rank()); k < pts.size(); k += size_t(Parallel::size())) {
            const auto p = post.evaluate(pts[k]);
            if (!p.valid()) {
                f.back() += 1.0;
                continue;
            }
            f[k] = -p.logpost;
            if (talk && k % 20 == 0 && k > 0)
                std::cout << "  " << k << "/" << pts.size() << "\n" << std::flush;
        }
        Parallel::allreduce_sum(f);
        if (f.back() > 0.0)
            throw std::runtime_error(
                "Hessian: a step leaves the prior range or the likelihood fails; reduce the steps");
        f.pop_back();
        return f;
    }

    std::vector<std::vector<double>>
    Minimizer::hessian(const std::vector<double> & x0, std::vector<double> & h, bool adapt_steps) const {
        const size_t d = post.dim();
        const auto & par = post.params();
        const bool talk = opt.verbose && Parallel::rank() == 0;
        auto plus_minus = [&](std::vector<Vec> & pts) { // 1+2i, 2+2i: +-h_i
            for (size_t i = 0; i < d; i++)
                for (double sgn : {1.0, -1.0}) {
                    auto y = x0;
                    y[i] += sgn * h[i];
                    pts.push_back(y);
                }
        };
        auto fit_prior = [&]() { // steps inside the prior box
            for (size_t i = 0; i < d; i++) {
                const double room = 0.95 * std::min(x0[i] - par[i].min, par[i].max - x0[i]);
                if (!(room > 0.0))
                    throw std::runtime_error("Hessian: " + par[i].name + " is on the boundary of its prior");
                h[i] = std::min(h[i], room);
            }
        };
        fit_prior();
        double f0 = 0.0;
        bool have_f0 = false;
        if (adapt_steps) {
            const double target = 0.5 * 1.5 * 1.5; // -ln P rises by this at 1.5 conditional sigma
            for (int pass = 0; pass < 4; pass++) {
                std::vector<Vec> pts;
                if (!have_f0)
                    pts.push_back(x0);
                plus_minus(pts);
                const Vec f = minus_logpost(pts);
                const size_t o = have_f0 ? 0 : 1;
                if (!have_f0) {
                    f0 = f[0];
                    have_f0 = true;
                }
                bool done = true;
                for (size_t i = 0; i < d; i++) {
                    const double rise = 0.5 * (f[o + 2 * i] + f[o + 2 * i + 1]) - f0; // ~ H_ii h_i^2 / 2
                    const double fac = rise > 0.0 ? std::clamp(std::sqrt(target / rise), 0.1, 10.0) : 10.0;
                    if (fac < 0.7 || fac > 1.4)
                        done = false;
                    h[i] *= fac;
                }
                fit_prior();
                if (talk) {
                    std::cout << "  Hessian steps (pass " << pass + 1 << "):";
                    for (size_t i = 0; i < d; i++)
                        std::cout << " " << std::setprecision(3) << h[i];
                    std::cout << "\n" << std::flush;
                }
                if (done)
                    break;
            }
        }
        std::vector<Vec> pts; // 0: centre; 1+2i, 2+2i: +-h_i; then the 4 corners of each pair
        pts.push_back(x0);
        plus_minus(pts);
        for (size_t i = 0; i < d; i++)
            for (size_t j = i + 1; j < d; j++)
                for (double si : {1.0, -1.0})
                    for (double sj : {1.0, -1.0}) {
                        auto y = x0;
                        y[i] += si * h[i];
                        y[j] += sj * h[j];
                        pts.push_back(y);
                    }
        if (talk)
            std::cout << "Hessian: " << pts.size() << " evaluations\n" << std::flush;
        const Vec f = minus_logpost(pts);
        std::vector<Vec> H(d, Vec(d));
        size_t k = 1 + 2 * d;
        for (size_t i = 0; i < d; i++)
            H[i][i] = (f[1 + 2 * i] - 2.0 * f[0] + f[2 + 2 * i]) / (h[i] * h[i]);
        for (size_t i = 0; i < d; i++)
            for (size_t j = i + 1; j < d; j++, k += 4)
                H[i][j] = H[j][i] = (f[k] - f[k + 1] - f[k + 2] + f[k + 3]) / (4.0 * h[i] * h[j]);
        return H;
    }

    bool Minimizer::covariance(const std::vector<std::vector<double>> & H,
                               const std::vector<double> & h,
                               std::vector<std::vector<double>> & C) {
        if (spd_inverse(H, C))
            return true;
        const size_t d = H.size();
        C.assign(d, Vec(d, 0.0));
        for (size_t i = 0; i < d; i++)
            C[i][i] = H[i][i] > 0.0 ? 1.0 / H[i][i] : h[i] * h[i];
        return false;
    }

    void Minimizer::write(const std::string & root, const MinimizerResult & r) const {
        post.write_paramnames(root);
        std::ofstream f(root + ".minimum.txt");
        if (!f)
            throw std::runtime_error("cannot write " + root + ".minimum.txt");
        f << std::setprecision(10);
        f << "# maximum of the posterior: -2 ln L = " << -2.0 * r.loglike << ", -2 ln posterior = " << -2.0 * r.logpost
          << " (" << r.evals << " evaluations, " << (r.converged ? "converged" : "not converged") << ")\n";
        if (all.size() > 1) {
            f << "# -2 ln posterior of the " << all.size() << " starts:";
            for (const auto & a : all)
                f << " " << -2.0 * a.logpost;
            f << "\n";
        }
        f << "# name  value  label (derived parameters marked with *)\n";
        const auto & p = post.params();
        for (size_t i = 0; i < p.size(); i++)
            f << p[i].name << " " << r.x[i] << " " << (p[i].label.empty() ? p[i].name : p[i].label) << "\n";
        for (size_t i = 0; i < r.derived.size() && i < post.derived_names().size(); i++)
            f << post.derived_names()[i] << "* " << r.derived[i] << " " << post.derived_labels()[i] << "\n";
    }

    void Minimizer::write_profile(const std::string & file, size_t i, const std::vector<MinimizerResult> & prof) const {
        std::ofstream f(file);
        if (!f)
            throw std::runtime_error("cannot write " + file);
        f << std::setprecision(10);
        f << "# profile in " << post.params()[i].name << ": minimum of -2 ln P over the other parameters\n";
        f << "# " << post.params()[i].name << " -2lnL -2lnP converged";
        for (const auto & n : post.names())
            f << " " << n;
        f << "\n";
        for (const auto & r : prof) {
            f << r.x[i] << " " << -2.0 * r.loglike << " " << -2.0 * r.logpost << " " << (r.converged ? 1 : 0);
            for (double v : r.x)
                f << " " << v;
            for (double v : r.derived)
                f << " " << v;
            f << "\n";
        }
    }

} // namespace FML::MCMC
