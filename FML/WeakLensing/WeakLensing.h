#ifndef FML_WEAK_LENSING_HEADER
#define FML_WEAK_LENSING_HEADER
// Flat GR Born projection primitives. All distances: comoving Mpc/h; k: h/Mpc;
// P: (Mpc/h)^3. No implicit factors of h, masks, bias or shot-noise subtraction.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace FML::WEAKLENSING {
    using Function = std::function<double(double)>;
    namespace detail {
        inline void require(bool ok, const char * message) {
            if (!ok)
                throw std::invalid_argument(message);
        }
        inline double finite(double x) {
            if (!std::isfinite(x))
                throw std::domain_error("WeakLensing: nonfinite evaluation");
            return x;
        }
        inline void grid(const std::vector<double> & x, bool allow_zero = true) {
            require(x.size() >= 2, "WeakLensing: need at least two radial nodes");
            for (std::size_t i = 0; i < x.size(); ++i)
                require(std::isfinite(x[i]) && (allow_zero ? x[i] >= 0 : x[i] > 0) && (i == 0 || x[i] > x[i - 1]),
                        "WeakLensing: nodes must be finite, nonnegative and strictly increasing");
        }
        inline std::vector<double> masses(const std::vector<double> & x, const std::vector<double> & density) {
            grid(x);
            require(x.size() == density.size(), "WeakLensing: distribution size mismatch");
            std::vector<double> weights(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) {
                require(std::isfinite(density[i]) && density[i] >= 0, "WeakLensing: negative/nonfinite density");
                const double width = ((i ? x[i] - x[i - 1] : 0.) + (i + 1 < x.size() ? x[i + 1] - x[i] : 0.)) / 2;
                weights[i] = finite(density[i] * width);
            }
            return weights;
        }
    } // namespace detail

    /// Positive source distances with normalized probability masses. Sampled
    /// densities are converted to trapezoid quadrature masses, not treated as masses.
    class Sources {
        std::vector<double> chi_, probability_;
        std::vector<long double> tail_mass_, tail_inverse_;
        double normalization_ = 0;

      public:
        Sources(std::vector<double> chi, std::vector<double> masses) {
            detail::require(!chi.empty() && chi.size() == masses.size(),
                            "WeakLensing: source size mismatch/empty sources");
            std::vector<std::pair<double, double>> entries;
            long double total = 0;
            for (std::size_t i = 0; i < chi.size(); ++i) {
                detail::require(std::isfinite(chi[i]) && chi[i] >= 0 && std::isfinite(masses[i]) && masses[i] >= 0,
                                "WeakLensing: invalid source distance/mass");
                if (masses[i] == 0)
                    continue;
                detail::require(chi[i] > 0, "WeakLensing: source probability at observer");
                total += masses[i];
                entries.emplace_back(chi[i], masses[i]);
            }
            detail::require(total > 0 && total <= std::numeric_limits<double>::max(),
                            "WeakLensing: invalid source normalization");
            normalization_ = double(total);
            std::sort(entries.begin(), entries.end());
            for (const auto & entry : entries) {
                const double p = double(static_cast<long double>(entry.second) / total);
                if (!chi_.empty() && chi_.back() == entry.first)
                    probability_.back() += p;
                else {
                    chi_.push_back(entry.first);
                    probability_.push_back(p);
                }
            }
            tail_mass_.assign(chi_.size() + 1, 0);
            tail_inverse_.assign(chi_.size() + 1, 0);
            for (std::size_t j = chi_.size(); j-- > 0;) {
                tail_mass_[j] = tail_mass_[j + 1] + probability_[j];
                tail_inverse_[j] = tail_inverse_[j + 1] + static_cast<long double>(probability_[j]) / chi_[j];
            }
        }
        static Sources plane(double distance) { return Sources({distance}, {1.}); }
        static Sources from_distance_density(const std::vector<double> & chi, const std::vector<double> & n_chi) {
            return Sources(chi, detail::masses(chi, n_chi));
        }
        static Sources from_redshift_density(const std::vector<double> & z,
                                             const std::vector<double> & n_z,
                                             const Function & chi_of_z) {
            detail::require(bool(chi_of_z), "WeakLensing: missing distance callback");
            const auto weights = detail::masses(z, n_z);
            std::vector<double> chi;
            for (double value : z)
                chi.push_back(detail::finite(chi_of_z(value)));
            detail::grid(chi);
            return Sources(std::move(chi), weights);
        }
        double input_normalization() const { return normalization_; }
        const std::vector<double> & distances() const { return chi_; }
        const std::vector<double> & probabilities() const { return probability_; }
        double efficiency(double chi) const {
            detail::require(std::isfinite(chi) && chi >= 0, "WeakLensing: invalid lens distance");
            const auto i = std::size_t(std::upper_bound(chi_.begin(), chi_.end(), chi) - chi_.begin());
            return std::clamp(double(tail_mass_[i] - static_cast<long double>(chi) * tail_inverse_[i]), 0., 1.);
        }
    };

    struct FlatGeometry {
        double omega_m = .3;
        double hubble_distance = 2997.92458; // c/H0 in Mpc/h
        Function z_of_chi;
        void validate() const {
            detail::require(std::isfinite(omega_m) && omega_m >= 0 && std::isfinite(hubble_distance) &&
                                hubble_distance > 0 && bool(z_of_chi),
                            "WeakLensing: invalid flat geometry");
        }
        double redshift(double chi) const {
            const double z = detail::finite(z_of_chi(chi));
            detail::require(z >= 0, "WeakLensing: negative redshift");
            return z;
        }
        double kernel(const Sources & sources, double chi) const {
            validate();
            const double g = sources.efficiency(chi);
            if (g == 0 || omega_m == 0)
                return 0;
            return detail::finite(1.5 * omega_m / hubble_distance / hubble_distance * chi * (1 + redshift(chi)) * g);
        }
    };
    inline Function convergence_kernel(Sources sources, FlatGeometry geometry) {
        geometry.validate();
        return [sources = std::move(sources), geometry = std::move(geometry)](double chi) {
            return geometry.kernel(sources, chi);
        };
    }

    /// Fixed-order Gauss-Legendre rule; convergence is checked by refining panels/order.
    class Quadrature {
        std::vector<double> nodes_, weights_;

      public:
        explicit Quadrature(int order = 8) {
            detail::require(order >= 2 && order <= 64, "WeakLensing: quadrature order must be 2..64");
            nodes_.resize(order);
            weights_.resize(order);
            for (int i = 0; i < (order + 1) / 2; ++i) {
                double x = std::cos(std::acos(-1.) * (i + .75) / (order + .5)), derivative = 0;
                bool converged = false;
                for (int it = 0; it < 100; ++it) {
                    double p = 1, previous = 0;
                    for (int j = 1; j <= order; ++j) {
                        const double next = ((2 * j - 1) * x * p - (j - 1) * previous) / j;
                        previous = p;
                        p = next;
                    }
                    derivative = order * (x * p - previous) / (x * x - 1);
                    const double dx = p / derivative;
                    if (std::abs(dx) < 4 * std::numeric_limits<double>::epsilon()) {
                        converged = true;
                        break;
                    }
                    x -= dx;
                }
                if (!converged)
                    throw std::runtime_error("WeakLensing: quadrature root did not converge");
                nodes_[i] = -x;
                nodes_[order - 1 - i] = x;
                weights_[i] = weights_[order - 1 - i] = 2 / ((1 - x * x) * derivative * derivative);
            }
        }
        template <class F>
        double integrate(F && f, double lo, double hi) const {
            detail::require(std::isfinite(lo) && std::isfinite(hi) && hi >= lo,
                            "WeakLensing: invalid integration interval");
            if (hi == lo)
                return 0;
            const double half = (hi - lo) / 2, middle = lo + half;
            long double sum = 0;
            for (std::size_t i = 0; i < nodes_.size(); ++i)
                sum += weights_[i] * detail::finite(f(middle + half * nodes_[i]));
            return detail::finite(double(half * sum));
        }
    };
    template <class F>
    double integrate_panels(F && f, double lo, double hi, std::size_t panels = 64, int order = 8) {
        detail::require(panels > 0, "WeakLensing: need positive panel count");
        detail::require(std::isfinite(lo) && std::isfinite(hi) && hi >= lo, "WeakLensing: invalid panel interval");
        Quadrature q(order);
        long double sum = 0;
        for (std::size_t i = 0; i < panels; ++i)
            sum += q.integrate(f, lo + (hi - lo) * (double(i) / panels), lo + (hi - lo) * (double(i + 1) / panels));
        return detail::finite(double(sum));
    }

    /// Precomputed finite-shell Born weights. Maps must share angular pixels;
    /// delta is constant radially within each shell, but the geometric kernel is integrated.
    class BornPlan {
        std::vector<double> edges_;
        std::vector<std::vector<double>> weights_; // bin, shell
      public:
        BornPlan(std::vector<double> edges,
                 const std::vector<Sources> & sources,
                 const FlatGeometry & geometry,
                 int order = 8)
            : edges_(std::move(edges)) {
            detail::grid(edges_);
            geometry.validate();
            detail::require(!sources.empty(), "WeakLensing: no source bins");
            Quadrature q(order);
            for (const auto & source : sources) {
                weights_.emplace_back(edges_.size() - 1, 0.);
                for (std::size_t j = 0; j + 1 < edges_.size(); ++j) {
                    double left = edges_[j];
                    const auto add = [&](double right) {
                        weights_.back()[j] +=
                            q.integrate([&](double chi) { return geometry.kernel(source, chi); }, left, right);
                        left = right;
                    };
                    for (double chi : source.distances())
                        if (chi > left && chi < edges_[j + 1])
                            add(chi);
                    add(edges_[j + 1]);
                    detail::finite(weights_.back()[j]);
                }
            }
        }
        std::size_t bins() const { return weights_.size(); }
        std::size_t shells() const { return edges_.size() - 1; }
        const std::vector<double> & edges() const { return edges_; }
        double weight(std::size_t bin, std::size_t shell) const { return weights_.at(bin).at(shell); }
    };
    /// Rank-local owned pixels: no replication or MPI collectives. Every shell
    /// must be added once, in order, before maps() can expose the complete result.
    class BornAccumulator {
        BornPlan plan_;
        std::vector<std::vector<double>> maps_;
        std::size_t next_ = 0, pixels_;

      public:
        BornAccumulator(BornPlan plan, std::size_t local_pixels) : plan_(std::move(plan)), pixels_(local_pixels) {
            maps_.assign(plan_.bins(), std::vector<double>(pixels_, 0.));
        }
        template <class T>
        void add_shell(std::size_t shell, const T * contrast, std::size_t count) {
            detail::require(shell == next_ && shell < plan_.shells(), "WeakLensing: duplicate/out-of-order shell");
            detail::require(count == pixels_ && (count == 0 || contrast), "WeakLensing: local pixel size mismatch");
            for (std::size_t p = 0; p < count; ++p)
                detail::finite(double(contrast[p]));
            // Validate the entire update before mutating any bin (including overflow).
            for (std::size_t b = 0; b < maps_.size(); ++b)
                for (std::size_t p = 0; p < count; ++p)
                    detail::finite(maps_[b][p] + plan_.weight(b, shell) * double(contrast[p]));
            for (std::size_t b = 0; b < maps_.size(); ++b)
                for (std::size_t p = 0; p < count; ++p)
                    maps_[b][p] += plan_.weight(b, shell) * double(contrast[p]);
            ++next_;
        }
        bool complete() const { return next_ == plan_.shells(); }
        const std::vector<std::vector<double>> & maps() const {
            if (!complete())
                throw std::logic_error("WeakLensing: incomplete Born projection");
            return maps_;
        }
    };

    /// Continuous piecewise-linear foreground selection p(chi), normalized in dchi.
    class RadialSelection {
        std::vector<double> chi_, density_;
        double normalization_;

      public:
        RadialSelection(std::vector<double> chi, std::vector<double> n_chi)
            : chi_(std::move(chi)), density_(std::move(n_chi)) {
            const auto weights = detail::masses(chi_, density_);
            const long double norm = std::accumulate(weights.begin(), weights.end(), static_cast<long double>(0));
            detail::require(norm > 0 && norm <= std::numeric_limits<double>::max(),
                            "WeakLensing: invalid foreground normalization");
            normalization_ = double(norm);
            for (auto & p : density_)
                p /= normalization_;
        }
        double input_normalization() const { return normalization_; }
        const std::vector<double> & distances() const { return chi_; }
        double operator()(double chi) const {
            detail::require(std::isfinite(chi) && chi >= 0, "WeakLensing: invalid foreground distance");
            if (chi < chi_.front() || chi > chi_.back())
                return 0;
            if (chi == chi_.back())
                return density_.back();
            const auto j = std::size_t(std::upper_bound(chi_.begin(), chi_.end(), chi) - chi_.begin() - 1);
            const double t = (chi - chi_[j]) / (chi_[j + 1] - chi_[j]);
            return (1 - t) * density_[j] + t * density_[j + 1];
        }
    };
    struct PowerSpectrum {
        std::function<double(double, double)> value; // P(k,z), including bias for galaxy-matter cross power
        double k_min = 0, k_max = 0, z_min = 0, z_max = 0;
        void validate() const {
            detail::require(bool(value) && std::isfinite(k_min) && std::isfinite(k_max) && k_min >= 0 &&
                                k_max > k_min && std::isfinite(z_min) && std::isfinite(z_max) && z_min >= 0 &&
                                z_max >= z_min,
                            "WeakLensing: invalid power-spectrum domain");
        }
        double operator()(double k, double z) const {
            if (!(k >= k_min && k <= k_max && z >= z_min && z <= z_max))
                throw std::out_of_range("WeakLensing: power query outside declared k/z domain");
            return detail::finite(value(k, z));
        }
    };
    /// Extended Limber C_ell = int dchi Wa Wb P((ell+1/2)/chi,z)/chi^2.
    /// Caller supplies radial panels, including all support/interpolation knots.
    inline std::vector<double> limber_spectrum(const std::vector<double> & ells,
                                               const Function & wa,
                                               const Function & wb,
                                               const PowerSpectrum & power,
                                               const FlatGeometry & geometry,
                                               const std::vector<double> & edges,
                                               int order = 8) {
        detail::grid(edges);
        geometry.validate();
        power.validate();
        detail::require(bool(wa) && bool(wb), "WeakLensing: missing radial kernel");
        for (double ell : ells)
            detail::require(std::isfinite(ell) && ell >= 2, "WeakLensing: ell must be finite and >=2");
        Quadrature q(order);
        std::vector<double> result;
        for (double ell : ells) {
            long double cl = 0;
            for (std::size_t j = 0; j + 1 < edges.size(); ++j)
                cl += q.integrate(
                    [&](double chi) {
                        const double a = detail::finite(wa(chi)), b = detail::finite(wb(chi));
                        if (a == 0 || b == 0)
                            return 0.;
                        return a / chi * (b / chi) * power((ell + .5) / chi, geometry.redshift(chi));
                    },
                    edges[j],
                    edges[j + 1]);
            result.push_back(detail::finite(double(cl)));
        }
        return result;
    }

    /// w_gm(R; Pi_max) = 2 int_0^Pi_max xi_gm(hypot(R,Pi)) dPi [Mpc/h].
    inline double projected_galaxy_matter(double radius,
                                          double pi_max,
                                          const Function & xi_gm,
                                          std::size_t panels = 64,
                                          int order = 8) {
        detail::require(std::isfinite(radius) && radius >= 0 && std::isfinite(pi_max) && pi_max > 0 && bool(xi_gm),
                        "WeakLensing: invalid projected correlation arguments");
        return detail::finite(2 *
                              integrate_panels([&](double pi) { return xi_gm(detail::finite(std::hypot(radius, pi))); },
                                               0.,
                                               pi_max,
                                               panels,
                                               order));
    }
    /// DeltaSigma_com = rho_m0 [2/R^2 int_0^R R' w_gm(R') dR' - w_gm(R)].
    /// rho_m0: (Msun/h)/(Mpc/h)^3; result: (Msun/h)/(Mpc/h)^2.
    inline double excess_surface_density(double radius,
                                         double rho_m0,
                                         const Function & w_gm,
                                         std::size_t panels = 64,
                                         int order = 8) {
        detail::require(std::isfinite(radius) && radius > 0 && std::isfinite(rho_m0) && rho_m0 >= 0 && bool(w_gm),
                        "WeakLensing: invalid surface-density arguments");
        // Scaled u=R'/R avoids division by R^2 and handles the central limit stably.
        const double outer = detail::finite(w_gm(radius));
        const double difference =
            2 * integrate_panels([&](double u) { return u * (detail::finite(w_gm(radius * u)) - outer); },
                                 0.,
                                 1.,
                                 panels,
                                 order);
        return detail::finite(rho_m0 * difference);
    }
    /// 1/SigmaCrit_com, where four_pi_G_over_c2 is in (Mpc/h)/(Msun/h).
    /// For a single lens plane, mean gamma_t = DeltaSigma_com * this value.
    inline double
    inverse_sigma_critical_comoving(double chi_lens, double z_lens, const Sources & sources, double four_pi_G_over_c2) {
        detail::require(std::isfinite(chi_lens) && chi_lens >= 0 && std::isfinite(z_lens) && z_lens >= 0 &&
                            std::isfinite(four_pi_G_over_c2) && four_pi_G_over_c2 > 0,
                        "WeakLensing: invalid critical-density arguments");
        return detail::finite(four_pi_G_over_c2 * (1 + z_lens) * chi_lens * sources.efficiency(chi_lens));
    }
    inline double physical_surface_density(double comoving, double z_lens) {
        detail::require(std::isfinite(z_lens) && z_lens >= 0, "WeakLensing: invalid lens redshift");
        return detail::finite(comoving * (1 + z_lens) * (1 + z_lens));
    }
} // namespace FML::WEAKLENSING
#endif
