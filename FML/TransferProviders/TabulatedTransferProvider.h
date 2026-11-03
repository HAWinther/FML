#ifndef FML_TABULATED_TRANSFER_PROVIDER_H
#define FML_TABULATED_TRANSFER_PROVIDER_H

#include <FML/InitialConditions/TransferProvider.h>
#include <FML/Spline/Spline.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

//=====================================================================================================================
// A TransferProvider from tables of the transfer functions at a set of redshifts (one table per output file of a
// Boltzmann code), already in the FML convention: density = delta/R (CAMB sign), velocity = theta/(a H R). The readers
// (CLASSTransferProvider, CAMBFileTransferProvider) fill a TransferTables with every species they can give, including
// the combinations CDMBaryon and TotalMatter (weighted as the code defines them), and the shared machinery is here:
//
//   - Interpolation bicubic in (log a, log k) with 4 or more redshifts (bilinear with 2-3).
//   - Growth rates dlog(delta)/dlog(a) from the derivative of the 4-point Lagrange polynomial in log a through the
//     tables around a (fewer on one side at the ends). The derivative of the bicubic interpolation is poor at the
//     ends of the table (f_cb 8% too large at the highest redshift).
//   - RangePolicy::Reject: queries outside the tables are errors. RangePolicy::SourceExtrapolation: above k_max a power
//     law in k through the last 0.1 in log k (as the old CAMBUtils reader did for CAMB tables), below k_min and
//     outside the redshifts of the tables the value at the end (growth rates: the value at the end of the redshift
//     range).
//=====================================================================================================================

namespace FML::INITIALCONDITIONS {

    /// The tables a TabulatedTransferProvider is made from
    struct TransferTables {
        using DVector = std::vector<double>;
        using DVector2D = std::vector<DVector>;
        static constexpr int nspecies = 5;
        static int index(Species s) { return static_cast<int>(s); }

        /// For messages
        std::string name{"TabulatedTransferProvider"};
        double h{0.0};
        /// The scale factors of the tables (increasing) and the k-grid in h/Mpc (increasing)
        DVector a;
        DVector k_hmpc;
        /// density[species][ia][ik] = delta/R and velocity[species][ia][ik] = theta/(aH R) (empty: not available)
        std::array<DVector2D, nspecies> density;
        std::array<DVector2D, nspecies> velocity;
        /// The N-body gauge source delta_extra/R [ia][ik] (empty: not available; see TransferProvider)
        DVector2D nbody_gauge_source;
        Gauge density_gauge{Gauge::SynchronousCDM};
        Gauge velocity_gauge{Gauge::Newtonian};
        RangePolicy range_policy{RangePolicy::Reject};
        /// The primordial spectrum: P_R(k) = A_s (k / k_pivot)^(n_s - 1)
        double A_s{0.0};
        double n_s{1.0};
        double k_pivot_mpc{0.05};
    };

    class TabulatedTransferProvider : public TransferProvider {
        using Spline2D = FML::INTERPOLATION::SPLINE::Spline2D;
        using DVector = TransferTables::DVector;

        std::string name;
        double hubble_h{0.0};
        Gauge dgauge{Gauge::SynchronousCDM};
        Gauge vgauge{Gauge::Newtonian};
        RangePolicy policy{RangePolicy::Reject};
        double A_s{0.0}, n_s{1.0}, k_pivot_mpc{0.05};
        double primordial_scale{1.0};
        TransferRange krange{};
        double logk_max{0.0};
        DVector loga_nodes;
        std::array<Spline2D, TransferTables::nspecies> d, v;
        std::array<bool, TransferTables::nspecies> have_d{}, have_v{};
        Spline2D nbody_source;
        bool have_nbody_source{false};

        /// The spline at (log a, log k), with the power-law extrapolation above k_max (SourceExtrapolation)
        double eval(const Spline2D & spline, double x, double y) const {
            if (y <= logk_max or policy != RangePolicy::SourceExtrapolation)
                return spline(x, y);
            const double dy = 0.1;
            const double T = spline(x, logk_max), T1 = spline(x, logk_max - dy);
            if (not(T / T1 > 0.0))
                return T;
            return T * std::exp(std::log(T / T1) / dy * (y - logk_max));
        }

      protected:
        TabulatedTransferProvider() = default;

        void set_tables(const TransferTables & t) {
            const std::string msg = "[" + t.name + "] ";
            const size_t na = t.a.size(), nk = t.k_hmpc.size();
            if (na < 2)
                throw std::invalid_argument(msg + "Need tables at 2 or more redshifts");
            if (nk < 4)
                throw std::invalid_argument(msg + "Need 4 or more k-values");
            if (not(t.h > 0.0) or not std::isfinite(t.h))
                throw std::invalid_argument(msg + "Invalid h");
            if (not(t.A_s > 0.0) or not std::isfinite(t.n_s) or not(t.k_pivot_mpc > 0.0))
                throw std::invalid_argument(msg + "Invalid primordial parameters A_s, n_s, k_pivot");
            for (size_t i = 0; i < na; i++)
                if (not(t.a[i] > 0.0) or (i > 0 and not(t.a[i] > t.a[i - 1])))
                    throw std::invalid_argument(msg + "The scale factors must be positive and increasing");
            for (size_t i = 0; i < nk; i++)
                if (not(t.k_hmpc[i] > 0.0) or (i > 0 and not(t.k_hmpc[i] > t.k_hmpc[i - 1])))
                    throw std::invalid_argument(msg + "The k-values must be positive and increasing");
            if (t.density[TransferTables::index(Species::CDM)].empty())
                throw std::invalid_argument(msg + "No CDM density");

            name = t.name;
            hubble_h = t.h;
            dgauge = t.density_gauge;
            vgauge = t.velocity_gauge;
            policy = t.range_policy;
            A_s = t.A_s;
            n_s = t.n_s;
            k_pivot_mpc = t.k_pivot_mpc;
            loga_nodes.resize(na);
            for (size_t i = 0; i < na; i++)
                loga_nodes[i] = std::log(t.a[i]);
            DVector logk(nk);
            for (size_t i = 0; i < nk; i++)
                logk[i] = std::log(t.k_hmpc[i]);
            logk_max = logk.back();
            krange = {t.k_hmpc.front(), t.k_hmpc.back(), t.a.front(), t.a.back()};

            const gsl_interp2d_type * type = na >= 4 ? gsl_interp2d_bicubic : gsl_interp2d_bilinear;
            const auto make = [&](const TransferTables::DVector2D & table, Spline2D & spline, const std::string & what) {
                if (table.size() != na)
                    throw std::invalid_argument(msg + what + ": one table per redshift needed");
                for (auto & row : table)
                    if (row.size() != nk)
                        throw std::invalid_argument(msg + what + ": one value per k needed");
                spline.create(loga_nodes, logk, table, t.name + " " + what, type);
            };
            for (int s = 0; s < TransferTables::nspecies; s++) {
                have_d[s] = not t.density[s].empty();
                have_v[s] = not t.velocity[s].empty();
                if (have_d[s])
                    make(t.density[s], d[s], "density " + std::to_string(s));
                if (have_v[s])
                    make(t.velocity[s], v[s], "velocity " + std::to_string(s));
            }
            have_nbody_source = not t.nbody_gauge_source.empty();
            if (have_nbody_source)
                make(t.nbody_gauge_source, nbody_source, "N-body gauge source");
        }

      public:
        double h() const override { return hubble_h; }
        TransferRange range() const override { return krange; }
        Gauge density_gauge() const override { return dgauge; }
        Gauge velocity_gauge() const override { return vgauge; }
        RangePolicy range_policy() const override { return policy; }
        bool has_density(Species s) const override { return have_d[TransferTables::index(s)]; }
        bool has_velocity(Species s) const override { return have_v[TransferTables::index(s)]; }

        double density_transfer(Species s, double k, double a) const override {
            check_query(s, k, a);
            return eval(d[TransferTables::index(s)], std::log(a), std::log(k));
        }

        /// dlog(delta)/dlog(a): the derivative of the Lagrange polynomial in log(a) through the 4 tables around a
        /// (fewer on one side at the ends of the table). With 2-3 tables: the slope between the two around a. Outside
        /// the redshift range: the value at the end
        double density_growth_rate(Species s, double k, double a) const override {
            check_query(s, k, a);
            const auto & spline = d[TransferTables::index(s)];
            const double y = std::log(k);
            const double x0 = std::min(std::max(std::log(a), loga_nodes.front()), loga_nodes.back());
            const int n = int(loga_nodes.size());
            double value = 0.0, deriv = 0.0;
            if (n < 4) {
                int j = int(std::upper_bound(loga_nodes.begin(), loga_nodes.end(), x0) - loga_nodes.begin()) - 1;
                j = std::max(0, std::min(j, n - 2));
                const double d0 = eval(spline, loga_nodes[j], y), d1 = eval(spline, loga_nodes[j + 1], y);
                const double t = (x0 - loga_nodes[j]) / (loga_nodes[j + 1] - loga_nodes[j]);
                value = d0 + t * (d1 - d0);
                deriv = (d1 - d0) / (loga_nodes[j + 1] - loga_nodes[j]);
            } else {
                int j = int(std::upper_bound(loga_nodes.begin(), loga_nodes.end(), x0) - loga_nodes.begin()) - 2;
                j = std::max(0, std::min(j, n - 4));
                for (int i = 0; i < 4; i++) {
                    const double xi = loga_nodes[j + i], di = eval(spline, xi, y);
                    double li = 1.0, dli = 0.0;
                    for (int m = 0; m < 4; m++) {
                        if (m == i)
                            continue;
                        const double xm = loga_nodes[j + m];
                        const double fac = (x0 - xm) / (xi - xm);
                        dli = dli * fac + li / (xi - xm);
                        li *= fac;
                    }
                    value += li * di;
                    deriv += dli * di;
                }
            }
            if (value == 0.0)
                throw std::domain_error("[" + name + "] Undefined density logarithmic derivative");
            return deriv / value;
        }

        double velocity_divergence_transfer(Species s, double k, double a) const override {
            check_query(s, k, a, true);
            return eval(v[TransferTables::index(s)], std::log(a), std::log(k));
        }

        bool has_nbody_gauge_source() const override { return have_nbody_source; }
        double nbody_gauge_source(double k, double a) const override {
            if (not have_nbody_source)
                return TransferProvider::nbody_gauge_source(k, a);
            if (not(k > 0) or not(a > 0))
                throw std::invalid_argument("[" + name + "] k_hmpc and a must be positive");
            if (policy == RangePolicy::Reject and not krange.contains(k, a))
                throw std::out_of_range("[" + name + "] Query outside the supported k/a range");
            return eval(nbody_source, std::log(a), std::log(k));
        }

        double primordial_power(double k) const override {
            if (not(k > 0) or not std::isfinite(k))
                throw std::invalid_argument("[" + name + "] Invalid k_hmpc");
            return primordial_scale * 2.0 * M_PI * M_PI / (k * k * k) * A_s *
                   std::pow(k * hubble_h / k_pivot_mpc, n_s - 1.0);
        }
        void rescale_primordial_power(double factor) override {
            if (not(factor > 0) or not std::isfinite(factor))
                throw std::invalid_argument("[" + name + "] Invalid primordial power rescaling");
            primordial_scale *= factor;
        }
    };

    /// A TabulatedTransferProvider from given tables
    class TransferTablesProvider final : public TabulatedTransferProvider {
      public:
        explicit TransferTablesProvider(const TransferTables & tables) { set_tables(tables); }
    };

    /// The tables of another provider (e.g. the internal solver) at the scale factors a and the k-values k_hmpc (inside
    /// its range): every species it has, in its gauges. The primordial spectrum A_s (k / k_pivot)^(n_s - 1) is given
    /// (the source's own rescaling is not carried over). With range_policy the result behaves outside the tables as
    /// the CAMB and CLASS tables do.
    inline TransferTables tabulate_transfers(const TransferProvider & source,
                                             const std::vector<double> & a,
                                             const std::vector<double> & k_hmpc,
                                             double A_s,
                                             double n_s,
                                             double k_pivot_mpc,
                                             RangePolicy range_policy,
                                             const std::string & name = "TabulatedTransferProvider") {
        TransferTables t;
        t.name = name;
        t.h = source.h();
        t.a = a;
        t.k_hmpc = k_hmpc;
        t.density_gauge = source.density_gauge();
        t.velocity_gauge = source.velocity_gauge();
        t.range_policy = range_policy;
        t.A_s = A_s;
        t.n_s = n_s;
        t.k_pivot_mpc = k_pivot_mpc;
        for (auto s : {Species::CDM, Species::Baryon, Species::CDMBaryon, Species::TotalMatter, Species::MassiveNeutrino}) {
            const int i = TransferTables::index(s);
            const bool dens = source.has_density(s), vel = source.has_velocity(s);
            if (dens)
                t.density[i].assign(a.size(), TransferTables::DVector(k_hmpc.size()));
            if (vel)
                t.velocity[i].assign(a.size(), TransferTables::DVector(k_hmpc.size()));
            for (size_t ia = 0; ia < a.size(); ia++)
                for (size_t ik = 0; ik < k_hmpc.size(); ik++) {
                    if (dens)
                        t.density[i][ia][ik] = source.density_transfer(s, k_hmpc[ik], a[ia]);
                    if (vel)
                        t.velocity[i][ia][ik] = source.velocity_divergence_transfer(s, k_hmpc[ik], a[ia]);
                }
        }
        if (source.has_nbody_gauge_source()) {
            t.nbody_gauge_source.assign(a.size(), TransferTables::DVector(k_hmpc.size()));
            for (size_t ia = 0; ia < a.size(); ia++)
                for (size_t ik = 0; ik < k_hmpc.size(); ik++)
                    t.nbody_gauge_source[ia][ik] = source.nbody_gauge_source(k_hmpc[ik], a[ia]);
        }
        return t;
    }

} // namespace FML::INITIALCONDITIONS
#endif
