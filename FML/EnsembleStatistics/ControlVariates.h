#ifndef FML_CONTROL_VARIATES_HEADER
#define FML_CONTROL_VARIATES_HEADER
#include <FML/Global/Global.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>
namespace FML::ENSEMBLE {
    using Vector = std::vector<double>;
    using Count = unsigned long long;
    namespace detail {
        inline void require(bool ok, const char* text) {
            if (!ok)
                throw std::invalid_argument(text);
        }
        inline bool finite(const Vector& x) {
            for (double v : x)
                if (!std::isfinite(v) || std::abs(v) > 1e100)
                    return false;
            return true;
        }
        inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }
        inline void same(Count hash) {
            Count lo = hash, hi = hash;
            FML::MinOverTasks(&lo);
            FML::MaxOverTasks(&hi);
            require(lo == hi, "Ensemble MPI schemas/models differ across ranks");
        }
        inline void word(Count& h, Count x) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        inline void text(Count& h, const std::string& s) {
            word(h, s.size());
            for (unsigned char c : s)
                word(h, c);
        }
        inline void values(Count& h, const Vector& a) {
            word(h, a.size());
            for (double x : a) {
                std::uint64_t bits;
                std::memcpy(&bits, &x, sizeof(bits));
                word(h, bits);
            }
        }
        template <class T> void sum(std::vector<T>& a) {
#ifdef USE_MPI
            for (size_t offset = 0; offset < a.size();) {
                int n = int(std::min(size_t(INT_MAX), a.size() - offset));
                MPI_Allreduce(MPI_IN_PLACE, a.data() + offset, n, FML::get_mpi_type<T>(), MPI_SUM, FML::communicator);
                offset += n;
            }
#else
            (void)a;
#endif
        }
        template <class T> void broadcast(std::vector<T>& a, int root) {
#ifdef USE_MPI
            for (size_t offset = 0; offset < a.size();) {
                int n = int(std::min(size_t(INT_MAX), a.size() - offset));
                MPI_Bcast(a.data() + offset, n, FML::get_mpi_type<T>(), root, FML::communicator);
                offset += n;
            }
#else
            (void)a;
            (void)root;
#endif
        }
    } // namespace detail
    struct Summary {
        Count count = 0;
        Vector mean, covariance;
    }; // covariance is row-major, sample normalization.
    class RunningCovariance {
        size_t d;
        Count n = 0;
        std::vector<long double> mean, m2;

      public:
        explicit RunningCovariance(size_t dimension) : d(dimension) {
            detail::require(d > 0 && d <= std::vector<long double>().max_size() / d, "Invalid ensemble dimension");
            mean.resize(d);
            m2.resize(d * d);
        }
        size_t dimension() const { return d; }
        Count count() const { return n; }
        void reset() {
            n = 0;
            std::fill(mean.begin(), mean.end(), 0);
            std::fill(m2.begin(), m2.end(), 0);
        }
        void add(const Vector& x) {
            detail::require(x.size() == d && detail::finite(x),
                            "Ensemble row has wrong dimension/nonfinite or excessive values");
            if (n == std::numeric_limits<Count>::max())
                throw std::overflow_error("Ensemble count overflow");
            std::vector<long double> delta(d);
            for (size_t i = 0; i < d; ++i)
                delta[i] = static_cast<long double>(x[i]) - mean[i];
            const long double next = static_cast<long double>(n) + 1, weight = static_cast<long double>(n) / next;
            for (size_t i = 0; i < d; ++i) {
                mean[i] += delta[i] / next;
                for (size_t j = 0; j <= i; ++j) {
                    m2[i * d + j] += weight * delta[i] * delta[j];
                    m2[j * d + i] = m2[i * d + j];
                }
            }
            ++n;
        }
        void merge(const RunningCovariance& other) {
            detail::require(d == other.d, "Cannot merge ensembles with different dimensions");
            if (other.n > std::numeric_limits<Count>::max() - n)
                throw std::overflow_error("Ensemble merge count overflow");
            if (other.n == 0)
                return;
            if (n == 0) {
                *this = other;
                return;
            }
            const Count total = n + other.n;
            std::vector<long double> delta(d);
            for (size_t i = 0; i < d; ++i)
                delta[i] = other.mean[i] - mean[i];
            const long double weight = static_cast<long double>(n) * other.n / total;
            for (size_t i = 0; i < d; ++i) {
                mean[i] += delta[i] * other.n / total;
                for (size_t j = 0; j <= i; ++j) {
                    m2[i * d + j] += other.m2[i * d + j] + weight * delta[i] * delta[j];
                    m2[j * d + i] = m2[i * d + j];
                }
            }
            n = total;
        }
        Summary summary() const {
            Summary s;
            s.count = n;
            s.mean.resize(d);
            s.covariance.resize(d * d);
            for (size_t i = 0; i < d; ++i)
                s.mean[i] = n ? double(mean[i]) : detail::nan();
            for (size_t i = 0; i < d * d; ++i)
                s.covariance[i] = n > 1 ? double(m2[i] / (n - 1)) : detail::nan();
            return s;
        }
        /// Collective, nonmutating two-pass merge. Samples themselves are never gathered.
        RunningCovariance global() const {
            detail::same(d);
            Count local_count = n;
            const auto counts = FML::GatherFromTasks(&local_count);
            Count total = 0;
            int first = -1;
            for (int t = 0; t < FML::NTasks; ++t) {
                if (counts[t] && first < 0)
                    first = t;
                if (counts[t] > std::numeric_limits<Count>::max() - total)
                    throw std::overflow_error("Global ensemble count overflow");
                total += counts[t];
            }
            RunningCovariance result(d);
            if (!total)
                return result;
            auto reference = mean;
            detail::broadcast(reference, first);
            std::vector<long double> shift(d);
            if (n)
                for (size_t i = 0; i < d; ++i)
                    shift[i] = (mean[i] - reference[i]) * n;
            detail::sum(shift);
            for (size_t i = 0; i < d; ++i)
                result.mean[i] = reference[i] + shift[i] / total;
            auto centered = m2;
            if (n)
                for (size_t i = 0; i < d; ++i)
                    for (size_t j = 0; j < d; ++j)
                        centered[i * d + j] += (mean[i] - result.mean[i]) * (mean[j] - result.mean[j]) * n;
            detail::sum(centered);
            result.m2 = std::move(centered);
            result.n = total;
            return result;
        }
    };
    enum class PhaseGenerator { Philox, NGenIC, External };
    struct Metadata {
        std::string cosmology, observable_schema, target_model, surrogate_model;
        bool fixed_amplitudes = false, phase_pair_averaged = false;
        PhaseGenerator phase_generator = PhaseGenerator::Philox;
        Count fingerprint() const {
            Count h = 1469598103934665603ULL;
            detail::text(h, cosmology);
            detail::text(h, observable_schema);
            detail::text(h, target_model);
            detail::text(h, surrogate_model);
            detail::word(h, fixed_amplitudes);
            detail::word(h, phase_pair_averaged);
            detail::word(h, Count(phase_generator));
            return h;
        }
        bool operator==(const Metadata& other) const {
            return std::tie(cosmology, observable_schema, target_model, surrogate_model, fixed_amplitudes,
                            phase_pair_averaged, phase_generator) ==
                   std::tie(other.cosmology, other.observable_schema, other.target_model, other.surrogate_model,
                            other.fixed_amplitudes, other.phase_pair_averaged, other.phase_generator);
        }
        void validate() const {
            detail::require(
                !cosmology.empty() && !observable_schema.empty() && !target_model.empty() && !surrogate_model.empty() &&
                    (phase_generator == PhaseGenerator::Philox || phase_generator == PhaseGenerator::NGenIC ||
                     phase_generator == PhaseGenerator::External),
                "Ensemble metadata tags must be explicit");
        }
    };
    struct PhaseKey {
        std::uint64_t realization = 0, seed = 0;
        std::uint32_t stream = 0;
        bool fixed_amplitudes = false;
        PhaseGenerator generator = PhaseGenerator::Philox;
        int phase_sign = 1; // +/-1: one realization; 0: an already averaged antithetic pair.
        bool operator==(const PhaseKey& k) const {
            return std::tie(realization, seed, stream, fixed_amplitudes, generator, phase_sign) ==
                   std::tie(k.realization, k.seed, k.stream, k.fixed_amplitudes, k.generator, k.phase_sign);
        }
    };
    namespace detail {
        class Registry {
            std::map<std::uint64_t, PhaseKey> ids;
            std::set<std::pair<std::uint64_t, std::uint32_t>> phases;

          public:
            size_t size() const { return ids.size(); }
            const auto& keys() const { return ids; }
            bool overlaps(const PhaseKey& k) const {
                return ids.count(k.realization) || phases.count({k.seed, k.stream});
            }
            void validate(const PhaseKey& key, const Metadata& m) const {
                require(
                    key.fixed_amplitudes == m.fixed_amplitudes && key.generator == m.phase_generator &&
                        (m.phase_pair_averaged ? key.phase_sign == 0 : (key.phase_sign == 1 || key.phase_sign == -1)),
                    "Phase key disagrees with ensemble sampling convention");
            }
            void add(const PhaseKey& key, const Metadata& m) {
                validate(key, m);
                require(!overlaps(key), "Duplicate realization or phase block in ensemble");
                ids.emplace(key.realization, key);
                phases.emplace(key.seed, key.stream);
            }
            void disjoint(const Registry& other) const {
                for (const auto& item : other.ids)
                    require(!overlaps(item.second), "Training, mean and evaluation phase blocks must be independent");
            }
            Registry global(const Metadata& metadata) const {
                Count local = ids.size();
                const auto counts = FML::GatherFromTasks(&local);
                Registry result;
                bool duplicate = false;
                for (int rank = 0; rank < FML::NTasks; ++rank) {
                    require(counts[rank] <= std::vector<Count>().max_size() / 4,
                            "Ensemble identity list exceeds addressable storage");
                    std::vector<Count> data(size_t(counts[rank]) * 4);
                    if (FML::ThisTask == rank) {
                        size_t i = 0;
                        for (const auto& item : ids) {
                            const auto& k = item.second;
                            data[i++] = k.realization;
                            data[i++] = k.seed;
                            data[i++] = k.stream;
                            data[i++] =
                                (Count(k.generator) << 3) + (k.fixed_amplitudes ? 4 : 0) + Count(k.phase_sign + 1);
                        }
                    }
                    broadcast(data, rank);
                    for (size_t i = 0; i < data.size(); i += 4) {
                        PhaseKey k{data[i],
                                   data[i + 1],
                                   std::uint32_t(data[i + 2]),
                                   bool(data[i + 3] & 4),
                                   PhaseGenerator(data[i + 3] >> 3),
                                   int(data[i + 3] & 3) - 1};
                        if (result.overlaps(k))
                            duplicate = true;
                        else
                            result.add(k, metadata);
                    }
                }
                require(!duplicate, "Duplicate realization/phase block across MPI ranks");
                return result;
            }
        };
    } // namespace detail
    class MeanEstimate {
        Metadata metadata_;
        Vector mean_, covariance_;
        detail::Registry identities_;
        Count count_ = 0;

      public:
        static MeanEstimate exact(Vector mean, const Metadata& metadata) {
            metadata.validate();
            detail::require(!mean.empty() && mean.size() <= std::vector<double>().max_size() / mean.size() &&
                                detail::finite(mean),
                            "Invalid exact surrogate mean");
            MeanEstimate m;
            m.metadata_ = metadata;
            m.mean_ = std::move(mean);
            m.covariance_.resize(m.mean_.size() * m.mean_.size());
            return m;
        }
        static MeanEstimate independent(const RunningCovariance& controls, const std::vector<PhaseKey>& keys,
                                        const Metadata& metadata) {
            metadata.validate();
            const auto summary = controls.summary();
            detail::require(summary.count >= 2 && summary.count == keys.size(),
                            "Independent mean needs at least two samples and complete phase provenance");
            auto m = exact(summary.mean, metadata);
            for (const auto& k : keys)
                m.identities_.add(k, metadata);
            m.count_ = summary.count;
            m.covariance_ = summary.covariance;
            for (auto& x : m.covariance_)
                x /= double(summary.count);
            return m;
        }
        const Vector& mean() const { return mean_; }
        const Vector& covariance_of_mean() const { return covariance_; }
        Count samples() const { return count_; }
        const Metadata& metadata() const { return metadata_; }
        const detail::Registry& identities() const { return identities_; }
    };
    enum class FitMode { Full, Diagonal };
    struct FitOptions {
        FitMode mode = FitMode::Full;
        double ridge = 1e-8;
    }; // Ridge in standardized-control units.
    class ControlModel {
        size_t ny_ = 0, nc_ = 0;
        Vector beta_;
        MeanEstimate control_mean_;
        detail::Registry training_;
        Metadata metadata_;
        Count training_count_ = 0;
        FitOptions options_;
        friend class PairedTraining;
        ControlModel() = default;

      public:
        ControlModel(const ControlModel&) = default;
        ControlModel(ControlModel&&) = default;
        ControlModel& operator=(const ControlModel&) = default;
        ControlModel& operator=(ControlModel&&) = default;
        size_t target_dimension() const { return ny_; }
        size_t control_dimension() const { return nc_; }
        const Vector& coefficients() const { return beta_; }
        const MeanEstimate& control_mean() const { return control_mean_; }
        const Metadata& metadata() const { return metadata_; }
        Count training_samples() const { return training_count_; }
        FitOptions fit_options() const { return options_; }
        void validate_pair(const PhaseKey& target, const PhaseKey& control) const {
            detail::require(target == control, "Target and surrogate must use the same phase key");
            training_.validate(target, metadata_);
            detail::require(!training_.overlaps(target) && !control_mean_.identities().overlaps(target),
                            "Evaluation overlaps training or surrogate-mean phases");
        }
        Vector apply(const Vector& target, const Vector& control, const PhaseKey& target_key,
                     const PhaseKey& control_key) const {
            detail::require(target.size() == ny_ && control.size() == nc_ && detail::finite(target) &&
                                detail::finite(control),
                            "Invalid control-variate evaluation vectors");
            validate_pair(target_key, control_key);
            Vector result = target;
            for (size_t i = 0; i < ny_; ++i) {
                long double value = target[i];
                for (size_t j = 0; j < nc_; ++j)
                    value -= static_cast<long double>(beta_[i * nc_ + j]) *
                             (static_cast<long double>(control[j]) - control_mean_.mean()[j]);
                if (!std::isfinite(value) || std::abs(value) > 1e100)
                    throw std::overflow_error("Controlled observable exceeds supported range");
                result[i] = double(value);
            }
            return result;
        }
        Vector mean_uncertainty() const {
            Vector out(ny_ * ny_);
            const auto& c = control_mean_.covariance_of_mean();
            // Two matrix products avoid O(ny^2 nc^2) contraction.
            std::vector<long double> intermediate(ny_ * nc_);
            for (size_t i = 0; i < ny_; ++i)
                for (size_t b = 0; b < nc_; ++b)
                    for (size_t a = 0; a < nc_; ++a)
                        intermediate[i * nc_ + b] += static_cast<long double>(beta_[i * nc_ + a]) * c[a * nc_ + b];
            for (size_t i = 0; i < ny_; ++i)
                for (size_t j = 0; j < ny_; ++j) {
                    long double value = 0;
                    for (size_t b = 0; b < nc_; ++b)
                        value += intermediate[i * nc_ + b] * beta_[j * nc_ + b];
                    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
                        throw std::overflow_error("Surrogate-mean uncertainty overflows");
                    out[i * ny_ + j] = double(value);
                }
            return out;
        }
        Count fingerprint() const {
            Count h = metadata_.fingerprint();
            detail::word(h, ny_);
            detail::word(h, nc_);
            detail::word(h, training_count_);
            detail::word(h, Count(options_.mode));
            detail::values(h, {options_.ridge});
            detail::values(h, beta_);
            detail::values(h, control_mean_.mean());
            detail::values(h, control_mean_.covariance_of_mean());
            for (const auto& x : training_.keys()) {
                detail::word(h, x.second.realization);
                detail::word(h, x.second.seed);
                detail::word(h, x.second.stream);
                detail::word(h, Count(x.second.phase_sign + 1));
            }
            for (const auto& x : control_mean_.identities().keys()) {
                detail::word(h, x.second.realization);
                detail::word(h, x.second.seed);
                detail::word(h, x.second.stream);
                detail::word(h, Count(x.second.phase_sign + 1));
            }
            return h;
        }
    };
    class PairedTraining {
        size_t ny, nc;
        Metadata metadata_;
        RunningCovariance joint;
        detail::Registry identities;

      public:
        static size_t checked_dimension(size_t targets, size_t controls) {
            detail::require(targets > 0 && controls > 0 && targets <= std::numeric_limits<size_t>::max() - controls,
                            "Invalid paired ensemble dimensions");
            return targets + controls;
        }
        PairedTraining(size_t targets, size_t controls, Metadata metadata)
            : ny(targets), nc(controls), metadata_(std::move(metadata)), joint(checked_dimension(targets, controls)) {
            detail::require(ny > 0 && nc > 0 && ny <= std::numeric_limits<size_t>::max() - nc,
                            "Invalid paired ensemble dimensions");
            metadata_.validate();
        }
        void add(const Vector& y, const Vector& c, const PhaseKey& target, const PhaseKey& control) {
            detail::require(target == control, "Training target/control phase keys do not match");
            detail::require(y.size() == ny && c.size() == nc && detail::finite(y) && detail::finite(c),
                            "Invalid paired training vectors");
            identities.add(target, metadata_);
            Vector row = y;
            row.insert(row.end(), c.begin(), c.end());
            joint.add(row);
        }
        PairedTraining global() const {
            Count hash = metadata_.fingerprint();
            detail::word(hash, ny);
            detail::word(hash, nc);
            detail::same(hash);
            auto result = *this;
            result.identities = identities.global(metadata_);
            result.joint = joint.global();
            return result;
        }
        Summary summary() const { return joint.summary(); }
        ControlModel fit(const MeanEstimate& mean, const FitOptions& options = {}) const {
            const auto s = joint.summary();
            detail::require(s.count >= 2 && mean.mean().size() == nc && mean.metadata() == metadata_ &&
                                options.ridge >= 0 && std::isfinite(options.ridge) &&
                                (options.mode == FitMode::Full || options.mode == FitMode::Diagonal),
                            "Invalid training data, surrogate mean or fit options");
            identities.disjoint(mean.identities());
            ControlModel model;
            model.ny_ = ny;
            model.nc_ = nc;
            model.beta_.resize(ny * nc);
            model.control_mean_ = mean;
            model.training_ = identities;
            model.metadata_ = metadata_;
            model.training_count_ = s.count;
            model.options_ = options;
            const size_t d = ny + nc;
            if (options.mode == FitMode::Diagonal) {
                detail::require(ny == nc, "Diagonal control variates need matching target/control widths");
                for (size_t i = 0; i < ny; ++i) {
                    const double variance = s.covariance[(ny + i) * d + ny + i];
                    if (variance > 0)
                        model.beta_[i * nc + i] = s.covariance[i * d + ny + i] / variance / (1 + options.ridge);
                }
                return model;
            }
            std::vector<size_t> active;
            std::vector<long double> scale(nc);
            for (size_t i = 0; i < nc; ++i)
                if (s.covariance[(ny + i) * d + ny + i] > 0) {
                    active.push_back(i);
                    scale[i] = std::sqrt(static_cast<long double>(s.covariance[(ny + i) * d + ny + i]));
                }
            const size_t n = active.size();
            std::vector<long double> lower(n * n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j <= i; ++j) {
                    long double value =
                        s.covariance[(ny + active[i]) * d + ny + active[j]] / (scale[active[i]] * scale[active[j]]);
                    if (i == j)
                        value += options.ridge;
                    for (size_t k = 0; k < j; ++k)
                        value -= lower[i * n + k] * lower[j * n + k];
                    if (i == j) {
                        if (!(value > 64 * std::numeric_limits<long double>::epsilon() * std::max(size_t(1), n)))
                            throw std::domain_error("Singular control covariance: use positive ridge regularization");
                        lower[i * n + j] = std::sqrt(value);
                    } else
                        lower[i * n + j] = value / lower[j * n + j];
                }
            for (size_t row = 0; row < ny; ++row) {
                std::vector<long double> solution(n);
                for (size_t i = 0; i < n; ++i) {
                    long double value = s.covariance[row * d + ny + active[i]] / scale[active[i]];
                    for (size_t j = 0; j < i; ++j)
                        value -= lower[i * n + j] * solution[j];
                    solution[i] = value / lower[i * n + i];
                }
                for (size_t k = n; k > 0; --k) {
                    size_t i = k - 1;
                    for (size_t j = i + 1; j < n; ++j)
                        solution[i] -= lower[j * n + i] * solution[j];
                    solution[i] /= lower[i * n + i];
                }
                for (size_t i = 0; i < n; ++i) {
                    const long double b = solution[i] / scale[active[i]];
                    if (!std::isfinite(b) || std::abs(b) > std::numeric_limits<double>::max())
                        throw std::overflow_error("Control coefficient overflows");
                    model.beta_[row * nc + active[i]] = double(b);
                }
            }
            return model;
        }
    };
    struct Estimate {
        Count count = 0;
        Vector mean, sample_covariance, covariance_of_mean;
        Metadata metadata;
    };
    class ControlEvaluation {
        ControlModel model_;
        RunningCovariance corrected;
        detail::Registry identities;

      public:
        explicit ControlEvaluation(ControlModel model)
            : model_(std::move(model)), corrected(model_.target_dimension()) {}
        void add(const Vector& y, const Vector& c, const PhaseKey& target, const PhaseKey& control) {
            auto value = model_.apply(y, c, target, control);
            identities.add(target, model_.metadata());
            corrected.add(value);
        }
        ControlEvaluation global() const {
            detail::same(model_.fingerprint());
            auto out = *this;
            out.identities = identities.global(model_.metadata());
            out.corrected = corrected.global();
            return out;
        }
        Estimate estimate() const {
            const auto s = corrected.summary();
            Estimate out;
            out.count = s.count;
            out.mean = s.mean;
            out.sample_covariance = s.covariance;
            out.covariance_of_mean = model_.mean_uncertainty();
            out.metadata = model_.metadata();
            for (size_t i = 0; i < out.covariance_of_mean.size(); ++i)
                out.covariance_of_mean[i] += s.count > 1 ? s.covariance[i] / double(s.count) : detail::nan();
            return out;
        }
    };
} // namespace FML::ENSEMBLE
#endif
