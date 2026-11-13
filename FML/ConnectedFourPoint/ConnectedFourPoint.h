#ifndef FML_CONNECTED_FOUR_POINT_H
#define FML_CONNECTED_FOUR_POINT_H
#include <FML/ConnectedFourPoint/Common.h>
#include <FML/EnsembleStatistics/ControlVariates.h>
#include <optional>
// Periodic covariance/trispectrum convention: Scoccimarro, Zaldarriaga & Hui (1999),
// arXiv:astro-ph/9901099. Wick-reference and finite-sample conventions: README.md.
namespace FML::FOURPOINT {
    struct FourPointResult {
        unsigned long long samples{};
        std::array<long double, 4> means{};
        std::array<long double, 16> second_moment{};
        long double fourth{}, disconnected{}, connected{};
        bool reference_covariance_used{false};
    };
    /// Joint fourth moment of four KNOWN zero-mean variables. Each rank supplies disjoint samples.
    /// With no reference, Wick products use this realization's second moments (a biased finite-sample plug-in).
    /// Supplying the exact ensemble covariance removes that plug-in bias for known zero means.
    /// Products of noisy estimated reference covariances can themselves be biased.
    template <class Sample>
    FourPointResult zero_mean_four_point(size_t local_count,
                                         Sample sample,
                                         const std::optional<std::array<double, 16>> & reference = std::nullopt) {
        std::vector<double> schema{double(bool(reference))};
        if (reference)
            schema.insert(schema.end(), reference->begin(), reference->end());
        agree(schema);
        if (reference)
            require(detail::valid_covariance(*reference), "[FourPoint] Invalid reference covariance");
        require(local_count <= std::numeric_limits<unsigned long long>::max() / FML::NTasks,
                "[FourPoint] Sample count overflow");
        FourPointResult result;
        result.samples = local_count;
        FML::SumOverTasks(&result.samples);
        require(result.samples > 0, "[FourPoint] No samples");
        std::array<long double, 21> sums{};
        bool finite = true;
        try {
            for (size_t i = 0; i < local_count; ++i) {
                const auto row = sample(i);
                long double product = 1;
                for (int a = 0; a < 4; ++a) {
                    finite = finite && std::isfinite(row[a]);
                    sums[a] += row[a];
                    product *= row[a];
                    for (int b = 0; b < 4; ++b)
                        sums[4 + 4 * a + b] += static_cast<long double>(row[a]) * row[b];
                }
                sums[20] += product;
            }
        } catch (...) {
            finite = false;
        }
        for (auto s : sums)
            finite = finite && std::isfinite(s);
        require(finite, "[FourPoint] Nonfinite sample/moment");
        FML::SumArrayOverTasks(sums.data(), int(sums.size()));
        for (int a = 0; a < 4; ++a)
            result.means[a] = sums[a] / result.samples;
        for (int a = 0; a < 16; ++a)
            result.second_moment[a] = sums[4 + a] / result.samples;
        result.fourth = sums[20] / result.samples;
        result.reference_covariance_used = bool(reference);
        auto covariance = [&](int i, int j) -> long double {
            return reference ? (*reference)[4 * i + j] : result.second_moment[4 * i + j];
        };
        result.disconnected = covariance(0, 1) * covariance(2, 3) + covariance(0, 2) * covariance(1, 3) +
                              covariance(0, 3) * covariance(1, 2);
        result.connected = result.fourth - result.disconnected;
        require(std::isfinite(result.connected), "[FourPoint] Moment overflow");
        return result;
    }
    struct ConnectedCovariance {
        unsigned long long realizations{};
        std::vector<double> total, gaussian, connected, collapsed_trispectrum;
    };
    /// Full signed-k counts, including both members of each Hermitian pair. Disjoint periodic shells only.
    /// sum_power_squared is sum_k P_total(k)^2 from an EXTERNAL model/independent ensemble, not sum |delta|^4.
    inline std::vector<double> gaussian_covariance_disjoint_shells(const std::vector<double> & sum_power_squared,
                                                                   const std::vector<size_t> & full_mode_count) {
        const size_t n = full_mode_count.size();
        if (!n || sum_power_squared.size() != n || n > std::vector<double>().max_size() / n)
            throw std::invalid_argument("[FourPoint] Gaussian covariance dimensions");
        std::vector<double> result(n * n);
        for (size_t i = 0; i < n; ++i) {
            if (!full_mode_count[i] || !std::isfinite(sum_power_squared[i]) || sum_power_squared[i] < 0)
                throw std::invalid_argument("[FourPoint] Invalid Gaussian mode statistics");
            result[i * n + i] = double(2.L * sum_power_squared[i] / full_mode_count[i] / full_mode_count[i]);
            if (!std::isfinite(result[i * n + i]))
                throw std::overflow_error("[FourPoint] Gaussian covariance overflow");
        }
        return result;
    }
    /// Pure post-processing of an N-1 sample covariance of independent periodic boxes.
    /// Gaussian reference must match binning, units, weighting, noise and mode counting. No SSC/window model.
    inline ConnectedCovariance connected_power_covariance(const FML::ENSEMBLE::Summary & ensemble,
                                                          const std::vector<double> & gaussian,
                                                          double boxsize) {
        const size_t n = ensemble.mean.size();
        if (ensemble.count < 2 || !n || n > std::vector<double>().max_size() / n ||
            ensemble.covariance.size() != n * n || gaussian.size() != n * n || !(std::isfinite(boxsize) && boxsize > 0))
            throw std::invalid_argument("[FourPoint] Invalid covariance input");
        const double volume = boxsize * boxsize * boxsize;
        if (!(std::isfinite(volume) && volume > 0))
            throw std::invalid_argument("[FourPoint] Invalid volume");
        for (auto mean : ensemble.mean)
            if (!std::isfinite(mean))
                throw std::invalid_argument("[FourPoint] Invalid ensemble mean");
        ConnectedCovariance result{ensemble.count,
                                   ensemble.covariance,
                                   gaussian,
                                   std::vector<double>(n * n),
                                   std::vector<double>(n * n)};
        for (size_t i = 0; i < n * n; ++i) {
            if (!std::isfinite(ensemble.covariance[i]) || !std::isfinite(gaussian[i]) ||
                gaussian[i] != gaussian[(i % n) * n + i / n] ||
                ensemble.covariance[i] != ensemble.covariance[(i % n) * n + i / n] ||
                (i / n == i % n && (gaussian[i] < 0 || ensemble.covariance[i] < 0)))
                throw std::invalid_argument("[FourPoint] Invalid covariance entries");
            result.connected[i] = ensemble.covariance[i] - gaussian[i];
            result.collapsed_trispectrum[i] = volume * result.connected[i];
            if (!std::isfinite(result.collapsed_trispectrum[i]))
                throw std::overflow_error("[FourPoint] Covariance overflow");
        }
        return result;
    }
} // namespace FML::FOURPOINT
#endif
