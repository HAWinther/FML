#ifndef FML_POLYSPECTRUM_WORKSPACE_H
#define FML_POLYSPECTRUM_WORKSPACE_H
#include <FML/ComputePowerSpectra/PolyspectrumBinning.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/MPICounts.h>
#include <algorithm>
#include <array>
#include <climits>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
namespace FML::CORRELATIONFUNCTIONS::detail {
    template <int N, int ORDER>
    class ShellCache {
        using Grid = FML::GRID::FFTWGrid<N>;
        struct Entry {
            int bin{-1};
            size_t age{0};
            Grid grid;
        };
        std::vector<Entry> entries;
        std::function<Grid(int)> build;
        PolyspectrumStatistics & stats;
        size_t & builds;
        size_t clock{0};

      public:
        ShellCache(int nbins,
                   int mesh,
                   const std::vector<std::array<int, ORDER>> & configs,
                   const PolyspectrumOptions & options,
                   PolyspectrumStatistics & statistics,
                   size_t & shell_builds,
                   std::function<Grid(int)> builder)
            : build(std::move(builder)), stats(statistics), builds(shell_builds) {
            size_t bytes = Grid::storage_bytes(mesh);
            FML::MaxOverTasks(&bytes);
            stats.shell_grid_bytes = bytes;
            size_t capacity =
                options.max_cached_shells ? std::min(size_t(nbins), options.max_cached_shells) : size_t(nbins);
            if (options.shell_memory_budget_bytes)
                capacity = std::min(capacity, options.shell_memory_budget_bytes / bytes);
            // Every rank must use the same eviction schedule and FFT collectives.
            FML::MinOverTasks(&capacity);
            size_t needed = 0;
            for (auto cfg : configs)
                needed = std::max(needed, size_t(std::unique(cfg.begin(), cfg.end()) - cfg.begin()));
            if (capacity < needed)
                throw std::invalid_argument(
                    "[polyspectrum] Shell cache budget cannot hold one requested configuration");
            entries.resize(capacity);
            stats.memory.cache_metadata_bytes =
                std::max(stats.memory.cache_metadata_bytes, sizeof(*this) + entries.capacity() * sizeof(Entry));
        }
        void track_reduction_storage(size_t bytes) {
            stats.memory.reduction_bytes = std::max(stats.memory.reduction_bytes, bytes);
        }
        std::array<const Grid *, ORDER> get(const std::array<int, ORDER> & cfg) {
            std::array<const Grid *, ORDER> result{};
            for (int d = 0; d < ORDER; d++) {
                auto found =
                    std::find_if(entries.begin(), entries.end(), [&](const Entry & e) { return e.bin == cfg[d]; });
                if (found == entries.end()) {
                    auto victim = entries.end();
                    for (auto it = entries.begin(); it != entries.end(); ++it) {
                        if (it->bin >= 0 && std::find(cfg.begin(), cfg.end(), it->bin) != cfg.end())
                            continue;
                        if (victim == entries.end() || it->bin < 0 || it->age < victim->age)
                            victim = it;
                        if (it->bin < 0)
                            break;
                    }
                    if (victim == entries.end())
                        throw std::logic_error("[polyspectrum] No evictable shell");
                    victim->grid.free();
                    victim->bin = -1;
                    victim->grid = build(cfg[d]);
                    victim->bin = cfg[d];
                    ++builds;
                    stats.shell_grid_bytes = std::max(stats.shell_grid_bytes, victim->grid.allocated_storage_bytes());
                    found = victim;
                    size_t resident =
                        std::count_if(entries.begin(), entries.end(), [](const Entry & e) { return e.bin >= 0; });
                    stats.peak_cached_shells = std::max(stats.peak_cached_shells, resident);
                    size_t resident_bytes = 0;
                    for (const auto & entry : entries)
                        resident_bytes += entry.grid.allocated_storage_bytes();
                    stats.memory.shell_bytes = std::max(stats.memory.shell_bytes, resident_bytes);
                }
                found->age = ++clock;
                result[d] = &found->grid;
            }
            return result;
        }
    };
    template <int N, int ORDER, class Store>
    void integrate_configurations(ShellCache<N, ORDER> & cache,
                                  const std::vector<std::array<int, ORDER>> & configs,
                                  size_t batch_size,
                                  size_t & reductions,
                                  Store store) {
        const size_t batch = std::max(size_t(1), std::min(size_t(INT_MAX), batch_size ? batch_size : configs.size()));
        for (size_t start = 0; start < configs.size(); start += batch) {
            std::vector<double> sums(std::min(batch, configs.size() - start), 0.0);
            cache.track_reduction_storage(sums.capacity() * sizeof(double));
            for (size_t i = 0; i < sums.size(); i++) {
                auto fields = cache.get(configs[start + i]);
                double sum = 0.0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : sum)
#endif
                for (int slice = 0; slice < fields[0]->get_local_nx(); slice++)
                    for (auto index : fields[0]->get_real_range(slice, slice + 1)) {
                        double product = 1.0;
                        for (auto field : fields)
                            product *= field->get_real_from_index(index);
                        sum += product;
                    }
                sums[i] = sum;
            }
            FML::SumArrayOverTasks(sums.data(), sums.size());
            ++reductions;
            for (size_t i = 0; i < sums.size(); i++)
                store(configs[start + i], sums[i]);
        }
    }
    template <int N, int ORDER, bool COMPACT>
    void finish_memory_report(PolyspectrumBinning<N, ORDER, COMPACT> & bins) {
        auto & m = bins.statistics.memory;
        m.result_bytes = bins.result_storage_bytes();
        m.bin_metadata_bytes = bins.metadata_storage_bytes();
        m.fft_workspace_allowance_bytes = bins.options.fft_workspace_allowance_bytes.value_or(0);
        std::array<size_t *, 11> fields{&m.input_bytes,
                                        &m.result_bytes,
                                        &m.bin_metadata_bytes,
                                        &m.configuration_bytes,
                                        &m.cache_metadata_bytes,
                                        &m.shell_bytes,
                                        &m.padding_source_bytes,
                                        &m.remapping_bytes,
                                        &m.metadata_work_bytes,
                                        &m.reduction_bytes,
                                        &m.fft_workspace_allowance_bytes};
        std::array<FML::COMMUNICATION::Count, 11> values{};
        for (size_t i = 0; i < fields.size(); ++i)
            values[i] = *fields[i];
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE,
                      values.data(),
                      int(values.size()),
                      MPI_UNSIGNED_LONG_LONG,
                      MPI_MAX,
                      FML::communicator);
#endif
        int supplied = bins.options.fft_workspace_allowance_bytes.has_value();
        FML::MinOverTasks(&supplied);
        FML::COMMUNICATION::Count total = 0;
        bool valid = true;
        for (size_t i = 0; i + 1 < fields.size(); ++i)
            valid &= FML::COMMUNICATION::add_count(total, values[i], std::numeric_limits<size_t>::max());
        m.accounted_peak_bytes = size_t(total);
        m.estimated_peak_bytes.reset();
        if (supplied) {
            valid &= FML::COMMUNICATION::add_count(total, values.back(), std::numeric_limits<size_t>::max());
            if (valid)
                m.estimated_peak_bytes = size_t(total);
        }
        FML::COMMUNICATION::require_count(valid, "[polyspectrum] Memory estimate exceeds size_t");
        for (size_t i = 0; i < fields.size(); ++i)
            *fields[i] = size_t(values[i]);
    }

    template <int N, int ORDER, bool COMPACT>
    void store_permutations(PolyspectrumBinning<N, ORDER, COMPACT> & bins,
                            std::vector<double> & output,
                            std::array<int, ORDER> cfg,
                            double value) {
        if constexpr (COMPACT) {
            output[bins.get_index_from_coord(cfg)] = value;
            return;
        }
        do {
            output[bins.get_index_from_coord(cfg)] = value;
        } while (std::next_permutation(cfg.begin(), cfg.end()));
    }
} // namespace FML::CORRELATIONFUNCTIONS::detail
#endif
