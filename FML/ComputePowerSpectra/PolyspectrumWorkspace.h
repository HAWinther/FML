#ifndef FML_POLYSPECTRUM_WORKSPACE_H
#define FML_POLYSPECTRUM_WORKSPACE_H
#include <FML/ComputePowerSpectra/PolyspectrumBinning.h>
#include <FML/FFTWGrid/FFTWGrid.h>
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
                    found = victim;
                    size_t resident =
                        std::count_if(entries.begin(), entries.end(), [](const Entry & e) { return e.bin >= 0; });
                    stats.peak_cached_shells = std::max(stats.peak_cached_shells, resident);
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
            FML::SumArrayOverTasks(sums.data(), int(sums.size()));
            ++reductions;
            for (size_t i = 0; i < sums.size(); i++)
                store(configs[start + i], sums[i]);
        }
    }
    template <int N, int ORDER>
    void store_permutations(PolyspectrumBinning<N, ORDER> & bins,
                            std::vector<double> & output,
                            std::array<int, ORDER> cfg,
                            double value) {
        do {
            output[bins.get_index_from_coord(cfg)] = value;
        } while (std::next_permutation(cfg.begin(), cfg.end()));
    }
} // namespace FML::CORRELATIONFUNCTIONS::detail
#endif
