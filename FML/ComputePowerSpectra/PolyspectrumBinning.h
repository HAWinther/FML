#ifndef POLYSPECTRUMBINNING_HEADER
#define POLYSPECTRUMBINNING_HEADER
#include <FML/Global/Global.h>
#include <array>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace FML {
    namespace CORRELATIONFUNCTIONS {

        struct PolyspectrumOptions {
            /// Zero retains all needed shells. Otherwise cap the resident shell count.
            size_t max_cached_shells{0};
            /// Per-rank cache budget; zero is unlimited. Excludes the input, one
            /// source-resolution padding scratch grid, MPI remapping buffers, FFT workspace and result arrays.
            size_t shell_memory_budget_bytes{0};
            /// Canonical configurations per collective; zero requests one batch (up to INT_MAX).
            size_t reduction_batch_size{4096};
        };
        struct PolyspectrumStatistics {
            size_t shell_grid_bytes{0}, peak_cached_shells{0};
            size_t geometry_shell_builds{0}, field_shell_builds{0};
            size_t geometry_reduction_batches{0}, field_reduction_batches{0}, metadata_reduction_batches{0};
        };

        /// Dense results by default; COMPACT stores one slot per canonical tuple.
        template <int N, int ORDER, bool COMPACT = false>
        class PolyspectrumBinning {
          public:
            static constexpr bool compact_storage = COMPACT;
            enum BinningType { LINEAR_SPACING, LOG_SPACING };

            /// Number of bins
            int n{0};
            /// Type of bins (LINEAR_SPACING or LOG_SPACING)
            int bin_type{LINEAR_SPACING};
            /// The minimum k to bin from (should be 0!)
            double kmin{0.0};
            /// The maximum k to bin to
            double kmax{0.0};

            /// Subtract shotnoise? Algorithms (compute_polyspectrum) using this class checks this value
            bool subtract_shotnoise{true};

            /// The polyspectrum
            std::vector<double> P123;
            /// The volume (number of generalized triangles) of each bin
            std::vector<double> N123;
            /// The power-spectrum
            std::vector<double> pofk;

            /// kbin is center of bin
            std::vector<double> kbin;
            /// klow is lower bin edge
            std::vector<double> klow;
            /// khigh is higher bin edge
            std::vector<double> khigh;
            /// kmean is the mean of wavenumbers added to the bin
            std::vector<double> kmean;

            /// If we have precomputed the volume factor (N123) or not and have it available for algorithms to use
            bool bincount_is_set{false};
            /// True integer-wavevector closure. False retains legacy modulo-grid closure.
            /// Nonlinear estimates exclude even-mesh Nyquist planes when enabled.
            bool dealias{true};
            PolyspectrumOptions options;
            PolyspectrumStatistics statistics;
            using Configuration = std::array<int, ORDER>;

            /// Empty restores all configurations. Inputs are canonicalized and deduplicated.
            void set_configurations(std::vector<Configuration> configurations) {
                for (auto & cfg : configurations) {
                    for (auto bin : cfg)
                        if (bin < 0 || bin >= n)
                            throw std::invalid_argument("[PolyspectrumBinning] Selected bin is out of range");
                    std::sort(cfg.begin(), cfg.end());
                }
                std::sort(configurations.begin(), configurations.end());
                configurations.erase(std::unique(configurations.begin(), configurations.end()), configurations.end());
                if (selected == configurations)
                    return;
                selected = std::move(configurations);
                bincount_is_set = false;
                if constexpr (COMPACT) {
                    std::vector<Configuration>().swap(compact_configs);
                    std::vector<Configuration>().swap(cached_selected);
                    std::vector<double>().swap(P123);
                    std::vector<double>().swap(N123);
                }
            }
            std::vector<Configuration> configurations() {
                std::vector<Configuration> result;
                if (!selected.empty()) {
                    for (const auto & cfg : selected)
                        if (compute_this_configuration(cfg))
                            result.push_back(cfg);
                    return result;
                }
                Configuration cfg{};
                std::function<void(int, int)> enumerate = [&](int depth, int first) {
                    if (depth == ORDER) {
                        if (compute_this_configuration(cfg))
                            result.push_back(cfg);
                        return;
                    }
                    for (int bin = first; bin < n; bin++) {
                        cfg[depth] = bin;
                        enumerate(depth + 1, bin);
                    }
                };
                enumerate(0, 0);
                return result;
            }

            /// Allocate compact results only when needed. No dense tensor is constructed.
            void prepare_storage() {
                if constexpr (COMPACT) {
                    auto configs = configurations();
                    if (configs != compact_configs || P123.size() != configs.size() || N123.size() != configs.size()) {
                        std::vector<double> powers(configs.size(), 0.0), counts(configs.size(), 0.0);
                        compact_configs = std::move(configs);
                        P123 = std::move(powers);
                        N123 = std::move(counts);
                        bincount_is_set = false;
                    }
                }
            }
            const std::vector<Configuration> & stored_configurations() const { return compact_configs; }
            static size_t dense_entries(int nbins) {
                if (nbins < 2)
                    throw std::invalid_argument("[PolyspectrumBinning] At least two bins are required");
                size_t entries = 1;
                const size_t limit = std::vector<double>().max_size();
                for (int d = 0; d < ORDER; ++d) {
                    if (entries > limit / size_t(nbins))
                        throw std::length_error("[PolyspectrumBinning] Dense result exceeds addressable storage");
                    entries *= size_t(nbins);
                }
                return entries;
            }
            bool valid_storage() const {
                const size_t entries = COMPACT ? compact_configs.size() : dense_entries(n);
                return P123.size() == entries && N123.size() == entries;
            }
            /// Explicit dense allocation and symmetry expansion; never used by estimators.
            PolyspectrumBinning<N, ORDER, false> to_dense();

            bool has_bincount(int nmesh) {
                if (!bincount_is_set || cached_low != klow || cached_high != khigh || cached_dealias != dealias ||
                    cached_selected != selected)
                    return false;
                if (cached_mesh == 0)
                    cached_mesh = nmesh; // legacy explicitly imported counts
                return cached_mesh == nmesh;
            }
            void mark_bincount_computed(int nmesh) {
                cached_mesh = nmesh;
                cached_low = klow;
                cached_high = khigh;
                cached_dealias = dealias;
                cached_selected = selected;
                bincount_is_set = true;
            }

          private:
            template <int, int, bool> friend class PolyspectrumBinning;
            std::vector<Configuration> compact_configs;
            size_t compact_index(Configuration cfg) const {
                for (auto bin : cfg)
                    if (bin < 0 || bin >= n)
                        throw std::out_of_range("[PolyspectrumBinning] Bin is out of range");
                std::sort(cfg.begin(), cfg.end());
                auto found = std::lower_bound(compact_configs.begin(), compact_configs.end(), cfg);
                return found != compact_configs.end() && *found == cfg ? size_t(found - compact_configs.begin()) : compact_configs.size();
            }
            std::vector<Configuration> selected, cached_selected;
            int cached_mesh{-1};
            std::vector<double> cached_low, cached_high;
            bool cached_dealias{true};

          public:
            PolyspectrumBinning() = default;
            PolyspectrumBinning(double _kmin, double _kmax, int nbins, int bin_type = LINEAR_SPACING);
            PolyspectrumBinning(int nbins, int Nmesh, int bin_type = LINEAR_SPACING);

            /// To physical units: k *= 1/boxsize and P *= boxsize^((ORDER-1) NDIM)
            void scale(const double boxsize);

            /// Clears the arrays
            void reset();

            /// Release result arrays/canonical slots, retaining shell metadata and selection.
            void free();

            /// Get the polyspectra at (k[ik1], k[ik2], ...)
            double get_spectrum(const std::array<int, ORDER> & ik);
            /// For ORDER = 2 get the monospectum P(i,j) with i,j index of bin in [0,nbins)
            double get_spectrum(int i, int j);
            /// For ORDER = 3 get the bispectum B(i,j,k) with i,j,k index of bin in [0,nbins)
            double get_spectrum(int i, int j, int k);
            /// For ORDER = 4 get the trispectum T(i,j,k,l) with i,j,k,l index of bin in [0,nbins)
            double get_spectrum(int i, int j, int k, int l);

            /// Get the reduced polyspectrum: P / fac
            /// where fac = sqrt(P(k1)P(k2)) for order=2 and P(k1)P(k2) + cyc for ORDER=3,4
            double get_reduced_spectrum(const std::array<int, ORDER> & ik);
            double get_reduced_spectrum(int i, int j);
            double get_reduced_spectrum(int i, int j, int k);
            double get_reduced_spectrum(int i, int j, int k, int l);

            /// Returns the volume factor N123
            double get_bincount(const std::array<int, ORDER> & ik);
            double get_bincount(int i, int j);
            double get_bincount(int i, int j, int k);
            double get_bincount(int i, int j, int k, int l);

            /* Simpler ways of doing the methods above
            template <typename... Ints>
            double get_spectrum(Ints... is);
            template <typename... Ints>
            double get_reduced_spectrum(Ints... is);
            template <typename... Ints>
            double get_bincount(Ints... is);
            */

            /// Copy over the volume factor N123 (useful to save computations if we do many calculations with
            /// the same setup)
            void set_bincount(const std::vector<double> & N123_external);
            /// Import with an explicit mesh identity (preferred to the legacy overload).
            void set_bincount(const std::vector<double> & counts, int nmesh) {
                set_bincount(counts);
                mark_bincount_computed(nmesh);
            }

            /// This is just to make it easier to add binning of several spectra... just for testing
            void combine(PolyspectrumBinning & rhs);
            int nbinnings{0};

            /// The coord (k1,k2,...) corresponding to a given index in the P123 and N123 array
            /// These methods determine how the data is stored and fetched
            std::array<int, ORDER> get_coord_from_index(size_t index);
            size_t get_index_from_coord(const std::array<int, ORDER> & ik);

            /// Symmetry: we only need to compute ik1 <= ik2 <= ...
            /// This function tells algorithms which configurations to compute and
            /// which to set by using symmetry from the configs we have computed
            bool compute_this_configuration(const std::array<int, ORDER> & ik);
        };

        template <int N, int ORDER, bool COMPACT>
        PolyspectrumBinning<N, ORDER, COMPACT>::PolyspectrumBinning(int nbins, int Nmesh, int bin_type)
            : PolyspectrumBinning(0.0, 2.0 * M_PI * Nmesh / 2.0, nbins, bin_type){};

        template <int N, int ORDER, bool COMPACT>
        PolyspectrumBinning<N, ORDER, COMPACT>::PolyspectrumBinning(double _kmin, double _kmax, int nbins, int bin_type) {
            assert_mpi(nbins >= 2, "[PolyspectrumBinning] We need at least 2 bins");
            this->bin_type = bin_type;
            n = nbins;
            kmin = _kmin;
            kmax = _kmax;
            if constexpr (!COMPACT) {
                const size_t ntot = dense_entries(n);
                P123.resize(ntot, 0.0);
                N123.resize(ntot, 0.0);
            }
            pofk.resize(n, 0.0);
            kbin.resize(n, 0.0);
            klow.resize(n, 0.0);
            khigh.resize(n, 0.0);
            kmean.resize(n, 0.0);

            // k[0] is leftmost bin edge and k[n-1] is rightmost bin edge and otherwise center of bin
            // Used to set kbin below
            std::vector<double> k(n);
            for (int i = 0; i < n; i++) {
                if (bin_type == LINEAR_SPACING) {
                    k[i] = kmin + (kmax - kmin) * i / double(n - 1);
                } else if (bin_type == LOG_SPACING) {
                    assert_mpi(kmin > 0.0, "[PolyspectrumBinning] For log spacing we cannot have kmin = 0.0");
                    k[i] = std::exp(std::log(kmin) + std::log(kmax / kmin) * i / double(n - 1));
                } else {
                    // Unknown binning type
                    assert_mpi(false, "[PolyspectrumBinning] Unknown bintype (not LINEAR_SPACING or LOG_SPACING)");
                }
            }

            // We require that klow = 0 in the smallest bin
            for (int i = 0; i < n; i++) {
                if (i == 0) {
                    klow[i] = std::min(k[0], 0.0);
                    khigh[i] = k[0] + (k[1] - k[0]) / 2.0;
                    if (bin_type == LOG_SPACING)
                        khigh[i] = std::exp(std::log(k[0]) + std::log(k[1] / k[0]) / 2.0);
                } else if (i < n - 1) {
                    klow[i] = khigh[i - 1];
                    khigh[i] = k[i] + (k[i + 1] - k[i]) / 2.0;
                    if (bin_type == LOG_SPACING)
                        khigh[i] = std::exp(std::log(k[i]) + std::log(k[i + 1] / k[i]) / 2.0);
                } else {
                    klow[i] = khigh[i - 1];
                    khigh[i] = k[nbins - 1];
                }
                kbin[i] = (klow[i] + khigh[i]) / 2.0;
            }
        }

        template <int N, int ORDER, bool COMPACT>
        std::array<int, ORDER> PolyspectrumBinning<N, ORDER, COMPACT>::get_coord_from_index(size_t index) {
            if constexpr (COMPACT)
                return compact_configs.at(index);
            std::array<int, ORDER> ik;
            for (int ii = ORDER - 1; ii >= 0; ii--) {
                ik[ii] = int(index % n);
                index /= size_t(n);
            }
            return ik;
        }

        template <int N, int ORDER, bool COMPACT>
        size_t PolyspectrumBinning<N, ORDER, COMPACT>::get_index_from_coord(const std::array<int, ORDER> & ik) {
            if constexpr (COMPACT) {
                const size_t index = compact_index(ik);
                if (index == compact_configs.size())
                    throw std::out_of_range("[PolyspectrumBinning] Configuration has no stored slot");
                return index;
            }
            size_t index = 0;
            for (int i = 0; i < ORDER; i++)
                index = index * n + ik[i];
            return index;
        }

        // Symmetry: we only need to compute ik1 <= ik2 <= ...
        template <int N, int ORDER, bool COMPACT>
        bool PolyspectrumBinning<N, ORDER, COMPACT>::compute_this_configuration(const std::array<int, ORDER> & ik) {
            if (!selected.empty() && !std::binary_search(selected.begin(), selected.end(), ik))
                return false;
            double ksum = 0.0;
            for (int ii = 1; ii < ORDER; ii++) {
                if (ik[ii - 1] > ik[ii]) {
                    return false;
                }
                ksum += khigh[ik[ii - 1]];
            }

            // No valid 'triangles' if k1+k2+... < kN so can just set too zero right away
            // Saves a bit of time computing, but not significantly so its fine to remove this just to be sure
            if (ksum < klow[ik[ORDER - 1]] and ORDER > 2)
                return false;

            return true;
        }

        template <int N, int ORDER, bool COMPACT>
        void PolyspectrumBinning<N, ORDER, COMPACT>::scale(const double boxsize) {
            for (int i = 0; i < n; i++) {
                kbin[i] *= 1.0 / boxsize;
                klow[i] *= 1.0 / boxsize;
                khigh[i] *= 1.0 / boxsize;
                kmean[i] *= 1.0 / boxsize;
            }

            // Scale polyspectrum
            double scale = std::pow(boxsize, N * (ORDER - 1));
            for (auto & p : P123)
                p *= scale;

            // Scale power-spectrum
            scale = std::pow(boxsize, N);
            for (auto & p : pofk)
                p *= scale;

            // In principle we should scale N123 by 1/boxsize^ORDER
            // however this is only used internally and then in
            // dimensionless units so we omit this here
        }

        template <int N, int ORDER, bool COMPACT>
        void PolyspectrumBinning<N, ORDER, COMPACT>::reset() {
            std::fill(P123.begin(), P123.end(), 0.0);
            std::fill(N123.begin(), N123.end(), 0.0);
            std::fill(pofk.begin(), pofk.end(), 0.0);
            std::fill(kmean.begin(), kmean.end(), 0.0);
            nbinnings = 0;
            // N123 is zeroed so it must be computed again
            bincount_is_set = false;
        }

        template <int N, int ORDER, bool COMPACT>
        void PolyspectrumBinning<N, ORDER, COMPACT>::free() {
            // We only bother to free the things that take up memory
            bincount_is_set = false;
            P123.clear();
            P123.shrink_to_fit();
            N123.clear();
            N123.shrink_to_fit();
            if constexpr (COMPACT) {
                std::vector<Configuration>().swap(compact_configs);
                std::vector<Configuration>().swap(cached_selected);
            }
        }

        /* Simpler way of doing the methods below
        template <int N, int ORDER, bool COMPACT>
        template<typename... Ints>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_spectrum(Ints... is) {
            static_assert(sizeof...(is) == ORDER,
                  "Calling get_spectrum(i1, i2, ..., in) requires n == ORDER (e.g. get_spetrum(i,j,k) can only be called
        when ORDER=3"); std::array<int, ORDER> ik{is...}; return get_spectrum(ik);
        }
        template <int N, int ORDER, bool COMPACT>
        template<typename... Ints>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_reduced_spectrum(Ints... is) {
            static_assert(sizeof...(is) == ORDER,
                  "Calling get_reduced_spectrum(i1, i2, ..., in) requires n == ORDER (e.g. get_reduced_spetrum(i,j,k)
        can only be called when ORDER=3"); std::array<int, ORDER> ik{is...}; return get_reduced_spectrum(ik);
        }
        template <int N, int ORDER, bool COMPACT>
        template<typename... Ints>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_bincount(Ints... is) {
            static_assert(sizeof...(is) == ORDER,
                  "Calling get_bincount(i1, i2, ..., in) requires n == ORDER (e.g. get_bincount(i,j,k) can only be
        called when ORDER=3"); std::array<int, ORDER> ik{is...}; return get_bincount(ik);
        }
        */

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_spectrum(int i, int j) {
            assert_mpi(ORDER == 2, "[PolyspectrumBinning::get_spectrum] This method can only be called for ORDER=2\n");
            std::array<int, 2> ik{i, j};
            return get_spectrum(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_spectrum(int i, int j, int k) {
            assert_mpi(ORDER == 3, "[PolyspectrumBinning::get_spectrum] This method can only be called for ORDER=3\n");
            std::array<int, 3> ik{i, j, k};
            return get_spectrum(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_spectrum(int i, int j, int k, int l) {
            assert_mpi(ORDER == 4, "[PolyspectrumBinning::get_spectrum] This method can only be called for ORDER=4\n");
            std::array<int, 4> ik{i, j, k, l};
            return get_spectrum(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_spectrum(const std::array<int, ORDER> & ik) {
            if constexpr (COMPACT) {
                const size_t index = compact_index(ik);
                return index < P123.size() ? P123[index] : 0.0;
            }
            assert_mpi(ik.size() == ORDER, "[PolyspectrumBinning::get_spectrum] ik != ORDER has the wrong size\n");
            return P123[get_index_from_coord(ik)];
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_reduced_spectrum(int i, int j) {
            assert_mpi(ORDER == 2,
                       "[PolyspectrumBinning::get_reduced_spectrum] This method can only be called for ORDER=2\n");
            std::array<int, 2> ik{i, j};
            return get_reduced_spectrum(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_reduced_spectrum(int i, int j, int k) {
            assert_mpi(ORDER == 3,
                       "[PolyspectrumBinning::get_reduced_spectrum] This method can only be called for ORDER=3\n");
            std::array<int, 3> ik{i, j, k};
            return get_reduced_spectrum(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_reduced_spectrum(int i, int j, int k, int l) {
            assert_mpi(ORDER == 4,
                       "[PolyspectrumBinning::get_reduced_spectrum] This method can only be called for ORDER=4\n");
            std::array<int, 4> ik{i, j, k, l};
            return get_reduced_spectrum(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_reduced_spectrum(const std::array<int, ORDER> & ik) {
            const double P = get_spectrum(ik);
            // An empty bin (or a configuration that is not a triangle) has P = 0 and maybe zero power below
            if (P == 0.0)
                return 0.0;
            std::vector<double> pi(ORDER);

            for (int i = 0; i < ORDER; i++)
                pi[i] = pofk[ik[i]];

            if (ORDER == 2) {
                // For gaussian delta(i,j) for ORDER = 2
                return P / std::sqrt(pofk[ik[0]] * pofk[ik[1]]);
            } else if (ORDER == 3 or ORDER == 4) {
                // P(k1,..) / (P1P2 + cyc) for ORDER = 3 or 4
                // So = 0 and = 1 if gaussian
                double pipj = pi[ORDER - 1] * pi[0];
                for (int i = 0; i < ORDER - 1; i++)
                    pipj += pi[i] * pi[i + 1];
                return P / pipj;
            }

            // The general case
            // <d1d2...d2n> / (Pi1Pi2...Pin + cyc) ~ 1 if gaussian
            // This will never be used so we don't implement it
            assert_mpi(false, "[PolyspectrumBinning::get_reduced_spectrum] Not implemented\n");

            return 0.0;
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_bincount(const std::array<int, ORDER> & ik) {
            if constexpr (COMPACT) {
                const size_t index = compact_index(ik);
                return index < N123.size() ? N123[index] : 0.0;
            }
            assert_mpi(ik.size() == ORDER, "[PolyspectrumBinning::get_bincount] ik != ORDER has the wrong size\n");
            return N123[get_index_from_coord(ik)];
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_bincount(int i, int j) {
            assert_mpi(ORDER == 2, "[PolyspectrumBinning::get_bincount] This method can only be called for ORDER=2\n");
            std::array<int, 2> ik{i, j};
            return get_bincount(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_bincount(int i, int j, int k) {
            assert_mpi(ORDER == 3, "[PolyspectrumBinning::get_bincount] This method can only be called for ORDER=3\n");
            std::array<int, 3> ik{i, j, k};
            return get_bincount(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        double PolyspectrumBinning<N, ORDER, COMPACT>::get_bincount(int i, int j, int k, int l) {
            assert_mpi(ORDER == 4, "[PolyspectrumBinning::get_bincount] This method can only be called for ORDER=4\n");
            std::array<int, 4> ik{i, j, k, l};
            return get_bincount(ik);
        }

        template <int N, int ORDER, bool COMPACT>
        void PolyspectrumBinning<N, ORDER, COMPACT>::combine(PolyspectrumBinning<N, ORDER, COMPACT> & rhs) {
            prepare_storage();
            rhs.prepare_storage();
            assert_mpi(n == rhs.n and bin_type == rhs.bin_type && klow == rhs.klow && khigh == rhs.khigh &&
                           dealias == rhs.dealias && selected == rhs.selected && compact_configs == rhs.compact_configs,
                       "[PolyspectrumBinning::combine] Incompatible binnings\n");
            if (nbinnings == 0) {
                P123 = rhs.P123;
                N123 = rhs.N123;
                pofk = rhs.pofk;
                kmean = rhs.kmean;
            } else {
                for (size_t i = 0; i < P123.size(); i++)
                    P123[i] = (P123[i] * nbinnings + rhs.P123[i]) / (nbinnings + 1.0);
                for (size_t i = 0; i < N123.size(); i++)
                    N123[i] += rhs.N123[i];
                for (size_t i = 0; i < pofk.size(); i++)
                    pofk[i] = (pofk[i] * nbinnings + rhs.pofk[i]) / (nbinnings + 1.0);
            }
            nbinnings++;
            bincount_is_set = false;
        }

        template <int N, int ORDER, bool COMPACT>
        void PolyspectrumBinning<N, ORDER, COMPACT>::set_bincount(const std::vector<double> & N123_external) {
            prepare_storage();
            assert_mpi(N123_external.size() == N123.size(), "[set_bincount] Incompatible size of N123 array");
            N123 = N123_external;
            if (!selected.empty() && !COMPACT) {
                const auto requested = configurations();
                std::vector<double> values;
                for (const auto & cfg : requested)
                    values.push_back(N123[get_index_from_coord(cfg)]);
                std::fill(N123.begin(), N123.end(), 0.0);
                for (size_t i = 0; i < requested.size(); i++) {
                    auto cfg = requested[i];
                    do {
                        N123[get_index_from_coord(cfg)] = values[i];
                    } while (std::next_permutation(cfg.begin(), cfg.end()));
                }
            }
            mark_bincount_computed(0);
        }

        template <int N, int ORDER, bool COMPACT>
        PolyspectrumBinning<N, ORDER, false> PolyspectrumBinning<N, ORDER, COMPACT>::to_dense() {
            if constexpr (!COMPACT) {
                return *this;
            } else {
                if (P123.size() != compact_configs.size() || N123.size() != compact_configs.size())
                    throw std::invalid_argument("[PolyspectrumBinning] Inconsistent compact result storage");
                PolyspectrumBinning<N, ORDER, false> dense(kmin, kmax, n, bin_type);
                dense.subtract_shotnoise = subtract_shotnoise;
                dense.dealias = dealias;
                dense.options = options;
                dense.statistics = statistics;
                dense.pofk = pofk;
                dense.kbin = kbin;
                dense.klow = klow;
                dense.khigh = khigh;
                dense.kmean = kmean;
                dense.nbinnings = nbinnings;
                dense.selected = selected;
                for (size_t i = 0; i < compact_configs.size(); ++i) {
                    auto cfg = compact_configs[i];
                    do {
                        const auto index = dense.get_index_from_coord(cfg);
                        dense.P123[index] = P123[i];
                        dense.N123[index] = N123[i];
                    } while (std::next_permutation(cfg.begin(), cfg.end()));
                }
                dense.bincount_is_set = bincount_is_set;
                dense.cached_mesh = cached_mesh;
                dense.cached_low = cached_low;
                dense.cached_high = cached_high;
                dense.cached_dealias = cached_dealias;
                dense.cached_selected = cached_selected;
                return dense;
            }
        }

        template <int N, int ORDER>
        using SparsePolyspectrumBinning = PolyspectrumBinning<N, ORDER, true>;
    } // namespace CORRELATIONFUNCTIONS
} // namespace FML

#endif
