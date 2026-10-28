#ifndef CHUNKPIX_HEADER
#define CHUNKPIX_HEADER
#ifdef USE_HEALPIX

#include <FML/Global/Global.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "healpix_map.h"
#include "healpix_map_fitsio.h"
#pragma GCC diagnostic pop

// Chunkpix is created by Albert Izard

namespace FML {

    namespace CHUNKPIX {

        /// Type of chunk indices (there are 12 nside_chunks^2 chunks, e.g. 786432 for nside_chunks = 256, so a
        /// 16 bit integer is not enough)
        typedef std::int64_t ChunkIndex;

        /// A sparse Healpix map: the npix = 12 nside^2 pixels are split into 12 nside_chunks^2 chunks of consecutive
        /// pixel indices and only the chunks that have been written to are allocated.
        ///
        /// The pixel indices are in the ordering() of the map: NEST when nside is a power of two (then a chunk is
        /// the set of pixels inside one pixel of an nside_chunks map, a compact patch of the sky, so a map that
        /// covers part of the sky allocates about that fraction of the chunks), RING otherwise (a chunk is then a
        /// run of (nside/nside_chunks)^2 pixels along the iso-latitude rings, e.g. 32 full rings for nside = 8192 and
        /// nside_chunks = 8, so a partial sky allocates most chunks). Compute the indices with a Healpix_Base of the
        /// same ordering. reconstruct_full_map always gives a RING map.
        template <class T>
        class Chunkpix {
          private:
            long nside{};
            long nside_chunks{};
            long npix{};
            long nchunks{};
            long npix_per_chunk{};
            Healpix_Ordering_Scheme scheme{Healpix_Ordering_Scheme::RING};

            // Data: one vector per active chunk (no reallocation and copy of all the chunks when one is added)
            std::vector<std::vector<T>> cmap;
            // The chunk index (in the full map) of the active chunks in the order they were activated
            std::vector<ChunkIndex> active_chunks_values;
            // Map from chunk index to its position (key) in active_chunks_values
            std::unordered_map<ChunkIndex, long> active_chunks_keys;
            long n_active_chunks{};

          public:
            Chunkpix() = default;

            void init(long nside, long nside_chunks);

            void initialize_cmap();

            double chunk_sparsity();

            ChunkIndex ipix2chunk(long ipix_full);

            std::pair<size_t, size_t> chunk2ipix_ranges(ChunkIndex chunk);

            void increase_ipix_count(long ipix_full, T count_increase);

            void reconstruct_full_map(Healpix_Map<T> & full_map);

            void clean();

            /// The ordering of the pixel indices passed to increase_ipix_count
            Healpix_Ordering_Scheme ordering() const { return scheme; }
            long get_nside() const { return nside; }
            long get_npix() const { return npix; }
            long get_n_active_chunks() const { return n_active_chunks; }
        };

        template <class T>
        void Chunkpix<T>::init(long _nside, long _nside_chunks) {
            if (_nside <= 0 or _nside_chunks <= 0)
                throw std::runtime_error("Chunkpix error: nside and nside_chunks must be positive");
            if (_nside_chunks > _nside)
                throw std::runtime_error("Chunkpix error, choose nside_chunks <= nside");
            // Otherwise npix is not a multiple of nchunks and the last pixels would be in no chunk
            if (_nside % _nside_chunks != 0)
                throw std::runtime_error("Chunkpix error: nside (" + std::to_string(_nside) +
                                         ") must be a multiple of nside_chunks (" + std::to_string(_nside_chunks) +
                                         ")");

            nside = _nside;
            nside_chunks = _nside_chunks;

            npix = 12 * nside * nside;
            nchunks = 12 * nside_chunks * nside_chunks;
            npix_per_chunk = npix / nchunks;
            // NEST needs nside = 2^k (then nside_chunks, a divisor of nside, is a power of two too)
            scheme = (nside & (nside - 1)) == 0 ? Healpix_Ordering_Scheme::NEST : Healpix_Ordering_Scheme::RING;

            initialize_cmap();
        }

        // Set chunkpix map to 0
        template <class T>
        void Chunkpix<T>::initialize_cmap() {
            n_active_chunks = 0;
            active_chunks_values.resize(0);
            active_chunks_values.shrink_to_fit();
            active_chunks_keys = std::unordered_map<ChunkIndex, long>();
            cmap.resize(0);
            cmap.shrink_to_fit();
        }

        // Fraction of memory of a cmap compared to a full map
        template <class T>
        double Chunkpix<T>::chunk_sparsity() {
            return nchunks > 0 ? double(n_active_chunks) / double(nchunks) : 0.0;
        }

        // Return the chunk in the full-size map where a pixel belongs to
        template <class T>
        ChunkIndex Chunkpix<T>::ipix2chunk(long ipix_full) {
            return ChunkIndex(ipix_full / npix_per_chunk);
        }

        // Return the range of the pixels belonging to a chunk (first and last pixel, inclusive)
        template <class T>
        std::pair<size_t, size_t> Chunkpix<T>::chunk2ipix_ranges(ChunkIndex chunk) {
            return {size_t(npix_per_chunk) * size_t(chunk), size_t(npix_per_chunk) * size_t(chunk + 1) - 1};
        }

        // Increase the count number in cmap for the corresponding pixel.
        // This operation may activate a new chunk.
        template <class T>
        void Chunkpix<T>::increase_ipix_count(long ipix_full, T count_increase) {
            if (ipix_full < 0 or ipix_full >= npix)
                throw std::runtime_error("Chunkpix error: pixel index " + std::to_string(ipix_full) +
                                         " is outside [0, npix = " + std::to_string(npix) + ")");
            // Chunk the pixel is in
            const ChunkIndex chunk_value = ipix2chunk(ipix_full);
            // Is this chunk active?
            long chunk_key;
            auto it = active_chunks_keys.find(chunk_value);
            if (it == active_chunks_keys.end()) {
                // Chunk not in the list. Activate it.
                chunk_key = n_active_chunks;
                n_active_chunks++;
                active_chunks_values.push_back(chunk_value);
                active_chunks_keys[chunk_value] = chunk_key;
                // Allocate and initialize new chunk pixels to 0
                cmap.emplace_back(size_t(npix_per_chunk), T(0));
            } else {
                chunk_key = it->second;
            }

            // Add count
            cmap[size_t(chunk_key)][size_t(ipix_full % npix_per_chunk)] += count_increase;
        }

        // Reconstruct a full-size map from cmap
        template <class T>
        void Chunkpix<T>::reconstruct_full_map(Healpix_Map<T> & full_map) {
            full_map.SetNside(int(nside), scheme);
            full_map.fill(T(0));

            // Copy chunk by chunk
            double sum_cmap = 0.0;
            for (size_t key = 0; key < size_t(n_active_chunks); key++) {
                const auto i_full = chunk2ipix_ranges(active_chunks_values[key]);
                const auto & chunk = cmap[key];
                for (size_t i = 0; i <= i_full.second - i_full.first; i++) {
                    full_map[i_full.first + i] = chunk[i];
                    sum_cmap += double(chunk[i]);
                }
            }

            // Check count conservation (in floating point: the counts can be weights)
            double sum_full = 0.0;
            double sum_abs_full = 0.0;
            for (long i = 0; i < npix; i++) {
                sum_full += double(full_map[i]);
                sum_abs_full += std::fabs(double(full_map[i]));
            }
            if (std::fabs(sum_full - sum_cmap) > 1e-6 * std::max(1.0, sum_abs_full)) {
                std::string err_msg = "Error reconstructing full healpix map from chunkpix, task " +
                                      std::to_string(FML::ThisTask) + "\n" + std::to_string(sum_cmap) + "\t" +
                                      std::to_string(sum_full) + "\n";
                std::cout << err_msg;
                throw std::runtime_error(err_msg);
            }
            // The full map is always RING
            if (scheme == Healpix_Ordering_Scheme::NEST)
                full_map.swap_scheme();
        }

        template <class T>
        void Chunkpix<T>::clean() {
            n_active_chunks = 0;
            cmap.clear();
            cmap.shrink_to_fit();
            active_chunks_values.clear();
            active_chunks_values.shrink_to_fit();
            active_chunks_keys = std::unordered_map<ChunkIndex, long>();
        }

    } // namespace CHUNKPIX
} // namespace FML

#endif
#endif
