#ifndef CHUNKPIX_HEADER
#define CHUNKPIX_HEADER
#ifdef USE_HEALPIX

#include <FML/Global/Global.h>
#include <numeric>

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wdeprecated-copy"
#include "healpix_map.h"
#include "healpix_map_fitsio.h"
#pragma GCC diagnostic pop

// Chunkpix is created by Albert Izard

namespace FML {

    namespace CHUNKPIX {
        
        typedef uint16_t int2;
        
        template <class T>
        class Chunkpix {
            private:
                long nside{};
                long nside_chunks{};
                long npix{};
                int nchunks{};
                int npix_per_chunk{};
                
                // Data
                std::vector<T> cmap;
                std::vector<int2> active_chunks_values;
                int n_active_chunks{};

            public:
                Chunkpix() = default;

                void init(long nside, long nside_chunks);

                void initialize_cmap();

                double chunk_sparsity();

                int2 ipix2chunk(int ipix_full);
                
                std::pair<size_t,size_t> chunk2ipix_ranges(int2 chunk);

                void increase_ipix_count(int ipix_full, T count_increase);

                void reconstruct_full_map(Healpix_Map<T> & full_map);

                void deallocate_full_map();

                void clean();

        };

        template <class T>
        void Chunkpix<T>::init(long _nside, long _nside_chunks) {
            nside = _nside;
            nside_chunks = _nside_chunks;

            npix = 12*nside*nside;
            nchunks = 12*nside_chunks*nside_chunks;
            npix_per_chunk = npix/nchunks;

            if (nside_chunks > nside){
                const std::string err_str = "Chunkpix error, choose nside_chunks < nside ";
                throw std::runtime_error(err_str);
            } 
            
            initialize_cmap();
        }

        // Set chunkpix map to 0
        template <class T>
        void Chunkpix<T>::initialize_cmap() {
            n_active_chunks = 0;
            active_chunks_values.resize(0);
            active_chunks_values.shrink_to_fit();
            cmap.resize(0);
            cmap.shrink_to_fit();
        }

        // Fraction of memory of a cmap compared to a full map
        template <class T>
        double Chunkpix<T>::chunk_sparsity() {
            return double(n_active_chunks / nchunks);
        }

        // Return the chunk in the full-size map where a pixel belongs to
        template <class T>
        int2 Chunkpix<T>::ipix2chunk(int ipix_full) {
            return int2(ipix_full / npix_per_chunk);
        }

        // Return the range of the pixels belonging to a chunk
        template <class T>
        std::pair<size_t,size_t> Chunkpix<T>::chunk2ipix_ranges(int2 chunk) {
            return {(size_t) npix_per_chunk*chunk, (size_t) npix_per_chunk*(chunk+1)-1 };
        }

        // Increase the count number in cmap for the corresponding pixel.
        // This operation may activate a new chunk.
        template <class T>
        void Chunkpix<T>::increase_ipix_count(int ipix_full, T count_increase) {
            // Chunk the pixel is in
            int2 chunk_value = ipix2chunk(ipix_full);
            // Is this chunk active?
            // TODO This can be a bottleneck. Optimize if necessary. Hash-table?
            auto it = std::find(active_chunks_values.begin(),
                                active_chunks_values.end(),
                                chunk_value);
            int2 chunk_key = it - active_chunks_values.begin();

            if(it == active_chunks_values.end()) {
                // Chunk not in the list. Activate it.
                n_active_chunks++;
                active_chunks_values.push_back(chunk_value);
                // Allocate and initialize new chunk pixels to 0
                cmap.resize(n_active_chunks*npix_per_chunk, 0.0);
            }

            // Add count
            std::size_t ipix_chunk = ipix_full % npix_per_chunk
                                                        + (size_t)chunk_key * npix_per_chunk; 
            cmap[ipix_chunk] += count_increase;
        }

        // Reconstruct a full-size map from cmap
        template <class T>
        void Chunkpix<T>::reconstruct_full_map(Healpix_Map<T> & full_map){
            full_map.SetNside(nside, Healpix_Ordering_Scheme::RING);
            full_map.fill(0.0);
            
            int counter = 0;
            // Copy chunk by chunk
            for(size_t key=0; key<(size_t)n_active_chunks; key++){
                auto i_full = chunk2ipix_ranges(active_chunks_values[key]);
                auto i_cmap = chunk2ipix_ranges(key);
                for(size_t i=0; i<=i_full.second-i_full.first; i++){
                    full_map[i_full.first + i] = cmap[i_cmap.first + i];
                    counter += cmap[i_cmap.first + i];
                }
            }

            // Check count conservation
            auto * start = &full_map[0];
            auto * end = &full_map[npix];
            auto c0 = std::accumulate(cmap.begin(), cmap.end(), 0);
            auto c1 = std::accumulate(start, end, 0);
            if ( c0 != c1) {
                std::string err_msg = "Error reconstructing full healpix map from chunkpix, task"
                        + std::to_string(FML::ThisTask) + " \n"
                        + std::to_string(c0) + "\t" + std::to_string(c1) + "\t"
                        + std::to_string(counter) + "\n";
                std::cout << err_msg;
                throw std::runtime_error(err_msg);
            }
        }
        
        template <class T>
        void Chunkpix<T>::clean(){
            cmap.clear();
            cmap.shrink_to_fit();
            active_chunks_values.clear();
            active_chunks_values.shrink_to_fit();
        }

    } // namespace CHUNKPIX
} // namespace FML

#endif
#endif
