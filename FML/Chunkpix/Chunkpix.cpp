#ifdef USE_HEALPIX

#include "Chunkpix.h"
#include <chealpix.h>
#include <numeric>

namespace FML {

    namespace CHUNKPIX {

        void Chunkpix::init(long _nside, long _nside_chunks) {
            nside = _nside;
            nside_chunks = _nside_chunks;

            npix = nside2npix(nside); 
            nchunks = nside2npix(nside_chunks);
            npix_per_chunk = npix/nchunks;

            if (nside_chunks > nside){
                const std::string err_str = "Chunkpix error, choose nside_chunks < nside ";
                throw std::runtime_error(err_str);
            } 
            
            initialize_cmap();
        }

        // Set chunkpix map to 0
        void Chunkpix::initialize_cmap() {
            n_active_chunks = 0;
            active_chunks_values.resize(0);
            active_chunks_values.shrink_to_fit();
            cmap.resize(0);
            cmap.shrink_to_fit();
        }

        // Fraction of memory of a cmap compared to a full map
        float Chunkpix::chunk_sparsity() {
            return (float)n_active_chunks / nchunks;
        }

        // Return the chunk in the full-size map where a pixel belongs to
        int2 Chunkpix::ipix2chunk(int ipix_full) {
            return (int2) (ipix_full / npix_per_chunk);
        }

        // Return the range of the pixels belonging to a chunk
        std::pair<size_t,size_t> Chunkpix::chunk2ipix_ranges(int2 chunk) {
            return {(size_t) npix_per_chunk*chunk, (size_t) npix_per_chunk*(chunk+1)-1 };
        }

        // Increase the count number in cmap for the corresponding pixel.
        // This operation may activate a new chunk.
        void Chunkpix::increase_ipix_count(int ipix_full, float count_increase) {
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
        void Chunkpix::reconstruct_full_map(std::vector<float>& full_map){
            full_map.resize(npix, 0.0);
            std::fill(full_map.begin(), full_map.end(), 0.0);
            
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
            auto c0 = std::accumulate(cmap.begin(), cmap.end(), 0);
            auto c1 = std::accumulate(full_map.begin(), full_map.end(), 0);
            if ( c0 != c1) {
                std::string err_msg = "Error reconstructing full healpix map from chunkpix, task"
                        + std::to_string(FML::ThisTask) + " \n"
                        + std::to_string(c0) + "\t" + std::to_string(c1) + "\t"
                        + std::to_string(counter) + "\n";
                std::cout << err_msg;
                throw std::runtime_error(err_msg);
            }
        }
        
        void Chunkpix::clean(){
            cmap.clear();
            cmap.shrink_to_fit();
            active_chunks_values.clear();
            active_chunks_values.shrink_to_fit();
        }

    } // namespace CHUNKPIX
} // namespace FML


#endif
