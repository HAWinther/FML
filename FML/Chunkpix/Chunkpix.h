#ifdef USE_HEALPIX
#ifndef CHUNKPIX_HEADER
#define CHUNKPIX_HEADER

#include <FML/Global/Global.h>

// Chunkpix is created by Albert Izard

namespace FML {

    namespace CHUNKPIX {
        
        typedef uint16_t int2;
        
        // template <class T>
        class Chunkpix {
            private:
                long nside{};
                long nside_chunks{};
                long npix{};
                int nchunks{};
                int npix_per_chunk{};
                
                // Data
                std::vector<float> cmap;
                std::vector<int2> active_chunks_values;
                int n_active_chunks{};

            public:
                Chunkpix() = default;

                void init(long nside, long nside_chunks);

                void initialize_cmap();

                float chunk_sparsity();

                int2 ipix2chunk(int ipix_full);
                
                std::pair<size_t,size_t> chunk2ipix_ranges(int2 chunk);

                void increase_ipix_count(int ipix_full, float count_increase);

                void reconstruct_full_map(std::vector<float> & full_map);

                void deallocate_full_map();

                void clean();

        };

    } // namespace CHUNKPIX
} // namespace FML

#endif
#endif
