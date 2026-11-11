#ifndef FML_PEANO_HILBERT_DECOMPOSITION_H
#define FML_PEANO_HILBERT_DECOMPOSITION_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

/// Standalone Peano-Hilbert keys and MPI key-range decomposition.
///
/// The ownership ranges are contiguous in Hilbert-key order. This module does
/// not migrate particles or modify FML::MPIParticles.
namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT {

    using Key = uint64_t;
    using Position = std::array<double, 3>;

    /// Return the number of usable bits per coordinate for a 64-bit key.
    constexpr unsigned max_coordinate_bits() { return 21; }

    namespace detail {
        inline void validate_bits(unsigned bits) {
            if (bits == 0 || bits > max_coordinate_bits())
                throw std::invalid_argument("Peano-Hilbert bits must be in [1, 21]");
        }

        // Skilling's axes-to-transpose transform, followed by bit interleaving.
        // Reference: J. Skilling, AIP Conf. Proc. 707, 381 (2004), doi:10.1063/1.1751381.
        inline Key key_from_integer_coordinates(std::array<uint32_t, 3> x, unsigned bits) {
            validate_bits(bits);
            const uint32_t limit = uint32_t(1) << bits;
            for (auto & axis : x)
                if (axis >= limit)
                    throw std::out_of_range("Peano-Hilbert integer coordinate is outside its grid");

            const uint32_t top = uint32_t(1) << (bits - 1);
            for (uint32_t q = top; q > 1; q >>= 1) {
                const uint32_t p = q - 1;
                for (int i = 0; i < 3; ++i) {
                    if (x[i] & q) {
                        x[0] ^= p;
                    } else {
                        const uint32_t t = (x[0] ^ x[i]) & p;
                        x[0] ^= t;
                        x[i] ^= t;
                    }
                }
            }
            for (int i = 1; i < 3; ++i)
                x[i] ^= x[i - 1];
            uint32_t t = 0;
            for (uint32_t q = top; q > 1; q >>= 1)
                if (x[2] & q)
                    t ^= q - 1;
            for (auto & axis : x)
                axis ^= t;

            Key key = 0;
            for (int bit = int(bits) - 1; bit >= 0; --bit)
                for (int axis = 0; axis < 3; ++axis)
                    key = (key << 1) | ((x[axis] >> bit) & 1U);
            return key;
        }

        inline std::array<uint32_t, 3> integer_coordinates_from_key(Key key, unsigned bits) {
            validate_bits(bits);
            const unsigned nkeybits = 3 * bits;
            if (key >= (Key(1) << nkeybits))
                throw std::out_of_range("Peano-Hilbert key is outside its configured resolution");

            // Undo the bit interleave to recover Skilling's transposed axes.
            std::array<uint32_t, 3> x{};
            for (int bit = int(bits) - 1; bit >= 0; --bit)
                for (int axis = 0; axis < 3; ++axis) {
                    x[axis] |= uint32_t((key >> (3 * bit + (2 - axis))) & 1U) << bit;
                }

            // Gray decode, then undo the exchange loop.
            const uint32_t gray = x[2] >> 1;
            for (int axis = 2; axis > 0; --axis)
                x[axis] ^= x[axis - 1];
            x[0] ^= gray;

            // Undo the excess exchange work.
            const uint32_t limit = uint32_t(1) << bits;
            for (uint32_t q = 2; q < limit; q <<= 1) {
                const uint32_t p = q - 1;
                for (int axis = 2; axis >= 0; --axis) {
                    if (x[axis] & q) {
                        x[0] ^= p;
                    } else {
                        const uint32_t t = (x[0] ^ x[axis]) & p;
                        x[0] ^= t;
                        x[axis] ^= t;
                    }
                }
            }
            return x;
        }

        inline int communicator_rank(
#ifdef USE_MPI
            MPI_Comm comm
#else
            int
#endif
        ) {
#ifdef USE_MPI
            int rank = 0;
            MPI_Comm_rank(comm, &rank);
            return rank;
#else
            return 0;
#endif
        }

        inline int communicator_size(
#ifdef USE_MPI
            MPI_Comm comm
#else
            int
#endif
        ) {
#ifdef USE_MPI
            int size = 1;
            MPI_Comm_size(comm, &size);
            return size;
#else
            return 1;
#endif
        }
    } // namespace detail

    /// Map a point in the unit periodic cube [0,1)^3 to a Hilbert key.
    inline Key key_from_position(const Position & position, unsigned bits) {
        detail::validate_bits(bits);
        const uint32_t ngrid = uint32_t(1) << bits;
        std::array<uint32_t, 3> cell{};
        for (int axis = 0; axis < 3; ++axis) {
            const double x = position[axis];
            if (!std::isfinite(x) || x < 0.0 || x >= 1.0)
                throw std::out_of_range("Peano-Hilbert positions must be finite and in [0,1)");
            cell[axis] = std::min(uint32_t(x * ngrid), ngrid - 1);
        }
        return detail::key_from_integer_coordinates(cell, bits);
    }

    /// Map an integer grid coordinate to a Hilbert key.
    inline Key key_from_grid_coordinate(const std::array<uint32_t, 3> & coordinate, unsigned bits) {
        return detail::key_from_integer_coordinates(coordinate, bits);
    }

    /// Inverse of key_from_grid_coordinate for the same bit resolution.
    inline std::array<uint32_t, 3> grid_coordinate_from_key(Key key, unsigned bits) {
        return detail::integer_coordinates_from_key(key, bits);
    }

    /// Rank intervals in Hilbert-key order, constructed from a collective key histogram.
    class Decomposition {
      public:
        Decomposition() = default;

        /// Build from each rank's local particle keys. All ranks in comm call this
        /// collectively with identical bits and histogram_bits.
        ///
        /// histogram_bits controls the memory/quantile resolution (default 16,
        /// at most 20). It is a prefix histogram over the full Hilbert key.
        /// Points with identical keys cannot be split across ranks.
        static Decomposition build(const std::vector<Key> & local_keys,
#ifdef USE_MPI
                                  MPI_Comm comm = MPI_COMM_WORLD,
#else
                                  int comm = 0,
#endif
                                  unsigned bits = 21,
                                  unsigned histogram_bits = 16) {
            const int nranks = detail::communicator_size(comm);
            const int rank = detail::communicator_rank(comm);
            if (nranks < 1)
                throw std::runtime_error("Peano-Hilbert decomposition requires at least one MPI rank");

#ifdef USE_MPI
            int local_config[2] = {int(bits), int(histogram_bits)};
            int min_config[2] = {}, max_config[2] = {};
            MPI_Allreduce(local_config, min_config, 2, MPI_INT, MPI_MIN, comm);
            MPI_Allreduce(local_config, max_config, 2, MPI_INT, MPI_MAX, comm);
            if (min_config[0] != max_config[0] || min_config[1] != max_config[1])
                throw std::invalid_argument("All ranks must use the same Peano-Hilbert resolution");
#endif
            detail::validate_bits(bits);
            const unsigned key_bits = 3 * bits;
            if (histogram_bits == 0 || histogram_bits > 20 || histogram_bits > key_bits)
                throw std::invalid_argument("Peano-Hilbert histogram_bits must be in [1, min(20, 3*bits)]");

            const Key key_limit = Key(1) << key_bits;
            bool local_valid = true;
            for (const Key key : local_keys)
                local_valid = local_valid && key < key_limit;
#ifdef USE_MPI
            int valid = local_valid ? 1 : 0;
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, comm);
            if (!valid)
                throw std::out_of_range("A rank supplied a key outside the configured Hilbert resolution");

#else
            if (!local_valid)
                throw std::out_of_range("A key is outside the configured Hilbert resolution");
#endif

            const size_t nbins = size_t(1) << histogram_bits;
            std::vector<uint64_t> histogram(nbins, 0);
            const unsigned shift = key_bits - histogram_bits;
            for (const Key key : local_keys) {
                auto & count = histogram[size_t(key >> shift)];
                if (count == std::numeric_limits<uint64_t>::max())
                    throw std::overflow_error("Local Peano-Hilbert histogram overflow");
                ++count;
            }
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, histogram.data(), int(nbins), MPI_UINT64_T, MPI_SUM, comm);
#endif

            uint64_t total = 0;
            for (const auto count : histogram) {
                if (count > std::numeric_limits<uint64_t>::max() - total)
                    throw std::overflow_error("Global Peano-Hilbert particle count overflow");
                total += count;
            }

            Decomposition result;
            result.bits_ = bits;
            result.histogram_bits_ = histogram_bits;
            result.rank_ = rank;
            result.nranks_ = nranks;
            result.total_particles_ = total;
            result.boundaries_.reserve(size_t(nranks) + 1);
            result.boundaries_.push_back(0);

            // Place each cut at whichever edge of the quantile bin is closer to
            // its target. This avoids assigning an entire large bin to one rank
            // where the opposite edge would produce a better balance.
            uint64_t cumulative = 0;
            size_t bin = 0;
            for (int r = 1; r < nranks; ++r) {
                const uint64_t target = (total / uint64_t(nranks)) * uint64_t(r) +
                                        ((total % uint64_t(nranks)) * uint64_t(r)) / uint64_t(nranks);
                while (bin < nbins && cumulative + histogram[bin] < target) {
                    cumulative += histogram[bin];
                    ++bin;
                }
                size_t cut_bin = bin;
                if (bin < nbins) {
                    const uint64_t before = cumulative;
                    const uint64_t after = cumulative + histogram[bin];
                    if (target - before > after - target)
                        cut_bin = bin + 1;
                }
                const Key cut = Key(cut_bin) << shift;
                result.boundaries_.push_back(std::max(result.boundaries_.back(), cut));
            }
            result.boundaries_.push_back(key_limit);
            return result;
        }

        int owner_of_key(Key key) const {
            require_built();
            if (key >= key_limit())
                throw std::out_of_range("Peano-Hilbert key is outside this decomposition");
            return int(std::upper_bound(boundaries_.begin(), boundaries_.end(), key) - boundaries_.begin()) - 1;
        }

        int owner_of_position(const Position & position) const { return owner_of_key(key_from_position(position, bits_)); }

        Key key_begin(int rank) const {
            require_rank(rank);
            return boundaries_[size_t(rank)];
        }
        Key key_end(int rank) const {
            require_rank(rank);
            return boundaries_[size_t(rank) + 1];
        }
        unsigned bits() const { require_built(); return bits_; }
        unsigned histogram_bits() const { require_built(); return histogram_bits_; }
        int rank() const { require_built(); return rank_; }
        int nranks() const { require_built(); return nranks_; }
        uint64_t total_particles() const { require_built(); return total_particles_; }

      private:
        Key key_limit() const { return Key(1) << (3 * bits_); }
        void require_built() const {
            if (nranks_ == 0)
                throw std::logic_error("Peano-Hilbert decomposition has not been built");
        }
        void require_rank(int rank) const {
            require_built();
            if (rank < 0 || rank >= nranks_)
                throw std::out_of_range("Peano-Hilbert rank is outside the decomposition");
        }

        unsigned bits_{0};
        unsigned histogram_bits_{0};
        int rank_{0};
        int nranks_{0};
        uint64_t total_particles_{0};
        std::vector<Key> boundaries_;
    };

} // namespace FML::DOMAIN_DECOMPOSITION::PEANO_HILBERT

#endif
