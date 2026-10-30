#ifndef FML_MPI_COUNTS_HEADER
#define FML_MPI_COUNTS_HEADER
#include <FML/Global/Global.h>
#include <climits>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace FML::COMMUNICATION {
    using Count = unsigned long long;
    static_assert(sizeof(size_t) <= sizeof(Count));

    inline bool add_count(Count & sum, Count value, Count limit = std::numeric_limits<Count>::max()) {
        if (sum > limit || value > limit - sum)
            return false;
        sum += value;
        return true;
    }
    inline bool multiply_count(Count a, Count b, Count & result, Count limit = std::numeric_limits<Count>::max()) {
        if (b && a > limit / b)
            return false;
        result = a * b;
        return true;
    }
    inline void require_count(bool valid, const char * message) {
        int good = valid;
        FML::MinOverTasks(&good);
        if (!good)
            throw std::overflow_error(message);
    }
    inline size_t checked_bytes(Count count, Count extent, Count limit = std::numeric_limits<size_t>::max()) {
        Count bytes = 0;
        require_count(multiply_count(count, extent, bytes, std::min(limit, Count(std::numeric_limits<size_t>::max()))),
                      "[MPI counts] Byte product exceeds supported range");
        return size_t(bytes);
    }
    struct SizeLayout {
        std::vector<size_t> offsets;
        size_t total = 0;
    };
    inline SizeLayout checked_size_layout(const std::vector<Count> & counts,
                                          size_t limit = std::numeric_limits<size_t>::max()) {
        SizeLayout result;
        result.offsets.resize(counts.size());
        Count total = 0;
        bool valid = true;
        for (size_t i = 0; i < counts.size(); ++i) {
            result.offsets[i] = size_t(total);
            if (!add_count(total, counts[i], limit)) {
                valid = false;
                break;
            }
        }
        require_count(valid, "[MPI counts] Aggregate count exceeds addressable storage");
        result.total = size_t(total);
        return result;
    }
    struct IntLayout {
        std::vector<int> counts, offsets;
        size_t total = 0;
    };
    template <class Integer>
    IntLayout checked_int_layout(const std::vector<Integer> & counts, Count scale = 1, Count limit = INT_MAX) {
        static_assert(std::is_integral_v<Integer>);
        IntLayout result;
        result.counts.resize(counts.size());
        result.offsets.resize(counts.size());
        Count total = 0;
        bool valid = scale > 0 && limit <= Count(INT_MAX);
        for (size_t i = 0; valid && i < counts.size(); ++i) {
            if constexpr (std::is_signed_v<Integer>)
                if (counts[i] < 0) {
                    valid = false;
                    break;
                }
            Count count = 0;
            valid = multiply_count(Count(counts[i]), scale, count, limit);
            const Count offset = total;
            valid = valid && add_count(total, count, limit);
            if (valid) {
                result.counts[i] = int(count);
                result.offsets[i] = int(offset);
            }
        }
        require_count(valid, "[MPI counts] Counts or displacements exceed the MPI int layout limit");
        result.total = size_t(total);
        return result;
    }

    /// Checked legacy Gatherv/Allgatherv for homogeneous trivially copyable records.
    /// Aggregate records are conservatively limited to INT_MAX; no payload chunking.
    template <class T>
    std::vector<T>
    gather_records_checked(const std::vector<T> & local, bool replicate = false, int root = 0, Count limit = INT_MAX) {
        static_assert(std::is_trivially_copyable_v<T>);
        require_count(root >= 0 && root < FML::NTasks && sizeof(T) <= size_t(INT_MAX),
                      "[MPI counts] Invalid gather root or record extent");
        Count n = local.size();
        const auto sizes = FML::GatherFromTasks(&n);
        const auto layout = checked_int_layout(sizes, 1, limit);
        require_count(layout.total <= local.max_size(), "[MPI counts] Gather exceeds vector capacity");
        checked_bytes(layout.total, sizeof(T), std::numeric_limits<std::ptrdiff_t>::max());
        if (FML::NTasks == 1)
            return local;
        std::vector<T> result;
#ifdef USE_MPI
        if (replicate || FML::ThisTask == root)
            result.resize(layout.total);
        MPI_Datatype type;
        MPI_Type_contiguous(int(sizeof(T)), MPI_BYTE, &type);
        MPI_Type_commit(&type);
        if (replicate)
            MPI_Allgatherv(local.data(),
                           layout.counts[FML::ThisTask],
                           type,
                           result.data(),
                           layout.counts.data(),
                           layout.offsets.data(),
                           type,
                           FML::communicator);
        else
            MPI_Gatherv(local.data(),
                        layout.counts[FML::ThisTask],
                        type,
                        result.data(),
                        layout.counts.data(),
                        layout.offsets.data(),
                        type,
                        root,
                        FML::communicator);
        MPI_Type_free(&type);
#else
        (void)replicate;
#endif
        return result;
    }
} // namespace FML::COMMUNICATION
#endif
