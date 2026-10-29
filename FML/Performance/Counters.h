#ifndef FML_PERFORMANCE_COUNTERS_H
#define FML_PERFORMANCE_COUNTERS_H
#include <array>
#include <atomic>
#include <cstddef>
#include <limits>

namespace FML::PERFORMANCE {
    using Count = unsigned long long;
    enum class Counter : size_t {
        GridAllocations,
        GridAllocatedBytes,
        GridCopyOperations,
        GridCopiedBytes,
        ForwardFFTs,
        InverseFFTs,
        PayloadSendrecvCalls,
        PayloadSendMessages,
        PayloadSentBytes,
        PayloadReceivedBytes,
        TransportBufferAllocations,
        TransportBufferAllocatedBytes,
        TransportPackedBytes,
        TransportUnpackedBytes,
        FFTPlans,
        Size
    };
    constexpr size_t number_of_counters = size_t(Counter::Size);
    inline constexpr std::array<const char *, number_of_counters> names{"grid_storage_allocations",
                                                                        "grid_storage_allocated_bytes",
                                                                        "grid_copy_operations",
                                                                        "grid_copied_bytes",
                                                                        "forward_ffts",
                                                                        "inverse_ffts",
                                                                        "payload_sendrecv_calls",
                                                                        "payload_send_messages",
                                                                        "payload_sent_bytes",
                                                                        "payload_received_bytes",
                                                                        "transport_buffer_allocations",
                                                                        "transport_buffer_allocated_bytes",
                                                                        "transport_packed_bytes",
                                                                        "transport_unpacked_bytes",
                                                                        "fftw_plan_creations"};
    struct Snapshot {
        std::array<Count, number_of_counters> values{};
        bool overflow = false;
        Count operator[](Counter counter) const { return values[size_t(counter)]; }
    };
#ifdef FML_ENABLE_PERFORMANCE_COUNTERS
    inline constexpr bool enabled = true;
    namespace detail {
        struct State {
            std::array<std::atomic<Count>, number_of_counters> values;
            std::atomic<bool> overflow{false};
            State() {
                for (auto &value : values)
                    value.store(0, std::memory_order_relaxed);
            }
        };
        inline State &state() {
            static State state;
            return state;
        }
    } // namespace detail
    inline void record(Counter counter, Count amount = 1) noexcept {
        auto &s = detail::state();
        auto &value = s.values[size_t(counter)];
        Count old = value.load(std::memory_order_relaxed);
        while (true) {
            const bool overflow = amount > std::numeric_limits<Count>::max() - old;
            const Count next = overflow ? std::numeric_limits<Count>::max() : old + amount;
            if (value.compare_exchange_weak(old, next, std::memory_order_relaxed)) {
                if (overflow)
                    s.overflow.store(true, std::memory_order_relaxed);
                return;
            }
        }
    }
    /// Reset/snapshot at quiescent boundaries; a reset is process-local and is not an MPI collective.
    inline void reset() noexcept {
        auto &s = detail::state();
        for (auto &value : s.values)
            value.store(0, std::memory_order_relaxed);
        s.overflow.store(false, std::memory_order_relaxed);
    }
    inline Snapshot snapshot() noexcept {
        Snapshot result;
        auto &s = detail::state();
        for (size_t i = 0; i < number_of_counters; ++i)
            result.values[i] = s.values[i].load(std::memory_order_relaxed);
        result.overflow = s.overflow.load(std::memory_order_relaxed);
        return result;
    }
#else
    inline constexpr bool enabled = false;
    inline void record(Counter, Count = 1) noexcept {}
    inline void reset() noexcept {}
    inline Snapshot snapshot() noexcept { return {}; }
#endif
    inline void grid_allocation(size_t bytes) noexcept {
        if (bytes) {
            record(Counter::GridAllocations);
            record(Counter::GridAllocatedBytes, bytes);
        }
    }
    inline void grid_copy(size_t bytes) noexcept {
        if (bytes) {
            record(Counter::GridCopyOperations);
            record(Counter::GridCopiedBytes, bytes);
        }
    }
    inline void transport_allocation(size_t bytes) noexcept {
        if (bytes) {
            record(Counter::TransportBufferAllocations);
            record(Counter::TransportBufferAllocatedBytes, bytes);
        }
    }
    inline void payload(size_t sent, size_t received) noexcept {
        record(Counter::PayloadSendrecvCalls);
        if (sent)
            record(Counter::PayloadSendMessages);
        record(Counter::PayloadSentBytes, sent);
        record(Counter::PayloadReceivedBytes, received);
    }
} // namespace FML::PERFORMANCE
#endif
