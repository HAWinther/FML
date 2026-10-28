#ifndef FML_CHUNKED_EXCHANGE_HEADER
#define FML_CHUNKED_EXCHANGE_HEADER

#include <FML/Global/Global.h>
#include <climits>
#include <limits>
#include <type_traits>
#include <vector>

namespace FML::COMMUNICATION {
    namespace detail {
        template <class Record>
        constexpr size_t records_per_chunk(size_t bytes) {
            return std::min(bytes, size_t(INT_MAX)) / sizeof(Record);
        }

        constexpr unsigned long long chunk_count(unsigned long long count, unsigned long long capacity) {
            return count / capacity + (count % capacity != 0);
        }
    } // namespace detail

    /// Consume outgoing records on their destination rank, without a flattened send buffer
    /// or a full receive buffer. Clears outgoing, including its per-peer capacities.
    /// Collective on FML::communicator; all ranks must call, including ranks without data.
    /// Records must be trivially copyable and have the same representation on every rank.
    /// consume(const Record *, size_t) runs synchronously; pointers are valid only during
    /// the call. It must not throw, retain pointers, or invoke MPI collectives.
    /// max_message_bytes bounds each MPI payload and receive scratch (default 8 MiB).
    /// Different limits across ranks use the smallest whole-record capacity.
    template <class Record, class Consume>
    void exchange_records_chunked(std::vector<std::vector<Record>> & outgoing,
                                  Consume && consume,
                                  size_t max_message_bytes = 8 * 1024 * 1024) {
        static_assert(std::is_trivially_copyable_v<Record>, "MPI records must be trivially copyable");
        static_assert(sizeof(size_t) <= sizeof(unsigned long long), "MPI record counts must fit unsigned long long");
        assert_mpi(outgoing.size() == size_t(FML::NTasks), "[exchange_records_chunked] Wrong number of peers");
        unsigned long long capacity = detail::records_per_chunk<Record>(max_message_bytes);
        FML::MinOverTasks(&capacity);
        assert_mpi(capacity > 0, "[exchange_records_chunked] Message limit cannot hold one record");

        auto & self = outgoing[FML::ThisTask];
        if (!self.empty())
            consume(self.data(), self.size());
        std::vector<Record>().swap(self);
#ifdef USE_MPI
        for (int step = 1; step < FML::NTasks; ++step) {
            const int dest = int((static_cast<long long>(FML::ThisTask) + step) % FML::NTasks);
            const int source = int((static_cast<long long>(FML::ThisTask) - step + FML::NTasks) % FML::NTasks);
            auto & send = outgoing[dest];
            const unsigned long long nsend = send.size();
            unsigned long long nrecv = 0;
            MPI_Sendrecv(&nsend, 1, MPI_UNSIGNED_LONG_LONG, dest, 0,
                         &nrecv, 1, MPI_UNSIGNED_LONG_LONG, source, 0,
                         FML::communicator, MPI_STATUS_IGNORE);
            assert_mpi(nrecv <= std::numeric_limits<size_t>::max(),
                       "[exchange_records_chunked] Receive count exceeds addressable range");
            auto rounds = detail::chunk_count(std::max(nsend, nrecv), capacity);
            // Every rank executes the same number of rounds: a zero-count edge must
            // still match its peer while another edge is transferring several chunks.
            FML::MaxOverTasks(&rounds);
            // Construct at the required size rather than growing a reused vector,
            // whose geometric capacity growth could exceed the scratch budget.
            std::vector<Record> receive(size_t(std::min(capacity, nrecv)));
            for (unsigned long long round = 0; round < rounds; ++round) {
                const auto offset = round * capacity;
                const auto scount = offset < nsend ? std::min(capacity, nsend - offset) : 0;
                const auto rcount = offset < nrecv ? std::min(capacity, nrecv - offset) : 0;
                // Whole-record chunking proves both byte counts fit int before narrowing.
                MPI_Sendrecv(scount ? send.data() + size_t(offset) : nullptr,
                             int(scount * sizeof(Record)), MPI_BYTE, dest, 1,
                             rcount ? receive.data() : nullptr,
                             int(rcount * sizeof(Record)), MPI_BYTE, source, 1,
                             FML::communicator, MPI_STATUS_IGNORE);
                if (rcount)
                    consume(receive.data(), size_t(rcount));
            }
            std::vector<Record>().swap(send);
        }
#endif
    }
} // namespace FML::COMMUNICATION
#endif
