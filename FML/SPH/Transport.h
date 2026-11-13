#ifndef FML_SPH_TRANSPORT_HEADER
#define FML_SPH_TRANSPORT_HEADER
#include <FML/Global/MPICounts.h>
#include <FML/Global/PeriodicSlabExchange.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
namespace FML::SPH::detail {
    namespace C = FML::COMMUNICATION;
    template <size_t N>
    void agree(const std::array<double, N> & values, const char * what) {
#ifdef USE_MPI
        std::array<double, N> lo{}, hi{};
        MPI_Allreduce(values.data(), lo.data(), int(N), MPI_DOUBLE, MPI_MIN, FML::communicator);
        MPI_Allreduce(values.data(), hi.data(), int(N), MPI_DOUBLE, MPI_MAX, FML::communicator);
        if (lo != hi)
            throw std::invalid_argument(what);
#else
        (void)values;
        (void)what;
#endif
    }
    inline std::vector<std::array<double, 2>> slabs() {
        std::vector<std::array<double, 2>> result;
        C::collective_try([&] { result.resize(FML::NTasks); }, "SPH slab metadata");
        const std::array<double, 2> local{FML::xmin_domain, FML::xmax_domain};
#ifdef USE_MPI
        MPI_Allgather(local.data(), 2, MPI_DOUBLE, result.data(), 2, MPI_DOUBLE, FML::communicator);
#else
        result[0] = local;
#endif
        bool good = result.front()[0] == 0 && result.back()[1] == 1;
        for (size_t j = 0; j < result.size(); ++j)
            good &= std::isfinite(result[j][0]) && std::isfinite(result[j][1]) && result[j][0] <= result[j][1] &&
                    (j == 0 || result[j][0] == result[j - 1][1]);
        if (!good)
            throw std::invalid_argument("[SPH] FML x slabs must partition [0,1]");
        return result;
    }
    template <class Record, class Destination>
    std::vector<Record> route(const std::vector<Record> & input, Destination destination, size_t message_bytes) {
        std::vector<std::vector<Record>> send;
        std::vector<unsigned long long> ns, nr;
        C::collective_try(
            [&] {
                send.resize(FML::NTasks);
                ns.resize(FML::NTasks);
                nr.resize(FML::NTasks);
                for (const auto & record : input) {
                    const int rank = destination(record);
                    if (rank < 0 || rank >= FML::NTasks)
                        throw std::invalid_argument("[SPH] Invalid destination");
                    send[rank].push_back(record);
                }
                for (int rank = 0; rank < FML::NTasks; ++rank)
                    ns[rank] = send[rank].size();
            },
            "SPH route staging");
#ifdef USE_MPI
        MPI_Alltoall(ns.data(), 1, MPI_UNSIGNED_LONG_LONG, nr.data(), 1, MPI_UNSIGNED_LONG_LONG, FML::communicator);
#else
        nr = ns;
#endif
        const auto layout = C::checked_size_layout(nr);
        std::vector<Record> result;
        C::collective_try([&] { result.reserve(layout.total); }, "SPH route reservation");
        C::exchange_records_chunked(
            send,
            [&](const Record * p, size_t n) noexcept { result.insert(result.end(), p, p + n); },
            message_bytes);
        return result;
    }
    inline void unique_ids(const std::vector<std::uint64_t> & ids, size_t bytes) {
        // Hash-partition IDs, never all-gather a catalogue. Check once on initialization.
        auto received = route(
            ids,
            [](std::uint64_t id) {
                id ^= id >> 30;
                id *= 0xbf58476d1ce4e5b9ULL;
                id ^= id >> 27;
                id *= 0x94d049bb133111ebULL;
                id ^= id >> 31;
                return int(id % std::uint64_t(FML::NTasks));
            },
            bytes);
        C::collective_try(
            [&] {
                std::sort(received.begin(), received.end());
                if (std::adjacent_find(received.begin(), received.end()) != received.end())
                    throw std::invalid_argument("[SPH] Particle IDs must be globally unique");
            },
            "SPH global ID validation");
    }
} // namespace FML::SPH::detail
#endif
