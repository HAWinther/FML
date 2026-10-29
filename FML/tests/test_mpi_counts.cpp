#include <FML/Global/MPICounts.h>
#include <climits>
#include <iostream>
#include <limits>
using namespace FML::COMMUNICATION;
int failures = 0;
void check(bool good, const char *message) {
    int bad = !good;
    FML::MaxOverTasks(&bad);
    failures += bad;
    if (!FML::ThisTask)
        std::cout << (bad ? "[FAIL] " : "[PASS] ") << message << '\n';
}
template <class F> void rejects(F &&function, const char *message) {
    bool caught = false;
    try {
        function();
    } catch (const std::overflow_error &) {
        caught = true;
    }
    check(caught, message);
}
struct Record {
    int owner, index;
    double value;
};
int main() {
    Count value = std::numeric_limits<Count>::max(), output = 91;
    check(!add_count(value, 1) && value == std::numeric_limits<Count>::max(),
          "overflowing count addition leaves its input intact");
    check(!multiply_count(value, 2, output) && output == 91, "overflowing byte products fail before multiplication");
    check(multiply_count(value, 0, output) && output == 0, "zero-size products are safe");
    auto wide = checked_size_layout({Count(INT_MAX) + 7, 12});
    check(wide.offsets[1] == size_t(INT_MAX) + 7 && wide.total == size_t(INT_MAX) + 19,
          "wide migration layouts preserve counts above INT_MAX without allocating payloads");
    auto edge = checked_int_layout(std::vector<Count>{INT_MAX, 0});
    check(edge.counts[0] == INT_MAX && edge.offsets[1] == INT_MAX && edge.total == size_t(INT_MAX),
          "MPI int layout accepts its exact boundary");
    rejects([] { checked_int_layout(std::vector<Count>{Count(INT_MAX) + 1}); },
            "oversized individual MPI counts are rejected");
    rejects([] { checked_int_layout(std::vector<Count>{INT_MAX, 1}); }, "aggregate displacement overflow is rejected");
    rejects([] { checked_int_layout(std::vector<int>{1, -1}); }, "negative receive counts are rejected");
    rejects([] { checked_int_layout(std::vector<Count>{Count(INT_MAX) / 3 + 1}, 3); },
            "field-scaled layouts are checked before narrowing");
    rejects([] { checked_size_layout({std::numeric_limits<Count>::max(), 1}); }, "wide aggregate overflow is rejected");
    rejects([] { checked_bytes(std::numeric_limits<Count>::max(), 2); }, "serialized byte products cannot wrap size_t");
    rejects([] { checked_int_layout(std::vector<Count>{FML::ThisTask == 0 ? Count(INT_MAX) + 1 : 1}); },
            "one-rank layout failure is collective and recoverable");
#ifdef USE_MPI
    std::vector<Count> send(FML::NTasks), receive(FML::NTasks);
    send[(FML::ThisTask + 1) % FML::NTasks] = Count(INT_MAX) + 100 + FML::ThisTask;
    MPI_Alltoall(send.data(), 1, MPI_UNSIGNED_LONG_LONG, receive.data(), 1, MPI_UNSIGNED_LONG_LONG, FML::communicator);
    int source = (FML::ThisTask + FML::NTasks - 1) % FML::NTasks;
    check(receive[source] == Count(INT_MAX) + 100 + source,
          "migration count wire format preserves more than 2^31 records");
#endif
    void (*wide_reduction)(double *, size_t) = &FML::SumArrayOverTasks<double>;
    double array[3] = {double(FML::ThisTask), 1., 2.};
    wide_reduction(array, size_t(3));
    check(array[0] == double(FML::NTasks) * (FML::NTasks - 1) / 2 && array[1] == FML::NTasks &&
              array[2] == 2 * FML::NTasks,
          "array reduction accepts size_t counts and preserves sums");
    std::vector<Record> local;
    const int n = FML::ThisTask % 3 + (FML::ThisTask == FML::NTasks - 1);
    for (int i = 0; i < n; ++i)
        local.push_back({FML::ThisTask, i, .5 + FML::ThisTask + i});
    auto validate = [](const std::vector<Record> &records) {
        size_t cursor = 0;
        for (int rank = 0; rank < FML::NTasks; ++rank)
            for (int i = 0; i < rank % 3 + (rank == FML::NTasks - 1); ++i) {
                if (cursor >= records.size())
                    return false;
                const auto &r = records[cursor++];
                if (r.owner != rank || r.index != i || r.value != .5 + rank + i)
                    return false;
            }
        return cursor == records.size();
    };
    const auto all = gather_records_checked(local, true);
    check(validate(all), "checked all-gather preserves record fields and rank order, including empty ranks");
    const int root = FML::NTasks - 1;
    const auto gathered = gather_records_checked(local, false, root);
    check(FML::ThisTask == root ? validate(gathered) : gathered.empty(),
          "checked gather supports a nonzero root and empty non-root results");
    rejects([&] { gather_records_checked(local, true, 0, 0); },
            "gather rejects an aggregate limit before payload communication");
    check(validate(gather_records_checked(local, true)),
          "valid communication succeeds after collective count failures");
    if (!FML::ThisTask)
        std::cout << (failures ? "MPI count tests FAILED\n" : "All MPI count tests passed\n");
    return failures ? 1 : 0;
}
