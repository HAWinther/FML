#include <FML/Global/Global.h>
#include <FML/Global/MPICounts.h>
#include <FML/MCMC/Parallel.h>

#include <cstdlib>

namespace FML::MCMC {

    namespace Parallel {
        void abort(int code) {
            FML::abort_mpi(code);
            std::exit(code);
        }
        int rank() { return FML::ThisTask; }
        int size() { return FML::NTasks; }
#ifdef USE_MPI
        bool enabled() { return true; }
        void barrier() { MPI_Barrier(FML::communicator); }
        void allreduce_sum(std::vector<double> & v) {
            FML::COMMUNICATION::require_count(v.size() <= size_t(INT_MAX), "[MCMC] Reduction count exceeds MPI int");
            auto count = FML::COMMUNICATION::Count(v.size());
            const auto counts = FML::GatherFromTasks(&count);
            FML::COMMUNICATION::require_count(
                std::all_of(counts.begin(), counts.end(), [&](auto n) { return n == count; }),
                "[MCMC] Reduction lengths differ between ranks");
            MPI_Allreduce(MPI_IN_PLACE, v.data(), int(v.size()), MPI_DOUBLE, MPI_SUM, FML::communicator);
        }
        std::vector<double> allgather(const std::vector<double> & local) {
            return FML::COMMUNICATION::gather_records_checked(local, true);
        }
#else
        bool enabled() { return false; }
        void barrier() {}
        void allreduce_sum(std::vector<double> &) {}
        std::vector<double> allgather(const std::vector<double> & local) { return local; }
#endif
    } // namespace Parallel

} // namespace FML::MCMC
