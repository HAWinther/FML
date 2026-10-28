#include <FML/Global/Global.h>
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
            MPI_Allreduce(MPI_IN_PLACE, v.data(), int(v.size()), MPI_DOUBLE, MPI_SUM, FML::communicator);
        }
        std::vector<double> allgather(const std::vector<double> & local) {
            const int n = size();
            int mine = int(local.size());
            std::vector<int> counts(n), displs(n, 0);
            MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, FML::communicator);
            for (int i = 1; i < n; i++)
                displs[i] = displs[i - 1] + counts[i - 1];
            std::vector<double> all(size_t(displs[n - 1] + counts[n - 1]));
            MPI_Allgatherv(local.data(),
                           mine,
                           MPI_DOUBLE,
                           all.data(),
                           counts.data(),
                           displs.data(),
                           MPI_DOUBLE,
                           FML::communicator);
            return all;
        }
#else
        bool enabled() { return false; }
        void barrier() {}
        void allreduce_sum(std::vector<double> &) {}
        std::vector<double> allgather(const std::vector<double> & local) { return local; }
#endif
    } // namespace Parallel

} // namespace FML::MCMC
