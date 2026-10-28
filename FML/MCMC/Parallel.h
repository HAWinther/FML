#ifndef FML_MCMC_PARALLEL_H
#define FML_MCMC_PARALLEL_H

// =============================================================================
// The MPI operations the samplers need (chains and minimizer starts distributed over processes), on top of FML's
// MPI setup (Global.h: FML::ThisTask, FML::NTasks, FML::communicator; MPI is initialized and finalized by FML).
// Compiled with MPI when USE_MPI is defined (make USE_MPI=true); otherwise a single process and every function
// is trivial, so the code using it needs no #ifdefs.
// =============================================================================

#include <vector>

namespace FML::MCMC {

    namespace Parallel {
        [[noreturn]] void abort(int code); // MPI_Abort (stops all processes), else std::exit
        int rank();                        // FML::ThisTask
        int size();                        // FML::NTasks
        bool enabled();                    // compiled with MPI
        void barrier();
        void allreduce_sum(std::vector<double> & v);                      // element-wise sum over processes
        std::vector<double> allgather(const std::vector<double> & local); // concatenation, ordered by rank
    } // namespace Parallel

} // namespace FML::MCMC

#endif
