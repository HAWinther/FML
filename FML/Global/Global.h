#ifndef GLOBAL_HEADER
#define GLOBAL_HEADER

//===========================================================================
//
// Global information. Keep info about MPI/OMP parallelizations and the domain
// decomposition, type information, macros etc.
//
// Compile time defines:
// USE_MPI               : Use MPI
// NO_AUTO_FML_SETUP     : Do not initialize FML/MPI/FFTW automatically
// USE_OMP               : Use OpenMP
// USE_FFTW              : Use FFTW (here its just for initialization for MPI/threads)
// MEMORY_LOGGING        : Log all (big) allocations with the standard container (see MemoryLogging.h)
//    MIN_BYTES_TO_LOG   : How many bytes to enable logging of allocation
//    MAX_ALLOCATIONS_IN_MEMORY : Maximum number of allocations to keep (above which we give up)
//
//===========================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef MEMORY_LOGGING
#include <FML/MemoryLogging/MemoryLogging.h>
#endif

namespace FML {

    // MPI and OpenMP information
    /// The MPI task number
    extern int ThisTask;
    /// Total number of MPI tasks
    extern int NTasks;
    /// Total number of threads availiable
    extern int NThreads;
    /// If MPI and threads can work together with FFTW
    extern bool MPIThreadsOK;  // Set by init_mpi
    extern bool FFTWThreadsOK; // Set by init_fftw

    /// The local extent of the domain (global domain goes from 0 to 1)
    extern double xmin_domain;
    /// The local extent of the domain (global domain goes from 0 to 1)
    extern double xmax_domain;

    auto uniform_random() -> double;

    //================================================
    // Allocator to allow for logging of memory
    // usage
    //================================================
    template <class T>
#ifdef MEMORY_LOGGING
    using Allocator = FML::LogAllocator<T>;
#else
    using Allocator = std::allocator<T>;
#endif

    //================================================
    // Standard container
    //================================================
    template <class T>
    using Vector = std::vector<T, Allocator<T>>;

    //================================================
    // Integer type for array indices
    //================================================
    using IndexIntType = long long int;

    //================================================
    // MPI functions
    //================================================
    void init_mpi(int * argc, char *** argv);
    void abort_mpi(int exit_code);
    void finalize_mpi();
    void printf_mpi(const char * fmt, ...);
    void info();

    //================================================
    // Initialize the globals within FML
    // Assumes MPI has been initialized
    //================================================
    void init_fml();

    std::pair<double, double> get_system_memory_use();

    /// Gather a single value from all tasks. Value from task ThisTask is stored in values[ThisTask]
    template <class T>
    std::vector<T> GatherFromTasks(T * value) {
        std::vector<T> values(FML::NTasks);
#ifdef USE_MPI
        MPI_Allgather(value, sizeof(T), MPI_BYTE, values.data(), sizeof(T), MPI_BYTE, MPI_COMM_WORLD);
#else
        values[0] = *value;
#endif
        return values;
    }

    /// Inplace max over tasks of a single value
    template <class T>
    void MaxOverTasks([[maybe_unused]] T * value) {
        if (FML::NTasks == 1)
            return;
        std::vector<T> values = GatherFromTasks(value);
        T maxvalue = *value;
        for (auto v : values)
            if (maxvalue < v)
                maxvalue = v;
        *value = maxvalue;
    }
    
    /// Inplace min over tasks of a single value
    template <class T>
    void MinOverTasks([[maybe_unused]] T * value) {
        if (FML::NTasks == 1)
            return;
        std::vector<T> values = GatherFromTasks(value);
        T minvalue = *value;
        for (auto v : values)
            if (minvalue > v)
                minvalue = v;
        *value = minvalue;
    }
    
    /// Inplace sum over tasks of a single value
    template <class T>
    void SumOverTasks([[maybe_unused]] T * value) {
        if (FML::NTasks == 1)
            return;
        std::vector<T> values = GatherFromTasks(value);
        T sum = 0;
        for (auto v : values) {
            sum += v;
        }
        *value = sum;
    }
    
    /// Inplace sum over tasks of a contigious array of values
    template <class T>
    void SumArrayOverTasks([[maybe_unused]] T * value, int n) {
        if (FML::NTasks == 1)
            return;
        for(int i = 0; i < n; i++){
          SumOverTasks(&value[i]);
        }
    }
    
    //============================================
    /// An assert function that calls MPI_Abort
    /// instead of just abort to avoid deadlock
    //============================================
    constexpr void __assert_mpi(const char * expr_str, bool expr, const char * file, int line, const char * msg) {
        if (not expr) {
            std::cout << "[assert_mpi] Assertion failed: [" << expr_str << "], File: [" << file << "], Line: [" << line
                      << "], Message: [" << msg << "]" << std::endl;
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            abort();
        }
    }
#define assert_mpi(Expr, Msg) __assert_mpi(#Expr, Expr, __FILE__, __LINE__, Msg)

    //============================================
    /// Simple integer a^b power-function by squaring
    //============================================
    constexpr long long int power(int base, int exponent) {
        return exponent == 0 ?
                   1 :
                   (exponent % 2 == 0 ?
                        power(base, exponent / 2) * power(base, exponent / 2) :
                        (long long int)(base)*power(base, (exponent - 1) / 2) * power(base, (exponent - 1) / 2));
    }

    //============================================
    // Overloads of arithmetic operations for the
    // container plus elementary math functions
    // using some simple macros
    //============================================

#define OPS(OP)                                                                                                        \
    template <class T>                                                                                                 \
    auto operator OP(const Vector<T> & lhs, const Vector<T> & rhs)->Vector<T> {                                        \
        const size_t nlhs = lhs.size();                                                                                \
        const size_t nrhs = rhs.size();                                                                                \
        if (nlhs != nrhs)                                                                                              \
            throw std::runtime_error("Error vectors need to have the same size:");                                     \
        Vector<T> y(nlhs);                                                                                             \
        for (size_t i = 0; i < nlhs; i++) {                                                                            \
            y[i] = lhs[i] OP rhs[i];                                                                                   \
        }                                                                                                              \
        return y;                                                                                                      \
    }
    OPS(+)
    OPS(-)
    OPS(*)
    OPS(/)
#undef OPS

#define OPS(OP)                                                                                                        \
    template <class T>                                                                                                 \
    auto operator OP(const Vector<T> & lhs, const double & rhs)->Vector<T> {                                           \
        const size_t nlhs = lhs.size();                                                                                \
        Vector<T> y(nlhs);                                                                                             \
        for (size_t i = 0; i < nlhs; i++) {                                                                            \
            y[i] = lhs[i] OP rhs;                                                                                      \
        }                                                                                                              \
        return y;                                                                                                      \
    }
    OPS(+)
    OPS(-)
    OPS(*)
    OPS(/)
#undef OPS

#define OPS(OP)                                                                                                        \
    template <class T>                                                                                                 \
    auto operator OP(const double & lhs, const Vector<T> & rhs)->Vector<T> {                                           \
        const size_t nrhs = rhs.size();                                                                                \
        Vector<T> y(nrhs);                                                                                             \
        for (size_t i = 0; i < nrhs; i++) {                                                                            \
            y[i] = lhs OP rhs[i];                                                                                      \
        }                                                                                                              \
        return y;                                                                                                      \
    }
    OPS(+)
    OPS(-)
    OPS(*)
    OPS(/)
#undef OPS

    template <class T>
    Vector<T> pow(const Vector<T> & x, const double exp) {
        const size_t n = x.size();
        Vector<T> y(n);
        for (size_t i = 0; i < n; i++) {
            y[i] = std::pow(x[i], exp);
        }
        return y;
    }

#define FUNS(FUN)                                                                                                      \
    template <class T>                                                                                                 \
    auto FUN(const Vector<T> & x)->Vector<T> {                                                                         \
        auto op_##FUN = [](double x) -> Vector<T> { return std::FUN(x); };                                             \
        Vector<T> y(x.size());                                                                                         \
        std::transform(x.begin(), x.end(), y.begin(), op_##FUN);                                                       \
        return y;                                                                                                      \
    }
    FUNS(exp)
    FUNS(log)
    FUNS(cos)
    FUNS(sin)
    FUNS(tan)
    FUNS(fabs)
    FUNS(atan)
#undef FUNS
} // namespace FML

namespace FML {

    //=============================================================
    /// Singleton for initializing and finalizing MPI automatically on
    /// startup and exit
    //=============================================================
    struct MPISetup {
        static MPISetup & init(int * argc = nullptr, char *** argv = nullptr) {
            static MPISetup instance(argc, argv);
            return instance;
        }

      private:
        MPISetup(int * argc = nullptr, char *** argv = nullptr) { init_mpi(argc, argv); }

        ~MPISetup() { finalize_mpi(); }
    };

#ifdef USE_FFTW
    namespace GRID {
#include <FML/FFTWGrid/FFTWGlobal.h>
        void init_fftw(int * argc, char *** argv);
        void finalize_fftw();
        void set_fftw_nthreads(int nthreads);
    } // namespace GRID

    //=============================================================
    /// Singleton for initializing and cleaning up FFTW
    /// with MPI and threads automatically
    //=============================================================
    struct FFTWSetup {
        static FFTWSetup & init(int * argc = nullptr, char *** argv = nullptr) {
            static FFTWSetup instance(argc, argv);
            return instance;
        }

      private:
        FFTWSetup(int * argc, char *** argv) { FML::GRID::init_fftw(argc, argv); }

        ~FFTWSetup() { FML::GRID::finalize_fftw(); }
    };
#endif

    //=============================================================
    /// Singleton for initializing and cleaning up FML
    /// Takes care of automatically initializing MPI and FFTW
    //=============================================================
    struct FMLSetup {
        static FMLSetup & init(int * argc = nullptr, char *** argv = nullptr) {
            static FMLSetup instance(argc, argv);
            return instance;
        }

      private:
        FMLSetup(int * argc, char *** argv) {
            // Initialize MPI
            [[maybe_unused]] MPISetup & m = MPISetup::init(argc, argv);

#ifdef USE_FFTW
            // Initialize FFTW
            [[maybe_unused]] FFTWSetup & f = FFTWSetup::init(argc, argv);
#endif

            // Initialize FML
            init_fml();

            // Show some info to see that all is ok
            info();
        }

        ~FMLSetup() = default;
    };
} // namespace FML

#endif
