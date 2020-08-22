#include <iomanip>
#include <iostream>
#include <random>
#include <stdarg.h>

#include "Global.h"
    
static std::string processor_name {"IsOnlyKnownWithMPI"};

namespace FML {

#ifndef NO_AUTO_MPI_SETUP
    //============================================
    // Automatic init and cleanup of MPI
    // If arguments are needed call it instead from main as:
    // FML::MPISetup & mpisetup = FML::MPISetup::init(&argc, &argv);
    //============================================
    MPISetup & mpisetup = MPISetup::init();
#endif

    //============================================
    // Automaticall init and cleanup of FFTW
    // with MPI and/or threads. Use the define
    // NO_AUTO_FFTW_SETUP to not do this and init
    // it yourself. Can instead be called with arguments from main as:
    // FML::FFTWSetup & fftwsetup = FML::FFTWSetup::init(&argv, &argv);
    //============================================
#ifndef NO_AUTO_FFTW_SETUP
#ifdef USE_FFTW
    [[maybe_unused]] FFTWSetup & fftwsetup = FFTWSetup::init();
#endif
#endif

#ifdef MEMORY_LOGGING
    // If we use memory logging initialize it
    [[maybe_unused]] MemoryLog * MemoryLog::instance = nullptr;
#endif

    // Number of MPI tasks and (max) OpenMP threads
    int ThisTask = 0;
    int NTasks = 1;
    int NThreads = 1;

    bool FFTWThreadsOK = false;
    bool MPIThreadsOK = false;

    // The local extent of the domain (global domain goes from 0 to 1)
    double xmin_domain = 0.0;
    double xmax_domain = 1.0;

    // A simple random [0,1) generator
    std::mt19937 generator;
    double uniform_random() {
        auto udist = std::uniform_real_distribution<double>(0.0, 1.0);
        return udist(generator);
    }

    //============================================
    // Initialize MPI
    //============================================
    void init_mpi([[maybe_unused]] int * argc, [[maybe_unused]] char *** argv) {
        ThisTask = 0;
        NTasks = 1;
        xmin_domain = 0.0;
        xmax_domain = 1.0;

        // Initialize MPI
#ifdef USE_MPI
#ifdef USE_OMP
        int provided;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
        MPIThreadsOK = provided >= MPI_THREAD_FUNNELED;
#else
        MPI_Init(argc, argv);
#endif
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        MPI_Comm_size(MPI_COMM_WORLD, &NTasks);

        int plen;
        char pname[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(pname, &plen);
        processor_name = std::string(pname);
#endif

        // Set range for local domain
        xmin_domain = ThisTask / double(NTasks);
        xmax_domain = (ThisTask + 1) / double(NTasks);

        std::vector<double> xmin_over_tasks(NTasks, 0);
        std::vector<double> xmax_over_tasks(NTasks, 0);
        xmin_over_tasks[ThisTask] = xmin_domain;
        xmax_over_tasks[ThisTask] = xmax_domain;
#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, xmin_over_tasks.data(), NTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, xmax_over_tasks.data(), NTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

        // Initialize OpenMP
#ifdef USE_OMP
#pragma omp parallel
        {
            int id = omp_get_thread_num();
            if (id == 0)
                NThreads = omp_get_num_threads();
        }
#endif

        // Initialize FFTW
#ifdef USE_FFTW
#ifndef NO_AUTO_FFTW_SETUP
        FML::FFTWSetup & fftwsetup = FML::FFTWSetup::init();
        (void)fftwsetup;
#endif
#endif

        // Show some info
        if (ThisTask == 0) {
            std::cout << "\n#=====================================================\n";
            std::cout << "# Initializing FML\n";
#ifdef USE_MPI
            std::cout << "# MPI enabled. Running with " << NTasks << " MPI tasks\n";
#else
            std::cout << "# MPI not enabled. Running with " << NTasks << " MPI tasks\n";
#endif
#ifdef USE_OMP
            std::cout << "# OpenMP enabled. Main task has " << NThreads << " threads availiable\n";
#else
            std::cout << "# OpenMP not enabled. Main task has " << NThreads << " threads availiable\n";
#endif
#if defined(USE_MPI) && defined(USE_OMP)
            std::cout << "# MPI + Threads is" << (FML::MPIThreadsOK ? " " : " not ") << "working\n";
#endif
#ifdef USE_FFTW
            std::cout << "# Using FFTW. ";
            std::cout << "Thread support is" << (FML::FFTWThreadsOK ? " " : " not ") << "enabled.\n";
#else
            std::cout << "# Not using FFTW\n";
#endif
            for (int i = 0; i < NTasks; i++) {
                std::cout << "# Task " << std::setw(4) << i << " [" << processor_name << "]\n";
                std::cout << "In charge of x-domain [" << std::setw(8) << xmin_over_tasks[i] << " , " << std::setw(8)
                          << xmax_over_tasks[i] << ")\n";
            }
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

    //============================================
    // Abort MPI
    //============================================
    void abort_mpi(int exit_code) {
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, exit_code);
#endif
        exit(exit_code);
    }

    //============================================
    // End MPI
    //============================================
    void finalize_mpi() {
#ifdef USE_MPI
        MPI_Finalize();
#endif
    }

    //============================================
    // Print function for MPI. Only task 0 prints
    //============================================
    void printf_mpi(const char * fmt, ...) {
        if (ThisTask == 0) {
            va_list argp;
            va_start(argp, fmt);
            std::vfprintf(stdout, fmt, argp);
            std::fflush(stdout);
            va_end(argp);
        }
    }

    //============================================
    // An assert function that calls MPI_Abort
    // instead of just abort to avoid deadlock
    //============================================
    void __assert_mpi(const char * expr_str, bool expr, const char * file, int line, const char * msg) {
        if (!expr) {
            std::cout << "[assert_mpi] Assertion failed: [" << expr_str << "], File: [" << file << "], Line: [" << line
                      << "], Message: [" << msg << "]\n";
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            abort();
        }
    }
} // namespace FML

namespace FML {
    namespace GRID {

        void init_fftw([[maybe_unused]] int * argc, [[maybe_unused]] char *** argv) {
            // We are assuming MPI has already been initialized
#ifdef USE_FFTW
#ifdef USE_FFTW_THREADS
            if (FML::MPIThreadsOK) {
                FML::FFTWThreadsOK = INIT_FFTW_THREADS();
                if (FML::FFTWThreadsOK) {
                    SET_FFTW_NTHREADS(FML::NThreads);
                }
            }
#endif
#ifdef USE_MPI
            INIT_FFTW_MPI();
#endif
#endif
        }

        void finalize_fftw() {
#ifdef USE_FFTW
#ifdef USE_MPI
            CLEANUP_FFTW_MPI();
#endif
#endif
        }

        void set_fftw_nthreads([[maybe_unused]] int nthreads) {
#ifdef USE_FFTW
#ifdef USE_FFTW_THREADS
            if (FML::FFTWThreadsOK)
                SET_FFTW_NTHREADS(nthreads);
#endif
#endif
        }
    } // namespace GRID
} // namespace FML
