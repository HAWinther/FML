#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdarg.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

#include "Global.h"
#include "SystemMemory.h"

static std::string processor_name{"NameIsOnlyKnownWithMPI"};

namespace FML {
    
    FML::UTILS::Timings global_timer;

    // Initialize FML, set the few globals we have and init MPI
    // and FFTW. If no auto setup you must init MPI and FFTW
    // yourself and then call init_fml
#ifndef NO_AUTO_FML_SETUP
    FMLSetup & fmlsetup = FMLSetup::init();
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
    /// Initialize MPI. This function is automatically
    /// called unless NO_AUTO_MPI_SETUP is defined
    //============================================
    void init_mpi([[maybe_unused]] int * argc, [[maybe_unused]] char *** argv) {
#ifdef USE_MPI
#ifdef USE_OMP
        int provided;
        MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
        MPIThreadsOK = provided >= MPI_THREAD_FUNNELED;
#else
        MPI_Init(argc, argv);
#endif
#endif
    }

    //============================================
    /// Initialize the global variables within FML
    /// We here assume MPI has been initialized
    //============================================
    void init_fml() {
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
        MPI_Comm_size(MPI_COMM_WORLD, &NTasks);
        int plen;
        char pname[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(pname, &plen);
        processor_name = std::string(pname);
#else
        ThisTask = 0;
        NTasks = 1;
        processor_name = "ComputerNameNotAvailiable";
#endif

        // Initialize OpenMP
#ifdef USE_OMP
#pragma omp parallel
        {
            int id = omp_get_thread_num();
            if (id == 0)
                NThreads = omp_get_num_threads();
        }
#else
        NThreads = 1;
#endif

        // Set range for local domain
        xmin_domain = ThisTask / double(NTasks);
        xmax_domain = (ThisTask + 1) / double(NTasks);
    }

    /// Show some info about FML
    void info() {
        if (ThisTask == 0) {

            std::cout << "\n";
            std::cout << "#=====================================================\n";
            std::cout << "#           ________________  .____         \n";
            std::cout << "#           \\_   _____/     \\ |    |      \n";
            std::cout << "#            |    __)/  \\ /  \\|    |      \n";
            std::cout << "#            |     \\/    Y    \\    |___   \n";
            std::cout << "#            \\___  /\\____|__  /_______ \\ \n";
            std::cout << "#                \\/         \\/        \\/ \n";
            std::cout << "#\n";

            std::cout << "# Initializing FML, MPI and FFTW\n";
#ifdef USE_MPI
            std::cout << "# MPI is enabled. Running with " << NTasks << " MPI tasks\n";
#else
            std::cout << "# MPI is *not* enabled\n";
#endif
#ifdef USE_OMP
            std::cout << "# OpenMP is enabled. Main task has " << NThreads << " threads availiable\n";
#else
            std::cout << "# OpenMP is *not* enabled\n";
#endif
#if defined(USE_MPI) && defined(USE_OMP)
            std::cout << "# MPI + Threads is" << (FML::MPIThreadsOK ? " " : " *not* ") << "working\n";
#endif
#ifdef USE_FFTW
            std::cout << "# FFTW is enabled. ";
            std::cout << "Thread support is" << (FML::FFTWThreadsOK ? " " : " *not* ") << "enabled\n";
#else
            std::cout << "# FFTW is *not* enabled\n";
#endif
            std::cout << "#\n";
            std::cout << "# List of tasks:\n";
        }

        for (int i = 0; i < NTasks; i++) {
            if (FML::ThisTask == i) {
                std::cout << "# Task " << std::setw(4) << i << " [" << processor_name << "]\n";
                std::cout << "#     x-domain [" << std::setw(8) << FML::xmin_domain << " , " << std::setw(8)
                          << FML::xmax_domain << ")" << std::endl;
            }
#ifdef USE_MPI
            // Sleep to syncronize output to screen (this is not guaranteed to ensure syncronized output, but the only
            // alternative is communiation and its not that important)
            MPI_Barrier(MPI_COMM_WORLD);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
        }

        if (FML::ThisTask == 0) {
            std::cout << std::flush;
            std::cout << "#\n";
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }

#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
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
#ifdef USE_FFTW
        // If we have MPI and FFTW then FFTW takes care of finalizing
        return;
#endif
#ifdef USE_MPI
        MPI_Finalize();
#endif
    }

    //============================================
    // Print function for MPI. Only task 0 prints
    //============================================
    void printf_mpi(const char * fmt, ...) {
        if (FML::ThisTask == 0) {
            va_list argp;
            va_start(argp, fmt);
            std::vfprintf(stdout, fmt, argp);
            std::fflush(stdout);
            va_end(argp);
        }
    }

    //============================================
    // Create a folder (only task 0 is allowed to
    // do this)
    //============================================
    bool create_folder(std::string foldername) {
        bool ok = true;
        if (FML::ThisTask == 0) {
            int status = mkdir(foldername.c_str(), 0777);
            ok = not((status < 0) && (errno != EEXIST));
            if (not ok) {
                throw std::runtime_error("Failed to create snapshot folder [" + foldername + "]");
            }
        }
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        return ok;
    }

    //============================================
    /// Fetch the resident set size currently and
    /// the peak value so far of it. Returns bytes
    //============================================
    std::pair<double, double> get_system_memory_use() {
        double currentSize_bytes = getCurrentRSS();
        double peakSize_bytes = getPeakRSS();
        return {currentSize_bytes, peakSize_bytes};
    }
    
    //============================================
    /// Print the resident set size and the peak
    /// value of it so far
    //============================================
    void print_system_memory_use() {
        auto sysmem = FML::get_system_memory_use();
        double cur_rss = sysmem.first / 1e6;
        double peak_rss = sysmem.second / 1e6;
        double min_cur_rss = cur_rss;
        double max_cur_rss = cur_rss;
        double min_peak = peak_rss;
        double max_peak = peak_rss;
        FML::MinOverTasks(&min_cur_rss);
        FML::MaxOverTasks(&max_cur_rss);
        FML::MinOverTasks(&min_peak);
        FML::MaxOverTasks(&max_peak);
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "# System memory use (resident set size): \n";
            std::cout << "# Current RSS: " << std::setw(15) << min_cur_rss << " MB (min over tasks)\n";
            std::cout << "# Current RSS: " << std::setw(15) << max_cur_rss << " MB (max over tasks)\n";
            std::cout << "# Peak RSS:    " << std::setw(15) << min_peak << " MB (min over tasks)\n";
            std::cout << "# Peak RSS:    " << std::setw(15) << max_peak << " MB (max over tasks)\n";
            std::cout << "#=====================================================\n";
        }
    }

    // Make sure the most common types we use for comm gets instansiated
#define TYPES(TYPE)                                                                                                    \
    template std::vector<TYPE> GatherFromTasks<TYPE>(TYPE *);                                                          \
    template void MinOverTasks<TYPE>(TYPE *);                                                                          \
    template void MaxOverTasks<TYPE>(TYPE *);                                                                          \
    template void SumOverTasks<TYPE>(TYPE *);                                                                          \
    template void SumArrayOverTasks<TYPE>(TYPE *, int);
    TYPES(char);
    TYPES(int);
    TYPES(unsigned int);
    TYPES(long long);
    TYPES(size_t);
    TYPES(ptrdiff_t);
    TYPES(float);
    TYPES(double);
    TYPES(long double);
#undef TYPES

} // namespace FML

namespace FML {
    namespace GRID {

        void init_fftw([[maybe_unused]] int * argc, [[maybe_unused]] char *** argv) {
#if defined(USE_FFTW) && defined(USE_FFTW_THREADS)
            if (FML::MPIThreadsOK) {
                FML::FFTWThreadsOK = INIT_FFTW_THREADS();
                if (FML::FFTWThreadsOK) {
                    SET_FFTW_NTHREADS(FML::NThreads);
                }
            }
#endif
#if defined(USE_FFTW) && defined(USE_MPI)
            INIT_FFTW_MPI();
#endif
        }

        void finalize_fftw() {
#if defined(USE_FFTW) && defined(USE_MPI)
            CLEANUP_FFTW_MPI();
            MPI_Finalize();
#endif
        }

        void set_fftw_nthreads([[maybe_unused]] int nthreads) {
#if defined(USE_FFTW) && defined(USE_FFTW_THREADS)
            if (FML::FFTWThreadsOK)
                SET_FFTW_NTHREADS(nthreads);
#endif
        }
    } // namespace GRID
} // namespace FML
