#ifndef MEMORYLOG_HEADER
#define MEMORYLOG_HEADER

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <typeinfo>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
namespace FML {
    // The MPI communicator used by the library (defined in Global.cpp)
    extern MPI_Comm communicator;
} // namespace FML
#endif

namespace FML {

    extern int ThisTask;
    extern int NTasks;
    extern std::pair<double, double> get_system_memory_use();

    //=======================================================
    // Do not log allocations smaller than a minimum size
    // The standard size is 0
    // Set a limit to how many allocations we can have in the log
    // at the same time to avoid it taking up too much memory
    // If this limit is reached then we clear the memory associated
    // with the logger and stop logging any more allocations.
    // The memory-vs-time history keeps at most this many entries
    // (later changes are not added to the history, but the
    // allocations and the peak are still tracked)
    //=======================================================
#ifndef MIN_BYTES_TO_LOG
#define MIN_BYTES_TO_LOG 0
#endif
#ifndef MAX_ALLOCATIONS_IN_MEMORY
#define MAX_ALLOCATIONS_IN_MEMORY 100000
#endif

    class MemoryLog;
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

    //=======================================================
    /// Logs all heap allocations in the code that are larger
    /// than min_bytes_to_log. Implemented as a singleton so only
    /// one instance of this object exists. Thread safe.
    //=======================================================
    class MemoryLog {
      private:
        // Lock (all members below are only accessed with it held)
        std::mutex mymutex{};

        // The first allocation
        TimePoint time_start{};

        // Containing all current allocations
        std::map<void *, size_t> allocations{};

        // For more info, after allocating we can (externally)
        // call add with the pointer and store a label
        std::map<void *, std::string> labels{};

        // The total memory as function of time
        std::map<TimePoint, size_t> memory_vs_time{};

        // Total memory in use at any given time
        size_t memory_in_use{0};
        size_t peak_memory_use{0};

        // The minimum allocation size in bytes to log
        size_t min_bytes_to_log = MIN_BYTES_TO_LOG;
        size_t max_allocations_to_log = MAX_ALLOCATIONS_IN_MEMORY;
        size_t nallocation{0};
        bool stop_logging{false};
        bool history_full{false};

        // Record the memory in use now (called with the lock held)
        void record_time() {
            if (history_full)
                return;
            memory_vs_time[std::chrono::steady_clock::now()] = memory_in_use;
            history_full = memory_vs_time.size() >= max_allocations_to_log;
        }

        // Private constructor and destructor
        MemoryLog() {
            time_start = std::chrono::steady_clock::now();
            memory_in_use = 0;
            memory_vs_time[time_start] = 0;
            nallocation = 0;
            stop_logging = false;
        }
        ~MemoryLog() = default;

      public:
        /// The (only) instance. Created on first use (thread safe) and never deleted, so it can be used until the
        /// end of the program (also from destructors of static objects)
        static MemoryLog * get() {
            static MemoryLog * instance = new MemoryLog;
            return instance;
        }

        MemoryLog(const MemoryLog & arg) = delete;
        MemoryLog(const MemoryLog && arg) = delete;
        MemoryLog & operator=(const MemoryLog & arg) = delete;
        MemoryLog & operator=(const MemoryLog && arg) = delete;

        /// Bytes currently allocated (of what we log)
        size_t get_memory_in_use() {
            std::lock_guard<std::mutex> guard(mymutex);
            return memory_in_use;
        }
        /// Peak of get_memory_in_use so far
        size_t get_peak_memory_use() {
            std::lock_guard<std::mutex> guard(mymutex);
            return peak_memory_use;
        }
        /// Number of allocations currently logged
        size_t get_number_of_allocations() {
            std::lock_guard<std::mutex> guard(mymutex);
            return allocations.size();
        }
        /// The label of an allocation ("" if it has none)
        std::string get_label(void * ptr) {
            std::lock_guard<std::mutex> guard(mymutex);
            auto it = labels.find(ptr);
            return it == labels.end() ? std::string("") : it->second;
        }
        /// Is the logger still logging (false after too many allocations at the same time)
        bool is_logging() {
            std::lock_guard<std::mutex> guard(mymutex);
            return not stop_logging;
        }

        // Add a new allocation label
        void add_label(void * ptr, std::string name) {
            std::lock_guard<std::mutex> guard(mymutex);
            if (stop_logging)
                return;
            if (allocations.find(ptr) == allocations.end()) {
                std::cout << "[MemoryLogging] Warning could not find pointer in allocation list to add label " + name +
                                 " to\n";
                return;
            }
            labels[ptr] = name;
        }
        void add_label(void * ptr, size_t size, std::string name) {
            if (size < min_bytes_to_log)
                return;
            add_label(ptr, name);
        }

        void add(void * ptr, size_t size) {
            if (size < min_bytes_to_log)
                return;
            std::lock_guard<std::mutex> guard(mymutex);
            if (stop_logging)
                return;
            allocations[ptr] = size;
            nallocation++;

            memory_in_use += size;
            if (memory_in_use > peak_memory_use)
                peak_memory_use = memory_in_use;
            record_time();

            // If we have too many allocations at the same time then stop logging
            if (allocations.size() >= max_allocations_to_log) {
                stop_logging = true;
                clear();
            }
        }

        // Free the memory of the log (called with the lock held)
        void clear() {
            allocations.clear();
            labels.clear();
            memory_vs_time.clear();
        }

        // Remove an allocation and log info
        void remove(void * ptr, size_t size) {
            if (size < min_bytes_to_log)
                return;
            std::lock_guard<std::mutex> guard(mymutex);
            if (stop_logging)
                return;
            auto it = allocations.find(ptr);
            if (it == allocations.end())
                return; // Allocated before we started logging (or below the limit)

#ifdef DEBUG_MEMORYLOG
            if (ThisTask == 0)
                std::cout << "===> MemoryLog::remove Freeing[" << labels[ptr] << "]" << std::endl;
#endif
            memory_in_use -= it->second;
            allocations.erase(it);
            labels.erase(ptr);
            record_time();
        }

        /// Print the memory in use (collective with MPI: all tasks must call it)
        void print() {
            std::lock_guard<std::mutex> guard(mymutex);

            // Check if any task has saturated the allocation limit
            char ok = stop_logging ? 0 : 1;
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_CHAR, MPI_MIN, FML::communicator);
#endif

            // Fetch system memory usage
            auto sysmem = get_system_memory_use();
            double min_sys_vmemory = sysmem.first;
            double max_sys_vmemory = sysmem.first;
            double mean_sys_vmemory = sysmem.first;
            double min_sys_rssmemory = sysmem.second;
            double max_sys_rssmemory = sysmem.second;
            double mean_sys_rssmemory = sysmem.second;

#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &min_sys_vmemory, 1, MPI_DOUBLE, MPI_MIN, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &max_sys_vmemory, 1, MPI_DOUBLE, MPI_MAX, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &mean_sys_vmemory, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &min_sys_rssmemory, 1, MPI_DOUBLE, MPI_MIN, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &max_sys_rssmemory, 1, MPI_DOUBLE, MPI_MAX, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &mean_sys_rssmemory, 1, MPI_DOUBLE, MPI_SUM, FML::communicator);
#endif
            mean_sys_vmemory /= NTasks;
            mean_sys_rssmemory /= NTasks;
            if (ThisTask == 0) {
                std::cout << "\n#=====================================================\n";
                std::cout << "#                   MemoryLogging\n";
                std::cout << "#=====================================================\n\n";
                std::cout << "Info about resident set size:\n";
                std::cout << "[Current resident set size]:\n";
                std::cout << "Min over tasks:  " << std::setw(15) << min_sys_vmemory / 1.0e6 << " MB\n";
                std::cout << "Mean over tasks: " << std::setw(15) << mean_sys_vmemory / 1.0e6 << " MB\n";
                std::cout << "Max over tasks:  " << std::setw(15) << max_sys_vmemory / 1.0e6 << " MB\n";
                std::cout << "[Peak resident set size]:\n";
                std::cout << "Min over tasks:  " << std::setw(15) << min_sys_rssmemory / 1.0e6 << " MB\n";
                std::cout << "Mean over tasks: " << std::setw(15) << mean_sys_rssmemory / 1.0e6 << " MB\n";
                std::cout << "Max over tasks:  " << std::setw(15) << max_sys_rssmemory / 1.0e6 << " MB\n";
            }

            if (ok == 0) {
                clear();
                stop_logging = true;
                if (ThisTask == 0)
                    std::cout << "\nWarning: The memory log of allocations has stopped working due to max number of "
                                 "allocations ("
                              << max_allocations_to_log << ") having been reached\n"
                              << "\n#=====================================================\n"
                              << std::flush;
                return;
            }

            if (ThisTask == 0) {
                std::cout << "\n";
                std::cout << "For info below we are only tracking allocation of standard\n";
                std::cout << "container (Vector) and only allocations larger than " << min_bytes_to_log << " bytes\n";
                std::cout << "(not FFTW grids, which use fftw_malloc)\n\n";
            }

            long long int min_memory = memory_in_use;
            long long int max_memory = memory_in_use;
            long long int mean_memory = memory_in_use;
            long long int peak_memory = peak_memory_use;
#ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &min_memory, 1, MPI_LONG_LONG, MPI_MIN, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &max_memory, 1, MPI_LONG_LONG, MPI_MAX, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &mean_memory, 1, MPI_LONG_LONG, MPI_SUM, FML::communicator);
            MPI_Allreduce(MPI_IN_PLACE, &peak_memory, 1, MPI_LONG_LONG, MPI_MAX, FML::communicator);
#endif
            mean_memory /= NTasks;
            if (ThisTask == 0) {
                std::cout << "Memory in use (Task 0):   " << std::setw(15) << double(memory_in_use) / 1.0e6 << " MB\n";
                std::cout << "Min over tasks:           " << std::setw(15) << double(min_memory) / 1.0e6 << " MB\n";
                std::cout << "Mean over tasks:          " << std::setw(15) << double(mean_memory) / 1.0e6 << " MB\n";
                std::cout << "Max over tasks:           " << std::setw(15) << double(max_memory) / 1.0e6 << " MB\n";
                std::cout << "\n";
                std::cout << "Peak memory use (Task 0): " << std::setw(15) << double(peak_memory_use) / 1.0e6
                          << " MB\n";
                std::cout << "Max over tasks:           " << std::setw(15) << double(peak_memory) / 1.0e6 << " MB\n";
                std::cout << "\n";
            }
            if (not allocations.empty()) {
                if (ThisTask == 0) {
                    std::cout << "\nWe have the following things allocated on task 0: \n";
                }
                for (auto && a : allocations) {
                    if (ThisTask == 0) {
                        std::string name = "";
                        if (labels.find(a.first) != labels.end())
                            name = labels[a.first];
                        std::string bytelabel = " (MB)";
                        double factor = 1e6;
                        std::cout << "Address: " << a.first << " Size: " << double(a.second) / factor << bytelabel
                                  << " Label: " << name << "\n";
                    }
                }
                if (ThisTask == 0)
                    std::cout << "\n";
            }
            if (not memory_vs_time.empty()) {
                if (ThisTask == 0)
                    std::cout << "Total memory as function of time"
                              << (history_full ?
                                      " (only the first " + std::to_string(max_allocations_to_log) + " changes)" :
                                      std::string(""))
                              << ":\n";
                for (auto && m : memory_vs_time) {
                    double time_in_sec =
                        std::chrono::duration_cast<std::chrono::duration<double>>(m.first - time_start).count();
                    if (ThisTask == 0) {
                        std::cout << " Time (sec): " << std::setw(13) << time_in_sec;
                        std::string bytelabel = " (MB)";
                        double factor = 1e6;
                        std::cout << " Memory: " << std::setw(13) << double(m.second) / factor << bytelabel << "\n";
                    }
                }
            }
            if (ThisTask == 0) {
                std::cout << "\n#=====================================================\n";
                std::cout << std::flush;
            }
        }
    };

    /// Custom allocator that logs allocations
    template <typename T>
    struct LogAllocator {
        using value_type = T;

        LogAllocator() = default;
        template <class U>
        LogAllocator(const LogAllocator<U> &) {}

        T * allocate(std::size_t size) {
            if (size <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                if (auto ptr = std::malloc(size * sizeof(T))) {
                    MemoryLog::get()->add(ptr, size * sizeof(T));
                    return static_cast<T *>(ptr);
                }
            }
            std::cout << "[LogAllocator] Allocation of " << size << " elements of type " << typeid(T).name()
                      << " of size " << sizeof(T) << " failed" << std::endl;
            throw std::bad_alloc();
        }
        void deallocate(T * ptr, std::size_t size) {
            MemoryLog::get()->remove(ptr, size * sizeof(T));
            std::free(ptr);
        }
    };

    template <typename T, typename U>
    inline bool operator==(const LogAllocator<T> &, const LogAllocator<U> &) {
        return true;
    }

    template <typename T, typename U>
    inline bool operator!=(const LogAllocator<T> & a, const LogAllocator<U> & b) {
        return not(a == b);
    }
} // namespace FML

#endif
