#ifndef MEMORYLOG_HEADER
#define MEMORYLOG_HEADER

#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>
#include <mutex>
#include <vector>
#include <typeinfo>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace FML {
  extern int NTasks;

  //=======================================================
  // Do not log allocations smaller than a minimum size
  // The standard size is 1MB
  // Set a limit to how many items we can have in the log
  // at the same time to avoid it taking up too much memory
  // If this limit is reached the logging will stop!
  //=======================================================
#ifndef MIN_BYTES_TO_LOG
#define MIN_BYTES_TO_LOG 1000000
#endif
#ifndef MAX_ALLOCATIONS_IN_MEMORY
#define MAX_ALLOCATIONS_IN_MEMORY 1000000
#endif

  class MemoryLog;
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

  //=======================================================
  // Logs all heap allocations in the code that are larger 
  // than min_bytes_to_log. Implemented as a singleton so only 
  // one instance of this object exists. Thread safe
  //=======================================================
  class MemoryLog {
    private:
      static MemoryLog *instance;

      // Lock
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
      size_t nallocation = 0;
      bool stop_logging = false;

      // Private constructor and destructor
      MemoryLog(){
        time_start = std::chrono::steady_clock::now();
        memory_in_use = 0;
        memory_vs_time[time_start] = 0;
        nallocation = 0;
        stop_logging = false;
      }
      ~MemoryLog() = default;

    public:

      static MemoryLog * get(){
        if(!instance)
          instance = new MemoryLog;
        return instance;
      }

      MemoryLog(const MemoryLog& arg) = delete; 
      MemoryLog(const MemoryLog&& arg) = delete;
      MemoryLog& operator=(const MemoryLog& arg) = delete; 
      MemoryLog& operator=(const MemoryLog&& arg) = delete;

      // Add a new allocation label
      void add_label(void *ptr, std::string name){
        if(allocations.find(ptr) == allocations.end()) {
          std::cout << "[MemoryLogging] Warning could not find pointer in allocation list to add label " + name + " to\n";
          return;
        }
        std::lock_guard<std::mutex> guard(mymutex);
        labels[ptr] = name;
      }
      void add_label(void *ptr, size_t size, std::string name){
        if(size < min_bytes_to_log) return;
        add_label(ptr, name);
      }

      void add(void *ptr, size_t size){
        if(size < min_bytes_to_log) return;
        if(stop_logging) return;
        std::lock_guard<std::mutex> guard(mymutex);
        allocations[ptr] = size;

        memory_in_use += size;
        if(memory_in_use > peak_memory_use)
          peak_memory_use = memory_in_use;

        auto time = std::chrono::steady_clock::now();
        memory_vs_time[time] = memory_in_use;
        nallocation++;
        stop_logging = nallocation >= max_allocations_to_log;
      }

      // Remove an allocation and log info
      void remove(void *ptr, size_t size){
        if(size < min_bytes_to_log) return;
        if(stop_logging) return;
        std::lock_guard<std::mutex> guard(mymutex);
        allocations.erase(ptr);

        memory_in_use -= size;

        auto time = std::chrono::steady_clock::now();
        memory_vs_time[time] = memory_in_use;
      }

      // Print the total memory in use
      void print(){
        int ThisTask = 0;
          
        if(stop_logging) {
          std::cout << "Warning: This method has stopped working on task " << ThisTask << " due to max number of allocation having been reached\n";
        }
#ifdef USE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD,&ThisTask);
#endif
        if(ThisTask == 0){
          std::cout << "\n#=====================================================\n\n";
          std::cout << "We are only tracking allocation of standard container\n";
          std::cout << "and only allocations larger than " << min_bytes_to_log << " bytes\n\n";
        }
#ifdef USE_MPI
        long long int min_memory  = memory_in_use;
        long long int max_memory  = memory_in_use;
        long long int mean_memory = memory_in_use;
        long long int peak_memory = peak_memory_use;
        MPI_Allreduce(MPI_IN_PLACE, &min_memory,  1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_memory,  1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &mean_memory, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &peak_memory, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        mean_memory /= NTasks;
        if(ThisTask == 0){
          std::cout << "Memory in use:   " << std::setw(15) << memory_in_use/1.0e6   << " MB\n";
          std::cout << "Min over tasks:  " << std::setw(15) << min_memory/1.0e6      << " MB\n";
          std::cout << "Mean over tasks: " << std::setw(15) << mean_memory/1.0e6     << " MB\n";
          std::cout << "Max over tasks:  " << std::setw(15) << max_memory/1.0e6      << " MB\n";
          std::cout << "\n";
          std::cout << "Peak memory use: " << std::setw(15) << peak_memory_use/1.0e6 << " MB\n";
          std::cout << "Max over tasks:  " << std::setw(15) << peak_memory/1.0e6     << " MB\n";
          std::cout << "\n";
        }
#else
        std::cout << "Memory in use:   " << std::setw(15) << memory_in_use/1.0e6   << " MB\n";
        std::cout << "Peak memory use: " << std::setw(15) << peak_memory_use/1.0e6 << " MB\n";
#endif
        if(!allocations.empty()){
          if(ThisTask == 0) {
            std::cout << "\nWe have the following things allocated on task 0: \n";
          }
          for(auto &&a : allocations){
            if(ThisTask == 0){
              std::string name = "";
              if(labels.find(a.first) != labels.end())
                name = labels[a.first];
              std::string bytelabel =  " (MB)";
              double factor = 1e6;
              std::cout << "Address: " << a.first << " Size: " << a.second/factor << bytelabel << " Label: " << name << "\n";
            }
          }
          if(ThisTask == 0)
            std::cout << "\n";
        }
        if(!memory_vs_time.empty()){
          if(ThisTask == 0)
            std::cout << "Total memory as function of time:\n";
          for(auto &&m : memory_vs_time){
            double time_in_sec = std::chrono::duration_cast<std::chrono::duration<double>>(m.first - time_start).count();
            if(ThisTask == 0){
              std::cout << " Time (sec): " << std::setw(13) <<  time_in_sec;
              std::string bytelabel = " (MB)";
              double factor = 1e6;
              std::cout << " Memory: " << std::setw(13) << m.second/factor << bytelabel << "\n";
            }
          }

        }
        if(ThisTask == 0){
          std::cout << "\n#=====================================================\n";
          std::cout << std::flush;
        }
      }
  };

  // Custom allocator with logging
  template <typename T>
    struct LogAllocator {
      using value_type = T;

      LogAllocator() = default;
      template <class U>
        LogAllocator(const LogAllocator<U>&) {}

      T* allocate(std::size_t size) {
        if (size <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
          if (auto ptr = std::malloc(size * sizeof(T))) {
            MemoryLog::get()->add(ptr,size * sizeof(T));
            return static_cast<T*>(ptr);
          }
        }
        std::cout << "[LogAllocator] Allocation of " << size << " elements of type " << typeid(T).name() << " of size " << sizeof(T) << " failed" << std::endl;
        throw std::bad_alloc();
      }
      void deallocate(T* ptr, std::size_t size) {
        MemoryLog::get()->remove(ptr, size * sizeof(T));
        std::free(ptr);
      }
    };

  template <typename T, typename U>
    inline bool operator == (const LogAllocator<T>&, const LogAllocator<U>&) {
      return true;
    }

  template <typename T, typename U>
    inline bool operator != (const LogAllocator<T>& a, const LogAllocator<U>& b) {
      return !(a == b);
    }
}

#endif
