#ifndef TIMINGS_HEADER
#define TIMINGS_HEADER

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
namespace FML {
    // The MPI communicator used by the library (defined in Global.cpp)
    extern MPI_Comm communicator;
} // namespace FML
#endif

namespace FML {

    /// This namespace contains useful things that don't fit anywhere else
    namespace UTILS {

        using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

        /// Class for performing timings of the code using std::chrono
        class Timings {
          private:
            // For keeping timings
            std::map<std::string, TimePoint> timings;
            std::map<std::string, double> elapsed_time_sec;
            mutable std::mutex timings_mutex;

            // Get the current time
            TimePoint getTime() { return std::chrono::steady_clock::now(); }

            // Convert an interval to seconds
            double timeInSeconds(const TimePoint & time_start, const TimePoint & time_end) {
                return std::chrono::duration_cast<std::chrono::duration<double>>(time_end - time_start).count();
            }

          public:
            Timings() = default;

            // Copies get the timings recorded so far (and their own lock)
            Timings(const Timings & rhs) {
                std::lock_guard<std::mutex> guard(rhs.timings_mutex);
                timings = rhs.timings;
                elapsed_time_sec = rhs.elapsed_time_sec;
            }
            Timings & operator=(const Timings & rhs) {
                if (this != &rhs) {
                    std::scoped_lock guard(timings_mutex, rhs.timings_mutex);
                    timings = rhs.timings;
                    elapsed_time_sec = rhs.elapsed_time_sec;
                }
                return *this;
            }

            /// Output all the recorded timings
            void PrintAllTimings() const {
#ifdef USE_MPI
                int ThisTask = 0;
                int initialized = 0;
                MPI_Initialized(&initialized);
                if (initialized)
                    MPI_Comm_rank(FML::communicator, &ThisTask);
                if (ThisTask > 0)
                    return;
#endif
                std::lock_guard<std::mutex> guard(timings_mutex);
                std::cout << "\n";
                std::cout << "============================================\n";
                std::cout << "All the recorded timings for task 0:\n";
                std::cout << "============================================\n";
                for (const auto & t : elapsed_time_sec) {
                    std::cout << "Total elapsed time for [" << std::setw(35) << t.first << "]: " << std::setw(10)
                              << t.second << " sec\n";
                }
                std::cout << "============================================\n";
                std::cout << "\n";
            }

            /// Start timing
            /// @param[in] name The label to give to the timing
            ///
            void StartTiming(const char * name) { StartTiming(std::string(name)); }

            /// Start timing
            void StartTiming(std::string name) {
                auto start_time = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> guard(timings_mutex);

                timings[name] = start_time;
                elapsed_time_sec.try_emplace(name, 0.0);
            }

            /// End timing (the time since the last StartTiming with this name is added to its total; a second
            /// EndTiming without a new StartTiming adds nothing and returns 0)
            /// @param[in] name The label to give to the timing
            /// @param[in] print Optional: print timing to screen right away
            ///
            double EndTiming(const char * name, bool print = false) { return EndTiming(std::string(name), print); }

            /// End timing and print to screen if print = true
            /// @param[in] name The label to give to the timing
            /// @param[in] print Optional: print timing to screen right away
            ///
            double EndTiming(std::string name, bool print = false) {
                auto end_time = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> guard(timings_mutex);

                double time_sec = 0.0;
                double tot_time_sec = 0.0;

                // Find the start time (and remove it: the timing is no longer running)
                auto it = timings.find(name);
                if (it == timings.end()) {
                    if (print)
                        std::cout << "Can't print elapsed time. No running timing [" << name << "]\n";
                    return 0.0;
                }
                time_sec = timeInSeconds(it->second, end_time);
                timings.erase(it);

                // Add to the total elapsed time
                auto & total = elapsed_time_sec[name];
                total += time_sec;
                tot_time_sec = total;

                if (print) {
                    if (tot_time_sec == time_sec) {
                        std::cout << "Elapsed time for [" << name << "]: " << time_sec << " sec\n";
                    } else {
                        std::cout << "Current elapsed time for [" << name << "]: " << time_sec
                                  << " sec. Total: " << tot_time_sec << " sec\n";
                    }
                }

                return time_sec;
            }

            /// Total time recorded for a label (0 if it has never been ended)
            double GetTotalTime(const std::string & name) const {
                std::lock_guard<std::mutex> guard(timings_mutex);
                auto it = elapsed_time_sec.find(name);
                return it == elapsed_time_sec.end() ? 0.0 : it->second;
            }

            /// Print to screen total time for a given label
            /// @param[in] name The label to print the time of
            ///
            void PrintTotalTime(const char * name) const { PrintTotalTime(std::string(name)); }

            /// Print to screen total time for a given label
            /// @param[in] name The label to print the time of
            ///
            void PrintTotalTime(std::string name) const {
                std::lock_guard<std::mutex> guard(timings_mutex);
                auto it = elapsed_time_sec.find(name);
                if (it != elapsed_time_sec.end()) {
                    double time_sec = it->second;
                    std::cout << "Total elapsed time for [" << name << "]: " << time_sec << " sec\n";
                }
            }
        };
    } // namespace UTILS
} // namespace FML
#endif
