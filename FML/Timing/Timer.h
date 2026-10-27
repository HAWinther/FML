#ifndef FML_TIMING_TIMER_H
#define FML_TIMING_TIMER_H
#include <chrono>
#include <iostream>
#include <string>
#include <utility>

namespace FML { namespace UTILS {
    /// A restartable wall-clock stopwatch. Use Timings for accumulated timings by label.
    class Timer {
      public:
        explicit Timer(std::string name = "") : name(std::move(name)) { start(); }
        void start() { t0 = std::chrono::steady_clock::now(); }
        double elapsed() const {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        void print(bool enabled = true) const {
            if (enabled)
                std::cout << "[Timer" << (name.empty() ? "" : " " + name) << "] " << elapsed() << " s\n";
        }
      private:
        std::string name;
        std::chrono::steady_clock::time_point t0;
    };
} }
#endif
