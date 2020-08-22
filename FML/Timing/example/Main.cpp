#include <FML/Timing/Timings.h>
#include <cmath>

using Timer = FML::UTILS::Timings;

int main() {
    Timer timer;
    timer.StartTiming("Whole Program");

    //==============================================
    // Example of how to do simple benchmarking
    // The 'timer' here is a global
    //==============================================

    // Add a timing
    timer.StartTiming("Label");
    for (int i = 0; i < 10000; i++)
        exp(i);
    timer.EndTiming("Label");

    // Add a new timing and print the time right away when we end
    timer.StartTiming("Label 1");
    for (int i = 0; i < 10000; i++)
        exp(i);
    timer.EndTiming("Label 1", true);

    // Add to the same label as above
    timer.StartTiming("Label 1");
    for (int i = 0; i < 10000; i++)
        exp(i);
    timer.EndTiming("Label 1");

    // Print all the timings we have in the code
    timer.EndTiming("Whole Program");
    timer.PrintAllTimings();
}
