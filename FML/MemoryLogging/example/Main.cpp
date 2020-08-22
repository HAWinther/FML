#include <FML/Global/Global.h>
#include <FML/MemoryLogging/MemoryLogging.h>
#include <vector>

// Vector using the custom log-allocator
using MyVector = std::vector<double, FML::LogAllocator<double>>;

int main() {

    auto * mem = FML::MemoryLog::get();

    //================================================
    // Example of memory logging
    //================================================

    // Allocte some memory
    MyVector a(100000);
    MyVector b(1000000);
    MyVector c;

    if (FML::ThisTask > 0) {
        c = MyVector(100000);
        mem->add_label(c.data(), "[This is C]");
    }

    // Add a label so its easier to know what we have allocated
    // when we print below (not required)
    mem->add_label(a.data(), "[This is A]");
    mem->add_label(b.data(), "[This is B]");

    mem->print();

    // Free the memory
    // a.clear(); a.shrink_to_fit();
    // b.clear(); b.shrink_to_fit();
    // c.clear(); c.shrink_to_fit();

    // Print the info about the memory and the history
    // of allocations
    FML::MemoryLog::get()->print();
}
