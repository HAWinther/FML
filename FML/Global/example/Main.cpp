#include <FML/Global/Global.h>
#include <iostream>
#include <stdio.h>

template <class T>
using Vector = FML::Vector<T>;

int main() {

    // If we don't want automatic initialization/finalization of FML/MPI/FFTW (NO_AUTO_FML_SETUP) then we can do it like this
    // (or the way you are used to; just remember to set MPIThreadsOK and FFTWThreadsOK and call init_fml last)
    // FML::init_mpi();
    // FML::GRID::init_fftw();
    // FML::init_fml();
    // FML::info();

    // The stuff we store in global
    std::cout << "Task " << FML::ThisTask << " Total Tasks: " << FML::NTasks << " xmin: " << FML::xmin_domain
              << " xmax: " << FML::xmax_domain << std::endl;
    
#ifdef MEMORY_LOGGING
    // Print the FML memory log
    FML::MemoryLog::get()->print();
   
    // Allocate some memory with standard container
    Vector<double> a(1000000);
    Vector<double> b(1000000);

    // Set memory labels
    auto * mem = FML::MemoryLog::get();
    mem->add_label(a.data(), "[Vector a]");
    mem->add_label(b.data(), "[Vector b]");

    // Print the FML memory log
    FML::MemoryLog::get()->print();

    // Clear the memory and print again
    a.clear();
    b.clear();
    a.shrink_to_fit();
    b.shrink_to_fit();

    // Print the FML memory log 
    FML::MemoryLog::get()->print();
#endif

    // If we don't want automatic initialization/finalization of FML/MPI/FFTW (NO_AUTO_FML_SETUP) then we can do it like this
    // FML::GRID::finalize_fftw();
    // FML::finalize_mpi();
}
