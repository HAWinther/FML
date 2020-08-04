#include <stdio.h>
#include <iostream>
#include <FML/Global/Global.h>

template<class T>
using Vector = FML::Vector<T>;

int main(int argc, char **argv){

  // The stuff we store in global
  std::cout << FML::ThisTask << " " << FML::NTasks << "\n";
  std::cout << FML::xmin_domain << " " << FML::xmax_domain << "\n";

  // Memory loggin with custom allocator
  Vector<double> a(1000000);
  Vector<double> b(1000000);
#ifdef MEMORY_LOGGING
  auto * mem = FML::MemoryLog::get();
  mem->add_label(a.data(), "[Vector a]");
  mem->add_label(b.data(), "[Vector b]");
#endif    
  std::cout << FML::ThisTask << "\n";

#ifdef MEMORY_LOGGING
  a = a + b + FML::pow(a,2);
  //FML::MemoryLog::get()->print();
#endif
}

