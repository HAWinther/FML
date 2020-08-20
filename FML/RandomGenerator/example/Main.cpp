#include <iostream>
#include <memory>
#include <FML/RandomGenerator/RandomGenerator.h>

int main(){
  
  //=========================================
  // Simple wrapper around C++ random or GSL
  // random number library (or whatever) to 
  // provide a simple interface for rng in this library
  //=========================================

  FML::RANDOM::RandomGenerator r;
  std::cout << r.generate_uniform() << "\n";

#ifdef USE_GSL
  // If you want to use GSL
  std::shared_ptr<FML::RANDOM::RandomGenerator> r1 = std::make_shared<FML::RANDOM::GSLRandomGenerator> ();
  std::cout << r1->generate_uniform() << "\n";
#endif
}
