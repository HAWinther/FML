#include <FML/RandomGenerator/RandomGenerator.h>
#include <iostream>
#include <memory>

int main() {

    //=========================================
    // Simple wrapper around C++ random or GSL
    // random number library (or whatever) to
    // provide a simple interface for rng in this library
    //=========================================

    const int seed = 1234;
    FML::RANDOM::RandomGenerator r(seed);

    // Make a clone of the rng
    auto rclone = r.clone();

    // Generate some numbers
    std::vector<double> numbers(5);
    for(auto & num : numbers){
      num = r.generate_uniform();
      std::cout << num << "\n";
    }
    std::cout << "\n";
    
    // The clone should give the same results as above
    for(auto & num : numbers){
      double num2 = rclone->generate_uniform();
      std::cout << num2 << "\n";
      assert( std::fabs( num - num2 ) < 1e-10 );
    }
    std::cout << "\n";
    
#ifdef USE_GSL
    // If you want to use GSL
    std::shared_ptr<FML::RANDOM::RandomGenerator> r1 = std::make_shared<FML::RANDOM::GSLRandomGenerator>(seed);
    
    // Make a clone of the rng
    auto r1clone = r1->clone();

    // Generate some numbers
    for(auto & num : numbers){
      num = r1->generate_uniform();
      std::cout << num << "\n";
    }
    std::cout << "\n";
    
    // The clone should give the same results as above
    for(auto & num : numbers){
      double num2 = r1clone->generate_uniform();
      std::cout << num2 << "\n";
      assert( std::fabs( num - num2 ) < 1e-10 );
    }
    std::cout << "\n";
    
#endif
}
