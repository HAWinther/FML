#include "RandomGenerator.h"

namespace FML {
  namespace RANDOM {

    RandomGenerator::RandomGenerator(){
      name = standardname;
      uniform_range_lower = fiducial_uniform_range_lower;
      uniform_range_upper = fiducial_uniform_range_upper;
      uniform_dist = std::uniform_real_distribution<double>(uniform_range_lower,uniform_range_upper);
      normal_mu = fiducial_normal_mu;
      normal_sigma = fiducial_normal_sigma;;
      normal_dist = std::normal_distribution<double> (normal_mu,normal_sigma);
    }

    RandomGenerator::RandomGenerator(unsigned int Seed){
      name = standardname;
      set_seed(Seed);
      uniform_range_lower = fiducial_uniform_range_lower;
      uniform_range_upper = fiducial_uniform_range_upper;
      uniform_dist = std::uniform_real_distribution<double>(uniform_range_lower,uniform_range_upper);
      normal_mu = fiducial_normal_mu;
      normal_sigma = fiducial_normal_sigma;;
      normal_dist = std::normal_distribution<double> (normal_mu,normal_sigma);
    }

    RandomGenerator::RandomGenerator(std::vector<unsigned int> Seed){
      name = standardname;
      set_seed(Seed);
      uniform_range_lower = fiducial_uniform_range_lower;
      uniform_range_upper = fiducial_uniform_range_upper;
      uniform_dist = std::uniform_real_distribution<double>(uniform_range_lower,uniform_range_upper);
      normal_mu = fiducial_normal_mu;
      normal_sigma = fiducial_normal_sigma;;
      normal_dist = std::normal_distribution<double> (normal_mu,normal_sigma);
    }

    RandomGenerator::~RandomGenerator(){}

    RandomGenerator::RandomGenerator(const RandomGenerator &rhs){
      name = rhs.name;
      Seed = rhs.Seed;
      generator = rhs.generator;
      uniform_range_lower = rhs.uniform_range_lower;
      uniform_range_upper = rhs.uniform_range_upper;
      uniform_dist = rhs.uniform_dist;
      normal_mu = rhs.normal_mu;
      normal_sigma = rhs.normal_sigma;
      normal_dist = rhs.normal_dist;
    }

    void RandomGenerator::set_uniform_range(double a, double b){
      uniform_range_lower = a;
      uniform_range_upper = b;
      uniform_dist = std::uniform_real_distribution<double>(a,b);
    }

    void RandomGenerator::set_normal_range(double mu, double sigma){
      normal_mu    = mu;
      normal_sigma = sigma;
      normal_dist = std::normal_distribution<double> (normal_mu,normal_sigma);
    }

    void RandomGenerator::set_seed(std::vector<unsigned int> Seed){
      this->Seed = Seed;
      std::seed_seq seeds(begin(Seed), end(Seed));
      generator = std_random_generator_type(seeds);
    }

    void RandomGenerator::set_seed(unsigned int Seed){
      set_seed(std::vector<unsigned int>(std_seed_size,Seed));
    } 

    double RandomGenerator::generate_uniform(){ 
      return uniform_dist(generator); 
    }

    double RandomGenerator::generate_normal(){
      return normal_dist(generator);
    }

    //=======================================================================
    //=======================================================================

#ifdef USE_GSL

    GSLRandomGenerator::GSLRandomGenerator(int seed) : RandomGenerator(seed){
      name = "gsl_rng_ranlxd1";
      random_generator = gsl_rng_alloc(gsl_random_generator_type);
      allocated = true;
      set_seed(seed);
      uniform_range_lower = 0.0;
      uniform_range_upper = 1.0;
      normal_mu = 0.0;
      normal_sigma = 1.0;
    }

    GSLRandomGenerator::GSLRandomGenerator(){
      name = "gsl_rng_ranlxd1";
      random_generator = gsl_rng_alloc(gsl_random_generator_type);
      allocated = true;
      set_seed(0);
      uniform_range_lower = 0.0;
      uniform_range_upper = 1.0;
      normal_mu = 0.0;
      normal_sigma = 1.0;
    }

    GSLRandomGenerator::~GSLRandomGenerator(){
      free();
    }

    GSLRandomGenerator::GSLRandomGenerator(const GSLRandomGenerator &rhs){
      if(rhs.allocated){ 
        if(allocated) free();
        random_generator = gsl_rng_alloc(gsl_random_generator_type); 
        allocated = true;
        memcpy(random_generator->state, rhs.random_generator->state, rhs.random_generator->type->size);
      } else {
        free();
      }
      name = rhs.name;
      uniform_range_lower = rhs.uniform_range_lower;
      uniform_range_upper = rhs.uniform_range_upper;
      normal_mu = rhs.normal_mu;
      normal_sigma = rhs.normal_sigma;
    }

    void GSLRandomGenerator::free(){
      if(allocated){
        gsl_rng_free(random_generator);
        allocated = false;
      }
    }

    void GSLRandomGenerator::set_seed(unsigned int seed){
      Seed = std::vector<unsigned int>(1,seed);
      gsl_rng_set(random_generator, seed);
    }

    double GSLRandomGenerator::generate_uniform(){
      return uniform_range_lower + (uniform_range_upper-uniform_range_lower)*gsl_rng_uniform(random_generator);
    }

    double GSLRandomGenerator::generate_normal(){
      return normal_mu + gsl_ran_gaussian(random_generator,normal_sigma);
    }

    void GSLRandomGenerator::set_uniform_range(double a, double b){
      uniform_range_lower = a;
      uniform_range_upper = b;
    }

    void GSLRandomGenerator::set_normal_range(double mu, double sigma){
      normal_mu    = mu;
      normal_sigma = sigma;
    }
#endif
  }
}

