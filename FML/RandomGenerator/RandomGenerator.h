#ifndef RANDOMGENERATOR_HEADER
#define RANDOMGENERATOR_HEADER

#include <vector>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdio>
#include <cassert>
#include <memory>
#ifdef USE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

namespace FML {
  namespace RANDOM {

    // Select the type of random generator and specify the seed size
    typedef std::mt19937 std_random_generator_type;
#define STDRANDOM_NAME "mt19937"

    // Gsl defines
#define GSLRANDOM_NAME "gsl_rng_ranlxd1"
#ifndef gsl_random_generator_type
#define gsl_random_generator_type gsl_rng_ranlxd1
#endif

    class RandomGenerator {
      protected:

        std::vector<unsigned int> Seed;
        std_random_generator_type generator;
        std::uniform_real_distribution<double> uniform_dist{0.0,1.0};
        std::normal_distribution<double> normal_dist{0.0,1.0};
        std::string name{STDRANDOM_NAME};

      public:

        RandomGenerator() = default;
        virtual ~RandomGenerator() = default;
        
        RandomGenerator(unsigned int seed){
          Seed = std::vector<unsigned int>(624,seed);;
          std::seed_seq seeds(begin(Seed), end(Seed));
          generator = std_random_generator_type(seeds);
        }

        RandomGenerator(std::vector<unsigned int> seed){
          set_seed(seed);
        }

        virtual void set_seed(std::vector<unsigned int> seed){
          Seed = seed;
          std::seed_seq seeds(begin(Seed), end(Seed));
          generator = std_random_generator_type(seeds);
        }

        virtual void set_seed(unsigned int Seed){
          set_seed(std::vector<unsigned int>(624,Seed));
        } 

        virtual double generate_uniform(){ 
          return uniform_dist(generator); 
        }

        virtual double generate_normal(){
          return normal_dist(generator);
        }

    };
    
    //=======================================================================
    //=======================================================================

#ifdef USE_GSL

    struct GSLRNG{
      gsl_rng * random_generator{nullptr};
      GSLRNG(){
        random_generator = gsl_rng_alloc(gsl_random_generator_type);
      }
      ~GSLRNG(){
        if(random_generator != nullptr)
          gsl_rng_free(random_generator);
      }
    };

    class GSLRandomGenerator : public RandomGenerator {
      private:

        GSLRNG rng;

      public:

        GSLRandomGenerator(){
          name = GSLRANDOM_NAME;
        }

        GSLRandomGenerator(int seed) {
          name = GSLRANDOM_NAME;
          set_seed(seed);
        }

        void set_seed(unsigned int seed) override{
          Seed = std::vector<unsigned int>(1,seed);
          gsl_rng_set(rng.random_generator, seed);
        }
        
        void set_seed(std::vector<unsigned int> seed) override{
          set_seed(seed[0]);
        }

        double generate_uniform() override{
          return gsl_rng_uniform(rng.random_generator);
        }

        double generate_normal() override{
          return gsl_ran_gaussian(rng.random_generator,1.0);
        }
    
    };

#endif

    //=======================================================================
    //=======================================================================

  }
}
#endif
