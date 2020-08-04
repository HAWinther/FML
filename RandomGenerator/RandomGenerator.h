#ifndef RANDOMGENERATOR_HEADER
#define RANDOMGENERATOR_HEADER

#include <vector>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdio>
#include <cassert>
#ifdef USE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

namespace FML {
  namespace RANDOM {

    // Select the type of random generator and specify the seed size
    typedef std::mt19937 std_random_generator_type;
    const size_t std_seed_size = 624;
    const std::string standardname = "mt19937";

    // Select type of GSL random generator
#ifdef USE_GSL
#ifndef gsl_random_generator_type
#define gsl_random_generator_type gsl_rng_ranlxd1
#endif
#endif

    // Default ranges for the uniform and normal distribution
    const double fiducial_uniform_range_lower = 0.0;
    const double fiducial_uniform_range_upper = 1.0;
    const double fiducial_normal_mu           = 0.0;
    const double fiducial_normal_sigma        = 1.0;

    class RandomGenerator {
      protected:

        std::vector<unsigned int> Seed;
        std_random_generator_type generator;
        std::uniform_real_distribution<double> uniform_dist;
        std::normal_distribution<double> normal_dist;

        std::string name;

        double uniform_range_lower;
        double uniform_range_upper;
        double normal_mu;
        double normal_sigma;

      public:

        RandomGenerator();
        RandomGenerator(unsigned int Seed);
        RandomGenerator(std::vector<unsigned int> Seed);
        ~RandomGenerator();
        RandomGenerator(const RandomGenerator &rhs);

        RandomGenerator& operator=(const RandomGenerator &rhs){
          Seed = rhs.Seed;
          generator = rhs.generator;
          uniform_range_lower = rhs.uniform_range_lower;
          uniform_range_upper = rhs.uniform_range_upper;
          uniform_dist = rhs.uniform_dist;
          normal_mu = rhs.normal_mu;
          normal_sigma = rhs.normal_sigma;
          normal_dist = rhs.normal_dist;
          return *this;
        }

        virtual void free(){}
        virtual void set_seed(std::vector<unsigned int> Seed);
        virtual void set_seed(unsigned int Seed);
        virtual void set_uniform_range(double a, double b);
        virtual void set_normal_range(double mu, double sigma);
        virtual double generate_uniform();
        virtual double generate_normal();
    };

#ifdef USE_GSL
    class GSLRandomGenerator : public RandomGenerator {
      private:

        gsl_rng *random_generator;
        bool allocated; 

      public:

        GSLRandomGenerator(int seed);
        GSLRandomGenerator();
        ~GSLRandomGenerator();
        GSLRandomGenerator(const GSLRandomGenerator &rhs);

        GSLRandomGenerator& operator=(const GSLRandomGenerator &rhs){
          if(this != &rhs){
            if(rhs.allocated){ 
              if(allocated) free();
              random_generator = gsl_rng_alloc(gsl_rng_ranlxd1); 
              allocated = true;
              memcpy(random_generator->state, rhs.random_generator->state, rhs.random_generator->type->size);
            } else {
              free();
            }
            uniform_range_lower = rhs.uniform_range_lower;
            uniform_range_upper = rhs.uniform_range_upper;
            normal_mu = rhs.normal_mu;
            normal_sigma = rhs.normal_sigma;
          }
          return *this;
        }

        void free();
        void set_seed(unsigned int seed);
        void set_uniform_range(double a, double b);
        void set_normal_range(double mu, double sigma);
        double generate_uniform();
        double generate_normal();
    };
#endif
  }
}

#endif
