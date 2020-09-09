#ifndef RANDOMGENERATOR_HEADER
#define RANDOMGENERATOR_HEADER

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>
#ifdef USE_GSL
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#endif

namespace FML {

    /// This namespace deals with generating random numbers and random fields.
    namespace RANDOM {

        // Select the type of random generator and specify the seed size
        typedef std::mt19937 std_random_generator_type;
#define STDRANDOM_NAME "mt19937"

        // Gsl defines
#define GSLRANDOM_NAME "gsl_rng_ranlxd1"
#ifndef gsl_random_generator_type
#define gsl_random_generator_type gsl_rng_ranlxd1
#endif

        /// This is a class for having a unified interface for random numbers in the library.
        /// If you want to use a different RNG other than c++ random (fiducial: mt19937) or GSL then make a class
        /// that inherits from this class and implement the 3-4 methods it has.
        class RandomGenerator {
          protected:
            std::vector<unsigned int> Seed;
            std_random_generator_type generator;
            std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
            std::normal_distribution<double> normal_dist{0.0, 1.0};
            std::string name{STDRANDOM_NAME};

            double sigma = 1.0;

          public:
            RandomGenerator() = default;
            virtual ~RandomGenerator() = default;

            RandomGenerator(unsigned int seed) {
                Seed = std::vector<unsigned int>(624, seed);
                ;
                std::seed_seq seeds(begin(Seed), end(Seed));
                generator = std_random_generator_type(seeds);
            }

            RandomGenerator(std::vector<unsigned int> seed) { set_seed(seed); }

            void set_normal_sigma(double sigma){
              normal_dist = std::normal_distribution<double>{0.0, sigma};
            }

            virtual void set_seed(std::vector<unsigned int> seed) {
                Seed = seed;
                std::seed_seq seeds(begin(Seed), end(Seed));
                generator = std_random_generator_type(seeds);
            }

            virtual std::shared_ptr<RandomGenerator> clone(){ return std::make_shared<RandomGenerator>(*this); }

            virtual void set_seed(unsigned int Seed) { set_seed(std::vector<unsigned int>(624, Seed)); }

            virtual double generate_uniform() { return uniform_dist(generator); }

            virtual double generate_normal() { return normal_dist(generator); }
        };

        //=======================================================================
        //=======================================================================

#ifdef USE_GSL

        struct GSLRNG {
            gsl_rng * random_generator{nullptr};
            GSLRNG() { random_generator = gsl_rng_alloc(gsl_random_generator_type); }
            ~GSLRNG() {
                if (random_generator != nullptr)
                    gsl_rng_free(random_generator);
            }
        };

        /// Generate random numbers using the GSL library (fiducial: gsl_rng_ranlxd1)
        class GSLRandomGenerator : public RandomGenerator {
          private:
            GSLRNG rng;

          public:
            GSLRandomGenerator() { name = GSLRANDOM_NAME; }

            GSLRandomGenerator(int seed) {
                name = GSLRANDOM_NAME;
                set_seed(seed);
            }

            void set_seed(unsigned int seed) override {
                Seed = std::vector<unsigned int>(1, seed);
                gsl_rng_set(rng.random_generator, seed);
            }
            
            virtual std::shared_ptr<RandomGenerator> clone(){ return std::make_shared<GSLRandomGenerator>(*this); }

            void set_seed(std::vector<unsigned int> seed) override { set_seed(seed[0]); }

            double generate_uniform() override { return gsl_rng_uniform(rng.random_generator); }

            double generate_normal() override { return gsl_ran_gaussian(rng.random_generator, sigma); }
        };

#endif

        //=======================================================================
        //=======================================================================

    } // namespace RANDOM
} // namespace FML
#endif
