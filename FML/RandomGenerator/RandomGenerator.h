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
#define STDRANDOM_SEEDSIZE 624

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

            /// Construct and initialize random number generator
            RandomGenerator(unsigned int seed) : RandomGenerator(std::vector<unsigned int>(STDRANDOM_SEEDSIZE, seed)) {}

            /// Construct and initialize random number generator
            RandomGenerator(std::vector<unsigned int> seed) : RandomGenerator() { set_seed(seed); }

            /// Set the standard deviation sigma used when generating a normal distributed random number
            void set_normal_sigma(double sigma) { normal_dist = std::normal_distribution<double>{0.0, sigma}; }

            /// Make a clone of the state of the random number generator
            virtual std::unique_ptr<RandomGenerator> clone() const { return std::make_unique<RandomGenerator>(*this); }

            /// Set the seed (vector of numbers as some rngs have a large state)
            virtual void set_seed(std::vector<unsigned int> seed) {
                assert(seed.size() > 0);
                Seed = seed;
                std::seed_seq seeds(begin(Seed), end(Seed));
                generator = std_random_generator_type(seeds);
            }

            /// Set the seed (single number)
            virtual void set_seed(unsigned int seed) { set_seed(std::vector<unsigned int>(STDRANDOM_SEEDSIZE, seed)); }

            /// Generate a random number uniformly distributed in [0,1)
            virtual double generate_uniform() { return uniform_dist(generator); }

            /// Generate a random number with a normal distribution N(0,sigma) where sigma by default is 1.
            virtual double generate_normal() { return normal_dist(generator); }
        };

        //=======================================================================
        //=======================================================================

#ifdef USE_GSL

        // A wrapper for holding the GSL data and making sure when we clone the object the state gets copied
        // not just the pointer to the state
        struct my_gsl_rng {

            std::shared_ptr<gsl_rng> random_generator;

            my_gsl_rng() {
                random_generator = std::shared_ptr<gsl_rng>(gsl_rng_alloc(gsl_random_generator_type), gsl_rng_free);
            }

            my_gsl_rng(const my_gsl_rng & rhs) {
                // We cannot copy a random generator that has not been created (should never be the case)
                assert(rhs.random_generator);
                random_generator = std::shared_ptr<gsl_rng>(gsl_rng_clone(rhs.random_generator.get()), gsl_rng_free);
            }

            gsl_rng * get() { return random_generator.get(); }
        };

        /// Generate random numbers using the GSL library (fiducial: gsl_rng_ranlxd1)
        class GSLRandomGenerator : public RandomGenerator {
          private:
            my_gsl_rng rng;

          public:
            GSLRandomGenerator() : RandomGenerator() { name = GSLRANDOM_NAME; }

            GSLRandomGenerator(unsigned int seed) : GSLRandomGenerator() { set_seed_gsl(seed); }

            GSLRandomGenerator(std::vector<unsigned int> seed) : GSLRandomGenerator() {
                assert(seed.size() > 0);
                set_seed_gsl(seed[0]);
            }

            virtual std::unique_ptr<RandomGenerator> clone() const override {
                return std::make_unique<GSLRandomGenerator>(*this);
            }

            // To avoid calling a virtual function in the constructor
            // and be sure the right method is called
            void set_seed_gsl(unsigned int seed) {
                Seed = std::vector<unsigned int>(1, seed);
                gsl_rng_set(rng.get(), seed);
            }

            virtual void set_seed(unsigned int seed) override { set_seed_gsl(seed); }

            virtual void set_seed(std::vector<unsigned int> seed) override {
                assert(seed.size() > 0);
                set_seed(seed[0]);
            }

            virtual double generate_uniform() override { return gsl_rng_uniform(rng.get()); }

            virtual double generate_normal() override { return gsl_ran_gaussian(rng.get(), sigma); }
        };

#endif

        //=======================================================================
        //=======================================================================

    } // namespace RANDOM
} // namespace FML
#endif
