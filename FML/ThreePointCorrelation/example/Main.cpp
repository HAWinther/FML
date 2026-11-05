#include <FML/ThreePointCorrelation/ThreePointCorrelation.h>
#include <FML/RandomGenerator/CounterBasedRandom.h>
#include <iostream>
#include <iomanip>
using namespace FML::THREEPOINT;
int main(int argc, char ** argv) {
    FML::FMLSetup::init(&argc, &argv);
    // Analytic toy: a uniform periodic Poisson sample with an independent uniform random catalogue.
    std::vector<Sample> data(500), random(1000);
    for (auto * cat : {&data, &random})
        for (size_t i = 0; i < cat->size(); ++i) {
            auto & p = (*cat)[i];
            p.id = i;
            FML::RANDOM::COUNTER_BASED::Stream rng(cat == &data ? 42 : 43, i);
            for (double & x : p.position)
                x = 500 * rng.uniform();
        }
    auto result = periodic(data, random, {500, {20, 45, 70, 100}, 4});
    if (FML::ThisTask == 0) {
        std::cout << "# Uniform periodic toy; finite random noise retained. Expected ensemble zeta=0.\n"
                  << "# ell r1_low r1_high r2_low r2_high zeta_ell\n"
                  << std::setprecision(12);
        const auto & e = result.options.edges;
        for (int l = 0; l <= result.options.lmax; ++l)
            for (size_t a = 0; a + 1 < e.size(); ++a)
                for (size_t b = a; b + 1 < e.size(); ++b)
                    std::cout << l << ' ' << e[a] << ' ' << e[a + 1] << ' ' << e[b] << ' ' << e[b + 1] << ' '
                              << result.zeta[result.index(l, a, b)] << '\n';
    }
}
