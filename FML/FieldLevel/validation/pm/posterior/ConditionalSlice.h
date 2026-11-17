#ifndef FML_TSC_CONDITIONAL_SLICE_H
#define FML_TSC_CONDITIONAL_SLICE_H
#include <cmath>
#include <random>
#include <stdexcept>
// Validation-only bounded slice kernel. Requires a continuous convex negative
// log density on [lower,upper], hence connected slices. Starting with the full
// support avoids a tuning width or an approximate stepping-out truncation.
template<class Potential>
double conditional_slice(double current,double lower,double upper,Potential potential,std::mt19937_64 & rng) {
    if(!(std::isfinite(lower)&&std::isfinite(upper)&&lower<upper&&current>=lower&&current<=upper))
        throw std::invalid_argument("Invalid conditional slice support");
    std::uniform_real_distribution<double> uniform(0.,1.);
    double u=0.;while(u==0.)u=uniform(rng);
    const double initial=potential(current),level=initial-std::log(u);
    if(!std::isfinite(initial))throw std::invalid_argument("Nonfinite conditional slice start");
    for(int attempt=0;attempt<10000;++attempt){
        const double proposal=lower+(upper-lower)*uniform(rng),value=potential(proposal);
        if(std::isfinite(value)&&value<=level)return proposal;
        if(proposal<current)lower=proposal;else upper=proposal;
    }
    throw std::runtime_error("Conditional slice exhausted shrinkage budget");
}
#endif
