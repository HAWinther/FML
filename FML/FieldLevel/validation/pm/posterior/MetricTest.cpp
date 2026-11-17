#include "FixedMetric.h"
#include <FML/MCMC/HMC.h>
#include <iomanip>
#include <iostream>
int main(){
    const std::vector<double> matrix={4.,1.,1.,2.};FixedMetric metric(matrix);
    std::vector<double> x={.3,-.7},v=x;metric.inverse(v);
    for(size_t i=0;i<2;++i){double value=0;for(size_t j=0;j<2;++j)value+=matrix[2*i+j]*v[j];if(std::abs(value-x[i])>1e-14)return 1;}
    if(metric.reconstruction_error()>1e-14)return 2;
    bool rejected=false;try{FixedMetric invalid({1.,2.,2.,1.});}catch(const std::invalid_argument &){rejected=true;}if(!rejected)return 3;
    using namespace FML::MCMC;HMCMass mass;mass.n_custom=2;mass.apply_sqrt=[&](auto & a){metric.sqrt_multiply(a);};mass.apply_inverse=[&](auto & a){metric.inverse(a);};
    HMCOptions options;options.warmup=100;options.samples=50000;options.n_leapfrog=6;options.step_size=.3;options.adapt_mass=false;options.adapt_step_size=false;options.seed=710;
    const auto result=run_hmc([&](const auto & q,auto & gradient){gradient={4*q[0]+q[1],q[0]+2*q[1]};return .5*(q[0]*gradient[0]+q[1]*gradient[1]);},x,mass,options,[](const auto & q){return q;});
    double first=0,second=0,c00=0,c01=0,c11=0;for(const auto & q:result.recorded){first+=q[0];second+=q[1];c00+=q[0]*q[0];c01+=q[0]*q[1];c11+=q[1]*q[1];}
    const double n=result.recorded.size();std::cout<<std::setprecision(17)<<"mean0,mean1,second00,second01,second11,acceptance,divergences\n"<<first/n<<','<<second/n<<','<<c00/n<<','<<c01/n<<','<<c11/n<<','<<result.acceptance<<','<<result.divergences<<'\n';
    if(std::abs(first/n)>.02||std::abs(second/n)>.02||std::abs(c00/n-2./7)>.02||std::abs(c01/n+1./7)>.02||std::abs(c11/n-4./7)>.02||result.acceptance<.9||result.divergences)return 4;
}
