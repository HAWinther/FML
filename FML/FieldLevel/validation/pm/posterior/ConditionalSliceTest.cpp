#include "ConditionalSlice.h"
#include <iomanip>
#include <iostream>
// Independent quadrature comparison is performed by check_conditionals.py.
int main(){
    std::cout<<std::setprecision(17)<<"case,mean,second,cdf\n";
    for(int which=0;which<3;++which){
        std::mt19937_64 rng(71000+which);double x=which==0?-3.9:-15.9,sum=0,square=0,cdf=0;
        const double lo=which==0?-4:-16,hi=which==0?4:-4,cut=which==0?0:-9.2;
        auto density=[&](double v){
            if(which==0)return .5*120*std::exp(-v)+61*v+2*v*v;
            if(which==1)return .5*256*std::exp(-(v-std::log(1e-4)))+128*(v-std::log(1e-4))+2*std::pow(v-std::log(1e-4),2);
            return .5*std::pow((v+15.95)/.08,2); // narrow truncated Gaussian at support edge
        };
        for(int i=-1000;i<200000;++i){x=conditional_slice(x,lo,hi,density,rng);if(i>=0){sum+=x;square+=x*x;cdf+=x<=cut;}}
        std::cout<<which<<','<<sum/200000<<','<<square/200000<<','<<cdf/200000<<'\n';
    }
    bool rejected=false;std::mt19937_64 rng(1);
    try{conditional_slice(5.,-4.,4.,[](double x){return x*x;},rng);}catch(const std::invalid_argument &){rejected=true;}
    if(!rejected)return 1;
}
