#include <FML/FieldLevel/PMAssignment.h>
#include <FML/tests/TestHelpers.h>
#include <vector>
#include <algorithm>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::TESTS::check;
struct Stencil { std::array<double,64> w{}; std::array<std::array<double,64>,3> dw{}; };
static Stencil stencil(std::array<double,3> x) {
    Stencil s;
    FF::DETAIL::tsc_stencil(4,x,[&](auto i,double w,const auto & dw){s.w[i]+=w;for(int d=0;d<3;++d)s.dw[d][i]+=dw[d];});
    return s;
}
// Independent periodic cardinal spline on every grid node, without nearest-node routing.
static double kernel(double x) { x=std::abs(x);return x<=.5?.75-x*x:x<1.5?.5*(1.5-x)*(1.5-x):0.; }
int main() {
    for (auto x : {std::array<double,3>{0.,0.,0.}, {.125,.375,.875}, {1.,0.,1.}, {-.01,.999999,.27}, {.211,.472,.713}}) {
        const auto s=stencil(x);double sum=0,err=0;std::array<double,3> total{};
        for(size_t i=0;i<64;++i){double expected=1;const int c[3]={int(i/16),int(i/4%4),int(i%4)};
            for(int d=0;d<3;++d){double z=4*x[d]-c[d];z-=4*std::round(z/4);expected*=kernel(z);total[d]+=s.dw[d][i];}
            sum+=s.w[i];err=std::max(err,std::abs(expected-s.w[i]));check(s.w[i]>=0,"TSC nonnegative weights");}
        check(std::abs(sum-1)<2e-15&&err<2e-15,"periodic TSC mass partition and independent cardinal kernel");
        for(double t:total)check(std::abs(t)<3e-15,"TSC derivative partition");
        for(int d=0;d<3;++d){double best=HUGE_VAL;
            for(double h:{1e-4,1e-5,1e-6,1e-7}){auto xp=x,xm=x;xp[d]+=h;xm[d]-=h;const auto p=stencil(xp),m=stencil(xm);double e=0;
                for(size_t i=0;i<64;++i)e=std::max(e,std::abs((p.w[i]-m.w[i])/(2*h)-s.dw[d][i]));best=std::min(best,e);}
            check(best<5e-6,"TSC derivative across half-cell knots and periodic seam");}
    }
    for(double knot:{0.,.125,.375,1.}){
        const std::array<double,3> x{knot,.23,.67};auto xp=x,xm=x;xp[0]+=1e-9;xm[0]-=1e-9;const auto p=stencil(xp),m=stencil(xm),s=stencil(x);double e=0;
        for(int d=0;d<3;++d)for(size_t i=0;i<64;++i)e=std::max({e,std::abs(p.dw[d][i]-s.dw[d][i]),std::abs(m.dw[d][i]-s.dw[d][i])});
        check(e<1e-7,"TSC first derivatives continuous at support changes and wrap");
    }
    return FML::TESTS::summary("FieldLevel TSC assignment");
}
