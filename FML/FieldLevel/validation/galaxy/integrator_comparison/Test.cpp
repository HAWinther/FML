#include "FieldHMC.h"
#include <iostream>
#include <sstream>
struct TestModel {
 size_t cells()const{return 2;}
 int workspace()const{return 0;}
 bool in_support(const std::vector<double>&p)const{return p.empty();}
 const std::vector<FML::FIELDLEVEL::FRAMEWORK::Parameter>& parameters()const{static const std::vector<FML::FIELDLEVEL::FRAMEWORK::Parameter> p;return p;}
};
namespace FML::FIELDLEVEL::FRAMEWORK {
 template<>struct Execution<TestModel>{
 static void validate(const TestModel&,const Field&w,const std::vector<double>&p){if(w.size()!=2||!p.empty())throw std::invalid_argument("Test dimensions");}
 static void hmc_controls(const TestModel&,bool,const FML::MCMC::HMCOptions&){}
 static FML::MCMC::HMCExecution hmc(const TestModel&){return {};}
 };
}
struct TestTarget {
 TestModel m;std::vector<double> precision;
 const TestModel& model()const{return m;}
 auto field_prior()const{return FML::FIELDLEVEL::FRAMEWORK::InitialFieldPrior::WhitenedGaussian;}
 double evaluate(const std::vector<double>&x,const std::vector<double>&,int&,std::vector<double>&g,std::vector<double>&gp)const{
  gp.clear();g.resize(2);if(std::abs(x[0])>1||std::abs(x[1])>1)return HUGE_VAL;
  double u=0;for(size_t j=0;j<2;++j){g[j]=precision[j]*x[j];u+=.5*precision[j]*x[j]*x[j];}return u;
 }
 std::vector<double> apply_field_metric(std::vector<double>x,const std::vector<double>&mass,int&,double exponent)const{for(size_t j=0;j<2;++j)x[j]*=std::pow(mass[j],exponent);return x;}
};
int main(){try{
 FIELD_HMC::Settings default_settings;if(default_settings.enabled||default_settings.step!=.15||default_settings.leaps!=8)throw std::runtime_error("Legacy defaults changed");
 for(const auto & text:{"FML_FIELD_HMC_1 0 8","FML_FIELD_HMC_1 0.1 0","FML_FIELD_HMC_1 0.1 129","FML_FIELD_HMC_1 nan 8","FML_FIELD_HMC_1 inf 8","FML_FIELD_HMC_1 0.1 8junk","FML_FIELD_HMC_1 0.1 8 extra","bad 0.1 8"}){
  std::istringstream in(text);std::string tag;in>>tag;bool rejected=false;try{FIELD_HMC::Settings s;s.read(in,tag);}catch(const std::invalid_argument&){rejected=true;}if(!rejected)throw std::runtime_error("Invalid settings accepted");
 }
 {std::istringstream in("0.075 16");FIELD_HMC::Settings s;s.read(in,"FML_FIELD_HMC_1");if(!s.enabled||s.step!=.075||s.leaps!=16)throw std::runtime_error("Valid settings rejected");}
 TestTarget target{{},{10000,4000}};const std::vector<double> initial={.01,-.015},mass={10000,4000};
 for(unsigned seed=1;seed<=16;++seed)for(int leaps:{8,16}){
  FML::MCMC::HMCOptions o;o.warmup=0;o.samples=1;o.jitter=false;o.adapt_mass=false;o.adapt_step_size=false;o.step_size=1.2/leaps;o.n_leapfrog=leaps;o.seed=seed;
  auto counted=FIELD_HMC::counted(target,initial,{},mass,o);
  int work=0;FML::MCMC::HMCMass reference_mass;reference_mass.n_custom=2;
  reference_mass.apply_sqrt=[&](auto &v){v=target.apply_field_metric(v,mass,work,.5);};
  reference_mass.apply_inverse=[&](auto &v){v=target.apply_field_metric(v,mass,work,-1.);};
  auto potential=[&](const auto&v,auto&g){std::vector<double>gp;return target.evaluate(v,{},work,g,gp);};
  auto native=FML::MCMC::run_hmc(potential,initial,reference_mass,o);
  if(counted.second!=uint64_t(leaps+1)||counted.first.last!=native.last||counted.first.acceptance!=native.acceptance)throw std::runtime_error("Counted/native mismatch");
  // Independent harmonic-oscillator transfer matrix, rather than replaying the kick/drift code.
  std::mt19937_64 rng(seed);std::normal_distribution<double> normal;std::uniform_real_distribution<double> uniform;
  std::vector<double>x=initial,momentum(2);for(size_t j=0;j<2;++j)momentum[j]=std::sqrt(mass[j])*normal(rng);
  auto energy=[&](const auto&a,const auto&b){double h=0;for(size_t j=0;j<2;++j)h+=.5*(target.precision[j]*a[j]*a[j]+b[j]*b[j]/mass[j]);return h;};const double h0=energy(x,momentum);
  for(int k=0;k<leaps;++k)for(size_t j=0;j<2;++j){const double h=o.step_size,lambda=target.precision[j],m=mass[j],a=1-h*h*lambda/(2*m),b=h/m,c=-h*lambda*(1-h*h*lambda/(4*m));const double next=a*x[j]+b*momentum[j];momentum[j]=c*x[j]+a*momentum[j];x[j]=next;}
  const double prob=std::min(1.,std::exp(h0-energy(x,momentum)));if(std::abs(prob-counted.first.acceptance)>1e-11)throw std::runtime_error("Analytic oscillator acceptance differs");
  if(uniform(rng)>=prob)x=initial;for(size_t j=0;j<2;++j)if(std::abs(x[j]-counted.first.last[j])>1e-12)throw std::runtime_error("Analytic oscillator state differs");
 }
 TestTarget unstable{{},{1e6,4e5}};FML::MCMC::HMCOptions o;o.warmup=0;o.samples=1;o.jitter=false;o.adapt_mass=false;o.adapt_step_size=false;o.step_size=.15;o.n_leapfrog=8;
 const auto failed=FIELD_HMC::counted(unstable,initial,{},std::vector<double>{1,1},o);
 if(failed.second!=2||failed.first.divergences!=1||failed.first.last!=initial)throw std::runtime_error("Early failure cost was counted nominally");
 std::cout<<"PASS: strict settings; 32 native/analytic Gaussian trajectories; exact full and early-stop gradient counts\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
