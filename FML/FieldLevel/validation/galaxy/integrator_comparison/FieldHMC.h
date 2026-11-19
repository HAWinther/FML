#ifndef FML_VALIDATION_CONFIGURED_FIELD_HMC_H
#define FML_VALIDATION_CONFIGURED_FIELD_HMC_H
#include <FML/FieldLevel/Framework.h>
#include <istream>
namespace FIELD_HMC {
 struct Settings {
  bool enabled=false;
  double step=.15;
  int leaps=8;
  void read(std::istream & in,const std::string & tag){
   std::string extra;
   if(tag!="FML_FIELD_HMC_1" || !(in>>step>>leaps) || !std::isfinite(step) || step<=0 || step>1 || leaps<1 || leaps>128 || (in>>extra))
    throw std::invalid_argument("Invalid explicit field integrator settings");
   enabled=true;
  }
 };
 template<class Target>
 std::pair<FML::MCMC::HMCResult,uint64_t> counted(const Target & target,const FML::FIELDLEVEL::Field & initial,
    const std::vector<double> & parameters,const std::vector<double> & frozen,FML::MCMC::HMCOptions options){
  namespace FF=FML::FIELDLEVEL::FRAMEWORK;
  using Model=std::decay_t<decltype(target.model())>;
  const auto & model=target.model();
  FF::Execution<Model>::validate(model,initial,parameters);FF::Execution<Model>::hmc_controls(model,false,options);
  if(target.field_prior()!=FF::InitialFieldPrior::WhitenedGaussian || !model.in_support(parameters))
   throw std::invalid_argument("Counted field update needs supported scalars and Gaussian white prior");
  options.lower.clear();options.upper.clear();auto work=model.workspace();uint64_t calls=0;
  FML::MCMC::HMCMass mass;mass.n_custom=initial.size();
  mass.apply_sqrt=[&](auto & v){v=target.apply_field_metric(v,frozen,work,.5);};
  mass.apply_inverse=[&](auto & v){v=target.apply_field_metric(v,frozen,work,-1.);};
  auto potential=[&](const auto & v,auto & g){++calls;std::vector<double> gp;return target.evaluate(v,parameters,work,g,gp);};
  return {FML::MCMC::run_hmc(potential,initial,std::move(mass),options,nullptr,FF::Execution<Model>::hmc(model)),calls};
 }
}
#endif
