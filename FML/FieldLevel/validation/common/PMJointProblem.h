#ifndef FML_PM_TSC_JOINT_VALIDATION_H
#define FML_PM_TSC_JOINT_VALIDATION_H
#include <FML/FieldLevel/validation/common/PMProblem.h>

// Validation-only recording of parameters at full-gradient evaluations. MAP
// trace counters identify the accepted evaluations, including any rejected ones.
struct JointModel : Model {
    using Model::Model;
    mutable bool recording=false;
    mutable size_t unsupported_trials=0;
    bool in_support(const std::vector<double> & p)const {const bool ok=Model::in_support(p);if(recording&&!ok)++unsupported_trials;return ok;}
    mutable std::vector<std::vector<double>> gradient_parameters;
    void pullback(const std::vector<double> & p,const FF::Modes & g,Workspace & w,Field & gs,std::vector<double> & gp)const {
        Model::pullback(p,g,w,gs,gp);
        if(recording)FML::COMMUNICATION::collective_try([&]{gradient_parameters.push_back(p);if(FML::ThisTask==0&&gradient_parameters.size()%10==0)std::cout<<"Joint full gradients="<<gradient_parameters.size()<<" A="<<p[0]<<" N="<<p[1]<<std::endl;},"Joint PM parameter trace");
    }
};
namespace FML::FIELDLEVEL::FRAMEWORK {template<>struct Execution<::JointModel>:Execution<::BaseModel>{};}

static inline JointModel make_joint(const Controls & c) {
    FF::Configuration config;config.mesh=c.n;config.density_mesh=c.m;config.boxsize=1.;config.evolution=FF::Evolution::Linear;
    config.initial_cutoff=config.bias_cutoff=2*M_PI*std::sqrt(9.5);config.likelihood_cutoff=2*M_PI*std::sqrt(16.5);
    // Same Gaussian mean/scale as the fixed fit; explicitly wider numerical
    // amplitude support to avoid clipping the whitened-coordinate joint mode.
    FF::Spectrum power{{{"log_amplitude",0.,0.,.5,-4.,4.}},
        [value=c.power](double,const auto & p){return FF::Power{value*std::exp(p[0]),{1.}};}};
    FF::PMSettings pm;pm.assignment=FF::PMAssignment::TSC;pm.force_window=FF::PMForceWindow::TSCCompensated;pm.scale_factors.clear();
    for(int j=0;j<=c.steps;++j)pm.scale_factors.push_back(.2+(c.final-.2)*j/c.steps);
    const double noise=std::log(noise_power);FF::PMObservationSettings observed;observed.reference_cic_mesh=16;
    return JointModel(config,power,pm,{"log_noise_power",noise,noise,.5,-16.,-4.},{8},FML::domain,observed);
}

#endif
