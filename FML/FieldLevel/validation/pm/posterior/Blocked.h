#ifndef FML_TSC_BLOCKED_H
#define FML_TSC_BLOCKED_H
#include <FML/FieldLevel/validation/pm/posterior/Target.h>
#include <FML/FieldLevel/validation/pm/posterior/ConditionalSlice.h>
// Experimental active-marginal kernel for this fixture, not a generic model adapter.
// Each composition preserves the same joint density: conditional field HMC,
// followed by four A,N slice sweeps. Adaptation ends before retained draws.
inline FML::MCMC::HMCResult run_blocked(CenteredTarget & target,Field z,FML::MCMC::HMCOptions options,
                                      const FML::MCMC::HMCExecution & execution,
                                      const FML::MCMC::HMCMass * field_mass=nullptr) {
    using namespace FML::MCMC;
    const size_t d=target.basis.dimension;
    options.lower.clear();options.upper.clear();options.verbose_every=0;
    HMCMass mass;if(field_mass)mass=*field_mass;else mass.diagonal.assign(d,21.);
    HMCState state;HMCRunControl control;control.state=&state;control.max_iterations=1;
    std::mt19937_64 scalar_rng(options.seed+11000);
    Field gradient;target.evaluate(z,gradient);
    HMCResult result;
    for(int iteration=0;iteration<options.warmup+options.samples;++iteration){
        const auto cache=target.snapshot();const Field previous(z.begin(),z.begin()+d);
        const auto field=run_hmc([&](const Field & q,Field & g){
            Field full=q;full.insert(full.end(),z.begin()+d,z.end());Field all;
            const double value=target.evaluate(full,all);g.assign(all.begin(),all.begin()+d);return value;
        },previous,mass,options,nullptr,execution,control);
        // Rejected trajectories leave the target cache at their last trial point.
        if(field.last==previous)target.restore(cache);
        std::copy(field.last.begin(),field.last.end(),z.begin());
        const size_t before=target.evaluations;
        for(int sweep=0;sweep<4;++sweep){
            z[d]=conditional_slice(z[d],-4.,4.,[&](double x){return target.conditional(0,x);},scalar_rng);
            z[d+1]=conditional_slice(z[d+1],-16.,-4.,[&](double x){return target.conditional(1,x);},scalar_rng);
        }
        // Refresh the resumed HMC force and energy after changing its conditioning
        // parameters; the reference-density cache makes this PM-free.
        const double value=target.evaluate(z,gradient);
        require(target.evaluations==before,"Scalar updates triggered a PM evaluation");
        state.potential=value;state.gradient.assign(gradient.begin(),gradient.begin()+d);
        if(iteration>=options.warmup){result.recorded.push_back(z);result.potential.push_back(value);}
        result.acceptance=field.acceptance;result.divergences=field.divergences;result.step_size=field.step_size;
        result.completed_iterations=field.completed_iterations;result.completed_samples=field.completed_samples;
        if(iteration%32==0&&FML::ThisTask==0)std::cout<<"[blocked] iteration "<<iteration<<" evaluations "<<target.evaluations
            <<" U "<<value<<" step "<<state.step_size<<std::endl;
    }
    result.last=z;result.diagonal=state.diagonal;return result;
}
#endif
