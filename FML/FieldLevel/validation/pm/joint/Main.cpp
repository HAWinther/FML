#include <FML/FieldLevel/validation/common/PMJointProblem.h>
static int integer(const char * text){size_t end;int n=std::stoi(text,&end);if(end!=std::string(text).size())throw std::invalid_argument("Invalid integer");return n;}
int main(int argc,char **argv){try{
    Controls c;c.seed=91;c.power=.002;c.final=1.;c.window=2;c.reference=16;c.assignment=3;int vary=1,fd=1;
    FML::COMMUNICATION::collective_try([&]{if(argc<7||(argc>9&&argc!=11))throw std::invalid_argument("pm_tsc_joint PREFIX DATA N M STEPS ITERATIONS [VARY] [FD] [IC_SEED NOISE_SEED]");c.prefix=argv[1];c.data=argv[2];c.n=integer(argv[3]);c.m=integer(argv[4]);c.steps=integer(argv[5]);c.fit=integer(argv[6]);if(argc>=8)vary=integer(argv[7]);if(argc>=9)fd=integer(argv[8]);if(argc==11){c.seed=integer(argv[9]);c.noise_seed=integer(argv[10]);}},"Joint PM controls");
    require(c.seed>=0&&c.noise_seed>=0&&c.n>=16&&c.n<=64&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64&&c.fit>=0&&c.fit<=200&&(vary==0||vary==1)&&(fd==0||fd==1),"Invalid bounded joint PM controls");
    FF::CHECKPOINT::same_string(std::to_string(c.seed)+","+std::to_string(c.noise_seed)+","+std::to_string(c.n)+","+std::to_string(c.m)+","+std::to_string(c.steps)+","+std::to_string(c.fit)+","+std::to_string(vary)+","+std::to_string(fd));
    FF::CHECKPOINT::same_string(c.prefix);FF::CHECKPOINT::same_string(c.data);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto model=make_joint(c);const auto p0=model.initial_parameters();auto work=model.workspace();
    const auto data=load_data(model,c);FF::BasicPosterior<JointModel> posterior(model,data);
    const auto truth=coefficients(c.seed);auto initial=coefficients(300);for(auto & x:initial)x*=.05;const auto start=expand(model,initial);
    auto identities=output(c.prefix+".identities.csv");if(FML::ThisTask==0)identities<<"point,potential,field_max,scalar_scaled_max,white_norm2,white_dot_gradient,amplitude_gradient,amplitude_expected,amplitude_error,noise_gradient,noise_expected,noise_error,chi2,dof\n";
    auto derivatives=output(c.prefix+".fd.csv");if(FML::ThisTask==0)derivatives<<"point,direction,h,analytic,numeric\n";
    auto direction_coeff=coefficients(11);const double length=std::sqrt(inner(direction_coeff,direction_coeff));for(auto & x:direction_coeff)x/=length;
    const auto direction=expand(model,direction_coeff);
    auto check_point=[&](const std::string & name,const Field & white,const std::vector<double> & p,bool finite_differences){
        Field g;std::vector<double> gp;const double u=posterior.evaluate(white,p,work,g,gp);const auto prediction=model.predict(white,p,work);
        double field_max=0,scalar_max=0,chi2=0,dof=0;for(double x:g)field_max=std::max(field_max,std::abs(x));field_max=model.maximum(field_max);
        for(size_t j=0;j<gp.size();++j)scalar_max=std::max(scalar_max,std::abs(gp[j])*model.parameters()[j].scale());
        for(size_t j=0;j<prediction.size();++j){const double weight=posterior.likelihood().weights()[j];chi2+=weight*std::norm(prediction[j]-data[j])*model.volume()/std::exp(p.back());dof+=weight;}
        chi2=model.sum(chi2);dof=model.sum(dof);const double square=model.sum(inner(white,white)),sg=model.sum(inner(white,g));
        const double amplitude=.5*(sg-square)+model.parameters()[0].derivative(p[0]);
        const double noise=.5*(dof-chi2)+model.parameters()[1].derivative(p[1]);
        if(FML::ThisTask==0)identities<<name<<','<<u<<','<<field_max<<','<<scalar_max<<','<<square<<','<<sg<<','<<gp[0]<<','<<amplitude<<','<<gp[0]-amplitude<<','<<gp[1]<<','<<noise<<','<<gp[1]-noise<<','<<chi2<<','<<dof<<'\n';
        if(finite_differences){const auto reference=posterior.value_reference(white,p,work);
            for(int kind=0;kind<3;++kind){const double da=kind?.2:0.,dn=kind?-.1:0.;
                const double analytic=(kind==1?0.:model.sum(inner(g,direction)))+da*gp[0]+dn*gp[1];
                for(double h:{.001,.0003,.0001,.00003,.00001,.000003,.000001}){
                    auto plus=white,minus=white;auto pp=p,pm=p;
                    if(kind!=1)for(size_t j=0;j<white.size();++j){plus[j]+=h*direction[j];minus[j]-=h*direction[j];}
                    pp[0]+=h*da;pm[0]-=h*da;pp[1]+=h*dn;pm[1]-=h*dn;
                    const auto up=posterior.value_change(plus,pp,work,white,reference),down=posterior.value_change(minus,pm,work,white,reference);
                    if(FML::ThisTask==0)derivatives<<name<<','<<kind<<','<<h<<','<<analytic<<','<<(up.change-down.change)/(2*h)<<'\n';
                }
            }
        }
        return std::make_pair(u,std::max(field_max,vary?scalar_max:0.));
    };
    const auto truth_white=expand(model,truth);check_point("truth",truth_white,p0,false);
    auto f=output(c.prefix+".truth_modes.csv");row(f,global_modes(model,model.predict(truth_white,p0,work)));
    const auto start_check=check_point("start",start,p0,fd);f=output(c.prefix+".start_modes.csv");row(f,global_modes(model,model.predict(start,p0,work)));
    FF::MAPOptions options;options.vary_parameters=vary;options.max_iterations=c.fit;options.tolerance=2e-5;options.history=5;options.stable_objective_changes=true;
    model.forwards=model.pullbacks=0;model.unsupported_trials=0;model.recording=true;const auto begin=std::chrono::steady_clock::now();const auto fit=FF::map(posterior,start,p0,options);model.recording=false;
    const double seconds=model.maximum(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());const auto forwards=model.forwards,pullbacks=model.pullbacks;
    require(model.gradient_parameters.size()==fit.gradient_evaluations,"Joint gradient parameter trace incomplete");
    require(model.gradient_parameters.front()==p0&&model.gradient_parameters.back()==fit.fit.parameters,"Joint parameter trace mismatch");
    const auto final_check=check_point("endpoint",fit.fit.white,fit.fit.parameters,fd);
    require(final_check.first==fit.fit.potential&&std::abs(final_check.second-fit.fit.gradient_norm)<1e-12,"Joint endpoint replay mismatch");
    const auto prediction=model.predict(fit.fit.white,fit.fit.parameters,work);const auto projected=project(model,fit.fit.white);
    auto physical=project(model,model.reconstruction(work).initial_density);for(auto & x:physical)x/=std::sqrt(double(model.layout().global_cells));
    f=output(c.prefix+".physical.csv");auto physical_truth=truth,physical_start=initial;for(auto & x:physical_truth)x*=std::sqrt(c.power);for(auto & x:physical_start)x*=std::sqrt(c.power);row(f,physical_truth);row(f,physical_start);row(f,physical);
    f=output(c.prefix+".coefficients.csv");row(f,truth);row(f,initial);row(f,projected);
    f=output(c.prefix+".fit_modes.csv");row(f,global_modes(model,prediction));f=output(c.prefix+".data.csv");if(FML::ThisTask==0)f<<identity(c)<<'\n';row(f,global_modes(model,data));
    f=output(c.prefix+".fit.csv");if(FML::ThisTask==0)f<<"initial_potential,potential,iterations,status,joint_gradient_norm,log_amplitude,log_noise_power,seconds,forwards,pullbacks,value_evaluations,gradient_evaluations,reference_evaluations,peak_history_pairs,unsupported_trials\n"<<start_check.first<<','<<fit.fit.potential<<','<<fit.fit.iterations<<','<<int(fit.fit.status)<<','<<fit.fit.gradient_norm<<','<<fit.fit.parameters[0]<<','<<fit.fit.parameters[1]<<','<<seconds<<','<<forwards<<','<<pullbacks<<','<<fit.value_evaluations<<','<<fit.gradient_evaluations<<','<<fit.reference_evaluations<<','<<fit.peak_history_pairs<<','<<model.unsupported_trials<<'\n';
    f=output(c.prefix+".trace.csv");if(FML::ThisTask==0){f<<"iteration,potential,gradient_norm,field_gradient_l2,step,log_amplitude,log_noise_power,value_evaluations,gradient_evaluations,reference_evaluations,accepted_change,armijo_bound\n";
        for(const auto & x:fit.trace){const auto & p=model.gradient_parameters.at(x.gradient_evaluations-1);f<<x.iteration<<','<<x.potential<<','<<x.gradient_norm<<','<<x.field_gradient_l2<<','<<x.step<<','<<p[0]<<','<<p[1]<<','<<x.value_evaluations<<','<<x.gradient_evaluations<<','<<x.reference_evaluations<<','<<x.accepted_change<<','<<x.armijo_bound<<'\n';}}
    f=output(c.prefix+".controls.csv");if(FML::ThisTask==0)f<<"n,m,steps,max_iterations,vary,fd,amplitude_lower,amplitude_upper,amplitude_mean,amplitude_sigma,noise_mean,noise_sigma,tolerance\n"<<c.n<<','<<c.m<<','<<c.steps<<','<<c.fit<<','<<vary<<','<<fd<<",-4,4,0,.5,"<<p0[1]<<",.5,2e-5\n";
    if(FML::ThisTask==0)std::cout<<"Joint PM status="<<int(fit.fit.status)<<" iterations="<<fit.fit.iterations<<" norm="<<fit.fit.gradient_norm<<" log_amplitude="<<fit.fit.parameters[0]<<" log_noise="<<fit.fit.parameters[1]<<std::endl;
    return 0;
}catch(const std::exception & e){if(FML::ThisTask==0)std::cerr<<e.what()<<std::endl;return 1;}}
