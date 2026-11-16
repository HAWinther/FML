#include <FML/FieldLevel/pm_nonlinear_validation/Validation.h>
// Bounded large-particle density study: a single forward trajectory, no pullback.
static void forward_only(const Controls & c,Model & model) {
    const auto truth=coefficients(c.seed);const auto white=expand(model,truth);
    const auto p=model.initial_parameters();auto work=model.workspace();
    const auto data=c.data=="-"?model.mock(white,p,92,work):load_data(model,c);
    FF::BasicPosterior<Model> posterior(model,data);
    const auto & predicted=model.predict(white,p,work);
    FF::Modes unused;double unused_noise;
    double u=model.sum(posterior.likelihood().evaluate(predicted,p.back(),unused,unused_noise)+.5*inner(white,white));
    for(size_t j=0;j<p.size();++j)u+=model.parameters()[j].penalty(p[j]);
    const auto & fields=model.reconstruction(work);
    auto physical=project(model,fields.initial_density);
    for(auto & x:physical)x/=std::sqrt(double(model.layout().global_cells));
    double mass=0,momentum=0;
    for(const auto & d:fields.diagnostics) {
        mass=std::max(mass,std::abs(d.mass_error));
        for(double v:d.mean_momentum)momentum=std::max(momentum,std::abs(v));
    }
    auto f=output(c.prefix+".data.csv");if(FML::ThisTask==0)f<<identity(c)<<'\n';row(f,global_modes(model,data));
    f=output(c.prefix+".modes.csv");row(f,global_modes(model,predicted));
    f=output(c.prefix+".initial.csv");row(f,physical);
    f=output(c.prefix+".forward_meta.csv");
    if(FML::ThisTask==0)f<<"n,m,steps,seed,power,final,active,potential,density_rms,diagnostic_nodes,max_mass_error,max_mean_momentum\n"
        <<c.n<<','<<c.m<<','<<c.steps<<','<<c.seed<<','<<c.power<<','<<c.final<<','<<truth.size()<<','<<u<<','
        <<fields.diagnostics.back().density_rms<<','<<fields.diagnostics.size()<<','<<mass<<','<<momentum<<'\n';
}
static void run(const Controls & c,Model & model) {
    const auto truth=coefficients(c.seed); const auto white=expand(model,truth);
    const auto p=model.initial_parameters();auto work=model.workspace();
    const auto data=c.data=="-"?model.mock(white,p,92,work):load_data(model,c);
    FF::BasicPosterior<Model> posterior(model,data);Field gradient;std::vector<double> gp;
    const double u=posterior.evaluate(white,p,work,gradient,gp);
    const auto report=work.adjoint_report();const auto predicted=model.predict(white,p,work);
    const auto & fields=model.reconstruction(work);
    auto physical=project(model,fields.initial_density);for(auto & x:physical)x/=std::sqrt(double(model.layout().global_cells));
    auto projected=project(model,gradient);projected.insert(projected.end(),gp.begin(),gp.end());
    auto file=output(c.prefix+".data.csv");if(FML::ThisTask==0)file<<identity(c)<<'\n';row(file,global_modes(model,data));
    auto f=output(c.prefix+".modes.csv");row(f,global_modes(model,predicted));
    f=output(c.prefix+".initial.csv");row(f,physical);f=output(c.prefix+".gradient.csv");row(f,projected);
    f=output(c.prefix+".meta.csv");
    if(FML::ThisTask==0) f<<"n,m,steps,seed,power,final,active,potential,density_rms,checkpoint_bytes,force_evaluations\n"
        <<c.n<<','<<c.m<<','<<c.steps<<','<<c.seed<<','<<c.power<<','<<c.final<<','<<truth.size()<<','<<u<<','
        <<fields.diagnostics.back().density_rms<<','<<report.checkpoint_payload_bytes<<','<<report.force_evaluations<<'\n';
    if(c.fd) {
        auto direction_coeff=coefficients(11);double norm=std::sqrt(inner(direction_coeff,direction_coeff));
        for(auto & x:direction_coeff)x/=norm;const auto direction=expand(model,direction_coeff);
        f=output(c.prefix+".fd.csv");if(FML::ThisTask==0)f<<"probe,h,analytic,numeric\n";
        for(int probe=0;probe<2;++probe) {
            const double da=probe?.2:0.,dn=probe?-.1:0.,analytic=model.sum(inner(gradient,direction))+da*gp[0]+dn*gp[1];
            for(double h:{.003,.001,.0003,.0001,.00003,.00001,.000003,.000001}) {
                auto plus=white,minus=white;auto pp=p,pm=p;
                for(size_t q=0;q<white.size();++q){plus[q]+=h*direction[q];minus[q]-=h*direction[q];}
                pp[0]+=da*h;pm[0]-=da*h;pp[1]+=dn*h;pm[1]-=dn*h;
                const double numeric=(value(model,posterior,plus,pp,work)-value(model,posterior,minus,pm,work))/(2*h);
                if(FML::ThisTask==0)f<<probe<<','<<h<<','<<analytic<<','<<numeric<<'\n';
            }
        }
    }
    if(c.fit) {
        auto initial=coefficients(300);for(auto & x:initial)x*=.05;
        const auto start=expand(model,initial);const double initial_u=value(model,posterior,start,p,work);
        f=output(c.prefix+".fit_start_modes.csv");row(f,global_modes(model,model.predict(start,p,work)));
        model.forwards=model.pullbacks=0;
        const auto begin=std::chrono::steady_clock::now();
        FF::MAPOptimizationResult optimization;
        const auto fit=[&] {
            if(c.optimizer==0)return FF::map(posterior,start,p,false,c.fit,2e-5);
            FF::MAPOptions options;options.algorithm=c.optimizer==1?FF::MAPAlgorithm::GradientDescent:FF::MAPAlgorithm::LBFGS;
            options.vary_parameters=false;options.max_iterations=c.fit;options.tolerance=2e-5;options.history=c.history;options.stable_objective_changes=c.stable;
            optimization=FF::map(posterior,start,p,options);return optimization.fit;
        }();
        const double seconds=model.maximum(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
        const auto forwards=model.forwards,pullbacks=model.pullbacks;
        if(c.optimizer==0)optimization.gradient_evaluations=pullbacks;
        struct rusage usage{};const bool rss_ok=getrusage(RUSAGE_SELF,&usage)==0;
#ifdef __APPLE__
        const double rss=rss_ok ? double(usage.ru_maxrss) : -1.;
#else
        const double rss=rss_ok ? double(usage.ru_maxrss)*1024 : -1.;
#endif
        const double total_rss=model.sum(rss),history_bytes=model.sum(double(optimization.peak_history_bytes));
        f=output(c.prefix+".fit_optimizer.csv");
        if(FML::ThisTask==0)f<<"optimizer,history,seconds,forwards,pullbacks,value_evaluations,gradient_evaluations,peak_history_pairs,total_history_bytes,sum_rank_peak_rss_bytes,curvature_skips,history_resets\n"
            <<c.optimizer<<','<<c.history<<','<<seconds<<','<<forwards<<','<<pullbacks<<','<<optimization.value_evaluations<<','<<optimization.gradient_evaluations<<','
            <<optimization.peak_history_pairs<<','<<history_bytes<<','<<total_rss<<','<<optimization.curvature_skips<<','<<optimization.history_resets<<'\n';
        if(c.optimizer) {
            f=output(c.prefix+".fit_trace.csv");
            if(FML::ThisTask==0){f<<"iteration,potential,gradient_norm,field_gradient_l2,step,value_evaluations,gradient_evaluations\n";
                for(const auto & x:optimization.trace)f<<x.iteration<<','<<x.potential<<','<<x.gradient_norm<<','<<x.field_gradient_l2<<','<<x.step<<','<<x.value_evaluations<<','<<x.gradient_evaluations<<'\n';}
        }
        if(c.stable) {
            f=output(c.prefix+".fit_change.csv");if(FML::ThisTask==0)f<<"reference_evaluations\n"<<optimization.reference_evaluations<<'\n';
            f=output(c.prefix+".fit_change_trace.csv");if(FML::ThisTask==0){f<<"iteration,accepted_change,armijo_bound,reference_evaluations\n";
                for(const auto & x:optimization.trace)f<<x.iteration<<','<<x.accepted_change<<','<<x.armijo_bound<<','<<x.reference_evaluations<<'\n';}
        }
        Field g;std::vector<double> scalar;posterior.evaluate(fit.white,p,work,g,scalar);
        const auto fitted=project(model,fit.white),pg=project(model,g);
        const auto final_modes=global_modes(model,model.predict(fit.white,p,work));
        f=output(c.prefix+".fit_modes.csv");row(f,final_modes);
        f=output(c.prefix+".fit_coefficients.csv");row(f,truth);row(f,initial);row(f,fitted);row(f,pg);
        f=output(c.prefix+".fit.csv");
        if(FML::ThisTask==0)f<<"initial_potential,potential,gradient_norm,iterations,status,projected_gradient_norm\n"
            <<initial_u<<','<<fit.potential<<','<<fit.gradient_norm<<','<<fit.iterations<<','<<int(fit.status)<<','<<std::sqrt(inner(pg,pg))<<'\n';
        if(FML::ThisTask==0)std::cout<<"MAP status="<<int(fit.status)<<" iterations="<<fit.iterations<<" gradient="<<fit.gradient_norm<<std::endl;
    }
}
int main(int argc,char ** argv) {try {
    Controls c;
    FML::COMMUNICATION::collective_try([&] {
        if(argc<11 || argc>18)throw std::invalid_argument("pm_nonlinear PREFIX N M STEPS SEED POWER FINAL_A DATA_PATH FD FIT_ITERATIONS [FORWARD_ONLY] [FORCE_WINDOW] [CIC_REFERENCE] [OPTIMIZER] [HISTORY] [STABLE_CHANGE] [ASSIGNMENT]");
        auto integer=[](const char * text){size_t end;int x=std::stoi(text,&end);if(end!=std::string(text).size())throw std::invalid_argument("Invalid integer");return x;};
        auto real=[](const char * text){size_t end;double x=std::stod(text,&end);if(end!=std::string(text).size()||!std::isfinite(x))throw std::invalid_argument("Invalid real");return x;};
        c.prefix=argv[1];c.n=integer(argv[2]);c.m=integer(argv[3]);c.steps=integer(argv[4]);c.seed=integer(argv[5]);c.power=real(argv[6]);c.final=real(argv[7]);c.data=argv[8];c.fd=integer(argv[9]);c.fit=integer(argv[10]);if(argc>=12)c.forward=integer(argv[11]);if(argc>=13)c.window=integer(argv[12]);if(argc>=14)c.reference=integer(argv[13]);if(argc>=15)c.optimizer=integer(argv[14]);if(argc>=16)c.history=integer(argv[15]);if(argc>=17)c.stable=integer(argv[16]);if(argc==18)c.assignment=integer(argv[17]);
    },"PM nonlinear controls");
    require(c.n>=16&&c.n<=(c.forward==1?128:64)&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64&&c.seed>=0&&c.seed<=1000000&&c.power>0&&c.power<=.01&&c.final>.2&&c.final<=1&&
            (c.fd==0||c.fd==1)&&c.fit>=0&&c.fit<=200&&(c.forward==0||c.forward==1)&&(c.window>=0&&c.window<=2)&&(c.assignment==2||c.assignment==3)&&(!c.window||c.window==c.assignment-1)&&c.reference>=0&&c.reference<=64&&c.optimizer>=0&&c.optimizer<=2&&c.history>=1&&c.history<=32&&(c.stable==0||c.stable==1)&&(!c.stable||c.optimizer>0)&&
            (!c.forward||(c.fd==0&&c.fit==0)),"Invalid bounded PM controls");
    std::ostringstream controls;controls<<std::setprecision(17)<<c.n<<','<<c.m<<','<<c.steps<<','<<c.seed<<','<<c.power<<','<<c.final<<','<<c.fd<<','<<c.fit<<','<<c.forward<<','<<c.window<<','<<c.reference<<','<<c.optimizer<<','<<c.history<<','<<c.stable<<','<<c.assignment;
    FF::CHECKPOINT::same_string(controls.str());FF::CHECKPOINT::same_string(c.prefix);FF::CHECKPOINT::same_string(c.data);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto model=make_model(c);auto window_file=output(c.prefix+".window.csv");if(FML::ThisTask==0)window_file<<"force_window\n"<<c.window<<'\n';auto observation_file=output(c.prefix+".observation.csv");if(FML::ThisTask==0)observation_file<<"reference_cic_mesh\n"<<c.reference<<'\n';if(c.assignment!=2){auto assignment_file=output(c.prefix+".assignment.csv");if(FML::ThisTask==0)assignment_file<<"assignment\n"<<c.assignment<<'\n';}if(c.forward)forward_only(c,model);else run(c,model);return 0;
} catch(const std::exception & e) {if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1;}}
