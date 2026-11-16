#include <FML/FieldLevel/pm_nonlinear_validation/Validation.h>
// Observes inputs/returned likelihood cotangents; the optimizer is unchanged.
struct CapturedModel : Model {
    using Model::Model;
    bool stable=false;
    mutable bool active=false,first=false,skip_reference=false;
    mutable Field last,point,gradient,trial;
    const FF::Modes & predict(const Field & s,const std::vector<double> & p,Workspace & w)const {
        if(active)FML::COMMUNICATION::collective_try([&]{if(first){if(skip_reference)skip_reference=false;else{trial=s;first=false;}}last=s;},"Capture PM input");
        return Model::predict(s,p,w);
    }
    void pullback(const std::vector<double> & p,const FF::Modes & g,Workspace & w,Field & gs,std::vector<double> & gp)const {
        Model::pullback(p,g,w,gs,gp);
        if(active)FML::COMMUNICATION::collective_try([&]{point=last;gradient=gs;for(size_t i=0;i<point.size();++i)gradient[i]+=point[i];first=true;skip_reference=stable;trial.clear();},"Capture PM accepted gradient");
    }
};
namespace FML::FIELDLEVEL::FRAMEWORK {template<>struct Execution<::CapturedModel>:Execution<::BaseModel>{};}
static int integer(const char * s){size_t end;int x=std::stoi(s,&end);if(end!=std::string(s).size())throw std::invalid_argument("Invalid integer");return x;}
static std::string state_identity(const Controls & c){return "PM-stationarity-v1,"+std::to_string(c.n)+","+std::to_string(c.m)+","+std::to_string(c.steps)+",compensated,Omega_m=.3,"+identity(c);}
int main(int argc,char **argv){try{
    Controls c;c.seed=91;c.power=.002;c.final=1.;c.window=1;c.reference=16;
    FML::COMMUNICATION::collective_try([&]{if(argc!=7&&argc!=8)throw std::invalid_argument("pm_stationarity_capture PREFIX DATA N M STEPS ITERATIONS [STABLE_CHANGE]");c.prefix=argv[1];c.data=argv[2];c.n=integer(argv[3]);c.m=integer(argv[4]);c.steps=integer(argv[5]);c.fit=integer(argv[6]);if(argc==8)c.stable=integer(argv[7]);},"Capture arguments");
    require(c.n>=16&&c.n<=64&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64&&c.fit>=0&&c.fit<=128&&(c.stable==0||c.stable==1),"Invalid capture controls");
    FF::CHECKPOINT::same_string(state_identity(c)+","+std::to_string(c.fit)+","+std::to_string(c.stable));FF::CHECKPOINT::same_string(c.prefix);FF::CHECKPOINT::same_string(c.data);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto model=make_model<CapturedModel>(c);model.stable=c.stable;auto p=model.initial_parameters();
    FF::BasicPosterior<CapturedModel> posterior(model,load_data(model,c));auto co=coefficients(300);for(auto & x:co)x*=.05;const auto start=expand(model,co);
    FF::MAPOptions options;options.max_iterations=c.fit;options.tolerance=2e-5;options.vary_parameters=false;options.history=5;options.stable_objective_changes=c.stable;
    model.active=true;const auto begin=std::chrono::steady_clock::now();const auto fit=FF::map(posterior,start,p,options);model.active=false;
    const double seconds=model.maximum(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());
    bool same=model.point==fit.fit.white;require(same,"Capture did not retain last accepted state");
    const bool failed=fit.fit.status==FF::MAPStatus::LineSearchFailed && !model.trial.empty();
    Field direction(model.cells()),unit_trial(model.cells());
    for(size_t i=0;i<direction.size();++i){direction[i]=failed?model.point[i]-model.trial[i]:model.gradient[i];unit_trial[i]=model.point[i]-direction[i];}
    auto work=model.workspace();Field fresh;std::vector<double> gp;const double u=posterior.evaluate(fit.fit.white,p,work,fresh,gp);double error=0;
    for(size_t i=0;i<fresh.size();++i)error=std::max(error,std::abs(fresh[i]-model.gradient[i]));
    require(model.maximum(error)==0. && u==fit.fit.potential,"Captured gradient/state fail fresh replay");
    FML::COMMUNICATION::collective_try([&]{std::ofstream f(c.prefix+".rank"+std::to_string(FML::ThisTask)+".csv");if(!f)throw std::runtime_error("Cannot write snapshot shard");f<<std::setprecision(17)<<"id,white,gradient,direction,unit_trial\n";
        for(size_t i=0;i<direction.size();++i)f<<model.layout().global_cell_id(i)<<','<<fit.fit.white[i]<<','<<fresh[i]<<','<<direction[i]<<','<<unit_trial[i]<<'\n';if(!f)throw std::runtime_error("Snapshot write failed");},"PM capture shards");
    auto f=output(c.prefix+".meta.csv");if(FML::ThisTask==0)f<<state_identity(c)<<'\n'<<"shards,status,iterations,potential,raw_gradient,direction_source,seconds\n"<<FML::NTasks<<','<<int(fit.fit.status)<<','<<fit.fit.iterations<<','<<u<<','<<fit.fit.gradient_norm<<','<<int(failed)<<','<<seconds<<'\n';
    f=output(c.prefix+".capture.csv");if(FML::ThisTask==0)f<<"stable_changes,reference_evaluations\n"<<c.stable<<','<<fit.reference_evaluations<<'\n';
    f=output(c.prefix+".parameters.csv");row(f,p);f=output(c.prefix+".projected.csv");row(f,project(model,fit.fit.white));row(f,project(model,fresh));
    f=output(c.prefix+".trace.csv");if(FML::ThisTask==0){f<<"iteration,potential,raw_gradient,step,value_evaluations,gradient_evaluations\n";for(const auto & x:fit.trace)f<<x.iteration<<','<<x.potential<<','<<x.gradient_norm<<','<<x.step<<','<<x.value_evaluations<<','<<x.gradient_evaluations<<'\n';}
    if(FML::ThisTask==0)std::cout<<"Captured status="<<int(fit.fit.status)<<" iterations="<<fit.fit.iterations<<" gradient="<<fit.fit.gradient_norm<<" seconds="<<seconds<<std::endl;
    return 0;
}catch(const std::exception &e){if(FML::ThisTask==0)std::cerr<<e.what()<<std::endl;return 1;}}
