#include <FML/FieldLevel/pm_tsc_posterior_validation/Target.h>
static int integer(const char * s){size_t end;int x=std::stoi(s,&end);if(end!=std::string(s).size())throw std::invalid_argument("Invalid integer");return x;}
int main(int argc,char **argv){try{
    Controls c;c.seed=91;c.power=.002;c.final=1.;c.assignment=3;c.window=2;c.reference=16;std::string mode,map_file;int id,warm,draws,leap,adapt,stride;double eps;
    FML::COMMUNICATION::collective_try([&]{if(argc!=15)throw std::invalid_argument("pm_tsc_posterior check|chain|full PREFIX DATA PHYSICAL_MAP N M STEPS ID WARM DRAWS LEAP EPS ADAPT STRIDE");mode=argv[1];c.prefix=argv[2];c.data=argv[3];map_file=argv[4];c.n=integer(argv[5]);c.m=integer(argv[6]);c.steps=integer(argv[7]);id=integer(argv[8]);warm=integer(argv[9]);draws=integer(argv[10]);leap=integer(argv[11]);size_t end;eps=std::stod(argv[12],&end);if(end!=std::string(argv[12]).size())throw std::invalid_argument("Invalid epsilon");adapt=integer(argv[13]);stride=integer(argv[14]);},"TSC posterior controls");
    require((mode=="check"||mode=="chain"||mode=="full")&&c.n>=16&&c.n<=64&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64&&id>=0&&id<4&&warm>=0&&warm<=1024&&draws>=2&&draws<=4096&&leap>=1&&leap<=32&&std::isfinite(eps)&&eps>0&&(adapt==0||adapt==1)&&stride>=1&&stride<=draws,"Invalid bounded TSC posterior controls");
    for(const auto & s:{mode,c.prefix,c.data,map_file})FF::CHECKPOINT::same_string(s);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto model=make_joint(c);model.controls({double(c.n),double(c.m),double(c.steps),double(id),double(warm),double(draws),double(leap),eps,double(adapt),double(stride)});
    const auto data=load_data(model,c);CenteredTarget target(model,data);auto z=read_physical(map_file);const size_t d=z.size();
    const double offsets[4]={-.6,.6,-.3,.3},noise_offsets[4]={-.4,.4,.2,-.2};
    for(size_t j=0;j<d;++j){FML::RANDOM::COUNTER_BASED::Stream rng(5100+id,j,0x54534350u);z[j]+=.1*(id+1)*rng.normal();}
    z.push_back(offsets[id]);z.push_back(std::log(noise_power)+noise_offsets[id]);model.controls(z);
    if(mode=="check"){
        auto f=output(c.prefix+".checks.csv");if(FML::ThisTask==0)f<<"probe,value_error,gradient_error,basis_error,roundtrip_error,fd_error\n";
        const auto direct=expand(model,Field(z.begin(),z.begin()+d));const auto fast=target.basis.expand_active(z);const auto roundtrip=target.basis.project_active(fast);double basis_error=0,roundtrip_error=0;
        for(size_t q=0;q<fast.size();++q)basis_error=std::max(basis_error,std::abs(fast[q]-direct[q]));basis_error=model.maximum(basis_error);
        for(size_t j=0;j<d;++j)roundtrip_error=std::max(roundtrip_error,std::abs(roundtrip[j]-z[j]));
        require(basis_error<2e-12&&roundtrip_error<2e-12,"Active FFT basis mismatch");
        for(int k=0;k<3;++k){auto x=z;x[d]=-.4+.4*k;x[d+1]=std::log(noise_power)+.2*(1-k);const auto error=target.direct_error(x);Field g;target.evaluate(x,g);Field direction(x.size());double norm=0;
            for(size_t j=0;j<x.size();++j){direction[j]=std::sin(.37*(j+1));norm+=direction[j]*direction[j];}for(auto & v:direction)v/=std::sqrt(norm);const double analytic=inner(g,direction);double best=HUGE_VAL;
            for(double h:{.001,.0001,.00001,.000001}){auto plus=x,minus=x;for(size_t j=0;j<x.size();++j){plus[j]+=h*direction[j];minus[j]-=h*direction[j];}Field unused;const double up=target.evaluate(plus,unused),down=target.evaluate(minus,unused);best=std::min(best,std::abs((up-down)/(2*h)-analytic)/std::max(1.,std::abs(analytic)));}
            if(FML::ThisTask==0)f<<k<<','<<error.first<<','<<error.second<<','<<basis_error<<','<<roundtrip_error<<','<<best<<'\n';
            require(std::abs(error.first)<2e-9&&error.second<2e-9&&best<2e-5,"Centered density/gradient/Jacobian check failed");
        }return 0;
    }
    FML::MCMC::HMCOptions o;o.warmup=warm;o.samples=draws;o.n_leapfrog=leap;o.step_size=eps;o.adapt_step_size=adapt;o.adapt_mass=false;o.target_acceptance=.85;o.divergence_threshold=10.;o.seed=12000+id;o.verbose_every=32;
    const auto begin=std::chrono::steady_clock::now();
    if(mode=="full"){
        auto white=target.basis.expand_active(z);for(auto & x:white)x*=std::exp(-z[d]/2);FF::BasicPosterior<JointModel> posterior(model,data);auto f=output(c.prefix+".full.csv");if(FML::ThisTask==0)f<<"draw,A,N,norm\n";
        const auto result=FF::sample(posterior,white,{z[d],z[d+1]},o,true,[&](const Field & s,const auto & p,size_t draw){const double norm=model.sum(inner(s,s));if(FML::ThisTask==0)f<<draw<<','<<p[0]<<','<<p[1]<<','<<norm<<'\n';});
        auto last=Field(result.last.begin(),result.last.begin()+model.cells());auto projected=target.basis.project_active(last);projected.insert(projected.end(),result.last.begin()+model.cells(),result.last.end());auto file=output(c.prefix+".snapshot.csv");row(file,projected);return 0;
    }
    FML::MCMC::HMCMass mass;mass.diagonal.assign(z.size(),21.);mass.diagonal[d]=65.;mass.diagonal[d+1]=132.;o.lower.assign(z.size(),-HUGE_VAL);o.upper.assign(z.size(),HUGE_VAL);o.lower[d]=-4.;o.upper[d]=4.;o.lower[d+1]=-16.;o.upper[d+1]=-4.;
    FML::MCMC::HMCExecution execution;execution.all_valid=[&](bool ok){return model.maximum(ok?0:1)==0;};
    const auto result=FML::MCMC::run_hmc([&](const auto & x,auto & g){return target.evaluate(x,g);},z,mass,o,nullptr,execution);
    const double seconds=model.maximum(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());require(result.recorded.size()==size_t(draws),"Lost posterior draws");
    auto states=output(c.prefix+".states.csv"),features=output(c.prefix+".features.csv"),prediction=output(c.prefix+".predictive.csv"),meta=output(c.prefix+".meta.csv");
    if(FML::ThisTask==0){states<<"draw";features<<"draw";for(size_t j=0;j<d;++j){states<<",q"<<j;features<<",q"<<j;}states<<",A,N,potential\n";for(size_t j=0;j<d;++j)features<<",q2_"<<j;features<<",A,N,potential,norm2\n";prediction<<"draw,A,N";for(size_t j=0;j<4608;++j)prediction<<",m"<<j;prediction<<'\n';}
    auto work=model.workspace();auto pref=model.initial_parameters();
    for(size_t i=0;i<result.recorded.size();++i){const auto & x=result.recorded[i];double norm=0;if(FML::ThisTask==0){states<<i;features<<i;for(size_t j=0;j<d;++j){states<<','<<x[j];features<<','<<x[j];norm+=x[j]*x[j];}states<<','<<x[d]<<','<<x[d+1]<<','<<result.potential[i]<<'\n';for(size_t j=0;j<d;++j)features<<','<<x[j]*x[j];features<<','<<x[d]<<','<<x[d+1]<<','<<result.potential[i]<<','<<norm<<'\n';}
        if(i%stride==0){const auto field=target.basis.expand_active(x);const auto modes=global_modes(model,model.predict(field,pref,work));if(FML::ThisTask==0){prediction<<i<<','<<x[d]<<','<<x[d+1];for(double v:modes)prediction<<','<<v;prediction<<'\n';}}
    }
    if(FML::ThisTask==0)meta<<"id,warmup,draws,leapfrog,adapt,epsilon,acceptance,divergences,seed,seconds,evaluations,n,m,steps,active\n"<<id<<','<<warm<<','<<draws<<','<<leap<<','<<adapt<<','<<result.step_size<<','<<result.acceptance<<','<<result.divergences<<','<<o.seed<<','<<seconds<<','<<target.evaluations<<','<<c.n<<','<<c.m<<','<<c.steps<<','<<d<<'\n';
    return 0;
}catch(const std::exception & e){if(FML::ThisTask==0)std::cerr<<e.what()<<std::endl;return 1;}}
