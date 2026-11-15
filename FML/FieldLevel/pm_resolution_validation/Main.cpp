#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/Frontend.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace FF=FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using Model=FF::PeriodicPMModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
struct Wave {int x,y,z;};
static std::vector<Wave> waves() {
    std::vector<Wave> out;
    for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z) {
        const int q=x*x+y*y+z*z;
        if(q>0 && q<=2 && (x>0 || (x==0 && y>0) || (x==0 && y==0 && z>0)))out.push_back({x,y,z});
    }
    return out;
}
static void require(bool condition,const char * text) {int bad=!condition;FML::MaxOverTasks(&bad);if(bad)throw std::invalid_argument(text);}
static std::ofstream output(const std::string & file) {
    std::ofstream out;FML::COMMUNICATION::collective_try([&]{if(FML::ThisTask==0){out.open(file);if(!out)throw std::runtime_error("Cannot open "+file);out<<std::setprecision(17);}},"PM resolution output");return out;
}
static void row(std::ofstream & file,const Field & values) {if(FML::ThisTask==0){for(size_t j=0;j<values.size();++j){if(j)file<<',';file<<values[j];}file<<'\n';}}
template<class Layout> static Field canonical(const Field & local,const Layout & layout) {
    require(local.size()==layout.owned_cells,"Canonical field size");Field global(layout.global_cells,0.);
    for(size_t q=0;q<local.size();++q)global[layout.global_cell_id(q)]=local[q];FML::SumArrayOverTasks(global.data(),global.size());return global;
}
static Model make_model(int n,int m,int steps) {
    FF::Configuration c;c.mesh=n;c.density_mesh=m;c.boxsize=1.;c.evolution=FF::Evolution::Linear;
    c.initial_cutoff=c.bias_cutoff=10.;c.likelihood_cutoff=15.;
    FF::Spectrum s{{{"log_amplitude",0.,0.,.5,-1.,1.}},[](double,const auto & p){return FF::Power{.002*std::exp(p[0]),{1.}};}};
    FF::PMSettings pm;pm.scale_factors.clear();for(int j=0;j<=steps;++j)pm.scale_factors.push_back(.2+.4*j/steps);
    const double noise=std::log(.0001);return Model(c,s,pm,{"log_noise_power",noise,noise,.5,-16.,-4.},{8});
}
static std::vector<Field> basis(const Model & model) {
    std::vector<Field> b(18,Field(model.cells()));size_t j=0;
    for(const auto & w:waves()) {
        for(size_t q=0;q<model.cells();++q) {
            const auto c=model.layout().global_coordinate(q);const double phase=2*M_PI*(w.x*c[0]+w.y*c[1]+w.z*c[2])/model.configuration().mesh;
            const double norm=std::sqrt(2./model.layout().global_cells);b[j][q]=norm*std::cos(phase);b[j+1][q]=norm*std::sin(phase);
        }
        j+=2;
    }
    return b;
}
static double inner(const Field & a,const Field & b){double sum=0;for(size_t q=0;q<a.size();++q)sum+=a[q]*b[q];return sum;}
static Field matched(const Model & model,const std::vector<Field> & b,bool with_null=false) {
    Field white(model.cells(),0.);
    if(with_null){white=model.white_noise(193);for(const auto & v:b){const double z=model.sum(inner(white,v));for(size_t q=0;q<white.size();++q)white[q]-=z*v[q];}}
    size_t j=0;for(const auto & w:waves()) {
        const std::uint64_t id=std::uint64_t((w.x+2)*25+(w.y+2)*5+w.z+2);
        FML::RANDOM::COUNTER_BASED::Stream rng(91,id,0x504d4346u);
        for(int part=0;part<2;++part,++j){const double z=rng.normal();for(size_t q=0;q<white.size();++q)white[q]+=z*b[j][q];}
    }
    return white;
}
static Field global_modes(const Model & model,const FF::Modes & local) {
    Field global(8*8*5*2,0.);
    for(size_t j=0;j<local.size();++j){const auto & n=model.modes()[j].n;const size_t i=(((n[0]+8)%8)*8+(n[1]+8)%8)*5+n[2];global[2*i]=local[j].real();global[2*i+1]=local[j].imag();}
    FML::SumArrayOverTasks(global.data(),global.size());return global;
}
static FF::Modes load_data(const Model & model,const std::string & path) {
    Field data;
    FML::COMMUNICATION::collective_try([&]{std::ifstream file(path);if(!file)throw std::runtime_error("Cannot read common data");std::string line,cell;std::getline(file,line);std::stringstream input(line);
        while(std::getline(input,cell,',')){size_t n;double x=std::stod(cell,&n);if(n!=cell.size()||!std::isfinite(x))throw std::invalid_argument("Invalid common data");data.push_back(x);}
        if(data.size()!=8*8*5*2)throw std::invalid_argument("Wrong common data size");},"Common PM data");
    FF::Modes local(model.modes().size());for(size_t j=0;j<local.size();++j){const auto & n=model.modes()[j].n;const size_t i=(((n[0]+8)%8)*8+(n[1]+8)%8)*5+n[2];local[j]={data[2*i],data[2*i+1]};}return local;
}
static double value(const Model & model,const FF::BasicPosterior<Model> & posterior,const Field & white,const std::vector<double> & p,Model::Workspace & work) {
    const auto & prediction=model.predict(white,p,work);FF::Modes g;double gn;
    double u=posterior.likelihood().evaluate(prediction,p.back(),g,gn);u=model.sum(u+.5*inner(white,white));
    for(size_t j=0;j<p.size();++j)u+=model.parameters()[j].penalty(p[j]);return u;
}
static void simulate(Model & model,const std::string & prefix,const std::string & path,bool finite_difference) {
    const auto b=basis(model);const auto white=matched(model,b);auto p=model.initial_parameters();auto work=model.workspace();
    const auto data=path=="-"?model.mock(white,p,92,work):load_data(model,path);FF::BasicPosterior<Model> posterior(model,data);
    Field g;std::vector<double> gp;const double u=posterior.evaluate(white,p,work,g,gp);const auto report=work.adjoint_report();
    const auto modes=model.predict(white,p,work);const auto & fields=model.reconstruction(work);
    auto data_file=output(prefix+".data.csv"),prediction=output(prefix+".modes.csv"),initial=output(prefix+".initial.csv"),gradient=output(prefix+".gradient.csv"),meta=output(prefix+".meta.csv");
    row(data_file,global_modes(model,data));row(prediction,global_modes(model,modes));Field physical,projected;
    for(const auto & v:b){physical.push_back(model.sum(inner(fields.initial_density,v))/std::sqrt(double(model.layout().global_cells)));projected.push_back(model.sum(inner(g,v)));}
    row(initial,physical);projected.insert(projected.end(),gp.begin(),gp.end());row(gradient,projected);
    if(FML::ThisTask==0){meta<<"n,m,steps,potential,force_evaluations,checkpoint_payload_bytes\n"<<model.configuration().mesh<<','<<model.configuration().density_mesh<<','<<model.dynamics().settings().scale_factors.size()-1<<','<<u<<','<<report.force_evaluations<<','<<report.checkpoint_payload_bytes<<'\n';}
    if(!finite_difference)return;
    auto fd=output(prefix+".fd.csv");if(FML::ThisTask==0)fd<<"probe,h,analytic,finite_difference\n";
    for(int probe=0;probe<2;++probe){Field direction(white.size(),0.);double norm=0;for(size_t j=0;j<b.size();++j)norm+=std::pow(std::sin(.7*(j+1)),2);norm=std::sqrt(norm);
        for(size_t j=0;j<b.size();++j)for(size_t q=0;q<white.size();++q)direction[q]+=std::sin(.7*(j+1))/norm*b[j][q];
        const double da=probe?.2:0.,dn=probe?-.1:0.;const double analytic=model.sum(inner(g,direction))+da*gp[0]+dn*gp[1];
        for(double h:{.003,.001,.0003,.0001,.00003,.00001,.000003,.000001}){auto plus=white,minus=white;auto pp=p,pm=p;for(size_t q=0;q<white.size();++q){plus[q]+=h*direction[q];minus[q]-=h*direction[q];}pp[0]+=h*da;pm[0]-=h*da;pp[1]+=h*dn;pm[1]-=h*dn;
            const double estimate=(value(model,posterior,plus,pp,work)-value(model,posterior,minus,pm,work))/(2*h);
            if(FML::ThisTask==0)fd<<probe<<','<<h<<','<<analytic<<','<<estimate<<'\n';}
    }
}
static void chain(Model & model,const std::string & prefix,int id,int warm,int draws,int leap,int stride) {
    const auto b=basis(model);const auto truth=matched(model,b,true);auto p=model.initial_parameters();auto work=model.workspace();
    const auto data=model.mock(truth,p,92,work);const auto & fields=model.reconstruction(work);
    auto truth_file=output(prefix+".truth.csv");row(truth_file,canonical(truth,model.layout()));row(truth_file,canonical(fields.initial_density,model.layout()));row(truth_file,canonical(fields.matter_density,model.observation_layout()));
    auto data_file=output(prefix+".data.csv");row(data_file,global_modes(model,data));
    auto start=model.white_noise(300+id);const double scales[4]={.5,1.,1.5,2.},offsets[4]={-.6,.6,-.3,.3};for(auto & x:start)x*=scales[id%4];p[0]=offsets[id%4];p[1]+=offsets[(id+1)%4];
    FF::BasicPosterior<Model> posterior(model,data);auto state=output(prefix+".states.csv"),predictive=output(prefix+".predictive.csv");
    if(FML::ThisTask==0){state<<"draw";for(size_t q=0;q<model.layout().global_cells;++q)state<<",w"<<q;state<<",log_amplitude,log_noise\n";predictive<<"draw";for(size_t q=0;q<model.layout().global_cells;++q)predictive<<",initial"<<q;for(size_t q=0;q<model.observation_layout().global_cells;++q)predictive<<",matter"<<q;predictive<<'\n';}
    FML::MCMC::HMCOptions o;o.warmup=warm;o.samples=draws;o.n_leapfrog=leap;o.step_size=.08;o.adapt_step_size=true;o.adapt_mass=false;o.target_acceptance=.9;o.seed=11000+id;o.divergence_threshold=10.;
    auto prediction_work=model.workspace();size_t count=0;const auto result=FF::sample(posterior,start,p,o,true,[&](const Field & white,const auto & parameters,size_t draw){
        const auto global=canonical(white,model.layout());if(FML::ThisTask==0){state<<draw;for(double x:global)state<<','<<x;for(double x:parameters)state<<','<<x;state<<'\n';}
        if(draw%stride==0){model.predict(white,parameters,prediction_work);const auto & r=model.reconstruction(prediction_work);const auto a=canonical(r.initial_density,model.layout()),d=canonical(r.matter_density,model.observation_layout());if(FML::ThisTask==0){predictive<<draw;for(double x:a)predictive<<','<<x;for(double x:d)predictive<<','<<x;predictive<<'\n';}}
        if(++count%128==0 && FML::ThisTask==0)std::cout<<"RETAINED "<<count<<'/'<<draws<<std::endl;
    });
    require(count==size_t(draws)&&result.potential.size()==size_t(draws),"Missing retained draws");auto potential=output(prefix+".potential.csv"),meta=output(prefix+".meta.csv");
    if(FML::ThisTask==0){potential<<"draw,potential\n";for(size_t j=0;j<result.potential.size();++j)potential<<j<<','<<result.potential[j]<<'\n';meta<<"n,m,steps,id,warmup,draws,leapfrog,stride,epsilon,acceptance,divergences\n"<<model.configuration().mesh<<','<<model.configuration().density_mesh<<','<<model.dynamics().settings().scale_factors.size()-1<<','<<id<<','<<warm<<','<<draws<<','<<leap<<','<<stride<<','<<result.step_size<<','<<result.acceptance<<','<<result.divergences<<'\n';}
}
int main(int argc,char ** argv){try{
    std::string mode,prefix,path;int n=0,m=0,steps=0,id=0,warm=0,draws=0,leap=0,stride=0,fd=0;
    FML::COMMUNICATION::collective_try([&]{if(argc<6)throw std::invalid_argument("pm_resolution case PREFIX N M STEPS DATA_PATH FD_FLAG | chain PREFIX N M STEPS ID WARM DRAWS LEAP STRIDE");
        auto integer=[](const char * s){size_t k;int x=std::stoi(s,&k);if(k!=std::string(s).size())throw std::invalid_argument("Invalid integer");return x;};mode=argv[1];prefix=argv[2];n=integer(argv[3]);m=integer(argv[4]);steps=integer(argv[5]);
        if(mode=="case" && argc==8){path=argv[6];fd=integer(argv[7]);}else if(mode=="chain" && argc==11){id=integer(argv[6]);warm=integer(argv[7]);draws=integer(argv[8]);leap=integer(argv[9]);stride=integer(argv[10]);}else throw std::invalid_argument("Invalid argument count/mode");
    },"PM resolution input");
    require(n>=4&&n<=32&&n%2==0&&m>=n&&m<=64&&m%2==0&&steps>=1&&steps<=64&&(fd==0||fd==1)&&id>=0&&id<16&&warm>=0&&warm<=4096&&draws>=0&&draws<=8192&&
        (mode!="chain"||(draws>0&&leap>=2&&leap<=64&&stride>0&&stride<=draws)),"Invalid bounded resolution controls");
    FF::CHECKPOINT::same_string(mode+":"+std::to_string(n)+":"+std::to_string(m)+":"+std::to_string(steps)+":"+std::to_string(id)+":"+std::to_string(warm)+":"+std::to_string(draws)+":"+std::to_string(leap)+":"+std::to_string(stride)+":"+std::to_string(fd));
    FF::CHECKPOINT::same_string(path);
    FML::FIELDLEVEL::FRONTEND::initialize(n,m);auto model=make_model(n,m,steps);model.controls({double(n),double(m),double(steps),double(id),double(warm),double(draws),double(leap),double(stride),double(fd),double(mode=="chain")});
    if(mode=="case")simulate(model,prefix,path,fd);else chain(model,prefix,id,warm,draws,leap,stride);return 0;
}catch(const std::exception & e){if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1;}}
