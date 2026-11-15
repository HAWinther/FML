#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/Frontend.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using Model = FF::PeriodicPMModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
struct Wave { int x, y, z; };
struct Controls {
    int n{}, m{}, steps{}, seed{}, fd{}, fit{};
    double power{}, final{};
    std::string prefix, data;
};
static constexpr int common_mesh = 16;
static constexpr double noise_power = 1e-4;
static void require(bool ok, const char * text) {
    int bad = !ok; FML::MaxOverTasks(&bad);
    if (bad) throw std::invalid_argument(text);
}
static std::vector<Wave> waves() {
    std::vector<Wave> result;
    for (int x=-3; x<=3; ++x) for (int y=-3; y<=3; ++y) for (int z=-3; z<=3; ++z) {
        const int q=x*x+y*y+z*z;
        if (q>0 && q<=9 && (x>0 || (x==0 && y>0) || (x==0 && y==0 && z>0)))
            result.push_back({x,y,z});
    }
    return result;
}
static double inner(const Field & a, const Field & b) {
    double sum=0; for (size_t q=0; q<a.size(); ++q) sum+=a[q]*b[q]; return sum;
}
// Generate/project one Fourier pair at a time: no O(active_modes * N^3) basis storage.
static Field coefficients(int seed) {
    Field result;
    for (const auto & k:waves()) {
        const auto id=std::uint64_t((k.x+4)*81+(k.y+4)*9+k.z+4);
        FML::RANDOM::COUNTER_BASED::Stream rng(seed,id,0x504d4e4cu);
        result.push_back(rng.normal()); result.push_back(rng.normal());
    }
    return result;
}
static Field expand(const Model & model, const Field & coefficients) {
    Field result(model.cells(),0.); size_t j=0;
    const double norm=std::sqrt(2./model.layout().global_cells);
    for (const auto & k:waves()) {
        for (size_t q=0; q<result.size(); ++q) {
            const auto x=model.layout().global_coordinate(q);
            const double phase=2*M_PI*(k.x*x[0]+k.y*x[1]+k.z*x[2])/model.configuration().mesh;
            result[q]+=norm*(coefficients[j]*std::cos(phase)+coefficients[j+1]*std::sin(phase));
        }
        j+=2;
    }
    return result;
}
static Field project(const Model & model, const Field & field) {
    Field result; const double norm=std::sqrt(2./model.layout().global_cells);
    for (const auto & k:waves()) {
        double c=0,s=0;
        for (size_t q=0; q<field.size(); ++q) {
            const auto x=model.layout().global_coordinate(q);
            const double phase=2*M_PI*(k.x*x[0]+k.y*x[1]+k.z*x[2])/model.configuration().mesh;
            c+=norm*std::cos(phase)*field[q]; s+=norm*std::sin(phase)*field[q];
        }
        result.push_back(model.sum(c)); result.push_back(model.sum(s));
    }
    return result;
}
static Model make_model(const Controls & c) {
    FF::Configuration config; config.mesh=c.n; config.density_mesh=c.m; config.boxsize=1;
    config.evolution=FF::Evolution::Linear;
    config.initial_cutoff=config.bias_cutoff=2*M_PI*std::sqrt(9.5);
    config.likelihood_cutoff=2*M_PI*std::sqrt(16.5);
    FF::Spectrum spectrum{{{"log_amplitude",0.,0.,.5,-1.,1.}},
                         [power=c.power](double,const auto & p) { return FF::Power{power*std::exp(p[0]),{1.}}; }};
    FF::PMSettings pm; pm.scale_factors.clear();
    for (int j=0; j<=c.steps; ++j) pm.scale_factors.push_back(.2+(c.final-.2)*j/c.steps);
    const double noise=std::log(noise_power);
    return Model(config,spectrum,pm,{"log_noise_power",noise,noise,.5,-16.,-4.},{8});
}
static size_t mode_index(const int * k) {
    return (size_t((k[0]+common_mesh)%common_mesh)*common_mesh+(k[1]+common_mesh)%common_mesh)*(common_mesh/2+1)+k[2];
}
static Field global_modes(const Model & model, const FF::Modes & modes) {
    Field result(common_mesh*common_mesh*(common_mesh/2+1)*2,0.);
    for (size_t j=0; j<modes.size(); ++j) {
        const auto i=mode_index(model.modes()[j].n); result[2*i]=modes[j].real(); result[2*i+1]=modes[j].imag();
    }
    FML::SumArrayOverTasks(result.data(),result.size()); return result;
}
static std::ofstream output(const std::string & path) {
    std::ofstream file;
    FML::COMMUNICATION::collective_try([&] {
        if (FML::ThisTask==0) {file.open(path); if (!file) throw std::runtime_error("Cannot write "+path); file<<std::setprecision(17);}
    },"PM nonlinear output");
    return file;
}
static void row(std::ofstream & file,const Field & values) {
    if (FML::ThisTask==0) {for (size_t j=0;j<values.size();++j) {if(j)file<<',';file<<values[j];}file<<'\n';}
}
static std::string identity(const Controls & c) {
    std::ostringstream text; text<<std::setprecision(17)<<"PM-nonlinear-v1,"<<c.seed<<','<<c.power<<','<<c.final<<",9,16,"<<noise_power;
    return text.str();
}
static FF::Modes load_data(const Model & model,const Controls & c) {
    Field values;
    FML::COMMUNICATION::collective_try([&] {
        std::ifstream file(c.data); std::string header,line,cell;
        if (!file || !std::getline(file,header) || header!=identity(c) || !std::getline(file,line))
            throw std::invalid_argument("Wrong PM data identity/header");
        std::stringstream input(line);
        while(std::getline(input,cell,',')) {size_t end;const double x=std::stod(cell,&end);if(end!=cell.size()||!std::isfinite(x))throw std::invalid_argument("Invalid PM data");values.push_back(x);}
        if (values.size()!=common_mesh*common_mesh*(common_mesh/2+1)*2)throw std::invalid_argument("Wrong PM data size");
        while (std::getline(file,line)) if(!line.empty())throw std::invalid_argument("Trailing PM data");
    },"PM nonlinear common data");
    FF::Modes result(model.modes().size());
    for(size_t j=0;j<result.size();++j){const auto i=mode_index(model.modes()[j].n);result[j]={values[2*i],values[2*i+1]};}
    return result;
}
// Independent value-only evaluation for the finite-difference checks.
static double value(const Model & model,const FF::BasicPosterior<Model> & posterior,
                    const Field & white,const std::vector<double> & p,Model::Workspace & work) {
    const auto & predicted=model.predict(white,p,work);FF::Modes gradient;double noise;
    double u=model.sum(posterior.likelihood().evaluate(predicted,p.back(),gradient,noise)+.5*inner(white,white));
    for(size_t j=0;j<p.size();++j)u+=model.parameters()[j].penalty(p[j]);
    return u;
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
        const auto fit=FF::map(posterior,start,p,false,c.fit,2e-5);
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
        if(argc!=11)throw std::invalid_argument("pm_nonlinear PREFIX N M STEPS SEED POWER FINAL_A DATA_PATH FD FIT_ITERATIONS");
        auto integer=[](const char * text){size_t end;int x=std::stoi(text,&end);if(end!=std::string(text).size())throw std::invalid_argument("Invalid integer");return x;};
        auto real=[](const char * text){size_t end;double x=std::stod(text,&end);if(end!=std::string(text).size()||!std::isfinite(x))throw std::invalid_argument("Invalid real");return x;};
        c.prefix=argv[1];c.n=integer(argv[2]);c.m=integer(argv[3]);c.steps=integer(argv[4]);c.seed=integer(argv[5]);c.power=real(argv[6]);c.final=real(argv[7]);c.data=argv[8];c.fd=integer(argv[9]);c.fit=integer(argv[10]);
    },"PM nonlinear controls");
    require(c.n>=16&&c.n<=64&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64&&c.seed>=0&&c.seed<=1000000&&c.power>0&&c.power<=.01&&c.final>.2&&c.final<=1&&
            (c.fd==0||c.fd==1)&&c.fit>=0&&c.fit<=200,"Invalid bounded PM controls");
    std::ostringstream controls;controls<<std::setprecision(17)<<c.n<<','<<c.m<<','<<c.steps<<','<<c.seed<<','<<c.power<<','<<c.final<<','<<c.fd<<','<<c.fit;
    FF::CHECKPOINT::same_string(controls.str());FF::CHECKPOINT::same_string(c.prefix);FF::CHECKPOINT::same_string(c.data);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto model=make_model(c);run(c,model);return 0;
} catch(const std::exception & e) {if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1;}}
