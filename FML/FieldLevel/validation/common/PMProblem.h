#ifndef FML_PM_NONLINEAR_VALIDATION_H
#define FML_PM_NONLINEAR_VALIDATION_H
#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/FieldLevel/MAP.h>
#include <chrono>
#include <sys/resource.h>
#include <FML/FieldLevel/Checkpoint.h>
#include <FML/FieldLevel/Frontend.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using BaseModel = FF::PeriodicPMModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
// Validation-only counters around public predictions/pullbacks (reset before MAP).
struct Model : BaseModel {
    using BaseModel::BaseModel;
    mutable size_t forwards{},pullbacks{};
    const FF::Modes & predict(const Field & s,const std::vector<double> & p,Workspace & w) const {
        ++forwards;return BaseModel::predict(s,p,w);
    }
    void pullback(const std::vector<double> & p,const FF::Modes & g,Workspace & w,Field & gs,std::vector<double> & gp) const {
        ++pullbacks;BaseModel::pullback(p,g,w,gs,gp);
    }
};
namespace FML::FIELDLEVEL::FRAMEWORK {
    template <> struct Execution<::Model> : Execution<::BaseModel> {};
}
struct Wave { int x, y, z; };
struct Controls {
    int n{}, m{}, steps{}, seed{}, fd{}, fit{}, forward{}, window{}, reference{}, optimizer{}, history{5}, stable{}, assignment{2};
    double power{}, final{};
    std::string prefix, data;
    int noise_seed{92};
};
static constexpr int common_mesh = 16;
static constexpr double noise_power = 1e-4;
static inline void require(bool ok, const char * text) {
    int bad = !ok; FML::MaxOverTasks(&bad);
    if (bad) throw std::invalid_argument(text);
}
static inline std::vector<Wave> waves() {
    std::vector<Wave> result;
    for (int x=-3; x<=3; ++x) for (int y=-3; y<=3; ++y) for (int z=-3; z<=3; ++z) {
        const int q=x*x+y*y+z*z;
        if (q>0 && q<=9 && (x>0 || (x==0 && y>0) || (x==0 && y==0 && z>0)))
            result.push_back({x,y,z});
    }
    return result;
}
static inline double inner(const Field & a, const Field & b) {
    double sum=0; for (size_t q=0; q<a.size(); ++q) sum+=a[q]*b[q]; return sum;
}
// Generate/project one Fourier pair at a time: no O(active_modes * N^3) basis storage.
static inline Field coefficients(int seed) {
    Field result;
    for (const auto & k:waves()) {
        const auto id=std::uint64_t((k.x+4)*81+(k.y+4)*9+k.z+4);
        FML::RANDOM::COUNTER_BASED::Stream rng(seed,id,0x504d4e4cu);
        result.push_back(rng.normal()); result.push_back(rng.normal());
    }
    return result;
}
static inline Field expand(const Model & model, const Field & coefficients) {
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
static inline Field project(const Model & model, const Field & field) {
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
template<class ModelType=Model> static inline ModelType make_model(const Controls & c,FF::PMAdjointOptions reverse={8}) {
    FF::Configuration config; config.mesh=c.n; config.density_mesh=c.m; config.boxsize=1;
    config.evolution=FF::Evolution::Linear;
    config.initial_cutoff=config.bias_cutoff=2*M_PI*std::sqrt(9.5);
    config.likelihood_cutoff=2*M_PI*std::sqrt(16.5);
    FF::Spectrum spectrum{{{"log_amplitude",0.,0.,.5,-1.,1.}},
                         [power=c.power](double,const auto & p) { return FF::Power{power*std::exp(p[0]),{1.}}; }};
    FF::PMSettings pm; pm.assignment=static_cast<FF::PMAssignment>(c.assignment); pm.force_window=static_cast<FF::PMForceWindow>(c.window); pm.scale_factors.clear();
    for (int j=0; j<=c.steps; ++j) pm.scale_factors.push_back(.2+(c.final-.2)*j/c.steps);
    const double noise=std::log(noise_power);
    FF::PMObservationSettings observed;observed.reference_cic_mesh=c.reference;
    return ModelType(config,spectrum,pm,{"log_noise_power",noise,noise,.5,-16.,-4.},reverse,FML::domain,observed);
}
static inline size_t mode_index(const int * k) {
    return (size_t((k[0]+common_mesh)%common_mesh)*common_mesh+(k[1]+common_mesh)%common_mesh)*(common_mesh/2+1)+k[2];
}
static inline Field global_modes(const Model & model, const FF::Modes & modes) {
    Field result(common_mesh*common_mesh*(common_mesh/2+1)*2,0.);
    for (size_t j=0; j<modes.size(); ++j) {
        const auto i=mode_index(model.modes()[j].n); result[2*i]=modes[j].real(); result[2*i+1]=modes[j].imag();
    }
    FML::SumArrayOverTasks(result.data(),result.size()); return result;
}
static inline std::ofstream output(const std::string & path) {
    std::ofstream file;
    FML::COMMUNICATION::collective_try([&] {
        if (FML::ThisTask==0) {file.open(path); if (!file) throw std::runtime_error("Cannot write "+path); file<<std::setprecision(17);}
    },"PM nonlinear output");
    return file;
}
static inline void row(std::ofstream & file,const Field & values) {
    if (FML::ThisTask==0) {for (size_t j=0;j<values.size();++j) {if(j)file<<',';file<<values[j];}file<<'\n';}
}
static inline std::string identity(const Controls & c) {
    std::ostringstream text; text<<std::setprecision(17)<<"PM-nonlinear-v1,"<<c.seed<<','<<c.power<<','<<c.final<<",9,16,"<<noise_power;
    if(c.reference)text<<",observation_cic,"<<c.reference;
    if(c.assignment!=2)text<<",assignment_tsc";
    if(c.noise_seed!=92)text<<",noise_seed,"<<c.noise_seed;
    return text.str();
}
static inline FF::Modes load_data(const Model & model,const Controls & c) {
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
static inline double value(const Model & model,const FF::BasicPosterior<Model> & posterior,
                    const Field & white,const std::vector<double> & p,Model::Workspace & work) {
    const auto & predicted=model.predict(white,p,work);FF::Modes gradient;double noise;
    double u=model.sum(posterior.likelihood().evaluate(predicted,p.back(),gradient,noise)+.5*inner(white,white));
    for(size_t j=0;j<p.size();++j)u+=model.parameters()[j].penalty(p[j]);
    return u;
}
#endif
