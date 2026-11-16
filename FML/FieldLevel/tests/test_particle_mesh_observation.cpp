#include <FML/FieldLevel/ParticleMeshCheckpoint.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
namespace FF=FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
static FF::Configuration configuration() {
    FF::Configuration c;c.mesh=c.density_mesh=8;c.boxsize=1;c.evolution=FF::Evolution::Linear;
    c.initial_cutoff=c.bias_cutoff=10.;c.likelihood_cutoff=15.;return c;
}
static FF::Spectrum spectrum() {
    return {{{"log_amplitude",.1,0.,.5,-1.,1.}},[](double,const auto & p){return FF::Power{.001*std::exp(p[0]),{1.}};}};
}
static FF::Parameter noise() {double x=std::log(1e-4);return {"log_noise_power",x,x,.5,-16.,-4.};}
static double dot(const Field & a,const Field & b) {double x=0;for(size_t j=0;j<a.size();++j)x+=a[j]*b[j];return x;}
static double triangle(int k,int m) {
    double sum=0;const int panels=256;
    for(int j=0;j<=panels;++j) {
        double u=double(j)/panels;
        sum+=(j==0||j==panels?1:(j%2?4:2))*2*(1-u)*std::cos(2*M_PI*k*u/m);
    }
    return sum/(3*panels);
}
template<class Model> static double value(const Model & m,const FF::BasicPosterior<Model> & posterior,
        const Field & white,const std::vector<double> & p,typename Model::Workspace & work) {
    const auto & pred=m.predict(white,p,work);FF::Modes unused;double ng;
    double u=m.sum(posterior.likelihood().evaluate(pred,p.back(),unused,ng)+.5*dot(white,white));
    for(size_t j=0;j<p.size();++j)u+=m.parameters()[j].penalty(p[j]);return u;
}
static double quadratic(int k,int m) {
    double sum=0;const int panels=256;
    for(int section=0;section<2;++section) {
        const double lo=section?.5:0.,hi=section?1.5:.5,step=(hi-lo)/panels;
        for(int j=0;j<=panels;++j){double u=lo+j*step,w=section?.5*(1.5-u)*(1.5-u):.75-u*u;
            sum+=step/3*(j==0||j==panels?1:(j%2?4:2))*2*w*std::cos(2*M_PI*k*u/m);}
    }return sum;
}
struct Snapshot {Field values;};
template<class Grid> static Snapshot exercise(const FML::Domain & domain,const std::string & label, FF::PMAssignment assignment=FF::PMAssignment::CIC) {
    using Model=FF::PeriodicPMModelT<Grid>;auto config=configuration();Snapshot result;
    for(int kind=1;kind<=4;++kind) {
        FF::PMSettings pm;pm.assignment=assignment;pm.scale_factors={.2,.35,.5};
        if(kind>=3)pm.force_window=assignment==FF::PMAssignment::TSC?FF::PMForceWindow::TSCCompensated:FF::PMForceWindow::CICCompensated;
        Model raw(config,spectrum(),pm,noise(),{2},domain);auto rw=raw.workspace();auto p=raw.initial_parameters();
        auto white=raw.white_noise(93);for(auto & x:white)x*=.3;
        auto truth=raw.white_noise(91);for(auto & x:truth)x*=.7;
        const auto baseline=raw.predict(white,p,rw);const auto fields=raw.reconstruction(rw).matter_density;
        const auto raw_mock=raw.mock(white,p,92,rw);
        FF::PMObservationSettings obs;
        if(kind==2||kind==3)obs.reference_cic_mesh=6;
        if(kind==1||kind==3||kind==4) {
            obs.response.multiplier.resize(raw.modes().size());
            for(size_t j=0;j<raw.modes().size();++j) {
                const auto & n=raw.modes()[j].n;const int q=n[0]*n[0]+n[1]*n[1]+n[2]*n[2];
                obs.response.multiplier[j]=kind==4?0.:(q==1?0.:(q==2?-.7:.35));
            }
        }
        Model model(config,spectrum(),pm,noise(),{2},domain,obs);auto work=model.workspace();
        const auto prediction=model.predict(white,p,work);const auto data=model.mock(truth,p,94,work);
        const auto same_mock=model.mock(white,p,92,work);
        bool correct=true;double err=0;
        for(size_t j=0;j<prediction.size();++j) {
            double expected=(kind==1||kind==3||kind==4)?obs.response.multiplier[j]:1.;
            if(obs.reference_cic_mesh)for(int k:model.modes()[j].n)expected*=triangle(k,6)/(assignment==FF::PMAssignment::TSC?quadratic(k,8):triangle(k,8));
            err=std::max(err,std::abs(expected-model.observation_response()[j]));
            correct&=std::abs(prediction[j]-baseline[j]*model.observation_response()[j])<2e-13;
            correct&=std::abs((same_mock[j]-prediction[j])-(raw_mock[j]-baseline[j]))<2e-13;
        }
        check(model.maximum(err)<1e-9,label+" response matches independent assignment-kernel quadrature");
        check(correct,label+" fixed response changes signal and leaves added noise unchanged");
        check(model.observation_response_configured(),label+" configured response retained on ranks with no modes");
        const auto & raw_fields=model.reconstruction(work).matter_density;double field_error=0;
        for(size_t j=0;j<fields.size();++j)field_error=std::max(field_error,std::abs(fields[j]-raw_fields[j]));
        check(model.maximum(field_error)<2e-13,label+" reconstruction remains raw dynamical density");
        FF::BasicPosterior<Model> posterior(model,data);Field g;std::vector<double> gp;
        const double u=posterior.evaluate(white,p,work,g,gp);check(std::isfinite(u),label+" finite normalized posterior");
        if(kind==4) {
            double error=0;for(size_t j=0;j<g.size();++j)error=std::max(error,std::abs(g[j]-white[j]));
            check(model.maximum(error)<2e-12 && std::abs(gp[0]-model.parameters()[0].derivative(p[0]))<2e-12,label+" zero signal response leaves white/spectrum priors intact");
            const auto other=model.mock(truth,p,92,work);check(other==same_mock,label+" zero-response mock is noise-only and independent of input field");
        } else {
            auto direction=model.white_noise(95);double length=std::sqrt(model.sum(dot(direction,direction)));
            for(auto & x:direction)x/=length;
            for(int probe=0;probe<2;++probe) {
                const double da=probe?.2:0.,dn=probe?-.1:0.;double analytic=model.sum(dot(g,direction))+da*gp[0]+dn*gp[1];
                int stable=0;double best=HUGE_VAL;size_t step=0;
                for(double h:{.003,.001,.0003,.0001,.00003,.00001,.000003,.000001}) {
                    auto plus=white,minus=white;auto pp=p,pm_=p;
                    for(size_t j=0;j<white.size();++j){plus[j]+=h*direction[j];minus[j]-=h*direction[j];}
                    pp[0]+=h*da;pm_[0]-=h*da;pp[1]+=h*dn;pm_[1]-=h*dn;
                    double numeric=(value(model,posterior,plus,pp,work)-value(model,posterior,minus,pm_,work))/(2*h);
                    double error=std::abs(numeric-analytic)/std::max(1.,std::abs(analytic));
                    if(step++>=4){best=std::min(best,error);stable+=error<1e-4;}
                }
                check(best<2e-5 && stable>=2,label+" response posterior directional gradient "+std::to_string(probe));
            }
        }
        if(kind==3) {
            auto scaled_config=config;scaled_config.boxsize=2.;scaled_config.initial_cutoff*=.5;
            scaled_config.bias_cutoff*=.5;scaled_config.likelihood_cutoff*=.5;
            auto power=spectrum();const auto original=power.evaluate;
            power.evaluate=[original](double k,const auto & params){auto v=original(2*k,params);v.value*=8;return v;};
            auto variance=noise();variance.initial+=std::log(8.);variance.mean+=std::log(8.);
            variance.lower+=std::log(8.);variance.upper+=std::log(8.);
            Model scaled(scaled_config,power,pm,variance,{2},domain,obs);auto sw=scaled.workspace();auto sp=scaled.initial_parameters();
            const auto sd=scaled.mock(truth,sp,94,sw);FF::BasicPosterior<Model> posterior_scaled(scaled,sd);
            Field sg;std::vector<double> sgp;double su=posterior_scaled.evaluate(white,sp,sw,sg,sgp),error=0.;
            for(size_t j=0;j<g.size();++j)error=std::max(error,std::abs(sg[j]-g[j]));
            for(size_t j=0;j<gp.size();++j)error=std::max(error,std::abs(sgp[j]-gp[j]));
            check(model.maximum(error)<2e-9 && std::abs(su-u)<2e-9,label+" observed posterior respects box/P/noise volume scaling");
        }
        // Canonical Fourier outputs and white gradients make backend ownership errors visible.
        Field global(2*8*8*5+512+gp.size()+1,0.);
        for(size_t j=0;j<model.modes().size();++j) {
            const auto & n=model.modes()[j].n;size_t k=((size_t((n[0]+8)%8)*8+(n[1]+8)%8)*5+n[2])*2;
            global[k]=prediction[j].real();global[k+1]=prediction[j].imag();
        }
        for(size_t j=0;j<g.size();++j)global[2*8*8*5+model.layout().global_cell_id(j)]=g[j];
        FML::SumArrayOverTasks(global.data(),global.size());
        for(size_t j=0;j<gp.size();++j)global[2*8*8*5+512+j]=gp[j];global.back()=u;
        result.values.insert(result.values.end(),global.begin(),global.end());
        // Size, nonfinite and cross-rank Hermitian errors are collective configuration failures.
        if(kind==1) {
            auto bad=obs;if(FML::ThisTask==0)bad.response.multiplier.push_back(1.);
            check(FML::TESTS::throws([&]{Model invalid(config,spectrum(),pm,noise(),{2},domain,bad);}),label+" wrong local response length rejected");
            bad=obs;if(FML::ThisTask==0&&!bad.response.multiplier.empty())bad.response.multiplier[0]=std::numeric_limits<double>::quiet_NaN();
            check(FML::TESTS::throws([&]{Model invalid(config,spectrum(),pm,noise(),{2},domain,bad);}),label+" one-rank nonfinite response rejected");
            bad=obs;for(size_t j=0;j<raw.modes().size();++j){const auto & n=raw.modes()[j].n;if(n[0]==1&&n[1]==0&&n[2]==0)bad.response.multiplier[j]+=.125;}
            check(FML::TESTS::throws([&]{Model invalid(config,spectrum(),pm,noise(),{2},domain,bad);}),label+" cross-rank non-Hermitian response rejected");
            bad={};bad.reference_cic_mesh=10;
            check(FML::TESTS::throws([&]{Model invalid(config,spectrum(),pm,noise(),{2},domain,bad);}),label+" amplifying CIC reference rejected");
            bad.reference_cic_mesh=4;
            check(FML::TESTS::throws([&]{Model invalid(config,spectrum(),pm,noise(),{2},domain,bad);}),label+" reference Nyquist below observed band rejected");
        }
    }
    return result;
}
int main(int argc,char ** argv) {
    FML::FMLSetup::init(&argc,&argv);const auto slab=FML::Domain::uniform_slab(FML::NTasks);FML::set_domain(slab);
    const auto baseline=exercise<FML::GRID::FFTWGrid<3>>(slab,"slab");
    const auto tsc=exercise<FML::GRID::FFTWGrid<3>>(slab,"TSC-slab",FF::PMAssignment::TSC);
#ifdef USE_MPI
    int px=FML::NTasks==4?2:FML::NTasks,py=FML::NTasks/px;
    const auto pencil=FML::Domain::uniform_pencil(px,py);FML::set_domain(pencil);
    const auto other=exercise<FML::GRID::PencilGrid<3>>(pencil,"pencil");double error=0;
    for(size_t j=0;j<baseline.values.size();++j)error=std::max(error,std::abs(baseline.values[j]-other.values[j]));
    const auto tsc_other=exercise<FML::GRID::PencilGrid<3>>(pencil,"TSC-pencil",FF::PMAssignment::TSC);
    for(size_t j=0;j<tsc.values.size();++j)error=std::max(error,std::abs(tsc.values[j]-tsc_other.values[j]));
    check(error<2e-9,"slab/pencil response predictions, posterior and gradients agree");
#endif
    return FML::TESTS::summary("PM observation response");
}
