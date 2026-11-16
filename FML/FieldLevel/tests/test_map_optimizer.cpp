#include <FML/FieldLevel/MAP.h>
#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/tests/TestHelpers.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
namespace FF=FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
// Independent coupled quadratic: distributed field, two replicated scalars,
// analytic minimizer, stiff directions, and an empty owner at four ranks.
struct Quadratic {
    struct Workspace {FF::Modes prediction,cotangent;FF::Modes & cotangent_buffer(){return cotangent;}};
    int first=3*FML::ThisTask/FML::NTasks,last=3*(FML::ThisTask+1)/FML::NTasks;
    bool metric=false,nonconvex=false;
    mutable size_t forwards{},pullbacks{};
    std::vector<FF::Parameter> spec{{"coupling",.2,0.,2.,-10.,10.},{"noise",-.1,0.,.7,-10.,10.}};
    void controls(const std::vector<double> & v)const {
        for(double x:v){double lo=x,hi=x;FML::MinOverTasks(&lo);FML::MaxOverTasks(&hi);
            if(lo!=hi)throw std::invalid_argument("inconsistent controls");}
    }
    size_t cells()const{return last-first;}
    const auto & parameters()const{return spec;}
    double A(size_t i)const{return nonconvex?1.:std::pow(8.,first+int(i));}
    double B(size_t i)const{return nonconvex?0.:.3*(first+int(i)+1);}
    double D(size_t i)const{return std::sin(first+int(i)+1.)+.7;}
    bool in_support(const std::vector<double> & p)const{return std::isfinite(p[0])&&std::isfinite(p[1])&&std::abs(p[0])<=10&&std::abs(p[1])<=10;}
    Workspace workspace()const{return {};}
    const FF::Modes & predict(const Field & s,const std::vector<double> & p,Workspace & w)const {
        ++forwards;w.prediction.resize(s.size());
        for(size_t i=0;i<s.size();++i)w.prediction[i]=A(i)*s[i]+B(i)*p[0];return w.prediction;
    }
    void pullback(const std::vector<double> &,const FF::Modes & g,Workspace &,Field & gs,std::vector<double> & gp)const {
        ++pullbacks;gs.resize(cells());gp.assign(2,0.);
        for(size_t i=0;i<gs.size();++i){gs[i]=A(i)*g[i].real();gp[0]+=B(i)*g[i].real();}
    }
    Field apply_field_metric(const Field & v,const std::vector<double>&,Workspace&,int)const {
        auto r=v;if(metric)for(size_t i=0;i<r.size();++i)r[i]/=1+A(i)*A(i);return r;
    }
};
struct QuadraticLikelihood {
    const Quadratic * model;
    double evaluate(const FF::Modes & pred,double noise,FF::Modes & g,double & ng)const {
        g.resize(pred.size());double u=0;ng=0;
        for(size_t i=0;i<pred.size();++i){double x=pred[i].real();
            if(model->nonconvex){u+=.25*x*x*x*x-2*x*x;g[i]=x*x*x-4*x;}
            else {double r=x-model->D(i);u+=.5*r*r;g[i]=r;}}
        if(FML::ThisTask==0){u+=.5*(noise-.6)*(noise-.6);ng=noise-.6;}return u;
    }
};
namespace FML::FIELDLEVEL::FRAMEWORK {
    template<>struct Execution<Quadratic> {
        static double sum(const Quadratic &,double v){FML::SumOverTasks(&v);return v;}
        static double maximum(const Quadratic &,double v){FML::MaxOverTasks(&v);return v;}
        static void validate(const Quadratic & m,const Field & s,const std::vector<double>&p){
            if(maximum(m,s.size()!=m.cells()||p.size()!=2)!=0)throw std::invalid_argument("shape");}
        static void map_controls(const Quadratic & m,bool vary,int n,double x){
            for(double v:{double(vary),double(n),x}){double lo=v;FML::MinOverTasks(&lo);
                if(maximum(m,v)!=lo)throw std::invalid_argument("inconsistent controls");}}
    };
}
struct GaussianQuadratic : Quadratic {
    struct Mode {int n[3];double k,weight;};
    std::vector<Mode> modes()const {std::vector<Mode> r;for(size_t i=0;i<cells();++i)r.push_back({{first+int(i)+1,0,1},1.,2.});if(FML::ThisTask==0)r.push_back({{100,0,1},1.,2.});return r;}
    const FF::Modes & predict(const Field & s,const std::vector<double>&p,Workspace&w)const {
        Quadratic::predict(s,p,w);if(FML::ThisTask==0)w.prediction.push_back(0.);return w.prediction;
    }
    void pullback(const std::vector<double>&p,const FF::Modes&g,Workspace&w,Field&gs,std::vector<double>&gp)const {
        Quadratic::pullback(p,g,w,gs,gp);for(auto & x:gs)x*=2;gp[0]*=2;
    }
};
namespace FML::FIELDLEVEL::FRAMEWORK {template<>struct Execution<::GaussianQuadratic>:Execution<::Quadratic>{};}
static void stable_changes() {
    GaussianQuadratic m;auto modes=m.modes();FF::Modes data;for(size_t i=0;i<m.cells();++i)data.push_back(m.D(i));if(FML::ThisTask==0)data.push_back(1e12);
    FF::BasicPosterior<GaussianQuadratic> post(m,FF::GaussianFieldLikelihood(modes,2.,1.,data));
    Field base(m.cells(),.1);std::vector<double> p{.2,0.};auto w=m.workspace();Field g;std::vector<double> gp;const double u=post.evaluate(base,p,w,g,gp);
    const auto reference=post.value_reference(base,p,w);auto trial=base;for(auto & x:trial)x+=1e-9;
    const auto delta=post.value_change(trial,p,w,base,reference);long double expected=0;
    for(size_t i=0;i<base.size();++i){long double dx=(long double)trial[i]-base[i];long double r=m.A(i)*base[i]+m.B(i)*p[0]-m.D(i),dr=m.A(i)*dx;expected+=2*r*dr+dr*dr+(long double)base[i]*dx+.5L*dx*dx;}
    const double exact=FF::Execution<GaussianQuadratic>::sum(m,double(expected));
    check(delta.value==u && delta.change!=0 && std::abs(delta.change-exact)<1e-10,"stable posterior change resolves changes hidden by a 1e24 constant");
    check(post.value_change(base,p,w,base,reference).change==0,"zero displacement has exactly zero change");
    auto outside=p;outside[1]=11;check(std::isinf(post.value_change(trial,outside,w,base,reference).change),"stable change rejects unsupported proposals");
    FF::BasicPosterior<GaussianQuadratic> other(m,FF::GaussianFieldLikelihood(modes,2.,1.,data));bool rejected=false;
    try{other.value_change(trial,p,w,base,reference);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"value reference rejects a foreign posterior");
    FF::MAPOptions o;o.vary_parameters=false;o.stable_objective_changes=true;o.max_iterations=150;o.tolerance=1e-7;
    m.forwards=m.pullbacks=0;const auto fit=FF::map(post,base,p,o);double err=0;
    for(size_t i=0;i<base.size();++i)err=std::max(err,std::abs(fit.fit.white[i]-2*m.A(i)*(m.D(i)-m.B(i)*p[0])/(1+2*m.A(i)*m.A(i))));
    check(fit.fit.status==FF::MAPStatus::Converged && FF::Execution<GaussianQuadratic>::maximum(m,err)<2e-7,"stable Armijo reaches analytic optimum despite huge constant target");
    check(m.forwards==fit.value_evaluations+fit.gradient_evaluations+fit.reference_evaluations && m.pullbacks==fit.gradient_evaluations,"stable reference prediction work is accounted for");
    bool sufficient=true;for(size_t i=1;i<fit.trace.size();++i)sufficient &= fit.trace[i].accepted_change<=fit.trace[i].armijo_bound;
    check(sufficient,"every stable accepted iterate satisfies change-form Armijo");
    // Noise-only differences, including tiny changes and large valid log variances.
    std::vector<GaussianQuadratic::Mode> local_modes{{{0,0,1},1.,2.}};FF::Modes zero{0.},signal{.7};FF::GaussianFieldLikelihood like(local_modes,2.,3.,zero);
    for(double dn:{1e-12,-1e-12,.2,-.2,2.,-2.}){const long double expected=.49L*3*std::expm1(-(long double)dn)+(long double)dn;
        check(std::abs(like.difference(signal,dn,signal,0.)-expected)<1e-14,"stable Gaussian difference includes noise normalization and variance");}
    check(std::abs(like.difference(zero,700,zero,-700)-1400.L)<1e-12,"large valid noise changes avoid overflow ratios");
    rejected=false;try{like.difference(zero,710,zero,0);}catch(const FF::UnrepresentableState&){rejected=true;}check(rejected,"unrepresentable stable noise variance rejects");
    // Joint white/parameter changes check scalar priors once across ranks. Use
    // moderate data so ordinary target differences remain an independent check.
    FF::Modes ordinary=data;if(FML::ThisTask==0)ordinary.back()=.3;FF::BasicPosterior<GaussianQuadratic> normal(m,FF::GaussianFieldLikelihood(modes,2.,1.,ordinary));
    const auto ref=normal.value_reference(base,p,w);auto p2=p;p2[0]+=.001;p2[1]+=.002;const auto joint=normal.value_change(trial,p2,w,base,ref);
    check(std::abs(joint.change-(normal.value(trial,p2,w)-normal.value(base,p,w)))<1e-10,"joint stable change includes replicated scalar priors exactly once");
    const auto reverse=normal.value_reference(trial,p2,w);
    check(std::abs(joint.change+normal.value_change(base,p,w,trial,reverse).change)<1e-12,"joint stable change is antisymmetric");
    GaussianQuadratic uniform;uniform.spec[0].prior=FF::Prior::Uniform;
    FF::BasicPosterior<GaussianQuadratic> uniform_post(uniform,FF::GaussianFieldLikelihood(uniform.modes(),2.,1.,ordinary));auto uw=uniform.workspace();
    const auto ur=uniform_post.value_reference(base,p,uw);const auto uc=uniform_post.value_change(trial,p2,uw,base,ur);
    check(std::abs(uc.change-(uniform_post.value(trial,p2,uw)-uniform_post.value(base,p,uw)))<1e-10,"uniform scalar prior contributes zero change");
#ifdef USE_MPI
    if(FML::NTasks>1){o.stable_objective_changes=FML::ThisTask!=0;rejected=false;try{FF::map(normal,base,p,o);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"inconsistent stable comparison options reject collectively");}
#endif
}
static void quadratic() {
    Quadratic m;FF::BasicPosterior<Quadratic,QuadraticLikelihood> p(m,QuadraticLikelihood{&m});
    Field s(m.cells(),.1);std::vector<double> theta{.2,-.1};auto w=m.workspace();Field g;std::vector<double> gp;
    const double u=p.evaluate(s,theta,w,g,gp),v=p.value(s,theta,w);check(u==v,"value-only and full target are identical");
    check(m.forwards==2&&m.pullbacks==1,"value-only evaluation avoids pullback");
    auto outside=theta;outside[0]=11;check(std::isinf(p.value(s,outside,w)),"value-only rejects unsupported parameters");
    double numerator=0,denominator=.25;
    for(size_t i=0;i<s.size();++i){double a=m.A(i),b=m.B(i);numerator+=b*m.D(i)/(1+a*a);denominator+=b*b/(1+a*a);}
    numerator=FF::Execution<Quadratic>::sum(m,numerator);denominator=FF::Execution<Quadratic>::sum(m,denominator-.25)+.25;
    const double expected_theta=numerator/denominator,expected_noise=.6/(1+1/.49);
    FF::MAPOptions options;options.max_iterations=150;options.tolerance=1e-7;
    for(bool metric:{false,true}){m.metric=metric;const auto result=FF::map(p,s,theta,options);
        double err=std::max(std::abs(result.fit.parameters[0]-expected_theta),std::abs(result.fit.parameters[1]-expected_noise));
        for(size_t i=0;i<s.size();++i)err=std::max(err,std::abs(result.fit.white[i]-m.A(i)*(m.D(i)-m.B(i)*expected_theta)/(1+m.A(i)*m.A(i))));
        if(FML::ThisTask==0)std::cout<<"quadratic metric="<<metric<<" status="<<int(result.fit.status)<<" it="<<result.fit.iterations<<" g="<<result.fit.gradient_norm<<" err="<<err<<" resets="<<result.history_resets<<std::endl;
        check(result.fit.status==FF::MAPStatus::Converged && err<2e-7,"L-BFGS reaches analytic coupled field/scalar solution");
        check(result.peak_history_pairs<=5 && result.peak_history_bytes<=10*(s.size()+theta.size())*sizeof(double),"history memory respects configured cap");
        check(result.gradient_evaluations==size_t(result.fit.iterations+1),"only accepted trials request gradients");
        bool monotonic=true;for(size_t j=1;j<result.trace.size();++j)monotonic &= result.trace[j].potential<=result.trace[j-1].potential;
        check(monotonic,"accepted objectives decrease");}
    m.metric=false;options.vary_parameters=false;options.max_iterations=20;options.history=2;
    const auto fixed=FF::map(p,s,theta,options);check(fixed.fit.parameters==theta,"fixed scalars remain bitwise unchanged");
    check(fixed.peak_history_pairs<=2,"short history remains bounded");
    options.algorithm=FF::MAPAlgorithm::GradientDescent;options.history=0;options.max_iterations=4;
    const auto fast=FF::map(p,s,theta,options);const auto old=FF::map(p,s,theta,false,4,options.tolerance);
    check(std::abs(old.potential-fast.fit.potential)<1e-12 && old.white==fast.fit.white,"value-only GD matches legacy accepted iterates");
    check(fast.value_evaluations>fast.gradient_evaluations && fast.peak_history_pairs==0,"rejected GD trials avoid gradients and history");
    options.max_iterations=0;const auto zero=FF::map(p,s,theta,options);check(zero.fit.white==s&&zero.fit.iterations==0,"zero iteration budget preserves input");
    options.max_iterations=4;options.max_line_search=1;const auto fail=FF::map(p,s,theta,options);check(fail.fit.status==FF::MAPStatus::LineSearchFailed&&fail.fit.white==s,"failed line search preserves last accepted state");
    options.algorithm=FF::MAPAlgorithm::LBFGS;options.max_line_search=40;options.max_iterations=8;options.history=3;m.nonconvex=true;
    const auto nonconvex=FF::map(p,s,theta,options);check(nonconvex.curvature_skips>0 && nonconvex.fit.potential<u,"negative curvature is skipped safely");
    options.history=0;bool rejected=false;try{FF::map(p,s,theta,options);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"invalid L-BFGS history rejects collectively");
    {auto unsupported=FF::MAPOptions{};unsupported.stable_objective_changes=true;bool caught=false;
     try{FF::map(p,s,theta,unsupported);}catch(const std::invalid_argument&){caught=true;}
     check(caught,"unsupported likelihood rejects explicit stable comparisons");}
#ifdef USE_MPI
    if(FML::NTasks>1){options.history=FML::ThisTask==0?2:3;rejected=false;try{FF::map(p,s,theta,options);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"inconsistent rank options reject collectively");}
#endif
}
template<class Grid> static void particle_mesh(const FML::Domain & domain) {
    FF::Configuration c;c.mesh=c.density_mesh=8;c.boxsize=1;c.evolution=FF::Evolution::Linear;c.initial_cutoff=c.bias_cutoff=10.;c.likelihood_cutoff=15.;
    FF::PMSettings pm;pm.scale_factors={.2,.3,.4};pm.force_window=FF::PMForceWindow::CICCompensated;
    FF::PMObservationSettings obs;obs.reference_cic_mesh=6;
    using Model=FF::PeriodicPMModelT<Grid>;
    Model m(c,FF::Spectrum::fixed([](double){return .0001;}),pm,{"log_noise_power",-7.,-7.,.5,-12.,-2.},{2},domain,obs);
    auto w=m.workspace();auto theta=m.initial_parameters();auto s=m.white_noise(12);for(auto & x:s)x*=.05;
    FF::BasicPosterior<Model> p(m,m.mock(m.white_noise(11),theta,13,w));Field g;std::vector<double> gp;
    double u=p.evaluate(s,theta,w,g,gp);check(std::abs(u-p.value(s,theta,w))<1e-12,"PM value/full target agreement with fixed response");
    FF::MAPOptions o;o.vary_parameters=false;o.max_iterations=12;o.tolerance=1e-5;
    auto fit=FF::map(p,s,theta,o);if(FML::ThisTask==0)std::cout<<"PM status="<<int(fit.fit.status)<<" it="<<fit.fit.iterations<<" g="<<fit.fit.gradient_norm<<" resets="<<fit.history_resets<<std::endl;check(std::isfinite(fit.fit.potential) && fit.fit.potential<u,"distributed nonlinear PM L-BFGS improves the target");
    check(std::abs(fit.fit.potential-p.value(fit.fit.white,fit.fit.parameters,w))<1e-11,"returned PM target matches fresh evaluation");
    FF::PMObservationSettings zero;zero.response.multiplier.assign(m.modes().size(),0.);
    Model null(c,FF::Spectrum::fixed([](double){return .0001;}),pm,{"log_noise_power",-7.,-7.,.5,-12.,-2.},{2},domain,zero);
    auto nw=null.workspace();FF::BasicPosterior<Model> prior(null,null.mock(s,theta,13,nw));
    const auto exact=FF::map(prior,s,theta,o);
    check(exact.fit.status==FF::MAPStatus::Converged && exact.fit.gradient_norm<o.tolerance,"real PM null response reaches the exact prior optimum");
}
int main(int argc,char **argv){FML::FMLSetup::init(&argc,&argv);quadratic();stable_changes();
    auto slab=FML::Domain::uniform_slab(FML::NTasks);FML::set_domain(slab);particle_mesh<FML::GRID::FFTWGrid<3>>(slab);
#ifdef USE_MPI
    auto pencil=FML::Domain::uniform_pencil(FML::NTasks==4?2:FML::NTasks,FML::NTasks==4?2:1);FML::set_domain(pencil);particle_mesh<FML::GRID::PencilGrid<3>>(pencil);
#endif
    return FML::TESTS::summary("MAP optimizer");}
