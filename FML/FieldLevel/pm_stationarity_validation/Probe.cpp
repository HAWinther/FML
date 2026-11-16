#include <FML/FieldLevel/pm_stationarity_validation/State.h>
// CIC stencil bases at all force/deposition nodes, using the forward's exact
// wrapped-position operation. These are perturbation changes at corresponding
// nodes, not ordinary physical crossings between consecutive time nodes.
class Trajectory {
    const Model & m;
    std::vector<std::uint16_t> base;
    size_t nodes;
    double local_min=HUGE_VAL,local_code=HUGE_VAL,local_u=0;
  public:
    size_t critical_node{},critical_id{};int critical_axis{};double critical_u{},min_face{};
    explicit Trajectory(const Model & model):m(model),nodes(m.dynamics().settings().scale_factors.size()){
        FML::COMMUNICATION::collective_try([&]{base.resize(nodes*m.cells()*3);},"PM stencil trace storage");}
    void baseline(size_t node,const Model::Reconstruction & r){
        for(size_t q=0;q<m.cells();++q){const auto x=r.position(q);for(int d=0;d<3;++d){const double u=x[d]*m.configuration().density_mesh;const auto cell=std::floor(u);base[(node*m.cells()+q)*3+d]=std::uint16_t(int(cell)%m.configuration().density_mesh);
            const double dist=std::min(u-cell,1-(u-cell));const double code=(node*m.layout().global_cells+m.layout().global_cell_id(q))*3+d;
            if(dist<local_min||(dist==local_min&&code<local_code)){local_min=dist;local_code=code;local_u=u;}}}}
    void finish(){min_face=-m.maximum(-local_min);double code=local_min==min_face?local_code:HUGE_VAL;code=-m.maximum(-code);
        critical_node=size_t(code)/(m.layout().global_cells*3);critical_id=(size_t(code)/3)%m.layout().global_cells;critical_axis=int(size_t(code)%3);critical_u=m.sum(local_code==code?local_u:0.);}
    struct Change {double axes{},particles{},first_node{-1},min_face{},critical_distance{};};
    template<class Work> Change compare(const Field & s,const std::vector<double> & theta,Work & work)const {
        double changed=0,first=double(nodes),closest=HUGE_VAL,cu=0.;std::vector<unsigned char> touched;
        FML::COMMUNICATION::collective_try([&]{touched.assign(m.cells(),0);},"PM trace comparison");
        m.dynamics().evolve_observed(s,theta,work,[&](size_t node,const auto & r){
            for(size_t q=0;q<m.cells();++q){const auto x=r.position(q);for(int d=0;d<3;++d){const double u=x[d]*m.configuration().density_mesh;const auto cell=std::floor(u);closest=std::min(closest,std::min(u-cell,1-(u-cell)));
                if(std::uint16_t(int(cell)%m.configuration().density_mesh)!=base[(node*m.cells()+q)*3+d]){++changed;touched[q]=1;first=std::min(first,double(node));}
                if(node==critical_node && m.layout().global_cell_id(q)==critical_id && d==critical_axis)cu=u;}}});
        const double u=m.sum(cu),mesh=m.configuration().density_mesh;double delta=u-critical_u;delta-=mesh*std::round(delta/mesh);
        first=-m.maximum(-first);return {m.sum(changed),m.sum(double(std::count(touched.begin(),touched.end(),1))),first==nodes?-1:first,-m.maximum(-closest),critical_u-std::round(critical_u)+delta};
    }
};
static double difference(const Model & m,const FF::BasicPosterior<Model> & posterior,const Field & base,const FF::Modes & prediction,
                         const Field & trial,const FF::Modes & next,const std::vector<double> & p){
    long double delta=0;const long double variance=std::exp(p.back())/m.volume();const auto & data=posterior.likelihood().data();const auto & weights=posterior.likelihood().weights();
    for(size_t q=0;q<base.size();++q){const long double dx=(long double)trial[q]-base[q];delta+=(long double)base[q]*dx+.5L*dx*dx;}
    for(size_t j=0;j<prediction.size();++j){const long double dr=(long double)next[j].real()-prediction[j].real(),di=(long double)next[j].imag()-prediction[j].imag();
        const long double rr=(long double)prediction[j].real()-data[j].real(),ri=(long double)prediction[j].imag()-data[j].imag();delta+=weights[j]*(rr*dr+ri*di+.5L*(dr*dr+di*di))/variance;}
    return m.sum(double(delta));
}
int main(int argc,char **argv){try{
    Controls c;c.seed=91;c.power=.002;c.final=1.;c.window=1;c.reference=16;std::string snapshot;int full=0;
    FML::COMMUNICATION::collective_try([&]{if(argc!=8)throw std::invalid_argument("pm_stationarity_probe PREFIX DATA SNAPSHOT N M STEPS FULL");c.prefix=argv[1];c.data=argv[2];snapshot=argv[3];c.n=integer(argv[4]);c.m=integer(argv[5]);c.steps=integer(argv[6]);full=integer(argv[7]);},"Probe arguments");
    require(c.n>=16&&c.n<=64&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64&&(full==0||full==1),"Invalid probe controls");
    FF::CHECKPOINT::same_string(state_identity(c)+","+std::to_string(full));for(const auto & path:{c.prefix,c.data,snapshot})FF::CHECKPOINT::same_string(path);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto model=make_model(c);const auto saved=load(model,c,snapshot);const auto p=saved.parameters;const std::vector<double> theta(p.begin(),p.end()-1);
    FF::BasicPosterior<Model> posterior(model,load_data(model,c));auto work=model.workspace();auto dyn=model.dynamics().workspace();
    auto metrics=output(c.prefix+".metrics.csv");if(FML::ThisTask==0)metrics<<"point,potential,raw_gradient,field_gradient_l2,saved_value_error,saved_gradient_error,value_only_offset,nearest_face,node,particle,axis\n";
    auto probes=output(c.prefix+".directions.csv");if(FML::ThisTask==0)probes<<"point,direction,h,sign,analytic,value_delta,stable_delta,predicted_delta,changed_axes,changed_particles,first_node,min_face,critical_distance,trajectory_error,manual_value_delta\n";
    auto lines=output(c.prefix+".line.csv");if(FML::ThisTask==0)lines<<"alpha,analytic,value_delta,stable_delta,predicted_delta,armijo_delta,changed_axes,changed_particles,first_node,critical_distance,unit_trial_error,manual_value_delta\n";
    auto caches=output(c.prefix+".cache.csv");if(FML::ThisTask==0)caches<<"point,budget,value_error,gradient_error,scalar_error,value_only_offset\n";
    for(int kind=0;kind<(full?2:1);++kind){
        const std::string name=kind?"initial":"stalled";Field white=saved.white;
        if(kind){auto co=coefficients(300);for(auto & x:co)x*=.05;white=expand(model,co);}
        Field g;std::vector<double> gp;const double u=posterior.evaluate(white,p,work,g,gp);const double value_only=posterior.value(white,p,work);const auto prediction=model.predict(white,p,work);
        double norm=std::sqrt(model.sum(inner(g,g))),raw=0,error=0;for(size_t q=0;q<g.size();++q){raw=std::max(raw,std::abs(g[q]));error=std::max(error,std::abs(g[q]-saved.gradient[q]));}raw=model.maximum(raw);error=model.maximum(error);
        Trajectory trace(model);const auto & direct=model.dynamics().evolve_observed(white,theta,dyn,[&](size_t node,const auto & r){trace.baseline(node,r);});trace.finish();
        double trajectory_error=0;const auto & matter=model.reconstruction(work).matter_density;for(size_t q=0;q<matter.size();++q)trajectory_error=std::max(trajectory_error,std::abs(matter[q]-direct.matter_density[q]));
        require(model.maximum(trajectory_error)==0.,"Observer changes trajectory");
        if(FML::ThisTask==0)metrics<<name<<','<<u<<','<<raw<<','<<norm<<','<<(kind?0:u-saved.meta[3])<<','<<(kind?0:error)<<','<<value_only-u<<','<<trace.min_face<<','<<trace.critical_node<<','<<trace.critical_id<<','<<trace.critical_axis<<'\n';
        auto evaluate=[&](const Field & state){const double public_value=posterior.value(state,p,work);const auto & predicted=model.predict(state,p,work);
            double unused_noise=0;double next=posterior.likelihood().evaluate(predicted,p.back(),work.cotangent_buffer(),unused_noise);
            for(double x:state)next+=.5*x*x;next=model.sum(next);for(size_t j=0;j<p.size();++j)next+=model.parameters()[j].penalty(p[j]);
            const double stable=difference(model,posterior,white,prediction,state,predicted,p);const auto change=trace.compare(state,theta,dyn);
            double err=0;const auto & observed=model.reconstruction(work).matter_density;const auto & raw_fields=dyn.result().matter_density;
            for(size_t q=0;q<observed.size();++q)err=std::max(err,std::abs(observed[q]-raw_fields[q]));
            return std::make_tuple(public_value-u,stable,change,model.maximum(err),next-u);};
        const Field hs=full?Field{.001,.0001,.00001,.000001,.0000001,3e-8,.00000001,3e-9,1e-9,3e-10,1e-10,1e-11,1e-12,1e-13}:Field{1e-4,1e-6,1e-8};
        for(int direction=0;direction<(full?3:1);++direction){Field d=g;if(direction==1)d=saved.direction;
            if(direction==2){auto co=coefficients(11);const double cn=std::sqrt(inner(co,co));for(auto & x:co)x/=cn;d=expand(model,co);}
            const double dn=std::sqrt(model.sum(inner(d,d)));require(dn>0,"Zero probe direction");for(auto & x:d)x*=direction==2?1/dn:-1/dn;const double analytic=model.sum(inner(g,d));
            for(double h:hs)for(int sign:{-1,1}){auto s=white;for(size_t q=0;q<s.size();++q)s[q]+=sign*h*d[q];
                const auto [delta,stable,change,err,manual]=evaluate(s);
                if(FML::ThisTask==0)probes<<name<<','<<direction<<','<<h<<','<<sign<<','<<analytic<<','<<delta<<','<<stable<<','<<sign*h*analytic<<','<<change.axes<<','<<change.particles<<','<<change.first_node<<','<<change.min_face<<','<<change.critical_distance<<','<<err<<','<<manual<<'\n';}
        }
        if(!kind && saved.meta[5]==1){const double analytic=-model.sum(inner(g,saved.direction));for(int k=0;k<(full?40:3);++k){const double alpha=std::ldexp(1.,-k);auto s=white;double first_error=0;
            for(size_t q=0;q<s.size();++q){s[q]-=alpha*saved.direction[q];if(k==0)first_error=std::max(first_error,std::abs(s[q]-saved.unit_trial[q]));}
            first_error=model.maximum(first_error);const auto [delta,stable,change,err,manual]=evaluate(s);require(err==0.,"Observed line trajectory mismatch");
            if(FML::ThisTask==0)lines<<alpha<<','<<analytic<<','<<delta<<','<<stable<<','<<alpha*analytic<<','<<1e-4*alpha*analytic<<','<<change.axes<<','<<change.particles<<','<<change.first_node<<','<<change.critical_distance<<','<<first_error<<','<<manual<<'\n';}}
        // Reused cache has seen all perturbations; compare with fresh workspaces and replay budgets.
        for(size_t budget:{size_t(0),size_t(1),size_t(4),size_t(8)}){auto other=make_model(c,{budget});auto ow=other.workspace();FF::BasicPosterior<Model> op(other,load_data(other,c));Field og;std::vector<double> opg;
            const double ou=op.evaluate(white,p,ow,og,opg);double ge=0,pe=0;for(size_t q=0;q<g.size();++q)ge=std::max(ge,std::abs(og[q]-g[q]));for(size_t j=0;j<gp.size();++j)pe=std::max(pe,std::abs(opg[j]-gp[j]));ge=model.maximum(ge);pe=model.maximum(pe);
            const double value_offset=op.value(white,p,ow)-ou;
            if(FML::ThisTask==0)caches<<name<<','<<budget<<','<<ou-u<<','<<ge<<','<<pe<<','<<value_offset<<'\n';}
        Field reused;std::vector<double> reused_gp;const double ru=posterior.evaluate(white,p,work,reused,reused_gp);double ge=0,pe=0;for(size_t q=0;q<g.size();++q)ge=std::max(ge,std::abs(reused[q]-g[q]));for(size_t j=0;j<gp.size();++j)pe=std::max(pe,std::abs(reused_gp[j]-gp[j]));ge=model.maximum(ge);pe=model.maximum(pe);
        const double reused_value_offset=posterior.value(white,p,work)-ru;
        if(FML::ThisTask==0)caches<<name<<",-1,"<<ru-u<<','<<ge<<','<<pe<<','<<reused_value_offset<<'\n';
    }
    if(FML::ThisTask==0)std::cout<<"Stationarity probes complete"<<std::endl;return 0;
}catch(const std::exception & e){if(FML::ThisTask==0)std::cerr<<e.what()<<std::endl;return 1;}}
