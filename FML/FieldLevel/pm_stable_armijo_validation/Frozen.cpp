#include <FML/FieldLevel/pm_stationarity_validation/State.h>
int main(int argc,char **argv){try{
    Controls c;c.seed=91;c.power=.002;c.final=1.;c.window=1;c.reference=16;std::string snapshot;
    FML::COMMUNICATION::collective_try([&]{if(argc!=7)throw std::invalid_argument("pm_stable_frozen PREFIX DATA SNAPSHOT N M STEPS");c.prefix=argv[1];c.data=argv[2];snapshot=argv[3];c.n=integer(argv[4]);c.m=integer(argv[5]);c.steps=integer(argv[6]);},"Stable frozen arguments");
    require(c.n>=16&&c.n<=64&&c.n%2==0&&c.m>=16&&c.m<=64&&c.m%2==0&&c.steps>=2&&c.steps<=64,"Invalid frozen controls");
    FF::CHECKPOINT::same_string(state_identity(c));for(const auto & path:{c.prefix,c.data,snapshot})FF::CHECKPOINT::same_string(path);
    FML::FIELDLEVEL::FRONTEND::initialize(c.n,c.m);auto m=make_model(c);const auto s=load(m,c,snapshot);const auto p=s.parameters;auto work=m.workspace();
    FF::BasicPosterior<Model> posterior(m,load_data(m,c));Field g;std::vector<double> gp;const double u=posterior.evaluate(s.white,p,work,g,gp);
    const auto reference=posterior.value_reference(s.white,p,work);const double slope=m.sum(inner(g,s.direction));
    auto file=output(c.prefix+".line.csv");if(FML::ThisTask==0)file<<"alpha,raw_delta,stable_delta,armijo_delta,raw_accept,stable_accept,returned_value_error\n";
    for(int k=0;k<40;++k){const double alpha=std::ldexp(1.,-k);auto candidate=s.white;for(size_t q=0;q<candidate.size();++q)candidate[q]-=alpha*s.direction[q];
        const auto change=posterior.value_change(candidate,p,work,s.white,reference);const double value=posterior.value(candidate,p,work),bound=-1e-4*alpha*slope;
        if(FML::ThisTask==0)file<<alpha<<','<<value-u<<','<<change.change<<','<<bound<<','<<int(value<=u+bound)<<','<<int(change.change<=bound)<<','<<change.value-value<<'\n';}
    if(FML::ThisTask==0)std::cout<<"Frozen stable Armijo probes complete"<<std::endl;return 0;
}catch(const std::exception &e){if(FML::ThisTask==0)std::cerr<<e.what()<<std::endl;return 1;}}
