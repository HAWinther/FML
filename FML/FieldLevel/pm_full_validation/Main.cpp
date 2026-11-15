#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <FML/FieldLevel/Frontend.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
using Model = FF::PeriodicPMModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
static void require(bool condition, const char * message) {
    int bad = !condition; FML::MaxOverTasks(&bad); if (bad) throw std::invalid_argument(message);
}
static std::ofstream output(const std::string & path) {
    std::ofstream file;
    FML::COMMUNICATION::collective_try([&] {
        if (FML::ThisTask == 0) { file.open(path); if (!file) throw std::runtime_error("Cannot open " + path); file << std::setprecision(17); }
    }, "PM full validation output");
    return file;
}
template<class Layout> static Field canonical(const Field & local, const Layout & layout) {
    Field global(layout.global_cells, 0.);
    require(local.size() == layout.owned_cells, "Invalid canonical field size");
    for (size_t q = 0; q < local.size(); ++q) global[layout.global_cell_id(q)] = local[q];
    FML::SumArrayOverTasks(global.data(), global.size()); return global;
}
static void row(std::ofstream & file, const Field & values) {
    if (FML::ThisTask == 0) { for (size_t j = 0; j < values.size(); ++j) { if (j) file << ','; file << values[j]; } file << '\n'; }
}
static Model make_model() {
    FF::Configuration c; c.mesh = 4; c.density_mesh = 8; c.boxsize = 1.;
    c.initial_cutoff = c.bias_cutoff = 7.; c.likelihood_cutoff = 15.; c.evolution = FF::Evolution::Linear;
    FF::Spectrum spectrum{{{"log_amplitude", 0., 0., .5, -1., 1.}},
        [](double, const auto & p) { return FF::Power{.02 * std::exp(p[0]), {1.}}; }};
    FF::PMSettings settings; settings.scale_factors = {.2,.3,.4,.5,.6};
    const double noise = std::log(.001);
    return Model(c, spectrum, settings, {"log_noise_power", noise, noise, .5, -14., -3.}, {3});
}
int main(int argc, char ** argv) {
    try {
        FML::FIELDLEVEL::FRONTEND::initialize(4,8);
        std::string prefix; int joint=0,id=0,warm=0,draws=0,leap=0,adapt=0,stride=0; double eps=0;
        FML::COMMUNICATION::collective_try([&] {
            if (argc != 11) throw std::invalid_argument("pm_full PREFIX JOINT ID WARM DRAWS EPS LEAP ADAPT PREDICTIVE_STRIDE SCHEMA_VERSION");
            auto integer=[](const char * s) { size_t n; int x=std::stoi(s,&n); if(n!=std::string(s).size())throw std::invalid_argument("Invalid integer"); return x; };
            prefix=argv[1];joint=integer(argv[2]);id=integer(argv[3]);warm=integer(argv[4]);draws=integer(argv[5]);
            size_t used;eps=std::stod(argv[6],&used);if(used!=std::string(argv[6]).size())throw std::invalid_argument("Invalid epsilon");
            leap=integer(argv[7]);adapt=integer(argv[8]);stride=integer(argv[9]);
            // Reserved schema version prevents accidental interpretation by another fixture.
            if(integer(argv[10])!=1)throw std::invalid_argument("Expected fixture schema 1");
        }, "PM full validation input");
        require((joint==0||joint==1) && id>=0 && id<16 && warm>=0 && warm<=8192 && draws>0 && draws<=16384 &&
                std::isfinite(eps) && eps>0 && leap>=2 && leap<=128 && (adapt==0||adapt==1) && stride>0 && stride<=draws,
                "Invalid bounded PM full controls");
        auto model=make_model(); model.controls({double(joint),double(id),double(warm),double(draws),eps,double(leap),double(adapt),double(stride)});
        auto work=model.workspace(); const auto truth=model.white_noise(91); auto parameters=model.initial_parameters();
        const auto data=model.mock(truth,parameters,92,work); const auto & truth_result=model.reconstruction(work);
        auto truth_file=output(prefix+".truth.csv"); row(truth_file,canonical(truth,model.layout()));
        row(truth_file,canonical(truth_result.initial_density,model.layout()));
        row(truth_file,canonical(truth_result.matter_density,model.observation_layout()));
        // Complete canonical rFFT data within the model's observed support; other entries zero.
        Field data_modes(8*8*5*2,0.);
        for(size_t j=0;j<data.size();++j) {
            const auto & n=model.modes()[j].n;
            const size_t index=((n[0]+8)%8*8+(n[1]+8)%8)*5+n[2];
            data_modes[2*index]=data[j].real();data_modes[2*index+1]=data[j].imag();
        }
        FML::SumArrayOverTasks(data_modes.data(),data_modes.size());
        auto data_file=output(prefix+".data.csv");row(data_file,data_modes);
        FF::PeriodicPMPosteriorT<Model::Grid> posterior(model,data);
        auto start=model.white_noise(300+id); const double scales[4]={.5,1.,1.5,2.};
        for(auto & x:start)x*=scales[id%4];
        if(joint) { const double offsets[4]={-.7,.7,-.4,.4}; parameters[0]=offsets[id%4];parameters[1]+=offsets[(id+1)%4]; }
        auto state=output(prefix+".states.csv"), predictive=output(prefix+".predictive.csv");
        if(FML::ThisTask==0) {
            state<<"draw";for(int q=0;q<64;++q)state<<",w"<<q;state<<",log_amplitude,log_noise\n";
            predictive<<"draw";for(int q=0;q<64;++q)predictive<<",initial"<<q;for(int q=0;q<512;++q)predictive<<",matter"<<q;predictive<<'\n';
        }
        FML::MCMC::HMCOptions o;o.warmup=warm;o.samples=draws;o.step_size=eps;o.n_leapfrog=leap;
        o.adapt_step_size=adapt;o.adapt_mass=false;o.target_acceptance=.85;o.seed=9000+id;o.divergence_threshold=10.;
        auto prediction_work=model.workspace();size_t count=0;
        const auto result=FF::sample(posterior,start,parameters,o,joint,[&](const Field & white,const auto & p,size_t draw) {
            const auto global=canonical(white,model.layout());
            if(FML::ThisTask==0) {state<<draw;for(double x:global)state<<','<<x;for(double x:p)state<<','<<x;state<<'\n';}
            if(draw%stride==0) {
                model.predict(white,p,prediction_work);const auto & r=model.reconstruction(prediction_work);
                const auto initial=canonical(r.initial_density,model.layout()),matter=canonical(r.matter_density,model.observation_layout());
                if(FML::ThisTask==0) {predictive<<draw;for(double x:initial)predictive<<','<<x;for(double x:matter)predictive<<','<<x;predictive<<'\n';}
            }
            ++count;if(FML::ThisTask==0 && count%256==0)std::cout<<"RETAINED "<<count<<'/'<<draws<<std::endl;
        });
        require(count==size_t(draws) && result.completed_samples==draws && result.potential.size()==size_t(draws),"Lost retained draws");
        auto potential=output(prefix+".potential.csv"),meta=output(prefix+".meta.csv");
        if(FML::ThisTask==0) {
            potential<<"draw,potential\n";for(size_t i=0;i<result.potential.size();++i)potential<<i<<','<<result.potential[i]<<'\n';
            meta<<"joint,id,warmup,draws,leapfrog,stride,adapt,epsilon,acceptance,divergences,seed\n"<<joint<<','<<id<<','<<warm<<','<<draws<<','<<leap<<','<<stride<<','<<adapt<<','<<result.step_size<<','<<result.acceptance<<','<<result.divergences<<','<<o.seed<<'\n';
        }
        require(std::isfinite(result.acceptance),"Nonfinite sampler summary");
        return 0;
    } catch(const std::exception & e) {if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1;}
}
