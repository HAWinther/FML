#include <FML/FieldLevel/ParticleMeshCheckpoint.h>
#include <FML/FieldLevel/Frontend.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
int main(int argc,char ** argv) {
    FML::FMLSetup::init(&argc,&argv);
    try {
        FF::CheckpointOptions io;io.max_iterations=40;io.every=25;
        FML::COMMUNICATION::collective_try([&] {
            if(argc<2 || argc>4)throw std::invalid_argument("Usage: particle_mesh_checkpoint DIRECTORY [start|resume] [trajectory_budget]");
            io.directory=argv[1];
            if(argc>=3) {
                const std::string phase=argv[2];
                if(phase!="start" && phase!="resume")throw std::invalid_argument("Expected start or resume");
                io.resume=phase=="resume";
            }
            if(argc==4) {
                size_t n;io.max_iterations=std::stoi(argv[3],&n);
                if(n!=std::string(argv[3]).size() || io.max_iterations<0)throw std::invalid_argument("Invalid trajectory budget");
            }
        },"PM checkpoint example input");
        FML::FIELDLEVEL::FRONTEND::initialize(4,8);
        FF::Configuration c;c.mesh=4;c.density_mesh=8;c.boxsize=1.;c.evolution=FF::Evolution::Linear;
        c.initial_cutoff=c.bias_cutoff=7.;c.likelihood_cutoff=15.;
        FF::Spectrum spectrum{{{"log_amplitude",0.,0.,.5,-1.,1.}},
            [](double,const auto & p){return FF::Power{.02*std::exp(p[0]),{1.}};}};
        FF::PMSettings settings;settings.scale_factors={.2,.4,.6};
        const double log_noise=std::log(.001);
        using Model=FF::PeriodicPMModelT<FML::FIELDLEVEL::FRONTEND::Grid>;
        Model model(c,spectrum,settings,{"log_noise_power",log_noise,log_noise,.5,-14.,-3.},{2});
        auto work=model.workspace();const auto parameters=model.initial_parameters();
        const auto data=model.mock(model.white_noise(91),parameters,92,work);
        FF::BasicPosterior<Model> posterior(model,data);
        // Version callback physics, mock inputs and relevant source/build changes.
        // std::function callbacks cannot be fingerprinted automatically.
        io.problem_id="FML-PM-example-1:P0.02-a1:mock91:noise92:CIC-KDK-v1";
        FML::MCMC::HMCOptions options;options.warmup=120;options.samples=300;
        options.n_leapfrog=4;options.step_size=.025;options.seed=13;
        const auto result=io.resume?FF::sample_checkpointed(posterior,{}, {},options,io):
            FF::sample_checkpointed(posterior,model.white_noise(300),parameters,options,io);
        FML::FIELDLEVEL::Field white(result.last.begin(),result.last.begin()+model.cells());
        const std::vector<double> current(result.last.begin()+model.cells(),result.last.end());
        model.predict(white,current,work);const auto & fields=model.reconstruction(work);
        double norm=0;for(double x:fields.matter_density)norm+=x*x;
        const double rms=std::sqrt(model.sum(norm)/model.observation_layout().global_cells);
        if(FML::ThisTask==0)std::cout<<"Completed trajectories: "<<result.completed_iterations<<'/'<<options.warmup+options.samples
            <<"\nCompleted draws: "<<result.completed_samples<<"; new rows this invocation: "<<result.recorded.size()
            <<"\nCumulative acceptance: "<<result.acceptance<<"; divergences: "<<result.divergences
            <<"\nCurrent final matter RMS: "<<rms<<'\n';
    }catch(const std::exception & e){if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1;}
    return 0;
}
