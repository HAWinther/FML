#define FML_FIELDLEVEL_CHECKPOINT_TEST_FAILURE
#include <FML/FieldLevel/ParticleMeshCheckpoint.h>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
#include <FML/tests/TestHelpers.h>
#include <iostream>
namespace FF = FML::FIELDLEVEL::FRAMEWORK;
namespace CP = FF::CHECKPOINT;
namespace C = FML::MCMC::CHECKPOINT;
namespace FS = std::filesystem;
using FML::FIELDLEVEL::Field;
using FML::TESTS::check;
struct Reference {
    FML::MCMC::HMCResult result;
    std::vector<Field> fields;
};
static void write_reference(const FS::path & file, const Reference & r) {
    C::Encoder e;
    e.vector(r.result.last); e.vector(r.result.diagonal); e.vector(r.result.potential);
    e.integer(r.result.recorded.size()); for (const auto & x : r.result.recorded) e.vector(x);
    e.integer(r.fields.size()); for (const auto & x : r.fields) e.vector(x);
    e.real(r.result.acceptance); e.real(r.result.step_size); e.integer(r.result.divergences);
    e.integer(r.result.completed_iterations); e.integer(r.result.completed_samples);
    C::write_file(file.string(), e.bytes);
}
static Reference read_reference(const FS::path & file, size_t cells, size_t np, int draws, bool joint) {
    const size_t n = cells + (joint ? np : 0);
    const auto bytes = C::read_file(file.string(), C::size_bound(n,np) + size_t(draws)*(cells+np+1)*16);
    C::Decoder d{bytes}; Reference r;
    r.result.last=d.vector(n);r.result.diagonal=d.vector(joint?np:0);r.result.potential=d.vector(draws);
    if(d.size(draws)!=size_t(draws))throw std::runtime_error("Reference row count");
    for(int i=0;i<draws;++i)r.result.recorded.push_back(d.vector(np));
    if(d.size(draws)!=size_t(draws))throw std::runtime_error("Reference field count");
    for(int i=0;i<draws;++i)r.fields.push_back(d.vector(cells));
    r.result.acceptance=d.real();r.result.step_size=d.real();r.result.divergences=long(d.size(LONG_MAX));
    r.result.completed_iterations=int(d.size(INT_MAX));r.result.completed_samples=int(d.size(INT_MAX));d.end();return r;
}
static bool same_summary(const FML::MCMC::HMCResult & a, const FML::MCMC::HMCResult & b) {
    return a.last==b.last && a.diagonal==b.diagonal && a.acceptance==b.acceptance &&
        a.step_size==b.step_size && a.divergences==b.divergences &&
        a.completed_iterations==b.completed_iterations && a.completed_samples==b.completed_samples;
}
static FF::Configuration configuration() {
    FF::Configuration c;c.mesh=4;c.density_mesh=8;c.boxsize=1.;c.evolution=FF::Evolution::Linear;
    c.initial_cutoff=c.bias_cutoff=7.;c.likelihood_cutoff=15.;return c;
}
static FF::Spectrum spectrum(double power=.02, double prior_sigma=.5) {
    return {{{"log_amplitude",0.,0.,prior_sigma,-1.,1.}},[power](double,const auto & p){return FF::Power{power*std::exp(p[0]),{1.}};}};
}
static FF::PMSettings settings() {FF::PMSettings s;s.scale_factors={.2,.4,.6};return s;}
static FF::Parameter noise() {const double x=std::log(.001);return {"log_noise_power",x,x,.5,-14.,-3.};}
static FML::MCMC::HMCOptions options(bool joint, bool dense=false) {
    FML::MCMC::HMCOptions o;o.warmup=joint?(dense?16:120):6;o.samples=joint?6:4;o.n_leapfrog=2;
    o.step_size=.025;o.seed=(std::uint64_t(1)<<40)+37;o.target_acceptance=.85;
    o.adapt_step_size=joint;o.adapt_mass=joint && !dense;return o;
}
template<class Grid>
void exercise(const FS::path & root,const std::string & phase,const FML::Domain & domain,bool dense=false, int force_window=0, int reference=0, int assignment=2) {
    using Model=FF::PeriodicPMModelT<Grid>;
    auto c=configuration();if(dense){c.mesh=8;c.density_mesh=4;c.likelihood_cutoff=9.;}if(dense&&reference)c.density_mesh=8;FF::PMObservationSettings observed;observed.reference_cic_mesh=reference;auto pm=settings();pm.assignment=static_cast<FF::PMAssignment>(assignment);pm.force_window=static_cast<FF::PMForceWindow>(force_window);const auto prior=noise();
    Model model(c,spectrum(),pm,prior,{2},domain,observed);auto work=model.workspace();const auto parameters=model.initial_parameters();
    const auto data=model.mock(model.white_noise(91),parameters,92,work);FF::BasicPosterior<Model> posterior(model,data);
    for(bool joint:{true,false}) {
        const std::string name=joint?"joint":"fixed";const auto o=options(joint,dense);
        FF::CheckpointOptions io;io.directory=(root/name).string();io.problem_id="PM-checkpoint-fixture-1:P0.02:mock91:noise92";io.every=17;
        const auto file=root/(name+".reference.rank"+std::to_string(FML::ThisTask));
        if(phase=="reject") {
            io.resume=true;io.max_iterations=0;
            check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,io,joint);}),name+" rejects changed ranks/backend/domain/thread configuration");
            continue;
        }
        if(phase=="prepare") {
            Reference reference;
            reference.result=FF::sample(posterior,model.white_noise(300),parameters,o,joint,[&](const auto & w,const auto &,size_t){reference.fields.push_back(w);});
            FML::COMMUNICATION::collective_try([&]{write_reference(file,reference);},"PM checkpoint reference write");
            std::vector<Field> fields;std::vector<std::vector<double>> rows;Field potentials;std::vector<size_t> indices;
            for(int budget:(joint?(dense?std::vector<int>{5,7,6}:std::vector<int>{27,33,55,7}):std::vector<int>{3})) {
                io.max_iterations=budget;
                auto sink=[&](const auto & white,const auto &,size_t draw){fields.push_back(white);indices.push_back(draw);};
                auto part=io.resume?FF::sample_checkpointed(posterior,{}, {},o,io,joint,sink):
                    FF::sample_checkpointed(posterior,model.white_noise(300),parameters,o,io,joint,sink);
                rows.insert(rows.end(),part.recorded.begin(),part.recorded.end());potentials.insert(potentials.end(),part.potential.begin(),part.potential.end());io.resume=true;
            }
            const size_t n=joint?2:0;
            check(rows==std::vector<std::vector<double>>(reference.result.recorded.begin(),reference.result.recorded.begin()+n) &&
                    potentials==Field(reference.result.potential.begin(),reference.result.potential.begin()+n) &&
                    fields==std::vector<Field>(reference.fields.begin(),reference.fields.begin()+n) && indices==(joint?std::vector<size_t>{0,1}:std::vector<size_t>{}),
                    name+" warmup/window/draw splits preserve full fields, rows, potentials and sink indices bitwise");
            const auto state=CP::load(io.directory,CP::identity(posterior,io.problem_id),model.cells(),parameters.size(),joint);
            check(state.frozen==parameters && state.sampler.next_iteration==(joint?(dense?18:122):3),name+" frozen parameters and iteration retained");
            if(joint)check(state.sampler.window==(dense?0:2) && state.sampler.diagonal.size()==2,name+" scalar mass-adaptation windows and diagonal retained");
            continue;
        }
        if(phase!="resume")throw std::invalid_argument("Expected prepare, resume or reject");
        Reference reference;FML::COMMUNICATION::collective_try([&]{reference=read_reference(file,model.cells(),parameters.size(),o.samples,joint);},"PM reference read");
        io.resume=true;std::vector<Field> fields;std::vector<size_t> indices;
        const auto suffix=FF::sample_checkpointed(posterior,{}, {},o,io,joint,[&](const auto & white,const auto &,size_t draw){fields.push_back(white);indices.push_back(draw);});
        const size_t start=joint?2:0;std::vector<size_t> expected;for(size_t i=start;i<size_t(o.samples);++i)expected.push_back(i);
        check(same_summary(suffix,reference.result) && suffix.recorded==std::vector<std::vector<double>>(reference.result.recorded.begin()+start,reference.result.recorded.end()) &&
            suffix.potential==Field(reference.result.potential.begin()+start,reference.result.potential.end()) &&
            fields==std::vector<Field>(reference.fields.begin()+start,reference.fields.end()) && indices==expected,
            name+" fresh-process resume matches fields, scalars, adaptation, acceptance, divergences and sink indices bitwise");
        io.max_iterations=0;
        const auto complete=FF::sample_checkpointed(posterior,{}, {},o,io,joint);
        check(same_summary(complete,reference.result) && complete.recorded.empty(),name+" completed zero-budget resume is a no-op");
        auto reject=[&](const auto & target,const std::string & reason){check(FML::TESTS::throws([&]{FF::sample_checkpointed(target,{}, {},o,io,joint);}),name+" rejects "+reason);};
        auto fingerprint_reject=[&](FF::Configuration config,FF::Spectrum power,FF::PMSettings physics,FF::Parameter n,FF::PMAdjointOptions reverse,const std::string & reason) {
            Model changed(config,std::move(power),std::move(physics),std::move(n),reverse,domain,observed);
            FF::BasicPosterior<Model> target(changed,data);reject(target,reason);
        };
        auto window_pm=pm;window_pm.force_window=pm.force_window==FF::PMForceWindow::None?(pm.assignment==FF::PMAssignment::TSC?FF::PMForceWindow::TSCCompensated:FF::PMForceWindow::CICCompensated):FF::PMForceWindow::None;
        fingerprint_reject(c,spectrum(),window_pm,prior,{2},"force-window convention");
        auto assignment_pm=pm;assignment_pm.assignment=pm.assignment==FF::PMAssignment::CIC?FF::PMAssignment::TSC:FF::PMAssignment::CIC;assignment_pm.force_window=pm.force_window==FF::PMForceWindow::None?FF::PMForceWindow::None:(assignment_pm.assignment==FF::PMAssignment::TSC?FF::PMForceWindow::TSCCompensated:FF::PMForceWindow::CICCompensated);
        fingerprint_reject(c,spectrum(),assignment_pm,prior,{2},"assignment convention");
        auto changed_pm=pm;changed_pm.background.omega_m=.31;fingerprint_reject(c,spectrum(),changed_pm,prior,{2},"background");
        changed_pm=pm;changed_pm.scale_factors[1]=.41;fingerprint_reject(c,spectrum(),changed_pm,prior,{2},"interior time node at fixed endpoints");
        changed_pm=pm;changed_pm.scale_factors={.2,.3,.4,.6};fingerprint_reject(c,spectrum(),changed_pm,prior,{2},"time-step count");
        fingerprint_reject(c,spectrum(),pm,prior,{1},"adjoint replay budget");
        auto changed_noise=prior;changed_noise.mean+=.01;fingerprint_reject(c,spectrum(),pm,changed_noise,{2},"noise prior");
        fingerprint_reject(c,spectrum(.02,.6),pm,prior,{2},"spectrum prior");
        fingerprint_reject(c,spectrum(.022),pm,prior,{2},"unversioned spectrum change by restored force/potential validation");
        auto modified=data;for(size_t i=0;i<modified.size();++i)if(FML::ThisTask==0 && (model.modes()[i].n[0]==0 && model.modes()[i].n[1]==0 && model.modes()[i].n[2]==1))modified[i]+=.001;
        FF::BasicPosterior<Model> different_data(model,modified);reject(different_data,"one-rank data change collectively");
        auto wrong=io;wrong.problem_id+="-changed";
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,wrong,joint);}),name+" rejects changed callback/problem version");
        auto changed_options=o;++changed_options.n_leapfrog;
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},changed_options,io,joint);}),name+" rejects changed HMC schedule");
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,io,!joint);}),name+" rejects changed fixed/joint mode");
        const auto generation=CP::current_generation(io.directory);
        if(FML::ThisTask==0)CP::fail_shard_write_once=true;
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,io,joint);}),name+" one-rank write failure propagates collectively");
        check(CP::current_generation(io.directory)==generation,name+" failed write preserves published generation");
        const auto original=CP::load(io.directory,CP::identity(posterior,io.problem_id),model.cells(),parameters.size(),joint);
        auto poisoned=original;if(FML::ThisTask==0)poisoned.sampler.gradient[0]+=1;CP::save(io.directory,poisoned);
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,io,joint);}),name+" rejects rank-local stored force mismatch");CP::save(io.directory,original);
        const auto shard=CP::shard(io.directory,CP::current_generation(io.directory));
        FML::COMMUNICATION::collective_try([&]{if(FML::ThisTask==0){FS::copy_file(shard,shard.string()+".backup",FS::copy_options::overwrite_existing);FS::resize_file(shard,13);}},"PM truncated shard fixture");
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,io,joint);}),name+" rejects truncated shard collectively");
        FML::COMMUNICATION::collective_try([&]{if(FML::ThisTask==0){FS::copy_file(shard.string()+".backup",shard,FS::copy_options::overwrite_existing);FS::rename(shard,shard.string()+".missing");}},"PM missing shard fixture");
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(posterior,{}, {},o,io,joint);}),name+" rejects missing shard collectively");
        FML::COMMUNICATION::collective_try([&]{if(FML::ThisTask==0){FS::rename(shard.string()+".missing",shard);FS::remove(shard.string()+".backup");}},"PM shard restore");
        check(FF::sample_checkpointed(posterior,{}, {},o,io,joint).last==reference.result.last,name+" restored checkpoint remains usable after failures");
    }
    // At zero white and zero data, changing Omega_m leaves the saved U/gradient
    // unchanged. Rejection must therefore come from the explicit PM fingerprint.
    if(phase=="resume") {
        FF::BasicPosterior<Model> zero(model,FF::Modes(model.modes().size(),0.));auto o=options(false);
        FF::CheckpointOptions io;io.directory=(root/"zero").string();io.problem_id="zero-PM-1";io.max_iterations=0;
        FF::sample_checkpointed(zero,Field(model.cells(),0.),parameters,o,io,false);
        auto changed_pm=pm;changed_pm.background.omega_m=.4;Model changed(c,spectrum(),changed_pm,prior,{2},domain,observed);
        FF::BasicPosterior<Model> target(changed,FF::Modes(changed.modes().size(),0.));auto w1=model.workspace(),w2=changed.workspace();Field g1,g2;std::vector<double> p1,p2;
        const double u1=zero.evaluate(Field(model.cells(),0.),parameters,w1,g1,p1),u2=target.evaluate(Field(model.cells(),0.),parameters,w2,g2,p2);
        check(u1==u2 && g1==g2 && p1==p2,"zero-state background change is invisible to force/potential check");
        io.resume=true;check(FML::TESTS::throws([&]{FF::sample_checkpointed(target,{}, {},o,io,false);}),"explicit PM identity rejects background change at indistinguishable saved state");
        auto window_pm=pm;window_pm.force_window=pm.force_window==FF::PMForceWindow::None?(pm.assignment==FF::PMAssignment::TSC?FF::PMForceWindow::TSCCompensated:FF::PMForceWindow::CICCompensated):FF::PMForceWindow::None;
        Model window_model(c,spectrum(),window_pm,prior,{2},domain,observed);
        FF::BasicPosterior<Model> window_target(window_model,FF::Modes(window_model.modes().size(),0.));
        auto window_work=window_model.workspace();Field wg;std::vector<double> wp;
        const double wu=window_target.evaluate(Field(model.cells(),0.),parameters,window_work,wg,wp);
        check(wu==u1 && wg==g1 && wp==p1,"force window is invisible at zero saved state");
        check(CP::identity(window_target,io.problem_id)!=CP::identity(zero,io.problem_id),"force window changes identity at indistinguishable saved state");
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(window_target,{}, {},o,io,false);}),"force window rejects zero-state checkpoint");
        auto other_assignment=pm;other_assignment.assignment=pm.assignment==FF::PMAssignment::CIC?FF::PMAssignment::TSC:FF::PMAssignment::CIC;
        other_assignment.force_window=pm.force_window==FF::PMForceWindow::None?FF::PMForceWindow::None:(other_assignment.assignment==FF::PMAssignment::TSC?FF::PMForceWindow::TSCCompensated:FF::PMForceWindow::CICCompensated);
        Model assignment_model(c,spectrum(),other_assignment,prior,{2},domain,observed);
        FF::BasicPosterior<Model> assignment_target(assignment_model,FF::Modes(assignment_model.modes().size(),0.));
        auto assignment_work=assignment_model.workspace();Field ag;std::vector<double> ap;
        const double au=assignment_target.evaluate(Field(model.cells(),0.),parameters,assignment_work,ag,ap);
        check(au==u1 && ag==g1 && ap==p1,"assignment change is invisible at zero saved state");
        check(CP::identity(assignment_target,io.problem_id)!=CP::identity(zero,io.problem_id),"assignment changes explicit identity at indistinguishable saved state");
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(assignment_target,{}, {},o,io,false);}),"assignment rejects zero-state checkpoint");
        auto altered_response=observed;
        altered_response.response.multiplier.assign(model.modes().size(),.8);
        Model response_model(c,spectrum(),pm,prior,{2},domain,altered_response);
        FF::BasicPosterior<Model> response_target(response_model,FF::Modes(response_model.modes().size(),0.));
        auto response_work=response_model.workspace();Field rg;std::vector<double> rp;
        double ru=response_target.evaluate(Field(model.cells(),0.),parameters,response_work,rg,rp);
        check(ru==u1 && rg==g1 && rp==p1,"observation response is invisible at zero saved state");
        int response_changed=CP::identity(response_target,io.problem_id)!=CP::identity(zero,io.problem_id);
        FML::MaxOverTasks(&response_changed);
        check(response_changed!=0,"cached response coefficients change distributed PM identity");
        check(FML::TESTS::throws([&]{FF::sample_checkpointed(response_target,{}, {},o,io,false);}),"changed response rejects zero-state checkpoint");
        auto reject_config=[&](FF::Configuration config,const std::string & label) {
            Model other(config,spectrum(),pm,prior,{2},domain,observed);
            FF::BasicPosterior<Model> different(other,FF::Modes(other.modes().size(),0.));
            check(CP::identity(different,io.problem_id)!=CP::identity(zero,io.problem_id),label+" changes PM identity");
            check(FML::TESTS::throws([&]{FF::sample_checkpointed(different,{}, {},o,io,false);}),"rejects "+label);
        };
        auto altered=c;altered.mesh=2*c.mesh;reject_config(altered,"IC mesh");
        altered=c;altered.density_mesh=16;reject_config(altered,"force/observation mesh");
        altered=c;altered.boxsize=1.1;reject_config(altered,"box length");
        altered=c;altered.initial_cutoff=8.;reject_config(altered,"initial cutoff");
        altered=c;altered.likelihood_cutoff=.9*c.likelihood_cutoff;reject_config(altered,"likelihood cutoff");

    }
}
#ifdef USE_MPI
static std::vector<double> edges(int parts,bool alternate) {
    std::vector<double> x(parts+1);for(int i=0;i<=parts;++i)x[i]=double(i)/parts;
    if(parts==2 && !alternate)x[1]=.25;return x;
}
#endif
int main(int argc,char ** argv) {
    FML::FMLSetup::init(&argc,&argv);
    try {
        std::string directory,phase,backend;int alternate=0,dense=0,force_window=0,reference=0,assignment=2;
        FML::COMMUNICATION::collective_try([&]{if(argc<4 || argc>9)throw std::invalid_argument("test_particle_mesh_checkpoint DIRECTORY prepare|resume|reject slab|pencil [alternate] [dense] [force_window] [reference_cic_mesh] [assignment]");directory=argv[1];phase=argv[2];backend=argv[3];if(argc>=5)alternate=std::stoi(argv[4]);if(argc>=6)dense=std::stoi(argv[5]);if(argc>=7)force_window=std::stoi(argv[6]);if(argc>=8)reference=std::stoi(argv[7]);if(argc==9)assignment=std::stoi(argv[8]);},"PM checkpoint test input");
        CP::same_string(directory);CP::same_string(phase);CP::same_string(backend);CP::same_string(std::to_string(alternate));CP::same_string(std::to_string(dense));CP::same_string(std::to_string(force_window));CP::same_string(std::to_string(reference));CP::same_string(std::to_string(assignment));
        FML::COMMUNICATION::collective_try([&]{if(FML::ThisTask==0)FS::create_directories(directory);},"PM test directory");
        if(backend=="slab")exercise<FML::GRID::FFTWGrid<3>>(directory,phase,FML::domain,dense!=0,force_window,reference,assignment);
#ifdef USE_MPI
        else if(backend=="pencil") {
            int dims[2]={0,0};MPI_Dims_create(FML::NTasks,2,dims);if(alternate==2){dims[0]=FML::NTasks;dims[1]=1;}
            const auto domain=FML::Domain::pencil(edges(dims[0],alternate!=0),edges(dims[1],alternate!=0));
            exercise<FML::GRID::PencilGrid<3>>(directory,phase,domain,dense!=0,force_window,reference,assignment);
        }
#endif
        else throw std::invalid_argument("Unsupported backend");
    }catch(const std::exception & e){if(FML::ThisTask==0)std::cerr<<e.what()<<'\n';return 1;}
    return FML::TESTS::summary("PM checkpoint");
}
