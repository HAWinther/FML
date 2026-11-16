#include <FML/FieldLevel/ParticleMesh.h>
#include <FML/tests/TestHelpers.h>
#include <limits>
#ifdef USE_MPI
#include <FML/PencilDomain/PencilGrid.h>
#endif
namespace FF=FML::FIELDLEVEL::FRAMEWORK;
using FML::TESTS::check;
template<class Grid>static void exercise(const FML::Domain & domain){
    FF::Configuration c;c.mesh=c.density_mesh=8;c.boxsize=1;c.evolution=FF::Evolution::Linear;c.initial_cutoff=c.bias_cutoff=10.;c.likelihood_cutoff=15.;
    for(auto window:{FF::PMForceWindow::None,FF::PMForceWindow::CICCompensated}){
        FF::PMSettings pm;pm.scale_factors={.2,.35,.5};pm.force_window=window;
        FF::ParticleMeshForwardT<Grid> model(c,FF::Spectrum::fixed([](double){return .001;}),pm,domain);
        auto work=model.workspace();auto white=model.white_noise(17);const auto p=model.initial_parameters();const auto baseline=model.evolve(white,p,work);size_t visits=0;bool valid=true;
        const auto observed=model.evolve_observed(white,p,work,[&](size_t node,const auto & state){valid &= node==visits++ && state.displacement[0].size()==model.cells();
            for(size_t q=0;q<model.cells();++q)for(double x:state.position(q))valid &= std::isfinite(x)&&x>=0&&x<=1;});
        check(valid&&visits==pm.scale_factors.size(),"observer visits every force position node in order");
        check(observed.initial_density==baseline.initial_density&&observed.matter_density==baseline.matter_density&&observed.displacement==baseline.displacement&&observed.momentum==baseline.momentum,"observer preserves all dynamical outputs bitwise");
        bool caught=false;try{model.evolve_observed(white,p,work,[&](size_t node,const auto &){if(FML::ThisTask==0&&node==1)throw std::runtime_error("test observer failure");});}catch(const std::exception&){caught=true;}check(caught,"one-rank observer exception propagates collectively");
        caught=false;try{work.result();}catch(const std::logic_error&){caught=true;}check(caught,"failed observer invalidates workspace output");
        check(model.evolve(white,p,work).displacement==baseline.displacement,"default evolution recovers after observer failure");
        auto state=baseline;for(auto & field:state.displacement)std::fill(field.begin(),field.end(),0.);
        bool canonical=true;for(size_t q=0;q<model.cells();++q)if(model.layout().global_cell_id(q)==0){state.displacement[0][q]=-std::numeric_limits<double>::denorm_min();double u=state.position(q)[0]*c.density_mesh;canonical &= int(std::floor(u))%c.density_mesh==0;}
        check(canonical,"trace canonicalizes the rounded periodic endpoint to cell zero");
    }
}
int main(int argc,char **argv){FML::FMLSetup::init(&argc,&argv);auto slab=FML::Domain::uniform_slab(FML::NTasks);FML::set_domain(slab);exercise<FML::GRID::FFTWGrid<3>>(slab);
#ifdef USE_MPI
    auto pencil=FML::Domain::uniform_pencil(FML::NTasks==4?2:FML::NTasks,FML::NTasks==4?2:1);FML::set_domain(pencil);exercise<FML::GRID::PencilGrid<3>>(pencil);
#endif
    return FML::TESTS::summary("PM read-only observer");}
