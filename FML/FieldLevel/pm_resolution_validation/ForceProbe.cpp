#include <FML/FieldLevel/ParticleMeshPosterior.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
namespace FF=FML::FIELDLEVEL::FRAMEWORK;
using FML::FIELDLEVEL::Field;
struct Wave {int x,y,z;};
int main(int argc,char ** argv){
    FML::FMLSetup::init(&argc,&argv);
    try {
        FML::COMMUNICATION::require_count(FML::NTasks==1,"Force probe requires one rank");
        if(argc!=3 && argc!=4)throw std::invalid_argument("pm_force_probe INITIAL_COEFFICIENTS.csv OUTPUT.csv [FORCE_WINDOW]");
        int window=0;if(argc==4){size_t used;window=std::stoi(argv[3],&used);if(used!=std::string(argv[3]).size()||(window!=0&&window!=1))throw std::invalid_argument("Invalid force window");}
        std::ifstream input(argv[1]);std::string line,cell;std::getline(input,line);std::stringstream row(line);Field coefficients;
        while(std::getline(row,cell,','))coefficients.push_back(std::stod(cell));
        if(coefficients.size()!=18)throw std::invalid_argument("Expected 18 initial coefficients");
        std::vector<Wave> waves;for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z)
            if(0<x*x+y*y+z*z && x*x+y*y+z*z<=2 && (x>0 || (x==0&&y>0) || (x==0&&y==0&&z>0)))waves.push_back({x,y,z});
        std::ofstream output(argv[2]);if(!output)throw std::runtime_error("Cannot open force probe output");
        output<<std::setprecision(17)<<"n,m,amplitude,delta_a,longitudinal_response,relative_force_error\n";
        const double a0=1e-4;
        for(const auto & grids:std::vector<std::pair<int,int>>{{8,8},{8,16},{8,32},{8,64},{4,32},{16,32},{32,32},{64,32},{32,16},{64,16},{64,8}})for(double da:(window?std::vector<double>{1e-8,1e-9,1e-10}:std::vector<double>{1e-8,1e-9})){
            const int n=grids.first,m=grids.second;FF::Configuration c;c.mesh=n;c.density_mesh=m;c.boxsize=1.;c.evolution=FF::Evolution::Linear;c.initial_cutoff=c.bias_cutoff=10.;c.likelihood_cutoff=15.;
            FF::PMSettings settings;settings.force_window=static_cast<FF::PMForceWindow>(window);settings.background.omega_m=1.;settings.scale_factors={a0,a0+da};
            FF::PeriodicPMModel model(c,FF::Spectrum::fixed([](double){return .002;}),settings);auto work=model.workspace();
            Field white(model.cells(),0.);std::array<Field,3> psi;for(auto & x:psi)x.assign(model.cells(),0.);
            for(size_t q=0;q<white.size();++q){const auto xyz=model.layout().global_coordinate(q);size_t j=0;
                for(const auto & k:waves){const double phase=2*M_PI*(k.x*xyz[0]+k.y*xyz[1]+k.z*xyz[2])/n;const double cc=std::sqrt(2.)*coefficients[j++],ss=std::sqrt(2.)*coefficients[j++];
                    white[q]+=(cc*std::cos(phase)+ss*std::sin(phase))/std::sqrt(.002*model.layout().global_cells);
                    const double value=(-cc*std::sin(phase)+ss*std::cos(phase))/(2*M_PI*(k.x*k.x+k.y*k.y+k.z*k.z));
                    psi[0][q]+=value*k.x;psi[1][q]+=value*k.y;psi[2][q]+=value*k.z;
                }
            }
            model.predict(white,model.initial_parameters(),work);const auto & fields=model.reconstruction(work);
            const double kick=settings.background.kick(a0,a0+da),amplitude=a0+.5*da;double dot=0,norm=0,error=0;
            for(int d=0;d<3;++d)for(size_t q=0;q<white.size();++q){
                const double force=(fields.momentum[d][q]-std::pow(a0,1.5)*psi[d][q])/kick;
                dot+=force*psi[d][q];norm+=psi[d][q]*psi[d][q];error+=std::pow(force-amplitude*psi[d][q],2);
            }
            output<<n<<','<<m<<','<<a0<<','<<da<<','<<dot/(amplitude*norm)<<','<<std::sqrt(error/(amplitude*amplitude*norm))<<'\n';
        }
        if(!output)throw std::runtime_error("Cannot finish force probe output");
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}return 0;
}
