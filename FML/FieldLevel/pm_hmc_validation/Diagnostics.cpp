#include <FML/MCMC/Diagnostics.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
int main(int argc, char ** argv) {
    try {
        if (argc < 4) throw std::invalid_argument("pm_hmc_diagnostics OUTPUT CHAIN1 CHAIN2 [...]");
        std::vector<std::vector<std::vector<double>>> columns(6);
        for (int i=2; i<argc; ++i) {
            std::ifstream file(argv[i]); if(!file)throw std::runtime_error("Cannot read chain");
            std::string line;std::getline(file,line);if(line!="draw,z0,z1,potential")throw std::invalid_argument("Unexpected chain schema");
            std::vector<std::vector<double>> chain(6);
            while(std::getline(file,line)) {
                std::stringstream row(line);std::string cell;std::vector<double> x;
                while(std::getline(row,cell,','))x.push_back(std::stod(cell));
                if(x.size()!=4)throw std::invalid_argument("Truncated chain row");
                const double a=x[1],b=x[2];const double values[6]={a,b,a*a,b*b,a*b,x[3]};
                for(int j=0;j<6;++j){if(!std::isfinite(values[j]))throw std::invalid_argument("Nonfinite chain");chain[j].push_back(values[j]);}
            }
            for(int j=0;j<6;++j)columns[j].push_back(std::move(chain[j]));
        }
        const char * names[6]={"z0","z1","z0_squared","z1_squared","cross","potential"};
        std::ofstream out(argv[1]);if(!out)throw std::runtime_error("Cannot write diagnostics");
        out<<std::setprecision(17)<<"feature,status,rhat,ess_bulk,ess_tail,mean\n";
        for(int j=0;j<6;++j) {
            const auto d=FML::MCMC::diagnose_chains(columns[j]);double mean=0;size_t n=0;
            for(const auto & chain:columns[j])for(double x:chain){mean+=x;++n;}
            out<<names[j]<<','<<FML::MCMC::diagnostic_status_name(d.status)<<','<<d.rhat<<','<<d.ess_bulk<<','<<d.ess_tail<<','<<mean/n<<'\n';
        }
        return 0;
    } catch(const std::exception & e) {std::cerr<<e.what()<<'\n';return 1;}
}
