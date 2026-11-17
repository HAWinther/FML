#include <FML/MCMC/Diagnostics.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
static std::vector<std::string> split(const std::string & line) {
    std::stringstream s(line);std::string cell;std::vector<std::string> out;while(std::getline(s,cell,','))out.push_back(cell);return out;
}
int main(int argc,char ** argv) {
    try {
        if(argc<4)throw std::invalid_argument("pm_full_diagnostics OUTPUT FEATURES1 FEATURES2 [...]");
        std::vector<std::string> header;std::vector<std::vector<std::vector<double>>> columns;
        for(int i=2;i<argc;++i) {
            std::ifstream file(argv[i]);if(!file)throw std::runtime_error("Cannot open features");std::string line;std::getline(file,line);
            auto h=split(line);if(i==2) {header=h;columns.resize(h.size()-1);}else if(h!=header)throw std::invalid_argument("Feature schemas differ");
            if(h.empty()||h[0]!="draw")throw std::invalid_argument("Missing draw column");
            std::vector<std::vector<double>> chain(columns.size());size_t row=0;
            while(std::getline(file,line)) {
                auto cells=split(line);if(cells.size()!=header.size()||std::stod(cells[0])!=double(row++))throw std::invalid_argument("Invalid feature row");
                for(size_t j=1;j<cells.size();++j) {size_t used;double x=std::stod(cells[j],&used);if(used!=cells[j].size()||!std::isfinite(x))throw std::invalid_argument("Invalid feature");chain[j-1].push_back(x);}
            }
            for(size_t j=0;j<columns.size();++j)columns[j].push_back(std::move(chain[j]));
        }
        std::ofstream out(argv[1]);if(!out)throw std::runtime_error("Cannot open diagnostics output");
        out<<std::setprecision(17)<<"feature,status,rhat,ess_bulk,ess_tail,mean,variance\n";
        for(size_t j=0;j<columns.size();++j) {
            auto d=FML::MCMC::diagnose_chains(columns[j]);double mean=0,var=0;size_t n=0;
            for(auto & c:columns[j])for(double x:c){mean+=x;++n;}mean/=n;
            for(auto & c:columns[j])for(double x:c)var+=(x-mean)*(x-mean);var/=n;
            out<<header[j+1]<<','<<FML::MCMC::diagnostic_status_name(d.status)<<','<<d.rhat<<','<<d.ess_bulk<<','<<d.ess_tail<<','<<mean<<','<<var<<'\n';
        }
        return 0;
    }catch(const std::exception & e){std::cerr<<e.what()<<'\n';return 1;}
}
