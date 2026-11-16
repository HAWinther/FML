#ifndef FML_PM_STATIONARITY_STATE_H
#define FML_PM_STATIONARITY_STATE_H
#include <FML/FieldLevel/pm_nonlinear_validation/Validation.h>
#include <cstdint>
static inline int integer(const char * s){size_t end;int x=std::stoi(s,&end);if(end!=std::string(s).size())throw std::invalid_argument("Invalid integer");return x;}
static inline std::string state_identity(const Controls & c){return "PM-stationarity-v1,"+std::to_string(c.n)+","+std::to_string(c.m)+","+std::to_string(c.steps)+",compensated,Omega_m=.3,"+identity(c);}
static inline Field parse(const std::string & text){Field r;std::stringstream input(text);std::string v;while(std::getline(input,v,',')){size_t end;double x=std::stod(v,&end);if(end!=v.size()||!std::isfinite(x))throw std::invalid_argument("Invalid snapshot number");r.push_back(x);}return r;}
struct Snapshot {Field white,gradient,direction,unit_trial,parameters,meta;};
static inline Snapshot load(const Model & model,const Controls & c,const std::string & prefix) {
    Snapshot result;
    FML::COMMUNICATION::collective_try([&]{
        std::ifstream meta(prefix+".meta.csv");std::string line;
        if(!std::getline(meta,line)||line!=state_identity(c)||!std::getline(meta,line)||line!="shards,status,iterations,potential,raw_gradient,direction_source,seconds"||!std::getline(meta,line))throw std::invalid_argument("Wrong snapshot metadata");
        result.meta=parse(line);if(result.meta.size()!=7||result.meta[0]<1||result.meta[0]>64||result.meta[0]!=int(result.meta[0]))throw std::invalid_argument("Invalid snapshot metadata values");
        if(std::getline(meta,line))throw std::invalid_argument("Trailing snapshot metadata");
        std::ifstream params(prefix+".parameters.csv");if(!std::getline(params,line))throw std::invalid_argument("Missing snapshot parameters");result.parameters=parse(line);
        if(result.parameters!=model.initial_parameters()||std::getline(params,line))throw std::invalid_argument("Changed snapshot parameters");
        const size_t total=model.layout().global_cells;std::array<Field,4> values;for(auto & v:values)v.resize(total);std::vector<unsigned char> seen(total,0);
        for(int rank=0;rank<int(result.meta[0]);++rank){std::ifstream f(prefix+".rank"+std::to_string(rank)+".csv");if(!std::getline(f,line)||line!="id,white,gradient,direction,unit_trial")throw std::invalid_argument("Missing snapshot shard");
            while(std::getline(f,line)){const auto row=parse(line);if(row.size()!=5||row[0]<0||row[0]>=total||row[0]!=size_t(row[0])||seen[size_t(row[0])])throw std::invalid_argument("Duplicate/invalid snapshot cell");const auto id=size_t(row[0]);seen[id]=1;for(int j=0;j<4;++j)values[j][id]=row[j+1];}}
        if(std::count(seen.begin(),seen.end(),0))throw std::invalid_argument("Missing snapshot cells");
        result.white.resize(model.cells());result.gradient.resize(model.cells());result.direction.resize(model.cells());result.unit_trial.resize(model.cells());
        for(size_t q=0;q<model.cells();++q){const auto id=model.layout().global_cell_id(q);result.white[q]=values[0][id];result.gradient[q]=values[1][id];result.direction[q]=values[2][id];result.unit_trial[q]=values[3][id];}
    },"Read full PM snapshot");return result;
}
#endif
