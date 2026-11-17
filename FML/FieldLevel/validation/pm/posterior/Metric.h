#ifndef FML_TSC_METRIC_H
#define FML_TSC_METRIC_H
#include <FML/FieldLevel/validation/pm/posterior/Target.h>
#include <FML/FieldLevel/validation/pm/posterior/FixedMetric.h>
// Freeze a regularized Hessian at the common physical reference field, A=0,N=N0.
// Finite differences are setup only: the sampler still uses the exact target.
inline void build_metric(CenteredTarget & target,Field reference,const std::string & prefix,double h) {
    const auto begin=std::chrono::steady_clock::now();const size_t n=target.basis.dimension;
    require(reference.size()==n&&h>=1e-7&&h<=.001,"Invalid metric reference or finite-difference step");
    reference.push_back(0.);reference.push_back(std::log(noise_power));Field matrix(n*n);
    for(size_t j=0;j<n;++j){auto plus=reference,minus=reference;plus[j]+=h;minus[j]-=h;Field gp,gm;target.evaluate(plus,gp);target.evaluate(minus,gm);
        for(size_t i=0;i<n;++i)matrix[i*n+j]=(gp[i]-gm[i])/(2*h);
        if(j%32==0&&FML::ThisTask==0)std::cout<<"[metric] column "<<j<<" evaluations "<<target.evaluations<<std::endl;
    }
    double scale=0,asymmetry=0;for(double v:matrix)scale=std::max(scale,std::abs(v));
    for(size_t i=0;i<n;++i)for(size_t j=0;j<i;++j){asymmetry=std::max(asymmetry,std::abs(matrix[i*n+j]-matrix[j*n+i]));const double mean=.5*(matrix[i*n+j]+matrix[j*n+i]);matrix[i*n+j]=matrix[j*n+i]=mean;}
    asymmetry/=std::max(1.,scale);require(asymmetry<.001,"Finite-difference Hessian is asymmetric");
    double hvp_error=0;
    for(int probe=0;probe<4;++probe){Field v(n);double norm=0;for(size_t j=0;j<n;++j){v[j]=std::sin((.31+.13*probe)*(j+1));norm+=v[j]*v[j];}for(double & x:v)x/=std::sqrt(norm);
        auto plus=reference,minus=reference;for(size_t j=0;j<n;++j){plus[j]+=.1*h*v[j];minus[j]-=.1*h*v[j];}Field gp,gm;target.evaluate(plus,gp);target.evaluate(minus,gm);double error=0,size=0;
        for(size_t i=0;i<n;++i){double expected=0;for(size_t j=0;j<n;++j)expected+=matrix[i*n+j]*v[j];const double actual=(gp[i]-gm[i])/(.2*h);error+=(actual-expected)*(actual-expected);size+=actual*actual;}
        hvp_error=std::max(hvp_error,std::sqrt(error/std::max(1.,size)));
    }
    require(hvp_error<.001,"Hessian directional refinement check failed");
    double ridge=0;bool positive=false;Field regularized;
    for(int attempt=0;attempt<16;++attempt){regularized=matrix;for(size_t i=0;i<n;++i)regularized[i*n+i]+=ridge;
        try{FixedMetric candidate(regularized);positive=true;break;}catch(const std::invalid_argument &){ridge=attempt==0?1e-8*std::max(1.,scale):10*ridge;}}
    require(positive,"Unable to regularize field metric");FixedMetric metric(regularized);require(metric.reconstruction_error()<1e-12,"Metric factorization failed");
    auto file=output(prefix+".matrix.csv");for(size_t i=0;i<n;++i)row(file,Field(regularized.begin()+i*n,regularized.begin()+(i+1)*n));
    auto meta=output(prefix+".meta.csv");if(FML::ThisTask==0)meta<<"dimension,fd_step,ridge,asymmetry,hvp_error,factor_error,evaluations,seconds\n"<<n<<','<<h<<','<<ridge<<','<<asymmetry<<','<<hvp_error<<','<<metric.reconstruction_error()<<','<<target.evaluations<<','<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
}
inline FixedMetric read_metric(const std::string & file,size_t dimension) {
    Field matrix;
    FML::COMMUNICATION::collective_try([&]{std::ifstream f(file);if(!f)throw std::invalid_argument("Missing field metric");std::string line,cell;size_t rows=0;
        while(std::getline(f,line)){if(++rows>dimension)throw std::invalid_argument("Too many metric rows");std::stringstream input(line);size_t columns=0;while(std::getline(input,cell,',')){size_t end;double value=std::stod(cell,&end);if(++columns>dimension||end!=cell.size()||!std::isfinite(value))throw std::invalid_argument("Invalid metric entry");matrix.push_back(value);}if(columns!=dimension)throw std::invalid_argument("Wrong metric columns");}
        if(rows!=dimension)throw std::invalid_argument("Wrong metric rows");FixedMetric check(matrix);
    },"Read fixed field metric");
    return FixedMetric(std::move(matrix));
}
#endif
