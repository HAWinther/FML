#ifndef FML_TSC_FIXED_METRIC_H
#define FML_TSC_FIXED_METRIC_H
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
// Small active-space validation metric. M is precision-like: p~N(0,M), qdot=M^-1 p.
// This dense helper is deliberately bounded; it is not a full-grid mass matrix.
class FixedMetric {
    size_t n;
    std::vector<double> matrix_,lower;
  public:
    explicit FixedMetric(std::vector<double> matrix):n(size_t(std::sqrt(matrix.size()))),matrix_(std::move(matrix)),lower(n*n,0.) {
        if(n==0||n>256||n*n!=matrix_.size())throw std::invalid_argument("Invalid dense metric dimensions");
        double scale=0;for(double v:matrix_){if(!std::isfinite(v))throw std::invalid_argument("Nonfinite metric");scale=std::max(scale,std::abs(v));}
        for(size_t i=0;i<n;++i)for(size_t j=0;j<=i;++j){
            if(std::abs(matrix_[i*n+j]-matrix_[j*n+i])>1e-12*std::max(1.,scale))throw std::invalid_argument("Asymmetric metric");
            double v=matrix_[i*n+j];for(size_t k=0;k<j;++k)v-=lower[i*n+k]*lower[j*n+k];
            if(i==j){if(!(v>1e-10*std::max(1.,scale)))throw std::invalid_argument("Metric is not positive definite");lower[i*n+j]=std::sqrt(v);}
            else lower[i*n+j]=v/lower[j*n+j];
        }
    }
    size_t dimension()const{return n;}
    const std::vector<double> & matrix()const{return matrix_;}
    void sqrt_multiply(std::vector<double> & x)const {
        if(x.size()!=n)throw std::invalid_argument("Wrong metric vector size");const auto input=x;
        for(size_t i=0;i<n;++i){x[i]=0;for(size_t j=0;j<=i;++j)x[i]+=lower[i*n+j]*input[j];}
    }
    void inverse(std::vector<double> & x)const {
        if(x.size()!=n)throw std::invalid_argument("Wrong metric vector size");
        for(size_t i=0;i<n;++i){for(size_t j=0;j<i;++j)x[i]-=lower[i*n+j]*x[j];x[i]/=lower[i*n+i];}
        for(size_t i=n;i-->0;){for(size_t j=i+1;j<n;++j)x[i]-=lower[j*n+i]*x[j];x[i]/=lower[i*n+i];}
    }
    double reconstruction_error()const {
        double error=0,scale=0;
        for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j){double v=0;for(size_t k=0;k<=std::min(i,j);++k)v+=lower[i*n+k]*lower[j*n+k];error=std::max(error,std::abs(v-matrix_[i*n+j]));scale=std::max(scale,std::abs(matrix_[i*n+j]));}
        return error/std::max(1.,scale);
    }
};
#endif
