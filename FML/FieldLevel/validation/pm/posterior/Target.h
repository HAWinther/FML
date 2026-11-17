#ifndef FML_PM_TSC_POSTERIOR_VALIDATION_H
#define FML_PM_TSC_POSTERIOR_VALIDATION_H
#include <FML/FieldLevel/validation/common/PMJointProblem.h>
#include <map>
// Validation-only orthonormal active Fourier basis, on the unchanged particle grid.
class ActiveBasis {
    using Grid=FML::FIELDLEVEL::FRONTEND::Grid;
    const JointModel & model;
    Grid grid;
    struct Slot {ptrdiff_t index;size_t pair;int sign;bool unique;};
    std::vector<Slot> slots;
  public:
    size_t dimension{2*waves().size()};
    explicit ActiveBasis(const JointModel & m):model(m),grid(FML::GRID::make_grid<Grid>(m.configuration().mesh,0,0,m.domain())) {
        FML::COMMUNICATION::collective_try([&]{std::map<std::array<int,3>,size_t> pairs;size_t pair=0;for(auto k:waves())pairs[{k.x,k.y,k.z}]=pair++;
            for(auto i:grid.get_fourier_range()){auto k=grid.get_fourier_integer_wavevector_from_index(i);int square=0;for(int x:k)square+=x*x;if(square==0||square>9)continue;
                const bool positive=k[0]>0||(k[0]==0&&k[1]>0)||(k[0]==0&&k[1]==0&&k[2]>0);const bool unique=k[2]>0||positive;
                std::array<int,3> key;for(int d=0;d<3;++d)key[d]=positive?k[d]:-k[d];slots.push_back({i,pairs.at(key),positive?1:-1,unique});}
        },"Active Fourier basis slots");
    }
    Field expand_active(const Field & z) {
        grid.fill_fourier_grid(0.);grid.set_grid_status_real(false);const double norm=std::sqrt(.5/model.layout().global_cells);
        for(const auto & s:slots)grid.set_fourier_from_index(s.index,FML::GRID::ComplexType(norm*z[2*s.pair],-s.sign*norm*z[2*s.pair+1]));
        grid.fftw_c2r();Field result(model.cells());size_t q=0;for(auto i:grid.get_real_range())result[q++]=grid.get_real_from_index(i);return result;
    }
    Field project_active(const Field & x) {
        size_t q=0;for(auto i:grid.get_real_range())grid.set_real_from_index(i,x[q++]);grid.set_grid_status_real(true);grid.fftw_r2c();
        Field result(dimension,0.);const double norm=std::sqrt(2.*model.layout().global_cells);
        for(const auto & s:slots)if(s.unique){const auto value=grid.get_fourier_from_index(s.index);result[2*s.pair]=norm*value.real();result[2*s.pair+1]=-s.sign*norm*value.imag();}
        FML::SumArrayOverTasks(result.data(),result.size());return result;
    }
};
// Exact active marginal in physical q=exp(A/2)*s coordinates. The inactive
// white directions integrate to a parameter-independent Gaussian constant.
class CenteredTarget {
    const JointModel & model;
    FF::BasicPosterior<JointModel> posterior;
    JointModel::Workspace work;
    std::vector<double> reference;
    Field last,gradient;
    double base{},chi{},square{},dof{};
    bool valid=false;
  public:
    ActiveBasis basis;
    size_t evaluations{};
    CenteredTarget(const JointModel & m,const FF::Modes & data):model(m),posterior(m,data),work(m.workspace()),reference(m.initial_parameters()),basis(m) {
        for(double weight:posterior.likelihood().weights())dof+=weight;dof=m.sum(dof);
    }
    double evaluate(const Field & z,Field & g) {
        require(z.size()==basis.dimension+2,"Wrong centered target dimension");model.controls(z);g.assign(z.size(),0.);
        const size_t d=basis.dimension;const double A=z[d],N=z[d+1];if(!model.in_support({A,N}))return HUGE_VAL;
        if(!valid||!std::equal(last.begin(),last.end(),z.begin())) {
            valid=false;last.assign(z.begin(),z.begin()+d);const auto white=basis.expand_active(last);Field gs;std::vector<double> gp;
            base=posterior.evaluate(white,reference,work,gs,gp);gradient=basis.project_active(gs);chi=dof-2*gp.back();square=inner(last,last);valid=true;++evaluations;
        }
        const double alpha=std::exp(-A),delta=N-reference[1],rho=std::exp(-delta);
        for(size_t j=0;j<d;++j)g[j]=rho*(gradient[j]-last[j])+alpha*last[j];
        g[d]=-.5*alpha*square+model.parameters()[0].derivative(A)+.5*d;
        g[d+1]=.5*(dof-rho*chi)+model.parameters()[1].derivative(N);
        return base+.5*std::expm1(-A)*square+.5*chi*std::expm1(-delta)+.5*dof*delta+
            model.parameters()[0].penalty(A)+model.parameters()[1].penalty(N)+.5*d*A;
    }
    // Only reference-density state is restored; the actual PM counter never rolls back.
    struct Cache {
        const CenteredTarget * owner;
        Field last, gradient;
        double base, chi, square;
        bool valid;
    };
    Cache snapshot() const {return {this,last,gradient,base,chi,square,valid};}
    void restore(const Cache & c) {
        if(c.owner!=this)throw std::invalid_argument("Foreign centered-target cache");
        last=c.last;gradient=c.gradient;base=c.base;chi=c.chi;square=c.square;valid=c.valid;
    }
    double conditional(size_t parameter,double x) const {
        if(!valid||parameter>1)throw std::invalid_argument("Invalid conditional cache");
        const auto & prior=model.parameters()[parameter];
        return prior.penalty(x)+(parameter==0 ? .5*std::exp(-x)*square+.5*basis.dimension*x :
            .5*chi*std::exp(-(x-reference[1]))+.5*dof*(x-reference[1]));
    }
    std::pair<double,double> direct_error(const Field & z) {
        Field g;const double u=evaluate(z,g);const size_t d=basis.dimension;auto white=basis.expand_active(z);const double scale=std::exp(-z[d]/2);for(auto & x:white)x*=scale;
        Field gs;std::vector<double> gp;const double direct=posterior.evaluate(white,{z[d],z[d+1]},work,gs,gp)+.5*d*z[d];
        auto projected=basis.project_active(gs);for(auto & x:projected)x*=scale;
        projected.push_back(gp[0]-.5*model.sum(inner(white,gs))+.5*d);projected.push_back(gp[1]);
        double error=0;for(size_t j=0;j<g.size();++j)error=std::max(error,std::abs(g[j]-projected[j]));return {u-direct,error};
    }
};
static inline Field read_physical(const std::string & file) {
    Field last;FML::COMMUNICATION::collective_try([&]{std::ifstream f(file);std::string line,cell;int rows=0;while(std::getline(f,line)){std::stringstream input(line);Field row;while(std::getline(input,cell,',')){size_t end;double x=std::stod(cell,&end);if(end!=cell.size()||!std::isfinite(x))throw std::invalid_argument("Invalid physical coefficient");row.push_back(x);}if(row.size()!=122)throw std::invalid_argument("Wrong physical coefficient count");last=std::move(row);++rows;}if(rows!=3)throw std::invalid_argument("Expected three physical coefficient rows");},"Physical MAP seed");for(auto & x:last)x/=std::sqrt(.002);return last;
}
#endif
