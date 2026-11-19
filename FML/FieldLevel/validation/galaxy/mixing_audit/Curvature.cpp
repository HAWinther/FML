#include <FML/FieldLevel/DistributedEulerianLPT.h>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace FF=FML::FIELDLEVEL::FRAMEWORK;
using Field=FML::FIELDLEVEL::Field;
int main(int argc,char **argv){FML::FMLSetup::init(&argc,&argv);try{
 if(argc!=3||FML::NTasks!=1)throw std::invalid_argument("curvature_probe FROZEN_STABILITY_INPUT OUTPUT.csv, serial only");
 std::ifstream in(argv[1]);std::string tag;unsigned mid,nm,nframes,trials,ncases;
 if(!(in>>tag>>mid>>nm>>nframes>>trials>>ncases)||tag!="FML_STABILITY_1"||mid!=203||nframes!=12||nm>64||ncases>8)throw std::invalid_argument("Unexpected bounded input");
 for(unsigned j=0;j<ncases;++j){std::string name;int current,leaps;double step;if(!(in>>name>>current>>step>>leaps))throw std::invalid_argument("Incomplete settings");}
 FF::Modes data(nm);for(auto &z:data){double re,im;if(!(in>>re>>im))throw std::invalid_argument("Incomplete data");z={re,im};}
 FF::EulerianLPTConfiguration c;c.mesh=4;c.product_mesh=8;c.density_mesh=16;c.boxsize=100;const double kf=2*M_PI/100;c.initial_cutoff=1.7*kf;c.matter_band={1.8*kf,FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};c.likelihood_band={1.6*kf,FML::FIELDLEVEL::EULERIAN::BandGeometry::Cube};c.matter_assignment=FF::MatterAssignment::PSWF;
 std::array<FF::Parameter,4> biases={{{"b_delta",1.6,0,1,0,4,FF::Prior::Uniform},{"b_delta2",.3,0,1,-4,4,FF::Prior::Uniform},{"b_K2",-.2,0,1,-4,4,FF::Prior::Uniform},{"b_laplacian",7,0,1,-100,100,FF::Prior::Uniform}}};
 FF::DistributedEulerianLPTModel model(c,FF::sigma8_spectrum([](double k){return 20000*std::exp(-5*k);}),biases);auto work=model.workspace();auto target=FF::fixed_noise_posterior(model,data,1000.);
 if(model.modes().size()!=nm)throw std::invalid_argument("Mode layout mismatch");std::ofstream out(argv[2]);out<<std::setprecision(17)<<"ensemble,draw,h,row,column,value\n";
 for(unsigned frame=0;frame<nframes;++frame){unsigned ensemble,draw;if(!(in>>ensemble>>draw))throw std::invalid_argument("Missing frame");std::vector<double>frozen(5),p(5);Field white(64);FF::Modes saved(nm);double savedq;
  for(auto *v:{&frozen,&p,&white})for(auto &x:*v)if(!(in>>x)||!std::isfinite(x))throw std::invalid_argument("Invalid vectors");
  for(auto &z:saved){double re,im;if(!(in>>re>>im))throw std::invalid_argument("Missing prediction");z={re,im};}if(!(in>>savedq))throw std::invalid_argument("Missing Q");
  const auto pred=model.predict(white,p,work);double q=0;for(size_t j=0;j<nm;++j){if(std::abs(pred[j]-saved[j])>1e-10)throw std::runtime_error("Forward changed");q+=.5*target.likelihood().weights()[j]*std::norm(pred[j]-data[j])*1000;}
  if(std::abs(q-savedq)>1e-11*std::max(1.,std::abs(q)))throw std::runtime_error("Q changed");
  for(double h:{1e-4,1e-5})for(size_t column=0;column<64;++column){Field unit(64,0);unit[column]=1;const auto v=target.apply_field_metric(unit,frozen,work,-.5);auto plus=white,minus=white;for(size_t j=0;j<64;++j){plus[j]+=h*v[j];minus[j]-=h*v[j];}
   Field a,b;std::vector<double>gp;const double up=target.evaluate(plus,p,work,a,gp),um=target.evaluate(minus,p,work,b,gp);if(!std::isfinite(up)||!std::isfinite(um))throw std::runtime_error("Nonfinite curvature endpoint");
   for(size_t j=0;j<64;++j)a[j]=(a[j]-b[j])/(2*h);const auto projected=target.apply_field_metric(a,frozen,work,-.5);
   for(size_t row=0;row<64;++row){if(!std::isfinite(projected[row]))throw std::runtime_error("Nonfinite curvature");out<<ensemble<<','<<draw<<','<<h<<','<<row<<','<<column<<','<<projected[row]<<'\n';}
  }
 }
 std::string extra;if(in>>extra)throw std::invalid_argument("Extra input");out.close();if(!out)throw std::runtime_error("Cannot write curvature");
 std::cout<<"PASS: all 12 source frames reproduced, 3072 field gradients evaluated for two-step curvature check\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
