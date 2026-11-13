#include "PencilGrid.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace FML::GRID {
namespace {
#if defined(SINGLE_PRECISION_FFTW)
#define PF_ALLOC fftwf_alloc_complex
#define PF_FREE fftwf_free
#define PF_PLAN_R2C fftwf_plan_guru_dft_r2c
#define PF_PLAN_C2R fftwf_plan_guru_dft_c2r
#define PF_PLAN_MANY fftwf_plan_many_dft
#define PF_EXEC fftwf_execute
#define PF_EXEC_DFT fftwf_execute_dft
#define PF_DESTROY fftwf_destroy_plan
const MPI_Datatype mpi_real = MPI_FLOAT, mpi_complex = MPI_C_FLOAT_COMPLEX;
#elif defined(LONG_DOUBLE_PRECISION_FFTW)
#define PF_ALLOC fftwl_alloc_complex
#define PF_FREE fftwl_free
#define PF_PLAN_R2C fftwl_plan_guru_dft_r2c
#define PF_PLAN_C2R fftwl_plan_guru_dft_c2r
#define PF_PLAN_MANY fftwl_plan_many_dft
#define PF_EXEC fftwl_execute
#define PF_EXEC_DFT fftwl_execute_dft
#define PF_DESTROY fftwl_destroy_plan
const MPI_Datatype mpi_real = MPI_LONG_DOUBLE, mpi_complex = MPI_C_LONG_DOUBLE_COMPLEX;
#else
#define PF_ALLOC fftw_alloc_complex
#define PF_FREE fftw_free
#define PF_PLAN_R2C fftw_plan_guru_dft_r2c
#define PF_PLAN_C2R fftw_plan_guru_dft_c2r
#define PF_PLAN_MANY fftw_plan_many_dft
#define PF_EXEC fftw_execute
#define PF_EXEC_DFT fftw_execute_dft
#define PF_DESTROY fftw_destroy_plan
const MPI_Datatype mpi_real = MPI_DOUBLE, mpi_complex = MPI_C_DOUBLE_COMPLEX;
#endif
    my_fftw_complex * fc(ComplexType * p) { return reinterpret_cast<my_fftw_complex *>(p); }
    MPI_Datatype subarray(MPI_Datatype base, std::array<int,3> sizes, std::array<int,3> subs, std::array<int,3> starts) {
        MPI_Datatype t; MPI_Type_create_subarray(3,sizes.data(),subs.data(),starts.data(),MPI_ORDER_C,base,&t); MPI_Type_commit(&t); return t;
    }
    MPI_Datatype strided3(MPI_Aint es, int no, MPI_Aint so, int nm, MPI_Aint sm, int ni, MPI_Aint si, MPI_Aint start) {
        MPI_Datatype a,b,c,d,e,t; MPI_Type_create_hvector(ni,1,si*es,mpi_complex,&a); MPI_Type_create_resized(a,0,sm*es,&b);
        MPI_Type_contiguous(nm,b,&c); MPI_Type_create_resized(c,0,so*es,&d); MPI_Type_contiguous(no,d,&e);
        int one=1; MPI_Aint disp=start*es; MPI_Type_create_hindexed(1,&one,&disp,e,&t); MPI_Type_commit(&t);
        MPI_Type_free(&a); MPI_Type_free(&b); MPI_Type_free(&c); MPI_Type_free(&d); MPI_Type_free(&e); return t;
    }
    void alltoallw(ComplexType * s, const std::vector<MPI_Datatype>& st, ComplexType * r,
                   const std::vector<MPI_Datatype>& rt, MPI_Comm c) {
        int n=int(st.size()); std::vector<int> counts(n,1), displs(n,0);
        MPI_Alltoallw(s,counts.data(),displs.data(),st.data(),r,counts.data(),displs.data(),rt.data(),c);
    }
}

PencilRealRange::PencilRealRange(int i0,int i1,int j0,int j1,int nz,int nzpad,int nyg,int gx,int gy)
 : i0_(i0),i1_(i1),j0_(j0),j1_(j1),nz_(nz),nzpad_(nzpad),nyg_(nyg),nghost_x_(gx),nghost_y_(gy) {}
PencilRealRange::PencilRealRange(size_t first,size_t last,int nx,int ny,int nz,int nzpad,int nyg,int gx,int gy)
 : i0_(0),i1_(0),j0_(0),j1_(0),nz_(nz),nzpad_(nzpad),nyg_(nyg),nghost_x_(gx),nghost_y_(gy),linear_(true),row_begin_(first),row_end_(last),nx_(nx),ny_(ny) {}
PencilRealRange::iterator PencilRealRange::make(int i) const {
    size_t idx=(size_t(i+nghost_x_)*nyg_+size_t(j0_+nghost_y_))*nzpad_;
    return iterator({i,j0_,0,idx},j0_,j1_,nz_,size_t(nzpad_-nz_),size_t(nyg_-(j1_-j0_))*nzpad_);
}
PencilRealRange::iterator PencilRealRange::begin() const {
    if(!linear_) return make(i0_);
    int i=int(row_begin_/ny_),j=int(row_begin_%ny_);
    size_t idx=(size_t(i+nghost_x_)*nyg_+size_t(j+nghost_y_))*nzpad_;
    return iterator({i,j,0,idx},nz_,row_begin_,row_end_,ny_,nyg_,nghost_x_,nghost_y_,nzpad_);
}
PencilRealRange::iterator PencilRealRange::end() const {
    if(!linear_) return make(i1_);
    return iterator({int(row_end_/ny_),int(row_end_%ny_),0,0},nz_,row_end_,row_end_,ny_,nyg_,nghost_x_,nghost_y_,nzpad_);
}
PencilRealRange::iterator& PencilRealRange::iterator::operator++() {
    if(linear_) {
        if(++cell_.k==nz_) { cell_.k=0; ++row_; if(row_<row_end_) { cell_.i=int(row_/ny_); cell_.j=int(row_%ny_); cell_.index=(size_t(cell_.i+nghost_x_)*nyg_+size_t(cell_.j+nghost_y_))*nzpad_; } }
        else ++cell_.index;
        return *this;
    }
    ++cell_.index;
    if (++cell_.k==nz_) { cell_.k=0; cell_.index+=row_skip_; if (++cell_.j==j1_) { cell_.j=j0_; cell_.index+=plane_skip_; ++cell_.i; } }
    return *this;
}
PencilFourierRange::iterator PencilFourierRange::begin() const {
    if(!linear_) return iterator({0,0,0,0},nkz_,nx_);
    return iterator({int(row_begin_/nkz_),int(row_begin_%nkz_),0,row_begin_*size_t(nx_)},row_begin_,row_end_,nkz_,nx_);
}
PencilFourierRange::iterator PencilFourierRange::end() const {
    if(!linear_) return iterator({nky_,0,0,size()},nkz_,nx_);
    return iterator({int(row_end_/nkz_),int(row_end_%nkz_),0,row_end_*size_t(nx_)},row_end_,row_end_,nkz_,nx_);
}
PencilFourierRange::iterator& PencilFourierRange::iterator::operator++() {
    if(linear_) { if(++cell_.ix==nx_) { cell_.ix=0; ++row_; cell_.index=row_*size_t(nx_); if(row_<row_end_) { cell_.a=int(row_/nkz_); cell_.b=int(row_%nkz_); } } else ++cell_.index; return *this; }
    ++cell_.index; if (++cell_.ix==nx_) { cell_.ix=0; if (++cell_.b==nkz_) { cell_.b=0; ++cell_.a; } } return *this;
}

PencilGrid::PencilGrid(std::shared_ptr<const PencilDomain> d,int g,unsigned flags)
 : decomposition_(std::move(d)),nghost_(g),fftw_flags_(flags) { ghost_types_.fill(MPI_DATATYPE_NULL); if(!decomposition_) throw std::invalid_argument("PencilGrid: null decomposition"); setup(); fill_real_grid(FloatType(0)); }
PencilGrid::~PencilGrid(){ release(); }
PencilGrid::PencilGrid(const PencilGrid& o):decomposition_(o.decomposition_),nghost_(o.nghost_),fftw_flags_(o.fftw_flags_){ ghost_types_.fill(MPI_DATATYPE_NULL); setup(); std::memcpy(main_,o.main_,nmain_*sizeof(ComplexType)); status_=o.status_; }
PencilGrid::PencilGrid(PencilGrid&& o) noexcept { ghost_types_.fill(MPI_DATATYPE_NULL); swap(o); }
void PencilGrid::swap(PencilGrid& o) noexcept {
    using std::swap; swap(decomposition_,o.decomposition_); swap(nghost_,o.nghost_); swap(nghost_y_,o.nghost_y_); swap(fftw_flags_,o.fftw_flags_); swap(mesh_,o.mesh_);
    swap(nzc_,o.nzc_);swap(nzpad_,o.nzpad_);swap(xs_,o.xs_);swap(nx_,o.nx_);swap(ys_,o.ys_);swap(ny_,o.ny_);swap(kys_,o.kys_);swap(nky_,o.nky_);swap(kzs_,o.kzs_);swap(nkz_,o.nkz_);swap(nxg_,o.nxg_);swap(nyg_,o.nyg_);
    swap(main_,o.main_);swap(nmain_,o.nmain_);swap(nscratch_,o.nscratch_);swap(plan_r2c_,o.plan_r2c_);swap(plan_c2r_,o.plan_c2r_);swap(plan_y_forward_,o.plan_y_forward_);swap(plan_y_backward_,o.plan_y_backward_);swap(plan_x_forward_,o.plan_x_forward_);swap(plan_x_backward_,o.plan_x_backward_);
    swap(type_a_,o.type_a_);swap(type_b_row_,o.type_b_row_);swap(type_b_column_,o.type_b_column_);swap(type_c_,o.type_c_);swap(ghost_types_,o.ghost_types_);swap(status_,o.status_);
}
void PencilGrid::release(){
    for(auto p:{plan_r2c_,plan_c2r_,plan_y_forward_,plan_y_backward_,plan_x_forward_,plan_x_backward_}) if(p) PF_DESTROY(p);
    plan_r2c_=plan_c2r_=plan_y_forward_=plan_y_backward_=plan_x_forward_=plan_x_backward_=nullptr;
    for(auto* v:{&type_a_,&type_b_row_,&type_b_column_,&type_c_}) { for(auto& t:*v) if(t!=MPI_DATATYPE_NULL) MPI_Type_free(&t); v->clear(); }
    for(auto& t:ghost_types_) if(t!=MPI_DATATYPE_NULL) MPI_Type_free(&t);
    ghost_types_.fill(MPI_DATATYPE_NULL);
    PF_FREE(reinterpret_cast<my_fftw_complex*>(main_)); main_=nullptr; nmain_=0;
}
void PencilGrid::setup(){
    mesh_=decomposition_->mesh(); nzc_=decomposition_->fourier_last_size(); nzpad_=2*nzc_;
    auto x=decomposition_->x_range(),y=decomposition_->y_range(),ky=decomposition_->ky_range(),kz=decomposition_->kz_range();
    xs_=x.start;nx_=x.count;ys_=y.start;ny_=y.count;kys_=ky.start;nky_=ky.count;kzs_=kz.start;nkz_=kz.count;
    int p0=decomposition_->process_grid()[0],p1=decomposition_->process_grid()[1];
    nghost_y_ = decomposition_->ndim() >= 2 ? nghost_ : 0;
    int minx=mesh_[0]/p0,miny=mesh_[1]/p1;
    if(nghost_<0||nghost_>minx||nghost_y_>miny) throw std::invalid_argument("PencilGrid: ghost width exceeds smallest pencil");
    nxg_=nx_+2*nghost_; nyg_=ny_+2*nghost_y_;
    nmain_=std::max(size_t(nxg_)*nyg_*nzc_,size_t(nky_)*nkz_*mesh_[0]);
    main_=reinterpret_cast<ComplexType*>(PF_ALLOC(nmain_)); if(!main_) throw std::bad_alloc();
    nscratch_=decomposition_->ndim()==1 ? 2*size_t(mesh_[0]) : size_t(nx_)*nkz_*mesh_[1]; decomposition_->workspace().reserve(nscratch_);
    create_plans(); create_datatypes();
}
void PencilGrid::create_plans(){
    auto* s=decomposition_->workspace().data(); int ny=mesh_[1],nx=mesh_[0];
    if(decomposition_->ndim()==1) {
        int len=mesh_[0];
        auto * scratch_complex=s+len;
        plan_x_forward_=PF_PLAN_MANY(1,&len,1,fc(scratch_complex),nullptr,1,len,fc(scratch_complex),nullptr,1,len,FFTW_FORWARD,fftw_flags_);
        plan_x_backward_=PF_PLAN_MANY(1,&len,1,fc(scratch_complex),nullptr,1,len,fc(scratch_complex),nullptr,1,len,FFTW_BACKWARD,fftw_flags_);
        if(!plan_x_forward_||!plan_x_backward_) throw std::runtime_error("PencilGrid: 1D FFTW planning failed");
        return;
    }
    fftw_iodim dz{mesh_[2],1,1}; fftw_iodim forward[2]={{nx_,nyg_*nzpad_,nyg_*nzc_},{ny_,nzpad_,nzc_}};
    fftw_iodim backward[2]={{nx_,nyg_*nzc_,nyg_*nzpad_},{ny_,nzc_,nzpad_}};
    size_t off=size_t(nghost_)*nyg_+nghost_; FloatType* r0=real_data()+off*nzpad_; auto* c0=fc(main_+off*nzc_);
    plan_r2c_=PF_PLAN_R2C(1,&dz,2,forward,r0,c0,fftw_flags_); plan_c2r_=PF_PLAN_C2R(1,&dz,2,backward,c0,r0,fftw_flags_);
    plan_y_forward_=PF_PLAN_MANY(1,&ny,nx_*nkz_,fc(s),nullptr,1,ny,fc(s),nullptr,1,ny,FFTW_FORWARD,fftw_flags_);
    plan_y_backward_=PF_PLAN_MANY(1,&ny,nx_*nkz_,fc(s),nullptr,1,ny,fc(s),nullptr,1,ny,FFTW_BACKWARD,fftw_flags_);
    plan_x_forward_=PF_PLAN_MANY(1,&nx,nky_*nkz_,fc(main_),nullptr,1,nx,fc(main_),nullptr,1,nx,FFTW_FORWARD,fftw_flags_);
    plan_x_backward_=PF_PLAN_MANY(1,&nx,nky_*nkz_,fc(main_),nullptr,1,nx,fc(main_),nullptr,1,nx,FFTW_BACKWARD,fftw_flags_);
    if(!plan_r2c_||!plan_c2r_||!plan_y_forward_||!plan_y_backward_||!plan_x_forward_||!plan_x_backward_) throw std::runtime_error("PencilGrid: FFTW planning failed");
}
void PencilGrid::create_datatypes(){
    auto P=decomposition_->process_grid(); MPI_Aint cs=sizeof(ComplexType);
    if(decomposition_->ndim()!=1) for(int j=0;j<P[1];++j){ auto kz=decomposition_->kz_range_of(j), yr=decomposition_->y_range_of(j);
        type_a_.push_back(subarray(mpi_complex,{nxg_,nyg_,nzc_},{nx_,ny_,kz.count},{nghost_,nghost_,kz.start}));
        type_b_row_.push_back(strided3(cs,nx_,MPI_Aint(nkz_)*mesh_[1],yr.count,1,nkz_,mesh_[1],yr.start)); }
    if(decomposition_->ndim()!=1) for(int i=0;i<P[0];++i){ auto ky=decomposition_->ky_range_of(i),xr=decomposition_->x_range_of(i);
        type_b_column_.push_back(subarray(mpi_complex,{nx_,nkz_,mesh_[1]},{nx_,nkz_,ky.count},{0,0,ky.start}));
        type_c_.push_back(strided3(cs,xr.count,1,nkz_,mesh_[0],nky_,MPI_Aint(nkz_)*mesh_[0],xr.start)); }
    if(nghost_){ std::array<int,3> sz{nxg_,nyg_,nzpad_},sx{nghost_,nyg_,nzpad_},sy{nxg_,nghost_y_,nzpad_};
        ghost_types_[0]=subarray(mpi_real,sz,sx,{nx_,0,0});ghost_types_[1]=subarray(mpi_real,sz,sx,{0,0,0});ghost_types_[2]=subarray(mpi_real,sz,sx,{nghost_,0,0});ghost_types_[3]=subarray(mpi_real,sz,sx,{nx_+nghost_,0,0});
        if(nghost_y_){ghost_types_[4]=subarray(mpi_real,sz,sy,{0,ny_,0});ghost_types_[5]=subarray(mpi_real,sz,sy,{0,0,0});ghost_types_[6]=subarray(mpi_real,sz,sy,{0,nghost_y_,0});ghost_types_[7]=subarray(mpi_real,sz,sy,{0,ny_+nghost_y_,0});}
        ghost_types_[8]=subarray(mpi_real,sz,{nghost_,ny_,nzpad_},{0,nghost_y_,0});ghost_types_[9]=subarray(mpi_real,sz,{nghost_,ny_,nzpad_},{nx_+nghost_,nghost_y_,0}); }
}
void PencilGrid::fftw_r2c(){
    if(status_!=Status::Real) throw std::logic_error("fftw_r2c requires real-space grid");
    if(decomposition_->ndim()==1) {
        std::vector<FloatType> packed(static_cast<size_t>(nx_),FloatType(0));
        for(int i=0;i<nx_;++i) packed[size_t(i)]=real(i,0,0);
        std::vector<int> counts(size_t(decomposition_->nranks())),displacements(counts.size());
        for(int r=0;r<decomposition_->nranks();++r){auto coord=decomposition_->coords_of(r);auto range=decomposition_->x_range_of(coord[0]);counts[size_t(r)]=range.count;displacements[size_t(r)]=range.start;}
        auto* s=decomposition_->workspace().data();
        MPI_Allgatherv(packed.data(),nx_,mpi_real,reinterpret_cast<FloatType*>(s),counts.data(),displacements.data(),mpi_real,decomposition_->communicator());
        auto* full=s+mesh_[0];auto* gathered=reinterpret_cast<FloatType*>(s);
        for(int i=0;i<mesh_[0];++i) full[i]=ComplexType(gathered[i],FloatType(0));
        PF_EXEC_DFT(plan_x_forward_,fc(full),fc(full));
        std::copy(full,full+mesh_[0],main_);status_=Status::Fourier;return;
    }
    auto* s=decomposition_->workspace().data(); PF_EXEC(plan_r2c_);alltoallw(main_,type_a_,s,type_b_row_,decomposition_->row_communicator());PF_EXEC_DFT(plan_y_forward_,fc(s),fc(s));alltoallw(s,type_b_column_,main_,type_c_,decomposition_->column_communicator());PF_EXEC(plan_x_forward_);status_=Status::Fourier;
}
void PencilGrid::fftw_c2r(){
    if(status_!=Status::Fourier) throw std::logic_error("fftw_c2r requires Fourier-space grid");
    if(decomposition_->ndim()==1) {
        auto* s=decomposition_->workspace().data();auto* full=s+mesh_[0];std::copy(main_,main_+mesh_[0],full);PF_EXEC_DFT(plan_x_backward_,fc(full),fc(full));
        auto range=decomposition_->x_range();FloatType norm=FloatType(1)/mesh_[0];status_=Status::Real;for(int i=0;i<nx_;++i)real(i,0,0)=full[range.start+i].real()*norm;return;
    }
    auto* s=decomposition_->workspace().data();PF_EXEC(plan_x_backward_);alltoallw(main_,type_c_,s,type_b_column_,decomposition_->column_communicator());PF_EXEC_DFT(plan_y_backward_,fc(s),fc(s));alltoallw(s,type_b_row_,main_,type_a_,decomposition_->row_communicator());PF_EXEC(plan_c2r_);FloatType norm=FloatType(1)/(mesh_[0]*mesh_[1]*mesh_[2]);status_=Status::Real;for(auto c:get_real_range())(*this)[c]*=norm;
}
void PencilGrid::communicate_boundaries(){ if(!nghost_)return;if(status_!=Status::Real)throw std::logic_error("ghost exchange requires real-space grid");auto* p=real_data();MPI_Comm c=decomposition_->communicator();int lo=decomposition_->neighbour(0,-1),hi=decomposition_->neighbour(0,1);MPI_Sendrecv(p,1,ghost_types_[0],hi,10,p,1,ghost_types_[1],lo,10,c,MPI_STATUS_IGNORE);MPI_Sendrecv(p,1,ghost_types_[2],lo,11,p,1,ghost_types_[3],hi,11,c,MPI_STATUS_IGNORE);if(nghost_y_){lo=decomposition_->neighbour(1,-1);hi=decomposition_->neighbour(1,1);MPI_Sendrecv(p,1,ghost_types_[4],hi,12,p,1,ghost_types_[5],lo,12,c,MPI_STATUS_IGNORE);MPI_Sendrecv(p,1,ghost_types_[6],lo,13,p,1,ghost_types_[7],hi,13,c,MPI_STATUS_IGNORE);} }
void PencilGrid::add_ghosts_to_owners(){ if(!nghost_)return;if(status_!=Status::Real)throw std::logic_error("ghost reduction requires real-space grid");auto* p=real_data();MPI_Comm c=decomposition_->communicator();std::vector<FloatType>b;auto add=[&](int i0,int i1,int j0,int j1){size_t n=0;for(int i=i0;i<i1;i++)for(int j=j0;j<j1;j++)for(int k=0;k<nzpad_;k++)real_data()[real_offset(i,j,k)]+=b[n++];};int lo,hi,cnt;
    if(nghost_y_){lo=decomposition_->neighbour(1,-1);hi=decomposition_->neighbour(1,1);cnt=nxg_*nghost_y_*nzpad_;b.resize(cnt);MPI_Sendrecv(p,1,ghost_types_[5],lo,20,b.data(),cnt,mpi_real,hi,20,c,MPI_STATUS_IGNORE);add(-nghost_,nx_+nghost_,ny_-nghost_y_,ny_);MPI_Sendrecv(p,1,ghost_types_[7],hi,21,b.data(),cnt,mpi_real,lo,21,c,MPI_STATUS_IGNORE);add(-nghost_,nx_+nghost_,0,nghost_y_);}
    lo=decomposition_->neighbour(0,-1);hi=decomposition_->neighbour(0,1);cnt=nghost_*ny_*nzpad_;b.resize(cnt);MPI_Sendrecv(p,1,ghost_types_[8],lo,22,b.data(),cnt,mpi_real,hi,22,c,MPI_STATUS_IGNORE);add(nx_-nghost_,nx_,0,ny_);MPI_Sendrecv(p,1,ghost_types_[9],hi,23,b.data(),cnt,mpi_real,lo,23,c,MPI_STATUS_IGNORE);add(0,nghost_,0,ny_); }

size_t PencilGrid::real_offset(int i,int j,int k)const{return (size_t(i+nghost_)*nyg_+size_t(j+nghost_y_))*nzpad_+k;}
size_t PencilGrid::fourier_offset(int a,int b,int x)const{return (size_t(a)*nkz_+b)*mesh_[0]+x;}
void PencilGrid::check_real(int i,int j,int k)const{if(status_!=Status::Real||i < -nghost_||i>=nx_+nghost_||j < -nghost_y_||j>=ny_+nghost_y_||k<0||k>=mesh_[2])throw std::out_of_range("PencilGrid real index/status invalid");}
void PencilGrid::check_fourier(int a,int b,int x)const{if(status_!=Status::Fourier||a<0||a>=nky_||b<0||b>=nkz_||x<0||x>=mesh_[0])throw std::out_of_range("PencilGrid Fourier index/status invalid");}
FloatType& PencilGrid::operator[](const PencilRealCell& c){check_real(c.i,c.j,c.k);if(c.index!=real_offset(c.i,c.j,c.k))throw std::invalid_argument("PencilRealCell from different layout");return real_data()[c.index];}
const FloatType& PencilGrid::operator[](const PencilRealCell& c)const{check_real(c.i,c.j,c.k);if(c.index!=real_offset(c.i,c.j,c.k))throw std::invalid_argument("PencilRealCell from different layout");return real_data()[c.index];}
ComplexType& PencilGrid::operator[](const PencilFourierCell& c){check_fourier(c.a,c.b,c.ix);if(c.index!=fourier_offset(c.a,c.b,c.ix))throw std::invalid_argument("PencilFourierCell from different layout");return main_[c.index];}
const ComplexType& PencilGrid::operator[](const PencilFourierCell& c)const{check_fourier(c.a,c.b,c.ix);if(c.index!=fourier_offset(c.a,c.b,c.ix))throw std::invalid_argument("PencilFourierCell from different layout");return main_[c.index];}
FloatType& PencilGrid::real(int i,int j,int k){check_real(i,j,k);return real_data()[real_offset(i,j,k)];} const FloatType& PencilGrid::real(int i,int j,int k)const{check_real(i,j,k);return real_data()[real_offset(i,j,k)];}
ComplexType& PencilGrid::fourier(int a,int b,int x){check_fourier(a,b,x);return main_[fourier_offset(a,b,x)];} const ComplexType& PencilGrid::fourier(int a,int b,int x)const{check_fourier(a,b,x);return main_[fourier_offset(a,b,x)];}
FloatType* PencilGrid::get_real_grid(){if(status_!=Status::Real)throw std::logic_error("grid is not in real space");return real_data();} const FloatType* PencilGrid::get_real_grid()const{if(status_!=Status::Real)throw std::logic_error("grid is not in real space");return real_data();}
ComplexType* PencilGrid::get_fourier_grid(){if(status_!=Status::Fourier)throw std::logic_error("grid is not in Fourier space");return main_;} const ComplexType* PencilGrid::get_fourier_grid()const{if(status_!=Status::Fourier)throw std::logic_error("grid is not in Fourier space");return main_;}
my_fftw_complex* PencilGrid::get_fftw_grid(){return reinterpret_cast<my_fftw_complex*>(main_);}
size_t PencilGrid::get_index_real(const std::array<int,3>& c)const{return real_offset(c[0],c[1],c[2]);}size_t PencilGrid::get_index_fourier(const std::array<int,3>& c)const{return fourier_offset(c[0],c[1],c[2]);}
std::array<int,3> PencilGrid::get_coord_from_index(size_t n)const{int k=int(n%nzpad_);n/=nzpad_;int j=int(n%nyg_)-nghost_y_;int i=int(n/nyg_)-nghost_;return{i,j,k};}
std::array<int,3> PencilGrid::get_fourier_coord_from_index(size_t n)const{int x=int(n%mesh_[0]);n/=mesh_[0];int b=int(n%nkz_),a=int(n/nkz_);return{a,b,x};}
FloatType PencilGrid::get_real(const std::array<int,3>& c)const{return real(c[0],c[1],c[2]);}FloatType PencilGrid::get_real_from_index(size_t n)const{auto c=get_coord_from_index(n);return real(c[0],c[1],c[2]);}void PencilGrid::set_real(const std::array<int,3>& c,FloatType v){real(c[0],c[1],c[2])=v;}void PencilGrid::set_real_from_index(size_t n,FloatType v){auto c=get_coord_from_index(n);real(c[0],c[1],c[2])=v;}void PencilGrid::add_real(const std::array<int,3>& c,FloatType v){real(c[0],c[1],c[2])+=v;}
ComplexType PencilGrid::get_fourier(const std::array<int,3>& c)const{return fourier(c[0],c[1],c[2]);}ComplexType PencilGrid::get_fourier_from_index(size_t n)const{auto c=get_fourier_coord_from_index(n);return fourier(c[0],c[1],c[2]);}void PencilGrid::set_fourier(const std::array<int,3>& c,ComplexType v){fourier(c[0],c[1],c[2])=v;}void PencilGrid::set_fourier_from_index(size_t n,ComplexType v){auto c=get_fourier_coord_from_index(n);fourier(c[0],c[1],c[2])=v;}
std::array<int,3> PencilGrid::global_cell(const PencilRealCell& c)const{auto w=[](int i,int n){i%=n;return i<0?i+n:i;};return{w(xs_+c.i,mesh_[0]),w(ys_+c.j,mesh_[1]),c.k};}
std::array<double,3> PencilGrid::get_real_position(const std::array<int,3>& c)const{return{(xs_+c[0])*decomposition_->cell_size(0),(ys_+c[1])*decomposition_->cell_size(1),c[2]*decomposition_->cell_size(2)};}
std::array<int,3> PencilGrid::get_fourier_integer_wavevector(const std::array<int,3>& c)const{return{fold(c[2],mesh_[0]),fold(kys_+c[0],mesh_[1]),kzs_+c[1]};}
std::array<double,3> PencilGrid::get_fourier_wavevector(const std::array<int,3>& c)const{auto n=get_fourier_integer_wavevector(c);double kf=2*M_PI/decomposition_->boxsize();return{kf*n[0],kf*n[1],kf*n[2]};}
std::array<double,3> PencilGrid::get_fourier_wavevector_from_index(size_t i)const{return get_fourier_wavevector(get_fourier_coord_from_index(i));}
double PencilGrid::hermitian_weight(const PencilFourierCell& c)const{int k=kzs_+c.b;return(k==0||(mesh_[2]%2==0&&k==mesh_[2]/2))?1.0:2.0;}
void PencilGrid::get_fourier_wavevector_and_norm_by_index(size_t i,std::array<double,3>& k,double& m)const{k=get_fourier_wavevector_from_index(i);m=std::sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);}
void PencilGrid::fill_real_grid(FloatType v){std::fill(real_data(),real_data()+2*nmain_,v);status_=Status::Real;}
void PencilGrid::fill_real_grid(const std::function<FloatType(std::array<double,3>&)>& f){status_=Status::Real;for(auto c:get_real_range()){auto x=get_real_position({c.i,c.j,c.k});(*this)[c]=f(x);}}
void PencilGrid::fill_fourier_grid(ComplexType v){std::fill(main_,main_+nmain_,v);status_=Status::Fourier;}
void PencilGrid::fill_fourier_grid(const std::function<ComplexType(std::array<double,3>&)>& f){status_=Status::Fourier;for(auto c:get_fourier_range()){auto k=get_fourier_wavevector({c.a,c.b,c.ix});(*this)[c]=f(k);}}
PencilRealRange PencilGrid::get_real_range_of_rows(size_t a,size_t b)const{size_t total=size_t(nx_)*ny_;if(a>b||b>total)throw std::out_of_range("real row range");return PencilRealRange(a,b,nx_,ny_,mesh_[2],nzpad_,nyg_,nghost_,nghost_y_);}
PencilFourierRange PencilGrid::get_fourier_range_of_rows(size_t a,size_t b)const{size_t total=size_t(nky_)*nkz_;if(a>b||b>total)throw std::out_of_range("Fourier row range");return PencilFourierRange(a,b,nky_,nkz_,mesh_[0]);}
void PencilGrid::info()const{std::cout<<"PencilGrid NDIM="<<decomposition_->ndim()<<" rank "<<decomposition_->rank()<<": real ["<<nx_<<'x'<<ny_<<'x'<<mesh_[2]<<"], Fourier ["<<nky_<<'x'<<nkz_<<'x'<<mesh_[0]<<"], ghosts "<<nghost_<<", bytes "<<allocated_storage_bytes()<<'\n';}
void PencilGrid::free(){release();}
}
