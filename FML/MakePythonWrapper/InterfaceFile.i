%module MyLibrary
%{
  #define SWIG_FILE_WITH_INIT

  // The functions and data structures you want availiable in Python
  struct Data {
    int n;
    double *x;
  };
  extern void test(int n);
  extern Data * getData(int n);
  extern void freeData(Data *f);
  extern void getNumpyArray(int nx, double *x, int ny, double *y);

%}

%include "numpy.i"
%init %{
import_array();
%}

// This defines an interface for numpy arrays. The C function getNumpyArray(int nx, double *x, int ny, double *y) 
// can with this be called as getNumpyArray(x,y) from python with x,y being 1D numpy arrays
%apply (int DIM1, double* IN_ARRAY1) {(int nx, double *x), (int ny, double *y)};

  
// The functions and data structures you want availiable in Python (have to be repeated here)
struct Data {
  int n;
  double *x;
};
extern void test(int n);
extern Data * getData(int n);
extern void freeData(Data *f);
extern void getNumpyArray(int nx, double *x, int ny, double *y);

// Here we define how to extract the data from the struct Data in python
%extend Data{
  int get_n(){
    return $self->n;
  }
  double get_x(int i) {
    if($self->n > 0)
      return $self->x[i];
    return 0.0;
  }
  ~Data(){
    freeData($self);
  }
}

