%module MyLibrary
%{
  #define SWIG_FILE_WITH_INIT
  #include <stdexcept>

  // The functions and data structures you want available in Python
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
%include "exception.i"
%init %{
import_array();
%}

// This defines an interface for numpy arrays. The C function getNumpyArray(int nx, double *x, int ny, double *y) 
// can with this be called as getNumpyArray(x,y) from python with x,y being 1D numpy arrays
%apply (int DIM1, double* IN_ARRAY1) {(int nx, double *x), (int ny, double *y)};

// getData returns a new object that python owns: the destructor (~Data below, which calls freeData) runs when
// the python object is deleted/garbage collected. freeData is therefore not exposed to python (calling it on an
// object python owns would free the memory twice)
%newobject getData;

// Turn C++ exceptions from the accessors into python exceptions
%exception Data::get_x {
  try {
    $action
  } catch (const std::out_of_range & e) {
    SWIG_exception(SWIG_IndexError, e.what());
  }
}
  
// The functions and data structures you want available in Python (have to be repeated here)
struct Data {
  int n;
  double *x;
};
extern void test(int n);
extern Data * getData(int n);
extern void getNumpyArray(int nx, double *x, int ny, double *y);

// Here we define how to extract the data from the struct Data in python
%extend Data{
  int get_n(){
    return $self->n;
  }
  double get_x(int i) {
    if (i < 0 or i >= $self->n)
      throw std::out_of_range("Data.get_x: index out of range");
    return $self->x[i];
  }
  ~Data(){
    freeData($self);
  }
}
