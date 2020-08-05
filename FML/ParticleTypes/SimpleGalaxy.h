#ifndef SIMPLEGALAXY_HEADER
#define SIMPLEGALAXY_HEADER

// A minimal class for holding basic info about a galaxy
class SimpleGalaxy {
  public:
    double RA;
    double DEC;
    double z;
    double weight { 1.0 };

    SimpleGalaxy(double _RA, double _DEC, double _z, double _weight = 1.0) : 
      RA(_RA),
      DEC(_DEC),
      z(_z),
      weight(_weight) {}

    double get_RA() const{
      return RA;
    }

    double get_DEC() const{
      return DEC;
    }

    double get_z() const{
      return z;
    }

    double get_weight(){
      return weight;
    }
};

#endif
