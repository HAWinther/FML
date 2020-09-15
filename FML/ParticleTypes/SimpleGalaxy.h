#ifndef SIMPLEGALAXY_HEADER
#define SIMPLEGALAXY_HEADER

/// A minimal class for holding basic info about a galaxy
/// If all galaxies have the same weight then we don't need to include it or its methods below
/// but algorithms will still be able to use it
class SimpleGalaxy {
  public:
    double RA;
    double DEC;
    double z;
    double weight{1.0};

    SimpleGalaxy(double _RA, double _DEC, double _z, double _weight = 1.0)
        : RA(_RA), DEC(_DEC), z(_z), weight(_weight) {}

    double get_RA() const { return RA; }
    double get_DEC() const { return DEC; }
    double get_z() const { return z; }
    double get_weight() { return weight; }
    void set_RA(double _RA) const { return RA = _RA; }
    void set_DEC(double _DEC) const { return DEC = _DEC; }
    void set_z(double _z) const { return z = _z; }
    void set_weight(double _weight) { return weight = _weight; }
};

#endif
