#ifndef EXAMPLE_PARTICLES
#define EXAMPLE_PARTICLES

//======================================================================
// Some examples of particles you can use depending on what you want to do.
// The LPT order we use for COLA is determined from what is availiable in the particles
// and some algorithms can do more things (or even require) if certain fields are
// availiable. E.g. watershed require get/set_volume and FoF halo finder will
// store fofid in the particle if present. Particles->Grid checks for get_mass
// and will use the mass, otherwise assumes all masses are the same. GADGET
// writer checks for get_id and will only output IDs in the files if this is present.
// IC generator stores displacement fields and ID and mass in particles if present.
//
// In all the fields below the choice of type is free, you can use floats for
// positions or long doubles if you want that. The ID type should be big enough to
// hold the number of particles so long long int (8 bytes) is usually the way to go
// for any sim with more than 1024^3 particles otherwise unsigned int works.
//
// If you need dynamically allocated memory inside the class then that is no
// problem, but you will have to implement three functions that appends the data
// to a communiation buffer, assigns the data from a communication buffer and a
// method that gives the byte-size of the particle. Otherwise communication of 
// the particles across tasks will not work.
//======================================================================

const int NDIM = 3;

//======================================================================
/// This is the particle that is what you will want to use
/// Its good for 2LPT COLA with and without scaledependent growth
/// and it has an ID which is useful for analyzing the output with
/// other codes that read GADGET files (and is often required)
//======================================================================
class FiducialParticle {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double Psi_1LPT[NDIM];
    double Psi_2LPT[NDIM];
    long long int id;

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_D_2LPT() { return Psi_2LPT; }
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
};

//======================================================================
/// This is the particle that is what you will want to use
/// if memory is an issue. Same as above just with floats
//======================================================================
class FiducialParticleLowMemory {
  private:
    float pos[NDIM];
    float vel[NDIM];
    float Psi_1LPT[NDIM];
    float Psi_2LPT[NDIM];
    long long int id;

  public:
    float * get_pos() { return pos; }
    float * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    float * get_D_1LPT() { return Psi_1LPT; }
    float * get_D_2LPT() { return Psi_2LPT; }
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
};

//======================================================================
/// Minimal particle that will work with the N-body solver
//======================================================================
class MinimalParticle {
  private:
    double pos[NDIM];
    double vel[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
};

//======================================================================
/// Minimal particle for normal 1LPT COLA
//======================================================================
class MinimalParticleCOLA_1LPT {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double Psi_1LPT[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_D_1LPT() { return Psi_1LPT; }
};

//======================================================================
/// Minimal particle for normal 2LPT COLA
//======================================================================
class MinimalParticleCOLA_2LPT {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double Psi_1LPT[NDIM];
    double Psi_2LPT[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_D_2LPT() { return Psi_2LPT; }
};

//======================================================================
/// Minimal particle for normal 3LPT COLA
//======================================================================
class MinimalParticleCOLA_3LPT {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double Psi_1LPT[NDIM];
    double Psi_2LPT[NDIM];
    double Psi_3LPTa[NDIM];
    double Psi_3LPTb[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_D_2LPT() { return Psi_2LPT; }
    double * get_D_3LPTa() { return Psi_3LPTa; }
    double * get_D_3LPTb() { return Psi_3LPTb; }
};

//======================================================================
/// Minimal particle for scaledependent 1LPT COLA
//======================================================================
class MinimalParticleScaledependentCOLA_1LPT {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double q[NDIM];
    double Psi_1LPT[NDIM];
    double dPsidloga_1LPT[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_q() { return q; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_dDdloga_1LPT() { return dPsidloga_1LPT; }
};

//======================================================================
/// Minimal particle for scaledependent 2LPT COLA
/// NB: The 1LPT and 2LPT fields are used for
/// temporary storage of Psi1lpt+Psi2lpt and its derivative
/// needed to compute the COLA kick and drift
//======================================================================
class MinimalParticleScaledependentCOLA_2LPT {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double q[NDIM];
    double Psi_1LPT[NDIM];
    double Psi_2LPT[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_q() { return q; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_D_2LPT() { return Psi_2LPT; }
};

//======================================================================
/// Minimal particle for scaledependent 3LPT COLA
/// NB: The 1LPT and 2LPT fields are used for
/// temporary storage of Psi1lpt+Psi2lpt+Psi3lpt and its derivative
/// needed to compute the COLA kick and drift
//======================================================================
class MinimalParticleScaledependentCOLA_3LPT {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double Psi_1LPT[NDIM];
    double Psi_2LPT[NDIM];
    double q[NDIM];

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_q() { return q; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_D_2LPT() { return Psi_2LPT; }
};

//======================================================================
/// A particle with almost everything you'll need (but requires twice
/// the memory of the fiducial one)
//======================================================================
class FatFuckingParticle {
  private:
    double pos[NDIM];
    double vel[NDIM];
    double Psi_1LPT[NDIM];
    double Psi_2LPT[NDIM];
    double Psi_3LPTa[NDIM];
    double Psi_3LPTb[NDIM];
    double q[NDIM];
    long long int id;
    double mass;
    double volume;
    int fofid;

  public:
    double * get_pos() { return pos; }
    double * get_vel() { return vel; }
    constexpr int get_ndim() const { return NDIM; }
    double * get_q() { return q; }
    double * get_D_1LPT() { return Psi_1LPT; }
    double * get_D_2LPT() { return Psi_2LPT; }
    double * get_D_3LPTa() { return Psi_3LPTa; }
    double * get_D_3LPTb() { return Psi_3LPTb; }
    long long int get_id() const { return id; }
    void set_id(long long int _id) { id = _id; }
    double get_mass() const { return mass; }
    void set_mass(double _mass) { mass = _mass; }
    double get_volume() const { return volume; }
    void set_volume(double _volume) { volume = _volume; }
    int get_fofid() const { return fofid; }
    void set_fofid(int _fofid) { fofid = _fofid; }
};
#endif
