#ifndef SIMPLEPARTICLE_HEADER
#define SIMPLEPARTICLE_HEADER

// Simple particle with all the methods needed to be used with MPIParticles
// Some algorithms might require more field like get_vol, id etc.
template<int NDIM>
struct SimpleParticle {
  
  double Pos[NDIM];
  double Vel[NDIM];
  
  int get_ndim(){
    return NDIM;
  }

  double *get_pos(){
    return Pos;
  }
  
  double *get_vel(){
    return Vel;
  }
  
  double get_mass(){
    return 1.0;
  }
  
  // Methods needed for commmunication of particle
  int get_particle_byte_size(){
    return 2 * sizeof(double) * NDIM;
  }
  
  void append_to_buffer(char *buffer){
    int bytes = sizeof(double) * NDIM;
    std::memcpy(buffer, Pos, bytes);
    buffer += bytes;
    bytes = sizeof(double) * NDIM;
    std::memcpy(buffer, Vel, bytes);
    buffer += bytes;
  }
  
  void assign_from_buffer(char *buffer){
    int bytes = sizeof(double) * NDIM;
    std::memcpy(Pos, buffer, bytes);
    buffer += bytes;
    bytes = sizeof(double) * NDIM;
    std::memcpy(Vel, buffer, bytes);
    buffer += bytes;
  }

};

#endif
