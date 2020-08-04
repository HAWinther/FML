#include <iostream>
#include "../GeneralParticle.h"

// General particle
const constexpr bool HasPOS  = true;
const constexpr bool HasVEL  = true;
const constexpr bool HasID   = true;
const constexpr bool HasMASS = true;
const constexpr bool HasVOL  = true;
const constexpr int NDIM = 3;
using PositionType = double;
using VelocityType = double;
using MassType     = double;
using VolType      = double;
using IDType       = long long int;
using Particle = GeneralParticle<NDIM, 
      PositionType, HasPOS, 
      VelocityType, HasVEL, 
      IDType, HasID, 
      MassType, HasMASS,
      VolType, HasVOL>;

int main(){
  Particle p;

  PositionType pos[NDIM];
  for(int idim = 0; idim < NDIM; idim++)
    pos[idim] = (rand() % 100)/100.;
  VelocityType vel[NDIM];
  for(int idim = 0; idim < NDIM; idim++)
    vel[idim] = 10.0 + (rand() % 100)/100.;
  IDType id = 538;
  MassType mass = 752.0;
  VolType vol = 123.0;

  p.set_pos(pos);
  p.set_vel(vel);
  p.set_id(id);
  p.set_mass(mass);
  p.set_vol(vol);

  std::cout << std::boolalpha;
  std::cout << " Bytesize: " << p.get_particle_byte_size() << "\n";

  std::cout << " Has pos:  " << p.has_pos() << "\n";
  auto *ppos = p.get_pos();
  if(ppos){
    for(int idim = 0; idim < NDIM; idim++)
      std::cout << ppos[idim] << " ";
    std::cout << std::endl;
  }

  std::cout << " Has vel:  " << p.has_vel() << "\n";
  auto *pvel = p.get_vel();
  if(pvel){
    for(int idim = 0; idim < NDIM; idim++)
      std::cout << pvel[idim] << " ";
    std::cout << std::endl;
  }

  std::cout << " Has ID:   " << p.has_id()  << "\n";
  auto pid = p.get_id();
  std::cout << pid << std::endl;

  std::cout << " Has mass: " << p.has_mass()  << "\n";
  auto pmass = p.get_mass();
  std::cout << pmass << std::endl;
  
  std::cout << " Has vol: " << p.has_vol()  << "\n";
  auto pvol = p.get_vol();
  std::cout << pvol << std::endl;
}


