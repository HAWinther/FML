# Shared gas primitives

`EquationOfState.h` supplies `FML::HYDRODYNAMICS::IdealGas`: pressure, squared sound
speed, pressure-to-specific-energy conversion and entropy proxy, with explicit
constant gamma>1. Inputs to these arithmetic primitives must be positive finite
physical density/internal energy; the consuming solver validates representability.
The class has no gravity, SPH, MPI or Riemann dependency.

The initial consumer is [SPH](../SPH/README.md). Future finite-volume/Riemann
implementations can use the same thermodynamic conventions. This primitive does
not implement backlog #51's Euler fluxes or Riemann solvers.
