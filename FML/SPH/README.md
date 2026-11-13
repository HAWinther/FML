# Conservative SPH for adiabatic gas

Header-only C++17 module in `FML::SPH`, with a shared ideal-gas primitive in
[`Hydrodynamics/EquationOfState.h`](../Hydrodynamics/EquationOfState.h).
It evolves periodic 1D/2D/3D gas particles using adaptive density-energy SPH,
variable smoothing lengths and grad-h corrections. It supports serial, OpenMP
and slab MPI execution without all-gathering particle payloads. FFTW and GSL are
not required. [tests/RESULTS.md](tests/RESULTS.md) records measured validation and
its limits. This is the initial implementation of backlog #50.

## Minimal use

Initialize FML once, as for the other modules. All ranks construct/call their
solver in the same order, including ranks with no particles. Input particles
can initially be on arbitrary ranks; construction redistributes them by x slab
and checks globally unique uint64 IDs by hash partition.

```cpp
#include <FML/SPH/SPH.h>
using namespace FML::SPH;
std::vector<Particle<3>> local; // Fill positions, velocities, mass, energy and unique ID.
Options options;
options.boxsize = 1;
options.gamma = 5.0 / 3;
options.h_min = .005;
options.h_max = .1;
Solver<3> gas(std::move(local), options);
while (gas.time() < .1)
    gas.advance(.1 - gas.time()); // Uses min(max_dt, global timestep recommendation).
auto fields = gas.evaluate();
auto totals = gas.diagnostics();
// gas.particles() and fields have the same local ordering, sorted by stable ID.
```

`Particle<D>` stores `position`, `velocity`, positive `mass`, positive specific
internal `energy`, a nonnegative smoothing-length guess `h` and `id`. A zero h
chooses the bracket midpoint. Positions are in `[0,L)` in each dimension. Use
one consistent length/mass/time unit system; velocity has length/time, energy
has velocity squared, and pressure has density times velocity squared.
`gas.particles()` is const: the solver owns particle evolution and ID stability.
Returned field arrays are owning values, so retain them explicitly when needed.

`evaluate()` solves density and updates owned h guesses, then returns density,
Omega, pressure, sound speed, divergence/curl, shear limiter, acceleration,
internal-energy rate, signal speed, global timestep and local ghost counts.
`step(dt)` takes an explicitly chosen common step; `advance(max_dt)` chooses
the smaller global CFL/acceleration/energy-change step without a redundant third
force evaluation. Timesteps must advance the epoch representably.

## Density, support and conservative equations

The cubic B-spline has `W=sigma_D h^-D f(r/h)`, sigma=(2/3,10/(7 pi),1/pi),
`f=1−3q²/2+3q³/4` below q=1 and `f=(2−q)³/4` from 1 to 2. **Support is 2h**;
this differs from the support-radius convention in ShortRangeGravity softening.
The density includes the self contribution. For each particle solve

```
rho_i = sum_j m_j W(r_ij,h_i)
h_i^D rho_i = eta^D m_i
Omega_i = 1 + h_i/(D rho_i) sum_j m_j dW(r_ij,h_i)/dh_i
F_i = P_i / (Omega_i rho_i^2)
a_i = -sum_j m_j [F_i grad_i W(h_i) + F_j grad_i W(h_j) + Pi_ij grad_i Wbar]
du_i/dt = sum_j m_j [F_i v_ij.grad_i W(h_i) + Pi_ij v_ij.grad_i Wbar/2]
```

The bracketed Newton/bisection solve has an explicit tolerance and iteration cap.
`h_min <= h <= h_max`, eta^D must exceed the kernel's self coefficient, and
`2*h_max < L/2` guarantees a unique minimum-image interaction. An unbracketed,
nonconvergent or degenerate root raises a collective exception; h is not silently
clamped. Choose the bounds to encompass the actual gas density range.

Force searches include the **union** of both particles' supports, so neighbors
outside 2h_i but inside 2h_j still contribute. Pair forces are antisymmetric and
the matched pressure-work equation conserves total gas energy at the
semi-discrete level. Unequal masses are supported. For static inviscid gas the
ODE also preserves the entropy proxy `(gamma−1)u/rho^(gamma−1)`. Finite-step RK2
introduces integration error; it is not an exact-energy or exact-entropy step.

The equations follow the variational density-energy formulation summarized in
[Price's SPH review](https://arxiv.org/abs/1012.1885). The implementation is
independent and the tests include a fixed-entropy finite-difference force check.

## Dissipation and time integration

For approaching physical pairs, viscosity uses
`mu=hbar*(v_ij.r_ij)/(r²+epsilon*hbar²)` and
`Pi=(-alpha*cbar*mu+beta*mu²)/rhobar`, with the arithmetic mean of the two
Balsara factors. The factor is `|div v|/(|div v|+|curl v|+epsilon_switch*c/h)`.
Default alpha=1, beta=2; the shear limiter can be disabled. Viscosity is zero
for separating pairs and converts kinetic energy to heat.

Optional conductivity (default alpha_u=0) adds
`alpha_u sum_j m_j/rhobar*(u_i−u_j)*sqrt(|P_i−P_j|/rhobar)*Wbar'` to du/dt.
It exchanges heat conservatively and increases thermal entropy for positive u.
It is artificial conductivity, not a physical thermal-diffusion model. The
pressure-based signal can be inappropriate in stratified gravitational equilibria;
validate the setting for the intended problem. Classic density-energy SPH can
retain contact/surface-tension and particle-disorder errors.

Evolution is explicit global-timestep Heun RK2. The recommendation combines
`courant*a*h/max_signal`, `acceleration_factor*sqrt(a*h/|dv/dt|)` and
`energy_factor*u/|du/dt|`; energy change limits apply to heating and cooling.
A failed predictor/corrector leaves the original particles and epoch intact.
There is no energy floor, no individual timestepping and no checkpoint API yet.
A caller can retry a rejected step with smaller dt. Side effects in external
callbacks are the caller's responsibility and cannot be rolled back.

## Gravity and cosmological conventions

A collective `ExternalForce` callback receives `(time, local_particles, extra)`
and fills one acceleration vector per owned particle. It is called twice per
step, at the actual predictor states. Returned acceleration is **dv/dt in the
solver's units**, added once. This permits a caller to connect the existing
particle/PM gravity infrastructure or an external force. The callback must use
matching particle ownership and source masses and coordinate any of its own MPI
errors; callbacks containing collectives must be called on all ranks. Presence,
output shape and finite values are checked. The core has no built-in gravity
solve, collisionless-species evolution or SimulationCode gas adapter.

Without a background callback, a=1 and H=0 and positions are physical periodic
coordinates. An optional 3D `BackgroundFunction(time)` supplies consistent a(t)
and H=adot/a in physical time. Positions and h are then comoving, stored velocity
is physical peculiar velocity, internal energy is physical, and returned density
is comoving. `Thermodynamics::pressure` is the coordinate-pressure `a^3 P_phys`;
physical density/pressure are obtained by division by a^3. The evolution uses

```
dx/dt = v_pec/a
dv_pec/dt = a_hydro_comoving/a - H v_pec + extra
du/dt = du_hydro_comoving/a - 3 (gamma-1) H u
```

Dissipative pair approach uses physical relative motion
`v_pec,i−v_pec,j + a H (x_i−x_j)` in viscosity and viscous heating. Velocity
sampling for the shear limiter includes physical divergence `div_x(v_pec)/a+3H`
and curl `curl_x(v_pec)/a`. Sound speeds remain physical. The external gravity
callback supplies a physical peculiar acceleration; the solver applies no
additional gravity normalization or factor of a to it. Cosmological gas energy
is not conserved; homogeneous drag/cooling and instantaneous scale factors have
separate reference tests. Coupled cosmological self-gravity/shock calibration
remains open.

## MPI, ownership and memory

The solver uses FML's x slabs (`xmin_domain`, `xmax_domain`), which must partition
[0,1]. Empty slabs/ranks are supported; pencil/space-filling-curve particle
ownership is not selected by this module. The density phase exchanges sources
within 2h_max; after density/velocity gradients, the force phase exchanges full
thermodynamic records within twice the largest actual h. Both include nonadjacent
slabs and append each source once per destination. No particle payload catalogue
is all-gathered. Large support can still make ghost memory substantial.

Flat offset/index cell lists bound cell metadata by `max_cells`. Source and cell
buffers reuse capacity; neighbor traversal allocates no per-pair or per-target
neighbor lists. OpenMP parallelizes owned density and force targets without
shared force writes. Each directed pair is evaluated separately. Stable ID
ordering improves rank/thread replay; bitwise equality across hardware is not
promised. Global timestep reduction includes external acceleration.

Chunked homogeneous-ABI MPI records reuse `Global/ChunkedExchange.h` and
`Global/PeriodicSlabExchange.h`. `max_message_bytes` bounds payload/receive scratch,
not outgoing staging, ghosts or total memory. RK2 carries original particle/rate
records through predictor migration and keeps the original state for rollback;
peak storage is several local particle arrays plus ghosts and O(Ntasks) metadata.
Input/configuration and dependent local phases are checked collectively. The
shared transport helper's allocation failures and arbitrary external callback
exceptions are not a general fault-tolerant MPI contract.

## Build and validation

```sh
make -C FML/SPH/tests run USE_MPI=false USE_OMP=false
make -C FML/SPH/example USE_MPI=false
# Use a separate clean build when switching MPI/OpenMP flags.
make -C FML/SPH/tests clean
make -C FML/SPH/tests run USE_MPI=true USE_OMP=true MPIRUN=mpirun NP_LIST='1 2 4'
```

The normal module-test runner discovers these tests. Scientific profiles/plots
are a separate bounded experiment:

```sh
python3 FML/SPH/tests/run_validation.py \
  --executable /path/to/serial/sph_example \
  --mpi-executable /path/to/mpi/sph_example \
  --mpirun 'mpirun' --output /fresh/output/directory
```

NumPy and Matplotlib are needed for plots; SciPy is not required. Only use
launcher flags supported by the selected MPI implementation. Tests compare
kernel normalization/derivatives, all-pair density/forces, conservation,
dissipation signs, shear limiting, migration/chunking, failure rollback,
background terms and rank/thread evolution. The fluid experiment includes an
independently solved exact Euler Sod reference, sound-wave refinement and a
bounded 3D thermal blast. It does not establish Sedov self-similarity, turbulent
mixing, hydrostatic/self-gravitating collapse or production cosmological accuracy.
Those, alternative kernels/neighbor convergence, native gravity/species coupling,
checkpointing and larger distributed performance studies remain follow-ups.
