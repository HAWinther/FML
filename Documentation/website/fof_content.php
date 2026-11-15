<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Friends of Friends</a><br>
<a href="#Methods">2. Available methods</a><br>
<a href="#SO">3. Spherical-overdensity masses</a><br>
<a href="#Subhalos">4. Subhaloes and merger trees</a><br>
<a href="#References">Method references and scope</a><br>
<a href="#Relaxed">Relaxed FoF for low-resolution simulations</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Friends of Friends</h1>
</header>
<hr/>  

<p>
The Friends-of-Friends (FoF) linking definition is the standard cosmological method introduced by <a href="https://doi.org/10.1086/163168" style="display: inline">Davis et al. (1985)</a>: particles linked within a chosen distance (typically $0.2$ times the mean particle separation) form a group. FML bins particles to a grid and searches neighboring cells, then exchanges boundary particles across MPI tasks. The grid search, ownership rule and buffer diagnostics are FML's implementation details. Each group is assigned to the task containing its center, avoiding duplicate groups. If a group reaches the buffer edge, increase the buffer. Results are returned as FoFGroups, and particle IDs are stored when supported by the particle type.
</p>

<p>
<b>Any box domain.</b> <tt>FoFDomain.h</tt> has <tt>FriendsOfFriends(domain, part, NumPart, ...)</tt>: the same arguments and results with the particles on a Slab or Pencil domain (e.g. <tt>MPIParticles::get_domain()</tt>), so halos can be found in a run on <a href="pencilgrid.php" style="display: inline">pencils</a>. Spherical-overdensity masses also work on pencils.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Available methods</h1>
</header>
<hr/>  

<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
// In the namespace FML::FOF (FriendsOfFriends/FoF.h)
template&lt;class T, int NDIM, class FoFHaloClass = FoFHalo&lt;T,NDIM&gt; &gt;
  void FriendsOfFriends(
      T *part,
      size_t NumPart,
      double linking_length,          // In units of the mean particle separation
      int nmin_FoF_group,             // Minimum number of particles in a group (e.g. 20)
      bool periodic,
      double Buffersize_over_Boxsize, // e.g. (3 Mpc/h) / Boxsize
      std::vector&lt;FoFHaloClass&gt; &amp; LocalFoFGroups,
      int Ngrid = FoF_Ngrid_max);

// Example:
std::vector&lt;FML::FOF::FoFHalo&lt;Particle, 3&gt;&gt; FoFGroups;
FML::FOF::FriendsOfFriends&lt;Particle, 3&gt;(
    part.get_particles_ptr(), part.get_npart(),
    0.2, 20, true, 3.0 / Boxsize, FoFGroups);
</pre>
</div>
</div>
</p>
<p>
The fiducial class FoFHalo (FriendsOfFriends/FoFBinning.h) computes the number of particles, the mass, the center of mass position, the mean velocity and the velocity dispersion of the group. If you want to bin up things that are not provided in the fiducial class see FoFBinning.h for how to make your own FoFHaloClass to do the binning. If you want to limit how many grid-cells we use (i.e. the memory) you can set FML::FOF::FoF_Ngrid_max or pass Ngrid. The time spent in the different parts of the algorithm is recorded in FML::FOF::foftimer.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Relaxed">Relaxed FoF for low-resolution simulations</h1>
</header>
<hr/>

<p>Set <tt>fof_relaxed = true</tt> in SimulationCode or SnapshotAnalyzer to use the multi-pass relaxed finder. It uses longer links for small halos and rejects groups by their velocity dispersion. The standalone API distinguishes FastPM and paper algorithms. Output catalogues add the pass, size bin, linking length and velocity-dispersion ratio, including an HDF5 <tt>RelaxedFoF</tt> group when enabled.</p>
<p>The defaults were fitted to a specific FastPM configuration and need recalibration for other resolutions and step schedules. With those defaults the rejection threshold becomes nonpositive above approximately $z=2.2$, rejecting every halo. Peculiar velocities and the correct redshift are required. See <a href="https://github.com/HAWinther/FML/blob/master/FML/FriendsOfFriends/RelaxedFoF.md" style="display: inline">the full relaxed-FoF parameters and algorithms</a> and the <a href="snapshotanalyzer.php" style="display: inline">snapshot workflow</a>.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="SO">Spherical-overdensity masses</h1>
</header>
<hr/>

<p>
FoF masses depend on the linking length and on the shape of the halo; spherical-overdensity (SO) masses are defined by the radius $R_\Delta$ within which the mean density is $\Delta$ times a reference density,
$$M_\Delta = \frac{4\pi}{3}R_\Delta^3\,\Delta\,\rho_{\rm ref},\qquad \rho_{\rm ref} = \bar\rho_m\ (\Delta{\rm m}),\ \rho_c\ (\Delta{\rm c}),\ \text{or}\ \Delta_{\rm vir}(z)\rho_c\ (\text{vir}).$$
<tt>SphericalOverdensity</tt> (FriendsOfFriends/SphericalOverdensity.h) takes seed centres (e.g. the FoF haloes), finds the centre with a shrinking sphere, then grows the sphere around it to the requested density threshold and reports the outermost crossing. The virial overdensity convention follows <a href="https://doi.org/10.1086/305262" style="display: inline">Bryan &amp; Norman (1998)</a>; shrinking-sphere centering is motivated by the convergence discussion of <a href="https://arxiv.org/abs/astro-ph/0201544" style="display: inline">Power et al. (2003)</a>. FML returns radius, mass, particle count, mean velocity and dispersion, and flags spheres that hit the search radius. Its MPI search, centering and outermost-crossing policies are implementation-specific.
</p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/FriendsOfFriends/SphericalOverdensity.h&gt;
// thresholds: Delta * rho_ref in units of the mean density of the box (e.g. 200 for 200m)
auto so = FML::FOF::SphericalOverdensity&lt;Particle, 3&gt;(
    part.get_particles_ptr(), part.get_npart(), total_mass,
    centers, initial_radius, {200.0, 500.0}, search_radius);
// so[i].pos, so[i].radius[j], so[i].mass[j], so[i].npart[j], so[i].vel, so[i].truncated
</pre>
</div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Subhalos">Subhaloes and merger trees</h1>
</header>
<hr/>

<p>
FoF cannot separate a subhalo from its host. The <a href="phasespace.php" style="display: inline">phase-space halo finder</a> finds hosts and subhaloes in 6D phase space with unbinding, and the halo-history module links catalogues of several snapshots into merger trees.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="References">Method references and scope</h1>
</header>
<hr/>

<p>
Relaxed FoF (<tt>FriendsOfFriends/RelaxedFoF.h</tt>) follows <a href="https://arxiv.org/abs/1908.05276" style="display: inline">Dai, Feng, Seljak &amp; Singh (2019), §2.1 / Algorithm 1</a>. It offers separate paper and FastPM modes, including the linking-length schedule and velocity-dispersion rejection. The default FastPM calibration is specific to its original simulations; it is not a universal halo-mass calibration. The <a href="https://github.com/HAWinther/FML/blob/master/FML/FriendsOfFriends/RelaxedFoF.h" style="display: inline">source documentation</a> records the differences and parameter limits.
</p>
