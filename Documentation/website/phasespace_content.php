<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: haloes, subhaloes and their histories</a><br>
<a href="#Finder">2. The phase-space halo finder (FML::PHASE_SPACE_HALOS)</a><br>
<a href="#Binding">3. Unbinding and halo properties</a><br>
<a href="#MPI">4. Distributed runs and adapters</a><br>
<a href="#History">5. Merger histories (FML::HALO_HISTORY)</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: haloes, subhaloes and their histories</h1>
</header>
<hr/>

<p>
The <a href="fof.php" style="display: inline">Friends-of-Friends</a> algorithm finds haloes as connected regions in configuration space but cannot separate a subhalo moving through its host. Adding the velocities resolves it: in six-dimensional phase space a subhalo is a distinct, dense clump. <tt>FML/PhaseSpaceHaloFinder</tt> finds hosts and subhaloes in a single snapshot in the spirit of ROCKSTAR (Behroozi et al. 2013), and <tt>FML/HaloHistory</tt> links halo catalogues of several snapshots into merger trees by shared particle IDs. Spherical-overdensity masses are on the FoF page.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Finder">The phase-space halo finder</h1>
</header>
<hr/>

<p>
The algorithm:
</p>
<ol>
<li><b>Spatial FoF parents</b> with linking length $b = 0.28$ times the mean interparticle separation (minimum-image distances, exact connected components).</li>
<li><b>Phase-space metric.</b> Each parent is normalized by its mass-weighted position and velocity dispersions $\sigma_x$, $\sigma_v$, giving the distance
$$d(1,2) = \left[\frac{|\vec x_1 - \vec x_2|^2}{\sigma_x^2} + \frac{|\vec v_1 - \vec v_2|^2}{\sigma_v^2}\right]^{1/2}.$$</li>
<li><b>Hierarchical refinement.</b> A new 6D linking length is chosen such that a fraction 0.7 of the particles have a neighbour within it; FoF in 6D splits the group into children, and the process repeats in each child (with its own dispersions) until a minimum core size or depth is reached. The leaves are the seeds.</li>
<li><b>Seed merging.</b> A seed is merged into a larger seed if their core positions and velocities agree within the Poisson uncertainties ($\sigma_x/\sqrt N$, $\sigma_v/\sqrt N$); each seed joins at most one larger seed, so a poorly measured fragment cannot chain a subhalo into its host. An additional density-saddle test (kNN 6D densities sampled along the line between two peaks) keeps peaks that are separated by a resolved dip.</li>
<li><b>Assignment.</b> Particles are assigned to the closest seed on the way back up the hierarchy, using a dynamical position scale set by the seed's circular velocity and the reference density (200 times the mean density).</li>
</ol>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Binding">Unbinding and halo properties</h1>
</header>
<hr/>

<p>
Unbound particles are removed iteratively with the full isolated Newtonian potential of the assigned members, computed with a monopole tree (opening angle 0 gives exact direct sums) and the same compact spline softening as the <a href="shortrange.php" style="display: inline">short-range gravity</a> module. A particle remains if
$$\frac{1}{2}\left|\vec v - \vec v_{\rm bulk} + H(\vec x - \vec x_{\rm bulk})\right|^2 + \Phi_{\rm self} &lt; 0,\qquad \Phi_{\rm self}(\vec x_i) = -G\sum_{j\ne i}m_j\,g(|\vec x_i - \vec x_j|),$$
with the bulk velocity and potential recomputed at each iteration (the Hubble term is optional). The catalogue contains exclusive particle-ID memberships, the bound mass and the inclusive mass (with all subhaloes), the core position and velocity (shrinking core), velocity dispersion, bound radius, $V_{\rm max} = \max_r\sqrt{GM(&lt;r)/r}$ and $R_{\rm max}$, and the host: the nearest more massive candidate in the same parent whose bound extent contains the centre.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/PhaseSpaceHaloFinder/PhaseSpaceHaloFinder.h&gt;
using namespace FML::PHASE_SPACE_HALOS;
std::vector&lt;Particle&gt; particles;   // unique uint64 ID, position, peculiar velocity, mass
Options options;
options.boxsize = boxsize_mpch;
options.softening_support = softening_mpch;
options.position_to_physical = a;   // comoving coordinates
options.gravitational_constant = SI::G * SI::Msun / (SI::Mpc * std::pow(SI::km / SI::s, 2));
auto catalogue = find(particles, options);
// catalogue.halos[i].{id, host, bound_mass, inclusive_bound_mass, position, velocity, vmax, rmax, particle_ids}</pre>
  </div>
</div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="MPI">Distributed runs and adapters</h1>
</header>
<hr/>

<p>
<tt>find_distributed</tt> runs on FML's periodic slabs without gathering the particles: owned particles plus link-radius ghosts form local FoF edges, group minima propagate through ID/label messages until a global fixed point (which stitches groups crossing any number of slabs), complete parents are routed to a hash owner for analysis and the assignments return to the input ranks. Adapters give seeds for the spherical-overdensity routine, haloes for the <a href="halomodel.php#HOD" style="display: inline">HOD</a> (with an explicitly supplied mass definition) and snapshots for the merger histories below. A comparison with ROCKSTAR on a Plummer sphere recovers $V_{\rm max}$ within 5%; cosmological comparisons and completeness calibration are still to be done, and the finder loses deeply embedded subhaloes in the cores of their hosts.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="History">Merger histories</h1>
</header>
<hr/>

<p>
Haloes in consecutive snapshots are matched by the particles they share. For a halo $a$ with $N_a$ particles in the earlier snapshot and $b$ with $N_b$ in the later one, sharing $N_{ab}$,
$$\mathcal{M}_{ab} = \frac{N_{ab}^2}{N_aN_b} = \frac{N_{ab}}{N_a}\cdot\frac{N_{ab}}{N_b},$$
and each source picks the descendant of highest merit (ties by shared count, then ID). A target's main progenitor is the best-merit source among those that chose it; several incoming links are a merger. Optionally a halo without a descendant is searched for in the next few snapshots (to recover temporary disappearances). The matching uses sorted (particle ID, halo) arrays merged in one pass, so only halo pairs that share particles are visited: two snapshots with $10^7$ particles in $10^5$ haloes take ~7 s on one core.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
#include &lt;FML/HaloHistory/HaloHistory.h&gt;
using namespace FML::HALO_HISTORY;
MatchOptions criteria;
criteria.minimum_shared = 2;
auto links   = match(earlier, later, criteria);        // all candidates + chosen descendants
auto history = build({snap0, snap1, snap2}, criteria, /*max_skipped_snapshots=*/1);
// history.nodes (main_progenitor, progenitors, descendant), history.links, history.roots</pre>
  </div>
</div>
<p>
Catalogues can be exchanged in a versioned text format (<tt>CatalogueIO.h</tt>) with explicit units, mass definition and ID namespace, and converted to HOD input.
</p>
