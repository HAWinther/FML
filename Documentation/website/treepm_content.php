<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. TreePMCode: a TreePM N-body code</a><br>
<a href="#Steps">2. Time stepping</a><br>
<a href="#Reference">3. Against SimulationCode's TreePM and EuclidEmulator2</a><br>
<a href="#Limitations">4. Tests and limitations</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">TreePMCode: a TreePM N-body code</h1>
</header>
<hr/>

<p>
<a href="https://github.com/HAWinther/FML/tree/master/FML/TreePMCode" style="display: inline">FML/TreePMCode</a> is a simple TreePM N-body code for collisionless particles in the spirit of <a href="https://arxiv.org/abs/astro-ph/0505010" style="display: inline">GADGET-2</a>: CDM (or CDM and baryons as one set of particles), or CDM and baryons as two sets. There is no hydrodynamics. It is built from FML's building blocks and kept separate from <a href="amr.php#AMRCode" style="display: inline">AMRCode</a> (the RAMSES-like code) and <a href="nbodycode.php" style="display: inline">SimulationCode</a>.
</p>
<pre class="prettyprint">
cd FML/TreePMCode
make &amp;&amp; mpirun -np 4 ./treepm parameterfile.txt
cd tests &amp;&amp; make run
</pre>
<ul>
<li><b>Initial conditions</b>: 1LPT/2LPT from a linear $P(k)$ at $z = 0$ or from transfer functions (<tt>transfer_file</tt>, <tt>transfer_source = tables | camb | class</tt>), CDM and baryons from their own transfer functions (<tt>particle_baryons</tt>, <tt>baryon_lattice = same | offset</tt>), or the particles of a GADGET file (<tt>ic_gadget_file</tt>).</li>
<li><b>Domain</b>: the particles live on a Peano&ndash;Hilbert domain, rebuilt with equal particles per task every (PM) step.</li>
<li><b>Gravity</b>, split with a Gaussian of scale $r_s$ = <tt>asmth</tt> PM cells as in GADGET-2. The long range: the density deposited on the <tt>Nmesh</tt>$^3$ slab grid from the Hilbert domain, the filtered Poisson equation with the window deconvolved, and the force interpolated back by gathering the part of the grid each task needs. The short range, with the <a href="shortrange.php" style="display: inline">ShortRangeGravity</a> kernels, over the ghosts within the cutoff: the octree (<tt>backend = tree</tt>; <tt>opening_criterion = geometric</tt> with <tt>opening_angle</tt>, or <tt>relative</tt> as GADGET-2 with <tt>relative_tolerance</tt>) or the cell list (<tt>backend = cell</tt>). <tt>softening</tt> is Plummer-equivalent in units of the mean particle spacing.</li>
<li><b>Models</b> (as AMRCode): $\Lambda$CDM, coupled quintessence and the Fading Dark Sector, and covariantised dark scattering. With two species the source is weighted by $f_c(a)$ and $f_b(a)$, and with coupled quintessence the CDM also feels $2\beta^2 f_c$ times the force of the CDM alone.</li>
<li><b>Outputs</b>: $P(k)$ (of every species with two), GADGET files and images at the output redshifts.</li>
</ul>
<pre class="prettyprint">
Npart_1D   = 256
box        = 256.0
Nmesh      = 512        # the number of tasks must divide it
asmth      = 1.25       # r_s in PM cells
softening  = 0.0333     # in units of the mean particle spacing
backend    = tree
opening_criterion  = relative
relative_tolerance = 0.005
timestep_method    = gadget
output_redshifts   = 1, 0
</pre>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Steps">Time stepping</h1>
</header>
<hr/>

<p>
<tt>timestep_method = fixed</tt> is kick-drift-kick with a common step uniform in $\ln a$ (<tt>nsteps</tt>, or <tt>nsteps_between_outputs</tt>). <tt>timestep_method = gadget</tt> steps as GADGET-2 does. The PM step keeps the rms displacement below <tt>max_rms_displacement</tt> times min(mean spacing, $r_s$) and below <tt>max_dlna</tt>. The short-range force uses individual steps $\Delta t_i = \sqrt{2\eta\epsilon/|a_i|}$ (<tt>eta</tt>) in power-of-two bins of the PM step: half a long-range kick, kick-drift-kick sub-steps in which only the active particles get the short-range force, then the new domain, the PM force and the other half kick. For comparisons AMRCode has the same option with RAMSES' steps (<tt>timestep_method = ramses</tt>).
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Reference">Against SimulationCode's TreePM and EuclidEmulator2</h1>
</header>
<hr/>

<p>
SimulationCode's TreePM reference runs are frozen in <tt>ShortRangeGravity/tests/simcode_reference</tt> ($32^3$ particles in 64 Mpc/$h$ from $z = 19$). They were run without deconvolving the CIC window of the force (SimulationCode's default <tt>force_deconvolve_window = false</tt>). That suppresses the long-range force near $r_s$, so their result depends on the split. With <tt>deconvolve = false</tt> TreePMCode reproduces them to 0.3% at all $k$. With the window deconvolved, as in GADGET-2, the result no longer depends on the split: PM grids of $64^3$ and $256^3$ agree to 0.5%. Against SimulationCode run with the window deconvolved (1024 steps), max $|P/P_{\rm ref} - 1|$ is:
</p>
<table class="table table-striped table-condensed">
<tr><th></th><th>$z = 1$, $k \leq 1\,h$/Mpc</th><th>$z = 0$, $k \leq 1\,h$/Mpc</th><th>$z = 0$, up to $k_{\rm Ny}$</th><th>time (2 cores)</th></tr>
<tr><td>TreePMCode, cell list</td><td>0.01%</td><td>0.11%</td><td>0.7%</td><td>8.7 min</td></tr>
<tr><td>TreePMCode, tree, relative criterion 0.002</td><td>0.08%</td><td>0.23%</td><td>0.9%</td><td>5.8 min</td></tr>
<tr><td>TreePMCode, tree, relative criterion 0.005</td><td>0.03%</td><td>0.16%</td><td>2.0%</td><td>5.3 min</td></tr>
<tr><td>The frozen reference (window not deconvolved)</td><td>7.2%</td><td>4.9%</td><td>6.6%</td><td></td></tr>
</table>
<p>
Against <a href="https://arxiv.org/abs/2010.11288" style="display: inline">EuclidEmulator2</a> ($64^3$ in 128 Mpc/$h$, $z = 0$), $P/P_{\rm EE2}$ at $k = 0.5$, 1 and $1.5\,h$/Mpc is 1.02, 0.89 and 0.86 with GADGET steps and 0.86, 0.54 and 0.33 with 40 common steps. For comparison it is 0.92, 0.73 and 0.63 for AMRCode with RAMSES steps, and 0.98, 0.78 and 0.62 for SimulationCode's PM on $128^3$. Above $k \sim 1\,h$/Mpc the mass resolution of $64^3$ particles limits every code. See the <a href="https://github.com/HAWinther/FML/blob/master/FML/TreePMCode/README.md" style="display: inline">TreePMCode README</a> for the details and timings.
</p>

<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Limitations">Tests and limitations</h1>
</header>
<hr/>

<ul>
<li><b>Tests</b> (1, 2 and 4 tasks):
  <ul>
  <li>The force of a Gaussian blob against the exact periodic force: rms errors of 3.7% down to 0.4% (the particle noise).</li>
  <li>A Zel'dovich plane wave: 2.2% in $x$ and 2.5% in $v$, the same on 1, 2 and 4 tasks to 14 digits.</li>
  <li>The CDM&ndash;baryon growth with coupled quintessence and dark scattering against the linear two-fluid growth: within 0.1%.</li>
  <li>GADGET steps against 160 common steps.</li>
  </ul></li>
<li><b>Limitations</b>:
  <ul>
  <li>The ghosts and the tree are rebuilt at every short-range sub-step (GADGET-2 drifts the tree nodes).</li>
  <li>The load is balanced by particle count, not by work.</li>
  <li>The number of tasks must divide <tt>Nmesh</tt>.</li>
  <li>With CDM and baryons on the same lattice, a force that resolves the particle spacing damps the CDM&ndash;baryon relative mode. Use a softening of about the spacing, or the offset lattice.</li>
  </ul></li>
</ul>
