<p class="page-toc"><a href="documentation.php">Back to documentation</a><br>
<a href="#Overview">Overview</a><br>
<a href="#Analyses">Available analyses</a><br>
<a href="#Usage">Usage</a><br>
<a href="#HybridBias">Hybrid bias input</a><br>
</p>
<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">SnapshotAnalyzer: analyse existing Gadget snapshots</h1>
</header>
<hr/>

<figure class="cinematic-figure" id="cinematic-12-one-universe-many-scales">
  <a href="images/cinematic/12-one-universe-many-scales.jpg" aria-label="View full-size illustration: One Universe, many scales">
    <img src="images/cinematic/12-one-universe-many-scales.jpg" width="1774" height="887" loading="lazy" decoding="async" alt="A conceptual zoom from a cosmic web through a cluster to a halo with bound substructure">
  </a>
  <figcaption><strong>One Universe, many scales.</strong> Conceptual journey across cosmic scales, from a large-volume web through a cluster to halo substructure; the continuous zoom compresses scale changes. <span class="cinematic-credit">AI-generated concept illustration.</span></figcaption>
</figure>

<p><tt>FML/SnapshotAnalyzer</tt> is a standalone post-processing executable for Gadget binary and HDF5 snapshots. It reads positions, velocities, IDs, types and masses, redistributes particles to their owning MPI slabs, and runs the library's analysis kernels without depending on SimulationCode. The default <tt>files_per_task</tt> reader reads each file once across the tasks; <tt>all_tasks</tt> is also available.</p>
<p>Lua and FFTW are required. MPI and OpenMP use the normal FML build configuration. HDF5 snapshots/catalogues need <tt>USE_HDF5=true</tt>; ZOBOV and DTFE additionally need CGAL. See <a href="https://github.com/HAWinther/FML/blob/master/FML/SnapshotAnalyzer/README.md" style="display: inline">the complete build and parameter guide</a>.</p><div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">make -C FML/SnapshotAnalyzer
mpirun -np 4 FML/SnapshotAnalyzer/snapshot_analyzer analysis.lua</pre></div></div>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Analyses">Available analyses</h1>
</header>
<hr/>

<div class="table-responsive"><table class="table table-striped table-condensed">
<thead><tr><th>Analysis</th><th>Settings / outputs</th></tr></thead><tbody>
<tr><td>Power, redshift-space multipoles and bispectrum</td><td><tt>pofk_*</tt>, <tt>pofk_multipole_*</tt>, <tt>bispectrum_*</tt>; mass weighting, assignment and interlacing options.</td></tr>
<tr><td>Standard or relaxed FoF and spherical overdensity</td><td><tt>fof_*</tt>, <tt>fof_relaxed</tt>, <tt>fof_so_definitions</tt>; catalogues, mass functions and overlap flags.</td></tr>
<tr><td>T-web and Minkowski functionals</td><td><tt>cosmicweb_*</tt>, <tt>minkowski_*</tt>; web mass/volume fractions and excursion-set morphology with Gaussian-field predictions.</td></tr>
<tr><td>Hybrid Lagrangian bias spectra</td><td><tt>hybridbias_*</tt>; 15 advected operator cross-spectra with a supplied or recovered initial linear field, and with <tt>hybridbias_rsd</tt> their redshift-space multipoles.</td></tr>
<tr><td>Images</td><td><tt>images_*</tt>; PNG (and <tt>.npy</tt>) of a projected slab of the density and the velocity divergence, as in SimulationCode.</td></tr>
<tr><td>Voids, DTFE and baryonification</td><td>Voxel watershed, spherical underdensity and optional ZOBOV; optional DTFE density/velocity fields and spectra; position-only baryonification and optional corrected Gadget output.</td></tr>
</tbody></table></div>
<p>The shared <tt>FML::ANALYSIS</tt> headers provide the context, settings and measurement/output routines. Power, multipoles, bispectrum, FoF/SO and baryonification share algorithms and parameter names with SimulationCode. Extra standalone analyses and their requirements are documented here rather than implying that every option is wired into in-run analysis.</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Usage">A small Lua parameter file</h1>
</header>
<hr/>

<div style="text-align: center"><div style="max-width:800px;display:inline-block;text-align:left;width:100%"><pre class="prettyprint">snapshot_input = &quot;./snapshot_001/snapshot_001&quot;
snapshot_format = &quot;gadget&quot; -- gadget_hdf5 with USE_HDF5=true
snapshot_read_mode = &quot;files_per_task&quot;
particle_types = {}        -- all types; use {1} for type 1 only
snapshot_length_unit_mpch = 1.0
snapshot_mass_unit_msunh = 1e10
output_folder = &quot;.&quot;
pofk = true
pofk_nmesh = 256            -- divisible by the MPI rank count
pofk_interlacing = true
pofk_subtract_shotnoise = false
fof = false
cosmicweb = true
cosmicweb_nmesh = 128
cosmicweb_smoothing_scale_mpch = 5.0
minkowski = true
minkowski_nmesh = 128
minkowski_smoothing_scale_mpch = 5.0</pre></div></div>
<p>Set the snapshot length and mass units explicitly. The reader applies the Gadget velocity convention. The analysis background defaults to LCDM from the snapshot header, with optional radiation; a table of $a,H(a)/H_0$ can supply another expansion history. Selected-particle mass sets the mean density used for analysis. Outputs use the SimulationCode redshift naming convention, such as <tt>pofk_z0.000.txt</tt>.</p>

<p>
<b>Pencils.</b> <tt>make USE_PENCIL_GRID=true</tt> runs every analysis on <a href="pencilgrid.php" style="display: inline">pencils</a> (from <tt>FML_PENCILS="px py"</tt> or as square as possible); the meshes must then be divisible by <tt>px</tt> and <tt>py</tt> instead of the rank count. On slabs the outputs are the same as before.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="HybridBias">Recovering the linear field for hybrid bias</h1>
</header>
<hr/>

<p>The five Lagrangian operators are $1,\delta_L,\delta_L^2-\langle\delta_L^2\rangle,s^2-\langle s^2\rangle,\nabla^2\delta_L$. Their 15 cross-spectra require the initial lattice and particle identities. Use an early IC snapshot via <tt>hybridbias_ic_snapshot</tt>, with a 2LPT correction when appropriate, or provide a real-space lattice field through <tt>hybridbias_linear_field_source = "file"</tt> and <tt>hybridbias_linear_field_file</tt>. SimulationCode can write that field with <tt>hybridbias_output_linear_field = true</tt>. With <tt>hybridbias_rsd = true</tt> the operators are also assigned at the redshift-space positions of the particles (so the snapshot needs velocities) and the multipoles up to <tt>hybridbias_rsd_ellmax</tt> are written to <tt>hybridbias_rsd_z&lt;z&gt;.txt</tt>; on a SimulationCode run they equal the in-run ones to $10^{-5}$&ndash;$4\cdot 10^{-4}$.</p>
<p>IC positions must uniquely identify lattice points within half a cell. File input uses the documented ID-to-lattice mapping, or an IC snapshot for arbitrary IDs. Growth rescaling defaults to the analysis background's scale-independent ratio; use <tt>hybridbias_growth_ratio</tt> explicitly when this is inappropriate. See <a href="https://github.com/HAWinther/FML/blob/master/FML/SnapshotAnalyzer/README.md" style="display: inline">input formats, matching rules and validation</a>. For morphology, smooth over several cells and refine resolution; the painted density retains its assignment window.</p>

