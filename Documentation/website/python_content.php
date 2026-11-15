<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Analysis">1. The fml-analysis Python package</a><br>
<a href="#Swig">2. Making your own wrappers with SWIG</a><br>
<a href="#Halomodel">3. The halo-model wrapper</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Analysis">The fml-analysis Python package</h1>
</header>
<hr/>

<p>
<tt>FML/PythonAnalysis</tt> is a pip-installable package (pybind11, no FFTW/GSL/MPI needed) that exposes a serial 3D catalogue backed by the C++ <a href="statistics.php#kNN" style="display: inline">kNN and counts-in-cells</a> code: $k$-nearest-neighbour distances and CDFs, sphere counts, and counts-in-cells summaries (occupancy and density PDFs, factorial moments), with periodic or non-periodic geometry and reproducible Philox query points. Inputs are copied into C++-owned storage, the GIL is released during the computation and the results are ordinary NumPy arrays.
</p>
<div style="text-align: center">
  <div style="width:700px;display: inline-block;text-align: left">
    <pre class="prettyprint">
# python3 -m pip install ./FML/PythonAnalysis
import numpy as np
from fml_analysis import Catalogue, uniform_queries

positions = np.random.rand(10000, 3)
cat = Catalogue(positions, boxsize=1.0)
queries = uniform_queries(100000, [[0, 0, 0], [1, 1, 1]], seed=17)
distances = cat.knn_distances(queries, [1, 2])                # shape (n_orders, n_queries)
cdf       = cat.knn_cdf(queries, [1, 2, 4], [0.02, 0.05, 0.1])   # dict: cdf, hits, ...
summary   = cat.counts_in_cells(queries, [0.05, 0.1])         # PDFs and factorial moments per radius</pre>
  </div>
</div>
<p>
The tests compare everything with independent NumPy brute-force implementations.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Swig">Making your own wrappers with SWIG</h1>
</header>
<hr/>

<p>
<tt>FML/MakePythonWrapper</tt> shows how to expose any C++ function of the library to Python with SWIG and <tt>numpy.i</tt>: write the functions in a C++ file, list them in the interface file, and <tt>make</tt> produces a Python module (see <tt>test.py</tt>). <tt>make -C FML python</tt> builds the SWIG wrappers of the library.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Halomodel">The halo-model wrapper</h1>
</header>
<hr/>

<p>
<tt>FML/HaloModel/PythonWrapper</tt> wraps the <a href="halomodel.php#Halomodel" style="display: inline">halo model and HMCode</a> to compute non-linear power spectra from Python.
</p>
