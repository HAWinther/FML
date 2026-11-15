<hr/>
<header class="entry-header text-center">
  <h1 class="entry-title">Table of contents</h1>
</header>
<hr/>

<p class="page-toc">
<a href="index.php">Back to the main page</a><br>
<a href="documentation.php">Back to the documentation</a><br>
<a href="#Overview">1. Overview: Tidal tensor classification</a><br>
<a href="#Methods">2. Available methods</a><br>
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Overview">Overview: Tidal tensor classification</h1>
</header>
<hr/>  

<p>
The cosmic web can be classified (void, sheet, filament, halo) by the eigenvalues (and eigenvectors) of the tidal tensor (see e.g. Section 3.3 of <a href="https://arxiv.org/pdf/1702.06817.pdf" style="display: inline">this</a>). We provide a few methods that are useful for computing such a classification (namespace FML::HESSIAN in TidalTensor/Hessian.h). A ready made T-web/V-web classification built on these is described on the <a href="cosmicweb.php" style="display: inline">cosmic web</a> page.
</p>

<header class="entry-header text-center">
  <h1 class="entry-title" id="Methods">Available methods</h1>
</header>
<hr/>  
<p>
<div style="text-align: center">
  <div style="width:600px;display: inline-block;text-align: left">
    <pre class="prettyprint">
//=================================================================================
// Computes the Hessian matrix of a grid [norm * f] via Fourier transforms.
// If hessian_of_potential_of_f is true then we compute the Hessian 
// phi_ij where D^2 phi = norm * f_real
// Since f_ij = f_ji we only compute the elements for j &gt;= i and they are stored in
// the order fxx fxy ... fyy fyz ... etc. in hessian_real
// In 2D: [fxx fxy fyy]
// In 3D: [fxx fxy fxz fyy fyz fzz]
//=================================================================================
template&lt;int N&gt;
  void ComputeHessianWithFT(
    const FFTWGrid&lt;N&gt; &amp; f_real,
    std::vector&lt; FFTWGrid&lt;N&gt; &gt; &amp; hessian_real,
    double norm = 1.0,
    bool hessian_of_potential_of_f = false);

//=================================================================================
// For each point in the grid compute eigenvectors and eigenvalues of the tensor
// H_ij where tensor_real contains the N(N+1)/2 grids [ 00,01,02,..,11,12,...,NN ]
//
// Eigenvalues are ordered in descending order 
// Eigenvectors are stored in row major order in the grid vector
// This allocates N grids if compute_eigenvectors = false and N(N+1) grids otherwise
//=================================================================================
template&lt;int N&gt;
  void SymmetricTensorEigensystem(
    std::vector&lt; FFTWGrid&lt;N&gt; &gt; &amp; tensor_real,
    std::vector&lt; FFTWGrid&lt;N&gt; &gt; &amp; eigenvalues,
    std::vector&lt; FFTWGrid&lt;N&gt; &gt; &amp; eigenvectors,
    bool compute_eigenvectors = false);

// Example: the tidal tensor phi_ij with D^2 phi = delta and its eigenvalues
std::vector&lt;FFTWGrid&lt;3&gt;&gt; tidal_tensor, eigenvalues, eigenvectors;
FML::HESSIAN::ComputeHessianWithFT(delta_real, tidal_tensor, 1.0, true);
FML::HESSIAN::SymmetricTensorEigensystem(tidal_tensor, eigenvalues, eigenvectors);

</pre>
</div>
</div>
</p>
