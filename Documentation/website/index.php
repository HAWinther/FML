<!DOCTYPE html>
<html lang="en-US">

  <head>

  <!-- Include head from file --> 
  <?php include('head.php'); ?>

  <!-- Set title and description -->
  <title>FML: a C++ library for large-scale-structure cosmology, from parallel grids and particles to simulations, analysis and inference.</title>
  <meta name="description" content="A C++ library for large-scale-structure cosmology: parallel grids and particles on slab, pencil and Peano-Hilbert domains, solvers, simulations, analysis and inference, for any cosmological model.">
  <meta name="keywords" content="Cosmology, Numerical cosmology, N-body, C++, multigrid solver, power-spectrum, bispectrum, trispectrum, polyspectrum, pair counting, two point correlation function, FFTLog, FFTW, fourier transforms, density assignment, NGP, CIC, TSC, PCS, tessellation, DTFE, friends of friends, halo finder, void finder, ZOBOV, ORIGAMI, cosmic web, watershed, non-gaussianity, initial conditions, MPI, OpenMP, COLA simulations, LPT, Lagrangian perturbation theory, smoothing, gaussian filter, tophat filter, sharp-k filter">

  </head>

  <body>

  <!-- Include page header from file --> 
  <?php include('header.php'); ?>

  <!-- Wrap main content to get footer to stick to bottom -->
  <div class="page-wrap">
    <main id="main" class="site-main" role="main">
    <div class="container">
      <div class="row">
        <div class="col-xs-12 col-md-12 col-lg-12 body-text-markup">

          <!-- Include page content -->
          <?php include('index_content.php'); ?>

          <hr/>
        </div>
      </div>
    </div>
    </main>
  </div>

  <!-- Include page footer from file --> 
  <?php include('footer.php'); ?>
  
  </body>
</html>
