<!DOCTYPE html>
<html lang="en-US">

  <head>

  <!-- Include head from file --> 
  <?php include('head.php'); ?>

  <!-- Set title and description -->
  <title>FML::Documentation</title>
  <meta name="description" content="Documentation of the FML C++ library.">
  <meta name="keywords" content="Cosmology, Numerical cosmology, N-body, C++, multigrid solver, power-spectrum, bispectrum, trispetrum, polyspectrum, pair counting, two point correlation function, FFTLog, FFTW, fourier transforms, density assignment, NGP, CIC, TSC, PCS, tesselation, friend of friends, halo finder, void finder, watershed, non-gaussianity, initial conditions, MPI, OpenMP, COLA simulations, LPT, Lagrangian perturbation theory, smoothing, gaussian filter, tophat filter, sharp-k filter">

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
          <?php include('documentation_content.php'); ?>

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
