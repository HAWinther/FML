<!DOCTYPE html>
<html lang="en-US">

  <head>

  <!-- Include head from file --> 
  <?php include('head.php'); ?>

  <!-- Set title and description -->
  <title>FML::HYDRODYNAMICS and FML::SPH: finite-volume Euler equations and SPH</title>
  <meta name="description" content="Riemann solvers and a conservative finite-volume Euler solver, and conservative smoothed particle hydrodynamics, with MPI and OpenMP">

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
          <?php include('hydro_content.php'); ?>

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
