<!DOCTYPE html>
<html lang="en-US">

  <head>

  <!-- Include head from file --> 
  <?php include('head.php'); ?>

  <!-- Set title and description -->

  <title>FML::COSMOLOGY::LPT</title>
  <meta name="description" content="Lagrangian perturbation theory. Compute displacement fields. 1LPT. 2LPT. Density reconstruction. RSD reconstruction.">

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
          <?php include('lpt_content.php'); ?>

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
