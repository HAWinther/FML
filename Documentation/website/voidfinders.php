<!DOCTYPE html>
<html lang="en-US">

  <head>

  <!-- Include head from file --> 
  <?php include('head.php'); ?>

  <!-- Set title and description -->
  <title>FML::VOIDFINDER</title>
  <meta name="description" content="Void finders: ZOBOV, voxel watershed and spherical underdensities">

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
          <?php include('voidfinders_content.php'); ?>

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
