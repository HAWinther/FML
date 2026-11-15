<!DOCTYPE html>
<html lang="en-US">

  <head>

  <!-- Include head from file -->
  <?php include('head.php'); ?>

  <!-- Set title and description -->
  <title>FML SnapshotAnalyzer: Gadget snapshot analysis</title>
  <meta name="description" content="Standalone MPI analysis of Gadget snapshots: spectra, halos, voids, hybrid bias, cosmic web and morphology">

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
          <?php include('snapshotanalyzer_content.php'); ?>

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
