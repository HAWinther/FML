<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<!-- Activating Header -->
<link rel='stylesheet' id='bootstrap-css' href='style/bootstrap.min.css?ver=3.3.6' type='text/css' media='all' />
<link rel='stylesheet' id='gcsn-style-css' href='style/theme.css' type='text/css' media='all' />

<link rel="canonical" href="#" />
<link rel='shortlink' href='#' />

<!-- Load MathJax -->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>

<!-- Google Analytics: the old Universal Analytics property (UA-64567090-3) stopped collecting data in 2023.
     To use analytics again create a GA4 property and put its measurement id (G-XXXXXXXXXX) below. -->
<!--
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'G-XXXXXXXXXX');
</script>
-->

<?php

if (($_SERVER['REMOTE_ADDR'] ?? '') == "13.66.139.158") die();

// Do a page load count (total for the whole site)
$myip1 = "84.209.184.106";
$myip2 = "37.191.179.126";
$myip3 = "89.162.118.135";
$myip_head_work = "193.157";
$counter_file = '/home/5/w/wintherscoming/fml/counter.txt';
$visitor_file = '/home/5/w/wintherscoming/fml/visitor.txt';

// Remove everything that is not printable from user provided strings before we write them to the log
function fml_clean_log_field($str, $maxlen = 512) {
  $str = preg_replace('/[^\x20-\x7E]/', '', (string) $str);
  return substr(str_replace(' ', '%20', $str), 0, $maxlen);
}

if (session_status() === PHP_SESSION_NONE and !headers_sent()) session_start();
if (empty($_SESSION['visited'])) {
  $ip = $_SERVER['HTTP_CLIENT_IP'] ?? ($_SERVER['HTTP_X_FORWARDED_FOR'] ?? ($_SERVER['REMOTE_ADDR'] ?? ''));
  // X-Forwarded-For can be a list "client, proxy1, proxy2"
  $ip = fml_clean_log_field(trim(explode(',', $ip)[0]), 64);
  $ip_head = substr($ip, 0, 7);
  if ($ip != $myip1 and $ip != $myip2 and $ip != $myip3 and $ip_head != $myip_head_work) {
    $counter = (int) @file_get_contents($counter_file) + 1;
    @file_put_contents($counter_file, $counter, LOCK_EX);
    // Add some user info statistics
    if ($counter < 100000) {
      $site = fml_clean_log_field($_SERVER['REQUEST_URI'] ?? '');
      $date = date('Y-m-d H:i:s');
      $ref  = fml_clean_log_field($_SERVER['HTTP_REFERER'] ?? '');
      @file_put_contents($visitor_file, "$ip $date $site $ref\n", FILE_APPEND | LOCK_EX);
    }
  }
}
$_SESSION['visited'] = true;
?>

<!-- Google code prettify -->
<script src="https://cdn.jsdelivr.net/gh/google/code-prettify@master/loader/run_prettify.js"></script>
<style>
  .prettyprint ol.linenums > li { list-style-type: decimal; }
</style>  
