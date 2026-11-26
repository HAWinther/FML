# The FML website

The source of https://fml.wintherscoming.no/ (PHP). Every page `X.php` includes the shared `head.php`, `header.php`
and `footer.php` around its content in `X_content.php`; images are in `images/`, styles in `style/`.

Preview it locally with PHP's built-in server and open http://localhost:8000:

```bash
php -S localhost:8000 -t Documentation/website     # from the repository root
```

Deploy by copying the folder to the web server. The Doxygen reference (`../doxygen`) is uploaded to its `doxygen/`
folder, and the book (`../main.pdf`, built with `make` in `Documentation/`) next to `index.php` as `main.pdf`, where
`header.php`, `index_content.php` and `documentation_content.php` link to it.

The SimulationCode validation reports linked from `nbodycode_content.php` (section "Validation reports") are expected in
`reports/` next to `index.php`, with the file names they have in the repository's top-level `Reports/` folder
(flat: `linear_step_report.pdf`, `high_redshift_starts.pdf`, `item59_phase1.pdf`, `item59_phase2.pdf`,
`item59_phase3.pdf`, `halos_pgd.pdf`). When a report is added, add a row to that table.

Table of contents: every page starts with a "Table of contents" heading and a `<p class="page-toc">` with the two
"Back to" links (main page, documentation; not on the front page) and then one numbered entry per section
(`<h1 class="entry-title" id="...">`), `1. ...`, `2. ...`, in the order of the sections on the page. When a section is
added, add its entry at that position and renumber the ones after it.

