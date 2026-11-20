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
