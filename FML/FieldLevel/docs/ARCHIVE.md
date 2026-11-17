# Historical FieldLevel archive

The cleanup preserved all 1141 original files, including uncommitted
sources/results, failed studies, raw chains, predictions, plots and build products.
The full source/layout is available as `FieldLevel-before-cleanup.zip`, archive ID
`2026-11-17_202326`. Every extracted file and ZIP member was checked against SHA256.

ZIP SHA256: `15a63f59e86ad9f765a83d8355e01841ad32067e54b4987abfc220a294abca22`

The archive is external to the library checkout. On the originating machine it
is under `/Users/hansw/Claude/FieldLevel/archives/2026-11-17_202326`. On another machine,
copy that archive folder to your scientific-data storage. It is not an automatic
or hidden build dependency. Metadata is in [archive.json](validation/archive.json).

The archive contains `snapshot/FieldLevel/` with the original sibling layout,
`manifest.json` with all original file hashes, the base Git commit, the pending
tracked diff, build settings and a reconstruction README. To rebuild historical
programs use an isolated checkout at that commit and restore the snapshot there.
Do not restore it over the cleaned active checkout. Reanalysis scripts that only
read saved files can use the original archived layout directly.

The maintained validation drivers and ordinary library tests are self-contained.
Only two small frozen posterior input fixtures remain in the source tree.
Historical report links to raw files or figures now point to this guide; their
original paths are shown in the report and listed in the archive manifest.
All failed outcomes are preserved; cleanup does not supersede numerical evidence.
