# The FML book

LaTeX source of the comprehensive documentation of FML: the equations each module solves, the algorithms, how to use
it, how it is validated and the references. Build with `make` (latexmk; needs a standard TeX Live with pdflatex,
bibtex and makeindex) or `pdflatex main; bibtex main; makeindex main; pdflatex main; pdflatex main`.
Output: `main.pdf`.

The other documentation of FML is next to the book: the website in [`website/`](website/README.md) and the Doxygen
API reference in `doxygen/` (`./run_doxygen.sh` there).

## Layout

| File | Content |
|---|---|
| `main.tex` | Parts and chapter order |
| `preamble.tex` | Packages, boxes (`modulebox`, `implementation`, `validation`, `caveat`, `planned`), listings (`cppcode`, `paramfile`), markup (`\file`, `\code`, `\cls`, `\param` (indexed), `\arxiv`, `\github`) and physics macros |
| `references.bib` | Bibliography (natbib, `unsrtnat`); keys `FirstAuthorYear`; every entry has an eprint or doi |
| `chapters/NN_name.tex` | One chapter per file; `A_module_index.tex` maps every module directory to its chapters |
| `figures/` | Figures made for the book. Figures of the Einstein-Boltzmann solver report are used in place from `../FML/Cosmology/EinsteinBoltzmannSolver/report/figures/` (see `\graphicspath`) |

## Conventions for writing a chapter

- First line under the title: `\chapterstatus{written|outline}{date}`. Outline chapters list the planned content in
  `planned` boxes; replace them when the chapter is written.
- Start with a `modulebox` listing the headers and module docs the chapter covers.
- For every feature: the equations (with the library's conventions, Chapter 2), the algorithm and numerical choices
  (`implementation` box), the main classes/functions and a short `cppcode` example or `paramfile`, the
  parameters (`\param{...}`, indexed), the validation (`validation` box) and references (`\cite`, add to
  `references.bib`; use `\arxiv{...}` for a reference whose bibliographic data is not checked).
- Take facts from the code and the module READMEs, not from memory; the module READMEs remain authoritative for
  interfaces and test procedures.

## Status

All 39 chapters and the appendix are written (2026-11-17; 247 pages). The build has no errors, unresolved references or
overfull boxes. When a module changes, update its chapter (facts, defaults, validation numbers) and the date in
`\chapterstatus`.

| Part | Chapters |
|---|---|
| I Introduction | 1 Introduction, 2 Conventions |
| II Infrastructure | 3 Parallelism, 4 Grids and FFTs, 5 Particles, 6 Mass assignment/NUFFT, 7 Numerical methods, 8 Multigrid, 9 I/O and utilities |
| III Linear theory | 10 Background, 11 Recombination, 12 Linear perturbations, 13 CMB and matter spectra, 14 Extended models, 15 Growth and transfer functions |
| IV Nonlinear models | 16 Perturbation theory, 17 Spherical collapse, 18 Halo model, 19 Fitting functions and emulators |
| V Initial conditions | 20 Random fields, 21 LPT and initial conditions |
| VI Simulations | 22 PM/COLA, 23 Short-range forces, 24 SimulationCode, 25 Modified gravity and dark sector, 26 Gas and wave dark matter, 27 PGD and baryonification |
| VII Structures | 28 Halo finding, 29 Cosmic web, 30 Voids and tessellations |
| VIII Statistics | 31 Power spectra, 32 Configuration space, 33 Four-point and ensembles, 34 Galaxy mocks |
| IX Surveys and inference | 35 Surveys, 36 Weak lensing, 37 Field-level inference, 38 Sampling and fitting |
| X Applications | 39 Snapshot analysis |
| Appendix | A Module index |
