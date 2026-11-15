# Build the FML book: make (pdf), make clean, make watch
MAIN = main
all: $(MAIN).pdf
$(MAIN).pdf: $(MAIN).tex preamble.tex references.bib chapters/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex
clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).ind $(MAIN).ilg $(MAIN).idx
.PHONY: all watch clean
