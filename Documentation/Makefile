# Build the FML book: make (pdf), make clean, make watch
MAIN = main
all: $(MAIN).pdf
$(MAIN).pdf: $(MAIN).tex preamble.tex references.bib chapters/*.tex chapters/B_simulationcode_parameters.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

# The appendix of SimulationCode parameters is generated from PARAMETERS.md (itself written by ./nbody --parameters)
chapters/B_simulationcode_parameters.tex: ../FML/SimulationCode/PARAMETERS.md tools/simcode_parameters.py
	python3 tools/simcode_parameters.py $< $@
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex
clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).ind $(MAIN).ilg $(MAIN).idx
.PHONY: all watch clean
