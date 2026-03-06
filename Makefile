.PHONY: test test-core test-scheduler test latex-build latex-clean latex-open latex-rebuild latex-run

test-core:
	$(MAKE) -C biomed_knowledge_platform test

test-scheduler:
	$(MAKE) -C scheduler_pubmed test

test: test-core test-scheduler

LATEX_SRC := docs/thesis/latex/src
LATEX_OUT := docs/thesis/latex/out
MAIN := main.tex
PDF := $(abspath $(LATEX_OUT))/$(MAIN:.tex=.pdf)

latex-build:
	mkdir -p $(LATEX_OUT)
	cd $(LATEX_SRC) && pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
	-output-format=pdf \
	-output-directory=$(abspath $(LATEX_OUT)) \
	$(MAIN)
	cd $(LATEX_SRC) && pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
	-output-format=pdf \
	-output-directory=$(abspath $(LATEX_OUT)) \
	$(MAIN)

latex-clean:
	rm -rf $(LATEX_OUT)/*

latex-open: latex-build
	open $(PDF)

latex-rebuild: latex-clean latex-build

latex-run: latex-rebuild latex-open