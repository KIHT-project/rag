.PHONY: test test-core test-scheduler test

test-core:
	$(MAKE) -C biomed_knowledge_platform test

test-scheduler:
	$(MAKE) -C scheduler_pubmed test

test: test-core test-scheduler

LATEX_SRC=docs/thesis/latex/src
LATEX_OUT=docs/thesis/latex/out
MAIN=main.tex

latex-build:
	mkdir -p $(LATEX_OUT)
	pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
	-output-format=pdf \
	-output-directory=$(LATEX_OUT) \
	$(LATEX_SRC)/$(MAIN)

latex-clean:
	rm -rf $(LATEX_OUT)/*

latex-open: latex-build
	open $(LATEX_OUT)/main.pdf