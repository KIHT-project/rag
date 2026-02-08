# ========= Testing =========
PYTHON ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; else echo python; fi)
PYTEST=$(PYTHON) -m pytest
COV_MODULE=src
REPORT_DIR=tests_output.ignore
COV_DIR=$(REPORT_DIR)/coverage
COV_XML=$(COV_DIR)/coverage.xml
COV_HTML=$(COV_DIR)
JUNIT_XML=$(REPORT_DIR)/junit.xml
HTML_REPORT=$(REPORT_DIR)/test_report.html
COV_BADGE=$(REPORT_DIR)/coverage.svg

export COVERAGE_FILE=$(REPORT_DIR)/.coverage

.PHONY: test clean output format lint check

test: clean $(REPORT_DIR)
	$(PYTEST) \
		--cov=$(COV_MODULE) \
		--cov-report=term \
		--cov-config=coverage.ini \
		--cov-report=html:$(COV_HTML) \
		--cov-report=xml:$(COV_XML) \
		--cov-fail-under=80 \
		--junitxml=$(JUNIT_XML) \
		--html=$(HTML_REPORT) \
		--self-contained-html
	$(PYTHON) -m coverage_badge -o $(COV_BADGE) -f

$(REPORT_DIR):
	mkdir -p $(COV_DIR)

clean:
	rm -rf $(REPORT_DIR)

# ========= Code Format =========
SRC = src

format:
	$(PYTHON) -m black $(SRC) --line-length=100

lint:
	$(PYTHON) -m flake8 $(SRC) --max-complexity=10 --max-line-length=100 --statistics

check:
	$(PYTHON) -m black --check $(SRC) --line-length=100
	$(PYTHON) -m flake8 $(SRC) --max-complexity=10 --max-line-length=100 --statistics
