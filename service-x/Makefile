# ========= Testing =========
PYTEST=pytest
COV_MODULE=src
REPORT_DIR=output
COV_DIR=$(REPORT_DIR)/coverage
COV_XML=$(COV_DIR)/coverage.xml
COV_HTML=$(COV_DIR)
JUNIT_XML=$(REPORT_DIR)/junit.xml
HTML_REPORT=$(REPORT_DIR)/test_report.html
COV_BADGE=$(REPORT_DIR)/coverage.svg

.PHONY: test clean output

test: clean $(REPORT_DIR)
	$(PYTEST) \
		--cov=$(COV_MODULE) \
		--cov-report=term \
		--cov-report=html:$(COV_HTML) \
		--cov-report=xml:$(COV_XML) \
		--cov-fail-under=80 \
		--junitxml=$(JUNIT_XML) \
		--html=$(HTML_REPORT) \
		--self-contained-html
	coverage-badge -o $(COV_BADGE) -f

$(REPORT_DIR):
	mkdir -p $(COV_DIR)

clean:
	rm -rf $(REPORT_DIR)

# ========= Code Format =========
SRC = src

format:
	black $(SRC) --line-length=90

lint:
	flake8 $(SRC) --max-complexity=10 --max-line-length=90 --statistics

check:
	black --check $(SRC) --line-length=90
	flake8 $(SRC) --max-complexity=10 --max-line-length=90 --statistics
