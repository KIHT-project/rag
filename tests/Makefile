.PHONY: bdd-up bdd-down bdd bdd-core bdd-scheduler bdd-http-up bdd-http-down bdd-http

PYTHON ?= $(shell if [ -x ../.venv/bin/python ]; then echo ../.venv/bin/python; else echo python; fi)
PYTEST=$(PYTHON) -m pytest

bdd-up:
	docker-compose -p biomed-bdd -f bdd/docker-compose.yaml up -d

bdd-http-up:
	docker-compose -p biomed-bdd -f bdd/docker-compose.yaml --profile http up -d

bdd-down:
	docker-compose -p biomed-bdd -f bdd/docker-compose.yaml down -v

bdd-http-down:
	docker-compose -p biomed-bdd -f bdd/docker-compose.yaml down -v

bdd:
	$(PYTEST) bdd -v

bdd-core:
	$(PYTEST) bdd -v -m "not scheduler"

bdd-scheduler:
	$(PYTEST) bdd -v -m "scheduler"

bdd-http:
	BDD_HTTP_MODE=true $(PYTEST) bdd -v -m "http"
