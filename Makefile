.PHONY: test test-core test-scheduler test-all bdd-up bdd-down bdd-core bdd-scheduler bdd-all bdd-http-up bdd-http-down bdd-http

test-core:
	$(MAKE) -C biomed_knowledge_platform test

test-scheduler:
	$(MAKE) -C scheduler_pubmed test

test: test-core test-scheduler

test-all: test bdd-all

bdd-up:
	$(MAKE) -C tests bdd-up

bdd-down:
	$(MAKE) -C tests bdd-down

bdd-core:
	$(MAKE) -C tests bdd-core

bdd-scheduler:
	$(MAKE) -C tests bdd-scheduler

bdd-all:
	$(MAKE) -C tests bdd

bdd-http-up:
	$(MAKE) -C tests bdd-http-up

bdd-http-down:
	$(MAKE) -C tests bdd-http-down

bdd-http:
	$(MAKE) -C tests bdd-http
