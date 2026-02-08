.PHONY: test test-core test-scheduler test

test-core:
	$(MAKE) -C biomed_knowledge_platform test

test-scheduler:
	$(MAKE) -C scheduler_pubmed test

test: test-core test-scheduler
