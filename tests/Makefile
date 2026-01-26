.PHONY: bdd-up bdd-down bdd

bdd-up:
	docker-compose -p biomed-bdd -f bdd/docker-compose.yaml up -d

bdd-down:
	docker-compose -p biomed-bdd -f bdd/docker-compose.yaml down -v

bdd:
	pytest bdd -v