# PubMed Scheduler Service

Standalone service scaffold for PubMed scheduling/orchestration. This project is isolated from the main service and is designed to run in its own deployment and PostgreSQL database.

## Current scope
- Project structure for a standalone async FastAPI service
- Dockerization for independent deployment
- Dedicated PostgreSQL container and schema bootstrap
- Scheduler configuration file (`config/pubmed_scheduler.yaml`)
- Alembic migrations for `pubmed_scheduler` schema
- Dummy app exposing healthcheck endpoints only

Business logic for PubMed execution and orchestration is intentionally deferred.

## Run with Docker
From this folder:

```bash
docker compose up --build
```

Service endpoints:
- API: `http://localhost:9000`
- OpenAPI docs: `http://localhost:9000/docs`

## Run migrations
From this folder:

```bash
python -m alembic -c alembic.ini upgrade head
```

Optional rollback:

```bash
python -m alembic -c alembic.ini downgrade -1
```

## Startup migrations
The API now runs `alembic upgrade head` during FastAPI startup (lifespan), same pattern as the main project.

Disable this behavior when needed (for example tests):

```bash
export PUBMED_SCHEDULER_RUN_MIGRATIONS_ON_STARTUP=false
```

## Unit test workflow (same approach as biomed platform)
From this folder:

```bash
make test
```

This runs pytest with:
- coverage threshold: `80%`
- JUnit report: `tests_output.ignore/junit.xml`
- HTML test report: `tests_output.ignore/test_report.html`
- coverage XML: `tests_output.ignore/coverage/coverage.xml`
- coverage HTML: `tests_output.ignore/coverage/index.html`
- coverage badge: `tests_output.ignore/coverage.svg`

Cleanup reports:

```bash
make clean
```


Model generation:
```shell
python -m datamodel_code_generator \
          --input scheduler_pubmed/config/api.yaml \
          --input-file-type openapi \
          --output scheduler_pubmed/src/api/models/schemas.py \
          --output-model-type pydantic_v2.BaseModel \
          --use-standard-collections \
          --use-union-operator \
          --enum-field-as-literal all \
          --target-python-version 3.14 \
          --collapse-root-models \
          --strict-nullable
```

Docke run:
```shell
docker build -t scheduler-pubmed:latest .
docker run --rm -p 7999:8000 scheduler-pubmed:latest
```
