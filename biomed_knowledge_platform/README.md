# Service X

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [API Schema Generation](#api-schema-generation)
4. [Exception Handling](#exception-handling)
5. [Core Business/Domain Concepts](#core-businessdomain-concepts)
6. [Retrieval Semantics](#retrieval-semantics)
7. [Local Run](#local-run)

## Architecture Overview
The system follows Clean Architecture with the addition of Ports and Adapters.
### Layer responsibilities
The API layer is located under `biomed_platform/api`
Responsibilities:
* HTTP request parsing
* DTO validation
* Delegation to use cases
* Mapping domain results to API schemas
* No business logic
* No database or Qdrant imports

### Application layer (Use Cases)
The application layer is located under `biomed_platform/core/use_cases`
Responsibilities:
* Orchestrate workflows
* Implement business logic
* Coordinate ports
* Decide error semantics

### Domain layer
The domain layer is located under `biomed_platform/core/domains`
Responsibilities:
* Domain entities
* Value objects
* Retrieval result structures
* No framework dependencies

### Ports and Adapters
Ports are interfaces that define the boundaries between layers.
Adapters are implementations of ports.
Ports and adapters are located under `biomed_platform/core/ports` and `biomed_platform/adapters`.
Port Responsibilities:
* Define required capabilities
* Dependency inversion boundary
* No infrastructure knowledge

Adapters Responsibilities:
* Implement ports
* No business logic
* No domain knowledge
* Infrastructure integration
* Catch vendor (QDrant) exceptions and raise typed domain errors

### Services layer
Services are processes that run in the background.
They are located under `biomed_platform/core/services`.
Responsibilities:
* Background workers
* Ingestion pipelines
* In memory implementations
* Backpressure handling

## Project Structure:
```
biomed_knowledge_platform/
├──alembic
│  ├──versions/...
│  ├──env.py
├── Dockerfile
├── docker-compose.yaml
├── Makefile
├── README.md
├── requirements.txt
├── requirements-embeddings.txt
├── configs/
│   ├── api.yaml
│   ├── llm.yaml
│   ├── logging.yaml
│   ├── qdrant.yaml
│   ├── rag.yaml
│   └── swagger.yaml
├── src/
│   └── biomed_platform/
│       ├── api/                       # Transport layer (FastAPI)
│       │   ├── app.py                 # Composition root and wiring
│       │   ├── router.py
│       │   ├── endpoints/             # HTTP endpoints (thin)
│       │   ├── error_handlers.py      # Centralized exception mapping
│       │   ├── mappers/               # Domain → API DTO mappers
│       │   └── models/generated/      # OpenAPI generated schemas
│       ├── core/
│       │   ├── domains/               # Pure domain models
│       │   ├── ports/                 # Interfaces (protocols)
│       │   ├── use_cases/             # Application services
│       │   ├── services/              # Long running workers and pipelines
│       │   └── errors/                # Typed domain errors
│       ├── adapters/
│       │   └── qdrant/                # Infrastructure adapters
│       └── common/                    # Logging, middleware, utils
│       └── db/                        # DB configuration and models
├── tests/
│   ├── api/
│   ├── core/
│   └── common/
```

## API Schema Generation
OpenAPI models are generated from the Swagger specification.

```shell
python -m datamodel_code_generator \
  --input configs/swagger.yaml \
  --input-file-type openapi \
  --output src/biomed_platform/api/models/generated/schemas.py \
  --output-model-type pydantic_v2.BaseModel \
  --use-standard-collections \
  --use-union-operator \
  --use-annotated \
  --disable-timestamp \
  --target-python-version 3.14
```
## Exception Handling
The system uses typed domain errors.
Examples:
* ValidationError
* NotFoundError
* ConflictError
* RateLimitError
* DependencyUnavailableError
* InternalError

All exceptions are:
* Raised in core or adapters
* Handled centrally in the API layer
* Returned with structured JSON including:
  * status code
  * error type
  * message
  * details
  * request_id
  * timestamp (UTC)

**Endpoints never catch exceptions.**

## Core Business/Domain Concepts

### Document identity

Each document is identified by:
```shell
doc_id = hash(doi_normalized, content_source)
```

This allows:
* Unique document identity
* Fast retrieval by DOI

### Chunk based indexing
* Each document is chunked
* Each chunk is stored as a separate vector
* Payload contains metadata and text
* Retrieval operates at chunk level
* Results are aggregated back to the document level (DOI)

## Retrieval semantics
* Vector search retrieves chunks
* The best chunk score per document is selected
  * The best chunk is the one with the highest cosine similarity.
  First, we identify the chunk closest to the query vector. Then, we assemble the full document using the DOI
  and its total chunk count, retaining the score from the highest-ranking chunk.
* Top K unique documents are chosen
* All chunks for each document are fetched
* Content is assembled in order
* One response object per document is returned

## Database migrations
This project uses SQLAlchemy 2.x for models and Alembic for schema migrations.
The database is Postgres, migrations are applied via Alembic.
The Alembic env is configured to add `src/` to `sys.path`.
We should run Alembic commands from the repo root, the folder that contains `alembic.ini`.

### Verify Alembic can talk to the DB:
```shell
alembic current
```
If the DB is empty, it will show no current revision until we run the upgrade.

### Bring DB to latest schema:
```shell
alembic upgrade head
```
### Add a new table, the standard workflow

Step 1, create a SQLAlchemy model add a new file under:
`src/biomed_platform/db/models/`

Example:
`src/biomed_platform/db/models/document.py`

Step 2, register the model in the models package:
`src/biomed_platform/db/models/__init__.py`
Import the new model, and keep existing imports.

Step 3, generate an Alembic migration
Make sure the database is at head first from Step 1. 

Generate the migration:
```shell
alembic revision --autogenerate -m "<YOUR MESSAGE>"
```
for example:
```shell
alembic revision --autogenerate -m "add documents table"
```
This creates a file like:
`alembic/versions/<revision_id>_add_documents_table.py`

Step 4, apply the migration
```shell
alembic upgrade head
```

Step 5, verify DB version:
```shell
alembic current
alembic heads
```

## Local Run
Start dependencies:
```shell
docker compose up
```

If you want to run the API separately:
```shell
uvicorn biomed_platform.api.app:app --reload
```

Run tests or clean test files:
```shell
make test
make clean
```