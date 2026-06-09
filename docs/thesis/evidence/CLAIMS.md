# CLAIMS

Purpose
Single source of truth for thesis claims that require evidence.
Chapter authors may only introduce claims that exist here.

Rule
A claim is valid only if it has at least one Evidence link.

Format
Cxxx, statement, scope, evidence, status.

## Retrieval and generation results

### C001
Statement:
RAG with HyDE improved nDCG at k compared to the no HyDE configuration for the canonical run.

Scope:
Chapter 7 Evaluation, Retrieval Metrics

Evidence:
evaluation_metrics/runs/old/feb-10-valid-1-full-run/20260209_114745_nogit/paper_benchmark_summary.json, json path: <fill exact key path once extracted>

Status:
draft, pending extraction of exact numbers and key paths

### C002
Statement:
The system evaluation includes three main configurations: LLM only, RAG without HyDE, RAG with HyDE.

Scope:
Chapter 7 Experimental Setup

Evidence:
evaluation_metrics/config/eval.yaml
evaluation_metrics/src/phases/phase3_generate.py

Status:
ready

### C003
Statement:
The scheduler supports incremental ingestion by DOI deduplication and persists scheduler runs with success and failure states.

Scope:
Chapter 3 System Architecture

Evidence:
scheduler_pubmed/src/core/domains/scheduler.py
scheduler_pubmed/src/core/services/scheduler_runtime.py

Status:
ready

### C004
Statement:
The Label Studio integration is implemented as a curation-side workflow that exports reviewed tasks, enriches them with bibliographic metadata, and separates clean ingestion-ready records from rejected records.

Scope:
Chapter 3 System Architecture, Chapter 5 Methodology, Chapter 10 Appendix

Evidence:
labelstudio-tools/src/app/python/download_data/main.py
labelstudio-tools/ls_data/tasks_raw.json
labelstudio-tools/ls_data/tasks_clean.json
labelstudio-tools/ls_data/tasks_rejected.json

Status:
ready

### C005
Statement:
The RAG handoff for curated Label Studio tasks batches ingestion requests and retries on transport errors, HTTP 429 responses, and 5xx responses using bounded backoff.

Scope:
Chapter 6 Implementation Details

Evidence:
labelstudio-tools/src/app/python/rag_importer/rag_integration.py

Status:
ready

### C006
Statement:
Initial Label Studio task preparation from Excel validates required columns, normalizes optional boolean review flags, applies configurable exclusion filters, and transforms valid rows into task payloads containing PMID, title, and abstract.

Scope:
Chapter 3 System Architecture, Chapter 6 Implementation Details

Evidence:
labelstudio-tools/src/app/python/config.py
labelstudio-tools/src/app/python/data_loader.py
labelstudio-tools/src/app/python/filters.py
labelstudio-tools/src/app/python/tasks.py

Status:
ready
