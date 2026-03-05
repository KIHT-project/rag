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